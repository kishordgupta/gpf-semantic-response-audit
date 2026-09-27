#!/usr/bin/env python3
"""Independent consistency checks for the released data and numeric artifacts.

Does not load an encoder or call a service. By default, uses saved embeddings to
independently recompute every matched semantic distance. The --without-embeddings
mode checks published matrices, tables, data, and cases without private caches;
it still independently refits and recomputes the TF-IDF lexical distances.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-artifacts", action="store_true", help="Require a local HTML dashboard and a compiled report PDF.")
    parser.add_argument("--without-embeddings", action="store_true", help="Validate released data/matrices/tables without requiring encoder tensors; semantic embeddings are not recomputed in this mode.")
    args = parser.parse_args()
    validation_mode = "published_artifacts_without_embeddings" if args.without_embeddings else "recompute_all_distances_from_cached_embeddings"
    output_path = ROOT / "results" / ("validation_without_embeddings.json" if args.without_embeddings else "validation.json")
    checks = []

    def check(label, condition, details=None):
        record = {"check": label, "pass": bool(condition)}
        if details is not None:
            record["details"] = details
        checks.append(record)
        if not condition:
            raise AssertionError(f"{label}: {details}")

    try:
        df = pd.read_csv(ROOT / "data/responses.csv", keep_default_na=False)
        summary = json.loads((ROOT / "results/summary.json").read_text())
        quality = json.loads((ROOT / "results/data_quality.json").read_text())
        models = summary["models"]
        cells = sorted(df.cell_id.unique())
        pairs = list(itertools.combinations(models, 2))
        metrics = ["mpnet", "minilm", "mpnet_masked", "tfidf"]
        source_hash = hashlib.sha256((ROOT / "data/responses.csv").read_bytes()).hexdigest()
        check("Dataset SHA256 agrees with analysis and audit", source_hash == summary["source_sha256"] == quality["input_sha256"])
        check("Dataset is 7056 rows × 28 specified columns", df.shape == (7056, 28) and df.columns.tolist() == quality["columns"])
        check("12 unique models, 588 complete exact prompt cells", len(models) == 12 and len(cells) == 588 and df.groupby("cell_id").model_name.nunique().eq(12).all() and df.groupby("cell_id").full_prompt.nunique().eq(1).all())
        check("Observation keys are unique and responses nonempty", not df.duplicated(["model_name", "cell_id"]).any() and df.response.str.strip().ne("").all())
        def clean(text):
            text = re.sub(r"^\s*Gemini said\s*\n+", "", text)
            text = re.sub(r"^\s*Response \d+:\s*\n+", "", text)
            return re.sub(r"\s+", " ", text).strip()
        texts = df.response.map(clean).tolist()
        labels = sorted(df.y_group_name.unique(), key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(label) for label in labels) + r")(?!\w)", re.I)
        masked = [pattern.sub("[IDENTITY]", text) for text in texts]
        positions = {(row.cell_id, row.model_name): i for i, row in enumerate(df.itertuples(index=False))}
        embeddings = {}
        for metric in metrics[:-1]:
            metadata = summary["encoder_metadata"][metric] if args.without_embeddings else json.loads((ROOT / f"embeddings/{metric}.json").read_text())
            if not args.without_embeddings:
                vectors = np.load(ROOT / f"embeddings/{metric}.npz")["embeddings"]
                check(f"{metric}: shape and finite normalized vectors", vectors.shape == (7056, metadata["dimensions"]) and np.isfinite(vectors).all() and np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)) < 1e-5)
                embeddings[metric] = vectors
            expected_hash = hashlib.sha256("\n".join(masked if metric == "mpnet_masked" else texts).encode()).hexdigest()
            check(f"{metric}: embedding text hash matches dataset", metadata["text_sha256"] == expected_hash)
        # Independently instantiate the vectorizer, using the original released
        # feature order instead of platform-dependent max_features tie selection.
        lexical_lock = json.loads((ROOT / "results/tfidf_vocabulary.json").read_text())
        lexical_terms = lexical_lock["terms_by_feature_index"]
        lexical_corpus_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        lexical_vocab_hash = hashlib.sha256(json.dumps(lexical_terms, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        check("TF-IDF input and ordered vocabulary checksums match frozen baseline", lexical_corpus_hash == lexical_lock["text_sha256"] and lexical_vocab_hash == lexical_lock["ordered_vocabulary_sha256"] and len(set(lexical_terms)) == len(lexical_terms) == lexical_lock["features"] == 50000)
        lexical_parameters = dict(lexical_lock["parameters"])
        lexical_parameters["ngram_range"] = tuple(lexical_parameters["ngram_range"])
        lexical = TfidfVectorizer(vocabulary={term: index for index, term in enumerate(lexical_terms)}, **lexical_parameters).fit_transform(texts)
        arrays = np.load(ROOT / "results/matched_distances.npz")
        pair_summary = pd.read_csv(ROOT / "results/pairwise_summary.csv")
        pair_prompt = pd.read_csv(ROOT / "results/pairwise_prompt.csv")
        check("Prompt-pair table complete", len(pair_prompt) == 4 * 588 * 66 and not pair_prompt.duplicated(["encoder", "cell_id", "model_a", "model_b"]).any())
        check("Pairwise summary complete", len(pair_summary) == 4 * 66 and not pair_summary.duplicated(["encoder", "model_a", "model_b"]).any())
        for metric in metrics:
            arr = arrays[metric]
            check(f"{metric}: matrix finite and in cosine-distance bounds", arr.shape == (588, 66) and np.isfinite(arr).all() and arr.min() >= -1e-7 and arr.max() <= 2 + 1e-7)
            if metric == "tfidf" or not args.without_embeddings:
                recomputed = []
                for a, b in pairs:
                    ia = [positions[(cell, a)] for cell in cells]
                    ib = [positions[(cell, b)] for cell in cells]
                    if metric == "tfidf":
                        values = 1 - np.asarray(lexical[ia].multiply(lexical[ib]).sum(axis=1)).ravel()
                    else:
                        values = 1 - np.einsum("ij,ij->i", embeddings[metric][ia], embeddings[metric][ib])
                    recomputed.append(np.clip(values, 0, 2))
                direct = np.array(recomputed).T
                check(f"{metric}: all 38808 distances independently recomputed", np.allclose(arr, direct, atol=2e-6), {"max_absolute_error": float(np.max(np.abs(arr - direct)))})
            lookup = pair_summary[pair_summary.encoder == metric].set_index(["model_a", "model_b"])
            means = np.array([lookup.loc[pair, "mean_distance"] for pair in pairs])
            check(f"{metric}: every summary mean equals its 588 distances", np.allclose(means, arr.mean(axis=0), atol=1e-7))
            check(f"{metric}: confidence intervals ordered and bounded", (lookup.ci_low <= lookup.ci_high).all() and lookup.ci_low.between(0, 2).all() and lookup.ci_high.between(0, 2).all())
            exported = pair_prompt[pair_prompt.encoder == metric].pivot(index="cell_id", columns=["model_a", "model_b"], values="distance").reindex(index=cells, columns=pd.MultiIndex.from_tuples(pairs)).to_numpy()
            check(f"{metric}: long-form CSV equals distance matrix", np.allclose(exported, arr, atol=1e-7))
        coverage = {"pairwise_by_y": 4 * 66 * 42, "pairwise_by_family": 4 * 66 * 7,
                    "pairwise_by_article": 4 * 66 * 2, "identity_sensitivity": 4 * 12 * 6,
                    "y_importance": 4 * 12 * 42, "length_sensitivity": 4 * 66, "source_proximity": 7056}
        for name, expected_rows in coverage.items():
            table = pd.read_csv(ROOT / f"results/{name}.csv")
            check(f"{name}: expected table coverage", len(table) == expected_rows, {"rows": len(table), "expected_rows": expected_rows})
        influence = pd.read_csv(ROOT / "results/y_importance.csv")
        within = pd.read_csv(ROOT / "results/identity_sensitivity.csv")
        avg = influence.groupby(["encoder", "model_name", "y_group_dimension"]).mean_distance_to_other_identities.mean()
        expected = within.set_index(["encoder", "model_name", "y_group_dimension"]).mean_distance
        check("Y-level distance averages equal corresponding dimension pairwise means", np.allclose(avg.sort_index(), expected.sort_index(), atol=1e-7))
        check("Five suspect observations explicitly retained and flagged", len(quality["suspected_alignment_rows"]) == 5 and quality["sensitivity"]["balanced_complete_cell_remaining"] == 583)
        chunk_path = ROOT / "results/chunk_validation.json"
        chunk_validation = json.loads(chunk_path.read_text())
        check("Tokenizer chunks never exceed model sequence length", chunk_validation["status"] == "pass" and all(not item["over_limit_chunks"] for item in chunk_validation["passes"]))
        for item in chunk_validation["passes"]:
            meta_path = ROOT / f"embeddings/{item['pass']}.json"
            if args.without_embeddings and item["pass"] in summary["encoder_metadata"]:
                meta = summary["encoder_metadata"][item["pass"]]
                check(f"{item['pass']}: saved chunk validation agrees with published metadata", item["chunks"] == meta["chunks"] and item["text_sha256"] == meta["text_sha256"])
            elif not args.without_embeddings and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                check(f"{item['pass']}: independently reconstructed chunk metadata agrees", item["chunks"] == meta["chunks"] and item["text_sha256"] == meta["text_sha256"])
        excluded = set(quality["sensitivity"]["alignment_sensitivity_excluded_cells"])
        keep = np.array([cell not in excluded for cell in cells])
        screened = pd.read_csv(ROOT / "results/pairwise_summary_screened.csv")
        check("Screened comparison is balanced 583 complete cells", keep.sum() == 583 and len(screened) == 4 * 66 and screened.n_prompts.eq(583).all())
        for metric in metrics:
            table = screened[screened.encoder == metric].set_index(["model_a", "model_b"])
            observed = np.array([table.loc[pair, "mean_distance"] for pair in pairs])
            check(f"{metric}: screened means equal independently selected matrix rows", np.allclose(observed, arrays[metric][keep].mean(axis=0), atol=1e-7))
        for filename in ["cases.json", "cases_screened.json"]:
            cases = json.loads((ROOT / f"results/{filename}").read_text())
            valid = True
            for case in cases:
                row_a = df.iloc[positions[(case["cell_id"], case["model_a"])]]
                row_b = df.iloc[positions[(case["cell_id"], case["model_b"])]]
                numeric = arrays["mpnet"][cells.index(case["cell_id"]), pairs.index((case["model_a"], case["model_b"]))]
                valid &= (case["response_a"] == row_a.response and case["response_b"] == row_b.response and case["prompt_question"] == row_a.prompt_question and abs(case["distance"] - numeric) < 1e-7)
                if "screened" in filename:
                    valid &= case["cell_id"] not in excluded
            check(f"{filename}: selected quotations and distances match canonical rows", valid, {"cases": len(cases)})
        html = ROOT / "dashboard/index.html"
        pdfs = list((ROOT / "report").glob("*.pdf"))
        if args.require_artifacts:
            check("Standalone HTML dashboard exists", html.exists() and html.stat().st_size > 10000 and "<html" in html.read_text().lower())
            check("Compiled report PDF exists", bool(pdfs) and any(p.read_bytes().startswith(b"%PDF") and p.stat().st_size > 10000 for p in pdfs))
        result = {"status": "pass", "checks_passed": len(checks), "require_artifacts": args.require_artifacts, "validation_mode": validation_mode,
                  "limitations": ["Published-artifact mode checks semantic matrices against reported tables; it does not recompute semantic encoder outputs or verify saved tokenizer behavior by running tokenizers."] if args.without_embeddings else [],
                  "input_sha256": source_hash, "checks": checks}
    except Exception as error:
        result = {"status": "fail", "checks_passed": sum(item["pass"] for item in checks), "validation_mode": validation_mode, "error": f"{type(error).__name__}: {error}", "checks": checks}
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        raise
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "validation_mode", "require_artifacts", "input_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
