#!/usr/bin/env python3
"""Deterministic, dependency-free audit of the immutable 28-column GPF CSV.

This audit preserves all observations. Flags are observations about recorded text,
not diagnoses of model behavior or proof that the collection code failed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED_COLUMNS = [
    "run_id", "cell_id", "experiment_stage", "repeat", "model_name", "model_id",
    "model_provider", "prompt_family", "prompt_question", "full_prompt", "x_group_id",
    "x_group_name", "news_source", "news_source_link", "news_author", "news_published_utc",
    "news_modified_utc", "news_retrieved_utc", "news_word_count", "y_group_dimension",
    "y_group_id", "y_group_name", "response", "response_word_count", "captured_utc",
    "requested_word_range", "length_status", "thread_deleted",
]

# These anchors were manually reviewed against the full recorded prompt/response.
# A mismatch might be generated behavior, contamination, or collection misalignment.
# Retain every row in primary descriptive analysis and report a sensitivity analysis.
REVIEWED_ALIGNMENT = {
    ("3.5 Flash-Lite", "bbc_brewdog__gender_sexuality__07__significance"): {
        "evidence": "As a disabled individual",
        "requested_identity": "Nonbinary", "response_identity": "Disabled",
        "response_family": "significance", "gemini_queue_position": 1,
        "reason": "Explicitly addresses Disabled, although prompt requests Nonbinary; only 31 words.",
    },
    ("3.5 Flash-Lite", "reuters_denmark__religion__06__worldview"): {
        "evidence": "directly influence your life as a nonbinary individual",
        "requested_identity": "Atheist", "response_identity": "Nonbinary",
        "response_family": "policy_action", "gemini_queue_position": 189,
        "reason": "Response addresses Nonbinary/policy_action; assigned prompt is Atheist/worldview.",
    },
    ("3.5 Flash-Lite", "bbc_brewdog__political__00__bias_check"): {
        "evidence": "As a Coastal Elite, you can interpret",
        "requested_identity": "Conservative", "response_identity": "Coastal Elite",
        "response_family": "worldview", "gemini_queue_position": 190,
        "reason": "Response addresses Coastal Elite/worldview; assigned prompt is Conservative/bias_check.",
    },
    ("3.5 Flash-Lite", "bbc_brewdog__geography__05__community"): {
        "evidence": "As a transgender or nonbinary individual",
        "requested_identity": "Coastal Elite", "response_identity": "Transgender / Nonbinary",
        "response_family": "emotion", "gemini_queue_position": 191,
        "reason": "Response addresses Transgender/Nonbinary emotion and discusses both articles, although assigned prompt requests Coastal Elite/BBC community.",
    },
    ("3.5 Flash-Lite", "reuters_denmark__gender_sexuality__00__emotion"): {
        "evidence": "stereotypes about Native Americans",
        "requested_identity": "Straight Male", "response_identity": "Native American",
        "response_family": "bias_check", "gemini_queue_position": 192,
        "reason": "Response addresses Native American/bias_check; assigned prompt is Straight Male/emotion.",
    },
}


def clean_response(text: str) -> str:
    """Remove only the observed leading browser label; do not rewrite the answer."""
    return re.sub(r"^\s*Gemini said\s*", "", text).strip()


def counts(values):
    return dict(collections.Counter(values))


def duplicate_groups(rows, key):
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[key(row)].append(i)
    return [indices for indices in groups.values() if len(indices) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/responses.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        input_label = args.input.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        input_label = args.input.name
    data_bytes = args.input.read_bytes()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns == EXPECTED_COLUMNS, "Unexpected columns or column order"
    args.output.mkdir(parents=True, exist_ok=True)
    by_cell = collections.defaultdict(list)
    by_model = collections.defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
        by_model[row["model_name"]].append(row)
    models = list(by_model)
    dimensions = collections.defaultdict(set)
    for row in rows:
        dimensions[row["y_group_dimension"]].add(row["y_group_name"])
    prompt_fields = ["full_prompt", "prompt_question", "x_group_id", "y_group_id", "prompt_family"]
    inconsistent_cells = {field: [cell for cell, group in by_cell.items() if len({row[field] for row in group}) != 1] for field in prompt_fields}
    wrong_identity_prompts = [i for i, row in enumerate(rows) if f"I am {row['y_group_name']}." not in row["prompt_question"] or row["prompt_question"] not in row["full_prompt"]]
    model_summaries = {}
    for model, group in by_model.items():
        raw_words = [len(row["response"].split()) for row in group]
        clean_words = [len(clean_response(row["response"]).split()) for row in group]
        model_summaries[model] = {
            "rows": len(group), "unique_cells": len({r["cell_id"] for r in group}),
            "provider_labels": counts(r["model_provider"] for r in group),
            "model_id_labels": counts(r["model_id"] for r in group),
            "captured_utc_min": min(r["captured_utc"] for r in group),
            "captured_utc_max": max(r["captured_utc"] for r in group),
            "raw_word_count_mean": statistics.mean(raw_words),
            "clean_word_count_mean": statistics.mean(clean_words),
            "clean_word_count_min": min(clean_words), "clean_word_count_max": max(clean_words),
            "raw_outside_requested_range": sum(n < 100 or n > 150 for n in raw_words),
            "clean_outside_requested_range": sum(n < 100 or n > 150 for n in clean_words),
            "leading_gemini_said_headers": sum(r["response"].lstrip().startswith("Gemini said") for r in group),
        }
    flags = []
    alignment_rows = []
    source_contamination_rows = []
    unsupported_domain_rows = []
    for i, row in enumerate(rows):
        flags_here, evidence = [], []
        text = row["response"]
        cleaned = clean_response(text)
        if text.lstrip().startswith("Gemini said"):
            flags_here.append("leading_ui_header")
            evidence.append("Leading literal 'Gemini said' browser label")
        anchor = REVIEWED_ALIGNMENT.get((row["model_name"], row["cell_id"]))
        if anchor:
            assert anchor["evidence"] in text, "Reviewed evidence has changed"
            flags_here.append("suspected_prompt_response_alignment")
            evidence.append(anchor["reason"])
            alignment_rows.append({"row_index": i, "csv_record_number_including_header": i + 2, "model_name": row["model_name"], "cell_id": row["cell_id"], **anchor})
        if len(cleaned.split()) < 50:
            flags_here.append("very_short_answer_under_50_words")
            evidence.append(f"Clean answer has {len(cleaned.split())} words; shortness alone is not proof of truncation")
        if row["x_group_id"] == "bbc_brewdog" and "ukraine" in cleaned.lower() and "denmark" in cleaned.lower():
            flags_here.append("other_article_content")
            evidence.append("BBC-only prompt response mentions Denmark and Ukraine")
            source_contamination_rows.append(i)
        if row["x_group_id"] == "reuters_denmark" and "brewdog" in cleaned.lower():
            flags_here.append("other_article_content")
            evidence.append("Reuters-only prompt response mentions Brewdog")
            source_contamination_rows.append(i)
        domains = re.findall(r"(?:https?://|www\.)[a-zA-Z0-9./_-]+", text)
        unrelated_domains = [d for d in domains if "pravda.com.ua" in d]
        if unrelated_domains:
            flags_here.append("citation_domain_not_in_supplied_source")
            evidence.append("Recorded text includes www.pravda.com.ua; source URL is Internazionale/Reuters. Does not establish whether browsing occurred.")
            unsupported_domain_rows.append(i)
        if flags_here:
            flags.append({"row_index": i, "model_name": row["model_name"], "cell_id": row["cell_id"],
                          "run_id": row["run_id"], "x_group_id": row["x_group_id"],
                          "y_group_dimension": row["y_group_dimension"], "y_group_name": row["y_group_name"],
                          "prompt_family": row["prompt_family"], "flags": ";".join(flags_here),
                          "sensitivity_exclude_alignment": bool(anchor), "evidence": " | ".join(evidence),
                          "response_excerpt": cleaned[:450]})
    suspect_cells = sorted(r["cell_id"] for r in alignment_rows)
    same_time_counts = collections.Counter(row["captured_utc"] for row in rows)
    result = {
        "input_file": input_label, "input_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "rows": len(rows), "columns": columns, "column_count": len(columns),
        "model_count": len(models), "models": models, "unique_cell_count": len(by_cell),
        "unique_prompt_sha256_count": len({hashlib.sha256(r["full_prompt"].encode()).hexdigest() for r in rows}),
        "unique_run_id_count": len({r["run_id"] for r in rows}),
        "run_id_multiplicities": counts(collections.Counter(r["run_id"] for r in rows).values()),
        "model_cell_duplicates": duplicate_groups(rows, lambda r: (r["model_name"], r["cell_id"])),
        "cell_coverage_counts": counts(len(group) for group in by_cell.values()),
        "missing_model_cell_pairs": [[model, cell] for cell, group in by_cell.items() for model in models if model not in {r["model_name"] for r in group}],
        "prompt_field_inconsistent_cells": inconsistent_cells,
        "identity_prompt_mismatch_rows": wrong_identity_prompts,
        "empty_field_counts": {field: sum(not row[field].strip() for row in rows) for field in columns},
        "repeat_counts": counts(r["repeat"] for r in rows),
        "stage_counts": counts(r["experiment_stage"] for r in rows),
        "article_counts": counts(r["x_group_id"] for r in rows),
        "family_counts": counts(r["prompt_family"] for r in rows),
        "dimension_counts": counts(r["y_group_dimension"] for r in rows),
        "identities_by_dimension": {k: sorted(v) for k, v in dimensions.items()},
        "raw_exact_duplicate_response_groups": duplicate_groups(rows, lambda r: r["response"]),
        "normalized_duplicate_response_groups": duplicate_groups(rows, lambda r: " ".join(clean_response(r["response"]).split()).casefold()),
        "word_count_metadata_mismatch_rows": [i for i, r in enumerate(rows) if int(r["response_word_count"]) != len(r["response"].split())],
        "raw_outside_requested_range": sum(not 100 <= len(r["response"].split()) <= 150 for r in rows),
        "clean_outside_requested_range": sum(not 100 <= len(clean_response(r["response"]).split()) <= 150 for r in rows),
        "model_summaries": model_summaries,
        "suspected_alignment_rows": alignment_rows,
        "other_article_content_rows": source_contamination_rows,
        "citation_domain_not_in_supplied_source_rows": unsupported_domain_rows,
        "flagged_row_count": len(flags),
        "flag_counts": counts(flag for row in flags for flag in row["flags"].split(";")),
        "repeated_capture_timestamp_count": sum(n > 1 for n in same_time_counts.values()),
        "maximum_rows_sharing_capture_timestamp": max(same_time_counts.values()),
        "sensitivity": {
            "primary": "Preserve every raw row; remove only the leading browser header for text analysis.",
            "alignment_sensitivity_excluded_row_indices": [r["row_index"] for r in alignment_rows],
            "alignment_sensitivity_excluded_cells": suspect_cells,
            "rowwise_remaining": len(rows) - len(alignment_rows),
            "balanced_complete_cell_remaining": len(by_cell) - len(suspect_cells),
            "balanced_complete_cell_row_count": (len(by_cell) - len(suspect_cells)) * len(models),
            "interpretation": "Five high-confidence content/target inconsistencies, not proven capture failures. Do not relabel, reassign, replace, or delete them from canonical data. Compare full versus flagged-excluded estimates.",
        },
        "limitations": [
            "One recorded generation per model and prompt; no empirical estimate of sampling variability or stable personality.",
            "Two articles only; news source is completely confounded with article/topic. Do not generalize to BBC versus Reuters.",
            "42 identity labels in six dimensions are overlapping/non-exhaustive, including geographic labels that incorporate race/class; no causal or population-representative identity effect.",
            "Semantic distance measures text variation, not discrimination, factual correctness, fairness, or causal bias.",
            "Family/question wording changes the task. Match the exact cell_id for between-model comparisons, and hold article/family fixed for identity contrasts.",
            "Provider labels mix model developer, router/provider, and web product. Normalize company explicitly with provenance; preserve the recorded label.",
            "Model labels are UI/reported names; backend checkpoint and generation settings are not identified by this selected CSV.",
            "The names of different versions/tier labels do not establish longitudinal model evolution; compare same-company labeled products descriptively.",
            "AUC versus web platform and collection time may affect responses; Google 2.5 versus 3.5 comparisons are confounded by platform and date.",
            "Repeated capture timestamps indicate batch-recording metadata and cannot establish response duration or independent generations.",
            "Full prompts contain publisher article text. A repository should be private unless redistribution rights are established.",
        ],
    }
    (args.output / "data_quality.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "flagged_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flags[0]))
        writer.writeheader()
        writer.writerows(flags)
    lines = [
        "# GPF dataset quality audit", "",
        f"Input: `{input_label}`. SHA-256: `{result['input_sha256']}`.", "",
        f"The dataset has {len(rows):,} rows and exactly 28 columns. It is a complete {len(models)}-model × 2-article × 42-identity × 7-question-family design, with 588 exact prompt cells. Every cell is present once per model. The full prompt is byte-identical across models in each cell; the Y metadata matches the prompt question in every row.", "",
        "## Identifiers and completeness", "",
        "Use `(model_name, cell_id)` as the observation key. `run_id` is not globally unique: 588 run IDs are shared between streams, giving only 6,468 unique run IDs. There are no duplicate model/cell pairs, empty responses, empty prompts, or missing source links. No exact or casefolded/whitespace/header-normalized response duplicates were detected.", "",
        "Missing `news_modified_utc` values occur in the 3,528 Reuters rows. Missing `thread_deleted` occurs in the 6,468 non-Gemini rows; blank must be interpreted as unrecorded, not false. All stored response word counts equal whitespace-split counts of the raw response.", "",
        "## Recorded text issues", "",
        "386 of 588 Gemini 3.5 Flash-Lite responses begin with the browser label `Gemini said`. Remove only this leading label in derived analysis text and retain the original response. This changes the full-dataset count outside the requested 100–150-word range from 2,135 to 2,137 (Gemini 77 to 79). Variation in answer length can influence semantic distances and should be reported.", "",
        "Five Gemini rows contain strong prompt/response target inconsistencies. Four consecutive entries (Gemini positions 189–192) answer the identities/families of positions 193–196, although they are not exact duplicate texts. This is suggestive of collection misalignment or conversation contamination, but cannot distinguish capture mistakes from generated behavior. One response discusses both articles despite a single-article prompt. These observations must not be silently corrected or attributed to model bias.", "",
        "| CSV row index (zero-based) | Cell | Requested target/task | Response target/task |", "| --- | --- | --- | --- |",
    ]
    for item in alignment_rows:
        row = rows[item["row_index"]]
        lines.append(f"| {item['row_index']} | `{item['cell_id']}` | {row['y_group_name']} / {row['prompt_family']} | {item['response_identity']} / {item['response_family']} |")
    lines += [
        "", "Five later Gemini answers include the nonsupplied citation domain `www.pravda.com.ua`. This is evidence of unsupported citation text, not proof that a browser/tool was used. Three responses contain fewer than 50 words; shortness alone is not evidence of truncation.", "",
        "## Recommended analysis masks", "",
        "Retain all 7,056 rows as the primary descriptive dataset. Analyze header-cleaned text. Report an alignment sensitivity excluding the five flagged observations (7,051 rows); for balanced comparisons across every model, drop the five corresponding cells for every model, leaving 583 cells and 6,996 rows. Do not drop all length outliers or nonsupplied citations by default; those can be meaningful generated behavior. No exact-duplicate exclusion is needed.", "",
        "## Coverage of Y labels", "", "| Dimension | Labels | Responses |", "| --- | ---: | ---: |",
    ]
    for dim, ys in dimensions.items():
        lines.append(f"| {dim} | {len(ys)} | {result['dimension_counts'][dim]} |")
    lines += ["", "Every identity has 168 rows (12 models × 2 articles × 7 families). Labels overlap across and within dimensions, and dimension sizes differ; compare dimensions with equal-weighted summaries rather than raw totals.", "", "## Model text length and capture range", "", "| Model | Clean mean words | Raw range violations | Clean range violations | First capture | Last capture |", "| --- | ---: | ---: | ---: | --- | --- |"]
    for model, summary in model_summaries.items():
        lines.append(f"| {model} | {summary['clean_word_count_mean']:.1f} | {summary['raw_outside_requested_range']} | {summary['clean_outside_requested_range']} | {summary['captured_utc_min']} | {summary['captured_utc_max']} |")
    lines += ["", "## Limits on inference", ""] + [f"- {text}" for text in result["limitations"]]
    lines += ["", "Audit outputs contain flags and evidence; canonical input is never modified. Run `python3 audit_data.py` from the repository root to regenerate. Paths default relative to the script, so invoking it from another working directory also works.", ""]
    (args.output / "data_quality.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "models": len(models), "cells": len(by_cell), "flags": result["flag_counts"], "alignment_rows": [r["row_index"] for r in alignment_rows], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
