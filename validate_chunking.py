#!/usr/bin/env python3
"""Validate frozen tokenizer chunking locally without recomputing embeddings."""
from pathlib import Path
import csv
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT.parent / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer


def clean(text):
    text = re.sub(r"^\s*Gemini said\s*\n+", "", text)
    text = re.sub(r"^\s*Response \d+:\s*\n+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    rows = list(csv.DictReader((ROOT / "data/responses.csv").open(newline="")))
    revisions = json.loads((ROOT / "encoder_lock.json").read_text())
    texts = [clean(row["response"]) for row in rows]
    labels = sorted({row["y_group_name"] for row in rows}, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(label) for label in labels) + r")(?!\w)", re.I)
    masked = [pattern.sub("[IDENTITY]", text) for text in texts]
    articles = {}
    for row in sorted(rows, key=lambda r: r["cell_id"]):
        articles.setdefault(row["x_group_id"], row["full_prompt"].split("\n\nARTICLE\n", 1)[1])
    passes = [("mpnet", "mpnet", texts, 384), ("minilm", "minilm", texts, 256),
              ("mpnet_masked", "mpnet", masked, 384),
              ("articles_mpnet", "mpnet", list(articles.values()), 384)]
    results = []
    tokenizers = {}
    for stem, key, inputs, model_max in passes:
        if key not in tokenizers:
            tokenizers[key] = AutoTokenizer.from_pretrained(revisions[key]["name"], revision=revisions[key]["revision"], local_files_only=True)
        tokenizer = tokenizers[key]
        metadata_path = ROOT / "embeddings" / f"{stem}.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            model_max = metadata["max_seq_length"]
            budget = metadata["chunk_token_budget"]
        else:
            metadata = None
            budget = min(220, model_max - 16)
        max_reencoded, max_original, chunks, multichunk, changed = 0, 0, 0, 0, 0
        violations = []
        for row_index, text in enumerate(inputs):
            token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            max_original = max(max_original, len(token_ids))
            multichunk += len(token_ids) > budget
            for start in range(0, max(1, len(token_ids)), budget):
                part = token_ids[start:start + budget]
                decoded = tokenizer.decode(part, skip_special_tokens=True)
                retokens = tokenizer.encode(decoded, add_special_tokens=True, truncation=False)
                noprefix = tokenizer.encode(decoded, add_special_tokens=False, truncation=False)
                changed += part != noprefix
                chunks += 1
                max_reencoded = max(max_reencoded, len(retokens))
                if len(retokens) > model_max:
                    violations.append({"row_index": row_index, "chunk_start_token": start, "retokenized_length_including_specials": len(retokens)})
        digest = hashlib.sha256("\n".join(inputs).encode()).hexdigest()
        if metadata:
            assert digest == metadata["text_sha256"], f"Text hash differs: {stem}"
            assert chunks == metadata["chunks"], f"Chunk count differs: {stem}"
        results.append({"pass": stem, "encoder": revisions[key]["name"], "revision": revisions[key]["revision"],
                        "texts": len(inputs), "text_sha256": digest, "model_max_seq_length": model_max,
                        "token_budget": budget, "chunks": chunks, "multi_chunk_texts": multichunk,
                        "max_original_text_tokens": max_original,
                        "max_retokenized_chunk_tokens_including_specials": max_reencoded,
                        "chunks_with_changed_tokenization_after_decode": changed,
                        "over_limit_chunks": violations, "saved_embedding_metadata_available": bool(metadata)})
    result = {"status": "pass" if all(not item["over_limit_chunks"] for item in results) else "fail",
              "method": "Local cached AutoTokenizer only; reconstruct original nonoverlapping token chunks, decode, then retokenize including special tokens with truncation disabled. No embedding inference or network calls.",
              "interpretation": "No over-limit retokenized chunks means no truncation from model maximum length under this exact procedure. Mid-word chunk boundaries can change tokenization during decode/retokenize; this is separately counted and is not a semantic-equivalence guarantee.",
              "passes": results}
    (ROOT / "results/chunk_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["status"] == "pass"


if __name__ == "__main__":
    main()
