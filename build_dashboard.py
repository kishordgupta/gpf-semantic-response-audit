#!/usr/bin/env python3
"""Build the dependency-free, offline GPF response-distance dashboard.

Usage: python build_dashboard.py [--root PATH]
The generated HTML contains the complete response dataset; keep it private when
the underlying source-text permissions or account metadata require that.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def read_csv(path: Path, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict]:
    for row in rows:
        for key in keys:
            if row.get(key, "") != "":
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
    return rows


def build(root: Path) -> Path:
    raw = read_csv(root / "data/responses.csv", required=True)
    if not raw:
        raise ValueError("The response dataset is empty")
    models = list(dict.fromkeys(row["model_name"] for row in raw))
    model_lookup = {name: i for i, name in enumerate(models)}
    cell_ids = list(dict.fromkeys(row["cell_id"] for row in raw))
    cell_lookup = {name: i for i, name in enumerate(cell_ids)}
    cells: list[dict | None] = [None] * len(cell_ids)
    responses = []
    occupied = set()
    fields = ["cell_id", "prompt_family", "prompt_question", "full_prompt",
              "x_group_id", "x_group_name", "news_source", "news_source_link",
              "y_group_dimension", "y_group_id", "y_group_name"]
    flags = read_csv(root / "results/flagged_rows.csv")
    run_flags: dict[str, list[dict]] = {}
    pair_flags: dict[tuple[str, str], list[dict]] = {}
    for flag in flags:
        if flag.get("model_name") and flag.get("cell_id"):
            pair_flags.setdefault((flag["model_name"], flag["cell_id"]), []).append(flag)
        elif flag.get("run_id"):
            run_flags.setdefault(flag["run_id"], []).append(flag)
    for row in raw:
        ci = cell_lookup[row["cell_id"]]
        mi = model_lookup[row["model_name"]]
        key = (ci, mi)
        if key in occupied:
            raise ValueError(f"Duplicate cell/model in dashboard: {key}")
        occupied.add(key)
        cell = {field: row[field] for field in fields}
        if cells[ci] is not None and cells[ci] != cell:
            raise ValueError(f"Prompt/cell metadata mismatch: {row['cell_id']}")
        cells[ci] = cell
        response_flags = run_flags.get(row["run_id"], []) + pair_flags.get((row["model_name"], row["cell_id"]), [])
        # Compact response record: cell, model, exact text, words, length status,
        # capture timestamp, run id, explicit audit flags, thread deletion status.
        responses.append([ci, mi, row["response"], int(row["response_word_count"]),
                          row.get("length_status", ""), row.get("captured_utc", ""),
                          row["run_id"], response_flags, row.get("thread_deleted", "")])
    tables = {}
    names = ["pairwise_summary", "pairwise_by_y", "pairwise_by_family",
             "pairwise_by_article", "identity_sensitivity", "y_importance"]
    names += [name + "_screened" for name in names]
    names.append("model_summary")
    for name in names:
        tables[name] = numeric(read_csv(root / f"results/{name}.csv"),
                               ("mean_distance", "ci_low", "ci_high", "n_prompts",
                                "n_pairs", "n_blocks", "rows", "mean_words",
                                "median_words", "length_compliance",
                                "mean_distance_to_other_identities"))
    summary_path = root / "results/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    distances = read_csv(root / "results/pairwise_prompt.csv")
    encoders = list(dict.fromkeys(row["encoder"] for row in tables["pairwise_summary"] + distances))
    encoder_lookup = {name: i for i, name in enumerate(encoders)}
    compact_distances = []
    for row in distances:
        if row["cell_id"] not in cell_lookup:
            raise ValueError(f"Unknown distance cell: {row['cell_id']}")
        compact_distances.append([encoder_lookup[row["encoder"]], cell_lookup[row["cell_id"]],
                                  model_lookup[row["model_a"]], model_lookup[row["model_b"]],
                                  round(float(row["distance"]), 7)])
    robustness_path = root / "results/robustness_summary.json"
    robustness = json.loads(robustness_path.read_text()) if robustness_path.exists() else {}
    if robustness:
        excluded = set(robustness.get("excluded_cells", []))
        assert excluded <= set(cell_ids), "Screening lists cells outside this dataset"
        assert len(cell_ids) - len(excluded) == robustness["screened_cells"]
        if not all(tables[name + "_screened"] for name in ("pairwise_summary", "pairwise_by_y", "pairwise_by_family", "identity_sensitivity")):
            raise ValueError("Screening metadata exists but the required screened tables are incomplete")
    payload = {"models": models, "cells": cells, "responses": responses,
               "encoders": encoders, "distances": compact_distances,
               "tables": tables, "summary": summary, "robustness": robustness,
               "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "schema": list(raw[0]), "flagged_rows": len(flags)}
    # Escape '<' so dataset content cannot terminate the application/json script.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (root / "dashboard/template.html").read_text(encoding="utf-8")
    output = root / "dashboard/index.html"
    output.write_text(template.replace("__GPF_DATA__", blob), encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "rows": len(raw), "cells": len(cells), "models": len(models),
                      "distance_rows": len(compact_distances), "encoders": encoders}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().root.resolve())
