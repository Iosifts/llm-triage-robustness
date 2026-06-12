#!/usr/bin/env python3
"""Merge Slurm shard outputs into results_full.csv and validate completeness."""
import argparse
import csv
import glob
import os
from collections import Counter

EXPECTED_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="results_full_parts/results_*_shard*.csv")
    ap.add_argument("--out", default="results_full.csv")
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--conditions", type=int, default=7)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.parts))
    expected_parts = len(EXPECTED_MODELS) * args.shards
    print(f"parts: {len(paths)} / expected {expected_parts}")
    if len(paths) != expected_parts:
        missing = expected_parts - len(paths)
        print(f"WARNING: part count mismatch ({missing:+d})")

    all_rows = []
    fieldnames = None
    for path in paths:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        print(path, len(rows))
        if not rows:
            print(f"WARNING: empty part: {path}")
            continue
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows found.")

    dataset_rows = list(csv.DictReader(open(args.dataset, encoding="utf-8")))
    expected_rows = len(dataset_rows) * args.conditions * args.repeats * len(EXPECTED_MODELS)

    print("")
    print("rows:", len(all_rows), "/ expected", expected_rows)
    print("by model:", Counter(r["model"] for r in all_rows))
    print("by condition:", Counter(r["condition"] for r in all_rows))
    print("parse_ok:", Counter(r["parse_ok"] for r in all_rows))
    print("verdicts:", Counter(r["parsed_verdict"] for r in all_rows))

    key_counts = Counter((r["model"], r["cve_id"], r["condition"], r["repeat"]) for r in all_rows)
    duplicates = [k for k, v in key_counts.items() if v > 1]
    if duplicates:
        print(f"WARNING: duplicate result cells: {len(duplicates)}")
        print("first duplicate:", duplicates[0])

    models = set(r["model"] for r in all_rows)
    missing_models = EXPECTED_MODELS - models
    if missing_models:
        print("WARNING: missing models:", sorted(missing_models))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("")
    print(f"wrote {args.out}")
    print(f"next: python run_and_score.py --metrics-only --out {args.out}")


if __name__ == "__main__":
    main()
