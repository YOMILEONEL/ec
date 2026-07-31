#!/usr/bin/env python3
import csv
import json
import pathlib
import sys
from collections import Counter

if len(sys.argv) != 3:
    raise SystemExit("Usage: step07_calculate_metrics.py <normalized_test_csv> <output_dir>")

input_csv = pathlib.Path(sys.argv[1]).resolve()
out_dir = pathlib.Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_tokens(value):
    return [t.strip() for t in str(value or "").split(",") if t.strip()]

def operation_score(reference, generated):
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    return sum((Counter(reference) & Counter(generated)).values()) / max(len(reference), len(generated))

def position_score(reference, generated):
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    return sum(a == b for a, b in zip(reference, generated)) / max(len(reference), len(generated))

def longest_common_block_length(reference, generated):
    if not reference or not generated:
        return 0
    prev = [0] * (len(generated) + 1)
    best = 0
    for i in range(1, len(reference) + 1):
        cur = [0] * (len(generated) + 1)
        for j in range(1, len(generated) + 1):
            if reference[i - 1] == generated[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best

def order_score(reference, generated):
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    return longest_common_block_length(reference, generated) / max(len(reference), len(generated))

def edit_distance(reference, generated):
    prev = list(range(len(generated) + 1))
    for i, a in enumerate(reference, 1):
        cur = [i]
        for j, b in enumerate(generated, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]

def edit_score(reference, generated):
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    return 1 - edit_distance(reference, generated) / max(len(reference), len(generated))

def add_metrics(row, prefix, ref, sol):
    row[f"{prefix}_operation_score"] = operation_score(ref, sol)
    row[f"{prefix}_position_score"] = position_score(ref, sol)
    row[f"{prefix}_order_score"] = order_score(ref, sol)
    row[f"{prefix}_edit_score"] = edit_score(ref, sol)
    row[f"{prefix}_token_edit_distance"] = edit_distance(ref, sol)
    row[f"{prefix}_longest_common_block_length"] = longest_common_block_length(ref, sol)

rows = read_rows(input_csv)
for row in rows:
    solved = str(row.get("solved", "")).lower() in {"true", "1", "yes"}
    row["accuracy"] = 1.0 if solved else 0.0
    ref_norm = parse_tokens(row.get("reference_tokens_normalized", ""))
    sol_norm = parse_tokens(row.get("solution_tokens_normalized", ""))
    ref_abs = parse_tokens(row.get("reference_operations_abstract", ""))
    sol_abs = parse_tokens(row.get("solution_operations_abstract", ""))
    add_metrics(row, "normalized", ref_norm, sol_norm)
    add_metrics(row, "abstract", ref_abs, sol_abs)

out_csv = out_dir / "step07_test_results_with_metrics.csv"
fieldnames = list(rows[0].keys()) if rows else []
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

def mean(col, subset=None):
    active = rows if subset is None else [r for r in rows if subset(r)]
    vals = [float(r.get(col, 0) or 0) for r in active]
    return sum(vals) / len(vals) if vals else 0.0

summary = {
    "test_tasks": len(rows),
    "test_solved": int(sum(float(r["accuracy"]) for r in rows)),
    "test_accuracy": mean("accuracy"),
    "trivial_solutions": int(sum(int(float(r.get("trivial_solution", 0) or 0)) for r in rows)),
    "normalized_exact_matches": int(sum(int(float(r.get("normalized_exact_match", 0) or 0)) for r in rows)),
    "abstract_exact_matches": int(sum(int(float(r.get("abstract_exact_match", 0) or 0)) for r in rows)),
}
for prefix in ["normalized", "abstract"]:
    for metric in ["operation_score", "position_score", "order_score", "edit_score"]:
        summary[f"{prefix}_{metric}_mean"] = mean(f"{prefix}_{metric}")
        summary[f"{prefix}_{metric}_mean_solved_only"] = mean(f"{prefix}_{metric}", lambda r: float(r.get("accuracy", 0) or 0) == 1.0)

(out_dir / "step07_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
