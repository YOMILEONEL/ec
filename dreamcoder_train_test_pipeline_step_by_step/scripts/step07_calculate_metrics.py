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

def program_operation_score(reference_tokens, generated_tokens):
    """
    Measures whether the same operations/tokens occur,
    independent of their position.

    It uses multiset overlap, so repeated operations are handled correctly.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    if not reference_tokens and not generated_tokens:
        return 0.0
    if not reference_tokens or not generated_tokens:
        return 0.0

    ref_counter = Counter(reference_tokens)
    gen_counter = Counter(generated_tokens)

    overlap = 0
    for token in ref_counter:
        overlap += min(ref_counter[token], gen_counter.get(token, 0))

    return overlap / max(len(reference_tokens), len(generated_tokens))

def program_position_score(reference_tokens, generated_tokens):
    """
    Measures how many tokens are equal at the same position.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    if not reference_tokens and not generated_tokens:
        return 0.0
    if not reference_tokens or not generated_tokens:
        return 0.0

    matches = 0
    for i in range(min(len(reference_tokens), len(generated_tokens))):
        if reference_tokens[i] == generated_tokens[i]:
            matches += 1

    return matches / max(len(reference_tokens), len(generated_tokens))

def longest_common_contiguous_length(reference_tokens, generated_tokens):
    """
    Longest common contiguous token block.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    if not reference_tokens or not generated_tokens:
        return 0

    dp = [[0] * (len(generated_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    best = 0

    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(generated_tokens) + 1):
            if reference_tokens[i - 1] == generated_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])

    return best

def program_sequence_score(reference_tokens, generated_tokens):
    """
    Measures whether the relative order of tokens is preserved.
    Based on normalized Longest Common contiguous token block.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    if not reference_tokens and not generated_tokens:
        return 0.0
    if not reference_tokens or not generated_tokens:
        return 0.0

    lccl = longest_common_contiguous_length(reference_tokens, generated_tokens)
    return lccl / max(len(reference_tokens), len(generated_tokens))

def levenshtein_distance(reference_tokens, generated_tokens):
    """
    Computes token-level Levenshtein distance.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    n = len(reference_tokens)
    m = len(generated_tokens)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference_tokens[i - 1] == generated_tokens[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[n][m]

def program_edit_score(reference_tokens, generated_tokens):
    """
    Converts edit distance into a similarity score in [0, 1].
    1 means identical programs.
    0 means maximally different according to normalized edit distance.

    Shared implementation, identical in the DeepCoder pipeline's
    compute_metrics.py - keep both in sync.
    """
    if not reference_tokens and not generated_tokens:
        return 0.0
    if not reference_tokens or not generated_tokens:
        return 0.0

    max_len = max(len(reference_tokens), len(generated_tokens))
    distance = levenshtein_distance(reference_tokens, generated_tokens)
    return 1.0 - (distance / max_len)

def add_metrics(row, prefix, ref, sol):
    row[f"{prefix}_operation_score"] = program_operation_score(ref, sol)
    row[f"{prefix}_position_score"] = program_position_score(ref, sol)
    row[f"{prefix}_sequence_score"] = program_sequence_score(ref, sol)
    row[f"{prefix}_edit_score"] = program_edit_score(ref, sol)
    row[f"{prefix}_token_edit_distance"] = levenshtein_distance(ref, sol)
    row[f"{prefix}_longest_common_block_length"] = longest_common_contiguous_length(ref, sol)

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
    for metric in ["operation_score", "position_score", "sequence_score", "edit_score"]:
        summary[f"{prefix}_{metric}_mean"] = mean(f"{prefix}_{metric}")
        summary[f"{prefix}_{metric}_mean_solved_only"] = mean(f"{prefix}_{metric}", lambda r: float(r.get("accuracy", 0) or 0) == 1.0)

(out_dir / "step07_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
