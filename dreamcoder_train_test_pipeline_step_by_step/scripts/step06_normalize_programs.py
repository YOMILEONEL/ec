#!/usr/bin/env python3
"""
Step 06: normalize reference programs and DreamCoder solutions.

Robustness improvements:
- Unsolved tasks are never counted as trivial solutions.
- Empty solution tokens are only trivial if the task is solved and the solution is
  explicitly an empty-program/identity/constant pattern.
- Exact matches are only counted for solved tasks.
"""
import csv
import json
import pathlib
import re
import sys
from collections import Counter

if len(sys.argv) != 3:
    raise SystemExit("Usage: step06_normalize_programs.py <test_results_csv> <output_dir>")

input_csv = pathlib.Path(sys.argv[1]).resolve()
out_dir = pathlib.Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z_?<>!=]+|\+\+|[+*/-]|\d+|\$\d+")
IGNORED_SOLUTION_TOKENS = {"lambda", "true", "false"}
SOLUTION_TOKEN_MAP = {
    "reverse": "REVERSE",
    "sort": "SORT",
    "sum": "SUM",
    "length": "LENGTH",
    "car": "HEAD",
    "cdr": "TAIL",
    "index": "ACCESS",
    "slice": "SLICE",
    "cons": "CONS",
    "empty": "EMPTY_CONST",
    "empty?": "EMPTY_CHECK",
    "++": "CONCAT",
    "map": "MAP",
    "mapi": "MAP",
    "filter": "FILTER",
    "fold": "FOLD",
    "reduce": "FOLD",
    "reducei": "FOLD",
    "unfold": "UNFOLD",
    "range": "RANGE",
    "+": "ARITH_ADD",
    "-": "ARITH_SUBTRACT",
    "*": "ARITH_MULT",
    "/": "ARITH_DIV",
    "mod": "ARITH_MOD",
    "negate": "ARITH_NEGATE",
    "gt?": "PREDICATE",
    "lt?": "PREDICATE",
    "eq?": "PREDICATE",
    "is-square": "PREDICATE",
    "is-prime": "PREDICATE",
    "if": "IF",
}


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reference_token(statement):
    parts = [p.strip() for p in statement.split(",")]
    operation = parts[0].upper() if parts else ""
    if operation in {"LIST", "INT", "BOOL", ""}:
        return ""
    parameterized = {
        "COUNT": {"<0": "COUNT_NEG", ">0": "COUNT_POS", "EVEN": "COUNT_EVEN", "ODD": "COUNT_ODD"},
        "FILTER": {"<0": "FILTER_NEG", ">0": "FILTER_POS", "EVEN": "FILTER_EVEN", "ODD": "FILTER_ODD"},
        "MAP": {
            "+1": "MAP_INCREMENT", "-1": "MAP_DECREMENT", "*-1": "MAP_NEGATE",
            "*2": "MAP_MULT2", "*3": "MAP_MULT3", "*4": "MAP_MULT4",
            "/2": "MAP_DIV2", "/3": "MAP_DIV3", "/4": "MAP_DIV4", "**2": "MAP_SQUARE",
            # Some filenames encode /2 as _2, /3 as _3, /4 as _4.
            "_2": "MAP_DIV2", "_3": "MAP_DIV3", "_4": "MAP_DIV4",
        },
        "SCAN1L": {"+": "SCAN_ADD", "-": "SCAN_SUBTRACT", "*": "SCAN_MULT", "min": "SCAN_MIN", "max": "SCAN_MAX"},
        "ZIPWITH": {"+": "ZIPWITH_ADD", "-": "ZIPWITH_SUBTRACT", "*": "ZIPWITH_MULT", "min": "ZIPWITH_MIN", "max": "ZIPWITH_MAX"},
    }
    if operation in parameterized:
        return parameterized[operation].get(parts[1], operation) if len(parts) > 1 else operation
    direct = {"ACCESS", "DROP", "HEAD", "MAXIMUM", "MINIMUM", "REVERSE", "SORT", "SUM", "TAIL", "TAKE"}
    return operation if operation in direct else operation


def reference_abstract(token):
    if token.startswith("COUNT"):
        return "COUNT"
    if token.startswith("FILTER"):
        return "FILTER"
    if token.startswith("MAP"):
        return "MAP"
    if token.startswith("SCAN"):
        return "SCAN1L"
    if token.startswith("ZIPWITH"):
        return "ZIPWITH"
    return token


def tokenize_reference(program):
    tokens = []
    for statement in str(program or "").split("|"):
        token = reference_token(statement.strip())
        if token:
            tokens.append(token)
    return tokens


def tokenize_solution(program):
    tokens = []
    unknown = []
    for raw in TOKEN_RE.findall(str(program or "")):
        if raw in IGNORED_SOLUTION_TOKENS or re.fullmatch(r"\d+|\$\d+", raw):
            continue
        mapped = SOLUTION_TOKEN_MAP.get(raw)
        if mapped is None:
            unknown.append(raw)
        else:
            tokens.append(mapped)
    return tokens, unknown


def solution_abstract(token):
    if token.startswith("ARITH"):
        return "ARITH"
    if token == "PREDICATE":
        return "PREDICATE"
    return token


def is_trivial_solution(program, solution_tokens, solved, degenerate_example_set=False):
    if not solved:
        return 0
    text = re.sub(r"\s+", " ", str(program or "").strip())
    trivial_exact = {
        "empty", "$0", "$1", "$2", "0",
        "(lambda empty)", "(lambda $0)", "(lambda $1)", "(lambda $2)", "(lambda 0)",
        "(lambda (lambda empty))", "(lambda (lambda $0))", "(lambda (lambda $1))", "(lambda (lambda $2))", "(lambda (lambda 0))",
        "(lambda (lambda (lambda empty)))", "(lambda (lambda (lambda $0)))", "(lambda (lambda (lambda $1)))", "(lambda (lambda (lambda $2)))", "(lambda (lambda (lambda 0)))",
    }
    if text in trivial_exact:
        return 1
    # A solved program whose normalized token list consists only of EMPTY_CONST
    # is also a constant-empty solution. Empty token list alone is not enough.
    if solution_tokens == ["EMPTY_CONST"]:
        return 1
    # A task whose examples all share the same output (e.g. every example
    # happens to output []) cannot distinguish a real solution from a program
    # that ignores its input and returns that constant, even if the solution
    # text itself looks non-trivial (e.g. "(lambda (reverse (filter (lambda $0)
    # empty)))" always returns [] but is wrapped in real-looking operations).
    if degenerate_example_set:
        return 1
    return 0


rows = read_rows(input_csv)
normalized = []
all_unknown = Counter()
for row in rows:
    solved = parse_bool(row.get("solved", ""))
    ref_norm = tokenize_reference(row.get("reference_program", ""))
    sol_norm, unknown = tokenize_solution(row.get("solution", "")) if solved else ([], [])
    all_unknown.update(unknown)
    ref_abs = [reference_abstract(token) for token in ref_norm]
    sol_abs = [solution_abstract(token) for token in sol_norm]
    new_row = dict(row)
    new_row.update({
        "reference_tokens_normalized": ",".join(ref_norm),
        "solution_tokens_normalized": ",".join(sol_norm),
        "reference_operations_abstract": ",".join(ref_abs),
        "solution_operations_abstract": ",".join(sol_abs),
        "unknown_solution_tokens": ",".join(sorted(set(unknown))),
        "normalized_exact_match": 1 if solved and ref_norm == sol_norm and ref_norm else 0,
        "abstract_exact_match": 1 if solved and ref_abs == sol_abs and ref_abs else 0,
        "trivial_solution": is_trivial_solution(
            row.get("solution", ""), sol_norm, solved,
            degenerate_example_set=parse_bool(row.get("degenerate_example_set", "")),
        ),
    })
    normalized.append(new_row)

out_csv = out_dir / "step06_normalized_test_programs.csv"
write_rows(out_csv, normalized)
summary = {
    "tasks": len(normalized),
    "solved_tasks": sum(1 for row in normalized if parse_bool(row.get("solved", ""))),
    "rows_with_unknown_solution_tokens": sum(1 for row in normalized if row["unknown_solution_tokens"]),
    "unknown_solution_tokens": dict(sorted(all_unknown.items())),
    "trivial_solutions": sum(int(row["trivial_solution"]) for row in normalized),
    "output_csv": str(out_csv),
}
(out_dir / "step06_normalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
