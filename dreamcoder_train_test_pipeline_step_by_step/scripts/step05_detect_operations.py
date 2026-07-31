#!/usr/bin/env python3
import csv
import pathlib
import re
import sys
from collections import Counter

if len(sys.argv) != 3:
    raise SystemExit("Usage: step05_detect_operations.py <test_results_csv> <output_dir>")

input_csv = pathlib.Path(sys.argv[1]).resolve()
out_dir = pathlib.Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z_?<>!=]+|\+\+|[+*/-]|\d+|\$\d+")
IGNORED = {"lambda", "true", "false"}
KNOWN_SOLUTION_TOKENS = {
    "reverse", "sort", "sum", "length", "car", "cdr", "index", "cons", "empty", "empty?", "++",
    "map", "mapi", "filter", "fold", "reduce", "reducei", "unfold", "range",
    "+", "-", "*", "/", "mod", "negate", "gt?", "lt?", "eq?", "is-square", "is-prime", "if",
}

def ref_ops(program):
    ops = []
    for st in str(program or "").split("|"):
        p = st.split(",")[0].strip().upper()
        if p and p not in {"LIST", "INT", "BOOL"}:
            ops.append(p)
    return ops

def solution_tokens(program):
    tokens = []
    for token in TOKEN_RE.findall(str(program or "")):
        if token in IGNORED or re.fullmatch(r"\d+|\$\d+", token):
            continue
        tokens.append(token)
    return tokens

with input_csv.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

ref_counter = Counter()
sol_counter = Counter()
unknown = Counter()
operation_rows = []
for row in rows:
    rops = ref_ops(row.get("reference_program", ""))
    stoks = solution_tokens(row.get("solution", ""))
    ref_counter.update(rops)
    sol_counter.update(stoks)
    unknown.update([t for t in stoks if t not in KNOWN_SOLUTION_TOKENS])
    operation_rows.append({
        "task_id": row.get("task_id", ""),
        "solved": row.get("solved", ""),
        "reference_operations": ",".join(rops),
        "solution_tokens": ",".join(stoks),
        "unknown_solution_tokens": ",".join(sorted(set(t for t in stoks if t not in KNOWN_SOLUTION_TOKENS))),
    })

with (out_dir / "step05_operation_check.csv").open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["task_id", "solved", "reference_operations", "solution_tokens", "unknown_solution_tokens"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(operation_rows)

for name, counter in [("reference_operations", ref_counter), ("solution_tokens", sol_counter), ("unknown_solution_tokens", unknown)]:
    with (out_dir / f"step05_{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "count"])
        for token, count in sorted(counter.items()):
            w.writerow([token, count])

summary = (
    f"Tasks: {len(rows)}\n"
    f"Different reference operations: {len(ref_counter)}\n"
    f"Different solution tokens: {len(sol_counter)}\n"
    f"Unknown solution tokens: {len(unknown)}\n"
)
(out_dir / "step05_operation_check_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
if unknown:
    print("Unknown tokens found. Extend the mapping in step06_normalize_programs.py.")
else:
    print("All solution tokens are covered by the current token map.")
