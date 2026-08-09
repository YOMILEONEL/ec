#!/usr/bin/env python3
import csv
import json
import pathlib
import sys

if len(sys.argv) != 5:
    raise SystemExit("Usage: step08_summarize_results.py <metrics_csv> <metrics_summary_json> <step04_summary_json> <output_dir>")

metrics_csv = pathlib.Path(sys.argv[1]).resolve()
metrics_summary_json = pathlib.Path(sys.argv[2]).resolve()
step04_summary_json = pathlib.Path(sys.argv[3]).resolve()
out_dir = pathlib.Path(sys.argv[4]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

with metrics_csv.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
metrics_summary = json.loads(metrics_summary_json.read_text(encoding="utf-8"))
step04_summary = json.loads(step04_summary_json.read_text(encoding="utf-8"))

complete_csv = out_dir / "dreamcoder_test_results_complete.csv"
with complete_csv.open("w", newline="", encoding="utf-8") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

solved_csv = out_dir / "dreamcoder_test_solved_tasks.csv"
unsolved_csv = out_dir / "dreamcoder_test_unsolved_tasks.csv"
for path, wanted in [(solved_csv, True), (unsolved_csv, False)]:
    subset = [r for r in rows if (str(r.get("solved", "")).lower() == "true") == wanted]
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(subset)

summary_txt = out_dir / "step08_summary.txt"
lines = []
lines.append("DreamCoder train/test summary")
lines.append("")
lines.append(f"Train tasks used: {step04_summary.get('train_tasks')}")
lines.append(f"Test tasks used: {step04_summary.get('test_tasks')}")
lines.append(f"Iterations: {step04_summary.get('iterations')}")
lines.append(f"Enumeration timeout: {step04_summary.get('enumeration_timeout')}")
lines.append(f"Testing timeout: {step04_summary.get('testing_timeout')}")
lines.append(f"Recognition model enabled: {step04_summary.get('use_recognition_model')}")
lines.append(f"Grammar consolidation disabled: {step04_summary.get('no_consolidation')}")
lines.append("")
lines.append(f"Test tasks: {metrics_summary.get('test_tasks')}")
lines.append(f"Solved test tasks: {metrics_summary.get('test_solved')}")
lines.append(f"Accuracy: {metrics_summary.get('test_accuracy'):.6f}")
lines.append(f"Trivial solutions: {metrics_summary.get('trivial_solutions')}")
lines.append(f"Normalized exact matches: {metrics_summary.get('normalized_exact_matches')}")
lines.append(f"Abstract exact matches: {metrics_summary.get('abstract_exact_matches')}")
lines.append("")
for prefix in ["normalized", "abstract"]:
    lines.append(prefix.capitalize() + " metrics over all test tasks:")
    for metric in ["operation_score", "position_score", "sequence_score", "edit_score"]:
        lines.append(f"  {metric}: {metrics_summary.get(prefix + '_' + metric + '_mean'):.6f}")
    lines.append(prefix.capitalize() + " metrics over solved test tasks only:")
    for metric in ["operation_score", "position_score", "sequence_score", "edit_score"]:
        lines.append(f"  {metric}: {metrics_summary.get(prefix + '_' + metric + '_mean_solved_only'):.6f}")
    lines.append("")
lines.append("Generated files:")
for p in [complete_csv, solved_csv, unsolved_csv, metrics_summary_json]:
    lines.append(str(p))
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_txt.read_text(encoding="utf-8"))
