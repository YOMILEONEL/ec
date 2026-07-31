#!/usr/bin/env python3
import json
import pathlib
import sys
from collections import Counter

if len(sys.argv) != 4:
    raise SystemExit("Usage: step01_validate_datasets.py <train_json> <test_json> <output_dir>")

train_path = pathlib.Path(sys.argv[1]).resolve()
test_path = pathlib.Path(sys.argv[2]).resolve()
out_dir = pathlib.Path(sys.argv[3]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return data


def type_name(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, list):
        if value and all(isinstance(x, bool) for x in value):
            return "list(bool)"
        return "list(int)"
    if value is None:
        return "null"
    return type(value).__name__


def summarize(path):
    data = load_dataset(path)
    lengths = []
    examples_per_task = []
    null_outputs = 0
    tasks_only_null = 0
    input_arities = Counter()
    output_types = Counter()
    missing_program = 0
    missing_examples = 0

    for task in data:
        program = str(task.get("program", "") or "")
        if not program:
            missing_program += 1
        operations = [p for p in program.split("|") if p and p not in {"LIST", "INT", "BOOL"}]
        lengths.append(len(operations))

        examples = task.get("examples", [])
        if not examples:
            missing_examples += 1
        examples_per_task.append(len(examples))

        usable = 0
        for ex in examples:
            output = ex.get("output")
            if output is None:
                null_outputs += 1
                output_types["null"] += 1
            else:
                usable += 1
                output_types[type_name(output)] += 1
            input_arities[str(len(ex.get("inputs", [])))] += 1
        if examples and usable == 0:
            tasks_only_null += 1

    return {
        "path": str(path),
        "tasks": len(data),
        "program_length_min": min(lengths) if lengths else 0,
        "program_length_max": max(lengths) if lengths else 0,
        "examples_per_task_min": min(examples_per_task) if examples_per_task else 0,
        "examples_per_task_max": max(examples_per_task) if examples_per_task else 0,
        "null_outputs": null_outputs,
        "tasks_with_only_null_outputs": tasks_only_null,
        "input_arities": dict(input_arities),
        "output_types": dict(output_types),
        "missing_program": missing_program,
        "missing_examples": missing_examples,
    }

summary = {
    "train": summarize(train_path),
    "test": summarize(test_path),
}

(out_dir / "step01_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
lines = []
for split in ["train", "test"]:
    s = summary[split]
    lines.extend([
        f"{split.upper()} DATASET",
        f"Path: {s['path']}",
        f"Tasks: {s['tasks']}",
        f"Program length: {s['program_length_min']}..{s['program_length_max']}",
        f"Examples per task: {s['examples_per_task_min']}..{s['examples_per_task_max']}",
        f"Null outputs: {s['null_outputs']}",
        f"Tasks with only null outputs: {s['tasks_with_only_null_outputs']}",
        f"Input arities: {s['input_arities']}",
        f"Output types: {s['output_types']}",
        "",
    ])
(out_dir / "step01_dataset_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2))
