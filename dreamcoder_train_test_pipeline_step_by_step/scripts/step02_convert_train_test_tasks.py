#!/usr/bin/env python3
import csv
import json
import pathlib
import re
import sys

if len(sys.argv) != 5:
    raise SystemExit("Usage: step02_convert_train_test_tasks.py <train_json> <test_json> <output_dir> <max_train_tasks_or_0>")

train_path = pathlib.Path(sys.argv[1]).resolve()
test_path = pathlib.Path(sys.argv[2]).resolve()
out_dir = pathlib.Path(sys.argv[3]).resolve()
max_train_tasks = int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)


def load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def infer_type(candidates):
    """Infer a type from every occurrence of one argument/output position across
    all examples, not just the first. A single example whose list output happens
    to be empty is not enough evidence to call it list(bool): Python's
    all(isinstance(x, bool) for x in []) is vacuously True, which previously
    misclassified empty-list outputs as list(bool) even though this dataset
    never contains a genuine boolean list.
    """
    saw_list = False
    saw_nonempty_list = False
    all_list_elems_bool = True
    saw_bool_scalar = False
    saw_int_scalar = False
    for v in candidates:
        if isinstance(v, bool):
            saw_bool_scalar = True
        elif isinstance(v, int):
            saw_int_scalar = True
        elif isinstance(v, list):
            saw_list = True
            if v:
                saw_nonempty_list = True
                if not all(isinstance(x, bool) for x in v):
                    all_list_elems_bool = False
        else:
            raise ValueError(f"Unsupported value type: {v!r}")
    if saw_list:
        if saw_nonempty_list:
            return "list(bool)" if all_list_elems_bool else "list(int)"
        # Every example produced an empty list at this position: there is no
        # evidence of element type, and this dataset has no genuine
        # list(bool) values, so list(int) is the safe default.
        return "list(int)"
    if saw_bool_scalar:
        return "bool"
    if saw_int_scalar:
        return "int"
    raise ValueError(f"Could not infer type from: {candidates!r}")


def request_from_examples(examples):
    if not examples:
        return None
    arity = len(examples[0].get("inputs", []))
    input_types = [
        infer_type([ex.get("inputs", [])[i] for ex in examples])
        for i in range(arity)
    ]
    output_type = infer_type([ex.get("output") for ex in examples])
    return " -> ".join(input_types + [output_type])


def safe_name(program):
    name = str(program or "program")
    name = name.replace("|", "_").replace(",", "_")
    name = re.sub(r"[^A-Za-z0-9_+*<>=-]+", "_", name)
    return name[:100]


def convert_one_dataset(data, split, max_tasks=0):
    tasks = []
    skipped = []
    used = 0
    for original_index, task in enumerate(data):
        if max_tasks and used >= max_tasks:
            break
        program = str(task.get("program", "") or "")
        examples_raw = task.get("examples", [])
        usable_examples = []
        skipped_null = 0
        for ex in examples_raw:
            if ex.get("output") is None:
                skipped_null += 1
                continue
            usable_examples.append({"inputs": ex.get("inputs", []), "output": ex.get("output")})
        if not program:
            skipped.append({"split": split, "original_index": original_index, "reason": "missing_program"})
            continue
        if not usable_examples:
            skipped.append({"split": split, "original_index": original_index, "program": program, "reason": "no_usable_examples"})
            continue
        try:
            request = request_from_examples(usable_examples)
        except Exception as e:
            skipped.append({"split": split, "original_index": original_index, "program": program, "reason": str(e)})
            continue
        task_id = f"{split}_{used:05d}"
        task_name = f"{split}_task_{used:05d}__{safe_name(program)}"
        tasks.append({
            "split": split,
            "task_id": task_id,
            "original_index": original_index,
            "task_name": task_name,
            "request": request,
            "reference_program": program,
            "examples": usable_examples,
            "usable_examples": len(usable_examples),
            "skipped_null_examples": skipped_null,
        })
        used += 1
    return tasks, skipped

train_tasks, train_skipped = convert_one_dataset(load(train_path), "train", max_train_tasks)
test_tasks, test_skipped = convert_one_dataset(load(test_path), "test", 0)
all_skipped = train_skipped + test_skipped

for split, tasks in [("train", train_tasks), ("test", test_tasks)]:
    json_path = out_dir / f"step02_{split}_tasks.json"
    jsonl_path = out_dir / f"step02_{split}_tasks.jsonl"
    csv_path = out_dir / f"step02_{split}_tasks.csv"
    json_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    fieldnames = ["split", "task_id", "original_index", "task_name", "request", "reference_program", "usable_examples", "skipped_null_examples"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tasks:
            w.writerow({k: t.get(k, "") for k in fieldnames})

(out_dir / "step02_skipped_tasks.json").write_text(json.dumps(all_skipped, indent=2), encoding="utf-8")
summary = {
    "train_tasks": len(train_tasks),
    "test_tasks": len(test_tasks),
    "skipped_tasks": len(all_skipped),
    "max_train_tasks": max_train_tasks,
    "train_json": str(out_dir / "step02_train_tasks.json"),
    "test_json": str(out_dir / "step02_test_tasks.json"),
}
(out_dir / "step02_conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
