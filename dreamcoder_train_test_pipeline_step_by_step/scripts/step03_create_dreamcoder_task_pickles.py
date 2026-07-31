#!/usr/bin/env python3
import json
import os
import pathlib
import pickle
import sys

if len(sys.argv) != 5:
    raise SystemExit("Usage: step03_create_dreamcoder_task_pickles.py <train_tasks_json> <test_tasks_json> <output_dir> <dreamcoder_repo_root>")

train_json = pathlib.Path(sys.argv[1]).resolve()
test_json = pathlib.Path(sys.argv[2]).resolve()
out_dir = pathlib.Path(sys.argv[3]).resolve()
repo_root = pathlib.Path(sys.argv[4]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo_root))
os.chdir(str(repo_root))

try:
    from dreamcoder.task import Task
    from dreamcoder.type import arrow, tint, tbool, tlist
except Exception as e:
    raise SystemExit(f"Could not import DreamCoder modules from {repo_root}: {e}")


def parse_type(t):
    t = t.strip()
    if t == "int":
        return tint
    if t == "bool":
        return tbool
    if t == "list(int)":
        return tlist(tint)
    if t == "list(bool)":
        return tlist(tbool)
    raise ValueError(f"Unsupported type for DreamCoder conversion: {t}")


def parse_request(req):
    parts = [p.strip() for p in req.split("->")]
    types = [parse_type(p) for p in parts]
    if len(types) == 1:
        return types[0]
    return arrow(*types)


def make_tasks(path):
    specs = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for spec in specs:
        request = parse_request(spec["request"])
        examples = [(tuple(ex["inputs"]), ex["output"]) for ex in spec["examples"]]
        task = Task(spec["task_name"], request, examples)
        task.pbe_task_id = spec["task_id"]
        task.pbe_reference_program = spec["reference_program"]
        task.pbe_request_string = spec["request"]
        task.pbe_split = spec["split"]
        task.pbe_original_index = spec["original_index"]
        tasks.append(task)
    return tasks

train_tasks = make_tasks(train_json)
test_tasks = make_tasks(test_json)

train_pickle = out_dir / "step03_train_tasks.pkl"
test_pickle = out_dir / "step03_test_tasks.pkl"
train_pickle.write_bytes(pickle.dumps(train_tasks))
test_pickle.write_bytes(pickle.dumps(test_tasks))

summary = (
    f"Created DreamCoder Task objects\n"
    f"Train tasks: {len(train_tasks)}\n"
    f"Test tasks: {len(test_tasks)}\n"
    f"Train pickle: {train_pickle}\n"
    f"Test pickle: {test_pickle}\n"
    f"DreamCoder repo: {repo_root}\n"
)
(out_dir / "step03_task_pickle_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
