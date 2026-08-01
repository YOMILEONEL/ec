#!/usr/bin/env python3
"""
Step 04: run DreamCoder on training tasks and evaluate held-out test tasks.

Robustness improvements in this version:
1. Searches several possible result attributes for frontiers.
2. Matches frontiers both by task object and task name.
3. Streams DreamCoder stdout into step04_stdout.log while the run is active.
4. Parses DreamCoder HIT lines from stdout as a fallback when held-out testing
   hits are printed but not stored in the result frontier structure.
5. Marks whether a solution came from a result frontier or from stdout fallback.
"""
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

if len(sys.argv) not in {13, 14}:
    raise SystemExit(
        "Usage: step04_run_dreamcoder_train_test.py "
        "<train_pickle> <test_pickle> <train_spec_json> <test_spec_json> <output_dir> "
        "<dreamcoder_repo_root> <enumeration_timeout> <testing_timeout> <iterations> "
        "<frontier_size> <use_recognition:true|false> <no_consolidation:true|false> [cpus]"
    )

train_pickle = pathlib.Path(sys.argv[1]).resolve()
test_pickle = pathlib.Path(sys.argv[2]).resolve()
train_spec_json = pathlib.Path(sys.argv[3]).resolve()
test_spec_json = pathlib.Path(sys.argv[4]).resolve()
out_dir = pathlib.Path(sys.argv[5]).resolve()
repo_root = pathlib.Path(sys.argv[6]).resolve()
enum_timeout = sys.argv[7]
testing_timeout = sys.argv[8]
iterations = sys.argv[9]
frontier_size = sys.argv[10]
use_recognition = sys.argv[11]
no_consolidation = sys.argv[12]
cpus = sys.argv[13] if len(sys.argv) == 14 else os.environ.get("DREAMCODER_CPUS", "1")
scripts_dir = pathlib.Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)

# Compression/search hyperparameters. Not exposed as positional CLI args (to
# avoid breaking run_all.sh / the notebook's existing call signature) - read
# from the environment instead, with defaults matching the official list-domain
# paper experiments (see the repository's "official_experiments" file), not
# ecIterator's own much more conservative defaults. Notably ecIterator defaults
# to arity=0, which means consolidation can only invent closed, zero-argument
# expressions - no reusable *parameterized* library functions at all. That is
# very likely why earlier runs found compression candidates but reported
# "No improvement possible": there was nothing worth inventing under arity=0.
arity = os.environ.get("DREAMCODER_ARITY", "3")
pseudo_counts = os.environ.get("DREAMCODER_PSEUDOCOUNTS", "30")
aic = os.environ.get("DREAMCODER_AIC", "1.0")
structure_penalty = os.environ.get("DREAMCODER_STRUCTURE_PENALTY", "1.5")
top_k = os.environ.get("DREAMCODER_TOPK", "2")
# Recognition training time/steps default to None in ecIterator, which makes
# it silently fall back to enumerationTimeout - i.e. with a 15s enumeration
# budget the recognition model only ever gets 15s to train, which is likely
# too little to learn anything useful. Give it its own budget.
recognition_timeout = os.environ.get("DREAMCODER_RECOGNITION_TIMEOUT", "")
recognition_steps = os.environ.get("DREAMCODER_RECOGNITION_STEPS", "")
if use_recognition.lower() == "true" and not recognition_timeout and not recognition_steps:
    recognition_timeout = enum_timeout

runner = out_dir / "step04_internal_train_test_runner.py"
runner.write_text(textwrap.dedent(r'''
#!/usr/bin/env python3
import csv
import importlib.util
import json
import os
import pathlib
import pickle
import sys
import traceback

train_pickle = pathlib.Path(sys.argv[1]).resolve()
test_pickle = pathlib.Path(sys.argv[2]).resolve()
train_spec_json = pathlib.Path(sys.argv[3]).resolve()
test_spec_json = pathlib.Path(sys.argv[4]).resolve()
output_dir = pathlib.Path(sys.argv[5]).resolve()
repo_root = pathlib.Path(sys.argv[6]).resolve()
enum_timeout = float(sys.argv[7])
testing_timeout = float(sys.argv[8])
iterations = int(sys.argv[9])
frontier_size = int(sys.argv[10])
use_recognition = sys.argv[11].lower() in {"true", "1", "yes"}
no_consolidation = sys.argv[12].lower() in {"true", "1", "yes"}
cpus = int(sys.argv[13])
scripts_dir = pathlib.Path(sys.argv[14]).resolve()
arity = int(sys.argv[15])
pseudo_counts = float(sys.argv[16])
aic = float(sys.argv[17])
structure_penalty = float(sys.argv[18])
top_k = int(sys.argv[19])
recognition_timeout = float(sys.argv[20]) if sys.argv[20] else None
recognition_steps = int(sys.argv[21]) if sys.argv[21] else None

output_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(repo_root))
os.chdir(str(repo_root))


def load_list_primitives():
    # Preferred: a DeepCoder-tailored primitive set (map/filter/zipWith/fold/
    # scanl1/...) that lives in a real, importable module (step04_deepcoder_
    # primitives.py, next to this script) rather than being defined inline in
    # this throwaway __main__ runner. DreamCoder's parallel solver moves
    # enumerated programs between worker processes via dill/multiprocess,
    # which reconstructs functions defined in __main__ as new, non-identical
    # objects; a real module is resolved consistently by import path in every
    # process instead.
    try:
        from step04_deepcoder_primitives import build_deepcoder_primitives
        primitives = build_deepcoder_primitives()
        if primitives:
            return primitives, "step04_deepcoder_primitives:build_deepcoder_primitives()"
    except Exception:
        (output_dir / "step04_deepcoder_primitives_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    # Fallback: auto-discover whatever primitive set DreamCoder ships with, in
    # case the DeepCoder-tailored set above fails to import in this checkout.
    candidates = []
    for module_name in [
        "dreamcoder.domains.list.listPrimitives",
        "dreamcoder.domains.list.main",
        "domains.list.listPrimitives",
    ]:
        try:
            mod = __import__(module_name, fromlist=["*"])
            candidates.append((module_name, mod))
        except Exception:
            pass

    list_py = repo_root / "bin" / "list.py"
    if list_py.exists():
        try:
            spec = importlib.util.spec_from_file_location("dreamcoder_bin_list_for_train_test_runner", str(list_py))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            candidates.append(("bin/list.py", mod))
        except Exception:
            (output_dir / "step04_import_bin_list_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    for origin, mod in candidates:
        for name, obj in vars(mod).items():
            if callable(obj) and "primitive" in name.lower():
                try:
                    value = obj()
                    if isinstance(value, list) and value:
                        return value, f"{origin}:{name}()"
                except Exception:
                    pass
        for name, value in vars(mod).items():
            if isinstance(value, list) and value:
                primitive_like = [x for x in value if hasattr(x, "tp") or hasattr(x, "type") or x.__class__.__name__.lower().endswith("primitive")]
                if len(primitive_like) >= max(1, len(value) // 2):
                    return value, f"{origin}:{name}"
    raise RuntimeError("Could not locate DreamCoder list-domain primitives.")


def load_feature_extractor():
    # ecIterator silently disables useRecognitionModel (with just a warning)
    # if no featureExtractor class is given - it never raises, so this was
    # going unnoticed. LearnedFeatureExtractor is the list-domain recurrent
    # feature extractor bin/list.py normally wires up for recognition.
    try:
        from dreamcoder.domains.list.main import LearnedFeatureExtractor
        return LearnedFeatureExtractor, None
    except Exception:
        return None, traceback.format_exc()


def spec_map(path):
    specs = json.loads(path.read_text(encoding="utf-8"))
    return {s["task_name"]: s for s in specs}


def _entries_from_frontier(frontier):
    if frontier is None:
        return []
    entries = getattr(frontier, "entries", None)
    if entries is None and isinstance(frontier, (list, tuple)):
        # Sometimes a frontier-like collection is already stored as a list.
        return list(frontier)
    if entries is None:
        return []
    if isinstance(entries, (list, tuple)):
        return list(entries)
    return [entries]


def _iter_frontier_candidates(result, task):
    # Direct mapping attributes seen across DreamCoder variants.
    for attr in [
        "allFrontiers",
        "frontiers",
        "testingFrontiers",
        "testFrontiers",
        "heldoutFrontiers",
        "trainingFrontiers",
    ]:
        mapping = getattr(result, attr, None)
        if not isinstance(mapping, dict):
            continue
        if task in mapping:
            yield mapping[task], f"{attr}[task]"
        for key, frontier in mapping.items():
            if getattr(key, "name", None) == getattr(task, "name", None):
                yield frontier, f"{attr}[task.name]"
            elif isinstance(key, str) and key == getattr(task, "name", None):
                yield frontier, f"{attr}[name]"

    # History attributes. Search latest non-empty frontier first.
    for attr in ["frontiersOverTime", "testingFrontiersOverTime", "testFrontiersOverTime"]:
        history_map = getattr(result, attr, None)
        if not isinstance(history_map, dict):
            continue
        histories = []
        if task in history_map:
            histories.append((history_map[task], f"{attr}[task]"))
        for key, history in history_map.items():
            if getattr(key, "name", None) == getattr(task, "name", None):
                histories.append((history, f"{attr}[task.name]"))
            elif isinstance(key, str) and key == getattr(task, "name", None):
                histories.append((history, f"{attr}[name]"))
        for history, source in histories:
            if isinstance(history, list):
                for frontier in reversed(history):
                    yield frontier, source
            else:
                yield history, source


def frontier_entries_from_result(result, task):
    diagnostics = []
    for frontier, source in _iter_frontier_candidates(result, task):
        entries = _entries_from_frontier(frontier)
        diagnostics.append({"source": source, "entries": len(entries)})
        if entries:
            return entries, source, diagnostics
    return [], "", diagnostics


def distinct_output_count(task):
    """Count distinct example outputs for a task. If a 'solved' task has only
    one distinct output across all of its examples (e.g. every example happens
    to output []), a constant/degenerate program that ignores its input can
    satisfy every example without encoding the intended logic at all. Such a
    task cannot actually verify that DreamCoder found a generalizing solution.
    """
    try:
        outputs = [y for _xs, y in task.examples]
    except Exception:
        return None
    seen = []
    for y in outputs:
        if y not in seen:
            seen.append(y)
    return len(seen)


def export_rows(result, tasks, specs, split, csv_path, diagnostics_path):
    rows = []
    diagnostics = []
    for task in tasks:
        entries, source, diag = frontier_entries_from_result(result, task)
        solved = len(entries) > 0
        best = entries[0] if solved else None
        spec = specs.get(task.name, {})
        task_diagnostics = {"task_name": task.name, "frontier_source": source, "diagnostics": diag}
        diagnostics.append(task_diagnostics)
        distinct_outputs = distinct_output_count(task)
        rows.append({
            "split": split,
            "task_id": spec.get("task_id", getattr(task, "pbe_task_id", "")),
            "original_index": spec.get("original_index", getattr(task, "pbe_original_index", "")),
            "task_name": task.name,
            "request": spec.get("request", str(getattr(task, "request", ""))),
            "reference_program": spec.get("reference_program", getattr(task, "pbe_reference_program", "")),
            "solved": bool(solved),
            "solution": str(getattr(best, "program", "")) if best is not None else "",
            "log_prior": getattr(best, "logPrior", "") if best is not None else "",
            "log_likelihood": getattr(best, "logLikelihood", "") if best is not None else "",
            "solution_source": "frontier" if solved else "",
            "stdout_fallback_used": 0,
            "frontier_source": source,
            "distinct_example_outputs": distinct_outputs if distinct_outputs is not None else "",
            "degenerate_example_set": bool(distinct_outputs is not None and distinct_outputs <= 1),
        })
    fieldnames = [
        "split", "task_id", "original_index", "task_name", "request", "reference_program",
        "solved", "solution", "log_prior", "log_likelihood", "solution_source",
        "stdout_fallback_used", "frontier_source", "distinct_example_outputs", "degenerate_example_set",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return rows

try:
    from dreamcoder.grammar import Grammar
    from dreamcoder.dreamcoder import ecIterator

    train_tasks = pickle.loads(train_pickle.read_bytes())
    test_tasks = pickle.loads(test_pickle.read_bytes())
    train_specs = spec_map(train_spec_json)
    test_specs = spec_map(test_spec_json)
    primitives, primitive_source = load_list_primitives()
    grammar = Grammar.uniform(primitives)

    feature_extractor, feature_extractor_error = (None, None)
    if use_recognition:
        feature_extractor, feature_extractor_error = load_feature_extractor()
        if feature_extractor is None:
            (output_dir / "step04_feature_extractor_error.txt").write_text(feature_extractor_error, encoding="utf-8")

    setup = {
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "primitive_source": primitive_source,
        "primitives": len(primitives),
        "enumeration_timeout": enum_timeout,
        "testing_timeout": testing_timeout,
        "iterations": iterations,
        "frontier_size": frontier_size,
        "use_recognition_model": use_recognition,
        "feature_extractor": getattr(feature_extractor, "__name__", None),
        "no_consolidation": no_consolidation,
        "cpus": cpus,
        "arity": arity,
        "pseudo_counts": pseudo_counts,
        "aic": aic,
        "structure_penalty": structure_penalty,
        "top_k": top_k,
        "recognition_timeout": recognition_timeout,
        "recognition_steps": recognition_steps,
    }
    (output_dir / "step04_setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    (output_dir / "step04_setup.txt").write_text("\n".join(f"{k}: {v}" for k, v in setup.items()) + "\n", encoding="utf-8")

    output_prefix = str(output_dir / "dreamcoder_train_test")
    kwargs = dict(
        iterations=iterations,
        enumerationTimeout=enum_timeout,
        maximumFrontier=frontier_size,
        solver="ocaml",
        useRecognitionModel=use_recognition,
        noConsolidation=no_consolidation,
        compressor="ocaml",
        testingTasks=test_tasks,
        testingTimeout=testing_timeout,
        evaluationTimeout=0.0005,
        outputPrefix=output_prefix,
        CPUs=cpus,
        cuda=False,
        arity=arity,
        pseudoCounts=pseudo_counts,
        aic=aic,
        structurePenalty=structure_penalty,
        topK=top_k,
    )
    if recognition_timeout is not None:
        kwargs["recognitionTimeout"] = recognition_timeout
    if recognition_steps is not None:
        kwargs["recognitionSteps"] = recognition_steps
    if feature_extractor is not None:
        kwargs["featureExtractor"] = feature_extractor

    # DreamCoder forks differ slightly in accepted keyword arguments.
    # Try the richest configuration first and remove compatibility-sensitive keys if necessary.
    tried = []
    for removable in [[], ["cuda"], ["CPUs"], ["cuda", "CPUs"], ["useRecognitionModel"]]:
        candidate = dict(kwargs)
        for key in removable:
            candidate.pop(key, None)
        try:
            tried.append(sorted(candidate.keys()))
            iterator = ecIterator(grammar, train_tasks, **candidate)
            break
        except TypeError:
            iterator = None
    if iterator is None:
        raise TypeError(f"Could not call ecIterator with tested keyword sets: {tried}")

    last_result = None
    for result in iterator:
        last_result = result
    if last_result is None:
        raise RuntimeError("ecIterator did not yield a result")

    result_pickle = output_dir / "step04_dreamcoder_train_test_result.pkl"
    result_pickle.write_bytes(pickle.dumps(last_result))

    train_csv = output_dir / "step04_train_results.csv"
    test_csv = output_dir / "step04_test_results.csv"
    train_rows = export_rows(last_result, train_tasks, train_specs, "train", train_csv, output_dir / "step04_train_frontier_diagnostics.json")
    test_rows = export_rows(last_result, test_tasks, test_specs, "test", test_csv, output_dir / "step04_test_frontier_diagnostics.json")

    summary = {
        **setup,
        "train_solved": sum(1 for row in train_rows if str(row["solved"]).lower() == "true"),
        "test_solved": sum(1 for row in test_rows if str(row["solved"]).lower() == "true"),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "result_pickle": str(result_pickle),
        "stdout_fallback_note": "Outer wrapper may patch CSV/summary from stdout HIT lines after this runner exits.",
    }
    (output_dir / "step04_train_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

except Exception:
    error = traceback.format_exc()
    (output_dir / "step04_error.txt").write_text(error, encoding="utf-8")
    print(error)
    sys.exit(1)
'''), encoding="utf-8")


def parse_stdout_hits(stdout_text):
    pattern = re.compile(
        r"^HIT\s+(?P<task_name>\S+)\s+w/\s+(?P<program>.*?)\s*;\s*"
        r"log prior\s*=\s*(?P<log_prior>[-+0-9.eE]+)\s*;\s*"
        r"log likelihood\s*=\s*(?P<log_likelihood>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    hits = []
    for match in pattern.finditer(stdout_text):
        task_name = match.group("task_name")
        split = "test" if task_name.startswith("test_task_") else "train" if task_name.startswith("train_task_") else "unknown"
        hits.append({
            "split": split,
            "task_name": task_name,
            "solution": match.group("program").strip(),
            "log_prior": match.group("log_prior"),
            "log_likelihood": match.group("log_likelihood"),
        })
    return hits


def read_csv(path):
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patch_csv_from_stdout(csv_path, hits, split):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        return 0, 0
    for extra in ["solution_source", "stdout_fallback_used", "frontier_source"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    hit_map = {h["task_name"]: h for h in hits if h["split"] == split}
    already_solved = 0
    patched = 0
    for row in rows:
        solved = str(row.get("solved", "")).lower() in {"true", "1", "yes"}
        if solved:
            already_solved += 1
            if not row.get("solution_source"):
                row["solution_source"] = "frontier"
            row["stdout_fallback_used"] = row.get("stdout_fallback_used", "0") or "0"
            continue
        hit = hit_map.get(row.get("task_name"))
        if hit:
            row["solved"] = "True"
            row["solution"] = hit["solution"]
            row["log_prior"] = hit["log_prior"]
            row["log_likelihood"] = hit["log_likelihood"]
            row["solution_source"] = "stdout_fallback"
            row["stdout_fallback_used"] = "1"
            row["frontier_source"] = row.get("frontier_source", "")
            patched += 1
        else:
            row["solution_source"] = row.get("solution_source", "")
            row["stdout_fallback_used"] = row.get("stdout_fallback_used", "0") or "0"
    write_csv(csv_path, rows, fieldnames)
    return already_solved, patched


def update_summary_after_fallback(output_dir, hits, train_patch, test_patch):
    summary_path = output_dir / "step04_train_test_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_rows, _ = read_csv(output_dir / "step04_train_results.csv")
    test_rows, _ = read_csv(output_dir / "step04_test_results.csv")
    summary.update({
        "stdout_train_hits": sum(1 for h in hits if h["split"] == "train"),
        "stdout_test_hits": sum(1 for h in hits if h["split"] == "test"),
        "frontier_train_solved_before_stdout_fallback": train_patch[0],
        "frontier_test_solved_before_stdout_fallback": test_patch[0],
        "train_patched_from_stdout": train_patch[1],
        "test_patched_from_stdout": test_patch[1],
        "train_solved": sum(1 for r in train_rows if str(r.get("solved", "")).lower() in {"true", "1", "yes"}),
        "test_solved": sum(1 for r in test_rows if str(r.get("solved", "")).lower() in {"true", "1", "yes"}),
        "stdout_fallback_used": bool(train_patch[1] or test_patch[1]),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


cmd = [
    sys.executable, str(runner), str(train_pickle), str(test_pickle), str(train_spec_json), str(test_spec_json),
    str(out_dir), str(repo_root), str(enum_timeout), str(testing_timeout), str(iterations), str(frontier_size),
    str(use_recognition), str(no_consolidation), str(cpus), str(scripts_dir),
    str(arity), str(pseudo_counts), str(aic), str(structure_penalty), str(top_k),
    str(recognition_timeout), str(recognition_steps),
]
print("Running DreamCoder train/test workflow:")
print(" ".join(cmd))

stdout_log = out_dir / "step04_stdout.log"
stdout_chunks = []
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
with stdout_log.open("w", encoding="utf-8") as log_file:
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_chunks.append(line)
        log_file.write(line)
        log_file.flush()
        print(line, end="")
    return_code = proc.wait()

stdout_text = "".join(stdout_chunks)
hits = parse_stdout_hits(stdout_text)
(out_dir / "step04_stdout_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
if hits:
    with (out_dir / "step04_stdout_hits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "task_name", "solution", "log_prior", "log_likelihood"])
        writer.writeheader()
        writer.writerows(hits)

if return_code != 0:
    raise SystemExit(return_code)

train_patch = patch_csv_from_stdout(out_dir / "step04_train_results.csv", hits, "train")
test_patch = patch_csv_from_stdout(out_dir / "step04_test_results.csv", hits, "test")
update_summary_after_fallback(out_dir, hits, train_patch, test_patch)

print("\nStdout HIT fallback summary:")
print(json.dumps({
    "stdout_train_hits": sum(1 for h in hits if h["split"] == "train"),
    "stdout_test_hits": sum(1 for h in hits if h["split"] == "test"),
    "train_patched_from_stdout": train_patch[1],
    "test_patched_from_stdout": test_patch[1],
}, indent=2))
