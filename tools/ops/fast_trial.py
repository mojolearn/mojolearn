#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The per-estimator `fast` trial on a BUILT install (.github/workflows/fast-trial.yml).

    fast_trial.py run --python VPY --tests-root SOURCE/python/mojolearn/tests --out OUT
                      --families 0xfffffff [--exact MASK] [--shard I --shards N] [--workers W]
    fast_trial.py summary --out OUT [--out OUT2 ...] --ledger fast_ledger.json --md SUMMARY.md

WHAT IT RUNS. `test_fast_selector.py -k BuiltInstallFamilies` against the installed wheel (through
tools/run_python_tests.py INSTALLED_BOOTSTRAP, so the source tree is never imported as the runtime) with
MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES=<mask> (the door families of core/mode_answer.mojo that answer `fast` for the
trial) and MOJOLEARN_FAST_LEDGER=<unit ledger>. `--exact MASK` adds a second trial with exactly a candidate set:
a trial with every bit answers the shared RESOLVER / METADATA steps for all families at once, so a family that is
whole only through them is not whole alone (fx-fast-selector2-RESULT.md, Doubts).

PARALLEL PLAN. A unit is one (trial, estimator): MOJOLEARN_FAST_ESTIMATORS=<name> picks it, so each estimator runs
in its own process (a crash in one estimator never hides the rest). Units are spread across `--workers` processes
(default: the cores, lowered only when MemAvailable / --unit-gib is smaller), and across matrix shards by index
(`--shard I --shards N`: unit k runs on shard k % N). Each unit's peak RSS (wait4) and seconds are recorded in
OUT/units.tsv, so --unit-gib can come from a measurement next time.

THE PRODUCT IS THE REPORT. A `mixed` row makes its pytest fail; that is expected and is not a failure of this tool.
`run` exits non-zero only when a unit left no ledger (crashed or timed out); `summary` writes SUMMARY.md with one
line per door family (answered / mixed / refused / unavailable / crashed) and, for every row that is not whole,
each refusing or differing operation with its message (the message names the door the next lane edits).

THE CPU COVERAGE MODE (lane fx-coverage-smokes, 2026-11-20; Andrew: "do we have cpu paths for all of our gpu paths?
maybe we have a smoke test for this"). The same loop, the same workflow (fast-trial.yml with mode=coverage), the
same installed CPU wheel:

    fast_trial.py coverage --python VPY --out OUT [--shard I --shards N] [--workers W] [--unit-timeout S]
    fast_trial.py coverage-summary --out OUT [--out OUT2 ...] --ledger coverage_ledger.json --md SUMMARY.md

A unit is one public class of the installed `mojolearn.__all__` (the capitalised names; functions and submodules are
listed, not probed). Its child (COVERAGE_CHILD, `python -I` in an empty cwd, MOJOLEARN_NUMERIC_MODE=identical)
constructs the class (`numeric_mode='identical'` when it takes the word), fits it on a 48x4 float32 fixture (a 48-point
series when `fit` takes `y` first), then calls each public method once with arguments chosen by parameter name
(value_for), and writes one JSON line before and after every call. Each call is `ran`, `refused` (it raised: the
message says why it does not run on this CPU install), `not_covered` (the probe could not form the call: an unknown
required parameter or a signature TypeError; a gap of this smoke, not of the product) or `crashed` (the process died
or hit --unit-timeout inside it). After a crash the parent starts the child again with every finished method
skipped, so one crashing method never hides the others. It gates nothing; `coverage` exits non-zero only when a unit
left no line at all.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = "test_fast_selector.py"
TEST_SELECT = "BuiltInstallFamilies"
MASK_ALL = (1 << 28) - 1  # core/mode_answer.mojo MODE_FAMILIES_ALL
UNIT_TIMEOUT = 1900  # the test's own child timeout is 1800 s
STATUSES = ("answered", "mixed", "refused", "unavailable", "no_parameter", "crashed")


def mask(text: str) -> str:
    """A family mask as the test takes it (hex or decimal), refused when outside MODE_FAMILIES_ALL."""
    value = int(text, 0)
    if not 0 <= value <= MASK_ALL:
        raise argparse.ArgumentTypeError(f"family mask {text} is outside 0..{MASK_ALL:#x}")
    return f"{value:#x}"


def load_test(tests_root: Path):
    """The trial's test module, loaded from its file (it imports only the standard library at module level)."""
    spec = importlib.util.spec_from_file_location("fast_selector_trial_plan", tests_root / TEST_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "selected_families") or not hasattr(module, "ESTIMATOR_FAMILY"):
        raise SystemExit(f"{tests_root / TEST_FILE} predates MOJOLEARN_FAST_ESTIMATORS: use a source commit that has it")
    return module


def trials(families: str, exact: str | None) -> list[tuple[str, str]]:
    """[(trial name, mask)]: the families trial, and the exact candidate set when one is given."""
    out = [("families", families)]
    if exact is not None and exact != "":
        out.append(("exact", exact))
    return out


def units(estimators: list[str], trial_list, shard: int = 0, shards: int = 1) -> list[tuple[str, str, str]]:
    """[(trial, mask, estimator)] of this shard: every (trial, estimator) once across the shards."""
    every = [(name, value, est) for name, value in trial_list for est in estimators]
    return [u for k, u in enumerate(every) if k % shards == shard]


def mem_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        return None
    return None


def worker_count(requested: int, unit_gib: float, n_units: int) -> int:
    if requested:
        cores = requested
    elif hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))  # the allocation, not the host's visible CPUs
    else:
        cores = os.cpu_count() or 1
    avail = mem_available_gib()
    by_memory = max(1, int(avail // unit_gib)) if avail and unit_gib > 0 else cores
    return max(1, min(cores, by_memory, n_units or 1))


def bootstrap() -> str:
    sys.path.insert(0, str(ROOT / "tools"))
    from run_python_tests import INSTALLED_BOOTSTRAP
    return INSTALLED_BOOTSTRAP


def run_unit(python: str, tests_root: Path, out: Path, unit, code: str) -> dict:
    trial, value, estimator = unit
    stem = f"{trial}-{estimator}"
    ledger = out / "units" / f"{stem}.json"
    log = out / "units" / f"{stem}.log"
    env = dict(os.environ, MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES=value, MOJOLEARN_FAST_LEDGER=str(ledger),
               MOJOLEARN_FAST_ESTIMATORS=estimator)
    argv = [python, "-I", "-c", code, str(tests_root), "-q", "-rfEs", "-p", "no:cacheprovider", "--import-mode=importlib",
            str(tests_root / TEST_FILE), "-k", TEST_SELECT]
    start = time.monotonic()
    with open(log, "w") as fh, tempfile.TemporaryDirectory() as cwd:  # an empty cwd: no source tree on sys.path
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc, rss_kb, timed_out = None, 0, False
        deadline = start + UNIT_TIMEOUT
        while rc is None:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                rc = os.waitstatus_to_exitcode(status)
                rss_kb = usage.ru_maxrss
            else:
                if not timed_out and time.monotonic() > deadline:
                    proc.kill()
                    timed_out = True
                time.sleep(0.5)
        proc.returncode = rc
    return dict(trial=trial, mask=value, estimator=estimator, rc=rc, timed_out=timed_out,
                seconds=round(time.monotonic() - start, 1), maxrss_kb=rss_kb, ledger=ledger.exists())


def cmd_run(a) -> int:
    tests_root = Path(a.tests_root).resolve()
    module = load_test(tests_root)
    estimators = [row[0] for row in module.FAMILIES + list(getattr(module, "WALK", []))]
    todo = units(estimators, trials(a.families, a.exact), a.shard, a.shards)
    out = Path(a.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    workers = worker_count(a.workers, a.unit_gib, len(todo))
    print(f"fast trial: {len(todo)} units on shard {a.shard}/{a.shards}, {workers} workers "
          f"(cores {os.cpu_count()}, MemAvailable {mem_available_gib()} GiB, {a.unit_gib} GiB per unit)", flush=True)
    code = bootstrap()
    done = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for result in pool.map(lambda u: run_unit(a.python, tests_root, out, u, code), todo):
            done.append(result)
            print(f"  {result['trial']:8} {result['estimator']:28} rc={result['rc']} {result['seconds']}s "
                  f"rss={result['maxrss_kb'] // 1024}MiB ledger={'yes' if result['ledger'] else 'NO'}", flush=True)
    with open(out / "units.tsv", "w") as fh:
        fh.write("trial\tmask\testimator\trc\ttimed_out\tseconds\tmaxrss_kb\tledger\n")
        for r in done:
            fh.write("\t".join(str(r[k]) for k in ("trial", "mask", "estimator", "rc", "timed_out", "seconds",
                                                   "maxrss_kb", "ledger")) + "\n")
    missing = [f"{r['trial']}-{r['estimator']}" for r in done if not r["ledger"]]
    if missing:
        print(f"fast trial: {len(missing)} unit(s) left no ledger: {' '.join(missing)}", flush=True)
    return 1 if missing else 0


def tail_reason(log: Path) -> str:
    try:
        lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return "no log"
    for ln in reversed(lines):
        if any(word in ln for word in ("Error", "error", "SKIPPED", "Fatal", "Segmentation", "Killed")):
            return ln[:300]
    return (lines[-1] if lines else "empty log")[:300]


def merge(out_dirs: list[Path]) -> dict:
    """{trial: {mask, answer, vendor, rows: {estimator: row}}} from every unit ledger (and a `crashed` row for a
    unit that left none)."""
    merged: dict = {}
    for out in out_dirs:
        units_tsv = out / "units.tsv"
        if not units_tsv.exists():
            continue
        for line in units_tsv.read_text().splitlines()[1:]:
            trial, value, estimator, rc, timed_out, seconds, rss, _ = line.split("\t")
            entry = merged.setdefault(trial, dict(mask=value, answer=None, vendor=None, rows={}))
            stem = out / "units" / f"{trial}-{estimator}"
            try:
                report = json.loads(Path(f"{stem}.json").read_text())
            except (OSError, ValueError):
                why = "timed out" if timed_out == "True" else tail_reason(Path(f"{stem}.log"))
                entry["rows"][estimator] = dict(status="crashed", why=f"rc={rc}: {why}", seconds=float(seconds))
                continue
            entry["answer"] = entry["answer"] or report.get("answer")
            entry["vendor"] = entry["vendor"] or report.get("vendor")
            for name, row in report.get("rows", {}).items():
                entry["rows"][name] = dict(row, seconds=float(seconds), maxrss_kb=int(rss))
    return merged


def family_status(rows: list[dict]) -> str:
    """One word for a family from its estimators' rows."""
    seen = {r["status"] for r in rows}
    for word in ("crashed", "mixed"):
        if word in seen:
            return word
    # unavailable (not runnable under identical) and no_parameter (the class takes no numeric_mode, so no door
    # answers for it) say nothing about the family's doors
    real = seen - {"unavailable", "no_parameter"}
    if not real:
        return "no_parameter" if "no_parameter" in seen else "unavailable"
    if len(real) == 1:
        return real.pop()
    return "mixed"  # its estimators disagree (one answered, one refused): not whole


def render(merged: dict, family_of: dict) -> str:
    lines = ["# per-estimator `fast` trial", ""]
    for trial, entry in merged.items():
        rows = entry["rows"]
        lines += [f"## trial `{trial}`: families {entry['mask']} (answer={entry['answer']}, vendor={entry['vendor']})",
                  ""]
        by_family: dict[str, list[tuple[str, dict]]] = {}
        for name, row in sorted(rows.items()):
            by_family.setdefault(row.get("family") or family_of.get(name, "?"), []).append((name, row))
        counts = {s: 0 for s in STATUSES}
        for family in sorted(by_family):
            members = by_family[family]
            word = family_status([r for _, r in members])
            counts[word] = counts.get(word, 0) + 1
            label = "none (no answer step)" if family == "none" else family
            parts = ", ".join(f"{n} {r['status']}" for n, r in members)
            lines.append(f"- {label}: **{word}** ({parts})")
        lines += ["", "families: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v), ""]
        details = [(n, r) for n, r in sorted(rows.items()) if r["status"] in ("mixed", "crashed")]
        if details:
            lines += ["### rows that are not whole (the message names the door to edit)", ""]
        for name, row in details:
            lines.append(f"- {row.get('family') or family_of.get(name, '?')} {name}: {row['status']}"
                         + (f" (attribute={row.get('attribute')}, param={row.get('param')})" if "ops" in row else ""))
            if row["status"] == "crashed":
                lines.append(f"  - {row.get('why', '')}")
            for op, value in sorted(row.get("ops", {}).items()):
                if value != "same":
                    lines.append(f"  - {op}: {value}")
        refused = [(n, r) for n, r in sorted(rows.items()) if r["status"] in ("refused", "unavailable", "no_parameter")]
        if refused:
            lines += ["", "### refused at fit / unavailable", ""]
            lines += [f"- {n}: {r['status']}: {r.get('why', '')}" for n, r in refused]
        lines.append("")
    return "\n".join(lines)


def cmd_summary(a) -> int:
    out_dirs = [Path(p) for p in a.out]
    merged = merge(out_dirs)
    family_of = {}
    if a.tests_root:
        family_of = load_test(Path(a.tests_root).resolve()).ESTIMATOR_FAMILY
    Path(a.ledger).write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    Path(a.md).write_text(render(merged, family_of))
    print(f"fast trial summary: {sum(len(e['rows']) for e in merged.values())} rows in {len(merged)} trial(s) -> "
          f"{a.ledger}, {a.md}")
    return 0 if merged else 1


# ---- the CPU coverage mode ------------------------------------------------------------------------------------------

COVERAGE_STATUSES = ("ran", "refused", "not_covered", "crashed")
#: the class word, worst first (coverage_class_word)
CLASS_WORDS = ("crashed", "refused", "partial", "not_covered", "ran")
COVERAGE_UNIT_TIMEOUT = 300  # seconds per child process; a method that runs past it is `crashed (timed out)`

#: The child: one public class, every public method, one JSON line before and after each call (argv: name, ledger,
#: comma separated methods to skip). Runs under `python -I` in an empty cwd against the INSTALLED wheel.
COVERAGE_CHILD = r'''
import inspect, json, os, sys, tempfile, time
NAME, LEDGER = sys.argv[1], sys.argv[2]
SKIP = set(n for n in sys.argv[3].split(",") if n)
_out = open(LEDGER, "a", buffering=1)
def emit(**row):
    _out.write(json.dumps(row, default=str) + "\n")
    _out.flush()
import numpy as np
X = (np.arange(192, dtype=np.float32).reshape(48, 4) * np.float32(0.37)) % np.float32(5.0) + np.float32(0.25)
Y_REG = np.ascontiguousarray(X[:, 0] * np.float32(2.0) - X[:, 1], dtype=np.float32)
Y_CLS = np.ascontiguousarray((X[:, 0] > X[:, 1]).astype(np.int32))
T = np.arange(48, dtype=np.float32)
SERIES = np.ascontiguousarray(np.sin(T * np.float32(0.4)) * np.float32(3.0) + np.float32(10.0) + T * np.float32(0.05),
                              dtype=np.float32)
WORK = tempfile.mkdtemp(prefix="cov-")
#: TypeError text that means the probe's call did not fit the signature (not_covered), not a refusal
SIGNATURE = ("positional argument", "keyword argument", "required argument", "takes no arguments", "takes from",
             "takes exactly", "were given", "was given", "missing 1", "missing 2", "missing 3")
#: constructor parameters made small when the class has them (a coverage smoke, not a fit)
SMALL = dict(n_estimators=4, max_iter=20, epochs=2, n_epochs=2, max_epochs=2, random_state=0)
#: methods that change the fitted state run after every reading method
LAST = ("save", "load", "set_params", "partial_fit")
CLS_WORDS = ("Classifier", "LogisticRegression", "Perceptron", "SVC", "NB", "DiscriminantAnalysis",
             "LabelPropagation", "LabelSpreading", "NearestCentroid", "Label")
def why(error):
    return f"{type(error).__name__}: {error}"[:400]
def is_classifier(cls):
    if getattr(cls, "_estimator_type", None) == "classifier":
        return True
    name = cls.__name__
    if "Regress" in name:
        return False
    return any(w in name for w in CLS_WORDS) or hasattr(cls, "predict_proba")
class NoValue(Exception):
    pass
def required(fn):
    """[(name, kind)] of the parameters a call must be given, or None when the signature cannot be read."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    return [(p.name, p.kind) for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.name not in ("self", "cls")
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
def value_for(pname, ctx):
    p = pname.lower()
    if p in ("x", "x_test", "x_train", "x_new", "data", "inputs", "input", "queries", "query", "q", "points",
             "vectors", "xq", "features"):
        # inverse_transform takes what transform gave (computed inside its own start/step lines)
        return model_transform(ctx) if ctx.get("want_xt") else X
    if p in ("y", "y_true", "y_train", "target", "targets", "labels", "series", "endog", "ts"):
        return ctx["y"]
    if p in ("h", "horizon", "steps", "n_periods", "fh", "n_ahead", "periods"):
        return 3
    if p in ("path", "filename", "file", "fname", "filepath", "directory"):
        if ctx.get("loading"):
            if "saved" not in ctx:
                raise NoValue("nothing was saved to load")
            return ctx["saved"]
        return os.path.join(WORK, "model.npz")
    if p in ("estimator", "base_estimator", "classifier", "regressor"):
        return ctx["base"]()
    if p == "estimators":
        return [("a", ctx["base"]()), ("b", ctx["base"]())]
    if p == "model":
        return ctx["fitted"]()
    if p == "classes":
        return np.unique(Y_CLS)
    if p in ("k", "n_neighbors", "n", "n_samples", "n_components", "n_clusters", "num", "size"):
        return 2
    if p == "radius":
        return 1.0
    if p in ("params", "parameters"):
        return ctx["params"]()
    raise NoValue(f"no probe value for required parameter {pname!r}")
def model_transform(ctx):
    if "transform" not in ctx:
        raise NoValue("inverse_transform needs a transform to invert")
    return ctx["transform"](X)
def call(fn, ctx):
    """(status, why, value): ran / refused / not_covered."""
    need = required(fn)
    args, kwargs = [], {}
    try:
        for name, kind in need or ():
            v = value_for(name, ctx)
            if kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = v
            else:
                args.append(v)
    except NoValue as error:
        return "not_covered", str(error), None
    try:
        value = fn(*args, **kwargs)
        if inspect.isgenerator(value):
            value = list(value)
        return "ran", "", value
    except TypeError as error:
        text = str(error)
        if any(s in text for s in SIGNATURE):
            return "not_covered", why(error), None
        return "refused", why(error), None
    except Exception as error:
        return "refused", why(error), None
import mojolearn
try:
    vendor = mojolearn.vendor()
except Exception as error:
    vendor = why(error)
emit(event="begin", name=NAME, vendor=vendor, skip=sorted(SKIP))
try:
    cls = getattr(mojolearn, NAME)
except Exception as error:
    emit(event="step", step="import", status="refused", why=why(error))
    emit(event="end")
    raise SystemExit(0)
if not inspect.isclass(cls) or issubclass(cls, BaseException):
    emit(event="not_a_class", kind=type(cls).__name__)
    emit(event="end")
    raise SystemExit(0)
classifier = is_classifier(cls)
fit_need = required(getattr(cls, "fit", None)) if hasattr(cls, "fit") else []
series_first = bool(fit_need) and fit_need[0][0] in ("y", "series", "endog", "ts")
ctx = dict(y=(Y_CLS if "Label" in NAME else SERIES) if series_first else (Y_CLS if classifier else Y_REG))
ctx["base"] = (lambda: mojolearn.LogisticRegression()) if classifier else (lambda: mojolearn.Ridge())
def fitted():
    m = mojolearn.RandomForestClassifier(n_estimators=4, random_state=0) if classifier else \
        mojolearn.RandomForestRegressor(n_estimators=4, random_state=0)
    return m.fit(X, Y_CLS if classifier else Y_REG)
ctx["fitted"] = fitted
# construction: required parameters by name, the small settings it has, the word when it takes it
t0 = time.monotonic()
try:
    sig = inspect.signature(cls)
    params = sig.parameters
except (TypeError, ValueError):
    params = {}
kwargs = {k: v for k, v in SMALL.items() if k in params}
if "numeric_mode" in params:
    kwargs["numeric_mode"] = "identical"
emit(event="start", step="construct")
try:
    for name, kind in required(cls) or ():
        kwargs[name] = value_for(name, ctx)
    model = cls(**kwargs)
    emit(event="step", step="construct", status="ran", why="", seconds=round(time.monotonic() - t0, 3),
         numeric_mode="numeric_mode" in params)
except NoValue as error:
    emit(event="step", step="construct", status="not_covered", why=str(error))
    emit(event="end")
    raise SystemExit(0)
except TypeError as error:
    st = "not_covered" if any(s in str(error) for s in SIGNATURE) else "refused"
    emit(event="step", step="construct", status=st, why=why(error))
    emit(event="end")
    raise SystemExit(0)
except Exception as error:
    emit(event="step", step="construct", status="refused", why=why(error))
    emit(event="end")
    raise SystemExit(0)
ctx["params"] = lambda: model.get_params() if hasattr(model, "get_params") else {}
if hasattr(model, "fit"):
    emit(event="start", step="fit")
    t0 = time.monotonic()
    status, reason, value = call(model.fit, ctx)
    emit(event="step", step="fit", status=status, why=reason, seconds=round(time.monotonic() - t0, 3))
    if status != "ran":
        emit(event="end")
        raise SystemExit(0)
if hasattr(model, "transform"):
    ctx["transform"] = model.transform
methods = []
try:
    members = inspect.getmembers(cls)
except Exception as error:
    members = []
    emit(event="step", step="list-methods", status="refused", why=why(error))
for name, member in members:
    if name.startswith("_") or name == "fit" or isinstance(member, property):
        continue
    if inspect.isroutine(member):
        methods.append(name)
order = sorted(methods, key=lambda n: (n.startswith("fit") or n in LAST, LAST.index(n) if n in LAST else -1, n))
emit(event="methods", methods=order)
for name in order:
    if name in SKIP:
        continue
    emit(event="start", step=name)
    ctx["want_xt"] = name == "inverse_transform"
    ctx["loading"] = name == "load"
    t0 = time.monotonic()
    try:
        fn = getattr(model, name)
    except Exception as error:
        emit(event="step", step=name, status="refused", why=why(error), seconds=0.0)
        continue
    status, reason, value = call(fn, ctx)
    if name == "save" and status == "ran":
        ctx["saved"] = os.path.join(WORK, "model.npz")
    emit(event="step", step=name, status=status, why=reason, seconds=round(time.monotonic() - t0, 3))
emit(event="end")
'''


def coverage_discover(python: str) -> list[str]:
    """The installed wheel's public names (`mojolearn.__all__`), read without loading any family."""
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run([python, "-I", "-c", "import json, mojolearn; print(json.dumps(list(mojolearn.__all__)))"],
                           cwd=cwd, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical"))
    if p.returncode != 0:
        raise SystemExit(f"cpu coverage: `import mojolearn` failed on the installed wheel: {p.stderr[-600:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def coverage_split(names: list[str]) -> tuple[list[str], list[str]]:
    """(classes to probe, other public names): a capitalised name is a class (functions, submodules are not)."""
    seen, probe, other = set(), [], []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        (probe if n[:1].isupper() else other).append(n)
    return probe, other


def read_events(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return events


def open_step(events: list[dict]) -> str | None:
    """The step a child started and never finished (the call it died in), or None."""
    started = None
    for e in events[events_since_last_begin(events):]:
        if e.get("event") == "start":
            started = e["step"]
        elif e.get("event") == "step" and e.get("step") == started:
            started = None
    return started


def events_since_last_begin(events: list[dict]) -> int:
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "begin":
            return i
    return 0


def finished_methods(events: list[dict]) -> set[str]:
    return {e["step"] for e in events if e.get("event") == "step" and e.get("step") not in ("construct", "fit", "import")}


def spawn_wait(argv: list[str], log: Path, env: dict, timeout: float) -> tuple[int | None, bool, int, float]:
    """(rc, timed_out, peak RSS KiB, seconds) of one child in an empty cwd."""
    start = time.monotonic()
    with open(log, "a") as fh, tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc, rss_kb, timed_out = None, 0, False
        while rc is None:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                rc = os.waitstatus_to_exitcode(status)
                rss_kb = usage.ru_maxrss
            else:
                if not timed_out and time.monotonic() > start + timeout:
                    proc.kill()
                    timed_out = True
                time.sleep(0.2)
        proc.returncode = rc
    return rc, timed_out, rss_kb, time.monotonic() - start


def coverage_unit(python: str, out: Path, name: str, timeout: float) -> dict:
    """One class: the child, started again after each crash with the finished methods skipped."""
    ledger = out / "units" / f"{name}.jsonl"
    log = out / "units" / f"{name}.log"
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    total, peak, runs, rc, timed_out = 0.0, 0, 0, None, False
    crashed: list[str] = []
    while True:
        events = read_events(ledger)
        skip = sorted(finished_methods(events) | set(crashed))
        rc, timed_out, rss, seconds = spawn_wait(
            [python, "-I", "-c", COVERAGE_CHILD, name, str(ledger), ",".join(skip)], log, env, timeout)
        runs += 1
        total += seconds
        peak = max(peak, rss)
        events = read_events(ledger)
        tail = events[events_since_last_begin(events):]
        if any(e.get("event") == "end" for e in tail):
            break
        died_in = open_step(events)
        reason = "timed out" if timed_out else f"rc={rc}: {tail_reason(log)}"
        with open(ledger, "a") as fh:
            fh.write(json.dumps(dict(event="step", step=died_in or "start-up", status="crashed", why=reason)) + "\n")
        if died_in in (None, "construct", "fit") or runs > 64:
            break  # nothing after it can run: the row ends here
        crashed.append(died_in)
    return dict(name=name, rc=rc, timed_out=timed_out, runs=runs, seconds=round(total, 1), maxrss_kb=peak,
                ledger=ledger.exists())


def cmd_coverage(a) -> int:
    names = coverage_discover(a.python)
    probe, other = coverage_split(names)
    todo = [n for k, n in enumerate(probe) if k % a.shards == a.shard]
    out = Path(a.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "public-names.json").write_text(json.dumps(dict(probed=probe, other=other), indent=1) + "\n")
    workers = worker_count(a.workers, a.unit_gib, len(todo))
    print(f"cpu coverage: {len(todo)} of {len(probe)} classes on shard {a.shard}/{a.shards}, {workers} workers "
          f"(cores {os.cpu_count()}, MemAvailable {mem_available_gib()} GiB, {a.unit_gib} GiB per unit); "
          f"{len(other)} public functions/modules listed, not probed", flush=True)
    done = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for r in pool.map(lambda n: coverage_unit(a.python, out, n, a.unit_timeout), todo):
            done.append(r)
            print(f"  {r['name']:36} runs={r['runs']} rc={r['rc']} {r['seconds']}s rss={r['maxrss_kb'] // 1024}MiB",
                  flush=True)
    with open(out / "units.tsv", "w") as fh:
        fh.write("name\trc\ttimed_out\truns\tseconds\tmaxrss_kb\tledger\n")
        for r in done:
            fh.write("\t".join(str(r[k]) for k in ("name", "rc", "timed_out", "runs", "seconds", "maxrss_kb",
                                                   "ledger")) + "\n")
    missing = [r["name"] for r in done if not read_events(out / "units" / f"{r['name']}.jsonl")]
    if missing:
        print(f"cpu coverage: {len(missing)} unit(s) left no line: {' '.join(missing)}", flush=True)
    return 1 if missing else 0


def coverage_row(events: list[dict]) -> dict:
    """{vendor, kind, steps: {construct, fit, <method>: {status, why, seconds}}, methods} of one class: the first
    result of each method, the last of construct/fit (a restarted child constructs and fits again)."""
    row: dict = dict(vendor=None, kind="class", steps={}, methods=[])
    for e in events:
        ev = e.get("event")
        if ev == "begin":
            row["vendor"] = row["vendor"] or e.get("vendor")
        elif ev == "not_a_class":
            row["kind"] = e.get("kind") or "not a class"
        elif ev == "methods":
            row["methods"] = e.get("methods", [])
        elif ev == "step":
            step = e["step"]
            entry = dict(status=e.get("status"), why=e.get("why", ""), seconds=e.get("seconds"))
            if step in ("construct", "fit", "import", "start-up") or step not in row["steps"]:
                row["steps"][step] = entry
    return row


def coverage_class_word(row: dict) -> str:
    """One word for a class: crashed / refused (it does not construct or fit) / partial (a method refused) /
    not_covered (the probe could not form construct or fit) / ran."""
    steps = row["steps"]
    if not steps:
        return "crashed"
    seen = {s["status"] for s in steps.values()}
    if "crashed" in seen:
        return "crashed"
    head = [steps[k]["status"] for k in ("import", "start-up", "construct", "fit") if k in steps]
    if "refused" in head:
        return "refused"
    if "not_covered" in head:
        return "not_covered"
    if "refused" in seen:
        return "partial"
    return "ran"


def coverage_merge(out_dirs: list[Path]) -> dict:
    rows: dict = {}
    other: list[str] = []
    for out in out_dirs:
        names = out / "public-names.json"
        if names.exists():
            other = other or json.loads(names.read_text()).get("other", [])
        units_tsv = out / "units.tsv"
        if not units_tsv.exists():
            continue
        for line in units_tsv.read_text().splitlines()[1:]:
            name, rc, timed_out, runs, seconds, rss, _ = line.split("\t")
            row = coverage_row(read_events(out / "units" / f"{name}.jsonl"))
            if not row["steps"] and row["kind"] == "class":
                why = "timed out" if timed_out == "True" else tail_reason(out / "units" / f"{name}.log")
                row["steps"]["start-up"] = dict(status="crashed", why=f"rc={rc}: {why}", seconds=None)
            row.update(runs=int(runs), seconds=float(seconds), maxrss_kb=int(rss))
            if row["kind"] == "class":
                row["word"] = coverage_class_word(row)
            rows[name] = row
    return dict(rows=rows, other=other)


def coverage_render(merged: dict) -> str:
    rows = {n: r for n, r in merged["rows"].items() if r["kind"] == "class"}
    vendors = sorted({str(r.get("vendor")) for r in rows.values()})
    lines = ["# CPU coverage of the public API (`identical`, installed CPU wheel)", "",
             f"vendor(s) reported: {', '.join(vendors)}" + ("" if vendors == ["cpu"] else
                                                             "  (NOT only `cpu`: read the rows with that in mind)"),
             ""]
    words = {w: sorted(n for n, r in rows.items() if r["word"] == w) for w in CLASS_WORDS}
    calls = {s: 0 for s in COVERAGE_STATUSES}
    for r in rows.values():
        for s in r["steps"].values():
            calls[s["status"]] = calls.get(s["status"], 0) + 1
    lines += ["| classes | count |", "|---|---|"]
    lines += [f"| {w} | {len(words[w])} |" for w in CLASS_WORDS]
    lines += [f"| **total** | **{len(rows)}** |", "",
              "| calls (construct, fit, each method) | count |", "|---|---|"]
    lines += [f"| {s} | {calls.get(s, 0)} |" for s in COVERAGE_STATUSES]
    lines += [f"| **total** | **{sum(calls.values())}** |", ""]
    skipped = sorted(n for n, r in merged["rows"].items() if r["kind"] != "class")
    lines.append(f"Public names listed but not probed: {len(merged.get('other', []))} functions/submodules"
                 + (f"; {len(skipped)} capitalised names that are not classes ({', '.join(skipped)})" if skipped else ""))
    lines += ["", "## Does NOT run on CPU (refused or crashed), with the message", ""]
    bad = 0
    for name in sorted(rows):
        r = rows[name]
        for step, s in r["steps"].items():
            if s["status"] in ("refused", "crashed"):
                lines.append(f"- {name}.{step}: **{s['status']}**: {s['why']}")
                bad += 1
    if not bad:
        lines.append("- none")
    lines += ["", "## Not covered by this smoke (the probe could not form the call; a gap of the smoke)", ""]
    gaps = [f"- {n}.{step}: {s['why']}" for n in sorted(rows) for step, s in rows[n]["steps"].items()
            if s["status"] == "not_covered"]
    lines += gaps or ["- none"]
    lines += ["", "## Every class", "", "| class | word | ran | refused | not covered | crashed | seconds |",
              "|---|---|---|---|---|---|---|"]
    for name in sorted(rows):
        r = rows[name]
        c = {s: sum(1 for v in r["steps"].values() if v["status"] == s) for s in COVERAGE_STATUSES}
        lines.append(f"| {name} | {r['word']} | {c['ran']} | {c['refused']} | {c['not_covered']} | {c['crashed']} | "
                     f"{r.get('seconds', '')} |")
    lines.append("")
    return "\n".join(lines)


def cmd_coverage_summary(a) -> int:
    merged = coverage_merge([Path(p) for p in a.out])
    Path(a.ledger).write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    Path(a.md).write_text(coverage_render(merged))
    rows = [r for r in merged["rows"].values() if r["kind"] == "class"]
    print(f"cpu coverage summary: {len(rows)} classes -> {a.ledger}, {a.md}; "
          + ", ".join(f"{w} {sum(1 for r in rows if r['word'] == w)}" for w in CLASS_WORDS))
    return 0 if merged["rows"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage", help="the CPU coverage smoke: every public class and method, identical")
    c.add_argument("--python", required=True, help="the venv python with the built CPU wheel")
    c.add_argument("--out", required=True)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--workers", type=int, default=0, help="0: the cores (lowered only by MemAvailable)")
    c.add_argument("--unit-gib", type=float, default=1.5,
                   help="peak memory per unit (estimate until OUT/units.tsv maxrss_kb measures it)")
    c.add_argument("--unit-timeout", type=float, default=COVERAGE_UNIT_TIMEOUT,
                   help="seconds per child process (a method past it is crashed: timed out)")
    cs = sub.add_parser("coverage-summary")
    cs.add_argument("--out", action="append", required=True, help="a `coverage` OUT (repeat for every shard)")
    cs.add_argument("--ledger", required=True)
    cs.add_argument("--md", required=True)
    r = sub.add_parser("run")
    r.add_argument("--python", required=True, help="the venv python with the built wheel and pytest")
    r.add_argument("--tests-root", required=True, help="SOURCE/python/mojolearn/tests at the wheel's commit")
    r.add_argument("--out", required=True)
    r.add_argument("--families", type=mask, default=f"{MASK_ALL:#x}")
    r.add_argument("--exact", type=lambda s: mask(s) if s else "", default="")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=0, help="0: the cores (lowered only by MemAvailable)")
    r.add_argument("--unit-gib", type=float, default=1.5,
                   help="peak memory per unit (estimate until OUT/units.tsv maxrss_kb measures it)")
    s = sub.add_parser("summary")
    s.add_argument("--out", action="append", required=True, help="a `run` OUT (repeat for every shard)")
    s.add_argument("--tests-root", default="")
    s.add_argument("--ledger", required=True)
    s.add_argument("--md", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "coverage":
        if not 0 <= a.shard < a.shards:
            ap.error("--shard must be in 0..--shards-1")
        return cmd_coverage(a)
    if a.cmd == "coverage-summary":
        return cmd_coverage_summary(a)
    if a.cmd == "run":
        if not 0 <= a.shard < a.shards:
            ap.error("--shard must be in 0..--shards-1")
        return cmd_run(a)
    return cmd_summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
