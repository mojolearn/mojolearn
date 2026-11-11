#!/usr/bin/env python3
"""Observe Python calls during public operations, with visibility controls.

This is a test controller, not product code or a timing benchmark. Independent
estimator cases run in separate processes across the CPU allocation. GPU cases
run in order on one device: parallel GPU ownership is a separate contract and
concurrent workloads would obscure attribution. Fit precedes query per model.

The profiler sees Python frames and some C calls, not all calls made by native
code through the CPython API. The calibration demonstrates that limitation.
Zero observed nested calls is never a Python-free qualification.

`--cpython-count perf` (Linux, root, `perf` with uprobes; lane
thin-unpack-kit, 2026-11-11) closes that gap for the thin-UNPACK work: every
CPython C-API function our native code calls (each entry of
bindings/cpython_table.mojo plus the standard library's own calls) gets a
uprobe, and each public operation runs in its own `perf stat` process with
counting enabled only around that one operation (perf's control FIFO). The
count is every entry into those functions during the operation, from native
code and from the interpreter alike. Run it at a small shape and at ten times
that shape (`--rows 32 320`): the summary's `cpython_scaling` lists both
counts per operation, and a door whose UNPACK/PACK reads one value per
argument has the same count at both. A count of calls, a testing tool, never
a source-count gate or a timing. Native entry
table tests and ownership tests supply separate evidence. PCA uses transform
because its public interface has no predict operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import traceback


CASES = ("Ridge", "LinearRegression", "LogisticRegression", "PCA", "KMeans",
         "RandomForestClassifier", "GradientBoosting",
         "GaussianProcessRegressor", "GaussianProcessClassifier",
         "LinearSVC_str", "SVC_precomputed", "SVR_precomputed")

# Cases that run a public class under another name (lane thin-unpack-svm):
# LinearSVC with string labels and a full-size score (the score UNPACK is
# per used class, so its count must not grow with rows); SVC/SVR with
# kernel='precomputed' (predict reads support_, which grows with rows).
ESTIMATOR_CLASS = {"LinearSVC_str": "LinearSVC", "SVC_precomputed": "SVC", "SVR_precomputed": "SVR"}
         # Mixture and clustering doors (lane thin-unpack-mixture-cluster).
         "GaussianMixture", "BayesianGaussianMixture", "MiniBatchKMeans", "BisectingKMeans",
         "SpectralClustering", "HDBSCAN", "OPTICS")

# Cases whose fit takes X alone.
UNSUPERVISED = ("PCA", "KMeans", "GaussianMixture", "BayesianGaussianMixture", "MiniBatchKMeans",
                "BisectingKMeans", "SpectralClustering", "HDBSCAN", "OPTICS")
# Cases without predict: the query is fit_predict on the 7-row query slice
# (a fixed shape, so only the fit's count scales with --rows).
FIT_PREDICT_QUERY = ("SpectralClustering", "HDBSCAN", "OPTICS")

# Classifier cases: integer 0/1 targets.
CLASSIFIERS = ("LogisticRegression", "RandomForestClassifier", "GaussianProcessClassifier", "SVC_precomputed")

# uprobe group of the CPython call count (`--cpython-count perf`).
PERF_GROUP = "mojolearn_cpy"

# C-API functions the standard library's PythonObject calls directly (the
# ones bindings/cpython_table.mojo does not list; it says the stdlib handle
# keeps reference counts, PyObject_Type, PyObject_Call, the tuple builders
# and errors). Together with the table they are every CPython entry a
# Mojolearn binding reaches.
STDLIB_CPYTHON = (
    "Py_IncRef", "Py_DecRef", "PyObject_Type", "PyObject_Call", "PyObject_CallObject",
    "PyObject_GetAttrString", "PyObject_SetAttrString", "PyObject_GetAttr", "PyObject_SetAttr",
    "PyObject_GetItem", "PyObject_SetItem", "PyObject_Length", "PyObject_Str", "PyObject_Repr",
    "PyObject_RichCompare", "PyObject_Hash", "PyTuple_New", "PyTuple_SetItem", "PyTuple_GetItem",
    "PyList_New", "PyList_SetItem", "PyList_GetItem", "PyDict_New", "PyDict_SetItem",
    "PyDict_SetItemString", "PyLong_FromSsize_t", "PyLong_FromLong", "PyLong_AsSsize_t",
    "PyLong_AsLong", "PyFloat_FromDouble", "PyBool_FromLong", "PyUnicode_FromStringAndSize",
    "PySequence_Contains", "PyErr_Fetch", "PyErr_SetString", "PyErr_Restore",
    "PyEval_SaveThread", "PyEval_RestoreThread",
)


class PythonCalls:
    """Record observable events without modifying any product callable."""

    def __init__(self):
        self.calls = Counter()
        self.returns = Counter()

    def record(self, frame, event, arg):
        if event in ("call", "return"):
            if frame.f_code.co_filename == __file__:
                return
            key = ("python", frame.f_globals.get("__name__", ""), frame.f_code.co_qualname)
        elif event in ("c_call", "c_return", "c_exception"):
            if arg is sys.setprofile:
                return
            key = ("c", getattr(arg, "__module__", "") or "",
                   getattr(arg, "__qualname__", getattr(arg, "__name__", "<unnamed>")))
        else:
            return
        if event in ("call", "c_call"):
            self.calls[key] += 1
        else:
            self.returns[key] += 1

    def run(self, function, *args):
        # Each product case is a fresh worker. Restore callers' profile hooks
        # even on failure; never leave an observer active after the operation.
        previous = sys.getprofile()
        previous_thread = threading.getprofile()
        threading.setprofile(self.record)
        sys.setprofile(self.record)
        try:
            return function(*args)
        finally:
            sys.setprofile(previous)
            threading.setprofile(previous_thread)

    def report(self):
        return {
            "coverage": "current thread and Python threads created during the operation",
            "unobserved": ["native-to-native CPython C calls may be invisible",
                           "already-running foreign threads", "UNPACK/RUN/PACK phase boundaries"],
            "calls": [dict(kind=kind, module=module, name=name, count=count,
                           returns=self.returns[(kind, module, name)])
                      for (kind, module, name), count in sorted(self.calls.items())],
            "python_calls": sum(n for (kind, _, _), n in self.calls.items() if kind == "python"),
            "mojolearn_python_calls": sum(n for (kind, module, _), n in self.calls.items()
                                         if kind == "python" and
                                         (module == "mojolearn" or module.startswith("mojolearn."))),
            "visible_c_calls": sum(n for (kind, _, _), n in self.calls.items() if kind == "c"),
            "python_free_qualification": "not established by profiling",
        }


def visibility_control():
    """Compare known builtin invocations from Python and from the C API."""
    call = ctypes.pythonapi.PyObject_CallObject
    call.restype = ctypes.py_object
    call.argtypes = (ctypes.py_object, ctypes.py_object)
    direct, indirect = PythonCalls(), PythonCalls()
    direct_result = direct.run(lambda: len((1, 2, 3)))
    indirect_result = indirect.run(call, len, ((1, 2, 3),))
    if direct_result != 3 or indirect_result != 3:
        raise AssertionError("profiler visibility control did not execute len")
    key = ("c", "builtins", "len")
    return {"known_invocations_per_route": 1,
            "python_to_builtin_observed": direct.calls[key],
            "c_api_to_builtin_observed": indirect.calls[key],
            "c_api_blind_spot_demonstrated": indirect.calls[key] != 1,
            "qualification": "visibility control only; never an estimator pass"}


def model_parameters(name):
    parameters = {
        "Ridge": {"alpha": 1.0}, "LinearRegression": {},
        "LogisticRegression": {"max_iter": 20},
        "PCA": {"n_components": 2},
        "KMeans": {"n_clusters": 3, "max_iter": 10, "random_state": 0},
        "RandomForestClassifier": {"n_estimators": 3, "max_depth": 3, "random_state": 0},
        "GradientBoosting": {"loss": "RMSE", "n_estimators": 3, "max_depth": 2,
                             "random_state": 0},
        # optimizer=None (both defaults): one fit per call, so the count
        # compares the same work at both shapes.
        "GaussianProcessRegressor": {"optimizer": None},
        "GaussianProcessClassifier": {"optimizer": None},
        "LinearSVC_str": {"max_iter": 50},
        "SVC_precomputed": {"kernel": "precomputed"},
        "SVR_precomputed": {"kernel": "precomputed"},
        # Fixed iteration counts (tol=0 where the estimator has one), so the
        # fit does the same number of steps at both shapes.
        "GaussianMixture": {"n_components": 2, "max_iter": 5, "tol": 0.0, "random_state": 0},
        "BayesianGaussianMixture": {"n_components": 2, "max_iter": 5, "tol": 0.0, "random_state": 0},
        "MiniBatchKMeans": {"n_clusters": 3, "max_iter": 5, "random_state": 0},
        "BisectingKMeans": {"n_clusters": 3, "max_iter": 5, "random_state": 0},
        "SpectralClustering": {"n_clusters": 2, "affinity": "rbf", "random_state": 0},
        "HDBSCAN": {"min_cluster_size": 5},
        "OPTICS": {"min_samples": 5},
    }
    return dict(numeric_mode="identical", **parameters[name])


def compare_observations(before, after):
    """Compare observed counts only for successful, matching fixture cases."""
    fields = ("estimator", "requested_backend", "rows", "input_sha256", "parameters")
    if before.get("status") != "observed" or after.get("status") != "observed":
        return {"status": "incomparable", "reason": "a public operation failed"}
    if any(key not in before or before[key] != after.get(key) for key in fields):
        return {"status": "incomparable", "reason": "fixture, parameters or requested backend differ"}
    old = {row["operation"]: row for row in before["operations"]}
    new = {row["operation"]: row for row in after["operations"]}
    if old.keys() != new.keys():
        return {"status": "incomparable", "reason": "operation coverage differs"}
    metrics = ("python_calls", "mojolearn_python_calls", "visible_c_calls")
    return {"status": "observed_counts_only", "python_free_qualification": "not established",
            "operations": [{"operation": operation,
                            **{key: {"before": old[operation]["profile"][key],
                                     "after": new[operation]["profile"][key]} for key in metrics}}
                           for operation in old]}


def _perf_toggle(command):
    """Enable or disable the enclosing `perf stat` (control FIFO) and wait
    for its acknowledgement; a no-op outside `--cpython-count perf`."""
    control = os.environ.get("MOJOLEARN_PROBE_PERF_CTL")
    if not control:
        return
    # O_RDWR opens of a FIFO never block on Linux; perf holds the other ends.
    ctl = os.open(control, os.O_RDWR)
    ack = os.open(os.environ["MOJOLEARN_PROBE_PERF_ACK"], os.O_RDWR)
    try:
        os.write(ctl, (command + "\n").encode())
        reply = os.read(ack, 16)
        if not reply.startswith(b"ack"):
            raise RuntimeError(f"perf control did not acknowledge {command!r}: {reply!r}")
    finally:
        os.close(ctl)
        os.close(ack)


def run_case(case):
    name, backend, rows, evidence = case[:4]
    # `--cpython-count perf`: the one operation counted in this process.
    counted = case[4] if len(case) > 4 else None
    os.environ["MOJOLEARN_VENDOR"] = backend
    result = {"estimator": name, "requested_backend": backend, "rows": rows,
              "status": "failed", "operations": [], "python_free_qualification": "pending"}
    try:
        import numpy as np
        import mojolearn as ml

        result["package"] = {"path": ml.__file__, "version": getattr(ml, "__version__", None)}
        result["visibility_control"] = visibility_control()
        random = np.random.default_rng(0)
        x = random.normal(size=(rows, 4)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] * np.float32(0.3)).astype(np.float32)
        if name in CLASSIFIERS:
            # --label-dtype (default int64): another integer width exercises
            # the label reads that are not the contiguous int64 fast path.
            y = (y > 0).astype(os.environ.get("MOJOLEARN_PROBE_LABEL_DTYPE", "int64"))
            result["label_dtype"] = str(y.dtype)
        result["input_sha256"] = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        result["parameters"] = model_parameters(name)
        model = getattr(ml, ESTIMATOR_CLASS.get(name, name))(**result["parameters"])
        query = query_operation(name)
        fit_x, query_x = x, x[:min(7, rows)]
        if name.endswith("_precomputed"):
            # Linear Gram matrices: train (rows, rows), query (7, rows).
            fit_x, query_x = (x @ x.T).astype(np.float32), (query_x @ x.T).astype(np.float32)
        if name == "LinearSVC_str":
            y = np.where(y > 0, "pos", "neg").astype(object)
        operations = [("fit", model.fit, (x,) if name in ("PCA", "KMeans") else (fit_x, y)),
                      (query, getattr(model, query), (query_x,))]
        if name == "LinearSVC_str":
            # The full-size score: its CPython count must not grow with rows.
            operations.append(("score", model.score, (x, y)))
        operations = [("fit", model.fit, (x,) if name in UNSUPERVISED else (x, y)),
                      (query, getattr(model, query), (x[:min(7, rows)],))]
        for operation, function, arguments in operations:
            # These two dependent operations cannot run concurrently: query
            # consumes the fitted state produced by this exact fit.
            observed = PythonCalls()
            record = {"operation": operation, "status": "failed"}
            result["operations"].append(record)
            try:
                if counted is None:
                    value = observed.run(function, *arguments)
                elif operation == counted:
                    # No profiler hook during the counted operation: its own
                    # Python work would enter the counted C-API functions.
                    _perf_toggle("enable")
                    try:
                        value = function(*arguments)
                    finally:
                        _perf_toggle("disable")
                    record["cpython_counted"] = True
                else:
                    value = function(*arguments)
                if operation == "score":
                    if not math.isfinite(float(value)):
                        raise AssertionError("score returned a nonfinite value")
                elif operation != "fit":
                    output = np.asarray(value)
                    if output.shape[0] != min(7, rows) or not np.isfinite(output).all():
                        raise AssertionError("query returned invalid shape or nonfinite output")
                    record["output_shape"] = list(output.shape)
                record["status"] = "observed"
            finally:
                record["profile"] = observed.report()
        # Report actual loaded artifact paths; a requested backend alone is
        # not independent proof of GPU execution or installed-wheel provenance.
        result["loaded_native_modules"] = sorted({str(getattr(module, "__file__", "") or "")
            for key, module in tuple(sys.modules.items())
            if "mojolearn" in key and str(getattr(module, "__file__", "") or "").endswith((".so", ".dylib", ".pyd"))})
        result["status"] = "observed"
    except Exception:
        result["error"] = traceback.format_exc()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    suffix = f".{counted}" if counted else ""
    target = Path(evidence) / f"{backend}-{name}-{rows}{suffix}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return {"estimator": name, "rows": rows, "backend": backend,
            "status": result["status"], "evidence": str(target)}


def cpython_symbols(root):
    """Every CPython function our bindings call: the entries of
    bindings/cpython_table.mojo plus STDLIB_CPYTHON."""
    import re
    table = (Path(root) / "bindings" / "cpython_table.mojo").read_text()
    names = re.findall(r'ExternalFunction\[\s*"(Py[A-Za-z_]+)"', table)
    return sorted(set(names) | set(STDLIB_CPYTHON))


def cpython_library():
    """The image that defines the C API: libpython when the interpreter is
    built shared, else the interpreter executable itself."""
    import sysconfig
    if sysconfig.get_config_var("Py_ENABLE_SHARED"):
        library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
        if library.exists():
            return str(library.resolve())
    return str(Path(sys.executable).resolve())


def perf_probes_add(root, evidence):
    """One uprobe per CPython symbol (root). Symbols absent from this
    interpreter are skipped and listed. Probe additions are independent
    and run across the allocation."""
    import subprocess
    library = cpython_library()
    subprocess.run(["perf", "probe", "-q", "-d", f"{PERF_GROUP}:*"], capture_output=True)

    def add(symbol):
        done = subprocess.run(["perf", "probe", "-q", "-x", library, "-a", f"{PERF_GROUP}:{symbol}={symbol}"],
                              capture_output=True, text=True)
        return symbol, done.returncode == 0, done.stderr[-300:]

    symbols = cpython_symbols(root)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        outcomes = list(pool.map(add, symbols))
    added = [symbol for symbol, ok, _ in outcomes if ok]
    skipped = {symbol: error for symbol, ok, error in outcomes if not ok}
    (Path(evidence) / "cpython_probes.json").write_text(json.dumps(
        {"library": library, "added": added, "skipped": skipped}, indent=2) + "\n")
    if not added:
        raise SystemExit("perf probe added no CPython uprobes; see cpython_probes.json")
    return added


def perf_probes_remove():
    import subprocess
    subprocess.run(["perf", "probe", "-q", "-d", f"{PERF_GROUP}:*"], capture_output=True)


def perf_counts(path):
    """{symbol: count} from a `perf stat -x,` output file."""
    counts = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 3 or not fields[2].startswith(PERF_GROUP + ":"):
            continue
        value = fields[0].strip()
        counts[fields[2].split(":", 1)[1]] = int(value) if value.isdigit() else None
    return counts


def query_operation(name):
    if name in FIT_PREDICT_QUERY:
        return "fit_predict"
    return "transform" if name == "PCA" else "predict"


def case_operations(name):
    """Every operation counted for a case, in run order."""
    return ("fit", query_operation(name)) + (("score",) if name == "LinearSVC_str" else ())


def merge_counted(case, counted):
    """The case record of `--cpython-count perf`: the fit process's record
    plus every counted operation's CPython calls."""
    name, backend, rows, evidence = case
    base = Path(evidence) / f"{backend}-{name}-{rows}.fit.json"
    result = json.loads(base.read_text()) if base.exists() else {
        "estimator": name, "rows": rows, "requested_backend": backend, "status": "failed", "operations": []}
    result["cpython_count"] = counted
    if any(row["exit_code"] != 0 or row["cpython_calls"] is None for row in counted):
        result["status"] = "failed"
    target = Path(evidence) / f"{backend}-{name}-{rows}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return {"estimator": name, "rows": rows, "backend": backend, "status": result["status"],
            "evidence": str(target),
            "cpython_calls": {row["operation"]: row["cpython_calls"] for row in counted}}


def launch_counted(unit, seconds):
    """One worker process under `perf stat`, counting only `operation`. The
    operations of one model are dependent (a query needs its fit), so the
    query's process repeats the fit uncounted; the processes themselves are
    independent units."""
    case, operation = unit
    import build_process
    name, backend, rows, evidence = case
    stem = Path(evidence) / f"{backend}-{name}-{rows}.{operation}"
    ctl, ack = stem.with_suffix(".perf-ctl"), stem.with_suffix(".perf-ack")
    for fifo in (ctl, ack):
        os.mkfifo(fifo)
    env = os.environ.copy()
    env["MOJOLEARN_PROBE_PERF_CTL"], env["MOJOLEARN_PROBE_PERF_ACK"] = str(ctl), str(ack)
    out = stem.with_suffix(".perf.csv")
    command = ["perf", "stat", "-x", ",", "-o", str(out), "-e", f"{PERF_GROUP}:*", "-D", "-1",
               "--control", f"fifo:{ctl},{ack}", "--",
               sys.executable, str(Path(__file__).resolve()), "--worker-spec",
               json.dumps(list(case) + [operation])]
    receipt = build_process.run(command, cwd=Path.cwd(), env=env, log=stem.with_suffix(".log"), seconds=seconds)
    for fifo in (ctl, ack):
        fifo.unlink(missing_ok=True)
    counts = perf_counts(out) if out.exists() else {}
    known = [value for value in counts.values() if value is not None]
    return {"operation": operation, "exit_code": receipt["exit_code"], "perf_output": str(out),
            "cpython_calls": sum(known) if known and receipt["exit_code"] == 0 else None,
            "by_symbol": {key: value for key, value in sorted(counts.items()) if value}}


def launch_case(case, seconds):
    """Isolate native aborts/timeouts without losing other completed cases."""
    import build_process
    name, backend, rows, evidence = case
    stem = Path(evidence) / f"{backend}-{name}-{rows}"
    receipt = build_process.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec", json.dumps(case)],
        cwd=Path.cwd(), env=os.environ.copy(), log=stem.with_suffix(".log"), seconds=seconds)
    target = stem.with_suffix(".json")
    if target.exists():
        result = json.loads(target.read_text())
    else:
        result = {"estimator": name, "rows": rows, "requested_backend": backend,
                  "status": "failed", "operations": [], "python_free_qualification": "pending"}
    result["execution"] = {key: receipt.get(key) for key in
                           ("status", "exit_code", "process_returncode", "peak_rss_bytes")}
    if receipt["exit_code"] != 0:
        result["status"] = "failed"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return {"estimator": name, "rows": rows, "backend": backend,
            "status": result["status"], "evidence": str(target)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--worker-spec":
        return int(run_case(json.loads(argv[1]))["status"] != "observed")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "hip"), default="cpu")
    parser.add_argument("--estimators", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 96])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Source/wheel identifier for before/after attribution")
    parser.add_argument("--before", type=Path, help="Previous observation summary.json for count comparison")
    parser.add_argument("--seconds", type=float, default=600, help="Per-case execution deadline, never a speed score")
    parser.add_argument("--label-dtype", default="int64",
                        help="NumPy integer dtype of classifier targets (e.g. int16)")
    parser.add_argument("--cpython-count", choices=("off", "perf"), default="off",
                        help="perf: count CPython C-API calls per public operation with uprobes (Linux, root)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if any(rows < 8 for rows in args.rows):
        parser.error("each case needs at least eight rows")
    os.environ["MOJOLEARN_PROBE_LABEL_DTYPE"] = args.label_dtype
    args.evidence = args.evidence.resolve()
    args.evidence.mkdir(parents=True, exist_ok=True)
    cases = [(name, args.backend, rows, str(args.evidence))
             for name in dict.fromkeys(args.estimators) for rows in dict.fromkeys(args.rows)]
    if (args.evidence / "summary.json").exists() or any(
            (args.evidence / f"{backend}-{name}-{rows}.json").exists() for name, backend, rows, _ in cases):
        parser.error("case evidence already exists; use a fresh evidence directory")
    if sys.platform == "linux":
        from build_sizing import box_resources
        allocation, _ = box_resources()
    else:
        allocation = os.cpu_count() or 1
    workers = min(allocation, len(cases)) if args.backend == "cpu" else 1
    # Every case executes in a fresh bounded subprocess. The existing process
    # controller retains full logs and exit/signal receipts even for native
    # aborts. An abort never destroys other workers or hides their results.
    from functools import partial
    count = None if args.cpython_count == "off" else args.cpython_count
    if count == "perf":
        if sys.platform != "linux":
            parser.error("--cpython-count perf needs Linux perf uprobes")
        perf_probes_add(Path(__file__).resolve().parent.parent, args.evidence)
    try:
        if count == "perf":
            # Every (case, operation) process is an independent unit: one pool.
            units = [(case, operation) for case in cases for operation in case_operations(case[0])]
            workers = min(allocation, len(units)) if args.backend == "cpu" else 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                counted = list(pool.map(partial(launch_counted, seconds=args.seconds), units))
            results = [merge_counted(case, [row for (owner, _), row in zip(units, counted) if owner is case])
                       for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(partial(launch_case, seconds=args.seconds), cases))
    finally:
        if count == "perf":
            perf_probes_remove()
    report = {"schema": "mojolearn-python-crossing-observation-v1", "label": args.label,
              "backend": args.backend, "workers": workers, "allocation": allocation,
              "cases": results, "failed": sum(row["status"] != "observed" for row in results),
              "python_free_qualification": "pending independent native execution evidence",
              "timing": "not measured"}
    if count == "perf":
        # Per (estimator, operation): the call count at each shape and the
        # largest/smallest ratio; a thin UNPACK/PACK has ratio 1.0.
        scaling = {}
        for row in results:
            for operation, calls in (row.get("cpython_calls") or {}).items():
                scaling.setdefault((row["estimator"], operation), {})[row["rows"]] = calls
        report["cpython_scaling"] = [
            {"estimator": estimator, "operation": operation,
             "calls_by_rows": {str(rows): calls for rows, calls in sorted(by_rows.items())},
             "max_over_min": (max(by_rows.values()) / min(by_rows.values())
                              if all(by_rows.values()) and min(by_rows.values()) > 0 else None)}
            for (estimator, operation), by_rows in sorted(scaling.items())]
    if args.before:
        previous = json.loads(args.before.read_text())
        lookup = {(row["estimator"], row["backend"], row["rows"]): row for row in previous["cases"]}
        report["before_label"] = previous["label"]
        report["comparisons"] = []
        for row in results:
            earlier = lookup.get((row["estimator"], row["backend"], row["rows"]))
            comparison = ({"status": "incomparable", "reason": "no matching prior case"} if earlier is None
                          else compare_observations(json.loads(Path(earlier["evidence"]).read_text()),
                                                    json.loads(Path(row["evidence"]).read_text())))
            report["comparisons"].append(dict(estimator=row["estimator"], rows=row["rows"], **comparison))
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("label", "workers", "allocation", "failed")}))
    for row in report.get("cpython_scaling", []):
        print(json.dumps(row))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
