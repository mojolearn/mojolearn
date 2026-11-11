#!/usr/bin/env python3
"""Observe Python calls during public operations, with visibility controls.

This is a test controller, not product code or a timing benchmark. Independent
estimator cases run in separate processes across the CPU allocation. GPU cases
run in order on one device: parallel GPU ownership is a separate contract and
concurrent workloads would obscure attribution. Fit precedes query per model.

The profiler sees Python frames and some C calls, not all calls made by native
code through the CPython API. The calibration demonstrates that limitation.
Zero observed nested calls is never a Python-free qualification. Native entry
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
         "RandomForestClassifier", "GradientBoosting")


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


def run_case(case):
    name, backend, rows, evidence = case
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
        if name in ("LogisticRegression", "RandomForestClassifier"):
            y = (y > 0).astype(np.int64)
        result["input_sha256"] = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        result["parameters"] = model_parameters(name)
        model = getattr(ml, name)(**result["parameters"])
        query = "transform" if name == "PCA" else "predict"
        operations = [("fit", model.fit, (x,) if name in ("PCA", "KMeans") else (x, y)),
                      (query, getattr(model, query), (x[:min(7, rows)],))]
        for operation, function, arguments in operations:
            # These two dependent operations cannot run concurrently: query
            # consumes the fitted state produced by this exact fit.
            observed = PythonCalls()
            record = {"operation": operation, "status": "failed"}
            result["operations"].append(record)
            try:
                value = observed.run(function, *arguments)
                if operation != "fit":
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
    target = Path(evidence) / f"{backend}-{name}-{rows}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return {"estimator": name, "rows": rows, "backend": backend,
            "status": result["status"], "evidence": str(target)}


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
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if any(rows < 8 for rows in args.rows):
        parser.error("each case needs at least eight rows")
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(partial(launch_case, seconds=args.seconds), cases))
    report = {"schema": "mojolearn-python-crossing-observation-v1", "label": args.label,
              "backend": args.backend, "workers": workers, "allocation": allocation,
              "cases": results, "failed": sum(row["status"] != "observed" for row in results),
              "python_free_qualification": "pending independent native execution evidence",
              "timing": "not measured"}
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
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
