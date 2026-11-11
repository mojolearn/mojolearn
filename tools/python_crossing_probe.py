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
from concurrent.futures import ProcessPoolExecutor
import ctypes
import hashlib
import json
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


def _model(ml, name):
    parameters = {
        "Ridge": {"alpha": 1.0}, "LinearRegression": {},
        "LogisticRegression": {"max_iter": 20},
        "PCA": {"n_components": 2},
        "KMeans": {"n_clusters": 3, "max_iter": 10, "random_state": 0},
        "RandomForestClassifier": {"n_estimators": 3, "max_depth": 3, "random_state": 0},
        "GradientBoosting": {"loss": "RMSE", "n_estimators": 3, "max_depth": 2,
                             "random_state": 0},
    }
    return getattr(ml, name)(numeric_mode="identical", **parameters[name])


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
        model = _model(ml, name)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "hip"), default="cpu")
    parser.add_argument("--estimators", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 96])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Source/wheel identifier for before/after attribution")
    args = parser.parse_args(argv)
    if any(rows < 8 for rows in args.rows):
        parser.error("each case needs at least eight rows")
    args.evidence.mkdir(parents=True, exist_ok=True)
    cases = [(name, args.backend, rows, str(args.evidence))
             for name in dict.fromkeys(args.estimators) for rows in dict.fromkeys(args.rows)]
    if sys.platform == "linux":
        from build_sizing import box_resources
        allocation, _ = box_resources()
    else:
        allocation = os.cpu_count() or 1
    workers = min(allocation, len(cases)) if args.backend == "cpu" else 1
    # Spawn prevents inherited product modules/backend state from contaminating
    # another case. One task per child keeps caches and profilers independent.
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        results = list(pool.map(run_case, cases))
    report = {"schema": "mojolearn-python-crossing-observation-v1", "label": args.label,
              "backend": args.backend, "workers": workers, "allocation": allocation,
              "cases": results, "failed": sum(row["status"] != "observed" for row in results),
              "python_free_qualification": "pending independent native execution evidence",
              "timing": "not measured"}
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("label", "workers", "allocation", "failed")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
