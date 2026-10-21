#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Testing-only whole-public-operation audit; never a product or timing path.

Run in a fresh process with PYTHONPATH pointing to the frozen validation
overlay. Input construction and result inspection are independent controller
work outside each measured native-runtime boundary. No compilation occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("cpu", "metal", "cuda", "hip"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--estimators", nargs="+", choices=(
        "KMeans", "LinearRegression", "Ridge", "LogisticRegression", "PCA",
        "ExtraTreesClassifier", "ExtraTreesRegressor"))
    args = parser.parse_args()
    os.environ["MOJOLEARN_VENDOR"] = args.vendor
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"

    import numpy as np
    import mojolearn as ml
    from mojolearn._verify_native_runtime import audit_native_call

    vendor = ml.vendor()
    mode = ml.numeric_mode()
    if vendor != args.vendor or mode != "identical":
        raise RuntimeError(f"requested {args.vendor}/identical, loaded {vendor}/{mode}")

    # Independent controller data: full-rank regression, separated class
    # labels, and nonzero covariance. No data work enters the product call.
    rng = np.random.default_rng(1607)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    x[:, 0] += np.where(np.arange(len(x)) % 2, 2.0, -2.0)
    regression = (x @ np.array([1.5, -0.75, 0.5], np.float32) + 2).astype(np.float32)
    labels = (np.arange(len(x)) % 2).astype(np.int32)
    queries = np.ascontiguousarray(x[::4])
    cases = []

    def record(estimator, operation, callable_, *values, expected_shape=None, **kwargs):
        audit = audit_native_call(callable_, *values, **kwargs)
        report = audit.report()
        row = {"estimator": estimator, "operation": operation, "audit": report}
        row["status"] = ("operation_error" if audit.exception is not None else
                         "native_pass" if report["passed"] else "runtime_violation")
        if (vendor == "cpu" and estimator.startswith("ExtraTrees") and
                operation == "fit" and audit.exception is not None and
                "training is GPU-only" in str(audit.exception)):
            row["expected_route_refusal"] = True
        if audit.exception is None and expected_shape is not None:
            # Consumption is test oracle work after the public operation.
            out = np.asarray(audit.result)
            valid = out.shape == expected_shape and bool(np.isfinite(out).all())
            row["consumed_output"] = {
                "shape": list(out.shape), "expected_shape": list(expected_shape),
                "dtype": str(out.dtype), "finite": bool(np.isfinite(out).all()),
                "sha256": hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                "shape_and_finite_pass": valid,
            }
            if not valid:
                row["status"] = "output_error"
        cases.append(row)
        return audit

    def exercise(name, cls, constructor, target, outputs):
        if args.estimators is not None and name not in args.estimators:
            return
        built = record(name, "construct", cls, **constructor)
        if built.exception is not None:
            cases.append({"estimator": name, "operation": "fit_and_outputs",
                          "status": "dependency_blocked", "reason": "constructor raised"})
            return
        model = built.result
        fit = record(name, "fit", model.fit, x, *(() if target is None else (target,)))
        if fit.exception is not None:
            for operation, width in outputs:
                cases.append({"estimator": name, "operation": operation,
                              "status": "dependency_blocked", "reason": "fit raised"})
            return
        for operation, width in outputs:
            shape = (len(queries),) if width is None else (len(queries), width)
            record(name, operation, getattr(model, operation), queries, expected_shape=shape)

    exercise("KMeans", ml.KMeans,
             dict(n_clusters=2, n_init=1, max_iter=5, random_state=17), None,
             [("predict", None), ("transform", 2)])
    exercise("LinearRegression", ml.LinearRegression, dict(fit_intercept=True),
             regression, [("predict", None)])
    exercise("Ridge", ml.Ridge, dict(alpha=1.0, fit_intercept=True),
             regression, [("predict", None)])
    exercise("LogisticRegression", ml.LogisticRegression,
             dict(max_iter=12, linesearch_max_iter=10, tol=1e-4), labels,
             [("predict", None), ("predict_proba", 2)])
    exercise("PCA", ml.PCA, dict(n_components=2, svd_solver="full"), None,
             [("transform", 2)])
    tree_options = dict(n_estimators=3, max_depth=3, random_state=17,
                        max_features=1.0, device="cpu" if vendor == "cpu" else "gpu")
    exercise("ExtraTreesClassifier", ml.ExtraTreesClassifier, tree_options,
             labels, [("predict", None), ("predict_proba", 2)])
    exercise("ExtraTreesRegressor", ml.ExtraTreesRegressor, tree_options,
             regression, [("predict", None)])

    native_modules = []
    for name, module in sorted(sys.modules.copy().items()):
        if type(module) is not ModuleType:
            continue
        path = vars(module).get("__file__")
        if path and ("mojolearn" in name or "mojolearn" in path) and path.endswith((".so", ".dylib", ".pyd")):
            resolved = Path(path).resolve()
            native_modules.append({"module": name, "path": path, "resolved_path": str(resolved),
                                   "sha256": digest(resolved)})
    source_files = {frame["file"] for row in cases for frame in row.get("audit", {}).get("python_frames", [])}
    sources = [{"path": path, "sha256": digest(path)} for path in sorted(source_files)
               if Path(path).is_file()]
    summary = dict(Counter(row["status"] for row in cases))
    report = {
        "schema": "mojolearn.public-classical-native-audit.v1",
        "kind": "source-validation-not-installed-wheel",
        "scope": "Representative whole public operations; no performance or exhaustive option-matrix claim.",
        "vendor": vendor, "numeric_mode": mode, "python": sys.version,
        "selected_estimators": args.estimators,
        "controller_path": str(Path(__file__).resolve()), "controller_sha256": digest(__file__),
        "package_path": ml.__file__, "numpy_version": np.__version__,
        "native_reuse_provenance": json.loads(args.provenance.read_text()),
        "native_reuse_provenance_path": str(args.provenance.resolve()),
        "native_reuse_provenance_sha256": digest(args.provenance),
        "qualification_limit": "Reused classical binaries do not qualify changed cast/label source closures.",
        "loaded_native_modules": native_modules, "traced_source_files": sources,
        "summary": summary, "total": len(cases), "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "vendor": vendor, "total": len(cases),
                      "summary": summary, "native_modules": len(native_modules)}))
    return int(any(row["status"] != "native_pass" for row in cases))


if __name__ == "__main__":
    raise SystemExit(main())
