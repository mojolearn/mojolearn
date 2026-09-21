#!/usr/bin/env python3
"""A/B repeated RF training with the fused bootstrap/label-gather arm.

Run each arm in a fresh process after installing its matching RF binding.  The
summary gate compares every model and prediction byte, quality metric, and the
median of repeated complete fits on exactly Taxi and Istella.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(np.asarray(value))
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def metadata(model):
    return {
        "classes": [float(v) for v in model.classes_],
        "n_classes": int(model.n_classes_),
        "n_features_in": int(model.n_features_in_),
    }


def load_data(name, rows, predict_rows):
    import speed_gbdt_arm as data
    if name == "taxi":
        d = data.load_taxi("shipped", regression=False)
        cols = [data.TAXI_FEATURES.index(c) for c in data.TAXI_NUMERIC]
        x = d.X_train[:, cols]
    elif name == "istella":
        d = data.load_istella("shipped", regression=False)
        x = d.X_train
    else:
        raise ValueError(name)
    n = min(rows, len(x))
    pn = min(predict_rows, n)
    return (np.ascontiguousarray(x[:n], dtype=np.float32),
            np.ascontiguousarray(d.y_train[:n], dtype=np.float32), pn)


def fit_once(x, y, args):
    import mojolearn as ml
    model = ml.RandomForestClassifier(
        n_estimators=args.trees, max_depth=args.depth,
        max_features=max(1, int(np.sqrt(x.shape[1]))), random_state=7,
        device="gpu", numeric_mode="identical", inference_engine="sequential",
    )
    t0 = time.perf_counter_ns()
    model.fit(x, y)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    return model, elapsed_ms


def run(args):
    x, y, pn = load_data(args.dataset, args.rows, args.predict_rows)
    if args.repeats < 5:
        raise ValueError("at least five retained complete fits are required")
    times = []
    repeats = []
    expected = int(args.arm == "fused")
    # One complete fit is the excluded warmup. This warms the allocator and
    # kernels without contributing either bytes or timing to the gate.
    warm, _ = fit_once(x, y, args)
    if int(warm._bind().rf_fused_bootstrap_gather()) != expected:
        raise RuntimeError("warmup binding selection mismatch")
    del warm
    for repeat in range(args.repeats):
        model, elapsed_ms = fit_once(x, y, args)
        selected = int(model._bind().rf_fused_bootstrap_gather())
        if selected != expected:
            raise RuntimeError("binding reports fused=%d; expected %d" %
                               (selected, expected))
        pred = np.asarray(model.predict(x[:pn]))
        proba = np.asarray(model.predict_proba(x[:pn]))
        model_parts = [getattr(model, name) for name in FOREST_ARRAYS]
        fitted = metadata(model)
        arrays_hash = digest(*model_parts)
        fitted_hash = hashlib.sha256(
            json.dumps(fitted, sort_keys=True).encode()).hexdigest()
        yi = y[:pn].astype(np.int64)
        chosen = proba[np.arange(pn), yi]
        rec = {
            "repeat": repeat, "fit_ms": elapsed_ms,
            "model_hash": hashlib.sha256(
                (arrays_hash + fitted_hash).encode()).hexdigest(),
            "model_arrays_hash": arrays_hash,
            "model_array_hashes": {name: digest(getattr(model, name))
                                     for name in FOREST_ARRAYS},
            "fitted_metadata": fitted,
            "fitted_metadata_hash": fitted_hash,
            "predict_hash": digest(pred), "proba_hash": digest(proba),
            "accuracy": float(np.mean(pred == y[:pn])),
            "logloss": float(-np.log(np.clip(chosen, 1e-300, 1.0)).mean()),
        }
        repeats.append(rec)
        times.append(elapsed_ms)
        print("FIT", args.dataset, args.arm, repeat,
              "ms=%.3f model=%s" % (elapsed_ms, rec["model_hash"][:16]),
              flush=True)
    stable_hashes = all(
        len({r[key] for r in repeats}) == 1
        for key in ("model_hash", "fitted_metadata_hash", "predict_hash", "proba_hash")
    )
    if not stable_hashes:
        raise RuntimeError("repeated fits were not byte-identical")
    out = {
        "dataset": args.dataset, "arm": args.arm, "outer": args.outer,
        "launch_position": args.launch_position,
        "train_shape": list(x.shape), "predict_rows": pn,
        "input": {"sha256": digest(x, y), "x_dtype": x.dtype.str,
                  "y_dtype": y.dtype.str, "x_shape": list(x.shape),
                  "y_shape": list(y.shape)},
        "trees": args.trees, "depth": args.depth,
        "median_fit_ms": statistics.median(times), "fit_ms": times,
        "fit_spread": max(times) / min(times),
        "fused_selected": bool(expected), "repeats": repeats,
    }
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


def summarize(args):
    records = []
    for pattern in args.inputs:
        for path in glob.glob(pattern):
            with open(path) as fh:
                records.append(json.load(fh))
    expected_keys = {
        (dataset, arm, outer)
        for dataset in ("taxi", "istella")
        for arm in ("baseline", "fused") for outer in range(3)
    }
    actual_keys = [(r["dataset"], r["arm"], r["outer"]) for r in records]
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        raise RuntimeError("matrix must contain one record for every Taxi/Istella x arm x outer 0,1,2")
    for r in records:
        expected_position = (0 if r["arm"] == "baseline" else 1)
        if r["outer"] % 2 == 1:
            expected_position = 1 - expected_position
        if r["launch_position"] != expected_position:
            raise RuntimeError("non-alternating launch order in %r" %
                               ((r["dataset"], r["arm"], r["outer"]),))
        if len(r["fit_ms"]) < 5 or r["fit_spread"] > args.spread:
            raise RuntimeError("unstable/short timing record for %r" %
                               ((r["dataset"], r["arm"], r["outer"]),))
    cells = []
    verdict = "promote"
    for dataset in ("taxi", "istella"):
        arms = {arm: [r for r in records
                      if r["dataset"] == dataset and r["arm"] == arm]
                for arm in ("baseline", "fused")}
        signatures = []
        inputs = set()
        for rs in arms.values():
            for record in rs:
                inputs.add(json.dumps(record["input"], sort_keys=True))
                for repeat in record["repeats"]:
                    signatures.append(tuple(
                        json.dumps(repeat[k], sort_keys=True) for k in (
                            "model_hash", "model_array_hashes",
                            "fitted_metadata", "fitted_metadata_hash",
                            "predict_hash", "proba_hash", "accuracy", "logloss"
                        )
                    ))
        exact = len(set(signatures)) == 1 and len(inputs) == 1
        base = [r["median_fit_ms"] for r in arms["baseline"]]
        fused = [r["median_fit_ms"] for r in arms["fused"]]
        conservative_ratio = max(fused) / min(base)
        if not exact or conservative_ratio > args.promote_ratio:
            verdict = "reject"
        cells.append({
            "dataset": dataset, "exact_model_prediction_quality": exact,
            "baseline_ms": statistics.median(base),
            "fused_ms": statistics.median(fused),
            "fused_over_baseline": statistics.median(fused) / statistics.median(base),
            "conservative_fused_over_baseline": conservative_ratio,
            "quality": {k: arms["baseline"][0]["repeats"][0][k]
                        for k in ("accuracy", "logloss")},
            "input": arms["baseline"][0]["input"],
            "process_fit_spreads": {arm: [r["fit_spread"] for r in rs]
                                    for arm, rs in arms.items()},
        })
    result = {"verdict": verdict, "cells": cells,
              "rule": "both datasets byte/quality exact and slowest fused / fastest baseline <= %.3f" % args.promote_ratio}
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", choices=("taxi", "istella"), required=True)
    p.add_argument("--arm", choices=("baseline", "fused"), required=True)
    p.add_argument("--outer", type=int, required=True)
    p.add_argument("--launch-position", type=int, choices=(0, 1), required=True)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--predict-rows", type=int, default=100_000)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--spread", type=float, default=1.10)
    s.add_argument("--promote-ratio", type=float, default=0.98)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": run, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
