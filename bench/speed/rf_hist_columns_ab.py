#!/usr/bin/env python3
"""Exact two-dataset process gate for the default-off RF histogram tile4."""
import argparse
from contextlib import contextmanager
import glob
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tools"))
import speed_gbdt_arm as data
from mojolearn import RandomForestClassifier

FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def digest(*values):
    out = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        out.update(array.dtype.str.encode())
        out.update(str(array.shape).encode())
        out.update(array.tobytes())
    return out.hexdigest()


def load_binding(path, arm):
    spec = importlib.util.spec_from_file_location("rf_tile_" + arm + "._mojolearn_rf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def bound(module):
    cls = RandomForestClassifier
    had = "_bind" in cls.__dict__
    old = cls.__dict__.get("_bind")
    cls._bind = lambda self, name=None: module
    try:
        yield
    finally:
        if had:
            cls._bind = old
        else:
            del cls._bind


def load_dataset(name, rows, predict_rows):
    loaded = (data.load_taxi("shipped", regression=False) if name == "taxi"
              else data.load_istella("shipped", regression=False))
    n = min(rows, len(loaded.X_train))
    x = np.ascontiguousarray(loaded.X_train[:n], dtype=np.float32)
    y = np.ascontiguousarray(loaded.y_train[:n], dtype=np.float32)
    return x, y, min(predict_rows, n)


def fit_once(module, x, y, predict_rows, args):
    # This integer is the effective public default sqrt(n_features), supplied
    # explicitly so the timing does not include the unrelated portable-math DSO.
    model = RandomForestClassifier(
        n_estimators=args.trees, max_depth=args.depth,
        max_features=max(1, int(math.sqrt(x.shape[1]))), random_state=7,
        device="gpu", numeric_mode="identical", inference_engine="sequential",
    )
    with bound(module):
        start = time.perf_counter_ns()
        model.fit(x, y)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        prediction = np.asarray(model.predict(x[:predict_rows]))
        probability = np.asarray(model.predict_proba(x[:predict_rows]))
    chosen = probability[np.arange(predict_rows), y[:predict_rows].astype(np.int64)]
    arrays = {name: digest(getattr(model, name)) for name in FOREST_ARRAYS}
    metadata = {"classes": list(model.classes_), "n_classes": int(model.n_classes_),
                "n_features": int(model.n_features_in_)}
    return elapsed_ms, {
        "model_array_hashes": arrays,
        "model_hash": hashlib.sha256(json.dumps(
            {"arrays": arrays, "metadata": metadata}, sort_keys=True).encode()).hexdigest(),
        "fitted_metadata": metadata, "predict_hash": digest(prediction),
        "proba_hash": digest(probability),
        "accuracy": float(np.mean(prediction == y[:predict_rows])),
        "logloss": float(-np.log(np.clip(chosen, 1e-300, 1.0)).mean()),
    }


def run(args):
    x, y, predict_rows = load_dataset(args.dataset, args.rows, args.predict_rows)
    module = load_binding(args.binding, args.arm)
    if module.rf_numeric_mode() != 1:
        raise RuntimeError("binding is not IDENTICAL")
    fit_once(module, x, y, predict_rows, args)  # excluded complete warmup
    records, times = [], []
    for repeat in range(args.repeats):
        elapsed, record = fit_once(module, x, y, predict_rows, args)
        record.update(repeat=repeat, fit_ms=elapsed)
        records.append(record)
        times.append(elapsed)
        print("FIT", args.dataset, args.arm, repeat, "ms=%.3f" % elapsed, flush=True)
    if len(records) < 5:
        raise RuntimeError("at least five retained complete fits are required")
    stable = ("model_hash", "predict_hash", "proba_hash", "accuracy", "logloss")
    if any(len({json.dumps(r[key], sort_keys=True) for r in records}) != 1 for key in stable):
        raise RuntimeError("retained fits are not byte/quality identical")
    output = {
        "dataset": args.dataset, "arm": args.arm, "outer": args.outer,
        "launch_position": args.launch_position,
        "binary_sha256": hashlib.sha256(open(args.binding, "rb").read()).hexdigest(),
        "input": {"sha256": digest(x, y), "x_shape": list(x.shape),
                  "y_shape": list(y.shape), "x_dtype": x.dtype.str, "y_dtype": y.dtype.str},
        "trees": args.trees, "depth": args.depth, "predict_rows": predict_rows,
        "median_fit_ms": statistics.median(times), "fit_ms": times,
        "fit_spread": max(times) / min(times), "records": records,
    }
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)


def probe(args):
    log = os.environ.get("RF_LAUNCH_LOG")
    if not log:
        raise RuntimeError("probe requires a fresh RF_LAUNCH_LOG")
    x, y, predict_rows = load_dataset(args.dataset, min(args.rows, 20000), 100)
    module = load_binding(args.binding, args.arm)
    fit_once(module, x, y, predict_rows, args)
    lines = open(log).read().splitlines()
    tiled = sum(line.startswith("histogram_binned_columns4_") for line in lines)
    normal = sum(line == "histogram_binned" for line in lines)
    if args.expect_tile4 and not tiled:
        raise RuntimeError("tile4 compile/selection witness missing")
    if not args.expect_tile4 and tiled:
        raise RuntimeError("baseline unexpectedly selected tile4")
    if not args.expect_tile4 and not normal:
        raise RuntimeError("baseline did not reach the comparable binned histogram route")
    print(json.dumps({"arm": args.arm, "expect_tile4": args.expect_tile4,
                      "tile4_launches": tiled, "normal_launches": normal}, sort_keys=True))


def summarize(args):
    paths = [path for pattern in args.inputs for path in glob.glob(pattern)]
    records = [json.load(open(path)) for path in paths]
    expected = {(dataset, arm, outer) for dataset in ("taxi", "istella")
                for arm in ("baseline", "columns4") for outer in range(3)}
    keys = [(r["dataset"], r["arm"], r["outer"]) for r in records]
    if set(keys) != expected or len(keys) != len(expected):
        raise RuntimeError("need exactly Taxi/Istella x baseline/columns4 x outer 0,1,2")
    cells, verdict = [], "promote"
    for record in records:
        expected_position = int(record["arm"] == "columns4")
        if record["outer"] % 2:
            expected_position = 1 - expected_position
        if record["launch_position"] != expected_position:
            raise RuntimeError("non-alternating process order")
        if len(record["fit_ms"]) < 5 or record["fit_spread"] > args.spread:
            raise RuntimeError("short/unstable timing cell " + repr(keys))
    for dataset in ("taxi", "istella"):
        arms = {arm: [r for r in records if r["dataset"] == dataset and r["arm"] == arm]
                for arm in ("baseline", "columns4")}
        signatures, inputs = [], set()
        for arm_records in arms.values():
            for record in arm_records:
                inputs.add(json.dumps(record["input"], sort_keys=True))
                signatures.extend(json.dumps({k: repeat[k] for k in (
                    "model_array_hashes", "model_hash", "fitted_metadata", "predict_hash",
                    "proba_hash", "accuracy", "logloss")}, sort_keys=True)
                    for repeat in record["records"])
        exact = len(set(signatures)) == 1 and len(inputs) == 1
        baseline = [r["median_fit_ms"] for r in arms["baseline"]]
        candidate = [r["median_fit_ms"] for r in arms["columns4"]]
        conservative = max(candidate) / min(baseline)
        speed_gated = not args.wide_only or dataset == "istella"
        if not exact or (speed_gated and conservative > args.promote_ratio):
            verdict = "reject"
        cells.append({"dataset": dataset, "exact_model_prediction_quality": exact,
                      "baseline_ms": statistics.median(baseline),
                      "columns4_ms": statistics.median(candidate),
                      "columns4_over_baseline": statistics.median(candidate) / statistics.median(baseline),
                      "conservative_columns4_over_baseline": conservative,
                      "speed_gated": speed_gated,
                      "process_fit_spreads": {arm: [r["fit_spread"] for r in rs]
                                              for arm, rs in arms.items()},
                      "input": arms["baseline"][0]["input"]})
    result = {"verdict": verdict, "cells": cells,
              "rule": (("exact bytes/quality on both; wide Istella slowest columns4 / fastest baseline <= %.3f; narrow Taxi route unchanged" if args.wide_only else
                        "exact bytes/quality and slowest columns4 / fastest baseline <= %.3f") % args.promote_ratio)}
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("taxi", "istella"), required=True)
    common.add_argument("--arm", choices=("baseline", "columns4"), required=True)
    common.add_argument("--binding", required=True)
    common.add_argument("--rows", type=int, default=1_000_000)
    common.add_argument("--predict-rows", type=int, default=100_000)
    common.add_argument("--trees", type=int, default=100)
    common.add_argument("--depth", type=int, default=16)
    run_parser = sub.add_parser("run", parents=[common])
    run_parser.add_argument("--outer", type=int, required=True)
    run_parser.add_argument("--launch-position", type=int, choices=(0, 1), required=True)
    run_parser.add_argument("--repeats", type=int, default=5)
    run_parser.add_argument("--json", required=True)
    probe_parser = sub.add_parser("probe", parents=[common])
    probe_parser.add_argument("--expect-tile4", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("inputs", nargs="+")
    summary.add_argument("--spread", type=float, default=1.10)
    summary.add_argument("--promote-ratio", type=float, default=0.98)
    summary.add_argument("--wide-only", action="store_true")
    summary.add_argument("--out", required=True)
    args = parser.parse_args()
    {"run": run, "probe": probe, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
