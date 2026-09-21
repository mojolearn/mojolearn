#!/usr/bin/env python3
"""Exact repeated-inference A/B for the ordered resident RF/ET candidate."""
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


def digest(value):
    a = np.ascontiguousarray(np.asarray(value))
    return "%s:%s:%s" % (hashlib.sha256(a.tobytes()).hexdigest(), a.dtype, tuple(a.shape))


def load_data(name):
    import speed_gbdt_arm as data
    if name == "taxi":
        d = data.load_taxi("shipped", regression=False)
        cols = [data.TAXI_FEATURES.index(c) for c in data.TAXI_NUMERIC]
        x = np.ascontiguousarray(d.X_train[:4_000_000, cols], dtype=np.float32)
        y = np.ascontiguousarray(d.y_train[:4_000_000], dtype=np.float32)
    elif name == "istella":
        d = data.load_istella("shipped", regression=False)
        x = np.ascontiguousarray(d.X_train, dtype=np.float32)
        y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    else:
        raise ValueError(name)
    return x, y


def prepare(args):
    import mojolearn as ml
    os.makedirs(args.out, exist_ok=True)
    manifest = {"trees": args.trees, "depth": args.depth, "train_rows": args.train_rows,
                "datasets": {}}
    for dataset in ("taxi", "istella"):
        x, y = load_data(dataset)
        n = min(args.train_rows, len(x))
        entry = {"shape": list(x.shape), "positive_fraction": float(y.mean()), "models": {}}
        for kind, cls in (("rf", ml.RandomForestClassifier),
                          ("et", ml.ExtraTreesClassifier)):
            model = cls(n_estimators=args.trees, max_depth=args.depth,
                        max_features="sqrt", random_state=7, device="gpu",
                        numeric_mode="identical", inference_engine="sequential")
            t0 = time.perf_counter()
            model.fit(x[:n], y[:n])
            path = os.path.join(args.out, "%s-%s.npz" % (dataset, kind))
            model.save(path)
            entry["models"][kind] = {"path": path, "fit_s": time.perf_counter() - t0}
            print("PREP", dataset, kind, "shape", x.shape, "fit_rows", n,
                  "fit_s", entry["models"][kind]["fit_s"], flush=True)
        manifest["datasets"][dataset] = entry
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def quality(op, result, y):
    a = np.asarray(result)
    if op == "predict":
        return {"accuracy": float(np.mean(a == y))}
    p = np.asarray(a, dtype=np.float64)
    chosen = np.where(y.astype(bool), p[:, 1], p[:, 0])
    return {"logloss": float(-np.log(np.clip(chosen, 1e-300, 1.0)).mean())}


def run(args):
    import mojolearn as ml
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    x, y = load_data(args.dataset)
    x = np.ascontiguousarray(x[:args.rows])
    y = np.ascontiguousarray(y[:args.rows])
    rows = []
    for kind, cls in (("rf", ml.RandomForestClassifier),
                      ("et", ml.ExtraTreesClassifier)):
        model = cls.load(os.path.join(args.models, "%s-%s.npz" % (args.dataset, kind)))
        native = model._bind()
        selected = int(native.forest_ordered_resident())
        expected = 1 if args.arm == "ordered" else 0
        if selected != expected:
            raise RuntimeError("%s binding reports ordered=%d, expected %d" %
                               (kind, selected, expected))
        model.inference_engine = "parallel_groves" if args.arm == "ordered" else "sequential"
        for op in ("predict", "proba"):
            call = model.predict if op == "predict" else model.predict_proba
            warm = call(x)
            hashes = [digest(warm)]
            samples = []
            last = warm
            for _ in range(args.rounds):
                t0 = time.perf_counter_ns()
                last = call(x)
                samples.append((time.perf_counter_ns() - t0) / 1e6)
                hashes.append(digest(last))
            ordered = sorted(samples)
            rec = {"dataset": args.dataset, "kind": kind, "operation": op,
                   "arm": args.arm, "outer": args.outer, "shape": list(x.shape),
                   "rounds_ms": samples, "median_ms": statistics.median(samples),
                   "spread": max(samples) / min(samples), "hash": hashes[0],
                   "hashes_equal": len(set(hashes)) == 1,
                   "quality": quality("predict" if op == "predict" else "proba", last, y),
                   "ordered_selected": bool(selected), "numeric_mode": ml.numeric_mode(),
                   "vendor": ml.vendor()}
            if len(ordered) < 5 or not rec["hashes_equal"]:
                raise RuntimeError("unstable output in %s/%s/%s" % (args.dataset, kind, op))
            rows.append(rec)
            print("TIME", args.dataset, kind, op, args.arm,
                  "median_ms=%.3f spread=%.3f hash=%s" %
                  (rec["median_ms"], rec["spread"], rec["hash"][:16]), flush=True)
    with open(args.json, "w") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)


def summarize(args):
    records = []
    for pattern in args.inputs:
        for path in glob.glob(pattern):
            with open(path) as fh:
                records.extend(json.load(fh))
    table = []
    verdict = "promote"
    for dataset in ("taxi", "istella"):
        for kind in ("rf", "et"):
            for op in ("predict", "proba"):
                cell = [r for r in records if (r["dataset"], r["kind"], r["operation"]) ==
                        (dataset, kind, op)]
                arms = {a: [r for r in cell if r["arm"] == a]
                        for a in ("sequential", "ordered")}
                if any(len(v) < args.outers for v in arms.values()):
                    raise RuntimeError("missing processes for %s/%s/%s" % (dataset, kind, op))
                hashes = {r["hash"] for r in cell}
                qualities = {json.dumps(r["quality"], sort_keys=True) for r in cell}
                exact = len(hashes) == 1 and len(qualities) == 1 and all(r["hashes_equal"] for r in cell)
                process_medians = {a: [r["median_ms"] for r in rs]
                                   for a, rs in arms.items()}
                process_spread = {a: max(xs) / min(xs)
                                  for a, xs in process_medians.items()}
                # The unit of replication is the alternating process median.
                # Retain every within-process max/min above, but do not let a
                # one-millisecond scheduling excursion veto a 5-call median.
                stable = process_spread["ordered"] <= args.spread
                base = statistics.median(r["median_ms"] for r in arms["sequential"])
                cand = statistics.median(r["median_ms"] for r in arms["ordered"])
                ratio = cand / base
                conservative_ratio = (max(process_medians["ordered"]) /
                                      min(process_medians["sequential"]))
                qualified = exact and stable
                if not qualified or conservative_ratio > args.promote_ratio:
                    verdict = "reject"
                table.append({"dataset": dataset, "kind": kind, "operation": op,
                              "sequential_ms": base, "ordered_ms": cand,
                              "ordered_over_sequential": ratio, "exact": exact,
                              "conservative_ordered_over_sequential": conservative_ratio,
                              "stable": stable, "qualified": qualified,
                              "process_median_spread": process_spread,
                              "quality": arms["sequential"][0]["quality"],
                              "process_spreads": {a: [r["spread"] for r in rs]
                                                  for a, rs in arms.items()}})
    result = {"verdict": verdict, "promotion_rule":
              "all 8 cells exact, ordered process-median spread <= %.3f, and slowest ordered / fastest sequential <= %.3f" %
              (args.spread, args.promote_ratio),
              "cells": table}
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=1_000_000)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--depth", type=int, default=16)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=("taxi", "istella"), required=True)
    r.add_argument("--arm", choices=("sequential", "ordered"), required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--outer", type=int, required=True)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--rows", type=int, default=1_000_000)
    r.add_argument("--json", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--outers", type=int, default=3)
    s.add_argument("--spread", type=float, default=1.10)
    s.add_argument("--promote-ratio", type=float, default=0.98)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
