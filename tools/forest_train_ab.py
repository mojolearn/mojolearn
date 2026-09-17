#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Forest TRAINING timing, one arm per process (lane/forest-train-speed, 2026-09-17).

    fit        --lane {rf,et} --dataset {taxi,taxireg,istella,istellareg} --rows N
               --rounds R --label NAME --json OUT [--score]
               One warmup fit on 20,000 rows (context, kernel load), then R timed
               fits of the speed board's forest configuration (100 trees, depth 16,
               sqrt or 1.0 features, 128 bins for rf, seed 7;
               tools/speed_gbdt_arm.py::lane_config). Every fitted model's five
               prediction arrays are hashed OUTSIDE the timer. The checkout that
               fits is the one PYTHONPATH names, so two arms are two trees and
               two processes; the loaded binding's path, size, mtime and sha256
               are recorded from inside the process.
    summarize  JSON... --before LABEL --after LABEL
               Pools every process of an arm. The gate is the fan-out brief's:
               5+ timed fits per arm, max/min at most 1.10 on both arms, model
               hashes equal across arms. An arm outside the gate is marked `u`
               and no ratio is quoted from it.

The Python side of `fit` is split by MOJOLEARN_STAGE_TIMES=1 (the wrappers print
BOUNDARY_PYTHON with the binding's wall time); this harness records the whole
`fit` call, which is what a user waits for.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

SPREAD_GATE = 1.10
MODEL_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def _digest(model):
    h = hashlib.sha256()
    for name in MODEL_ARRAYS:
        a = np.ascontiguousarray(np.asarray(getattr(model, name)))
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _binding_record(model, module_name):
    """The binding THIS process fitted through: path, size, mtime, sha256."""
    path = getattr(model._bind(module_name), "__file__", None)
    if not path:
        return None
    st = os.stat(path)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return dict(path=path, bytes=st.st_size, mtime=st.st_mtime, sha256=sha)


def _make(lane, cfg, task, max_features):
    import mojolearn
    common = dict(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        max_features=max_features, min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=cfg["min_samples_split"],
        min_impurity_decrease=cfg["min_impurity_decrease"],
        bootstrap=cfg["bootstrap"], random_state=cfg["seed"], device="gpu")
    if lane == "rf":
        common["n_bins"] = cfg["n_bins"]
        if task == "regression":
            return mojolearn.RandomForestRegressor(criterion="squared_error", **common)
        return mojolearn.RandomForestClassifier(criterion="gini", **common)
    if task == "regression":
        return mojolearn.ExtraTreesRegressor(criterion="squared_error", **common)
    return mojolearn.ExtraTreesClassifier(criterion="gini", **common)


def cmd_fit(args):
    import speed_gbdt_arm as spec
    import mojolearn
    data = spec.load_dataset(args.dataset, "shipped", args.rows)
    cfg = spec.lane_config(args.lane, "shipped")
    if args.trees:
        cfg["n_estimators"] = args.trees
    max_features = spec.max_features_for(data)
    x = np.ascontiguousarray(data.X_train, dtype=np.float32)
    y = np.ascontiguousarray(data.y_train, dtype=np.float32)
    xt = np.ascontiguousarray(data.X_test[:args.score_rows], dtype=np.float32)
    yt = np.asarray(data.y_test[:args.score_rows])
    n_rows, n_cols = x.shape
    print("FTRAIN-HEADER lane=%s dataset=%s rows=%d cols=%d task=%s label=%s mode=%s vendor=%s"
          % (args.lane, args.dataset, n_rows, n_cols, data.task, args.label,
             os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset"), mojolearn.vendor()), flush=True)
    warm = _make(args.lane, cfg, data.task, max_features)
    t0 = time.perf_counter()
    warm.fit(x[:20000], y[:20000])
    warm_ms = (time.perf_counter() - t0) * 1000.0
    del warm
    times, hashes = [], []
    model = None
    for r in range(args.rounds):
        model = _make(args.lane, cfg, data.task, max_features)
        t0 = time.perf_counter()
        model.fit(x, y)
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        hashes.append(_digest(model))
        print("FTRAIN lane=%s dataset=%s label=%s round=%d ms=%.1f hash=%s"
              % (args.lane, args.dataset, args.label, r, ms, hashes[-1][:16]), flush=True)
    quality = None
    if args.score and model is not None:
        pred = np.asarray(model.predict(xt))
        if data.task == "regression":
            quality = dict(metric="rmse", value=float(np.sqrt(np.mean(
                (pred.astype(np.float64) - yt.astype(np.float64)) ** 2))))
        else:
            quality = dict(metric="accuracy", value=float(np.mean(
                pred.astype(np.int64) == yt.astype(np.int64))))
        quality["rows"] = int(xt.shape[0])
        print("FTRAIN-QUALITY", quality, flush=True)
    module = "_mojolearn_rf" if args.lane == "rf" else "_mojolearn_trees"
    rec = dict(
        lane=args.lane, dataset=args.dataset, rows=int(n_rows), cols=int(n_cols),
        task=data.task, label=args.label, config=cfg, max_features=max_features,
        numeric_mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset"),
        vendor=mojolearn.vendor(), warmup_ms=warm_ms, ms=times, hashes=hashes,
        quality=quality, binding=_binding_record(model, module),
        fit_numeric_mode=getattr(model, "_fit_numeric_mode", None),
        checkout=ROOT, stage_times=os.environ.get("MOJOLEARN_STAGE_TIMES", ""),
    )
    with open(args.json, "w") as fh:
        json.dump(rec, fh, indent=1)
    return 0


def cmd_summarize(args):
    cells = {}
    for pattern in args.json:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                r = json.load(fh)
            key = (r["lane"], r["dataset"], r["rows"])
            arm = cells.setdefault(key, {}).setdefault(
                r["label"], dict(ms=[], hashes=set(), quality=[], modes=set()))
            arm["ms"] += r["ms"]
            arm["hashes"].update(r["hashes"])
            arm["modes"].add(str(r.get("fit_numeric_mode") or r["numeric_mode"]))
            if r.get("quality"):
                arm["quality"].append(r["quality"]["value"])
    out = []
    for key in sorted(cells):
        row = dict(lane=key[0], dataset=key[1], rows=key[2], arms={})
        for label, arm in sorted(cells[key].items()):
            ms = arm["ms"]
            spread = max(ms) / min(ms)
            stable = len(ms) >= 5 and spread <= SPREAD_GATE
            row["arms"][label] = dict(
                n=len(ms), median_ms=statistics.median(ms), min_ms=min(ms), max_ms=max(ms),
                spread=spread, stable=stable, hashes=sorted(arm["hashes"]),
                quality=sorted(set(arm["quality"])), modes=sorted(arm["modes"]))
        a, b = row["arms"].get(args.before), row["arms"].get(args.after)
        if a and b:
            row["hashes_equal"] = a["hashes"] == b["hashes"] and len(a["hashes"]) == 1
            if a["stable"] and b["stable"]:
                row["ratio_after_over_before"] = b["median_ms"] / a["median_ms"]
        out.append(row)
        print("%s %s rows=%d" % key)
        for label, arm in row["arms"].items():
            print("  %-12s n=%d median=%.1f ms min=%.1f max=%.1f spread=%.3f%s quality=%s hashes=%s"
                  % (label, arm["n"], arm["median_ms"], arm["min_ms"], arm["max_ms"],
                     arm["spread"], "" if arm["stable"] else " u", arm["quality"],
                     [h[:12] for h in arm["hashes"]]))
        if "hashes_equal" in row:
            print("  hashes_equal=%s ratio(%s/%s)=%s" % (
                row["hashes_equal"], args.after, args.before,
                "%.4f" % row["ratio_after_over_before"]
                if "ratio_after_over_before" in row else "not quoted (u)"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--lane", choices=["rf", "et"], required=True)
    f.add_argument("--dataset", required=True)
    f.add_argument("--rows", type=int, required=True)
    f.add_argument("--rounds", type=int, default=3)
    f.add_argument("--trees", type=int, default=0)
    f.add_argument("--label", required=True)
    f.add_argument("--json", required=True)
    f.add_argument("--score", action="store_true")
    f.add_argument("--score-rows", type=int, default=200000)
    s = sub.add_parser("summarize")
    s.add_argument("json", nargs="+")
    s.add_argument("--before", default="before")
    s.add_argument("--after", default="after")
    s.add_argument("--out", default="")
    args = p.parse_args()
    return cmd_fit(args) if args.cmd == "fit" else cmd_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
