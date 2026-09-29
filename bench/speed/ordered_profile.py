#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One Ordered fit of the gbdt-ordered lane, for profiling (lane/ordered-speed).

    MOJOLEARN_STAGE_TIMES=1 pixi run -e default python bench/speed/ordered_profile.py \
        --dataset taxi --trees 20

The lane's own parameters (`speed_gbdt_arm.lane_config('gbdt-ordered')`,
`forest_speed_arm.our_gbdt_arm`) with `n_estimators` overridden by
`--trees`, on the lane's dataset (optionally `--rows` capped). Prints one
`ORD-PROFILE` line per arm: fit wall ms, tree count, total leaves, the
per-tree depth histogram, test AUC and logloss, and a hash of the test
predictions (so a before/after pair can be checked bit for bit). With
`MOJOLEARN_STAGE_TIMES=1` our fit also prints its per-stage table (drains
per stage; a triage table, not a benchmark).

`--catboost-cpu` fits CatBoost CPU (boosting_type='Ordered') on the same
data and parameters (`speed_gbdt_arm.catboost_arms`), `--no-ours` skips our
arm. NOT the board: one round, no interleaving.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import forest_speed_arm as fsa          # noqa: E402
spec = fsa.spec


def _depths(model_leaf_counts):
    hist = {}
    for c in model_leaf_counts:
        d = int(c).bit_length() - 1
        hist[d] = hist.get(d, 0) + 1
    return " ".join(f"d{d}:{n}" for d, n in sorted(hist.items()))


def _report(name, ms, leaf_counts, d, pred, train=None):
    y = np.asarray(d.y_test)
    p = np.asarray(pred, dtype=np.float64)
    h = hashlib.sha256(np.asarray(pred, dtype=np.float32).tobytes()).hexdigest()[:16]
    q = ""
    if d.task == "binary":
        q = f"auc={spec.auc(y, p):.6f} logloss={spec.logloss(y, p):.6f}"
    else:
        q = f"rmse={spec.rmse(y, p):.6f}"
    if train is not None and d.task == "binary":
        ty, tp = train
        q += f" train_auc={spec.auc(np.asarray(ty), np.asarray(tp, dtype=np.float64)):.6f}"
    lc = [int(c) for c in leaf_counts]
    print(f"ORD-PROFILE arm={name} dataset={d.name} rows={d.X_train.shape[0]} "
          f"trees={len(lc)} leaves={sum(lc)} fit_ms={ms:.1f} {q} pred_hash={h} "
          f"depths[{_depths(lc)}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--catboost-cpu", action="store_true")
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--train-auc", action="store_true",
                    help="also score the first 500,000 training rows")
    ap.add_argument("--ours-ab", action="append", default=[],
                    metavar="PARAM=VALUE", help="one estimator keyword changed")
    a = ap.parse_args()

    lane = "gbdt-ordered"
    cfg = dict(spec.lane_config(lane, "shipped"))
    cfg["n_estimators"] = a.trees
    t0 = time.perf_counter()
    d = spec.load_dataset(a.dataset, "shipped", a.rows)
    print(f"ORD-PROFILE load dataset={d.name} shape={d.X_train.shape} "
          f"test={d.X_test.shape} task={d.task} "
          f"load_s={time.perf_counter() - t0:.1f}", flush=True)

    if not a.no_ours:
        fsa.prepare_our_inputs(d)
        extra = {}
        for kv in a.ours_ab:
            k, v = kv.split("=", 1)
            extra[k] = eval(v)  # noqa: S307 -- a local profiling knob
        arm = fsa.our_gbdt_arm(lane, cfg, d, extra or None)
        m = arm.make()
        t0 = time.perf_counter()
        arm.fit(m, d)
        if arm.sync:
            arm.sync()
        ms = (time.perf_counter() - t0) * 1e3
        if d.task == "binary":
            pred = np.asarray(m.predict_proba(d._ours_Xtest))[:, 1]
        else:
            pred = np.asarray(m.predict(d._ours_Xtest))
        train = None
        if a.train_auc and d.task == "binary":
            xt = d._ours_X[:500000]
            train = (d.y_train[:500000], np.asarray(m.predict_proba(xt))[:, 1])
        _report("ours", ms, np.asarray(m.get_tree_leaf_counts()), d, pred, train)

    if a.catboost_cpu:
        import catboost
        arms = spec.catboost_arms(lane, cfg, d, ("cpu",))
        cb = [x for x in arms if x.name == "catboost-cpu"][0]
        m = cb.make()
        t0 = time.perf_counter()
        cb.fit(m, d)
        ms = (time.perf_counter() - t0) * 1e3
        if d.task == "binary":
            pred = m.predict_proba(d.X_test)[:, 1]
        else:
            pred = m.predict(d.X_test)
        _report(f"catboost-cpu-{catboost.__version__}", ms,
                m.get_tree_leaf_counts(), d, pred)


if __name__ == "__main__":
    main()
