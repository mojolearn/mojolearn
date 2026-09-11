#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where the host time of one IDENTICAL forest fit goes, Python side.

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python3 tools/forest_host_split.py \
        --lane rf|et|iforest --dataset taxi|istella --rows 1000000 [--reps 3]

Lane forest-speed (2026-09-11). Wraps the module-level helpers each estimator
calls (the column-major conversion, label encoding, the native fit entry and
the model export) with perf_counter clocks and prints one SPLIT line per
wrapped call plus the whole fit, per repetition after one warm-up. Diagnostic
only: the wrappers add a Python call each and no drain, and nothing here is a
certifiable timing (the FSPEED rows are). Inputs are prepared exactly as
bench/speed/forest_speed_arm.py prepares them (C-order float32).
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speed_gbdt_arm as spec  # noqa: E402

CLOCK = defaultdict(float)


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            CLOCK[name] += time.perf_counter() - t0
    return wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lane", required=True, choices=("rf", "et", "iforest"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--rows", type=int, default=1000000)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()

    data = spec.load_dataset(a.dataset, "shipped", a.rows)
    cfg = spec.lane_config(a.lane, "shipped")
    X = np.ascontiguousarray(data.X_train, dtype=np.float32)
    y = np.ascontiguousarray(data.y_train, dtype=np.float32)

    import mojolearn
    from mojolearn import _forest_protocol, randomforest, extratrees, _iforest_impl

    for mod in (randomforest, extratrees):
        for helper in ("as_f32_colmajor", "encode_labels", "_forest_fit_arrays"):
            if hasattr(mod, helper):
                setattr(mod, helper, timed(mod.__name__.rsplit(".", 1)[1] + "." + helper,
                                           getattr(mod, helper)))
    _forest_protocol._export_fit_result = timed("export_fit_result",
                                                _forest_protocol._export_fit_result)
    _iforest_impl.as_f32_c = timed("iforest.as_f32_c", _iforest_impl.as_f32_c)

    def make():
        common = dict(n_estimators=cfg["n_estimators"], random_state=cfg["seed"])
        if a.lane == "iforest":
            return mojolearn.IsolationForest(max_samples=cfg["max_samples"],
                                             max_features=cfg["max_features"],
                                             bootstrap=cfg["bootstrap"], **common)
        common.update(max_depth=cfg["max_depth"], max_features="sqrt",
                      min_samples_leaf=1, min_samples_split=2,
                      min_impurity_decrease=0.0, bootstrap=cfg["bootstrap"],
                      device="gpu")
        if a.lane == "rf":
            return mojolearn.RandomForestClassifier(criterion="gini", n_bins=cfg["n_bins"], **common)
        return mojolearn.ExtraTreesClassifier(criterion="gini", **common)

    for rep in range(a.reps + 1):
        CLOCK.clear()
        t0 = time.perf_counter()
        m = make()
        if a.lane == "iforest":
            m.fit(X)
        else:
            m.fit(X, y)
        total = time.perf_counter() - t0
        tag = "warmup" if rep == 0 else "rep%d" % rep
        for name, s in sorted(CLOCK.items()):
            print("SPLIT lane=%s dataset=%s %s %s=%.1f ms" % (a.lane, a.dataset, tag, name, s * 1e3))
        print("SPLIT lane=%s dataset=%s %s fit_total=%.1f ms unwrapped=%.1f ms"
              % (a.lane, a.dataset, tag, total * 1e3,
                 (total - sum(v for k, v in CLOCK.items() if k != "export_fit_result"
                              and not k.endswith("_forest_fit_arrays"))) * 1e3), flush=True)


if __name__ == "__main__":
    main()
