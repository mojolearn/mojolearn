#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quality of ONE build and environment for the boosted estimators of the
trees family (DART, AdaBoost), paired by subset across arms: for each seed (a
different 300k training subset, the same 200k test rows, the estimator's
random_state = the seed) fit at the board settings of
bench/speed/trees_apple_profile.py and print the test metrics.

    pixi run -e default python bench/speed/trees_apple3_member_quality.py <tag> dart,adaboost taxi,taxireg 5

Every line begins `MQ ` and is one JSON record. Two arms (two tags) of the
same job are compared seed by seed."""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("TAP_REPO") or os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec  # noqa: E402
import trees_apple_profile as tap  # noqa: E402


def main(argv):
    tag = argv[1]
    ests = argv[2].split(",")
    datasets = argv[3].split(",")
    seeds = int(argv[4]) if len(argv) > 4 else 5
    sub = int(os.environ.get("MQ_ROWS", "300000"))
    for ds in datasets:
        d = spec.load_dataset(ds, "shipped", 2_000_000)
        x_all = np.asarray(d.X_train, dtype=np.float32)
        y_all = np.asarray(d.y_train)
        xt = np.ascontiguousarray(np.asarray(d.X_test, dtype=np.float32)[:200000])
        yt = np.asarray(d.y_test)[:200000]
        for est in ests:
            for seed in range(seeds):
                rng = np.random.default_rng(1000 + seed)
                idx = np.sort(rng.choice(len(x_all), size=min(sub, len(x_all)), replace=False))
                x = np.ascontiguousarray(x_all[idx])
                y = np.ascontiguousarray(y_all[idx], dtype=np.float32)
                if d.task != "regression":
                    y = y.astype(np.int64)
                model = tap._make(est, d.task)
                model.random_state = seed
                t = time.time()
                model.fit(x, y)
                fit_s = time.time() - t
                pred = np.asarray(model.predict(xt))
                rec = dict(est=est, ds=ds, seed=seed, tag=tag, fit_s=round(fit_s, 3),
                           digest=tap._digest([pred]))
                if d.task == "regression":
                    rec["rmse"] = float(np.sqrt(np.mean(
                        (pred.astype(np.float64) - yt.astype(np.float64)) ** 2)))
                else:
                    rec["acc"] = float(np.mean(pred.astype(np.int64) == yt.astype(np.int64)))
                    if hasattr(model, "predict_proba"):
                        p = np.asarray(model.predict_proba(xt), dtype=np.float64)
                        if p.ndim == 2 and p.shape[1] == 2:
                            p1 = np.clip(p[:, 1], 1e-12, 1 - 1e-12)
                            rec["logloss"] = float(-np.mean(
                                yt * np.log(p1) + (1 - yt) * np.log(1 - p1)))
                print("MQ " + json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv)
