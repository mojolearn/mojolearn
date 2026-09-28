#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AdaBoostRegressor member diagnosis (trees-apple, 2026-09-28): FAST on the
M3 Ultra scored RMSE 15.99 on taxireg against IDENTICAL's 6.18 with nearly
the same member errors. This fits the board's AdaBoostRegressor once and
prints, per member, its error, weight and the range / NaN count / RMSE of
its own predictions on the training rows, so a member that predicts
outlandish values on a few rows (which, under the linear loss, divides every
other row's error by a huge maximum and wins a huge weight) is visible.

    MOJOLEARN_NUMERIC_MODE=fast pixi run -e default python bench/speed/trees_adaboost_reg_diag.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def seeds_main():
    """TAP_SEEDS="7,11,13,17,19" TAP_DATASETS="taxireg,istellareg": the paired
    quality table (one ADSEED line per dataset and seed: members, test RMSE)."""
    import speed_gbdt_arm as spec
    import mojolearn as m
    rows = int(os.environ.get("TAP_ROWS", "1000000"))
    for ds in os.environ.get("TAP_DATASETS", "taxireg").split(","):
        data = spec.load_dataset(ds, "shipped", rows)
        x = np.ascontiguousarray(data.X_train, dtype=np.float32)
        y = np.ascontiguousarray(data.y_train, dtype=np.float32)
        xt = np.ascontiguousarray(data.X_test[:200000], dtype=np.float32)
        yt = np.asarray(data.y_test[:200000], dtype=np.float64)
        for seed in [int(v) for v in os.environ["TAP_SEEDS"].split(",")]:
            model = m.AdaBoostRegressor(m.DecisionTreeRegressor(max_depth=3), n_estimators=50,
                                        random_state=seed).fit(x, y)
            rmse = float(np.sqrt(np.mean((np.asarray(model.predict(xt), dtype=np.float64) - yt) ** 2)))
            print("ADSEED " + json.dumps(dict(mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset"),
                                              dataset=ds, seed=seed, members=len(model.estimators_),
                                              rmse=round(rmse, 5))), flush=True)


def main():
    if os.environ.get("TAP_SEEDS"):
        return seeds_main()
    import speed_gbdt_arm as spec
    import mojolearn as m
    rows = int(os.environ.get("TAP_ROWS", "1000000"))
    data = spec.load_dataset(os.environ.get("TAP_DATASET", "taxireg"), "shipped", rows)
    x = np.ascontiguousarray(data.X_train, dtype=np.float32)
    y = np.ascontiguousarray(data.y_train, dtype=np.float32)
    xt = np.ascontiguousarray(data.X_test[:200000], dtype=np.float32)
    yt = np.asarray(data.y_test[:200000], dtype=np.float64)
    seed = int(os.environ.get("TAP_SEED", "7"))
    model = m.AdaBoostRegressor(m.DecisionTreeRegressor(max_depth=3), n_estimators=50,
                                random_state=seed).fit(x, y)
    print("ADIAG mode=%s seed=%d members=%d rmse_test=%.4f y_range=[%.3f, %.3f]" % (
        os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset"), seed, len(model.estimators_),
        float(np.sqrt(np.mean((np.asarray(model.predict(xt), dtype=np.float64) - yt) ** 2))),
        float(y.min()), float(y.max())), flush=True)
    for i, (est, err, w) in enumerate(zip(model.estimators_, model.estimator_errors_,
                                          model.estimator_weights_)):
        p = np.asarray(est.predict(x), dtype=np.float64)
        pt = np.asarray(est.predict(xt), dtype=np.float64)
        rec = dict(i=i, err=round(float(err), 6), w=round(float(w), 4),
                   pmin=float(np.nanmin(p)), pmax=float(np.nanmax(p)),
                   nan=int(np.isnan(p).sum()),
                   rmse_train=round(float(np.sqrt(np.nanmean((p - y) ** 2))), 4),
                   rmse_test=round(float(np.sqrt(np.nanmean((pt - yt) ** 2))), 4),
                   leaves=sorted(set(np.round(np.asarray(est._leaves, dtype=np.float64), 3).tolist()))[:8])
        print("ADIAG " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
