# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CatBoost CPU arm for the LOGLOSS interleaved row, self-contained.

A separate module from `catboost_arm.py` ON PURPOSE: that file carries
another lane's uncommitted work, and a new file cannot collide with it.

Settings are the repository's pinned set with exactly two changes, both
theirs: `loss_function="Logloss"`, and Logloss's OWN estimation default --
Newton at 10 iterations (`GetEstimationMethodDefaults`,
`catboost_options.cpp:157-164`, the GPU arm the Mojo side transcribed).
Pinning iterations to 1 here would be the `leaf_estimation_iterations=1`
cheat this repository already priced for RMSE; BOTH arms get 10.

The target is binarized at 0.5 on BOTH sides from the SAME y file:
epsilon's labels are -1/+1, so `y > 0.5` puts -1 in class 0 and +1 in
class 1, matching the Mojo arm's `logloss_border = 0.5`. The reported
loss is computed HERE from raw predictions with the identical formula
the Mojo fit returns -- `mean(log(1 + e^a) - t*a)` -- so the two columns
compare as numbers, not as two libraries' reporting conventions
(`np.logaddexp` for the overflow their `functionValue` avoids the same
way).
"""
import time

import numpy as np
import catboost

_CACHE = {}


def _pool(scratch, prefix, border_count):
    key = (str(prefix), int(border_count))
    if key not in _CACHE:
        x = np.load("%s/%s_X.npy" % (scratch, prefix))
        y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
        p = catboost.Pool(x, (y > 0.5).astype(np.float32))
        p.quantize(border_count=int(border_count))
        _CACHE[key] = p
    return _CACHE[key]


def fit_seconds(scratch, prefix, border_count, trees, depth):
    """Warm-up entry, same shape as catboost_arm's."""
    return fit_seconds_and_logloss(
        scratch, prefix, border_count, trees, depth
    )[0]


def fit_seconds_and_logloss(scratch, prefix, border_count, trees, depth):
    p = _pool(str(scratch), str(prefix), int(border_count))
    x = np.load("%s/%s_X.npy" % (scratch, prefix))
    y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
    t = (y > 0.5).astype(np.float64)
    m = catboost.CatBoostClassifier(
        iterations=int(trees),
        depth=int(depth),
        learning_rate=0.3,
        l2_leaf_reg=3.0,
        border_count=int(border_count),
        loss_function="Logloss",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=10,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    m.fit(p)
    dt = time.perf_counter() - t0
    raw = m.predict(x, prediction_type="RawFormulaVal").astype(np.float64)
    loss = float(np.mean(np.logaddexp(0.0, raw) - t * raw))
    return [dt, loss]
