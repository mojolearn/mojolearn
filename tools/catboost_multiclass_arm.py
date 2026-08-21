"""The CatBoost CPU arm for the MULTICLASS covtype row, self-contained.

A separate module from `catboost_arm.py` for the same reason
`catboost_logloss_arm.py` is: that file carries another lane's
uncommitted work, and a new file cannot collide with it.

Pinned settings with the loss's own defaults, both THEIRS:
`loss_function="MultiClass"` and Newton at ONE iteration
(`GetEstimationMethodDefaults`, `catboost_options.cpp:106-112`, the arm
whose case list is `case MultiClass: case MultiClassOneVsAll:`).

Labels ride raw (covtype's 1..7); CatBoost maps distinct values to
classes in sorted order, which is the same 0..6 the Mojo arm gets from
`y - 1`, so class k is the same class on both sides. The reported loss
is the multiclass cross-entropy `-mean(log p[target])` computed from
`predict_proba`, which is their `functionValue` (`multilogit.cu:87-88`,
`w * log p[targetClass]`) with the sign flipped and divided by the row
count -- the same convention the Mojo `fit` returns, so the two columns
compare as numbers.
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
        p = catboost.Pool(x, y.astype(np.int32))
        p.quantize(border_count=int(border_count))
        _CACHE[key] = p
    return _CACHE[key]


def fit_seconds(scratch, prefix, border_count, trees, depth):
    """Warm-up entry, same shape as the other arms'."""
    return fit_seconds_and_loss(
        scratch, prefix, border_count, trees, depth
    )[0]


def fit_seconds_and_loss(scratch, prefix, border_count, trees, depth):
    p = _pool(str(scratch), str(prefix), int(border_count))
    x = np.load("%s/%s_X.npy" % (scratch, prefix))
    y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
    classes = np.sort(np.unique(y))
    target_idx = np.searchsorted(classes, y)
    m = catboost.CatBoostClassifier(
        iterations=int(trees),
        depth=int(depth),
        learning_rate=0.3,
        l2_leaf_reg=3.0,
        border_count=int(border_count),
        loss_function="MultiClass",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=1,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    m.fit(p)
    dt = time.perf_counter() - t0
    proba = m.predict_proba(x).astype(np.float64)
    n = len(y)
    loss = float(
        -np.mean(np.log(np.maximum(proba[np.arange(n), target_idx],
                                   1e-38)))
    )
    return [dt, loss]
