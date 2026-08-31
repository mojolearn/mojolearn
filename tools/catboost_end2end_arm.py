# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CatBoost arm for the HONEST end-to-end row: quantization INSIDE.

Every interleaved row so far quantizes outside the timed region on both
arms -- correct for kernel comparisons, and CatBoost gets that courtesy
because our arm consumes a prebuilt compressed index. THIS arm is the
row a user experiences: `catboost.Pool(X, y)` build, `fit`, everything
from raw floats inside the timer, against our `train(X, y)` doing the
same. Separate module so no other lane's file is touched.

The mojo harness hands the fixture stem; X rides as a raw col-major
f32 file to keep the two arms' inputs byte-identical, transposed here
to the row-major CatBoost wants OUTSIDE the timer (a layout courtesy,
not a quantization one -- our arm receives col-major natively, so
neither arm is charged for the other's preferred layout).
"""
import time

import numpy as np
import catboost

_CACHE = {}


def _xy(scratch, prefix, n_rows, n_feats):
    key = (str(prefix), int(n_rows), int(n_feats))
    if key not in _CACHE:
        xcol = np.fromfile(
            "%s/%s_Xcol.f32" % (scratch, prefix), dtype=np.float32
        ).reshape(int(n_feats), int(n_rows))
        x = np.ascontiguousarray(xcol.T)
        y = np.fromfile(
            "%s/%s_y.f32" % (scratch, prefix), dtype=np.float32
        )[: int(n_rows)]
        _CACHE[key] = (x, y)
    return _CACHE[key]


def fit_seconds_end2end(scratch, prefix, n_rows, n_feats, border_count,
                        trees, depth):
    x, y = _xy(scratch, prefix, n_rows, n_feats)
    m = catboost.CatBoostRegressor(
        iterations=int(trees),
        depth=int(depth),
        learning_rate=0.3,
        l2_leaf_reg=3.0,
        border_count=int(border_count),
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        rsm=1.0,
        has_time=True,
        random_seed=0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    pool = catboost.Pool(x, y)
    m.fit(pool)
    dt = time.perf_counter() - t0
    mse = float(np.mean((m.predict(x) - y) ** 2))
    return [dt, mse]
