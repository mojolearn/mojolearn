"""The CatBoost CPU arm of the interleaved benchmark, called from Mojo.

One function, positional args only (Mojo interop calls it directly). The
pool is quantized OUTSIDE the timed region on both sides of the comparison:
the Mojo side consumes a prebuilt compressed index, so CatBoost gets the
same courtesy.
"""
import time
import numpy as np
import catboost

_CACHE = {}

def _pool(scratch, prefix, border_count):
    key = (prefix, border_count)
    if key not in _CACHE:
        x = np.load("%s/%s_X.npy" % (scratch, prefix))
        y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
        p = catboost.Pool(x, y)
        p.quantize(border_count=int(border_count))
        _CACHE[key] = p
    return _CACHE[key]

def fit_seconds(scratch, prefix, border_count, trees, depth):
    p = _pool(str(scratch), str(prefix), int(border_count))
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
    m.fit(p)
    dt = time.perf_counter() - t0
    return dt

def fit_seconds_and_mse(scratch, prefix, border_count, trees, depth):
    """Same fit, but also returns the final train mse so every interleaved
    run PROVES accuracy parity instead of assuming it from the oracle."""
    import numpy as np
    p = _pool(str(scratch), str(prefix), int(border_count))
    x = np.load("%s/%s_X.npy" % (scratch, prefix))
    y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
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
    m.fit(p)
    dt = time.perf_counter() - t0
    mse = float(np.mean((m.predict(x) - y) ** 2))
    return [dt, mse]
