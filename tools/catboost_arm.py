# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CatBoost CPU arm of the interleaved benchmark, called from Mojo.

One function, positional args only (Mojo interop calls it directly). The
pool is quantized OUTSIDE the timed region on both sides of the comparison:
the Mojo side consumes a prebuilt compressed index, so CatBoost gets the
same courtesy.
"""
import os
import time
import numpy as np
import catboost

# WHICH DEVICE CATBOOST RUNS ON, added 2026-08-20 for the NVIDIA column.
#
# Every number this repository has recorded was taken on Apple silicon, where
# CatBoost has NO GPU arm at all -- `task_type="GPU"` raises there -- so the
# CPU arm was not a choice, it was the only thing that existed. That is the
# whole thesis and it is why the default here stays "CPU": flipping it would
# silently change what every existing result means.
#
# On NVIDIA the thesis does not hold. CatBoost ships a real CUDA learner, it
# is the arm any NVIDIA user would actually run, and conda-forge's linux-64
# build in this repository's own lock is `catboost-1.2.10-cuda129`. Quoting
# our GPU against their CPU on a box where their GPU is one keyword away
# would be picking the weaker opponent. So on NVIDIA the honest arm is
# `MOJOLEARN_CATBOOST_TASK_TYPE=GPU`, and we expect to LOSE that one --
# recording the size of the loss is the point of running it.
_TASK_TYPE = os.environ.get("MOJOLEARN_CATBOOST_TASK_TYPE", "CPU").upper()
if _TASK_TYPE not in ("CPU", "GPU"):
    raise SystemExit("MOJOLEARN_CATBOOST_TASK_TYPE must be CPU or GPU, got "
                     + _TASK_TYPE)

def task_type():
    """Read back by the harness so the printed table names the arm it timed.
    A device is not something a reader should have to infer from a filename."""
    return _TASK_TYPE

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
        task_type=_TASK_TYPE,
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
        task_type=_TASK_TYPE,
    )
    t0 = time.perf_counter()
    m.fit(p)
    dt = time.perf_counter() - t0
    mse = float(np.mean((m.predict(x) - y) ** 2))
    return [dt, mse]

def predict_prep(scratch, prefix, border_count, trees, depth):
    """Train once and cache (model, RAW pool) for `predict_seconds`. The
    raw pool mirrors their model_evaluation_speed notebook: `cb.Pool(X)`
    built OUTSIDE the timed region, quantization inside `predict` against
    the model's own borders."""
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
        task_type=_TASK_TYPE,
    )
    m.fit(p)
    x = np.load("%s/%s_X.npy" % (scratch, prefix))
    raw = catboost.Pool(x)
    _CACHE[("predict", str(prefix), int(border_count))] = (m, raw)
    return 0.0

def predict_seconds(prefix, border_count, threads):
    """One timed `model.predict(raw_pool)`, their notebook's timed line."""
    m, raw = _CACHE[("predict", str(prefix), int(border_count))]
    t0 = time.perf_counter()
    _ = m.predict(raw, thread_count=int(threads))
    return time.perf_counter() - t0

def fit_seconds_and_mse_bayesian(scratch, prefix, border_count, trees, depth):
    """The same pinned fit but at THEIR GPU-default sampling: Bayesian
    bootstrap, temperature 1. Stochastic, so the harness compares mse
    BANDS between arms, not bits."""
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
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        rsm=1.0,
        has_time=True,
        random_seed=0,
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
        task_type=_TASK_TYPE,
    )
    t0 = time.perf_counter()
    m.fit(p)
    dt = time.perf_counter() - t0
    mse = float(np.mean((m.predict(x) - y) ** 2))
    return [dt, mse]

def fit_and_test_mse_bayesian(scratch, prefix, border_count, trees, depth,
                              train_rows, seed):
    """The holdout-quality arm: train on the first `train_rows` at
    Bayesian temperature 1 with `seed`, quantized against the SAME
    full-data borders tsv the Mojo arm binned with (`input_borders`), and
    score the held-out tail. Returns [fit_seconds, train_mse, test_mse]."""
    x = np.load("%s/%s_X.npy" % (scratch, prefix))
    y = np.load("%s/%s_y.npy" % (scratch, prefix)).astype(np.float32)
    tr = int(train_rows)
    xt, yt = x[:tr], y[:tr]
    xh, yh = x[tr:], y[tr:]
    key = ("holdout", str(prefix), int(border_count), tr)
    if key not in _CACHE:
        p = catboost.Pool(xt, yt)
        p.quantize(input_borders="%s/%s_borders_%d.tsv"
                   % (scratch, prefix, int(border_count)))
        _CACHE[key] = p
    p = _CACHE[key]
    m = catboost.CatBoostRegressor(
        iterations=int(trees),
        depth=int(depth),
        learning_rate=0.3,
        l2_leaf_reg=3.0,
        border_count=int(border_count),
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        rsm=1.0,
        has_time=True,
        random_seed=int(seed),
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
        task_type=_TASK_TYPE,
    )
    t0 = time.perf_counter()
    m.fit(p)
    dt = time.perf_counter() - t0
    train_mse = float(np.mean((m.predict(xt) - yt) ** 2))
    test_mse = float(np.mean((m.predict(xh) - yh) ** 2))
    return [dt, train_mse, test_mse]

def fit_and_test_mse_cat(scratch, prefix, trees, depth, train_rows, seed):
    """The CATEGORICAL arm: real cat features, frequency information ONLY
    (max_ctr_complexity=1 disables combinations), one_hot_max_size 2.

    PINNED FROM THEIR OWN ERROR: `FeatureFreq` -- the GPU learner's
    frequency ctr and the formula our fixture carries -- "is not
    implemented on CPU yet" (catboost_options.cpp:509). The CPU spelling
    of the same information is `Counter` (a slightly different
    normalization of the same counts), so this arm is the same
    INFORMATION CLASS, not the same bits: quality bands compare, values
    do not."""
    import catboost.datasets
    # catboost.datasets fetches over the network on EVERY call (no local
    # cache dir on this box), and a 3-rep run once stalled half an hour
    # inside rep 2's sock_connect before recovering. Download once per
    # process.
    raw_key = ("raw_df", str(prefix))
    if raw_key not in _CACHE:
        if str(prefix) == "amazon":
            _CACHE[raw_key] = catboost.datasets.amazon()[0]
        elif str(prefix) == "adult":
            _CACHE[raw_key] = catboost.datasets.adult()[0]
        else:
            raise ValueError(prefix)
    train_df = _CACHE[raw_key]
    if str(prefix) == "amazon":
        y = train_df["ACTION"].to_numpy().astype(np.float32)
        xdf = train_df.drop(columns=["ACTION"])
        cat_idx = list(range(xdf.shape[1]))
    elif str(prefix) == "adult":
        y = (train_df["income"] == ">50K").to_numpy().astype(np.float32)
        xdf = train_df.drop(columns=["income"])
        cat_idx = [i for i, c in enumerate(xdf.columns)
                   if xdf[c].dtype == object]
    else:
        raise ValueError(prefix)
    if str(prefix) == "adult":
        # adult's object columns carry real float NaN among strings, and
        # catboost.Pool refuses them outright ("NaN values should be
        # converted to string"). The fixture side (tools/ctr_prep.py) maps
        # them to the string "nan", so both arms see the same category.
        xdf = xdf.copy()
        for i in cat_idx:
            c = xdf.columns[i]
            xdf[c] = xdf[c].fillna("nan").astype(str)
    x = xdf.astype(str) if str(prefix) == "amazon" else xdf
    tr = int(train_rows)
    m = catboost.CatBoostRegressor(
        iterations=int(trees),
        depth=int(depth),
        learning_rate=0.3,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        one_hot_max_size=2,
        simple_ctr="Counter",
        max_ctr_complexity=1,
        rsm=1.0,
        has_time=True,
        random_seed=int(seed),
        model_shrink_rate=0.0,
        boost_from_average=False,
        leaf_estimation_iterations=1,
        random_strength=0.0,
        logging_level="Silent",
        allow_writing_files=False,
        task_type=_TASK_TYPE,
    )
    t0 = time.perf_counter()
    m.fit(catboost.Pool(x[:tr], y[:tr], cat_features=cat_idx))
    dt = time.perf_counter() - t0
    train_mse = float(np.mean((m.predict(x[:tr]) - y[:tr]) ** 2))
    test_mse = float(np.mean((m.predict(x[tr:]) - y[tr:]) ** 2))
    return [dt, train_mse, test_mse]
