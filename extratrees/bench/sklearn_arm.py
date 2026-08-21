#!/usr/bin/env python3
"""scikit-learn's ExtraTrees arm, called FROM the Mojo driver in one process.

Why a Python module and not a second process: this box drifts. The repository
has it measured -- 1.7x on the same binary and the same fixture inside twenty
minutes -- so two numbers taken in two processes minutes apart are not
comparable. `sklearn_interleaved.mojo` alternates the arms inside a single
process and this file is the other arm.

THE DATA IS THE SAME BYTES, not the same generator. Both arms read
`<dir>/<name>_Xcol.f32` and `<dir>/<name>_y.f32` off disk. There is no
recurrence to re-implement and no chance of one arm getting an easier
dataset.

WHAT IS TIMED. `fit` and nothing else. The load, the transpose to
row-major, the dtype conversion and the accuracy computation all happen
outside the timed region, and the loaded arrays are cached across reps so
only the first rep pays for I/O.

n_jobs IS AN ARM, NOT A SETTING. scikit-learn's ExtraTreesClassifier
defaults to `n_jobs=None`, which is one thread. That is the number a user
gets by typing the constructor, and it is the arm that matches ours
parameter for parameter. But a user who types `n_jobs=-1` gets all ten of
this machine's cores, and hiding that would be choosing the flattering
comparison. Both are run and both are reported.
"""

import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

_CACHE = {}


def load(data_dir, name, n_rows, n_features):
    """The same bytes the Mojo arm reads, as a row-major float32 matrix.

    The file is COLUMN-major (feature 0's rows, then feature 1's rows), which
    is what the port's `Dataset` takes. scikit-learn wants (n_samples,
    n_features), so this reshapes to (n_features, n_rows) and transposes. When
    `n_rows` is smaller than the file, the FIRST n_rows of each column are
    taken -- the same subset, taken the same way, as `subset_columns` in the
    driver.
    """
    key = (data_dir, name, n_rows, n_features)
    if key in _CACHE:
        return _CACHE[key]
    # memmap, not fromfile: epsilon's matrix is 3.2 GB and only the first
    # n_rows of each column are wanted. The Mojo arm seeks per column for the
    # same reason and takes the same slice.
    xcol = np.memmap(
        "%s/%s_Xcol.f32" % (data_dir, name), dtype=np.float32, mode="r"
    )
    total_rows = xcol.shape[0] // n_features
    if total_rows * n_features != xcol.shape[0]:
        raise ValueError("X file is not a multiple of n_features")
    if n_rows > total_rows:
        raise ValueError("asked for %d rows, file has %d" % (n_rows, total_rows))
    x = np.ascontiguousarray(
        xcol.reshape(n_features, total_rows)[:, :n_rows].T
    )
    del xcol
    y = np.fromfile("%s/%s_y.f32" % (data_dir, name), dtype=np.float32)[:n_rows]
    # THE MAPPING IS COMPUTED, NOT ASSUMED. covtype's fixture is 1-based and
    # epsilon's is -1/+1. `np.unique(..., return_inverse=True)` ranks the
    # distinct values ascending; `dense_class_ids` in the Mojo arm is the same
    # function, so both arms call the same original label class 0.
    y = np.unique(y, return_inverse=True)[1].astype(np.int32)
    _CACHE[key] = (x, y)
    return x, y


def sklearn_version():
    import sklearn

    return sklearn.__version__


def _max_features(spec):
    """`spec` as scikit-learn spells it. `sqrt`/`log2`/`all` are their strings
    (with `all` being their `None`); anything else is an integer COUNT, which
    scikit-learn and the port both take literally."""
    if spec in ("sqrt", "log2"):
        return spec
    if spec == "all":
        return None
    # `k27` is an integer COUNT. The Mojo arm spells it the same way and
    # passes the string through unchanged, so there is one spelling and not a
    # translation table that can drift.
    if spec.startswith("k"):
        return int(spec[1:])
    return int(spec)


def fit_seconds_and_accuracy(
    data_dir, name, n_rows, n_features, n_trees, depth, seed, n_jobs, spec
):
    """`(seconds, train_accuracy, total_nodes)` for one ExtraTreesClassifier fit.

    Every parameter that both libraries have is set to the same value on both
    sides: `bootstrap=False`, `min_samples_split=2`,
    `min_samples_leaf=1`, `criterion='gini'`, the same `max_depth`, the same
    `max_features` and the same tree count. `random_state` is set on both, but the two RNGs are different
    designs (theirs is a sequential xorshift per split, ours is counter-based
    and keyed), so the forests are NOT the same forest and the accuracy column
    is what says whether they are the same QUALITY.
    """
    x, y = load(data_dir, name, n_rows, n_features)
    clf = ExtraTreesClassifier(
        n_estimators=n_trees,
        criterion="gini",
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=_max_features(spec),
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    clf.fit(x, y)
    secs = time.perf_counter() - t0
    acc = float((clf.predict(x) == y).mean())
    nodes = int(sum(t.tree_.node_count for t in clf.estimators_))
    return (secs, acc, nodes)


def load_regression(data_dir, name, n_rows, n_features):
    """The same matrix, with COLUMN 0 as the target and the rest as features.

    THE TARGET IS CHOSEN BY POSITION, NOT BY RESULT. covtype's column 0 is
    Elevation, a genuine continuous quantity, and it is column 0 -- picked
    before any number existed, which is what keeps this from being a dataset
    built to flatter us. The remaining 53 columns are the features.
    """
    key = ("reg", data_dir, name, n_rows, n_features)
    if key in _CACHE:
        return _CACHE[key]
    xcol = np.memmap(
        "%s/%s_Xcol.f32" % (data_dir, name), dtype=np.float32, mode="r"
    )
    total_rows = xcol.shape[0] // n_features
    view = xcol.reshape(n_features, total_rows)[:, :n_rows]
    y = np.ascontiguousarray(view[0]).astype(np.float64)
    x = np.ascontiguousarray(view[1:].T)
    del xcol
    _CACHE[key] = (x, y)
    return x, y


def fit_regression_seconds_and_mse(
    data_dir, name, n_rows, n_features, n_trees, depth, seed, n_jobs, spec
):
    """`(seconds, train_mse, total_nodes)` for one ExtraTreesRegressor fit.

    Same parameter matching as the classifier arm, and `criterion` is
    `squared_error`, which is what our CRITERION_MSE is.
    """
    x, y = load_regression(data_dir, name, n_rows, n_features)
    reg = ExtraTreesRegressor(
        n_estimators=n_trees,
        criterion="squared_error",
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=_max_features(spec),
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    reg.fit(x, y)
    secs = time.perf_counter() - t0
    pred = reg.predict(x)
    mse = float(((pred - y) ** 2).mean())
    nodes = int(sum(t.tree_.node_count for t in reg.estimators_))
    return (secs, mse, nodes)


def threads_available():
    import joblib

    return int(joblib.cpu_count())
