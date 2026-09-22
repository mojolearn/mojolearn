# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Regressions found by the Sep 22 pip-install smoke of the classical family.

1. `mojolearn.Array` had no `__array__`, so scikit-learn's metrics refused a
   classifier's `predict` output ("Expected array-like, got Array").
2. `import mojolearn.metrics` / `from mojolearn.metrics import ...` raised
   ModuleNotFoundError (the attribute is an alias of `_metrics_impl`).
3. `SVC().fit(X, y)` with string labels raised "buffer format '<U1'" while
   LogisticRegression and LinearSVC took the same labels.
4. The classical estimators had no `get_params`, so `cross_val_score`
   refused `mojolearn.Ridge()` and scikit-learn's `clone` refused them all.
5. The float32-only metrics refused plain Python lists of floats.
6. `bootstrap(...).confidence_interval` had no SciPy `.low` / `.high`.

The checks that fit or score need a binding and are skipped, saying so,
without one.
"""

import importlib
import math
import os
import tempfile

import numpy as np
import pytest

import mojolearn
from mojolearn._array import Array


def test_array_dunder_array_is_a_zero_copy_view():
    a = Array.from_list([[1.0, 2.0], [3.0, 4.0]], "<f4")
    v = a.__array__()
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32 and v.shape == (2, 2)
    assert v.__array_interface__["data"][0] == a.__array_interface__["data"][0]
    assert v.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_array_dunder_array_dtype_and_copy():
    a = Array.from_list([1, 2, 3], "<i8")
    f = a.__array__(dtype=np.float64)
    assert f.dtype == np.float64 and f.tolist() == [1.0, 2.0, 3.0]
    c = a.__array__(copy=True)
    assert c.__array_interface__["data"][0] != a.__array_interface__["data"][0]
    assert c.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        a.__array__(dtype=np.float64, copy=False)


def test_numpy_asarray_still_takes_the_interface_path():
    a = Array.from_list([5, 6], "<i8")
    v = np.asarray(a)
    assert v.__array_interface__["data"][0] == a.__array_interface__["data"][0]
    assert v.tolist() == [5, 6]


def test_sklearn_metrics_accept_an_array_of_labels():
    metrics = pytest.importorskip("sklearn.metrics")
    y_true = Array.from_list([0, 1, 1, 0, 2], "<i8")
    y_pred = Array.from_list([0, 1, 0, 0, 2], "<i8")
    assert metrics.accuracy_score(y_true, y_pred) == pytest.approx(0.8)
    assert metrics.confusion_matrix(y_true, y_pred).tolist() == [
        [2, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert metrics.f1_score(y_true, y_pred, average="macro") > 0


def test_mojolearn_metrics_is_importable_as_a_submodule():
    namespace = {}
    exec("import mojolearn.metrics as m\n"
         "from mojolearn.metrics import accuracy_score, r2_score\n", namespace)
    assert namespace["m"] is mojolearn.metrics
    assert namespace["accuracy_score"] is mojolearn.metrics.accuracy_score
    assert importlib.import_module("mojolearn.metrics") is mojolearn.metrics


def _svc_or_skip(x, y, **kw):
    from mojolearn._svm_impl import SVC
    try:
        from mojolearn._cpu_reference import reference_training
        with reference_training():
            return SVC(**kw).fit(x, y)
    except (ImportError, NotImplementedError, OSError) as exc:
        pytest.skip(f"no SVM binding that fits on this install: {exc}")


def test_svc_string_labels_match_the_numeric_fit():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 3)).astype(np.float32)
    codes = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)
    names = np.array(["neg", "pos"])[codes]
    m_num = _svc_or_skip(x, codes, kernel="rbf", gamma=0.5)
    m_str = _svc_or_skip(x, names, kernel="rbf", gamma=0.5)
    assert m_str.classes_ == ["neg", "pos"]
    # labels 0/1 are the codes the string fit hands the solver: same model.
    assert np.asarray(m_str.decision_function(x)).tobytes() == \
        np.asarray(m_num.decision_function(x)).tobytes()
    pred = m_str.predict(x)
    assert list(pred) == [["neg", "pos"][int(c)] for c in np.asarray(m_num.predict(x))]
    assert m_str.score(x, names) == m_num.score(x, codes)
    path = os.path.join(tempfile.mkdtemp(), "svc_str.npz")
    m_str.save(path)
    back = type(m_str).load(path)
    assert back.classes_ == ["neg", "pos"] and list(back.predict(x)) == list(pred)


# 4. The classical estimators had no get_params, so mojolearn's own
#    cross_val_score (and scikit-learn's clone) refused them.
_CLONE_CASES = {
    "LinearRegression": {}, "Ridge": dict(alpha=2.0), "Lasso": dict(alpha=0.3),
    "ElasticNet": dict(alpha=0.3, l1_ratio=0.2), "LogisticRegression": dict(C=0.5),
    "QNRegressor": dict(l2_strength=0.1), "LinearSVC": dict(C=0.3), "LinearSVR": dict(C=0.3),
    "SVC": dict(C=2.0, gamma=0.5), "SVR": dict(C=2.0, kernel="linear"),
    "KernelRidge": dict(alpha=0.3, kernel="rbf"), "Nystroem": dict(n_components=20),
    "RBFSampler": dict(n_components=20), "GaussianProcessRegressor": dict(alpha=2.0 ** -20),
    "GaussianProcessClassifier": {}, "KNeighborsClassifier": dict(n_neighbors=3),
    "KNeighborsRegressor": dict(n_neighbors=3), "NearestNeighbors": dict(n_neighbors=3),
    "RadiusNeighbors": dict(radius=0.5), "PCA": dict(n_components=2),
    "TruncatedSVD": dict(n_components=2), "Cholesky": {}, "ARIMA": dict(order=(2, 1, 0)),
    "KernelDensity": dict(bandwidth=0.5),
}


@pytest.mark.parametrize("name", sorted(_CLONE_CASES))
def test_get_params_and_clone(name):
    from mojolearn.model_selection import _clone
    kw = _CLONE_CASES[name]
    est = getattr(mojolearn, name)(**kw)
    params = est.get_params(deep=False)
    assert "numeric_mode" in params or name in ("Lasso", "ElasticNet")
    for key, value in kw.items():
        assert params[key] is value
    twin = _clone(est)
    assert type(twin) is type(est)
    assert twin.get_params(deep=False).keys() == params.keys()
    base = pytest.importorskip("sklearn.base")
    assert base.clone(est).get_params(deep=False).keys() == params.keys()


def test_set_params_rebuilds_through_the_constructor():
    est = mojolearn.Ridge(alpha=1.0)
    assert est.set_params(alpha=3.0) is est and est.alpha == 3.0
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(nope=1)
    with pytest.raises(NotImplementedError):
        mojolearn.Lasso().set_params(positive=True)


def test_estimator_type_picks_stratified_default_folds():
    from mojolearn.model_selection import _classifier
    for name in ("LogisticRegression", "LinearSVC", "SVC", "KNeighborsClassifier",
                 "GaussianProcessClassifier"):
        assert _classifier(getattr(mojolearn, name)()), name
    for name in ("LinearRegression", "Ridge", "Lasso", "SVR", "KernelRidge",
                 "KNeighborsRegressor", "GaussianProcessRegressor"):
        assert not _classifier(getattr(mojolearn, name)()), name
    utils = pytest.importorskip("sklearn.base")
    assert utils.is_classifier(mojolearn.LogisticRegression())
    assert utils.is_regressor(mojolearn.Ridge())


def test_cross_val_score_takes_mojolearn_ridge():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 3)).astype(np.float32)
    y = (x @ np.array([1.0, -2.0, 0.5], np.float32) + 0.1).astype(np.float32)
    try:
        from mojolearn._cpu_reference import reference_training
        with reference_training():
            scores = mojolearn.cross_val_score(mojolearn.Ridge(alpha=1e-3), x, y, cv=3)
    except (ImportError, OSError) as exc:
        pytest.skip(f"no linear-model binding on this install: {exc}")
    assert len(scores) == 3 and min(scores) > 0.99


# 5. Plain Python lists of floats for the float32-only metrics.
def test_float32_metrics_take_plain_lists():
    try:
        mse = mojolearn.metrics.mean_squared_error([1.0, 2.0, 3.0], [1.5, 2.0, 2.0])
        auc = mojolearn.metrics.roc_auc_score([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.3])
        ll = mojolearn.metrics.log_loss([0, 1], [[0.75, 0.25], [0.5, 0.5]])
    except (ImportError, OSError) as exc:
        pytest.skip(f"no metrics binding on this install: {exc}")
    assert mse == pytest.approx(1.25 / 3, rel=1e-6)
    assert auc == 1.0
    assert ll == pytest.approx(-(math.log(0.75) + math.log(0.5)) / 2, rel=1e-5)
    with pytest.raises(TypeError, match="float32"):
        mojolearn.metrics.mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# 6. SciPy's `confidence_interval.low` / `.high` spelling.
def test_confidence_interval_has_low_and_high():
    from mojolearn.resample import ConfidenceInterval, BootstrapResult
    r = BootstrapResult(1.0, None, None, 0.1, 0.5, 1.5, 3, 97)
    assert isinstance(r.confidence_interval, ConfidenceInterval)
    assert r.confidence_interval.low == 0.5 and r.confidence_interval.high == 1.5
    lo, hi = r.confidence_interval
    assert (lo, hi) == (0.5, 1.5)
