# SPDX-License-Identifier: Apache-2.0
"""LocalOutlierFactor, NearestCentroid and KNNImputer: each public method is
one native call (bindings/neighbors_local_native.mojo) and returns, bit for
bit, what the former Python path returned (tests/_legacy_neighbors_loc.py),
with the same fitted attributes and the same refusals."""
import numpy as np
import pytest

import mojolearn
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_neighbors_loc as legacy


def _bits(value):
    if isinstance(value, list):
        return ("list", [_bits(v) for v in value])
    if isinstance(value, (int, float, str, bool)) or value is None:
        return ("scalar", type(value).__name__, repr(value))
    a = np.asarray(value)
    return ("array", a.dtype.str, a.shape, a.tobytes())


def _same(a, b):
    assert _bits(a) == _bits(b)


def _attrs(model, names):
    return {n: _bits(getattr(model, n)) for n in names}


def _data(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[::9] *= 6.0
    return X


# ---------------------------------------------------------------- LOF

_LOF_ATTRS = ("negative_outlier_factor_", "offset_", "n_neighbors_", "n_features_in_", "n_samples_fit_")


@pytest.mark.parametrize("contamination", ["auto", 0.1, 0.25, 0.5])
@pytest.mark.parametrize("k", [1, 7, 200])
def test_lof_fit_and_fit_predict_match_former_path(contamination, k):
    X = _data()
    new = mojolearn.LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    old = legacy.LegacyLocalOutlierFactor(n_neighbors=k, contamination=contamination)
    assert audit_native_call(new.fit, X).assert_native() is new
    old.fit(X)
    assert _attrs(new, _LOF_ATTRS) == _attrs(old, _LOF_ATTRS)
    _same(mojolearn.LocalOutlierFactor(n_neighbors=k, contamination=contamination).fit_predict(X),
          legacy.LegacyLocalOutlierFactor(n_neighbors=k, contamination=contamination).fit_predict(X))


@pytest.mark.parametrize("contamination", ["auto", 0.2])
def test_lof_novelty_methods_match_former_path(contamination):
    X, Q = _data(), _data(25, seed=3)
    new = mojolearn.LocalOutlierFactor(n_neighbors=6, contamination=contamination, novelty=True).fit(X)
    old = legacy.LegacyLocalOutlierFactor(n_neighbors=6, contamination=contamination, novelty=True).fit(X)
    _same(audit_native_call(new.decision_function, Q).assert_native(), old.decision_function(Q))
    _same(audit_native_call(new.predict, Q).assert_native(), old.predict(Q))


@pytest.mark.parametrize("kwargs, kind, text", [
    (dict(metric="manhattan"), NotImplementedError, "LocalOutlierFactor: metric='manhattan' (p=2) is not implemented"),
    (dict(metric="minkowski", p=3), NotImplementedError, "(p=3) is not implemented"),
    (dict(metric_params={"a": 1}), NotImplementedError, "metric_params is not implemented"),
    (dict(contamination=0.7), ValueError, "contamination must be 'auto' or in (0, 0.5]"),
    (dict(n_neighbors=0), ValueError, "n_neighbors must be >= 1"),
])
def test_lof_refusals_keep_their_wording(kwargs, kind, text):
    with pytest.raises(kind, match=None) as err:
        mojolearn.LocalOutlierFactor(**kwargs).fit(_data())
    assert text in str(err.value)
    with pytest.raises(kind) as old:
        legacy.LegacyLocalOutlierFactor(**kwargs).fit(_data())
    assert str(err.value) == str(old.value)


def test_lof_one_sample_and_novelty_refusals():
    with pytest.raises(ValueError, match="minimum of 2 is required"):
        mojolearn.LocalOutlierFactor().fit(_data(1))
    with pytest.raises(AttributeError):
        mojolearn.LocalOutlierFactor(novelty=True).fit_predict(_data())


# ------------------------------------------------------- NearestCentroid

_NC_ATTRS = ("centroids_", "deviations_", "within_class_std_dev_", "class_prior_", "classes_", "n_features_in_")


def _labels(n, kind, seed=1):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, n)
    if kind == "int":
        return codes.astype(np.int64) * 5 - 3
    if kind == "float":
        return codes.astype(np.float64) / 2.0
    if kind == "str":
        return [("a", "bb", "c")[c] for c in codes]
    return [bool(c % 2) for c in codes]


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("priors", ["uniform", "empirical", [0.2, 0.3, 0.5], [2.0, 1.0, 1.0]])
@pytest.mark.parametrize("shrink", [None, 0.3])
@pytest.mark.parametrize("kind", ["int", "float", "str"])
def test_nearest_centroid_matches_former_path(metric, priors, shrink, kind):
    X, Q = _data(), _data(30, seed=4)
    y = _labels(60, kind)
    new = mojolearn.NearestCentroid(metric=metric, priors=priors, shrink_threshold=shrink)
    old = legacy.LegacyNearestCentroid(metric=metric, priors=priors, shrink_threshold=shrink)
    assert audit_native_call(new.fit, X, y).assert_native() is new
    old.fit(X, y)
    assert _attrs(new, _NC_ATTRS) == _attrs(old, _NC_ATTRS)
    _same(audit_native_call(new.predict, Q).assert_native(), old.predict(Q))
    yq = _labels(30, kind, seed=5)
    assert audit_native_call(new.score, Q, yq).assert_native() == old.score(Q, yq)
    if metric == "euclidean":
        _same(audit_native_call(new.decision_function, Q).assert_native(), old.decision_function(Q))
        _same(audit_native_call(new.predict_proba, Q).assert_native(), old.predict_proba(Q))
    else:
        with pytest.raises(AttributeError, match="metric='euclidean' only"):
            new.decision_function(Q)


def test_nearest_centroid_binary_decision_and_weighted_score():
    X, Q = _data(), _data(20, seed=6)
    y = (X[:, 0] > 0).astype(np.int64)
    new = mojolearn.NearestCentroid().fit(X, y)
    old = legacy.LegacyNearestCentroid().fit(X, y)
    _same(new.decision_function(Q), old.decision_function(Q))
    w = np.linspace(0.5, 2.0, 20)
    yq = (Q[:, 1] > 0).astype(np.int64)
    assert new.score(Q, yq, sample_weight=w) == old.score(Q, yq, sample_weight=w)


@pytest.mark.parametrize("kwargs, y, text", [
    (dict(metric="cosine"), None, "NearestCentroid: metric must be 'euclidean' or 'manhattan'"),
    (dict(), "one", "The number of classes has to be greater than one; got 1 class"),
    (dict(priors=[0.5, 0.5]), None, "priors must have one entry per class"),
    (dict(priors=[0.5, -0.5, 1.0]), None, "priors must be non-negative"),
])
def test_nearest_centroid_refusals_keep_their_wording(kwargs, y, text):
    X = _data()
    labels = np.zeros(60, np.int64) if y == "one" else _labels(60, "int")
    with pytest.raises(ValueError) as err:
        mojolearn.NearestCentroid(**kwargs).fit(X, labels)
    with pytest.raises(ValueError) as old:
        legacy.LegacyNearestCentroid(**kwargs).fit(X, labels)
    assert text in str(err.value) and str(err.value) == str(old.value)


def test_nearest_centroid_zero_variance_refusal():
    X = np.ones((10, 3), np.float32)
    with pytest.raises(ValueError, match="All features have zero variance"):
        mojolearn.NearestCentroid().fit(X, np.arange(10) % 2)


# ---------------------------------------------------------- KNNImputer


def _holes(n=50, d=6, seed=2, value=np.nan):
    X = _data(n, d, seed)
    rng = np.random.default_rng(seed + 10)
    X[rng.random((n, d)) < 0.2] = value
    X[:, 2] = value  # a feature missing everywhere
    return X


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("indicator", [False, True])
@pytest.mark.parametrize("missing", [np.nan, -1.0])
def test_knn_imputer_matches_former_path(weights, keep, indicator, missing):
    X, Q = _holes(value=missing), _holes(20, seed=7, value=missing)
    kw = dict(weights=weights, keep_empty_features=keep, add_indicator=indicator, missing_values=missing, n_neighbors=4)
    new, old = mojolearn.KNNImputer(**kw), legacy.LegacyKNNImputer(**kw)
    assert audit_native_call(new.fit, X).assert_native() is new
    old.fit(X)
    assert new._valid == old._valid and new._miss_cols == old._miss_cols and new.n_features_in_ == old.n_features_in_
    _same(audit_native_call(new.transform, Q).assert_native(), old.transform(Q))
    _same(audit_native_call(mojolearn.KNNImputer(**kw).fit_transform, X).assert_native(),
          legacy.LegacyKNNImputer(**kw).fit_transform(X))


@pytest.mark.parametrize("kwargs, kind, text", [
    (dict(missing_values="x"), NotImplementedError, "missing_values must be a number or NaN"),
    (dict(metric="euclidean"), NotImplementedError, "metric must be 'nan_euclidean'"),
    (dict(weights="w"), NotImplementedError, "weights must be 'uniform' or 'distance'"),
])
def test_knn_imputer_refusals_keep_their_wording(kwargs, kind, text):
    with pytest.raises(kind, match=text):
        mojolearn.KNNImputer(**kwargs).fit(_holes())


def test_knn_imputer_width_and_k_refusals():
    m = mojolearn.KNNImputer().fit(_holes())
    with pytest.raises(ValueError, match="different number of features"):
        m.transform(_holes(d=4))
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        mojolearn.KNNImputer(n_neighbors=0).fit(_holes()).transform(_holes())
