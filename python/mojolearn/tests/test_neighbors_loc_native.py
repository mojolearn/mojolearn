# SPDX-License-Identifier: Apache-2.0
"""Public neighbor-family call, metadata and refusal seams.

LOF numerical comparisons live in direct native fixtures with independent
mathematical references. Remaining centroid/imputer legacy comparisons keep
their existing coverage until their numerical migration is complete.
"""
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

@pytest.mark.parametrize("contamination", ["auto", 0.1, 0.25, 0.5])
@pytest.mark.parametrize("k", [1, 7, 200])
def test_lof_fit_and_fit_predict_preserve_public_state(contamination, k):
    X = _data()
    model = mojolearn.LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    assert audit_native_call(model.fit, X).assert_native() is model
    scores = np.asarray(model.negative_outlier_factor_)
    assert scores.shape == (60,)
    assert scores.dtype == np.float32
    assert isinstance(model.offset_, float)
    assert model.n_neighbors_ == min(k, 59)
    assert model.n_features_in_ == 5
    assert model.n_samples_fit_ == 60
    labels = np.asarray(audit_native_call(
        mojolearn.LocalOutlierFactor(n_neighbors=k, contamination=contamination).fit_predict, X
    ).assert_native())
    assert labels.shape == (60,)
    assert labels.dtype == np.int64


@pytest.mark.parametrize("contamination", ["auto", 0.2])
def test_lof_novelty_methods_preserve_public_outputs(contamination):
    X, Q = _data(), _data(25, seed=3)
    model = mojolearn.LocalOutlierFactor(n_neighbors=6, contamination=contamination, novelty=True).fit(X)
    decision = np.asarray(audit_native_call(model.decision_function, Q).assert_native())
    predicted = np.asarray(audit_native_call(model.predict, Q).assert_native())
    assert decision.shape == predicted.shape == (25,)
    assert decision.dtype == np.float64
    assert predicted.dtype == np.int64


@pytest.mark.parametrize("kwargs, kind, text", [
    (dict(metric="manhattan"), NotImplementedError, "LocalOutlierFactor: metric='manhattan' (p=2) is not implemented; euclidean (or minkowski with p=2) only"),
    (dict(metric="minkowski", p=3), NotImplementedError, "LocalOutlierFactor: metric='minkowski' (p=3) is not implemented; euclidean (or minkowski with p=2) only"),
    (dict(metric_params={"a": 1}), NotImplementedError, "metric_params is not implemented"),
    (dict(contamination=0.7), ValueError, "contamination must be 'auto' or in (0, 0.5]"),
    (dict(n_neighbors=0), ValueError, "n_neighbors must be >= 1"),
])
def test_lof_refusals_keep_their_wording(kwargs, kind, text):
    with pytest.raises(kind, match=None) as err:
        mojolearn.LocalOutlierFactor(**kwargs).fit(_data())
    assert str(err.value) == text


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
        # Integral float labels are classes; non-integral floats are a
        # continuous target, which scikit-learn's NearestCentroid refuses
        # (test_nearest_centroid_refuses_a_continuous_target).
        return codes.astype(np.float64) * 2.0 - 1.0
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
    yq = _labels(30, kind, seed=5)
    if metric == "manhattan" and not old._uniform():
        # As scikit-learn: with non-uniform priors predict is the argmax of
        # decision_function, which exists for metric='euclidean' only.
        with pytest.raises(AttributeError, match="metric='euclidean' only"):
            new.predict(Q)
        with pytest.raises(AttributeError, match="metric='euclidean' only"):
            new.score(Q, yq)
        return
    _same(audit_native_call(new.predict, Q).assert_native(), old.predict(Q))
    if kind == "float":
        # score is accuracy, whose labels are integers or strings (the
        # accuracy_score contract); float labels are refused there.
        with pytest.raises(TypeError, match="floating labels"):
            new.score(Q, yq)
    else:
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


def test_nearest_centroid_refuses_a_continuous_target():
    y = np.random.default_rng(1).integers(0, 3, 60).astype(np.float64) / 2.0
    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        mojolearn.NearestCentroid().fit(_data(), y)


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
    with pytest.raises(ValueError, match="X has 4 features, but KNNImputer is expecting 6 features"):
        m.transform(_holes(d=4))
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        mojolearn.KNNImputer(n_neighbors=0).fit(_holes()).transform(_holes())
