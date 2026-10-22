# SPDX-License-Identifier: Apache-2.0
"""Real-path and refusal tests for preprocessing and naive Bayes endpoints changed since 0.8.55."""
import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn import preprocessing as reference_prep
from sklearn.exceptions import NotFittedError

import mojolearn as ml
from mojolearn import preprocessing as pp

_rng = np.random.default_rng(0)
X = _rng.normal(size=(30, 3)).astype(np.float32)
Y = (X[:, 0] > 0).astype(np.int32)
X_CAT = _rng.integers(0, 3, (30, 3)).astype(np.float32)
X_GAPS = X.copy()
X_GAPS[::4, 1] = np.nan


# ---------------------------------------------------------------- MaxAbsScaler
def test_maxabs_fit_transform_matches_reference():
    got = ml.MaxAbsScaler().fit_transform(X)
    want = reference_prep.MaxAbsScaler().fit_transform(X)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ml.MaxAbsScaler().fit(X).transform(X)))


@pytest.mark.parametrize("args,kw,err", [
    pytest.param((X[:, 0],), {}, ValueError, id="1d"),
    pytest.param((X,), {"unknown": 1}, TypeError, id="unknown-keyword"),
    pytest.param(([["a", "b"]],), {}, TypeError, id="strings"),
])
def test_maxabs_fit_transform_refuses_bad_input(args, kw, err):
    with pytest.raises(err):
        ml.MaxAbsScaler().fit_transform(*args, **kw)


# ---------------------------------------------------------------- Binarizer
def test_binarizer_transform_thresholds():
    got = np.asarray(ml.Binarizer(threshold=0.5).fit(X).transform(X))
    np.testing.assert_array_equal(got, (X > 0.5).astype(got.dtype))


def test_binarizer_transform_refusals():
    with pytest.raises(NotFittedError):
        ml.Binarizer().transform(X)
    fitted = ml.Binarizer().fit(X)
    with pytest.raises(ValueError, match="features"):
        fitted.transform(X[:, :2])
    with pytest.raises(ValueError):
        fitted.transform(X[:, 0])


# ---------------------------------------------------------------- naive Bayes
def test_categorical_nb_partial_fit_matches_fit():
    whole = ml.CategoricalNB().fit(X_CAT, Y)
    parts = ml.CategoricalNB().partial_fit(X_CAT[:15], Y[:15], classes=[0, 1])
    parts.partial_fit(X_CAT[15:], Y[15:])
    np.testing.assert_array_equal(np.asarray(parts.predict(X_CAT)), np.asarray(whole.predict(X_CAT)))
    np.testing.assert_allclose(np.asarray(parts.predict_proba(X_CAT)),
                               np.asarray(whole.predict_proba(X_CAT)), rtol=1e-5, atol=1e-6)


def test_categorical_nb_partial_fit_refusals():
    with pytest.raises(ValueError, match="classes"):
        ml.CategoricalNB().partial_fit(X_CAT, Y)
    with pytest.raises(ValueError, match="Negative"):
        ml.CategoricalNB().partial_fit(-X_CAT - 1, Y, classes=[0, 1])
    started = ml.CategoricalNB().partial_fit(X_CAT, Y, classes=[0, 1])
    with pytest.raises(ValueError, match="initial classes"):
        started.partial_fit(X_CAT, Y + 5)
    with pytest.raises(ValueError, match="features"):
        started.partial_fit(X_CAT[:, :2], Y)


@pytest.mark.parametrize("est", [ml.CategoricalNB, ml.GaussianNB], ids=["CategoricalNB", "GaussianNB"])
def test_classifier_has_no_fit_transform(est):
    # fit_transform exists only on transformers (scikit-learn's TransformerMixin).
    model = est()
    assert not hasattr(model, "fit_transform")
    with pytest.raises(AttributeError, match="fit_transform"):
        model.fit_transform(X_CAT, Y)


def test_gaussian_nb_fit_learns_class_means():
    model = ml.GaussianNB().fit(X, Y)
    want = np.stack([X[Y == c].mean(0) for c in (0, 1)])
    np.testing.assert_allclose(np.asarray(model.theta_), want, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(model.classes_), [0, 1])


@pytest.mark.parametrize("make", [
    pytest.param(lambda: ml.GaussianNB().fit(X, Y[:-1]), id="short-y"),
    pytest.param(lambda: ml.GaussianNB().fit(np.where(X > 1, np.nan, X), Y), id="nan"),
    pytest.param(lambda: ml.GaussianNB().fit(X, Y, sample_weight=-np.ones(30, np.float32)), id="negative-weight"),
    pytest.param(lambda: ml.GaussianNB(priors=[0.9, 0.9]).fit(X, Y), id="priors-sum"),
])
def test_gaussian_nb_fit_refusals(make):
    with pytest.raises(ValueError):
        make()


def test_gaussian_nb_partial_fit_matches_fit():
    whole = ml.GaussianNB().fit(X, Y)
    parts = ml.GaussianNB().partial_fit(X[:15], Y[:15], classes=[0, 1]).partial_fit(X[15:], Y[15:])
    np.testing.assert_allclose(np.asarray(parts.theta_), np.asarray(whole.theta_), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(parts.var_), np.asarray(whole.var_), rtol=1e-3, atol=1e-5)


def test_gaussian_nb_partial_fit_refusals():
    with pytest.raises(ValueError, match="classes"):
        ml.GaussianNB().partial_fit(X, Y)
    started = ml.GaussianNB().partial_fit(X, Y, classes=[0, 1])
    with pytest.raises(ValueError, match="features"):
        started.partial_fit(X[:, :2], Y)
    with pytest.raises(ValueError, match="initial classes"):
        started.partial_fit(X, Y + 5)


# ---------------------------------------------------------------- IterativeImputer
def test_iterative_imputer_fit_transform_fills_gaps():
    filled = np.asarray(ml.IterativeImputer(random_state=0).fit(X_GAPS).transform(X_GAPS))
    assert filled.shape == X.shape and np.isfinite(filled).all()
    seen = ~np.isnan(X_GAPS)
    np.testing.assert_allclose(filled[seen], X_GAPS[seen], rtol=1e-6)
    once = np.asarray(ml.IterativeImputer(random_state=0).fit_transform(X_GAPS))
    np.testing.assert_allclose(once, filled, rtol=1e-5, atol=1e-6)


def test_iterative_imputer_refusals():
    with pytest.raises(NotFittedError):
        ml.IterativeImputer().transform(X_GAPS)
    fitted = ml.IterativeImputer(random_state=0).fit(X_GAPS)
    with pytest.raises(ValueError, match="features"):
        fitted.transform(X_GAPS[:, :2])
    with pytest.raises(ValueError):
        ml.IterativeImputer().fit(X_GAPS[:, 0])
    with pytest.raises(ValueError, match="strategy"):
        ml.IterativeImputer(initial_strategy="nope").fit(X_GAPS)
    with pytest.raises(ValueError, match="strategy"):
        ml.IterativeImputer(initial_strategy="nope").fit_transform(X_GAPS)


# ---------------------------------------------------------------- KBinsDiscretizer
def test_kbins_fit_transform_inverse_match_reference():
    kw = dict(n_bins=3, encode="ordinal", strategy="uniform")
    ours = ml.KBinsDiscretizer(**kw).fit(X)
    theirs = reference_prep.KBinsDiscretizer(**kw).fit(X)
    for a, b in zip(ours.bin_edges_, theirs.bin_edges_):
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-5, atol=1e-6)
    codes = np.asarray(ours.transform(X))
    np.testing.assert_array_equal(codes, theirs.transform(X))
    np.testing.assert_array_equal(np.asarray(ml.KBinsDiscretizer(**kw).fit_transform(X)), codes)
    np.testing.assert_allclose(np.asarray(ours.inverse_transform(codes)), theirs.inverse_transform(codes),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kw", [dict(n_bins=1), dict(strategy="nope"), dict(encode="nope")],
                         ids=["one-bin", "strategy", "encode"])
def test_kbins_fit_and_fit_transform_refuse_bad_options(kw):
    with pytest.raises(ValueError):
        ml.KBinsDiscretizer(**kw).fit(X)
    with pytest.raises(ValueError):
        ml.KBinsDiscretizer(**kw).fit_transform(X)


def test_kbins_transform_and_inverse_refusals():
    with pytest.raises(ValueError, match="NaN"):
        ml.KBinsDiscretizer().fit(np.where(X > 1, np.nan, X))
    with pytest.raises(NotFittedError):
        ml.KBinsDiscretizer().transform(X)
    with pytest.raises(NotFittedError):
        ml.KBinsDiscretizer().inverse_transform(X)
    fitted = ml.KBinsDiscretizer(n_bins=3, encode="ordinal").fit(X)
    with pytest.raises(ValueError, match="features"):
        fitted.transform(X[:, :2])
    with pytest.raises(ValueError, match="features"):
        fitted.inverse_transform(np.asarray(fitted.transform(X))[:, :2])
    with pytest.raises(ValueError, match="no bin"):
        fitted.inverse_transform(np.full_like(X, 7))


# ---------------------------------------------------------------- scalers
def test_minmax_inverse_transform_round_trips():
    scaler = pp.MinMaxScaler().fit(X)
    np.testing.assert_allclose(np.asarray(scaler.inverse_transform(scaler.transform(X))), X,
                               rtol=1e-5, atol=1e-6)


def test_minmax_inverse_transform_refusals():
    with pytest.raises(NotFittedError):
        pp.MinMaxScaler().inverse_transform(X)
    fitted = pp.MinMaxScaler().fit(X)
    with pytest.raises(ValueError, match="feature count"):
        fitted.inverse_transform(X[:, :2])
    with pytest.raises(TypeError, match="float32"):
        fitted.inverse_transform(X.astype(np.float64))


@pytest.mark.parametrize("scaler", [pp.MinMaxScaler, pp.StandardScaler], ids=["MinMaxScaler", "StandardScaler"])
def test_scaler_load_round_trips(scaler, tmp_path):
    path = tmp_path / "scaler.npz"
    fitted = scaler().fit(X)
    fitted.save(path)
    loaded = scaler.load(path)
    np.testing.assert_array_equal(np.asarray(loaded.transform(X)), np.asarray(fitted.transform(X)))


@pytest.mark.parametrize("scaler,other", [(pp.MinMaxScaler, pp.StandardScaler),
                                          (pp.StandardScaler, pp.MinMaxScaler)],
                         ids=["MinMaxScaler", "StandardScaler"])
def test_scaler_load_refusals(scaler, other, tmp_path):
    with pytest.raises(FileNotFoundError):
        scaler.load(tmp_path / "missing.npz")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not an npz")
    with pytest.raises(ValueError):
        scaler.load(junk)
    wrong_fields = tmp_path / "fields.npz"
    np.savez(wrong_fields, foo=np.zeros(3))
    with pytest.raises(ValueError, match="field"):
        scaler.load(wrong_fields)
    other_kind = tmp_path / "other.npz"
    other().fit(X).save(other_kind)
    with pytest.raises(ValueError):
        scaler.load(other_kind)
