# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One native call per public method (release 0.8.57): the sequence and MLP
classifiers' `score`, PLS / CCA `score`, `FactorAnalysis.fit_transform`,
`lu_factor` and `lu_solve`.

Each migrated method equals, bit for bit, the composition of public steps it
replaced (the same kernels in the same order): accuracy of `predict` through
`accuracy_fraction`, R^2 of `predict` through `r2_score`, `fit` then
`transform`. `lu_factor` / `lu_solve` keep their contract: the factor
reconstructs P A, the solve answers A x = b (and A^T x = b), repeat calls
return the same bits, and the refusals keep their words.
"""
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._buffer import as_f32_c
from mojolearn._expansion_metrics import accuracy_fraction
from mojolearn._metrics_impl import r2_score


def _bits(value):
    return np.asarray(value).tobytes()


def _seq_data(n=32, T=4, D=3, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D)).astype(np.float32)
    y = ((X[:, -1, 0] + 0.5 * X[:, -2, 1]) > 0).astype(np.int64)
    return X, y


def _cls_data(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int64)
    X = (rng.standard_normal((n, d)) + y[:, None] * 2.0).astype(np.float32)
    return X, y


def _pls_data(n=40, d=5, t=2, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (X[:, :t] * 1.5 + 0.1 * rng.standard_normal((n, t))).astype(np.float32)
    return X, Y


SEQ_CLASSIFIERS = {
    "LSTMClassifier": lambda: ml.LSTMClassifier(hidden_size=4, batch_size=16, max_epochs=2),
    "GRUClassifier": lambda: ml.GRUClassifier(hidden_size=4, batch_size=16, max_epochs=2),
    "RNNClassifier": lambda: ml.RNNClassifier(hidden_size=4, batch_size=16, max_epochs=2),
}


@pytest.mark.parametrize("name", sorted(SEQ_CLASSIFIERS))
def test_sequence_classifier_score_is_accuracy_of_predict(name):
    X, y = _seq_data()
    model = SEQ_CLASSIFIERS[name]().fit(X, y)
    w = np.linspace(0.5, 1.5, len(y)).astype(np.float32)
    for sample_weight in (None, w):
        expected = accuracy_fraction(y, model.predict(X), sample_weight, model.numeric_mode)
        assert _bits(model.score(X, y, sample_weight)) == _bits(expected)
    with pytest.raises(ValueError):
        model.score(X, y[:-1])


def test_mlp_classifier_score_is_accuracy_of_predict():
    X, y = _cls_data()
    model = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y)
    w = np.linspace(0.5, 1.5, len(y)).astype(np.float32)
    for sample_weight in (None, w):
        expected = accuracy_fraction(y, model.predict(X), sample_weight, model.numeric_mode)
        assert _bits(model.score(X, y, sample_weight)) == _bits(expected)
    with pytest.raises(ValueError):
        model.score(X, y[:-1])


@pytest.mark.parametrize("cls", [ml.PLSRegression, ml.PLSCanonical, ml.CCA])
@pytest.mark.parametrize("targets", [1, 2])
def test_pls_score_is_r2_of_predict(cls, targets):
    X, Y = _pls_data()
    Y = Y[:, 0] if targets == 1 else Y
    if cls is not ml.PLSRegression and targets == 1:
        pytest.skip("PLSCanonical and CCA take a 2-D Y")
    model = cls(n_components=1).fit(X, Y)
    w = np.linspace(0.5, 1.5, len(X)).astype(np.float32)
    # a float64 y is cast to float32 first, as RegressorScoreMixin did
    for y, sample_weight in ((Y, None), (Y.astype(np.float64), None), (Y, w)):
        truth, _ = as_f32_c(y, ndim=None, name="y")
        pred, _ = as_f32_c(model.predict(X), ndim=None, name="prediction")
        expected = r2_score(truth, pred, sample_weight=sample_weight, numeric_mode=model.numeric_mode)
        assert _bits(model.score(X, y, sample_weight)) == _bits(expected)
    with pytest.raises(ValueError):
        model.score(X, Y[:-1])


def test_factor_analysis_fit_transform_is_fit_then_transform():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 6)).astype(np.float32)
    joint = ml.FactorAnalysis(n_components=2, random_state=0)
    Z = joint.fit_transform(X)
    split = ml.FactorAnalysis(n_components=2, random_state=0).fit(X)
    assert _bits(Z) == _bits(split.transform(X))
    assert _bits(joint.components_) == _bits(split.components_)
    assert _bits(joint.noise_variance_) == _bits(split.noise_variance_)


def _system(n=8, seed=5):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) + n * np.eye(n)).astype(np.float32)
    return A, rng.random(n).astype(np.float32), rng.random((n, 3)).astype(np.float32)


def _unpack(lu, piv):
    lu, piv = np.asarray(lu, np.float64), np.asarray(piv)
    n = lu.shape[0]
    L, U = np.tril(lu, -1) + np.eye(n), np.triu(lu)
    perm = np.arange(n)
    for i, p in enumerate(piv):
        perm[[i, p]] = perm[[p, i]]
    return L, U, perm


def test_lu_factor_reconstructs_and_repeats():
    A, _, _ = _system()
    pair = ml.lu_factor(A)
    lu, piv = pair
    assert np.asarray(lu).dtype == np.float32 and np.asarray(piv).dtype == np.int32
    L, U, perm = _unpack(lu, piv)
    np.testing.assert_allclose(L @ U, A.astype(np.float64)[perm], rtol=1e-5, atol=1e-5)
    lu2, piv2 = ml.lu_factor(A)
    assert _bits(lu2) == _bits(lu) and _bits(piv2) == _bits(piv)


@pytest.mark.parametrize("trans", [0, 1, 2])
def test_lu_solve_answers_the_system(trans):
    A, b, B = _system()
    pair = ml.lu_factor(A)
    M = A.astype(np.float64) if trans == 0 else A.astype(np.float64).T
    for rhs in (b, B):
        x = ml.lu_solve(pair, rhs, trans=trans)
        assert np.asarray(x).shape == rhs.shape
        np.testing.assert_allclose(M @ np.asarray(x, np.float64), rhs, rtol=1e-4, atol=1e-4)
        assert _bits(ml.lu_solve(pair, rhs, trans=trans)) == _bits(x)


def test_lu_solve_plain_tuple_pair():
    # a pair without the factor's own A or device buffers takes the host route
    A, b, _ = _system()
    lu, piv = ml.lu_factor(A)
    x = ml.lu_solve((np.asarray(lu), np.asarray(piv).astype(np.int64)), b)
    np.testing.assert_allclose(A.astype(np.float64) @ np.asarray(x, np.float64), b, rtol=1e-4, atol=1e-4)


def test_lu_refusals_keep_their_words():
    A, b, _ = _system()
    pair = ml.lu_factor(A)
    with pytest.raises(ValueError, match="trans must be 0, 1 or 2"):
        ml.lu_solve(pair, b, trans=3)
    with pytest.raises(ValueError, match="expected a square matrix, got 8 x 7"):
        ml.lu_factor(A[:, :7])
    with pytest.raises(ValueError, match="b has 7 rows, the factorization has 8"):
        ml.lu_solve(pair, b[:7])
    lu, piv = pair
    bad = np.asarray(piv).copy()
    bad[0] = 8
    with pytest.raises(ValueError, match=r"piv must hold one row index in \[0, n\) per row"):
        ml.lu_solve((np.asarray(lu), bad), b)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        ml.lu_solve((np.asarray(lu),), b)
    with pytest.raises(TypeError, match="cannot unpack non-iterable"):
        ml.lu_solve(3, b)


def test_lu_factor_singular_warns():
    A = np.ones((3, 3), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lu, piv = ml.lu_factor(A)
    assert any(issubclass(w.category, RuntimeWarning) and "exactly zero. Singular matrix." in str(w.message)
               for w in caught)
    assert any(w.filename == __file__ for w in caught)
    assert np.asarray(lu).shape == (3, 3) and np.asarray(piv).shape == (3,)
