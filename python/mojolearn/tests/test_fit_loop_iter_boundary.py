# SPDX-License-Identifier: Apache-2.0
"""`fit_loop_` at the public boundary of the fixed-cost iterative fits: NMF,
FastICA, FactorAnalysis, AlternatingLeastSquares (decomposition),
LabelPropagation, LabelSpreading (neighbors) and TSNE (manifold). After fit it
is a tuple (int, float): the executed iterations (n_iter_, or the sweeps for
ALS) and a non-negative loop time in milliseconds. Before fit, reading it
fails like the estimator's other fitted attributes."""
import numpy as np
import pytest
import mojolearn as ml


def _rows(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d), dtype=np.float32)


def _semi():
    X = _rows(60, 4)
    y = (X[:, 0] > 0.5).astype(np.int64)
    y[10:] = -1
    return X, y


# (factory, fit arguments, the capped count or None, the fitted attribute the count equals, a reference attribute)
_CASES = {
    "nmf": (lambda: ml.NMF(n_components=2, solver="mu", max_iter=4, tol=0.0, random_state=0),
            lambda: (_rows(30, 5),), 4, "n_iter_", "components_"),
    "fastica": (lambda: ml.FastICA(n_components=2, max_iter=4, tol=0.0, random_state=0),
                lambda: (_rows(40, 3),), 4, "n_iter_", "components_"),
    # EM can stop when the log-likelihood gain falls below tol even at 0; the count is n_iter_'s
    "factor-analysis": (lambda: ml.FactorAnalysis(n_components=1, max_iter=4, tol=0.0),
                        lambda: (_rows(40, 4),), None, "n_iter_", "components_"),
    "als": (lambda: ml.AlternatingLeastSquares(factors=2, iterations=3, random_state=0),
            lambda: ((_rows(20, 10) < 0.3).astype(np.float32),), 3, None, "user_factors_"),
    "label-propagation": (lambda: ml.LabelPropagation(kernel="knn", n_neighbors=3, max_iter=4, tol=0.0),
                          _semi, 4, "n_iter_", "transduction_"),
    "label-spreading": (lambda: ml.LabelSpreading(kernel="knn", n_neighbors=3, max_iter=4, tol=0.0),
                        _semi, 4, "n_iter_", "transduction_"),
    "tsne": (lambda: ml.TSNE(perplexity=5.0, max_iter=250, random_state=0),
             lambda: (_rows(30, 3),), 250, "n_iter_", "embedding_"),
}


@pytest.mark.parametrize("lane", sorted(_CASES))
def test_fit_loop_after_fit(lane):
    factory, args, cap, count_attr, _ = _CASES[lane]
    model = factory()
    assert model.fit(*args()) is model
    loop = model.fit_loop_
    assert type(loop) is tuple and len(loop) == 2
    iterations, milliseconds = loop
    assert type(iterations) is int and type(milliseconds) is float
    assert milliseconds >= 0.0
    if count_attr is not None:
        assert iterations == getattr(model, count_attr)
    if cap is not None:
        assert iterations == cap


@pytest.mark.parametrize("lane", sorted(_CASES))
def test_fit_loop_before_fit_fails_like_other_fitted_attributes(lane):
    factory, _, _, _, reference = _CASES[lane]
    model = factory()
    with pytest.raises(Exception) as expected:
        getattr(model, reference)
    with pytest.raises(type(expected.value)):
        model.fit_loop_


def test_fit_loop_is_the_last_fit():
    X = _rows(30, 5)
    model = ml.NMF(n_components=2, solver="mu", max_iter=4, tol=0.0, random_state=0).fit(X)
    assert model.fit_loop_[0] == 4
    model.set_params(max_iter=6).fit(X)
    assert model.fit_loop_[0] == 6 == model.n_iter_
