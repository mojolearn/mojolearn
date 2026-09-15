# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QuerySoftMax, the softmax ranking GradientBoosting loss, and the ranking
losses' CatBoost loss parameters (lane/gbdt-rest).

What the fit must do independent of any vendor: learn on uneven queries at the
reference's defaults (lambda 0.01, beta 1.0, Gradient leaves at 100
iterations), read `loss_lambda`, `loss_beta`, `loss_permutations` and
`loss_decay` where their loss reads them and refuse them everywhere else, and
refuse a nonpositive total weighted target in the reference's words. Bit
identity across the Metal and CPU columns is the gbdt-query-softmax lane of
tools/identity_break.py, not this module. Fits run inside the private
reference context so a CPU-only install reaches the host binding.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training


def _fixture(n=192, d=4, seed=9):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    rel = np.digitize(score, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    sizes = [1, 2, 7, 3, 16, 1, 5, 40, 2, 9]
    ids = []
    q = 0
    while len(ids) < n:
        ids.extend([q] * sizes[q % len(sizes)])
        q += 1
    return X, rel, np.asarray(ids[:n], dtype=np.int64)


def _ndcg(pred, rel, groups):
    """Mean per-query NDCG, all positions (ties: lower grade first)."""
    out = []
    for q in np.unique(groups):
        idx = np.flatnonzero(groups == q)
        p, t = pred[idx].astype(np.float64), rel[idx].astype(np.float64)
        decay = 1.0 / np.log2(np.arange(len(idx)) + 2.0)
        order = sorted(range(len(idx)), key=lambda i: (-p[i], t[i]))
        ideal = float(np.dot(np.sort(t)[::-1], decay))
        out.append(float(np.dot(t[order], decay)) / ideal if ideal > 0 else 1.0)
    return float(np.mean(out))


def _model(**kw):
    params = dict(n_estimators=4, max_depth=3, loss="QuerySoftMax")
    params.update(kw)
    return GradientBoosting(**params)


@reference_training()
def test_learns_on_uneven_queries():
    X, rel, g = _fixture()
    m = _model().fit(X, rel, group_id=g)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))
    assert np.any(pred != 0)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert _ndcg(pred, rel, g) > _ndcg(np.zeros_like(pred), rel, g)


@reference_training()
def test_leaf_default_is_gradient_at_a_hundred_iterations():
    X, rel, g = _fixture()
    default = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    gradient = np.asarray(
        _model(leaf_estimation_method="Gradient").fit(X, rel, group_id=g).predict(X))
    newton = np.asarray(
        _model(leaf_estimation_method="Newton").fit(X, rel, group_id=g).predict(X))
    assert default.tobytes() == gradient.tobytes()
    assert default.tobytes() != newton.tobytes()


@reference_training()
def test_loss_parameters_are_read_and_their_defaults_are_the_references():
    X, rel, g = _fixture()
    default = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    spelled = np.asarray(
        _model(loss_lambda=0.01, loss_beta=1.0).fit(X, rel, group_id=g).predict(X))
    other_beta = np.asarray(
        _model(loss_beta=2.0).fit(X, rel, group_id=g).predict(X))
    assert default.tobytes() == spelled.tobytes()
    assert default.tobytes() != other_beta.tobytes()


@reference_training()
def test_lambda_is_read_where_the_second_derivative_is_read():
    """MEASURED, not assumed: `lambda` enters only `der2`
    (`query_softmax.cu:182`), and QuerySoftMax's default leaf estimator is
    Gradient, whose Hessian is the leaf's weight sum, under a Cosine score
    whose weight plane is the row weight. So at the defaults `lambda` moves
    nothing, and under Newton leaves, which read `der2`, it does."""
    X, rel, g = _fixture()
    default = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    inert = np.asarray(_model(loss_lambda=0.5).fit(X, rel, group_id=g).predict(X))
    assert default.tobytes() == inert.tobytes()
    newton = np.asarray(
        _model(leaf_estimation_method="Newton").fit(X, rel, group_id=g).predict(X))
    newton_lambda = np.asarray(
        _model(leaf_estimation_method="Newton", loss_lambda=0.5)
        .fit(X, rel, group_id=g).predict(X))
    assert newton.tobytes() != newton_lambda.tobytes()


@reference_training()
def test_yeti_rank_loss_parameters_are_read():
    X, rel, g = _fixture()
    def yeti(**kw):
        return np.asarray(GradientBoosting(
            n_estimators=3, max_depth=3, loss="YetiRank", **kw
        ).fit(X, rel, group_id=g).predict(X))
    default = yeti()
    assert default.tobytes() == yeti(loss_permutations=10, loss_decay=0.85).tobytes()
    assert default.tobytes() != yeti(loss_permutations=3).tobytes()
    assert default.tobytes() != yeti(loss_decay=0.5).tobytes()


@reference_training()
def test_loss_parameters_are_refused_where_their_loss_does_not_read_them():
    X, rel, g = _fixture()
    with pytest.raises(ValueError, match="YetiRank loss parameter"):
        _model(loss_permutations=5)
    with pytest.raises(ValueError, match="QuerySoftMax loss parameter"):
        GradientBoosting(loss="YetiRank", loss_lambda=0.5)
    with pytest.raises(ValueError, match="QuerySoftMax loss parameter"):
        GradientBoosting(loss="RMSE", loss_beta=2.0)
    with pytest.raises(ValueError, match="loss_permutations must be a positive integer"):
        GradientBoosting(loss="YetiRank", loss_permutations=0)
    with pytest.raises(ValueError, match="loss_decay must be a finite number"):
        GradientBoosting(loss="YetiRank", loss_decay=float("inf"))


@reference_training()
def test_nonpositive_total_weighted_target_is_refused_in_their_words():
    X, rel, g = _fixture()
    zeros = np.zeros_like(rel)
    with pytest.raises(Exception, match="Total weighted target should be greater, than zero"):
        _model().fit(X, zeros, group_id=g)


@reference_training()
def test_refusals_by_name():
    X, rel, g = _fixture()
    with pytest.raises(Exception, match="with a bootstrap is not implemented"):
        _model(bootstrap_type="Bernoulli", subsample=0.5).fit(X, rel, group_id=g)
    with pytest.raises(Exception, match="QuerySoftMax"):
        _model(grow_policy="Depthwise").fit(X, rel, group_id=g)
    with pytest.raises(Exception, match="read by loss='PairLogit' only"):
        _model().fit(X, rel, group_id=g, pairs=[(0, 1)])


@reference_training()
def test_two_fits_of_one_configuration_agree_bit_for_bit():
    X, rel, g = _fixture()
    a = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    b = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    assert a.tobytes() == b.tobytes()


@reference_training()
def test_queries_of_one_row_train_nothing():
    """A query of one row has softmax 1, so its derivative is
    `beta * (-t * 1 + t) == 0`: the reference's grouping rule
    (`TWithoutQueriesGrouping`) leaves the model at zero, as for QueryRMSE."""
    X, rel, _ = _fixture()
    no_groups = np.asarray(_model().fit(X, rel).predict(X))
    one_each = np.asarray(
        _model().fit(X, rel, group_id=np.arange(X.shape[0])).predict(X))
    assert np.all(no_groups == 0.0)
    assert no_groups.tobytes() == one_each.tobytes()
