"""QueryRMSE, the querywise GradientBoosting loss (lane/gbdt-learning-to-rank, stage 2).

What the fit must do, independent of any vendor: learn on real queries, train
nothing on queries of one row (the CatBoost reference's `TWithoutQueriesGrouping`,
`gpu_data/doc_parallel_dataset.h:26-38`), read group ids by their spelling, and
refuse by name the arms this implementation does not carry. Bit identity across
the Metal and CPU columns is the gbdt-query-rmse lane of tools/identity_break.py,
not this module. Every fit runs inside the private reference context so a
CPU-only install reaches the host binding.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training


def _fixture(n=256, d=4, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # graded relevance with ties, driven by two columns
    score = X[:, 0] + 0.5 * X[:, 1]
    rel = np.digitize(score, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    sizes = [1, 2, 7, 3, 16, 1, 5, 40, 2, 9]
    ids = []
    q = 0
    while len(ids) < n:
        ids.extend([q] * sizes[q % len(sizes)])
        q += 1
    return X, rel, np.asarray(ids[:n], dtype=np.int64)


def _model(**kw):
    params = dict(n_estimators=10, max_depth=4, loss="QueryRMSE")
    # pinned to the configuration these tests were written against: the
    # SymmetricTree defaults became CatBoost's GPU ones (lane/catboost-parity,
    # 2026-09-19), whose query bootstrap is refused by name for this loss
    params.update(bootstrap_type="No", random_strength=0.0)
    params.update(kw)
    return GradientBoosting(**params)


@reference_training()
def test_learns_on_uneven_queries():
    X, rel, g = _fixture()
    m = _model().fit(X, rel, group_id=g)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert np.any(pred != 0)


@reference_training()
def test_queries_of_one_row_train_nothing():
    X, rel, _ = _fixture()
    no_groups = np.asarray(_model().fit(X, rel).predict(X))
    one_each = np.asarray(_model().fit(X, rel, group_id=np.arange(X.shape[0])).predict(X))
    assert np.all(no_groups == 0.0)
    assert no_groups.tobytes() == one_each.tobytes()


@reference_training()
def test_group_id_spelling_does_not_change_the_model():
    X, rel, g = _fixture()
    as_int = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    as_str = np.asarray(_model().fit(X, rel, group_id=[f"q{v}" for v in g]).predict(X))
    assert as_int.tobytes() == as_str.tobytes()


@reference_training()
def test_refusals_by_name():
    X, rel, g = _fixture()
    with pytest.raises(Exception, match="with a bootstrap is not implemented"):
        _model(bootstrap_type="Bernoulli", subsample=0.5).fit(X, rel, group_id=g)
    # the device refuses in the QueryRMSE sentence; the host binding refuses
    # eval_set for every loss first, in its own by-name sentence
    with pytest.raises(Exception, match="with eval_set is not implemented|no CPU implementation of .* for eval_set"):
        _model().fit(X, rel, group_id=g, eval_set=(X[:32], rel[:32]))
    with pytest.raises(Exception, match="QueryRMSE"):
        _model(grow_policy="Depthwise").fit(X, rel, group_id=g)
    coded = X.copy()
    coded[:, 0] = np.floor(np.abs(coded[:, 0]) * 3).astype(np.float32)
    with pytest.raises(Exception, match="cat_features|categorical"):
        _model(cat_features=[0]).fit(coded, rel, group_id=g)


@reference_training()
def test_split_query_still_refused():
    X, rel, g = _fixture()
    g = g.copy()
    g[-1] = 0
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        _model().fit(X, rel, group_id=g)
