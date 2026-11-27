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
    # 2026-09-19); stochastic defaults are covered separately.
    params.update(bootstrap_type="No", random_strength=0.0)
    params.update(kw)
    return GradientBoosting(**params)


def test_learns_on_uneven_queries():
    X, rel, g = _fixture()
    m = _model().fit(X, rel, group_id=g)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert np.any(pred != 0)


def test_group_id_spelling_does_not_change_the_model():
    X, rel, g = _fixture()
    as_int = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    as_str = np.asarray(_model().fit(X, rel, group_id=[f"q{v}" for v in g]).predict(X))
    assert as_int.tobytes() == as_str.tobytes()


def test_refusals_by_name():
    X, rel, g = _fixture()
    with pytest.raises(Exception, match="eval_group_id"):
        _model().fit(X, rel, group_id=g, eval_set=(X[:32], rel[:32]))
    with pytest.raises(Exception, match="QueryRMSE|SymmetricTree"):
        _model(grow_policy="Depthwise").fit(X, rel, group_id=g)


@pytest.mark.parametrize("as_list", [False, True])
def test_categorical_ranking_accepts_separate_validation_queries(as_list):
    X, rel, groups = _fixture(32)
    X[:, 0] = (X[:, 0] > 0).astype(np.float32)
    valid, target = X[-8:].copy(), rel[-8:]
    valid[0, 0] = 2  # a held-out category
    held = (valid, target, ["first"] * 3 + ["second"] * 5)
    ranker = _model(cat_features=[0], n_estimators=2)
    assert ranker.fit(X[:-8], rel[:-8], group_id=groups[:-8], eval_set=[held] if as_list else held) is ranker
    assert len(ranker.test_loss_curve_) == 2
    assert np.asarray(ranker.predict(valid)).shape == (8,)


def test_validation_refuses_an_iterable_without_invoking_it():
    class Callback:
        def __iter__(self):
            pytest.fail("eval_set called a user iterator")

    X, rel, groups = _fixture(16)
    with pytest.raises(Exception, match="eval_set"):
        _model().fit(X, rel, group_id=groups, eval_set=Callback())


@pytest.mark.parametrize("loss", ["lambdarank", "rank_xendcg"])
def test_lightgbm_objectives_require_query_ids_and_integer_grades(loss):
    X, rel, groups = _fixture(16)
    ranker = _model(loss=loss, n_estimators=1)
    with pytest.raises(Exception, match="require group_id"):
        ranker.fit(X, rel)
    with pytest.raises(Exception, match="eval_set"):
        ranker.fit(X, rel, group_id=groups, eval_set=(X, rel))
    cat_x = X.copy()
    cat_x[:, 0] = (cat_x[:, 0] > 0).astype(np.float32)
    model = _model(loss=loss, cat_features=[0], n_estimators=1)
    assert model.fit(cat_x, rel, group_id=groups, eval_set=(cat_x, rel, groups), eval_at=1) is model
    for invalid_cutoff in (0, -1, 1.5, True, 2**31):
        with pytest.raises(ValueError, match="eval_at"):
            model.fit(cat_x, rel, group_id=groups, eval_set=(cat_x, rel, groups), eval_at=invalid_cutoff)
    for bad_grade in (-1, 0.5, 31):
        invalid = rel.copy()
        invalid[0] = bad_grade
        with pytest.raises(Exception, match=r"integer relevance grades in \[0, 30\]"):
            ranker.fit(X, invalid, group_id=groups)
        with pytest.raises(Exception, match="integer relevance grades"):
            ranker.fit(X, rel, group_id=groups, eval_set=(X, invalid, groups))


def test_split_query_still_refused():
    X, rel, g = _fixture()
    g = g.copy()
    g[-1] = 0
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        _model().fit(X, rel, group_id=g)
