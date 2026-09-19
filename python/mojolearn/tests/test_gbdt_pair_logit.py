"""PairLogit, the pairwise ranking GradientBoosting loss (lane/gbdt-learning-to-rank, stage 3).

What the fit must do independent of any vendor: learn on uneven queries, generate
the same pairs the CatBoost reference's default path does (every two rows of a
query with different grades, the higher grade the winner), accept explicit
`pairs` with `pairs_weight`, and refuse by name what this implementation does not
carry. Bit identity across the Metal and CPU columns is the gbdt-pair-logit lane
of tools/identity_break.py, not this module. Fits run inside the private
reference context so a CPU-only install reaches the host binding.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training
from mojolearn.ensemble import _pairs_arrays


def _fixture(n=240, d=4, seed=11):
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


def _generated_pairs(rel, group):
    """The reference's GenerateBruteForce, restated in Python for the test."""
    pairs = []
    begin = 0
    for q in np.unique(group):
        size = int(np.count_nonzero(group == q))
        for a in range(begin, begin + size):
            for b in range(a + 1, begin + size):
                if rel[a] == rel[b]:
                    continue
                pairs.append((a, b) if rel[a] > rel[b] else (b, a))
        begin += size
    return pairs


def _model(**kw):
    params = dict(n_estimators=8, max_depth=4, loss="PairLogit")
    # pinned to the configuration these tests were written against: the
    # SymmetricTree defaults became CatBoost's GPU ones (lane/catboost-parity,
    # 2026-09-19), whose query bootstrap is refused by name for this loss
    params.update(bootstrap_type="No", random_strength=0.0, leaf_estimation_iterations=10)
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
    # a pair's loss starts at log 2 per unit pair weight
    assert abs(m.loss_curve_[0] - np.log(2.0)) < 0.1


@reference_training()
def test_explicit_unit_pairs_equal_the_generated_pairs():
    X, rel, g = _fixture()
    generated = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    explicit = np.asarray(_model().fit(X, rel, group_id=g, pairs=_generated_pairs(rel, g)).predict(X))
    assert generated.tobytes() == explicit.tobytes()


@reference_training()
def test_pairs_weight_changes_the_model():
    X, rel, g = _fixture()
    pairs = _generated_pairs(rel, g)
    unit = np.asarray(_model().fit(X, rel, group_id=g, pairs=pairs).predict(X))
    weights = np.linspace(0.5, 2.0, len(pairs)).astype(np.float32)
    weighted = np.asarray(_model().fit(X, rel, group_id=g, pairs=pairs, pairs_weight=weights).predict(X))
    assert unit.tobytes() != weighted.tobytes()


@reference_training()
def test_refusals_by_name():
    X, rel, g = _fixture()
    pairs = _generated_pairs(rel, g)
    with pytest.raises(NotImplementedError, match="pairs without group_id"):
        _model().fit(X, rel, pairs=pairs)
    with pytest.raises(NotImplementedError, match="read by loss='PairLogit' only"):
        GradientBoosting(n_estimators=2, loss="QueryRMSE", bootstrap_type="No").fit(X, rel, group_id=g, pairs=pairs)
    with pytest.raises(Exception, match="Cannot generate pairs for data without groups"):
        _model().fit(X, rel)
    with pytest.raises(Exception, match="Target data is constant"):
        _model().fit(X, np.ones_like(rel), group_id=g)
    with pytest.raises(Exception, match="must belong to the same group"):
        _model().fit(X, rel, group_id=g, pairs=[(0, X.shape[0] - 1)])
    with pytest.raises(Exception, match="with a bootstrap is not implemented"):
        _model(bootstrap_type="Bernoulli", subsample=0.5).fit(X, rel, group_id=g)


def test_pairs_validation():
    assert _pairs_arrays([(3, 1), [0, 2]], None, 4) == ([3, 1, 0, 2], [1.0, 1.0])
    assert _pairs_arrays(np.array([[1, 0]]), [2.5], 2) == ([1, 0], [2.5])
    with pytest.raises(ValueError, match="isn't equal to 2"):
        _pairs_arrays([(1, 2, 3)], None, 4)
    with pytest.raises(ValueError, match="must be an integer"):
        _pairs_arrays([(1.0, 2)], None, 4)
    with pytest.raises(ValueError, match="outside the 4 rows"):
        _pairs_arrays([(1, 4)], None, 4)
    with pytest.raises(ValueError, match="with itself"):
        _pairs_arrays([(2, 2)], None, 4)
    with pytest.raises(ValueError, match="is not equal to len\\(pairs\\)"):
        _pairs_arrays([(0, 1)], [1.0, 2.0], 4)
    with pytest.raises(ValueError, match="finite nonnegative"):
        _pairs_arrays([(0, 1)], [-1.0], 4)
    with pytest.raises(ValueError, match="pairs is empty"):
        _pairs_arrays([], None, 4)
