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
    # 2026-09-19); stochastic defaults are covered separately.
    params.update(bootstrap_type="No", random_strength=0.0, leaf_estimation_iterations=10)
    params.update(kw)
    return GradientBoosting(**params)


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


@reference_training()
def test_pairs_validation():
    X, rel, g = _fixture()
    n = X.shape[0]
    pairs = _generated_pairs(rel, g)
    # a list of tuples, a list of lists and an int array are the same pairs;
    # no pairs_weight is unit weights
    base = np.asarray(_model(n_estimators=2).fit(X, rel, group_id=g, pairs=pairs).predict(X))
    for same in ([list(p) for p in pairs], np.asarray(pairs, dtype=np.int64)):
        got = np.asarray(_model(n_estimators=2).fit(X, rel, group_id=g, pairs=same).predict(X))
        assert got.tobytes() == base.tobytes()
    unit = np.ones(len(pairs), dtype=np.float32)
    got = np.asarray(_model(n_estimators=2).fit(X, rel, group_id=g, pairs=pairs, pairs_weight=unit).predict(X))
    assert got.tobytes() == base.tobytes()

    def refused(match, pairs, pairs_weight=None):
        with pytest.raises(ValueError, match=match):
            _model(n_estimators=2).fit(X, rel, group_id=g, pairs=pairs, pairs_weight=pairs_weight)

    refused("isn't equal to 2", [(1, 2, 3)])
    refused("must be an integer", [(1.0, 2)])
    refused(f"outside the {n} rows", [(1, n)])
    refused("with itself", [(2, 2)])
    refused("is not equal to len\\(pairs\\)", [(0, 1)], [1.0, 2.0])
    refused("finite nonnegative", [(0, 1)], [-1.0])
    refused("pairs is empty", [])
