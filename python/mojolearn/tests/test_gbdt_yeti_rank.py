"""YetiRank, the sampled-permutation ranking GradientBoosting loss (lane/gbdt-learning-to-rank, stage 4).

What stays here is the public door: the default `l2_leaf_reg` (None) the door
resolves to 0 for YetiRank only, and the refusals by name. Learning on uneven
queries and the seed contract are the native check `gbdt_public_fits`
(`yeti_learns`, `yeti_seed`). Bit identity across the Metal and CPU columns is the gbdt-yeti-rank lane
of tools/identity_break.py, not this module. Fits run inside the private
reference context so a CPU-only install reaches the host binding.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting


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


def _model(**kw):
    params = dict(n_estimators=10, max_depth=4, loss="YetiRank")
    # pinned to the configuration these tests were written against: the
    # SymmetricTree defaults became CatBoost's GPU ones (lane/catboost-parity,
    # 2026-09-19); stochastic defaults are covered separately.
    params.update(bootstrap_type="No", random_strength=0.0)
    params.update(kw)
    return GradientBoosting(**params)


def test_l2_default_is_zero_for_yeti_rank_only():
    X, rel, g = _fixture()
    default = np.asarray(_model().fit(X, rel, group_id=g).predict(X))
    explicit0 = np.asarray(_model(l2_leaf_reg=0.0).fit(X, rel, group_id=g).predict(X))
    three = np.asarray(_model(l2_leaf_reg=3.0).fit(X, rel, group_id=g).predict(X))
    assert default.tobytes() == explicit0.tobytes()
    assert default.tobytes() != three.tobytes()
    assert GradientBoosting().l2_leaf_reg is None


def test_refusals_by_name():
    X, rel, g = _fixture()
    with pytest.raises(Exception, match="changing the leaf_estimation_method parameter is prohibited"):
        _model(leaf_estimation_method="Gradient").fit(X, rel, group_id=g)
    with pytest.raises(Exception, match="read by loss='PairLogit' only"):
        _model().fit(X, rel, group_id=g, pairs=[(0, 1)])
    big = np.zeros(1100, dtype=np.int64)
    Xb = np.random.default_rng(3).standard_normal((1100, 4)).astype(np.float32)
    rb = (np.arange(1100) % 3).astype(np.float32)
    with pytest.raises(Exception, match="max query size supported on GPU is 1023, got 1100"):
        _model(n_estimators=1).fit(Xb, rb, group_id=big)
