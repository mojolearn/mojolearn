# SPDX-License-Identifier: Apache-2.0
"""mojolearn.Harmony at the Python API boundary (lane sc-stats-1, 2026-11-28): parameters, refusals, the attributes
fit publishes and their invariants, several batch covariates, and that the correction removes a planted batch
shift while keeping the biological groups apart. The kernels' exact checks are native
(checks/native_tests/x_cluster_harmony_check.mojo)."""
import numpy as np
import pytest

import mojolearn


def _embedding(n=600, d=8, seed=0):
    """Three cell types, two batches; batch 1 shifted along every dimension."""
    rng = np.random.default_rng(seed)
    cell = np.arange(n) % 3
    batch = (np.arange(n) // 3) % 2
    centers = rng.normal(scale=3.0, size=(3, d))
    z = centers[cell] + rng.normal(scale=0.4, size=(n, d)) + 1.5 * batch[:, None]
    return z.astype(np.float32), batch, cell


def test_defaults_and_params():
    m = mojolearn.Harmony()
    p = m.get_params()
    assert p["n_clusters"] is None and p["theta"] == 2.0 and p["sigma"] == 0.1 and p["lamb"] == 1.0
    assert p["max_iter_harmony"] == 10 and p["max_iter_kmeans"] == 20 and p["block_size"] == 0.05
    assert p["random_state"] == 0


def test_fit_attributes():
    z, batch, _ = _embedding()
    m = mojolearn.Harmony(random_state=0).fit(z, batch)
    emb = np.asarray(m.embedding_)
    r = np.asarray(m.R_)
    assert emb.shape == z.shape and emb.dtype == np.float32
    assert m.n_clusters_ == 20  # round(600 / 30)
    assert r.shape == (600, 20)
    np.testing.assert_allclose(r.sum(axis=1), 1.0, atol=1e-4)
    assert np.asarray(m.cluster_centers_).shape == (20, 8)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(m.cluster_centers_), axis=1), 1.0, atol=1e-4)
    assert 1 <= m.n_iter_ <= 10
    assert len(m.objective_) == m.n_iter_ + 1
    assert m.n_features_in_ == 8
    out = mojolearn.Harmony(random_state=0).fit_transform(z, batch)
    assert np.asarray(out).tobytes() == emb.tobytes()


def test_correction_removes_the_batch_shift():
    z, batch, cell = _embedding()
    emb = np.asarray(mojolearn.Harmony(random_state=0).fit_transform(z, batch))

    def shift(x):
        return np.mean([np.linalg.norm(x[(cell == c) & (batch == 1)].mean(0) - x[(cell == c) & (batch == 0)].mean(0))
                        for c in range(3)])

    def spread(x):
        m = [x[cell == c].mean(0) for c in range(3)]
        return min(np.linalg.norm(m[a] - m[b]) for a in range(3) for b in range(a + 1, 3))

    assert shift(emb) < 0.5 * shift(z)
    assert spread(emb) > 0.5 * spread(z)


def test_several_covariates_and_per_covariate_parameters():
    z, batch, _ = _embedding()
    tech = np.arange(len(batch)) % 4
    m = mojolearn.Harmony(n_clusters=10, theta=[2.0, 1.0], lamb=[1.0, 0.5], tau=5.0, max_iter_harmony=2,
                          random_state=3).fit(z, [batch, tech])
    assert np.asarray(m.embedding_).shape == z.shape
    assert np.asarray(m.R_).shape == (len(batch), 10)
    names = np.array(["s1", "s2"])[batch]
    s = mojolearn.Harmony(n_clusters=10, max_iter_harmony=2).fit(z, (names,))
    assert np.asarray(s.embedding_).shape == z.shape


def test_same_seed_same_result():
    z, batch, _ = _embedding()
    a = np.asarray(mojolearn.Harmony(random_state=4).fit_transform(z, batch))
    b = np.asarray(mojolearn.Harmony(random_state=4).fit_transform(z, batch))
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("kwargs", [dict(sigma=0.0), dict(lamb=0.0), dict(theta=-1.0), dict(block_size=0.0),
                                    dict(block_size=1.5), dict(max_iter_harmony=0), dict(n_clusters=0),
                                    dict(n_clusters=10_000), dict(theta=[1.0, 2.0]), dict(tau=-1.0)])
def test_refusals(kwargs):
    z, batch, _ = _embedding()
    with pytest.raises(ValueError):
        mojolearn.Harmony(**kwargs).fit(z, batch)


def test_input_refusals():
    z, batch, _ = _embedding()
    with pytest.raises(ValueError):
        mojolearn.Harmony().fit(z, batch[:-1])
    bad = z.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        mojolearn.Harmony().fit(bad, batch)
