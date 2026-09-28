# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GaussianProcessRegressor.predict(return_cov=True) (lane x-neighbors-gp-cov,
2026-09-27). Skips, saying so, without a GP binding."""

import numpy as np
import pytest

import mojolearn as ml


def _fit_or_skip(normalize_y=False):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.1 * rng.normal(size=60) + (5.0 if normalize_y else 0.0)).astype(np.float32)
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    try:
        m = ml.GaussianProcessRegressor(kernel=k, normalize_y=normalize_y).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no GP binding on this install: {exc}")
    return m, x, y, k


def test_both_flags_are_refused():
    m, x, _, _ = _fit_or_skip()
    with pytest.raises(RuntimeError, match="at most one"):
        m.predict(x, return_std=True, return_cov=True)


@pytest.mark.parametrize("normalize_y", [False, True])
def test_cov_is_symmetric_and_its_diagonal_is_the_variance(normalize_y):
    m, x, _, _ = _fit_or_skip(normalize_y)
    q = x[:20] + np.float32(0.25)
    mean, cov = m.predict(q, return_cov=True)
    mean, cov = np.asarray(mean), np.asarray(cov)
    assert cov.shape == (20, 20) and cov.dtype == np.float32
    assert (cov == cov.T).all()
    assert mean.tobytes() == np.asarray(m.predict(q)).tobytes()
    _, std = m.predict(q, return_std=True)
    np.testing.assert_allclose(np.sqrt(np.clip(np.diag(cov), 0, None)), np.asarray(std), rtol=1e-3, atol=1e-4)
    again = np.asarray(m.predict(q, return_cov=True)[1])
    assert again.tobytes() == cov.tobytes()


def test_matches_scikit_learn():
    gp = pytest.importorskip("sklearn.gaussian_process")
    kr = pytest.importorskip("sklearn.gaussian_process.kernels")
    m, x, y, _ = _fit_or_skip()
    k = kr.ConstantKernel(1.0) * kr.RBF(1.0) + kr.WhiteKernel(0.1)
    r = gp.GaussianProcessRegressor(kernel=k, optimizer=None).fit(x.astype(np.float64), y)
    q = x[:15] + np.float32(0.5)
    _, cov = m.predict(q, return_cov=True)
    _, rcov = r.predict(q.astype(np.float64), return_cov=True)
    np.testing.assert_allclose(np.asarray(cov), rcov, atol=2e-3)
