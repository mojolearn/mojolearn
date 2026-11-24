# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GaussianProcessRegressor.predict(return_cov=True) (lane x-neighbors-gp-cov,
2026-09-27): the scikit-learn oracle. The refusal and the returned arrays are in
test_gp_boundary.py. Skips, saying so, without a GP binding."""

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
