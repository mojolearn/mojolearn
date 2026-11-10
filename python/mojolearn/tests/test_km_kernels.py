# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Kernel-method public estimator, buffer and scikit-learn comparison seams.

Direct kernel cells and the numerical Nystroem basis reconstruction run in
checks/native_tests/kernel_methods_cells_*.mojo. Public negative-input refusals,
KernelRidge's whole-estimator reference and Nystroem's public owners stay here.
"""

import numpy as np
import pytest

from mojolearn.kernel_methods import KernelRidge, Nystroem


def _data(n=80, d=6, seed=4):
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(size=(n, d))).astype(np.float32)
    x[::7, 2] = 0.0
    y = (x @ rng.normal(size=d)).astype(np.float32)
    return x, y


def _fit_or_skip(est, *args):
    try:
        return est.fit(*args)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no kernel-methods binding on this install: {exc}")


def test_negative_input_is_refused_by_the_chi2_kernels():
    x, y = _data()
    for k in ("chi2", "additive_chi2"):
        with pytest.raises(ValueError, match="negative"):
            KernelRidge(kernel=k).fit(-x, y)
        with pytest.raises(ValueError, match="negative"):
            Nystroem(kernel=k, n_components=8).fit(-x)


@pytest.mark.parametrize("kernel, gamma, alpha", [
    ("cosine", None, 1.0), ("chi2", None, 1.0), ("chi2", 0.3, 1.0), ("additive_chi2", None, 400.0),
])
def test_kernel_ridge_matches_scikit_learn(kernel, gamma, alpha):
    sk = pytest.importorskip("sklearn.kernel_ridge")
    x, y = _data()
    m = _fit_or_skip(KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma), x, y)
    # scikit-learn 1.9's KernelRidge passes gamma=None through to chi2_kernel,
    # whose `K *= gamma` then fails on an object array; chi2_kernel's own
    # default (the value gamma=None means here) is 1.0
    sk_gamma = 1.0 if (kernel == "chi2" and gamma is None) else gamma
    r = sk.KernelRidge(alpha=alpha, kernel=kernel, gamma=sk_gamma).fit(x.astype(np.float64), y)
    np.testing.assert_allclose(np.asarray(m.predict(x)), r.predict(x.astype(np.float64)), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("kernel", ["chi2", "cosine"])
def test_nystroem_basis_public_shapes_and_owners(kernel):
    """Public input/output owners and basis indexing stay at the boundary."""
    x, _ = _data()
    before = x.copy()
    n = _fit_or_skip(Nystroem(kernel=kernel, n_components=12, random_state=1), x)
    idx = np.asarray(n.component_indices_).astype(np.int64)
    assert idx.shape == (12,) and len(set(idx.tolist())) == 12
    assert np.all((0 <= idx) & (idx < len(x)))
    np.testing.assert_array_equal(np.asarray(n.components_), x[idx])
    assert np.asarray(n.normalization_).shape == (12, 12)
    basis_output = np.asarray(n.transform(x[idx]))
    full_output = np.asarray(n.transform(x))
    assert basis_output.shape == (12, 12) and full_output.shape == (80, 12)
    assert basis_output.dtype == np.float32 and full_output.dtype == np.float32
    np.testing.assert_array_equal(x, before)
