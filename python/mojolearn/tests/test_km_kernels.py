# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelRidge / Nystroem with scikit-learn's cosine, chi2 and additive_chi2
kernels (lane x-neighbors-km-kernels, 2026-09-27).

Refusals always run; value checks run through whichever kernel-methods binding
this install loads and skip, saying so, without one. scikit-learn is compared
to a tolerance (float32 cells against its float64); repeat calls must be the
same bits.
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
    again = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma).fit(x, y)
    assert np.asarray(again.dual_coef_).tobytes() == np.asarray(m.dual_coef_).tobytes()
    assert np.asarray(again.predict(x)).tobytes() == np.asarray(m.predict(x)).tobytes()


def test_kernel_matrix_cells_match_scikit_learn():
    """The approximation Z Z^T over the basis rows equals the exact kernel
    there (Nystroem is exact on its own basis)."""
    pw = pytest.importorskip("sklearn.metrics.pairwise")
    x, _ = _data()
    for kernel, ref in (("chi2", pw.chi2_kernel(x.astype(np.float64))),
                        ("cosine", pw.cosine_similarity(x.astype(np.float64)))):
        n = _fit_or_skip(Nystroem(kernel=kernel, n_components=12, random_state=1), x)
        idx = np.asarray(n.component_indices_).astype(np.int64)
        z = np.asarray(n.transform(x[idx])).astype(np.float64)
        np.testing.assert_allclose(z @ z.T, ref[np.ix_(idx, idx)], atol=5e-3)
        assert np.asarray(n.transform(x)).tobytes() == np.asarray(n.transform(x)).tobytes()
