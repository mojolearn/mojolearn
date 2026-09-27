# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelRidge sample_weight and kernel='precomputed' (lane
x-neighbors-krr-options, 2026-09-27).

Refusal checks always run; value checks run through whichever kernel-methods
binding this install loads and skip, saying so, without one. Exact checks:
an integer-valued X has an exact linear Gram under any summation order, so
kernel='precomputed' on that Gram must give kernel='linear''s bits, and a unit
weight must give the unweighted bits. scikit-learn is compared to a tolerance.
"""

import numpy as np
import pytest

from mojolearn.kernel_methods import KernelRidge


def _data(n=96, d=5, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=(n, d)).astype(np.float32)
    y = (a @ rng.normal(size=d) + rng.normal(size=n)).astype(np.float32)
    return a, y


def _fit_or_skip(est, *args, **kw):
    try:
        return est.fit(*args, **kw)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no kernel-methods binding on this install: {exc}")


def test_refusals():
    a, y = _data()
    with pytest.raises(ValueError, match="square"):
        KernelRidge(kernel="precomputed").fit(a, y)
    with pytest.raises(ValueError, match="sample_weight"):
        KernelRidge().fit(a, y, sample_weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match="entries"):
        KernelRidge().fit(a, y, sample_weight=np.ones(3))


def test_precomputed_is_the_linear_kernel_bit_for_bit():
    a, y = _data()
    lin = _fit_or_skip(KernelRidge(alpha=2.0, kernel="linear"), a, y)
    k = (a.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    pre = KernelRidge(alpha=2.0, kernel="precomputed").fit(k, y)
    assert np.asarray(pre.dual_coef_).tobytes() == np.asarray(lin.dual_coef_).tobytes()
    q = a[:20]
    kq = (q.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    assert np.asarray(pre.predict(kq)).tobytes() == np.asarray(lin.predict(q)).tobytes()
    assert np.asarray(pre.predict(kq)).tobytes() == np.asarray(pre.predict(kq)).tobytes()


def test_unit_weight_is_the_unweighted_fit():
    a, y = _data()
    base = _fit_or_skip(KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1), a, y)
    one = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(a, y, sample_weight=1.0)
    assert np.asarray(one.dual_coef_).tobytes() == np.asarray(base.dual_coef_).tobytes()


def test_weights_match_scikit_learn():
    sk = pytest.importorskip("sklearn.kernel_ridge")
    a, y = _data()
    w = 0.5 + 0.5 * (np.arange(len(y)) % 4)
    w[5] = 0.0
    m = _fit_or_skip(KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1), a, y, sample_weight=w)
    r = sk.KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(a, y, sample_weight=w)
    np.testing.assert_allclose(np.asarray(m.dual_coef_), r.dual_coef_, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.predict(a)), r.predict(a), rtol=1e-3, atol=1e-3)
    assert float(np.asarray(m.dual_coef_)[5]) == 0.0
    k = (a.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    p = KernelRidge(alpha=1.0, kernel="precomputed").fit(k, y, sample_weight=w)
    rp = sk.KernelRidge(alpha=1.0, kernel="precomputed").fit(k.astype(np.float64), y, sample_weight=w)
    np.testing.assert_allclose(np.asarray(p.dual_coef_), rp.dual_coef_, rtol=1e-2, atol=1e-3)


def test_repeat_calls_are_the_same_bits():
    a, y = _data()
    w = 0.25 + (np.arange(len(y)) % 3)
    one = _fit_or_skip(KernelRidge(alpha=0.5, kernel="rbf", gamma=0.2), a, y, sample_weight=w)
    two = KernelRidge(alpha=0.5, kernel="rbf", gamma=0.2).fit(a, y, sample_weight=w)
    assert np.asarray(one.dual_coef_).tobytes() == np.asarray(two.dual_coef_).tobytes()
    assert np.asarray(one.predict(a)).tobytes() == np.asarray(two.predict(a)).tobytes()
