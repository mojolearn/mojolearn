# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: feature selection against scikit-learn."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.feature_selection as skfs
import mojolearn as ml


def _data(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) * np.linspace(0.3, 2.0, d)).astype(np.float32)
    X[:, 2] = np.float32(4.0)
    return X


def test_variance_threshold():
    X = _data()
    for th in (0.0, 0.5):
        m, r = ml.VarianceThreshold(th).fit(X), skfs.VarianceThreshold(th).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.variances_), r.variances_, rtol=1e-4, atol=1e-6)
        assert list(m.get_support()) == list(r.get_support())
        np.testing.assert_array_equal(np.asarray(m.transform(X)), r.transform(X))


if __name__ == "__main__":
    test_variance_threshold()
    print("PASS test_x_prep_selection")


def test_scores_and_kbest():
    rng = np.random.default_rng(3)
    n, d = 400, 7
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 0.6, d)).astype(np.float32)
    yr = (X[:, 3] * 0.4 + rng.standard_normal(n)).astype(np.float32)
    for mine, ref, Xa, ya in ((ml.f_classif, skfs.f_classif, X, y), (ml.chi2, skfs.chi2, np.abs(X), y),
                              (ml.f_regression, skfs.f_regression, X, yr)):
        s, p = mine(Xa, ya)
        rs, rp = ref(Xa.astype(np.float64), ya)
        np.testing.assert_allclose(np.asarray(s), rs, rtol=2e-3, atol=1e-4)
        ok = rp > 1e-6
        np.testing.assert_allclose(np.asarray(p)[ok], rp[ok], rtol=2e-2, atol=1e-6)
    m = ml.SelectKBest(k=3).fit(X, y)
    r = skfs.SelectKBest(k=3).fit(X, y)
    assert list(m.get_support()) == list(r.get_support())


if __name__ == "__main__":
    test_scores_and_kbest()
    print("PASS test_x_prep_selection (kbest)")
