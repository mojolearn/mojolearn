# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: IterativeImputer (BayesianRidge) against scikit-learn
on correlated data with 15% missing."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SkII
import mojolearn as ml


def _data(seed=0, n=400, d=5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def test_iterative():
    X = _data()
    Xh = _data(1)
    for kw in (dict(), dict(max_iter=3, imputation_order="descending", initial_strategy="median"),
               dict(min_value=-1.0, max_value=1.0)):
        m, r = ml.IterativeImputer(**kw), SkII(**kw)
        a, b = np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64))
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)
        assert m.n_iter_ == r.n_iter_, (m.n_iter_, r.n_iter_)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)),
                                   rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    test_iterative()
    print("PASS test_x_prep_iterative")
