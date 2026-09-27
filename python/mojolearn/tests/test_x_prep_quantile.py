# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: QuantileTransformer against scikit-learn (no subsample,
so the quantiles are the reference's own)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import QuantileTransformer as SkQT
import mojolearn as ml


def _data(seed=0, n=700, d=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1] * 2)
    X[::11, 2] = np.nan
    X[:, 3] = np.exp(X[:, 3])
    return X


def test_quantile():
    X = _data()
    Xh = _data(1)
    for kw in (dict(n_quantiles=100), dict(n_quantiles=50, output_distribution="normal")):
        m = ml.QuantileTransformer(**kw).fit(X)
        r = SkQT(**kw).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.quantiles_), r.quantiles_, rtol=1e-5, atol=1e-5)
        a, b = np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=2e-4)


if __name__ == "__main__":
    test_quantile()
    print("PASS test_x_prep_quantile")
