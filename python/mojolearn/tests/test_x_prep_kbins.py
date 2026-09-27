# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: KBinsDiscretizer against scikit-learn (every strategy,
both quantile methods, a constant column, ties that collapse bins)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import KBinsDiscretizer as SkKB
import mojolearn as ml


def _data(seed=0, n=500, d=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1])          # ties: some quantile bins collapse
    X[:, 2] = np.float32(1.5)            # constant
    return X


def test_strategies():
    X = _data()
    for kw in (dict(strategy="uniform"), dict(strategy="quantile"),
               dict(strategy="quantile", quantile_method="linear"), dict(strategy="kmeans")):
        m = ml.KBinsDiscretizer(n_bins=5, encode="ordinal", **kw).fit(X)
        r = SkKB(n_bins=5, encode="ordinal", **kw).fit(X.astype(np.float64))
        # k-means on the tie column (integers equidistant from two centres)
        # is decided by the reference's distance rounding; compare the rest
        cols = [0, 2, 3] if kw["strategy"] == "kmeans" else range(X.shape[1])
        for j in cols:
            np.testing.assert_allclose(np.asarray(m.bin_edges_[j]), r.bin_edges_[j], rtol=1e-4, atol=1e-4)
            assert int(np.asarray(m.n_bins_)[j]) == int(r.n_bins_[j])
        got, want = np.asarray(m.transform(X))[:, cols], r.transform(X.astype(np.float64))[:, cols]
        assert np.mean(got == want) > 0.995, (kw, np.mean(got == want))
        # inverse_transform: the bin centres of the reference's own codes
        Z = r.transform(X.astype(np.float64))
        a, b = np.asarray(m.inverse_transform(Z)), r.inverse_transform(Z)
        np.testing.assert_allclose(a[:, cols], b[:, cols], rtol=1e-4, atol=1e-4)


def test_onehot():
    X = _data(1)
    m = ml.KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="uniform").fit(X)
    r = SkKB(n_bins=4, encode="onehot-dense", strategy="uniform").fit(X.astype(np.float64))
    got, want = np.asarray(m.transform(X)), r.transform(X.astype(np.float64))
    assert got.shape == want.shape and np.mean(got == want) > 0.995
    a, b = np.asarray(m.inverse_transform(want)), r.inverse_transform(want)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_strategies()
    test_onehot()
    print("PASS test_x_prep_kbins")
