# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: PowerTransformer against scikit-learn (lambdas at a
search tolerance, outputs at float32 tolerance)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import PowerTransformer as SkPT
import mojolearn as ml


def _data(seed=0, n=600, d=4):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.standard_normal(n), rng.exponential(2.0, n), rng.gamma(2.0, 1.5, n),
                         rng.lognormal(0, 0.7, n)]).astype(np.float32)
    return X


def test_power():
    X = _data()
    Xh = _data(1)
    for kw in (dict(), dict(standardize=False), dict(method="box-cox")):
        if kw.get("method") == "box-cox":
            X, Xh = np.abs(X) + np.float32(0.1), np.abs(Xh) + np.float32(0.1)
        m = ml.PowerTransformer(**kw).fit(X)
        r = SkPT(**kw).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.lambdas_), r.lambdas_, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)),
                                   rtol=5e-3, atol=5e-3)
        # inverse_transform against the reference's inverse at OUR lambdas and scaler
        r.lambdas_ = np.asarray(m.lambdas_).astype(np.float64)
        if kw.get("standardize", True):
            r._scaler.mean_ = np.asarray(m._mean).astype(np.float64)
            r._scaler.scale_ = np.asarray(m._scale).astype(np.float64)
        Z = np.asarray(m.transform(Xh))
        Z[3, 0] = np.nan
        a, b = np.asarray(m.inverse_transform(Z)), r.inverse_transform(Z.astype(np.float64))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-4)
        ok = ~np.isnan(Z)
        np.testing.assert_allclose(a[ok], Xh[ok], rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    test_power()
    print("PASS test_x_prep_power")
