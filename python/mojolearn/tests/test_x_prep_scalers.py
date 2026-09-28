# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: RobustScaler and MaxAbsScaler against scikit-learn at a
float32 tolerance, on small data with ties, a constant column, a zero column
and NaN. Not an identity claim (that is tools/algos_lane_check.sh)."""
import numpy as np
try:
    import pytest
    sk = pytest.importorskip("sklearn.preprocessing")
except ImportError:                         # run as a script on a box without pytest
    import sklearn.preprocessing as sk
import mojolearn as ml


def _data(seed=0, n=301, d=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32) * np.float32(3)
    X[:, 1] = np.round(X[:, 1])            # ties
    X[:, 2] = np.float32(2.5)              # constant
    X[:, 3] = np.float32(0)                # all zero
    X[::7, 4] = np.nan                     # missing
    return X


def test_robust_scaler():
    X = _data()
    for kw in ({}, dict(quantile_range=(10.0, 90.0)), dict(with_centering=False),
               dict(unit_variance=True), dict(unit_variance=True, quantile_range=(5.0, 80.0))):
        m = ml.RobustScaler(**kw).fit(X)
        r = sk.RobustScaler(**kw).fit(X.astype(np.float64))
        if r.center_ is not None and m.center_ is not None:
            np.testing.assert_allclose(np.asarray(m.center_), r.center_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(m.scale_), r.scale_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(m.transform(X)), r.transform(X.astype(np.float64)),
                                   rtol=1e-4, atol=1e-4)


def test_maxabs_scaler():
    X = _data(1)
    m = ml.MaxAbsScaler().fit(X)
    r = sk.MaxAbsScaler().fit(X.astype(np.float64))
    np.testing.assert_allclose(np.asarray(m.scale_), r.scale_, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(m.transform(X)), r.transform(X.astype(np.float64)), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    test_robust_scaler()
    test_maxabs_scaler()
    print("PASS test_x_prep_scalers")
