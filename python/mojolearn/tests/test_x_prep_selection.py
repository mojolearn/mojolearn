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
