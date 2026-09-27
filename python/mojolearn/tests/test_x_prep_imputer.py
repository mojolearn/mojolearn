# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: SimpleImputer against scikit-learn (every strategy, an
all-missing column, a numeric missing_values)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.impute import SimpleImputer as SkSI
import mojolearn as ml


def _data(seed=0, n=203, d=5):
    rng = np.random.default_rng(seed)
    X = np.round(rng.standard_normal((n, d)) * 2).astype(np.float32)
    X[rng.random((n, d)) < 0.2] = np.nan
    X[:, 3] = np.nan
    return X


def test_strategies():
    X = _data()
    Xh = _data(1)
    for kw in (dict(strategy="mean"), dict(strategy="median"), dict(strategy="most_frequent"),
               dict(strategy="constant", fill_value=-3.0), dict(strategy="median", keep_empty_features=True)):
        m = ml.SimpleImputer(**kw).fit(X)
        r = SkSI(**kw).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.statistics_), np.asarray(r.statistics_, dtype=np.float64), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)), rtol=1e-5, atol=1e-6)


def test_numeric_missing():
    X = np.nan_to_num(_data(2), nan=-1.0)
    m = ml.SimpleImputer(missing_values=-1.0, strategy="mean").fit(X)
    r = SkSI(missing_values=-1.0, strategy="mean").fit(X.astype(np.float64))
    np.testing.assert_allclose(np.asarray(m.transform(X)), r.transform(X.astype(np.float64)), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_strategies()
    test_numeric_missing()
    print("PASS test_x_prep_imputer")
