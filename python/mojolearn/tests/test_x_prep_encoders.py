# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: OrdinalEncoder and OneHotEncoder against scikit-learn
(exact: the outputs are codes), numeric categories with ties and -0.0."""
import numpy as np
try:
    import pytest
    sk = pytest.importorskip("sklearn.preprocessing")
except ImportError:
    import sklearn.preprocessing as sk
import mojolearn as ml


def _data(seed=0, n=257, d=5):
    rng = np.random.default_rng(seed)
    X = np.floor(rng.standard_normal((n, d)) * 2).astype(np.float32)
    X[:, 2] = np.float32(7)
    X[0, 3] = np.float32(-0.0)
    X[1, 3] = np.float32(0.0)
    X[:, 4] = rng.integers(0, 2, n).astype(np.float32)   # binary column
    return X


def test_ordinal():
    X = _data()
    m = ml.OrdinalEncoder().fit(X)
    r = sk.OrdinalEncoder().fit(X)
    for a, b in zip(m.categories_, r.categories_):
        np.testing.assert_array_equal(np.asarray(a), b)
    np.testing.assert_array_equal(np.asarray(m.transform(X)), r.transform(X))
    Xh = _data(1)
    Xh[0, 0] = np.float32(99)
    m = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X)
    r = sk.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X)
    np.testing.assert_array_equal(np.asarray(m.transform(Xh)), r.transform(Xh))
    try:
        ml.OrdinalEncoder().fit(X).transform(Xh)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown category not refused")


def test_onehot():
    X = _data()
    Xh = _data(2)
    Xh[3, 1] = np.float32(55)
    for kw in (dict(handle_unknown="ignore"), dict(drop="first"), dict(drop="if_binary", handle_unknown="ignore")):
        m = ml.OneHotEncoder(**kw).fit(X)
        r = sk.OneHotEncoder(sparse_output=False, **kw).fit(X)
        np.testing.assert_array_equal(np.asarray(m.transform(X)), r.transform(X))
        if kw.get("handle_unknown") == "ignore":
            np.testing.assert_array_equal(np.asarray(m.transform(Xh)), r.transform(Xh))


if __name__ == "__main__":
    test_ordinal()
    test_onehot()
    print("PASS test_x_prep_encoders")
