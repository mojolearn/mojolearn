# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: PolynomialFeatures against scikit-learn."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import PolynomialFeatures as SkP
import mojolearn as ml


def test_poly():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    for kw in (dict(), dict(degree=3), dict(degree=(2, 3), interaction_only=True, include_bias=False)):
        m, r = ml.PolynomialFeatures(**kw).fit(X), SkP(**kw).fit(X)
        np.testing.assert_array_equal(np.asarray(m.powers_), r.powers_)
        np.testing.assert_allclose(np.asarray(m.transform(X)), r.transform(X.astype(np.float64)), rtol=1e-5, atol=1e-6)
    c = np.asarray(ml.PolynomialFeatures(degree=3).fit(X).transform(X))
    f = np.asarray(ml.PolynomialFeatures(degree=3, order="F").fit(X).transform(X))
    assert f.flags["F_CONTIGUOUS"] and not f.flags["C_CONTIGUOUS"]
    assert SkP(degree=3, order="F").fit(X).transform(X).flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(c.view(np.uint32), f.view(np.uint32))


if __name__ == "__main__":
    test_poly()
    print("PASS test_x_prep_polynomial")
