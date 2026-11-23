# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: SplineTransformer against scikit-learn (held-out rows
beyond the training range exercise the extrapolation)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import SplineTransformer as SkS
import mojolearn as ml


# test_spline moved to checks/native_tests/prep_spline.mojo::test_spline (lane py-delete, 2026-11-23).


def test_spline_options():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 3)).astype(np.float32)
    Xh = (rng.standard_normal((100, 3)) * 1.7).astype(np.float32)
    kn = np.array([[-2.0, -1.5, -1.0], [-0.5, 0.0, 0.1], [0.4, 0.3, 0.8], [2.0, 1.1, 1.5]])
    sw = rng.integers(0, 4, 300).astype(np.float64)
    cases = [
        (dict(knots=kn), None), (dict(knots=kn, degree=2, extrapolation="linear"), None),
        (dict(extrapolation="linear"), None),
        (dict(extrapolation="periodic"), None), (dict(extrapolation="periodic", degree=2, n_knots=4), None),
        (dict(extrapolation="periodic", include_bias=False, knots=kn), None),
        (dict(extrapolation="periodic", degree=3, n_knots=4), None),
        (dict(), sw), (dict(knots="quantile", n_knots=6), sw), (dict(knots="quantile", n_knots=4), None),
        (dict(order="F", degree=2), None),
    ]
    for kw, w in cases:
        m = ml.SplineTransformer(**kw).fit(X, sample_weight=w)
        r = SkS(**kw).fit(X.astype(np.float64), sample_weight=w)
        a, b = np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=5e-4, err_msg=str(kw))
    # degree <= 1 'linear': the reference raises `degree` inside its feature loop, so every later
    # feature reads the wrong knot range (x_prep/NOT_IMPLEMENTED.tsv, DIFFERS BY NAME); one feature
    for kw in (dict(extrapolation="linear", degree=1), dict(extrapolation="linear", degree=0)):
        m, r = ml.SplineTransformer(**kw).fit(X[:, :1]), SkS(**kw).fit(X[:, :1].astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.transform(Xh[:, :1])), r.transform(Xh[:, :1].astype(np.float64)),
                                   rtol=1e-3, atol=5e-4, err_msg=str(kw))
    # handle_missing='zeros': NaN left out of the knots and encoded as a zero block
    Xn, Xhn = X.copy(), Xh.copy()
    Xn[::7, 1] = np.nan
    Xhn[::5, 0] = np.nan
    # the reference has no handle_missing in 1.7 (the newest on Python 3.10)
    missing_cases = (dict(handle_missing="zeros"), dict(handle_missing="zeros", knots="quantile"),
                     dict(handle_missing="zeros", extrapolation="periodic"))
    for kw in (missing_cases if "handle_missing" in SkS().get_params() else ()):
        m, r = ml.SplineTransformer(**kw).fit(Xn), SkS(**kw).fit(Xn.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.transform(Xhn)), r.transform(Xhn.astype(np.float64)),
                                   rtol=1e-3, atol=5e-4, err_msg=str(kw))
    for est in (ml.SplineTransformer(), SkS()):
        try:
            est.fit(Xn)
            raise AssertionError("no ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    test_spline_options()
    print("PASS test_x_prep_spline")
