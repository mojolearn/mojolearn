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


def test_spline():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3)).astype(np.float32)
    Xh = (rng.standard_normal((100, 3)) * 1.5).astype(np.float32)
    for kw in (dict(), dict(n_knots=6, degree=2, knots="quantile"), dict(extrapolation="continue"),
               dict(include_bias=False, degree=1)):
        m, r = ml.SplineTransformer(**kw).fit(X), SkS(**kw).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)),
                                   rtol=1e-3, atol=2e-4)


if __name__ == "__main__":
    test_spline()
    print("PASS test_x_prep_spline")
