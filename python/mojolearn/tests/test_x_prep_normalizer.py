# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: Normalizer against scikit-learn (a zero row included)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import Normalizer as SkN
import mojolearn as ml


def test_normalizer():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 7)).astype(np.float32)
    X[3] = 0
    for norm in ("l1", "l2", "max"):
        np.testing.assert_allclose(np.asarray(ml.Normalizer(norm=norm).fit(X).transform(X)),
                                   SkN(norm=norm).fit(X).transform(X.astype(np.float64)), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    test_normalizer()
    print("PASS test_x_prep_normalizer")
