# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: Binarizer against scikit-learn (exact)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import Binarizer as SkB
import mojolearn as ml


def test_binarizer():
    rng = np.random.default_rng(0)
    X = np.round(rng.standard_normal((60, 5)), 1).astype(np.float32)
    for th in (0.0, 0.3, -1.0):
        np.testing.assert_array_equal(np.asarray(ml.Binarizer(threshold=th).fit(X).transform(X)),
                                      SkB(threshold=th).fit(X).transform(X))


if __name__ == "__main__":
    test_binarizer()
    print("PASS test_x_prep_binarizer")
