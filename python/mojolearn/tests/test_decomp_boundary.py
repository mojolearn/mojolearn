# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the decomp family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_ne2_g_decomp_entries.py
# The fused cast and finiteness scan of the decomposition inputs (N352, N353)
# casts as before and still refuses NaN and infinity.
import numpy as np
import pytest

import mojolearn as ml


def _ne2g_data(rows=64, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def test_decomp_input_cast_and_scan_in_one_pass():
    x = _ne2g_data(20, 4, 10)
    model = ml.TruncatedSVD(n_components=2).fit(x)
    as_f64 = np.asarray(model.transform(x.astype(np.float64)))
    np.testing.assert_array_equal(as_f64, np.asarray(model.transform(x)))
    ints = np.arange(80, dtype=np.int64).reshape(20, 4)
    np.testing.assert_array_equal(np.asarray(model.transform(ints)),
                                  np.asarray(model.transform(ints.astype(np.float32))))
    bad = x.astype(np.float64)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        model.transform(bad)
    bad[3, 1] = np.inf
    with pytest.raises(ValueError):
        model.transform(bad)


def test_pca_input_cast_and_scan_in_one_pass():
    x = _ne2g_data(20, 4, 11)
    model = ml.PCA(n_components=2).fit(x)
    np.testing.assert_array_equal(np.asarray(model.transform(x.astype(np.float64))),
                                  np.asarray(model.transform(x)))
    bad = x.astype(np.float64)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.transform(bad)
