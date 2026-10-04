# SPDX-License-Identifier: Apache-2.0
"""Real GPU binding regression for separately rounded OOB sums and means.

Run with MOJOLEARN_NUMERIC_MODE=identical on each supported GPU backend.
These assertions compare bits, including zero-count prediction semantics.
"""
import math

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


@pytest.mark.parametrize("target", [[1.0], [-1.0], [2.0**60, 1.0, -(2.0**60)],
                                    [2.0, -2.0, 0.0], [1.0] * 65])
@pytest.mark.parametrize("covered", [False, True])
def test_oob_finite_sum_mean_bits(target, covered):
    assert ml.numeric_mode() == "identical", "Run this regression in IDENTICAL mode"
    binding = _backend.binding("_mojolearn_x_trees", "identical")
    y = np.asarray(target, dtype=np.float32)
    counts = np.full(len(y), 2 if covered else 0, dtype=np.int32)
    acc = y.astype(np.float64) * (2 if covered else 0)
    pred = np.full(len(y), np.nan, dtype=np.float64)
    words = np.full(4, np.nan, dtype=np.float64)
    flags = np.full(4, -1, dtype=np.int32)
    before = [a.tobytes() for a in (acc, counts, y)]
    binding.x_trees_oob_r2([a.ctypes.data for a in (acc, counts, y, pred, words, flags)], [len(y)])
    expected_pred = y.astype(np.float64) if covered else np.zeros(len(y), dtype=np.float64)
    total = math.fsum(float(v) for v in y)
    mean = total / len(y)
    expected = np.asarray([total,
        math.fsum((float(v) - mean) ** 2 for v in y),
        math.fsum((float(v) - float(p)) ** 2 for v, p in zip(y, expected_pred)), mean], dtype=np.float64)
    np.testing.assert_array_equal(flags, np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(pred.view(np.uint64), expected_pred.view(np.uint64))
    np.testing.assert_array_equal(words.view(np.uint64), expected.view(np.uint64))
    assert [a.tobytes() for a in (acc, counts, y)] == before
