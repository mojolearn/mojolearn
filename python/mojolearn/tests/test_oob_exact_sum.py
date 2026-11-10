# SPDX-License-Identifier: Apache-2.0
"""Python pointer-list ABI and buffer ownership for the OOB R2 native entry.

The five-target numerical matrix is checked directly in Mojo through the
host and device functions (checks/native_tests/oob_exact_sum_*), including
separately rounded cancellation sums, means and zero-count predictions.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


@pytest.mark.parametrize("covered", [False, True])
def test_oob_pointer_list_abi_writes_results_and_preserves_inputs(covered):
    assert ml.numeric_mode() == "identical", "Run this regression in IDENTICAL mode"
    binding = _backend.binding("_mojolearn_x_trees", "identical")
    y = np.asarray([1.0], dtype=np.float32)
    counts = np.asarray([2 if covered else 0], dtype=np.int32)
    acc = np.asarray([2.0 if covered else 0.0], dtype=np.float64)
    pred = np.full(1, np.nan, dtype=np.float64)
    words = np.full(4, np.nan, dtype=np.float64)
    flags = np.full(4, -1, dtype=np.int32)
    before = [a.tobytes() for a in (acc, counts, y)]
    binding.x_trees_oob_r2([a.ctypes.data for a in (acc, counts, y, pred, words, flags)], [1])
    # Literal route/packing sentinels; numerical oracles and the full target
    # matrix are native. No Python arithmetic computes expected OOB values.
    assert pred[0] == (1.0 if covered else 0.0)
    np.testing.assert_array_equal(words, [1.0, 0.0, 0.0 if covered else 1.0, 1.0])
    np.testing.assert_array_equal(flags, [0, 0, 0, 0])
    assert [a.tobytes() for a in (acc, counts, y)] == before
