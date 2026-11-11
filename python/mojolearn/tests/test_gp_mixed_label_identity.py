# SPDX-License-Identifier: Apache-2.0
"""Public GP class identities for admitted mixed native scalar comparisons."""
import sys

import numpy as np
import pytest

from mojolearn._cpu_reference import reference_training
from mojolearn._gpc_impl import GaussianProcessClassifier


@pytest.mark.parametrize("labels", [
    [np.int64(1), 1.0, np.float32(2), 2, np.float64(3), 3],
    [np.float64(-0.0), 0, np.int64(1), 1.0, np.float32(2), 2],
    [0, np.float64(0), sys.hash_info.modulus, sys.hash_info.modulus,
     2 * sys.hash_info.modulus, np.float32(3)],
    [1 << 90, np.float64(2.0 ** 90), (1 << 90) + 1,
     np.float64(2.0 ** 90), np.int64(2), 2.0],
])
def test_mixed_scalar_classes_retain_first_original_objects(labels):
    # This independent boundary oracle is deliberately the documented scalar
    # rule, not a Float64 normalization or a reproduction of the native index.
    expected = sorted(dict.fromkeys(labels))
    x = np.array([[-3, 0], [-2, 1], [-1, -1], [1, 1], [2, -1], [3, 0]], dtype=np.float32)
    with reference_training():
        model = GaussianProcessClassifier(max_iter_predict=3).fit(x, labels)
    assert len(model.classes_) == len(expected)
    assert all(actual is original for actual, original in zip(model.classes_, expected))
