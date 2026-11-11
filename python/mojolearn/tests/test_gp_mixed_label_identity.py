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


@pytest.mark.parametrize("labels", [
    [np.str_("b"), np.str_("a"), np.str_("b"), np.str_("c"), np.str_("a"), np.str_("c")],
    ["b\ud800", "a\ud800", "b\ud800", "c\ud800", "a\ud800", "c\ud800"],
])
def test_nested_text_labels_keep_native_text_routes_and_class_identity(labels):
    expected = sorted(dict.fromkeys(labels))
    nested = [tuple(labels[:2]), [labels[2], (labels[3], labels[4])], labels[5]]
    x = np.array([[-3, 0], [-2, 1], [-1, -1], [1, 1], [2, -1], [3, 0]], dtype=np.float32)
    with reference_training():
        model = GaussianProcessClassifier(max_iter_predict=3).fit(x, nested)
    assert len(model.classes_) == len(expected)
    assert all(actual is original for actual, original in zip(model.classes_, expected))


@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_native_array_label_capture_preserves_logical_matrix_order(layout):
    from mojolearn._array import Array

    storage = np.array([[-3, 7, -3], [7, -3, 7]], dtype=np.int64, order="F" if layout == "F" else "C")
    if layout == "reversed":
        storage = storage[::-1, ::-1]
    labels = Array.from_buffer(storage)
    expected_codes = (storage.ravel(order="C") == 7).astype(np.float32)
    x = np.array([[-3, 0], [-2, 1], [-1, -1], [1, 1], [2, -1], [3, 0]], dtype=np.float32)
    with reference_training():
        model = GaussianProcessClassifier(max_iter_predict=3).fit(x, labels)
    assert list(model.classes_) == [-3, 7]
    np.testing.assert_array_equal(np.asarray(model.y_train_).reshape(-1), expected_codes)
