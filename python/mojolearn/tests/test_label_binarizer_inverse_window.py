# SPDX-License-Identifier: Apache-2.0
"""Tiny inverse-label regressions: never read beyond the output arena."""
import numpy as np
from mojolearn._expansion_prep import LabelBinarizer


def test_binary_one_and_two_columns():
    model = LabelBinarizer(neg_label=-1, pos_label=2).fit([3, 7, 3, 7])
    scores = np.array([-1, 0.5, 0.75, 2], dtype=np.float32)
    expected = np.array([3, 3, 7, 7])
    one = scores[:, None]
    two = np.column_stack([np.array([99, -99, 0, 1], dtype=np.float32), scores])
    np.testing.assert_array_equal(model.inverse_transform(one), expected)
    np.testing.assert_array_equal(model.inverse_transform(two), expected)
    np.testing.assert_array_equal(model.inverse_transform(two, threshold=1), [3, 3, 3, 7])
    np.testing.assert_array_equal(model.inverse_transform(model.transform([7, 3, 7])), [7, 3, 7])


def test_single_class_two_columns():
    model = LabelBinarizer().fit([11, 11])
    np.testing.assert_array_equal(model.inverse_transform(np.array([[0, 1], [1, 0]], dtype=np.float32)), [11, 11])


def test_multiclass_inverse_unchanged():
    model = LabelBinarizer().fit([3, 7, 11])
    scores = np.array([[0, 2, 1], [4, 0, -1], [0, 1, 2]], dtype=np.float32)
    np.testing.assert_array_equal(model.inverse_transform(scores), [7, 3, 11])


def test_binary_rejects_three_columns():
    model = LabelBinarizer().fit([3, 7])
    try:
        model.inverse_transform(np.zeros((2, 3), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError('binary inverse must reject three columns')
