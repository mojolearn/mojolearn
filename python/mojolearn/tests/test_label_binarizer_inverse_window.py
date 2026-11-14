# SPDX-License-Identifier: Apache-2.0
"""LabelBinarizer inverse public seams.

The inverse programs' class words and output window (binary one and two
columns, a single class, multiclass first maximum) are checked natively in
checks/native_tests/prep_label_binarizer_inverse_window_suite.mojo.
"""
import numpy as np
from mojolearn._expansion_prep import LabelBinarizer


def test_binary_transform_round_trip():
    model = LabelBinarizer(neg_label=-1, pos_label=2).fit([3, 7, 3, 7])
    np.testing.assert_array_equal(model.inverse_transform(model.transform([7, 3, 7])), [7, 3, 7])


def test_binary_rejects_three_columns():
    model = LabelBinarizer().fit([3, 7])
    try:
        model.inverse_transform(np.zeros((2, 3), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError('binary inverse must reject three columns')
