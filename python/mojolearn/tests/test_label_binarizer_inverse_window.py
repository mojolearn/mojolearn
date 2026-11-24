# SPDX-License-Identifier: Apache-2.0
"""LabelBinarizer inverse public seams.

The inverse programs' class words and output window (binary one and two
columns, a single class, multiclass first maximum) are checked natively in
checks/native_tests/prep_label_binarizer_inverse_window.mojo; the refusal of a
three-column binary inverse is in test_prep_boundary.py.
"""
import numpy as np
from mojolearn._expansion_prep import LabelBinarizer


def test_binary_transform_round_trip():
    model = LabelBinarizer(neg_label=-1, pos_label=2).fit([3, 7, 3, 7])
    np.testing.assert_array_equal(model.inverse_transform(model.transform([7, 3, 7])), [7, 3, 7])
