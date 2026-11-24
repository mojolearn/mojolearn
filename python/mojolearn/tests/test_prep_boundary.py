# SPDX-License-Identifier: Apache-2.0
"""The preprocessing family at the Python boundary: refusals by type. Their bits on each side are the verify lanes."""
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# LabelBinarizer inverse (moved from test_label_binarizer_inverse_window.py).
from mojolearn._expansion_prep import LabelBinarizer


def test_binary_rejects_three_columns():
    model = LabelBinarizer().fit([3, 7])
    try:
        model.inverse_transform(np.zeros((2, 3), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError('binary inverse must reject three columns')
