# SPDX-License-Identifier: Apache-2.0
"""THE maximize= SEAM (DEVIATION 6200, IDENTITY_PATHS row 200), Python side.

The bit seam itself (a `maximize=True` step on `g` equals, bit for bit, a
`maximize=False` step on `-g` with the sign bit flipped; the fixture must
separate `-g` from `0.0 - g`; the caller's gradient is never left negated;
SGD with Nesterov and decay, plain SGD, Adam and AdamW, clipping off and on,
three steps) runs natively in checks/native_tests/optimizer_maximize_*.mojo
(pixi tasks check-native-optimizer-maximize-cpu/-gpu) and in the identity lane
training/checks/maximize_check.mojo (tools/identity_lanes/neural.checks).
This file keeps the Python admission: a non-bool `maximize` is refused.
"""
import numpy as np
import pytest

import mojolearn as ml


def test_optimizer_refuses_a_non_bool_maximize():
    params = [np.zeros((16, 8), np.float32), np.zeros((16,), np.float32), np.zeros((3, 16), np.float32)]
    with pytest.raises(TypeError):
        ml.training.Adam(params, maximize=1)
