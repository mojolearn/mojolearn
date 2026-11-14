# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The LayerNorm module's public gradient attributes. Forward and backward
against the float64 restatement of F.layer_norm, and the module's
normalised rows, are checked natively in
checks/native_tests/sequence_norm_{cpu,gpu}.mojo."""
import numpy as np

import mojolearn as ml


def test_module_gradient_attributes():
    ln = ml.LayerNorm(4)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    ln(x)
    ln.backward(np.ones_like(x))
    assert ln.weight_grad.shape == (4,) and ln.bias_grad.shape == (4,)
