# SPDX-License-Identifier: Apache-2.0
"""Mamba block public boundary: the caller's buffers."""
import os

import numpy as np

# A public-caller contract: without the marker this module's calls count as library-internal
# (providers/python/estimator_lifecycle_pyapi_unpack.mojo:16 estimator_outside).
__mojolearn_user_boundary__ = True
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")


# ---------------------------------------------------------------- moved from test_c3_c5_no_copies.py (vdv-pytest-a)
# 0.8.66 regression C5: a block forward reads a C-contiguous float32 `x` in place and never writes it or its flags.


@pytest.mark.parametrize("kind", ["mamba1", "mamba2", "mamba3"])
def test_block_forward_reads_x_in_place_and_matches_the_copy(kind):
    from mojolearn.tests.test_mamba_native_public import _block, _tokens
    block, dm = _block(kind)
    x = _tokens(2, 7, dm)
    before = x.tobytes()
    y_in_place = np.asarray(block.forward(x)).tobytes()
    y_copy = np.asarray(block.forward(x.copy())).tobytes()
    assert y_in_place == y_copy
    assert x.tobytes() == before
    assert x.flags.writeable  # a forward never freezes its input
    strided = np.ascontiguousarray(np.repeat(x, 2, axis=2))[:, :, ::2]
    assert np.asarray(block.forward(strided)).tobytes() == y_copy
