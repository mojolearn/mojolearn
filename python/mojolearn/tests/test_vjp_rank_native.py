# SPDX-License-Identifier: Apache-2.0
"""The vjp refusal of a direction with the right size and the wrong full shape,
and the session's recovery after it. The rank and broadcast oracles run
natively in checks/native_tests/core_vjp_rank_*.mojo."""
import numpy as np
import pytest
from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def native(function, *args, **kwargs):
    return audit_native_call(function, *args, **kwargs).assert_native()


@pytest.mark.parametrize("wrong_shape", [(6, 2), (3, 2, 2)])
def test_vjp_rejects_same_flat_storage_wrong_shape_and_recovers(wrong_shape):
    with ops.Session(record=True) as s:
        x = native(s.tensor, np.ones((2, 3, 2), np.float32), requires_grad=True)
        y = native(s.scale, x, 2)
        wrong = native(s.tensor, np.ones(wrong_shape, np.float32))
        rejected = audit_native_call(s.vjp, y, [x], wrong)
        assert isinstance(rejected.exception, ValueError)
        assert "full shape" in str(rejected.exception)
        assert not rejected.frames and not rejected.wrapper_work and not rejected.forbidden_native
        seed = native(s.tensor, np.ones((2, 3, 2), np.float32))
        (dx,) = native(s.vjp, y, [x], seed)
        np.testing.assert_array_equal(native(s.numpy, dx), np.full((2, 3, 2), 2, np.float32))

