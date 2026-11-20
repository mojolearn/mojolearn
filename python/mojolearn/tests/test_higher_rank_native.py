# SPDX-License-Identifier: Apache-2.0
"""A direction of the wrong full shape is refused and leaves the session usable.

The rank-preserving gradient, HVP and JVP arithmetic runs natively in
checks/native_tests/ops_higher_{cpu,gpu}.mojo (fixtures 7..15)."""
import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def native(function, *args, **kwargs):
    return audit_native_call(function, *args, **kwargs).assert_native()


def fixture(shape):
    value = np.arange(12, dtype=np.float32).reshape(shape) / np.float32(16)
    direction = np.arange(1, 13, dtype=np.float32).reshape(shape) / np.float32(8)
    return value, direction


def close(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("method", ["hvp", "jvp"])
@pytest.mark.parametrize("wrong_shape", [(6, 2), (3, 2, 2)])
def test_direction_full_shape_refusal_preserves_session(method, wrong_shape):
    value, direction = fixture((2, 3, 2))
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        y = native(s.exp, x)
        output = native(s.sum, y) if method == "hvp" else y
        operation = getattr(s, method)
        wrong = native(s.tensor, direction.reshape(wrong_shape))
        rejected = audit_native_call(operation, output, [x], [wrong])
        assert isinstance(rejected.exception, ValueError)
        assert "full shape" in str(rejected.exception)
        assert not rejected.frames and not rejected.wrapper_work and not rejected.forbidden_native
        v = native(s.tensor, direction)
        result = native(operation, output, [x], [v])
        handle = result[0] if method == "hvp" else result
        close(native(s.numpy, handle), np.exp(value.astype(np.float64)) * direction)
