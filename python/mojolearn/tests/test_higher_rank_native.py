# SPDX-License-Identifier: Apache-2.0
"""Existing primitive higher derivatives retain rank and differentiable views."""
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


@pytest.mark.parametrize("shape", [(2, 3, 2), (2, 1, 3, 2)])
def test_recorded_gradient_preserves_alias_shapes_and_second_derivative(shape):
    value, direction = fixture(shape)
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        alias = native(s.reshape, x, (6, 2))
        y = native(s.exp, alias)
        loss = native(s.sum, y)
        native(s.backward, loss, create_graph=True)
        gx, ga = native(s.grad_tensor, x), native(s.grad_tensor, alias)
        close(native(s.numpy, gx), np.exp(value.astype(np.float64)))
        close(native(s.numpy, ga), np.exp(value.astype(np.float64)).reshape(6, 2))
        assert native(s.shape, x) == shape
        assert native(s.shape, alias) == (6, 2)
        v = native(s.tensor, direction)
        weighted = native(s.mul, gx, v)
        native(s.backward, native(s.sum, weighted))
        close(native(s.grad, x), np.exp(value.astype(np.float64)) * direction)


@pytest.mark.parametrize("shape", [(2, 3, 2), (2, 1, 3, 2)])
@pytest.mark.parametrize("curved", [False, True])
def test_hvp_zero_and_nonzero_curvature_preserve_full_rank(shape, curved):
    value, direction = fixture(shape)
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        y = native(s.exp, x) if curved else native(s.scale, x, 2)
        loss = native(s.sum, y)
        v = native(s.tensor, direction)
        (hv,) = native(s.hvp, loss, [x], [v])
        expected = np.exp(value.astype(np.float64)) * direction if curved else np.zeros(shape)
        close(native(s.numpy, hv), expected)


def test_hvp_disconnected_parameter_has_its_own_full_rank():
    value, direction = fixture((2, 3, 2))
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        unused = native(s.tensor, np.ones((2, 1, 3, 2), np.float32), requires_grad=True)
        loss = native(s.sum, native(s.exp, x))
        v = native(s.tensor, direction)
        u = native(s.tensor, np.ones((2, 1, 3, 2), np.float32))
        hv, hu = native(s.hvp, loss, [x, unused], [v, u])
        close(native(s.numpy, hv), np.exp(value.astype(np.float64)) * direction)
        close(native(s.numpy, hu), np.zeros((2, 1, 3, 2)))


@pytest.mark.parametrize("curved", [False, True])
def test_jvp_preserves_output_rank_distinct_from_input_rank(curved):
    value, direction = fixture((2, 3, 2))
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        y = native(s.exp, x) if curved else native(s.scale, x, 2)
        output = native(s.reshape, y, (2, 1, 3, 2))
        v = native(s.tensor, direction)
        ju = native(s.jvp, output, [x], [v])
        expected = np.exp(value.astype(np.float64)) * direction if curved else 2 * direction
        close(native(s.numpy, ju), expected.reshape(2, 1, 3, 2))


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
