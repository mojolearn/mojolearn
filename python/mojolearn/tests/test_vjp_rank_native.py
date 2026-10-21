# SPDX-License-Identifier: Apache-2.0
"""First derivative shape/refusal oracles; fixtures run only in this controller."""
import numpy as np
import pytest
from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def native(function, *args, **kwargs):
    return audit_native_call(function, *args, **kwargs).assert_native()


@pytest.mark.parametrize("shape", [(2, 3, 2), (2, 1, 3, 2)])
def test_vjp_preserves_full_gradient_rank(shape):
    value = np.arange(12, dtype=np.float32).reshape(shape) / np.float32(16)
    seed = np.arange(1, 13, dtype=np.float32).reshape(shape) / np.float32(8)
    with ops.Session(record=True) as s:
        x = native(s.tensor, value, requires_grad=True)
        y = native(s.scale, x, 2)
        v = native(s.tensor, seed)
        (dx,) = native(s.vjp, y, [x], v)
        actual = native(s.numpy, dx)
        assert actual.shape == shape
        np.testing.assert_array_equal(actual, seed * np.float32(2))


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


def test_vjp_import_and_layout_rules_have_independent_broadcast_gradient():
    a = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / np.float32(16)
    bias = np.array([[[0.25, -0.125]]], np.float32)
    seed = np.arange(1, 13, dtype=np.float32).reshape(3, 2, 2) / np.float32(8)
    with ops.Session(record=True) as s:
        x = native(s.tensor, a, requires_grad=True)
        b = native(s.tensor, bias, requires_grad=True)
        shifted = native(s.add, x, b)
        y = native(s.permute, shifted, (1, 0, 2))
        v = native(s.tensor, seed)
        dx, db = native(s.vjp, y, [x, b], v)
        expected_dx = np.transpose(seed, (1, 0, 2)).copy()
        expected_db = expected_dx.sum(axis=(0, 1), keepdims=True)
        actual_dx, actual_db = native(s.numpy, dx), native(s.numpy, db)
        assert actual_dx.shape == a.shape and actual_db.shape == bias.shape
        np.testing.assert_array_equal(actual_dx, expected_dx)
        np.testing.assert_array_equal(actual_db, expected_db)
