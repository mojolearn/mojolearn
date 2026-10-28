"""Bounded LayerNorm whole-public-operation controls; requires rebuilt owner."""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


def test_layernorm_constructor_backward_and_finished_gradients_are_native():
    layer = audit_native_call(ml.LayerNorm, (2, 3), numeric_mode='identical').assert_native()
    np.testing.assert_array_equal(layer.weight, np.ones((2, 3), dtype='f4'))
    np.testing.assert_array_equal(layer.bias, np.zeros((2, 3), dtype='f4'))
    with pytest.raises(RuntimeError, match='call forward first'):
        layer.backward(np.ones((4, 2, 3), dtype='f4'))
    x = np.arange(24, dtype='f4').reshape(4, 2, 3) / 24
    y = audit_native_call(layer.forward, x).assert_native()
    dy = np.linspace(-1, 1, 24, dtype='f4').reshape(x.shape)
    dx = audit_native_call(layer.backward, dy).assert_native()
    expected = ml.layer_norm_backward(dy, x, (2, 3), layer.weight, layer.bias, layer.eps, 'identical')
    assert y.shape == x.shape
    for actual, wanted in zip((dx, layer.weight_grad, layer.bias_grad), expected):
        assert actual.tobytes() == wanted.tobytes()
    assert layer.weight_grad.shape == layer.bias_grad.shape == (2, 3)


def test_layernorm_refuses_scalar_hook_before_it_runs():
    seen = []

    class FloatHook:
        def __float__(self):
            seen.append(True)
            raise AssertionError('Python conversion hook ran')

    with pytest.raises(TypeError):
        ml.LayerNorm(3, eps=FloatHook(), numeric_mode='identical')
    assert not seen
