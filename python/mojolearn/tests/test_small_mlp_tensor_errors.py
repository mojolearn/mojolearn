"""Actual public tensor iteration failures: no fake bindings or patched Torch.

Requires the selected native SmallMLP owner and installed Torch. Missing either
is a failure, not a skipped qualification. No GPU computation is needed: these
cases fail at native iteration before training/prediction.
"""
import pytest
import torch
from mojolearn import SmallMLPTrainer
import numpy as np

__mojolearn_user_boundary__ = True


def _trainer():
    shapes = ((16, 8), (16,), (3, 16), (3,))
    weights = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    return SmallMLPTrainer(*weights, data_schedule={"dataset": "tensor-error-contract"})


def test_unsupported_gradient_container_is_a_type_refusal():
    model = _trainer()
    with pytest.raises(TypeError):
        model.apply_gradients(object())
    assert model.step_ == 0


def test_admitted_tensor_unbind_failure_keeps_exception_type_and_message():
    # Scalar is an exact admitted Tensor, but native unbind(0) cannot operate
    # on rank zero. Compare the installed Torch operation's own exception.
    value = torch.tensor(1.0)
    with pytest.raises(Exception) as expected:
        value.unbind(0)
    model = _trainer()
    with pytest.raises(type(expected.value)) as actual:
        model.apply_gradients(value)
    assert str(actual.value) == str(expected.value)
    assert model.step_ == 0


def test_tensor_mode_refusal_remains_specific_and_does_not_call_user_hook():
    from torch.overrides import TorchFunctionMode
    calls = []
    class Mode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            calls.append(func)
            raise AssertionError("user Torch mode executed")
    value = torch.ones((4, 1))
    model = _trainer()
    with Mode(), pytest.raises(TypeError, match="TorchFunctionMode"):
        model.apply_gradients(value)
    assert calls == []
    assert model.step_ == 0
