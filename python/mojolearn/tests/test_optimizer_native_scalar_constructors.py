"""Native optimizer scalar conversion ignores mutable builtin constructors."""
import builtins
import dis

import numpy as np
import pytest

from mojolearn import _backend, _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


def poisoned_call(monkeypatch, name, operation, *args, **kwargs):
    calls = []
    def hook(*args, **kwargs):
        calls.append(name)
        raise AssertionError("mutable scalar constructor entered")
    with monkeypatch.context() as patch:
        # Keep controller disassembly independent of the product's live
        # builtin field. Neither the public call nor its native callees see
        # this testing-only module global.
        patch.setattr(dis, name, getattr(builtins, name), raising=False)
        patch.setattr(builtins, name, hook)
        audit = audit_native_call(operation, *args, **kwargs)
    result = audit.assert_native()
    assert not calls
    return result


def compare_state(actual, expected):
    left = audit_native_call(actual.state_dict).assert_native()
    right = expected.state_dict()
    assert left["t"] == right["t"]
    for name in ("exp_avg", "exp_avg_sq", "buf_initialized"):
        assert np.asarray(left[name]).tobytes() == np.asarray(right[name]).tobytes()
    assert np.asarray(actual.params[0]).tobytes() == np.asarray(expected.params[0]).tobytes()


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("name", ["float", "int", "bool"])
def test_optimizer_constructor_uses_connected_scalar_types(monkeypatch, cls, name):
    params = np.array([0.5, -0.25, 1.0], dtype="f4")
    kwargs = dict(lr=0.0125, weight_decay=0.02, accumulation_steps=2,
                  numeric_mode="identical", resident=False, maximize=False)
    if cls is T.SGD:
        kwargs.update(momentum=0.1, nesterov=True)
    else:
        kwargs.update(betas=(0.8, 0.95))
    expected = cls(params.copy(), **kwargs)
    actual = poisoned_call(monkeypatch, name, cls, params.copy(), **kwargs)
    compare_state(actual, expected)
    left, right = actual.get_params(), expected.get_params()
    assert tuple(left) == tuple(right)
    for key in left:
        if key != "params":
            assert left[key] == right[key]


@pytest.mark.parametrize("operation", ["step", "step_accumulated", "clip", "load", "accumulate"])
@pytest.mark.parametrize("name", ["float", "int", "bool"])
def test_optimizer_operations_use_connected_scalar_constructors(monkeypatch, operation, name):
    params = np.array([0.5, -0.25, 1.0], dtype="f4")
    gradient = np.array([0.125, -0.5, 0.25], dtype="f4")
    kwargs = dict(numeric_mode="identical", resident=False,
                  accumulation_steps=2 if operation == "step_accumulated" else 1)
    actual, expected = [T.AdamW(params.copy(), **kwargs) for _ in range(2)]
    if operation in ("step", "step_accumulated"):
        method = getattr(actual, operation)
        args = (gradient.copy(),) if operation == "step" else ([gradient.copy(), gradient.copy()], 1024)
        reference = getattr(expected, operation)(*args)
        result = poisoned_call(monkeypatch, name, method, *args)
        assert result == reference
        compare_state(actual, expected)
    elif operation == "clip":
        value, reference = gradient.copy(), gradient.copy()
        norm = T.clip_grad_norm_([reference], 0.25, numeric_mode="identical")
        result = poisoned_call(monkeypatch, name, T.clip_grad_norm_, [value], 0.25,
                               numeric_mode="identical")
        assert result == norm and value.tobytes() == reference.tobytes()
    elif operation == "load":
        expected.step(gradient.copy())
        state = expected.state_dict()
        poisoned_call(monkeypatch, name, actual.load_state_dict, state)
        # load_state_dict intentionally leaves parameter buffers unchanged.
        snapshot = audit_native_call(actual.state_dict).assert_native()
        assert snapshot["t"] == state["t"]
        for key in ("exp_avg", "exp_avg_sq", "buf_initialized"):
            assert np.asarray(snapshot[key]).tobytes() == np.asarray(state[key]).tobytes()
        assert np.asarray(actual.params[0]).tobytes() == params.tobytes()
    else:
        values = [gradient.copy(), gradient.copy()]
        reference = T.accumulate_grads(values, 1024, numeric_mode="identical")
        result = poisoned_call(monkeypatch, name, T.accumulate_grads, values, 1024,
                               numeric_mode="identical")
        assert np.asarray(result).tobytes() == np.asarray(reference).tobytes()


@pytest.mark.parametrize("operation", ["reshape", "param_device"])
def test_device_scalar_controls_use_retained_native_integer(monkeypatch, operation):
    if _backend.vendor() == "cpu":
        pytest.skip("device scalar controls require a GPU training pool")
    source = np.arange(12, dtype="f4").reshape(4, 3)
    if operation == "reshape":
        owner = T.train_to_device(source, numeric_mode="identical")
        method, args = owner.reshape, ((2, 6),)
    else:
        owner = T.AdamW([source], numeric_mode="identical")
        owner.params_to_device()
        method, args = owner.param_device, (0,)
    output = poisoned_call(monkeypatch, "int", method, *args)
    result = audit_native_call(output.to_host).assert_native()
    assert np.asarray(result).tobytes() == source.tobytes()
    assert result.shape == ((2, 6) if operation == "reshape" else source.shape)
    audit_native_call(T.TrainDeviceArray.__del__, output).assert_native()
    if operation == "reshape":
        audit_native_call(T.TrainDeviceArray.__del__, owner).assert_native()
