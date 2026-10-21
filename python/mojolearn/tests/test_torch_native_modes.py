# SPDX-License-Identifier: Apache-2.0
"""Real Torch mode hooks cannot be entered through native buffer admission."""
import os

import numpy as np
import pytest

try:
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception as error:
    if os.environ.get("MOJOLEARN_REQUIRE_TORCH") == "1":
        raise
    pytest.skip(f"Torch is unavailable: {error}", allow_module_level=True)

from mojolearn import LinearRegression, Ridge, Sequential, ops
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def native_ridge_fit(model, x, y):
    return ENTRY(model, "ridge_fit", (x, y, None), CONTEXT)


def native_ridge_predict(model, x):
    return ENTRY(model, "predict", (x,), CONTEXT)


def _mode(kind, calls):
    if kind == "function":
        class PoisonFunction(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                calls.append("torch_function")
                raise AssertionError("Python TorchFunctionMode reached")
        return PoisonFunction()

    class PoisonDispatch(TorchDispatchMode):
        @classmethod
        def _should_skip_dynamo(cls):
            return False

        def __torch_dispatch__(self, *args, **kwargs):
            calls.append("torch_dispatch")
            raise AssertionError("Python TorchDispatchMode reached")
    return PoisonDispatch()


def _clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


def _state(parameter=False):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8
    bias = torch.tensor([0.25, -0.5])
    if parameter:
        weight, bias = torch.nn.Parameter(weight), torch.nn.Parameter(bias)
    return {"0.weight": weight, "0.bias": bias}


@pytest.mark.parametrize("kind", ["function", "dispatch"])
@pytest.mark.parametrize("operation", ["classical_fit", "classical_predict", "sequential_weights", "session_weights"])
def test_active_torch_mode_is_refused_before_any_hook(kind, operation):
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 8
    y = torch.arange(6, dtype=torch.float32) / 3
    state = _state()
    model = Ridge()
    if operation == "classical_predict":
        _clean(audit_native_call(native_ridge_fit, model, x, y))
    sequential = Sequential().linear(2)
    calls = []
    mode = _mode(kind, calls)
    if operation == "session_weights":
        with ops.Session() as session:
            weight = session.tensor(np.zeros((3, 2), dtype="f4"))
            bias = session.tensor(np.zeros((1, 2), dtype="f4"))
            mapping = [(weight, "0.weight", "transpose"), (bias, "0.bias")]
            with mode:
                result = audit_native_call(session.load_weights, state, mapping)
            _clean(result, TypeError)
    else:
        with mode:
            if operation == "classical_fit":
                result = audit_native_call(native_ridge_fit, model, x, y)
            elif operation == "classical_predict":
                result = audit_native_call(native_ridge_predict, model, x)
            else:
                result = audit_native_call(sequential.load_state_dict, state)
        _clean(result, TypeError)
    assert calls == []
    assert "mode" in str(result.exception).lower()


@pytest.mark.parametrize("parameter", [False, True])
def test_ordinary_torch_tensors_and_parameters_remain_native(parameter):
    state = _state(parameter)
    expected = np.concatenate([state["0.weight"].detach().numpy().T.ravel(),
                               state["0.bias"].detach().numpy()])
    sequential = Sequential().linear(2)
    _clean(audit_native_call(sequential.load_state_dict, state))
    np.testing.assert_array_equal(np.asarray(sequential.params_).view("u4"), expected.view("u4"))
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 8
    y = torch.arange(8, dtype=torch.float32) / 3
    if parameter:
        x, y = torch.nn.Parameter(x), torch.nn.Parameter(y)
    reference = Ridge().fit(x.detach().numpy(), y.detach().numpy())
    wanted = np.asarray(reference.predict(x.detach().numpy())).view("u4")
    model = Ridge()
    _clean(audit_native_call(native_ridge_fit, model, x, y))
    got = _clean(audit_native_call(native_ridge_predict, model, x))
    np.testing.assert_array_equal(np.asarray(got).view("u4"), wanted)


def test_native_torch_descriptors_bypass_instance_python_method_shadows():
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 8
    y = torch.arange(8, dtype=torch.float32) / 3
    state = _state()
    calls = []

    def poison(*args, **kwargs):
        calls.append("shadow")
        raise AssertionError("instance Tensor method shadow reached")

    x.detach = poison
    x.numpy = poison
    state["0.weight"].data_ptr = poison
    state["0.weight"].stride = poison
    _clean(audit_native_call(native_ridge_fit, Ridge(), x, y))
    _clean(audit_native_call(Sequential().linear(2).load_state_dict, state))
    assert calls == []


@pytest.mark.parametrize("kind", ["function", "dispatch"])
def test_samba_native_inputs_do_not_touch_torch_and_tensor_refusal_stays_native(kind):
    # Samba's present public buffer contract admits ndarray/Array, not Torch
    # tensors. Preserve that explicit boundary; do not claim Tensor support.
    from mojolearn import SambaConfig, SambaStack
    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11, dropout=0)
    weights = {name: np.full(shape, 0.025, dtype="f4") for name, shape in config.registry()}
    model = SambaStack(config, weights=weights)
    tokens = np.array([[1, 3], [2, 4]], dtype="i4")
    tensor = torch.from_numpy(tokens)
    calls = []
    with _mode(kind, calls):
        good = audit_native_call(model.forward, tokens)
        refused = audit_native_call(model.forward, tensor)
    _clean(good)
    _clean(refused, TypeError)
    assert calls == []
