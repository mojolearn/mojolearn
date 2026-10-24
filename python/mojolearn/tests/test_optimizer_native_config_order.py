"""Fixed optimizer parameter ordering never enters a replaced sorting callback."""
import builtins
import dis

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connection_refusal import assert_connection_refusal


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("operation", ["get_params", "set_params"])
def test_optimizer_parameter_names_ignore_mutable_sorted(monkeypatch, cls, operation):
    params = np.array([1.0, -0.5, 0.25], dtype="f4")
    owner = cls(params, numeric_mode="identical", resident=False)
    expected = owner.get_params()
    expected_names = tuple(sorted(expected))
    before = np.asarray(owner.params[0]).tobytes()
    lr_before = owner.lr
    method = getattr(owner, operation)
    kwargs = {"lr": 0.002} if operation == "set_params" else {}
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("mutable sorting function entered")
    with monkeypatch.context() as patch:
        # Auditor disassembly uses sorted before tracing the operation.
        # Preserve only that testing global; the product sees the poison.
        patch.setattr(dis, "sorted", builtins.sorted, raising=False)
        patch.setattr(builtins, "sorted", forbidden)
        result = audit_native_call(method, **kwargs)
    # The replaced connected builtin refuses before any hook or state change;
    # with the setup-time export back the fixed order is native.
    assert_connection_refusal(result, "sorted", calls)
    assert owner.lr == lr_before and owner.t == 0
    output = audit_native_call(method, **kwargs).assert_native()
    actual = output if operation == "get_params" else owner.get_params()
    assert tuple(actual) == expected_names
    if operation == "get_params":
        assert all(actual[name] is expected[name] for name in expected_names)
    else:
        assert output is owner
        assert owner.lr == 0.002
        assert owner.t == 0
        assert np.asarray(owner.params[0]).tobytes() == before
