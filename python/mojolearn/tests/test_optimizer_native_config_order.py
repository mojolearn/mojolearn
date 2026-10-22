"""Fixed optimizer parameter ordering never enters mutable sorting callbacks."""
import builtins
import dis

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("operation", ["get_params", "set_params"])
def test_optimizer_parameter_names_ignore_mutable_sorted(monkeypatch, cls, operation):
    params = np.array([1.0, -0.5, 0.25], dtype="f4")
    owner = cls(params, numeric_mode="identical", resident=False)
    expected = owner.get_params()
    expected_names = tuple(sorted(expected))
    before = np.asarray(owner.params[0]).tobytes()
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
    output = result.assert_native()
    assert not calls
    actual = output if operation == "get_params" else owner.get_params()
    assert tuple(actual) == expected_names
    if operation == "get_params":
        assert all(actual[name] is expected[name] for name in expected_names)
    else:
        assert output is owner
        assert owner.lr == 0.002
        assert owner.t == 0
        assert np.asarray(owner.params[0]).tobytes() == before
