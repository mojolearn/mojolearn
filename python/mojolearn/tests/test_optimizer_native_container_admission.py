"""Mutable builtin names cannot admit Python container or scalar hooks."""
import builtins

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


def replacement(name, calls):
    def hook(*args, **kwargs):
        calls.append(name)
        raise AssertionError("mutable builtin container hook entered")
    if name == "memoryview":
        return type("ForeignView", (), {"obj": property(hook)})
    # Inherit the ordinary native constructor so the testing auditor's own
    # list/dict construction continues to work. Product input remains an
    # unsupported Python subclass with explicit conversion/protocol hooks.
    return type("ForeignContainer", (getattr(builtins, name),), {
        "__float__": hook, "__int__": hook, "__bool__": hook,
        "__len__": hook, "__getitem__": hook,
    })


def assert_native_refusal(audit, calls):
    assert audit.exception is not None, audit.report()
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not audit.report()[key], audit.report()


@pytest.mark.parametrize("name", ["list", "dict", "bytearray", "memoryview"])
@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
def test_scalar_configuration_never_admits_replaced_builtin_containers(monkeypatch, name, operation):
    calls = []
    cls = replacement(name, calls)
    value = cls()
    x = np.ones((2, 3), dtype="f4")
    weight = np.ones(3, dtype="f4")
    args = (x, weight) if operation.endswith("forward") else (x.copy(), x, weight)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, name, cls)
        audit = audit_native_call(getattr(T, operation), *args, eps=value,
                                  numeric_mode="identical")
    assert_native_refusal(audit, calls)


@pytest.mark.parametrize("operation", ["accumulate", "embedding", "cross_entropy"])
def test_tensor_preparation_never_admits_replaced_list_type(monkeypatch, operation):
    calls = []
    cls = replacement("list", calls)
    value = cls()
    if operation == "accumulate":
        method, args, kwargs = T.accumulate_grads, (value, 512), {}
    elif operation == "embedding":
        method, args, kwargs = T.embedding_forward, (np.ones((3, 2), dtype="f4"), value), {}
    else:
        method, args, kwargs = T.cross_entropy, (np.ones((2, 3), dtype="f4"), value), {}
    kwargs["numeric_mode"] = "identical"
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "list", cls)
        audit = audit_native_call(method, *args, **kwargs)
    assert_native_refusal(audit, calls)


@pytest.mark.parametrize("name", ["list", "dict", "set", "frozenset", "memoryview"])
def test_loss_truth_never_admits_replaced_builtin_container(monkeypatch, name):
    calls = []
    cls = replacement(name, calls)
    value = cls()
    x = np.ones((2, 3), dtype="f4")
    target = np.array([0, 1], dtype="i4")
    with monkeypatch.context() as patch:
        patch.setattr(builtins, name, cls)
        audit = audit_native_call(T.cross_entropy, x, target, return_grad=value,
                                  numeric_mode="identical")
    assert_native_refusal(audit, calls)


def test_optimizer_state_uses_retained_list_constructor(monkeypatch):
    owner = T.AdamW(np.ones(3, dtype="f4"), numeric_mode="identical", resident=False)
    calls = []
    cls = replacement("list", calls)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "list", cls)
        audit = audit_native_call(owner.state_dict)
    state = audit.assert_native()
    assert not calls
    assert state["t"] == 0
    assert np.asarray(state["exp_avg"]).tobytes() == np.zeros(3, dtype="f4").tobytes()
    assert np.asarray(state["exp_avg_sq"]).tobytes() == np.zeros(3, dtype="f4").tobytes()
