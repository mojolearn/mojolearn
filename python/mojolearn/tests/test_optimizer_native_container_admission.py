"""Mutable builtin names cannot admit Python container or scalar hooks."""
import builtins
import dis

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connection_refusal import assert_connection_refusal


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


def assert_native_refusal(audit, calls, expected=TypeError, message=None):
    assert type(audit.exception) is expected, audit.report()
    if message is not None:
        assert str(audit.exception) == message, audit.report()
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not audit.report()[key], audit.report()


def pin_observer_constructor(patch, name):
    # The auditor disassembles wrappers before the measured product call.
    # CPython dis constructs dict/set objects and then indexes/iterates them;
    # poisoning those observer objects fails before product execution. Pin
    # only dis's own globals, while builtins remains poisoned for the product.
    if name in ("dict", "set"):
        patch.setattr(dis, name, getattr(builtins, name), raising=False)


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
        pin_observer_constructor(patch, name)
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
    if operation == "accumulate":
        # The accumulation door reads the connected builtins first, so the
        # replaced list export refuses by name (commit 31fc85981) before the
        # container admission sees the foreign value. py_typed_call keeps the
        # TypeError class through the nested binding.
        assert_connection_refusal(audit, "list", calls)
        # With the setup-time export back, the foreign subclass itself is
        # still refused by admission without entering its hooks.
        audit = audit_native_call(method, *args, **kwargs)
        assert_native_refusal(audit, calls, TypeError,
                              "optimizer requires native buffers and lists/tuples; Python protocols and callbacks cannot execute")
    else:
        assert_native_refusal(audit, calls)


@pytest.mark.parametrize("name", ["list", "dict", "set", "frozenset", "memoryview"])
def test_loss_truth_never_admits_replaced_builtin_container(monkeypatch, name):
    calls = []
    cls = replacement(name, calls)
    value = cls()
    x = np.ones((2, 3), dtype="f4")
    target = np.array([0, 1], dtype="i4")
    with monkeypatch.context() as patch:
        pin_observer_constructor(patch, name)
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
    assert_connection_refusal(audit, "list", calls)
    state = audit_native_call(owner.state_dict).assert_native()
    assert state["t"] == 0
    assert np.asarray(state["exp_avg"]).tobytes() == np.zeros(3, dtype="f4").tobytes()
    assert np.asarray(state["exp_avg_sq"]).tobytes() == np.zeros(3, dtype="f4").tobytes()


@pytest.mark.parametrize("operation", ["construct", "set_params", "step", "step_accumulated",
                                       "accumulate-flat", "accumulate-nested"])
def test_native_optimizer_preparation_uses_retained_list_constructor(monkeypatch, operation):
    params = np.array([0.5, -0.25, 1.0], dtype="f4")
    gradient = np.array([0.125, -0.5, 0.25], dtype="f4")
    calls = []
    cls = replacement("list", calls)
    owner = T.AdamW([params.copy()], resident=False, numeric_mode="identical",
                    accumulation_steps=2 if operation == "step_accumulated" else 1)
    expected = T.AdamW([params.copy()], resident=False, numeric_mode="identical",
                       accumulation_steps=owner.accumulation_steps)
    if operation == "construct":
        method, args = T.AdamW, ([params.copy()],)
        kwargs = {"resident": False, "numeric_mode": "identical"}
        reference = None
    elif operation == "set_params":
        method, args, kwargs = owner.set_params, (), {"lr": 0.002}
        reference = expected.set_params(**kwargs)
    elif operation in ("step", "step_accumulated"):
        method, kwargs = getattr(owner, operation), {}
        args = ([gradient.copy()],) if operation == "step" else ([gradient.copy(), gradient.copy()], 1024)
        reference = getattr(expected, operation)(*args)
    else:
        pieces = [gradient.copy(), gradient.copy()]
        if operation == "accumulate-nested":
            pieces = [[value] for value in pieces]
        method, args = T.accumulate_grads, (pieces, 1024)
        kwargs = {"numeric_mode": "identical"}
        reference = T.accumulate_grads(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "list", cls)
        audit = audit_native_call(method, *args, **kwargs)
    # Refused before any state change; the clean call then runs once.
    assert_connection_refusal(audit, "list", calls)
    result = audit_native_call(method, *args, **kwargs).assert_native()
    if operation == "construct":
        assert np.asarray(result.params[0]).tobytes() == params.tobytes()
        assert result.t == 0
    elif operation == "set_params":
        assert result is owner and owner.lr == expected.lr and owner.t == 0
    elif operation.startswith("accumulate"):
        actual = result[0] if operation.endswith("nested") else result
        oracle = reference[0] if operation.endswith("nested") else reference
        assert np.asarray(actual).tobytes() == np.asarray(oracle).tobytes()
    else:
        assert result == reference
        assert owner.t == expected.t
        assert np.asarray(owner.params[0]).tobytes() == np.asarray(expected.params[0]).tobytes()
        state = audit_native_call(owner.state_dict).assert_native()
        expected_state = expected.state_dict()
        for name in ("exp_avg", "exp_avg_sq", "buf_initialized"):
            assert np.asarray(state[name]).tobytes() == np.asarray(expected_state[name]).tobytes()
