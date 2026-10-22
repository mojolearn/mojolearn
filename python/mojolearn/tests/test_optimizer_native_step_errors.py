"""A second native call boundary preserves the optimizer's public errors."""
import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("optimizer", ["SGD", "Adam", "AdamW"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_native_accumulated_step_preserves_count_error_and_state(optimizer, count):
    param = np.arange(8, dtype="f4") / 8
    before = param.tobytes()
    owner = getattr(T, optimizer)([param], accumulation_steps=2, numeric_mode="identical")
    gradient = np.ones(8, dtype="f4")
    result = audit_native_call(owner.step_accumulated, [[gradient] for _ in range(count)], tokens=512)
    assert type(result.exception) is ValueError, result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    assert param.tobytes() == before
    assert owner.t == 0


@pytest.mark.parametrize("case,category", [("count", ValueError), ("shape", ValueError), ("dtype", TypeError)])
def test_native_step_preserves_gradient_refusal_category(case, category):
    param = np.ones(8, dtype="f4")
    owner = T.AdamW([param], numeric_mode="identical")
    gradient = np.ones(8, dtype="f8" if case == "dtype" else "f4")
    gradients = [] if case == "count" else [gradient.reshape(2, 4) if case == "shape" else gradient]
    result = audit_native_call(owner.step, gradients)
    assert type(result.exception) is category, result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    assert np.array_equal(param, np.ones(8, dtype="f4"))
    assert owner.t == 0


def test_native_step_refuses_replaced_compiled_entry(monkeypatch):
    owner = T.AdamW([np.ones(8, dtype="f4")], numeric_mode="identical")
    calls = []
    def hook(*args):
        calls.append(args)
    monkeypatch.setitem(T._OPT_CONTEXT["bindings"]["identical"], "optimizer_door_step", hook)
    result = audit_native_call(owner.step, [np.ones(8, dtype="f4")])
    assert type(result.exception) is TypeError, result.report()
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
