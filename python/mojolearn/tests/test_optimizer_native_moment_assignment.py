"""Moment assignment validates real Python exports before state publication."""
import gc
import weakref

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


def _set_moment(owner, name, value):
    # Audit the actual public property wrapper; the test controller is outside
    # product execution and never installed as a callback/provider.
    return audit_native_call(getattr(T._Optimizer, name).fset, owner, value)


def _snapshot(owner):
    state = audit_native_call(owner.state_dict).assert_native()
    return state["t"], tuple(np.asarray(state[name]).tobytes()
                             for name in ("exp_avg", "exp_avg_sq", "buf_initialized"))


@pytest.mark.parametrize("name", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("bad_kind", ["empty", "undersized", "dtype", "readonly"])
def test_moment_assignment_refuses_before_publishing_or_uploading(name, resident, bad_kind):
    parameter = np.arange(8, dtype="f4") / 8
    control_parameter = parameter.copy()
    owner = T.AdamW([parameter], resident=resident, numeric_mode="identical")
    control = T.AdamW([control_parameter], resident=resident, numeric_mode="identical")
    gradient = np.full(8, 0.25, dtype="f4")
    audit_native_call(owner.step, [gradient]).assert_native()
    audit_native_call(control.step, [gradient]).assert_native()
    before = _snapshot(owner)
    assert before == _snapshot(control)
    old_first, old_second = owner.exp_avg, owner.exp_avg_sq
    field_state = (owner._m, owner._v, owner._native_optimizer_owner, owner.t, bool(owner.resident_))
    parameter_bytes = parameter.tobytes()
    candidate = np.ones(0 if bad_kind == "empty" else 1 if bad_kind == "undersized" else 8,
                        dtype="f8" if bad_kind == "dtype" else "f4")
    if bad_kind == "readonly":
        candidate.flags.writeable = False
    candidate_bytes = candidate.tobytes()
    audit = _set_moment(owner, name, candidate)
    expected = TypeError if bad_kind == "dtype" else ValueError
    assert type(audit.exception) is expected, audit.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not audit.report()[key], audit.report()
    assert owner._m is field_state[0] and owner._v is field_state[1]
    assert owner._native_optimizer_owner is field_state[2]
    assert (owner.t, bool(owner.resident_)) == field_state[3:]
    assert owner.exp_avg is old_first and owner.exp_avg_sq is old_second
    assert _snapshot(owner) == before
    assert parameter.tobytes() == parameter_bytes and candidate.tobytes() == candidate_bytes
    # A native resident state must also be unchanged, not merely the Python
    # mirrors inspected above. The next public step must agree with an
    # untouched control owner; this is rollback coverage, not a math oracle.
    audit_native_call(owner.step, [gradient]).assert_native()
    audit_native_call(control.step, [gradient]).assert_native()
    assert parameter.tobytes() == control_parameter.tobytes()
    assert _snapshot(owner) == _snapshot(control)


@pytest.mark.parametrize("name", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("resident", [False, True])
def test_moment_assignment_retains_exact_borrowed_owner_and_checkpoint_isolation(name, resident):
    owner = T.AdamW([np.ones(8, dtype="f4")], resident=resident, numeric_mode="identical")
    audit_native_call(owner.step, [np.full(8, 0.25, dtype="f4")]).assert_native()
    base = np.arange(10, dtype="f4") / 16
    candidate = base[1:9]
    expected = candidate.tobytes()
    candidate_ref, base_ref = weakref.ref(candidate), weakref.ref(base)
    _set_moment(owner, name, candidate).assert_native()
    published = audit_native_call(getattr(T._Optimizer, name).fget, owner).assert_native()
    assert published is candidate
    assert published.tobytes() == expected
    checkpoint = audit_native_call(owner.state_dict).assert_native()
    saved = np.asarray(checkpoint[name])
    assert saved.tobytes() == expected and not np.shares_memory(saved, published)
    del base, candidate, published
    gc.collect()
    assert candidate_ref() is not None and base_ref() is not None
    held = audit_native_call(getattr(T._Optimizer, name).fget, owner).assert_native()
    assert held is candidate_ref() and held.tobytes() == expected
    held[0] = np.float32(77)
    assert saved.tobytes() == expected
    # The caller-owned array remains a valid owner after optimizer finalization.
    del owner
    gc.collect()
    assert held[0] == np.float32(77) and saved.tobytes() == expected
