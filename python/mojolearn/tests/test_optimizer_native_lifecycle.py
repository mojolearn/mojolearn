"""Actual optimizer owner finalization, provider pins and refusal boundaries.

Exact native handle release/failed-close retry expectations live in the direct
optimizer_owner CPU/GPU suites. Python retains the actual wrapper, metadata
hook, callback admission, exception and public buffer lifetime contracts.
"""
import gc
import sys

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


def test_partial_optimizer_finalization_uses_no_python_getter(monkeypatch):
    owner = object.__new__(T._Optimizer)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python optimizer metadata getter executed")

    with monkeypatch.context() as patch:
        patch.setattr(T._Optimizer, "__getattribute__", forbidden)
        audit_native_call(T._Optimizer.__del__, owner).assert_native()
    assert vars(owner) == {}


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
def test_real_native_owner_finalization_is_idempotent_and_keeps_public_buffers(monkeypatch, cls):
    owner = cls(np.array([1., -2., .5, .25], dtype="f4"), lr=.01,
                numeric_mode="identical", resident=True)
    audit_native_call(owner.step, np.array([.1, -.2, .3, .4], dtype="f4")).assert_native()
    assert bool(owner.resident_) == (_backend.vendor() != "cpu")
    parameter = owner.params[0]
    parameter_bytes = np.asarray(parameter).tobytes()
    native_owner = vars(owner)["_native_optimizer_owner"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Python optimizer metadata getter executed")

    with monkeypatch.context() as patch:
        patch.setattr(T._Optimizer, "__getattribute__", forbidden)
        audit_native_call(T._Optimizer.__del__, owner).assert_native()
        audit_native_call(T._Optimizer.__del__, owner).assert_native()
    assert not owner.resident_
    assert vars(owner)["_native_optimizer_owner"] is native_owner
    assert np.asarray(parameter).tobytes() == parameter_bytes
    del owner, native_owner
    gc.collect()
    assert np.asarray(parameter).tobytes() == parameter_bytes


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
def test_factory_callback_refusal_preserves_state_and_retained_owner_cleanup(monkeypatch, cls):
    live = cls(np.ones(4, dtype="f4"), numeric_mode="identical", resident=True)
    audit_native_call(live.step, np.ones(4, dtype="f4")).assert_native()
    fresh = cls(np.ones(4, dtype="f4"), numeric_mode="identical", resident=True)
    native_owner = vars(fresh)["_native_optimizer_owner"]
    parameter_bytes = np.asarray(fresh.params[0]).tobytes()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Python optimizer descriptor callback executed")

    with monkeypatch.context() as patch:
        patch.setitem(T._OPT_CONTEXT["bindings"]["identical"], "optimizer_native_api_v1", forbidden)
        result = audit_native_call(fresh.step, np.ones(4, dtype="f4"))
        assert type(result.exception) is TypeError, result.report()
        for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
            assert not result.report()[key], result.report()
        assert fresh.t == 0
        assert vars(fresh)["_native_optimizer_owner"] is native_owner
        assert np.asarray(fresh.params[0]).tobytes() == parameter_bytes
        # The live owner retained its real provider/driver before poisoning.
        # Finalization must use that native owner without looking up a callback.
        audit_native_call(T._Optimizer.__del__, live).assert_native()
        audit_native_call(T._Optimizer.__del__, live).assert_native()
        assert not calls
    assert not live.resident_
    # Refusal also releases the operation lease: the same actual owner retries.
    audit_native_call(fresh.step, np.ones(4, dtype="f4")).assert_native()
    assert fresh.t == 1
    audit_native_call(T._Optimizer.__del__, fresh).assert_native()


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
def test_native_finalizer_preserves_pending_python_exception(cls):
    owner = cls(np.ones(4, dtype="f4"), numeric_mode="identical", resident=True)
    audit_native_call(owner.step, np.ones(4, dtype="f4")).assert_native()
    pending = RuntimeError("caller exception must survive native owner cleanup")
    with pytest.raises(RuntimeError) as caught:
        try:
            raise pending
        finally:
            assert sys.exception() is pending
            audit_native_call(T._Optimizer.__del__, owner).assert_native()
            assert sys.exception() is pending
    assert caught.value is pending
    assert not owner.resident_
    audit_native_call(T._Optimizer.__del__, owner).assert_native()
