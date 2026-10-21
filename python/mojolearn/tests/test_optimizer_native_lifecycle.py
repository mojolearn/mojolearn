"""Native optimizer cleanup, independent release observations and refusal audits."""
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn._training_impl import _Optimizer, _optimizer_finalize_native
from mojolearn._verify_native_runtime import audit_native_call


def test_partial_optimizer_finalization_uses_no_python_getter(monkeypatch):
    owner = object.__new__(_Optimizer)
    def forbidden(*args, **kwargs):
        raise AssertionError('Python optimizer metadata getter executed')
    with monkeypatch.context() as patch:
        patch.setattr(_Optimizer, '__getattribute__', forbidden)
        # Obtain the public wrapper before the poison can intercept test work.
        audit_native_call(_Optimizer.__del__, owner).assert_native()
    assert vars(owner) == {}


def test_native_close_entry_called_once_with_exact_handle(monkeypatch):
    events = []
    owner = object.__new__(_Optimizer)
    fields = vars(owner)
    fields['_res'] = 17
    # A C builtin is an independent observer of the native dispatch. Actual
    # resident GPU release is separately tested below.
    fields['_res_binding'] = SimpleNamespace(optimizer_resident_close=events.append)
    def forbidden(*args, **kwargs):
        raise AssertionError('Python optimizer metadata getter executed')
    with monkeypatch.context() as patch:
        patch.setattr(_Optimizer, '__getattribute__', forbidden)
        audit_native_call(_Optimizer.__del__, owner).assert_native()
        audit_native_call(_Optimizer.__del__, owner).assert_native()
    assert events == [17]
    assert fields['_res'] is None and fields['_res_binding'] is None


def test_python_close_callback_is_refused_without_invocation():
    called = []
    owner = object.__new__(_Optimizer)
    fields = vars(owner)
    fields['_res'] = 17
    fields['_res_binding'] = SimpleNamespace(optimizer_resident_close=lambda value: called.append(value))
    try:
        audit = audit_native_call(_optimizer_finalize_native, owner)
        assert audit.exception is not None and 'compiled close' in str(audit.exception)
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not audit.report()[key], audit.report()
        assert called == [] and fields['_res'] == 17
    finally:
        fields['_res'] = None  # Test-only deliberately invalid resource record.


def test_real_resident_resource_is_released_by_native_finalizer():
    if _backend.vendor() == 'cpu':
        pytest.skip('CPU optimizer has no resident allocation to release')
    binding = _backend.binding('_mojolearn_training', 'identical')
    handle = binding.optimizer_resident_open(8)
    owner = object.__new__(_Optimizer)
    fields = vars(owner)
    fields.update(_res=handle, _res_binding=binding)
    try:
        audit_native_call(_Optimizer.__del__, owner).assert_native()
        assert fields['_res'] is None
        m = np.empty(8, 'f4')
        v = np.empty(8, 'f4')
        with pytest.raises(Exception, match='handle|closed|not open|size'):
            binding.optimizer_resident_download(handle, m.ctypes.data, v.ctypes.data, 8)
        audit_native_call(_Optimizer.__del__, owner).assert_native()
    finally:
        if fields.get('_res') is not None:
            binding.optimizer_resident_close(handle)
            fields['_res'] = None
