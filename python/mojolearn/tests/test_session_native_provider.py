# SPDX-License-Identifier: Apache-2.0
"""Bounded public candidate Session constructor/lifetime proof; requires rebuild."""
import gc
import weakref

import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def test_session_constructor_uses_selected_native_registry_not_packed_entries(monkeypatch):
    entry = ops._SESSION_CONNECTIONS['identical']
    assert entry.native_provider is not None
    seen = []

    def forbidden(*args, **kwargs):
        seen.append(True)
        raise AssertionError('constructor entered packed CPython dispatch')

    assert callable(entry.open)
    monkeypatch.setattr(entry, 'open', forbidden)
    monkeypatch.setattr(entry, 'numeric_mode', forbidden)
    session = audit_native_call(ops.Session, record=True, numeric_mode=' identical ').assert_native()
    handle = audit_native_call(session.tensor, np.arange(6, dtype='f4').reshape(2, 3)).assert_native()
    np.testing.assert_array_equal(session.numpy(handle), np.arange(6, dtype='f4').reshape(2, 3))
    assert audit_native_call(session.close).assert_native() is None
    assert audit_native_call(session.close).assert_native() is None
    with pytest.raises(ValueError, match='closed'):
        session.numpy(handle)
    assert not seen


def test_session_finalizer_closes_exact_registry_after_last_owner():
    entry = ops._SESSION_CONNECTIONS['identical']
    session = audit_native_call(ops.Session, numeric_mode='identical').assert_native()
    tensor = session.tensor(np.ones((2, 3), 'f4'))
    sid = session._id
    ref = weakref.ref(session)
    del session
    gc.collect()
    assert ref() is None
    # Test-only raw native query establishes the actual registry was closed;
    # disappearance of the Python wrapper alone is not sufficient evidence.
    with pytest.raises(Exception, match='not open'):
        entry.binding.ops_shape(sid, tensor)


def test_session_rejects_handle_or_owner_replacement_and_remains_usable():
    first = ops.Session(numeric_mode='identical')
    second = ops.Session(numeric_mode='identical')
    original_id, original_owner = first._id, first._lifetime
    try:
        first._id = original_id + 1000
        with pytest.raises(TypeError, match='handle'):
            first.tensor(np.ones((1, 1), 'f4'))
        first._id = original_id
        first._lifetime = second._lifetime
        with pytest.raises(TypeError, match='handle'):
            first.tensor(np.ones((1, 1), 'f4'))
        first._lifetime = original_owner
        assert audit_native_call(first.tensor, np.ones((1, 1), 'f4')).assert_native() > 0
    finally:
        first._id, first._lifetime = original_id, original_owner
        first.close()
        second.close()


def test_session_refuses_changed_owner_hook_before_callback(monkeypatch):
    seen = []

    def forbidden(self, name, value):
        seen.append(name)
        raise AssertionError('Session attribute hook entered')

    monkeypatch.setattr(ops.Session, '__setattr__', forbidden)
    audit = audit_native_call(ops.Session, numeric_mode='identical')
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not seen
    assert not audit.report()['python_frames'], audit.report()


def test_session_refuses_foreign_constructor_protocols_and_replaced_execution_entry(monkeypatch):
    seen = []

    class ModeHook(str):
        def strip(self, *args):
            seen.append('mode')
            raise AssertionError('mode callback')

    class RecordHook:
        def __bool__(self):
            seen.append('record')
            raise AssertionError('record callback')

    for kwargs in ({'numeric_mode': ModeHook('identical')}, {'record': RecordHook()}):
        audit = audit_native_call(ops.Session, **kwargs)
        assert isinstance(audit.exception, (TypeError, ValueError)), audit.report()
        assert not audit.report()['python_frames'], audit.report()
    entry = ops._SESSION_CONNECTIONS['identical']

    def forbidden(*args):
        seen.append('entry')
        raise AssertionError('replaced execution entry')

    monkeypatch.setattr(entry, 'call', forbidden)
    audit = audit_native_call(ops.Session, numeric_mode='identical')
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not seen
