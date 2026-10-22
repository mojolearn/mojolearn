# SPDX-License-Identifier: Apache-2.0
"""Whole resident lifetime operations preserve ownership and release failures."""
from array import array
import builtins
import json
import os
import subprocess
import sys
from types import MappingProxyType
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
import pytest

from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._buffer import _native
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call


def _refused(audit):
    report = audit.report()
    assert type(audit.exception) is TypeError, report
    for field in ('wrapper_work', 'python_frames', 'forbidden_native_calls'):
        assert not report[field], report


@pytest.fixture
def owner():
    live = _native('msel_live')
    before = int(live())
    result = _Resident()
    try:
        yield result
    finally:
        result.close()
        assert int(live()) == before


def _populate(owner):
    values = np.arange(12, dtype='f4').reshape(4, 3)
    indices = np.array([2, 0], dtype='i8')
    # Input preparation/take is separate pending migration. This controller
    # calls it only to establish actual store ownership before auditing life.
    owner.take(values, indices)
    return values, indices


@pytest.mark.parametrize('subclass', [False, True])
def test_constructor_reinitialization_and_context_lifetime_are_native(subclass):
    class Ordinary(_Resident):
        pass
    cls = Ordinary if subclass else _Resident
    live = _native('msel_live')
    before = int(live())
    result = audit_native_call(cls).assert_native()
    try:
        result.annotation = {'unchanged': (1, 2)}
        values, indices = _populate(result)
        assert int(live()) == before + 2
        assert audit_native_call(result.__init__).assert_native() is None
        assert result._entries == {} and int(live()) == before
        assert result.annotation == {'unchanged': (1, 2)}
        _populate(result)
        assert audit_native_call(result.__enter__).assert_native() is result
        assert audit_native_call(result.__exit__, None, None, None).assert_native() is False
        assert int(live()) == before
        assert audit_native_call(result.close).assert_native() is None
    finally:
        result.close()


@pytest.mark.parametrize('kind', ['numpy', 'list', 'tuple', 'array', 'memoryview', 'Array', 'zero'])
@pytest.mark.parametrize('operation', ['release', 'close', 'reinit'])
def test_native_lifetime_preserves_supported_buffer_owners(owner, kind, operation):
    values = np.arange(12, dtype='f4').reshape(4, 3)
    if kind == 'list': values = values.tolist()
    elif kind == 'tuple': values = tuple(tuple(row) for row in values.tolist())
    elif kind == 'array': values = array('f', range(12))
    elif kind == 'memoryview': values = memoryview(values)
    elif kind == 'Array': values = Array.from_list(values.tolist(), '<f4')
    elif kind == 'zero': values = np.empty((0, 3), dtype='f4')
    owner._entry(values)
    retained = values
    previous = owner._entries
    snapshot = dict(previous)
    if operation == 'release':
        audit_native_call(owner.release, values).assert_native()
    elif operation == 'close':
        audit_native_call(owner.close).assert_native()
    else:
        audit_native_call(owner.__init__).assert_native()
    assert owner._entries == {} and retained is values
    if operation == 'release': assert owner._entries is previous
    else: assert owner._entries is not previous and previous == snapshot


def _frame(kind):
    result = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    if kind == 'series': return result['a']
    if kind == 'nullable': return result.astype('Float64')
    if kind == 'nullable_integer': return result.astype('Int64')
    if kind == 'object_numeric': return result.astype(object)
    if kind == 'strings': result.index = ['x', 'y']
    elif kind == 'categorical': result.index = pd.CategoricalIndex(['x', 'y'])
    elif kind == 'datetime': result.index = pd.date_range('2020-01-01', periods=2)
    elif kind == 'timezone': result.index = pd.date_range('2020-01-01', periods=2, tz='UTC')
    elif kind == 'zoneinfo': result.index = pd.date_range('2020-01-01', periods=2, tz='America/New_York')
    elif kind == 'dateutil_timezone': result.index = pd.date_range('2020-01-01', periods=2, tz='dateutil/America/New_York')
    elif kind == 'dateutil_offset':
        from dateutil.tz import tzoffset
        result.index = pd.date_range('2020-01-01', periods=2, tz=tzoffset('offset', 3600))
    elif kind == 'timedelta': result.index = pd.timedelta_range('1 day', periods=2)
    elif kind == 'period': result.index = pd.period_range('2020', periods=2, freq='Y')
    elif kind == 'interval': result.index = pd.IntervalIndex.from_breaks([0, 1, 2])
    elif kind == 'multi': result.index = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)])
    elif kind == 'calendar_offset': result.index = pd.date_range('2020-01-01', periods=2, freq=pd.DateOffset(months=1))
    if kind != 'numeric':
        result.index.get_loc(result.index[0])
        result.index.get_indexer(result.index)
    return result


@pytest.mark.parametrize('kind', ['numeric', 'series', 'nullable', 'nullable_integer', 'object_numeric',
                                  'strings', 'categorical', 'datetime', 'timezone', 'timedelta',
                                  'period', 'interval', 'multi', 'calendar_offset', 'zoneinfo',
                                  'dateutil_timezone', 'dateutil_offset'])
@pytest.mark.parametrize('operation', ['close', 'reinit'])
def test_native_lifetime_preserves_ordinary_pandas_owner_graphs(owner, kind, operation):
    frame = _frame(kind)
    before = frame.copy(deep=True)
    owner._entry(frame)
    assert audit_native_call(owner.close if operation == 'close' else owner.__init__).assert_native() is None
    assert owner._entries == {}
    if kind == 'series': pd.testing.assert_series_equal(frame, before)
    else: pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('kind', ['float_columns', 'int_columns', 'string_index'])
@pytest.mark.parametrize('operation', ['close', 'reinit'])
def test_native_lifetime_preserves_ordinary_arrow_owners(owner, kind, operation):
    # These are existing supported inputs, not an optional feature to delete.
    # Until native Arrow ownership admission exists, these cases deliberately
    # keep the resident lifecycle draft from qualifying for activation.
    import pyarrow
    frame = pd.DataFrame({'x': [1., 2.], 'y': [3., 4.]})
    if kind == 'float_columns': frame = frame.astype('float64[pyarrow]')
    elif kind == 'int_columns': frame = frame.astype('int64[pyarrow]')
    else: frame.index = pd.Index(pd.array(['x', 'y'], dtype='string[pyarrow]'))
    before = frame.copy(deep=True)
    owner._entry(frame)
    try:
        audit_native_call(owner.close if operation == 'close' else owner.__init__).assert_native()
        pd.testing.assert_frame_equal(frame, before)
    finally:
        # Controller cleanup also works while exercising an incomplete draft.
        # It does not turn the audited assertion above into a passing case.
        for _, _, sid in tuple(owner._entries.values()):
            if sid >= 0: _native('msel_free')(sid)
        owner._entries.clear()


@pytest.mark.parametrize('operation', ['close', 'reinit'])
def test_native_lifetime_refuses_hidden_arrow_finalizers_without_replacement(owner, operation):
    import pyarrow as pa
    calls = []
    class Foreign:
        def __del__(self): calls.append('finalizer')
    values = np.array([1., 2.], dtype='f8')
    foreign = Foreign()
    buffer = pa.foreign_buffer(values.ctypes.data, values.nbytes, base=foreign)
    array = pa.Array.from_buffers(pa.float64(), 2, [None, buffer])
    frame = pd.DataFrame({'x': pd.arrays.ArrowExtensionArray(array)})
    owner._entry(frame)
    before = dict(owner._entries)
    try:
        _refused(audit_native_call(owner.close if operation == 'close' else owner.__init__))
        assert calls == [] and owner._entries == before
    finally:
        for _, _, sid in tuple(owner._entries.values()):
            if sid >= 0: _native('msel_free')(sid)
        owner._entries.clear()


@pytest.mark.parametrize('operation', ['close', 'reinit', 'exit'])
def test_native_cleanup_attempts_remaining_slots_and_preserves_first_failure(owner, operation):
    values, indices = _populate(owner)
    third = np.array([7, 8], dtype='i8')
    owner._entry(third)
    free = _native('msel_free')
    live = _native('msel_live')
    before = int(live())
    first = owner._entries[id(values)][2]
    last = owner._entries[id(third)][2]
    free(first)
    free(last)
    method = owner.close if operation == 'close' else owner.__init__ if operation == 'reinit' else owner.__exit__
    audit = audit_native_call(method)
    report = audit.report()
    assert type(audit.exception) is Exception, report
    assert str(audit.exception) == f'msel store: id {first} is not live'
    for field in ('wrapper_work', 'python_frames', 'forbidden_native_calls'):
        assert not report[field], report
    assert int(live()) == before - 3
    assert set(owner._entries) == {id(values), id(third)}
    # Retain failed entries for diagnosis/retry; controller removes already
    # invalid test ids, then proves an ordinary second close is idempotent.
    owner._entries.clear()
    audit_native_call(owner.close).assert_native()


def test_failed_release_retains_its_owner_for_retry(owner):
    values, indices = _populate(owner)
    sid = owner._entries[id(values)][2]
    _native('msel_free')(sid)
    audit = audit_native_call(owner.release, values)
    assert type(audit.exception) is Exception
    assert owner._entries[id(values)][0] is values
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    del owner._entries[id(values)]
    audit_native_call(owner.release, indices).assert_native()


@pytest.mark.parametrize('location', ['array_owner', 'pandas_attrs', 'pandas_index_cache', 'weakref', 'interval_buffer_weakref'])
@pytest.mark.parametrize('operation', ['close', 'reinit'])
def test_native_cleanup_refuses_actual_lifetime_hooks_before_state_changes(owner, location, operation):
    calls = []
    class Foreign:
        def __del__(self): calls.append('finalizer')
    foreign = Foreign()  # Controller retains the object beyond the audited call.
    source = (_frame('strings') if location.startswith('pandas') else _frame('interval')
              if location == 'interval_buffer_weakref' else Array.from_list([[1., 2.]], '<f4'))
    if location == 'array_owner': source._pin = foreign
    elif location == 'pandas_attrs': source.attrs['owner'] = foreign
    elif location == 'pandas_index_cache': source.index._cache['owner'] = foreign
    elif location == 'interval_buffer_weakref':
        # Cython omits this typed-memoryview owner from tp_traverse. The
        # actual buffer callback still has to be inspected before cleanup.
        retained_buffer = source.index._engine.root.left.base
        reference = weakref.ref(retained_buffer, lambda unused: calls.append('weakref'))
    else:
        source = np.ones((2, 2), dtype='f4')
        reference = weakref.ref(source, lambda unused: calls.append('weakref'))
    owner._entry(source)
    before = dict(owner._entries)
    _refused(audit_native_call(owner.close if operation == 'close' else owner.__init__))
    assert calls == [] and owner._entries == before
    if location == 'array_owner': source._pin = None
    elif location == 'pandas_attrs': source.attrs.clear()
    elif location == 'pandas_index_cache': del source.index._cache['owner']
    else: del reference


def test_native_cleanup_refuses_pandas_class_finalizers_before_release(owner):
    calls = []
    frame = _frame('numeric')
    owner._entry(frame)
    previous = dict(owner._entries)
    with patch.object(pd.DataFrame, '__del__', lambda unused: calls.append('finalizer'), create=True):
        _refused(audit_native_call(owner.close))
        assert calls == [] and owner._entries == previous


def test_release_identity_miss_never_calls_application_protocols(owner):
    calls = []
    class Foreign:
        def __hash__(self): calls.append('hash'); raise AssertionError('hash')
        def __eq__(self, other): calls.append('eq'); raise AssertionError('eq')
    assert audit_native_call(owner.release, Foreign()).assert_native() is None
    assert calls == []


def test_native_cleanup_uses_connected_container_constructors(owner):
    frame = _frame('numeric')
    owner._entry(frame)
    original = list
    calls = []
    def foreign(value=()):
        if type(value) in (MappingProxyType, type({}.items())):
            calls.append('list')
        return original(value)
    with patch.object(builtins, 'list', foreign):
        audit = audit_native_call(owner.close)
    audit.assert_native()
    assert calls == [] and owner._entries == {}


def test_device_rows_keep_their_resident_snapshot_across_lifecycle_calls(owner):
    values = np.arange(12, dtype='f4').reshape(4, 3)
    indices = np.array([2, 0], dtype='i8')
    rows = DeviceRows(values, indices, owner.take)
    result = rows.materialize()
    expected = result.tobytes()
    audit_native_call(owner.close).assert_native()
    values[:] = -1
    assert rows.materialize() is result and result.tobytes() == expected
    assert owner._entries == {}


def test_native_lifecycle_refuses_custom_dictionary_keys_before_lookup(owner):
    calls = []
    class ForeignKey(str):
        __hash__ = str.__hash__
        def __eq__(self, other): calls.append('key'); raise AssertionError('key')
    original = vars(owner)
    poisoned = {ForeignKey('_entries'): original['_entries']}
    poisoned.update((key, value) for key, value in original.items() if key != '_entries')
    object.__setattr__(owner, '__dict__', poisoned)
    try:
        _refused(audit_native_call(owner.close))
        assert calls == []
    finally:
        object.__setattr__(owner, '__dict__', original)


@pytest.mark.parametrize('hook', ['dictionary', 'getattribute', 'setattr'])
def test_native_lifecycle_refuses_application_class_hooks(hook):
    calls = []
    def foreign(*args): calls.append(hook); raise AssertionError(hook)
    namespace = ({'__dict__': property(foreign)} if hook == 'dictionary' else
                 {'__getattribute__': foreign} if hook == 'getattribute' else {'__setattr__': foreign})
    cls = type('Foreign', (_Resident,), namespace)
    instance = object.__new__(cls)
    _refused(audit_native_call(_Resident.__init__, instance))
    assert calls == []


def test_native_empty_lifecycle_preserves_numpy_free_startup(tmp_path):
    script = '''
import json, sys
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules and 'pandas' not in sys.modules
owner = audit_native_call(_Resident).assert_native()
assert audit_native_call(owner.__enter__).assert_native() is owner
assert audit_native_call(owner.__exit__).assert_native() is False
assert audit_native_call(owner.close).assert_native() is None
assert 'numpy' not in sys.modules and 'pandas' not in sys.modules
print(json.dumps({'passed': True}))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {'passed': True}
