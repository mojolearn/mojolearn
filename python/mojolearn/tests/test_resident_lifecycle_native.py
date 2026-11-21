# SPDX-License-Identifier: Apache-2.0
"""model_selection._Resident lifecycle: one native call per method.

The entries live in the native state behind the instance's one `_owner`
slot. Owners are never inspected: every free runs while the owners are
still held, an entry leaves the state before its free, and the owners are
released last. NumPy, pandas and Arrow-backed pandas owners are all
supported; an owner's own finalizer runs after the native work. `_entry`
answers (owner, prepared Array, store id) and puts at most once.

Deleted with the instance dictionary (2026-11-21, arena unit E): the
assertions on `_entries` identity and on an application alias of the old
mapping, the replaced `_free` store entry (no such field exists; the store
is the helper table's, resolved at construction) and the `__setattr__`
hook refusal (the owner is written through the slot's own descriptor, so a
subclass hook is never reached). A failed entry that stays for a retry is
dropped by the owner's destructor (best effort), not by clearing a mapping.
"""
import gc

import numpy as np
import pytest

from mojolearn._buffer import _native
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call


def _live():
    return int(_native('msel_live')())


def _free_behind_its_back(store_id):
    # Test-controller fault injection: the model-selection store's own free.
    _native('msel_free')(store_id)


@pytest.fixture
def owner():
    before = _live()
    result = _Resident()
    try:
        yield result
    finally:
        result.close()
        gc.collect()
        assert _live() == before


def _numpy():
    return np.arange(12, dtype='f4').reshape(4, 3)


def _pandas(kind):
    pd = pytest.importorskip('pandas')
    frame = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [5., 6., 7., 8.]})
    if kind == 'pandas':
        return frame
    pytest.importorskip('pyarrow')
    if kind == 'arrow_float':
        return frame.astype('float64[pyarrow]')
    if kind == 'arrow_int':
        return frame.astype('int64[pyarrow]')
    frame.index = pd.Index(pd.array(['w', 'x', 'y', 'z'], dtype='string[pyarrow]'))
    return frame


def _owner_value(kind):
    return _numpy() if kind == 'numpy' else _pandas(kind)


KINDS = ['numpy', 'pandas', 'arrow_float', 'arrow_int', 'arrow_string_index']
INDICES = np.array([2, 0], dtype='i8')


def test_constructor_and_reinitialization_are_native_and_free_live_entries():
    before = _live()
    result = audit_native_call(_Resident).assert_native()
    try:
        assert _Resident.__slots__ == ('_owner',)
        with pytest.raises(AttributeError):
            result.note = 'kept'
        values = _numpy()
        result.take(values, INDICES)
        assert _live() == before + 2
        assert audit_native_call(result.__init__).assert_native() is None
        assert _live() == before
        # The reinitialized state is empty: the next take puts again.
        result.take(values, INDICES)
        assert _live() == before + 2
    finally:
        result.close()
    assert _live() == before


@pytest.mark.parametrize('kind', KINDS)
def test_take_gathers_rows_for_every_supported_owner(owner, kind):
    values = _owner_value(kind)
    expected = np.asarray(values, dtype='f8' if kind != 'numpy' else 'f4')[[2, 0]]
    rows = owner.take(values, INDICES)
    assert np.array_equal(np.asarray(rows.tolist(), dtype=expected.dtype), expected)
    live = _live()
    # The entry holds the caller's object; asking again puts nothing.
    assert owner._entry(values)[0] is values and _live() == live


def test_take_on_numpy_owner_is_one_native_call(owner):
    values = _numpy()
    rows = audit_native_call(owner.take, values, INDICES).assert_native()
    assert rows.tolist() == values[[2, 0]].tolist()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('operation', ['close', 'reinit', 'exit', 'release'])
def test_lifecycle_is_native_for_every_supported_owner(owner, kind, operation):
    values = _owner_value(kind)
    before = _live()
    owner.take(values, INDICES)
    assert _live() == before + 2
    if operation == 'close':
        assert audit_native_call(owner.close).assert_native() is None
    elif operation == 'reinit':
        assert audit_native_call(owner.__init__).assert_native() is None
    elif operation == 'exit':
        assert audit_native_call(owner.__exit__, None, None, None).assert_native() is False
    else:
        assert audit_native_call(owner.release, values).assert_native() is None
        assert _live() == before + 1
        assert audit_native_call(owner.release, INDICES).assert_native() is None
    assert _live() == before


def test_arrow_owner_finalizer_runs_after_every_free_and_publication(owner):
    pa = pytest.importorskip('pyarrow')
    pd = pytest.importorskip('pandas')
    seen = []

    class Foreign:
        def __del__(self):
            seen.append(_live())

    data = np.array([1., 2., 3., 4.], dtype='f8')
    before = _live()
    buffer = pa.foreign_buffer(data.ctypes.data, data.nbytes, base=Foreign())
    column = pa.Array.from_buffers(pa.float64(), 4, [None, buffer])
    frame = pd.DataFrame({'x': pd.arrays.ArrowExtensionArray(column)})
    del buffer, column
    assert owner.take(frame, INDICES).tolist() == [[3.], [1.]]
    del frame
    gc.collect()
    assert seen == []
    owner.close()
    gc.collect()
    # The owner chain is released only after the frees: the finalizer saw
    # no live slot of this resident.
    assert seen == [before]


def test_repeated_close_and_close_after_release(owner):
    values = _numpy()
    before = _live()
    owner.take(values, INDICES)
    owner.release(values)
    assert _live() == before + 1
    assert audit_native_call(owner.close).assert_native() is None
    assert audit_native_call(owner.close).assert_native() is None
    assert audit_native_call(owner.release, values).assert_native() is None
    assert _live() == before


def test_context_exit_on_error_closes_and_propagates(owner):
    before = _live()
    with pytest.raises(KeyError, match='inside'):
        with owner as entered:
            assert entered is owner
            entered.take(_numpy(), INDICES)
            raise KeyError('inside')
    assert _live() == before


@pytest.mark.parametrize('operation', ['close', 'reinit', 'exit'])
def test_error_order_first_failure_all_slots_attempted_failed_retained(operation):
    initial = _live()
    resident = _Resident()
    first, second = _numpy(), _numpy() + 1
    third = np.array([3, 1], dtype='i8')
    resident.take(first, INDICES)
    resident._entry(second)
    resident._entry(third, '<i8')
    before = _live()
    bad_first = resident._entry(first)[2]
    bad_last = resident._entry(third, '<i8')[2]
    assert _live() == before
    _free_behind_its_back(bad_first)
    _free_behind_its_back(bad_last)
    method = {'close': resident.close, 'reinit': resident.__init__,
              'exit': lambda: resident.__exit__(None, None, None)}[operation]
    audit = audit_native_call(method)
    report = audit.report()
    # The first failure in entry order is raised in the store's words; the
    # healthy slots between the two failures are still freed.
    assert audit.exception is not None, report
    assert f'msel store: id {bad_first} is not live' in str(audit.exception)
    assert not report['python_frames'] and not report['forbidden_native_calls'], report
    assert _live() == before - 2 - 2
    # The two failed entries stay for a retry (a failed reinitialization keeps
    # the old owner with them): asking again puts nothing.
    assert resident._entry(first)[2] == bad_first and resident._entry(third, '<i8')[2] == bad_last
    assert _live() == before - 4
    # The destructor drops what is left (best effort: those slots are gone).
    del resident, audit
    gc.collect()
    assert _live() == initial


def test_context_exit_failure_chains_the_body_error():
    initial = _live()
    resident = _Resident()
    values = _numpy()
    resident.take(values, INDICES)
    _free_behind_its_back(resident._entry(values)[2])
    # A cleanup failure in __exit__ replaces the body's error, as Python's
    # with-statement does for any failing __exit__.
    with pytest.raises(Exception, match='is not live') as caught:
        with resident:
            raise KeyError('body')
    assert not isinstance(caught.value, KeyError)
    del resident, caught
    gc.collect()
    assert _live() == initial


def test_failed_release_retains_its_owner_for_retry():
    initial = _live()
    resident = _Resident()
    values = _numpy()
    resident.take(values, INDICES)
    sid = resident._entry(values)[2]
    _free_behind_its_back(sid)
    audit = audit_native_call(resident.release, values)
    assert audit.exception is not None
    assert not audit.report()['python_frames']
    live = _live()
    assert resident._entry(values)[0] is values and resident._entry(values)[2] == sid and _live() == live
    assert audit_native_call(resident.release, INDICES).assert_native() is None
    del resident, audit
    gc.collect()
    assert _live() == initial


def test_a_subclass_attribute_hook_is_never_reached():
    calls = []

    class Hooked(_Resident):
        __slots__ = ()

        def __setattr__(self, name, value):
            calls.append(name)
            raise AssertionError('hook ran')
    before = _live()
    resident = Hooked()
    values = _numpy()
    assert resident.take(values, INDICES).tolist() == values[[2, 0]].tolist()
    resident.close()
    assert calls == [] and _live() == before
