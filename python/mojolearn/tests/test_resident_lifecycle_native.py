# SPDX-License-Identifier: Apache-2.0
"""model_selection._Resident lifecycle: one native call per method.

Owners are never inspected: every free runs while the owners are still held,
the emptied mapping is published before the first free, and the owners are
released last. NumPy, pandas and Arrow-backed pandas owners are all
supported; an owner's own finalizer runs after the native work."""
import gc

import numpy as np
import pytest

from mojolearn._buffer import _native
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call


def _live():
    return int(_native('msel_live')())


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
        result.note = 'kept'
        values = _numpy()
        result.take(values, INDICES)
        assert _live() == before + 2
        assert audit_native_call(result.__init__).assert_native() is None
        assert result._entries == {} and _live() == before and result.note == 'kept'
    finally:
        result.close()


@pytest.mark.parametrize('kind', KINDS)
def test_take_gathers_rows_for_every_supported_owner(owner, kind):
    values = _owner_value(kind)
    expected = np.asarray(values, dtype='f8' if kind != 'numpy' else 'f4')[[2, 0]]
    rows = owner.take(values, INDICES)
    assert np.array_equal(np.asarray(rows.tolist(), dtype=expected.dtype), expected)
    assert owner._entries[id(values)][0] is values


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
    snapshot = owner._entries
    copy = dict(snapshot)
    if operation == 'close':
        assert audit_native_call(owner.close).assert_native() is None
    elif operation == 'reinit':
        assert audit_native_call(owner.__init__).assert_native() is None
    elif operation == 'exit':
        assert audit_native_call(owner.__exit__, None, None, None).assert_native() is False
    else:
        assert audit_native_call(owner.release, values).assert_native() is None
        assert owner._entries is snapshot and id(values) not in owner._entries
        assert audit_native_call(owner.release, INDICES).assert_native() is None
    assert owner._entries == {} and _live() == before
    if operation != 'release':
        # The old mapping is never mutated: an application alias keeps it.
        assert owner._entries is not snapshot and snapshot == copy


def test_arrow_owner_finalizer_runs_after_every_free_and_publication(owner):
    pa = pytest.importorskip('pyarrow')
    pd = pytest.importorskip('pandas')
    seen = []

    class Foreign:
        def __del__(self):
            seen.append((dict(owner._entries), _live()))

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
    # The owner chain is released only after the frees and the publication:
    # the finalizer saw a closed resident and no live slot.
    assert seen == [({}, before)]


def test_repeated_close_and_close_after_release(owner):
    values = _numpy()
    before = _live()
    owner.take(values, INDICES)
    owner.release(values)
    assert _live() == before + 1
    assert audit_native_call(owner.close).assert_native() is None
    assert audit_native_call(owner.close).assert_native() is None
    assert audit_native_call(owner.release, values).assert_native() is None
    assert owner._entries == {} and _live() == before


def test_context_exit_on_error_closes_and_propagates(owner):
    before = _live()
    with pytest.raises(KeyError, match='inside'):
        with owner as entered:
            assert entered is owner
            entered.take(_numpy(), INDICES)
            raise KeyError('inside')
    assert owner._entries == {} and _live() == before


@pytest.mark.parametrize('operation', ['close', 'reinit', 'exit'])
def test_error_order_first_failure_all_slots_attempted_failed_retained(owner, operation):
    first, second = _numpy(), _numpy() + 1
    third = np.array([3, 1], dtype='i8')
    owner.take(first, INDICES)
    owner._entry(second)
    owner._entry(third, '<i8')
    before = _live()
    bad_first = owner._entries[id(first)][2]
    bad_last = owner._entries[id(third)][2]
    _native('msel_free')(bad_first)
    _native('msel_free')(bad_last)
    method = {'close': owner.close, 'reinit': owner.__init__, 'exit': owner.__exit__}[operation]
    audit = audit_native_call(method)
    report = audit.report()
    # The first failure in entry order is raised unchanged; later failures
    # are dropped; the healthy slots between them are still freed.
    assert type(audit.exception) is Exception, report
    assert str(audit.exception) == f'msel store: id {bad_first} is not live'
    assert not report['python_frames'] and not report['forbidden_native_calls'], report
    assert _live() == before - 2 - 2
    assert set(owner._entries) == {id(first), id(third)}
    assert owner._entries[id(first)][0] is first
    owner._entries.clear()


def test_context_exit_failure_chains_the_body_error(owner):
    values = _numpy()
    owner.take(values, INDICES)
    sid = owner._entries[id(values)][2]
    _native('msel_free')(sid)
    # A cleanup failure in __exit__ replaces the body's error, as Python's
    # with-statement does for any failing __exit__.
    with pytest.raises(Exception, match='is not live') as caught:
        with owner:
            raise KeyError('body')
    assert type(caught.value) is Exception
    owner._entries.clear()


def test_failed_release_retains_its_owner_for_retry(owner):
    values = _numpy()
    owner.take(values, INDICES)
    sid = owner._entries[id(values)][2]
    _native('msel_free')(sid)
    audit = audit_native_call(owner.release, values)
    assert type(audit.exception) is Exception
    assert owner._entries[id(values)][0] is values
    assert not audit.report()['python_frames']
    del owner._entries[id(values)]
    assert audit_native_call(owner.release, INDICES).assert_native() is None


def test_lifecycle_refuses_application_attribute_hooks():
    class Hooked(_Resident):
        def __setattr__(self, name, value):
            raise AssertionError('hook ran')
    with pytest.raises(TypeError, match='attribute hooks'):
        Hooked()


def test_take_refuses_replaced_store_entries(owner):
    owner._free = lambda sid: None
    with pytest.raises(TypeError, match='connected native entries'):
        owner.take(_numpy(), INDICES)
    owner.__init__()
