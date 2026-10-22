# SPDX-License-Identifier: Apache-2.0
"""A DeviceRows host materialization is a retained snapshot, not a callback."""
import numpy as np
from array import array
import pytest
import mojolearn as ml
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('estimator', ['MaxAbsScaler', 'RobustScaler'])
@pytest.mark.parametrize('ranges', ['0', '1'])
@pytest.mark.parametrize('field', ['base', 'indices', 'dtype', '_host'])
def test_row_metadata_keys_are_refused_before_python_equality(estimator, ranges, field, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', ranges)
    calls = []
    class ForeignKey(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            calls.append('equality')
            return str.__eq__(self, other)
    rows = DeviceRows(Array.from_list([[1.,2.],[3.,4.]], '<f4'), Array.from_list([0], '<i8'), None)
    value = rows.__dict__.pop(field)
    rows.__dict__[ForeignKey(field)] = value
    calls.clear()
    audit = audit_native_call(getattr(ml, estimator)().fit, rows)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert 'metadata keys' in str(audit.exception)
    assert not calls
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('estimator', ['MaxAbsScaler', 'RobustScaler'])
@pytest.mark.parametrize('hook', ['descriptor', 'getattribute', 'finalizer'])
def test_row_class_layout_is_refused_before_dictionary_or_lifecycle_hooks(estimator, hook, monkeypatch):
    from mojolearn import _expansion_prep
    context = getattr(_expansion_prep, '_MAXABS_CONTEXT' if estimator == 'MaxAbsScaler' else '_ROBUST_CONTEXT')
    calls = []
    def forbidden(*args):
        calls.append(hook)
        raise AssertionError('Python row metadata hook ran')
    rows = DeviceRows(Array.from_list([[1.,2.],[3.,4.]], '<f4'), Array.from_list([0], '<i8'), None)
    model = getattr(ml, estimator)()
    with monkeypatch.context() as patch:
        if hook == 'descriptor':
            # CPython does not allow replacing an existing class's __dict__
            # member. Simulate a wrongly connected class retaining a Python
            # property; the retained native descriptor must reject it.
            class ForeignRows:
                __dict__ = property(forbidden)
            rows = object.__new__(ForeignRows)
            patch.setitem(context, 'maxabs_rows_class', ForeignRows)
        else:
            patch.setattr(DeviceRows, '__getattribute__' if hook == 'getattribute' else '__del__', forbidden, raising=False)
        audit = audit_native_call(model.fit, rows)
    # Restore finalizer metadata while the source is still retained.
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('estimator', ['MaxAbsScaler', 'RobustScaler'])
@pytest.mark.parametrize('preexisting', [False, True])
def test_host_rows_snapshot_survives_base_and_index_mutation(estimator, preexisting, monkeypatch):
    # This explicitly selects the established host-materialization path on
    # every vendor; GPU range gather has its own resident-base contract.
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '0')
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('Python materialization hook ran')
    base = Array.from_list([[1.,2.],[3.,4.],[5.,6.]], '<f4')
    rows = DeviceRows(base, Array.from_list([0,2], '<i8'), forbidden)
    if preexisting:
        rows._host = Array.from_list([[1.,2.],[5.,6.]], '<f4')
        base._mv[0] = 100.
    model = getattr(ml, estimator)()
    before = audit_native_call(model.fit_transform, rows).assert_native()
    snapshot = rows._host
    assert snapshot is not None
    base._mv[0] = 200.
    rows.indices._mv[0] = 1
    after = audit_native_call(model.fit_transform, rows).assert_native()
    assert before.tobytes() == after.tobytes()
    assert rows._host is snapshot and not calls
    field = model.max_abs_ if estimator == 'MaxAbsScaler' else model.center_
    np.testing.assert_array_equal(field, [5.,6.] if estimator == 'MaxAbsScaler' else [3.,4.])


@pytest.mark.parametrize('estimator', ['MaxAbsScaler', 'RobustScaler'])
@pytest.mark.parametrize('dtype', ['<f8', '<i8'])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_materialization_preserves_original_typed_snapshot_before_float32_cast(estimator, dtype, order):
    values = np.array([[1.00000001, 2.], [3., 4.], [5.00000001, 6.]], dtype=dtype)
    if dtype == '<i8':
        values[:, 0] = [16777217, 3, 16777219]
    base = Array._owned(array('d' if dtype == '<f8' else 'q', values.ravel(order=order).tolist()), values.shape, dtype, order)
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('Python typed gather ran')
    rows = DeviceRows(base, Array.from_list([0,2], '<i8'), forbidden)
    actual = audit_native_call(getattr(ml, estimator)().fit_transform, rows).assert_native()
    expected = getattr(ml, estimator)().fit_transform(values[[0,2]].astype('f4'))
    assert actual.tobytes() == expected.tobytes()
    assert rows._host.dtype == dtype and rows._host.shape == (2,2)
    assert rows._host.tobytes() == values[[0,2]].tobytes()
    assert not calls
