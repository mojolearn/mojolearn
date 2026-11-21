# SPDX-License-Identifier: Apache-2.0
"""A DeviceRows host materialization is a retained snapshot, not a callback.

The snapshot is kept on the fold's native state; `rows.materialize()`
returns it once it exists (before any gather owner is consulted).

Deleted with the instance dictionary (2026-11-21, arena unit E): the test
that rebuilt `rows.__dict__` with hooked keys (a DeviceRows has one slot),
and the pre-set `rows._host`; a preexisting snapshot is now made by a
first `materialize` through `model_selection._Resident.take`.
"""
import gc
import numpy as np
from array import array
import pytest
import mojolearn as ml
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call


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
    # Earlier refusal audits retain exception/frame cycles until collection.
    # Drain those before installing a class-wide hook: the audit's own
    # pre-trace collection must not finalize a previous test's DeviceRows
    # under this test's temporary __del__.
    gc.collect()
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
    assert not calls
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    if hook == 'descriptor':
        # A wrongly connected class has no native rows state behind the slot.
        assert isinstance(audit.exception, TypeError), audit.report()
    else:
        # The state is read through the slot: a class hook is never reached.
        audit.assert_native()


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
    resident = _Resident()
    rows = DeviceRows(base, Array.from_list([0,2], '<i8'), resident.take if preexisting else forbidden)
    if preexisting:
        first = rows.materialize()
        base._mv[0] = 100.
    model = getattr(ml, estimator)()
    before = audit_native_call(model.fit_transform, rows).assert_native()
    snapshot = rows.materialize()
    if preexisting:
        assert snapshot is first
    base._mv[0] = 200.
    rows.indices._mv[0] = 1
    after = audit_native_call(model.fit_transform, rows).assert_native()
    assert before.tobytes() == after.tobytes()
    assert rows.materialize() is snapshot and not calls
    resident.close()
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
    snapshot = rows.materialize()
    assert snapshot.dtype == dtype and snapshot.shape == (2,2)
    assert snapshot.tobytes() == values[[0,2]].tobytes()
    assert not calls
