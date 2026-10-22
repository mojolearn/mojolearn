# SPDX-License-Identifier: Apache-2.0
"""A DeviceRows host materialization is a retained snapshot, not a callback."""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


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
