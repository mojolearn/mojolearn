# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The query X of a Gaussian-process predict (lane a7a-device-x-svm-gp-krr,
# 2026-11-09; plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section
# 3.2 steps 3-4).
#
#   gp_query(source, est, context, entry_tests) -> GpQuery
#       the query X once: a cross-validation fold stays on the device when
#       the entry gathers it (`entry_tests`: the resident GPR predict, the
#       GPC predict_all entries), else the host float32 C-order words (the
#       public door's registered X as it is, anything else admitted once)
#   gp_query_scan(query, est, context)
#       the finite test of a query the door did not test, for a route whose
#       entry returns no status word (one parallel host pass)
#   gp_query_refuse(finite, in_range, query, est)
#       the refusal of a query an entry flagged
#
# Every route refuses in the one wording 'Input X contains NaN or infinity
# (<est>).' (admitted_finite_refuse / door_rows_refuse); the GP entries'
# DEVIATION 1768 query refusals are no longer reached from the doors.
# Numerics: none (the same words reach the same kernels).
from std.python import PythonObject
from bindings.admitted_family import admit_family
from bindings.classical_arrays import classical_address
from bindings.admitted_input import admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import DoorRowsStore, door_is_device_rows, door_rows_refuse
from bindings.gp_svm_gp_state import gp_x_known, gp_x_deferred


struct GpQuery(Movable):
    """The admitted query of a GP predict: `q` the host float32 C-order
    array (None for a device fold), `store` a fold's resident ids
    (`store.active()`), `tested` True when every host word was tested
    finite already (by the public door's admission)."""
    var q: PythonObject
    var store: DoorRowsStore
    var rows: Int
    var cols: Int
    var tested: Bool

    def __init__(out self, q: PythonObject, store: DoorRowsStore, rows: Int, cols: Int, tested: Bool):
        self.q = q
        self.store = store
        self.rows = rows
        self.cols = cols
        self.tested = tested


def gp_query(source: PythonObject, est: String, context: PythonObject, entry_tests: Bool) raises -> GpQuery:
    """The query X of a GP predict (see the module note)."""
    if entry_tests and door_is_device_rows(source, context):
        # admit_x `device_x`: the fold's store ids, no host gather; the
        # entry gathers and tests the rows (finite=False: no host pass).
        var x = admit_x_unrecorded(source, est, context, device_x=True, finite=False)
        return GpQuery(x.array, x.store, x.rows, x.cols, False)
    if gp_x_known(source, context):
        # The public door's admitted C-order float32 X (no second snapshot);
        # tested unless the door deferred its finite test to the entry.
        return GpQuery(source, DoorRowsStore.none(), Int(py=source.shape[0]), Int(py=source.shape[1]),
                       not gp_x_deferred(source))
    var q = admit_family(source, 2, "X", context).array
    return GpQuery(q, DoorRowsStore.none(), Int(py=q.shape[0]), Int(py=q.shape[1]), False)


def gp_query_address(query: GpQuery) raises -> PythonObject:
    """The host query's data address (0 for a device fold), as the entries
    take it."""
    if query.store.active():
        return PythonObject(0)
    return classical_address(query.q)


def gp_query_scan(query: GpQuery, est: String, context: PythonObject) raises:
    """The finite test of a host query nobody tested yet, for a route whose
    entry returns no status word: one parallel pass, scikit-learn's
    wording. A device fold never reaches such a route."""
    if query.tested or query.store.active():
        return
    _ = admit_family(query.q, 2, "X", context, finite=True,
                     refusal="Input X contains NaN or infinity (" + est + ").")


def gp_query_refuse(finite: Int, in_range: Int, query: GpQuery, est: String) raises:
    """The refusal of a query the entry flagged (its status words)."""
    if query.store.active():
        door_rows_refuse(finite, in_range, est, query.store.base_rows)
    else:
        admitted_finite_refuse(finite, est)
