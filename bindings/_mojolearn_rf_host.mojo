# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_rf` family, the RandomForest classifier and
regressor (workstream E batch 3, rf-clf and rf-reg, 2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`ensemble/host/rf_oracle.mojo::rf_host_fit`, the device trainer
(`ensemble/randomforest.mojo::fit_forest` and the batched-level builder)
restated on the host from its kernels, so the five model arrays are meant to
be the GPU columns' bytes. The predict entries are
`core/forest_host_predict.mojo::rf_host_trees` and `rf_host_predict`, the
walk the forest host binding runs and the forest host gate holds to the
recorded GPU predictions on seven CPUs, exported here under the GPU
binding's `rf_predict_proba` and `rf_predict_reg` names and address
contract, so a fitted or loaded model predicts through them on a CPU-only
install (the infer column and the model column's reload check).

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/randomforest.py` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_rf": "_mojolearn_rf_host"`):
`rf_classifier_fit_export`, `rf_regressor_fit_export` and their
`_rowmajor_export` forms (the 16-slot params list and the criterion
argument of `bindings/_mojolearn_rf.mojo`), `forest_export`,
`forest_export_legacy`, `forest_export_release`, `rf_predict_proba`,
`rf_predict_reg`, `rf_vendor` answering "cpu" and `rf_numeric_mode`.
ADDED 2026-09-15 (rf-reg-poisson, rf-reg-gamma-ig,
rf-clf-balanced-parallel): the POISSON, GAMMA and INVERSE_GAUSSIAN criteria
fit through the oracle; `rf_classifier_fit_weighted` and its `_export` form
take the per-row class weights (the weighted BOOTSTRAP, and since
2026-09-27 weights without a bootstrap, the weighted objective);
and `forest_prepare_gpu`, `forest_predict_resident_reuse_gpu` and
`forest_release_gpu`, the resident `parallel_groves` entries, predict over
`core/forest_host_groves.mojo` (`RF_INPUT=True`: the input flushed).
ADDED 2026-10-02 (lane/fix-dart-host): the data session entries
`rf_data_session_open`, `rf_data_session_close`,
`rf_regressor_fit_session_rows_export`
and `rf_classifier_fit_weighted_session_export`, which DART opens by default
(e3372c826); each member fit is the plain entries' body on the session's X.
ADDED 2026-10-05 (box-run-2-dart-host): `rf_regressor_fit_dart_export`,
the DART member fit on the host DART session's X and target plane
(xtrees/dart_host.mojo), the host twin of cpu4-forest's device entry.
ABSENT, and so refused BY NAME through `_HostBinding`: the global tree-ID
shard fits (`rf_*_fit_shard`, the multi-GPU driver's), the non-resident
`rf_predict_*_gpu_parallel` and the pool and comparison entries.

WHY A FAMILY OF ITS OWN AND NOT THE FOREST HOST BINDING. The batch 3 brief
named `bindings/_mojolearn_forest_host.mojo` as the home. That binding is
loaded BY PATH and exports `forest_host_*` names under a different address
contract, and `_backend` deliberately does not route `_mojolearn_rf` to it
(its `_HOST_MODULES` comment says why). Routing the RandomForest classes on
a CPU-only install needs a binding that exports the GPU binding's own names,
which is exactly the shape the Extra Trees lane took with the `trees`
family, so this is the `rf` family, gated by the routed set's
`-D MOJOLEARN_HOST_SABOTAGE=1`.

The sabotage arm (`rf_host_sabotage`) is
`ensemble/host/rf_oracle.mojo::RF_ORACLE_HOST_SABOTAGE`: every bootstrap row
is drawn from the next Philox subsequence, so every forest this binary fits
differs.
"""
from core.experiment_guards_trees import TREES_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call as _balanced_call, py_type as _exact_type
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.forest_export_binding_unpack import (
    forest_export_py_list_f64,
    forest_export_py_list_i64,
)
from bindings.forest_export_binding_pack import (
    forest_export_descriptor_py,
)
from core.mojolearn_rf_host_impl import (
    RfHostSession,
    _apply_min_weight_fraction,
    rf_host_dart_member_xy,
    rf_host_forest_lists,
    rf_host_session_rows_x,
)
from core.hostptr import (
    f32_ptr,
    i32_ptr,
)
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_predict import rf_host_predict, rf_host_trees
from bindings.forest_host_strict_binding_pack import (
    forest_strict_prepare_host_binding, forest_strict_predict_host_binding, forest_strict_release_host_binding,
    forest_strict_predict_host_native,
)
from core.forest_host_strict_run import forest_strict_prepare_host_native, forest_strict_release_host_native
from bindings.forest_host_groves_binding_pack import (
    forest_predict_resident_host_binding,
    forest_prepare_host_binding,
    forest_release_host_binding,
)
from core.forest_host_groves_run import (
    forest_prepare_host_native, forest_predict_resident_host_native, forest_release_host_native,
)
from bindings.forest_fit_boundary_unpack import forest_fit_args_py
from core.forest_fit_request import (
    ForestFitArgs,
    XF_CAP_STRICT,
    XF_CAP_DART,
    XF_CAP_SESSION,
)
from core.tree_member import tree_member_publish_image
from core.forest_member import forest_member_publish_image
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_entry_table import FOREST_TABLE_RF
from core.native_descriptor_abi import NativeDescriptorError
from xtrees.dart_host import DART_HOST, DART_HOST_SESSIONS
from ensemble.nan_refusal import RF_NONFINITE_REFUSAL
from core.host_finite_scan import host_all_finite_f32
from ensemble.host.rf_oracle import (
    RF_ORACLE_HOST_SABOTAGE,
    RfHostForest,
    rf_host_fit,
)
from core.mojolearn_rf_host_run import (
    RF_HOST_EXPORTS,
    _read_x_col_major,
    _retain_rf_handle,
    _min_weight_fraction,
    _read_weights,
    _host_drop_zero_weight_rows,
    _rf_fit_colmajor,
    RF_HOST_SESSIONS,
    _session_params,
    _session_x,
    _rf_export_into_native,
    _rf_export_verify_native,
    _rf_fit_forest,
    _rf_predict_walk,
)


def _index(value: PythonObject) raises -> Int:
    # An exact int (every address, count and slot the wrappers pass) is its
    # own operator.index: read it directly (p3-trees: two attribute reads, a
    # class-name string and a module lookup per integer argument before).
    if _exact_type(value) is _exact_type(PythonObject(0)):
        return Int(py=value)
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("rf host: integers expected, not booleans")
    var operator_module = native_loaded_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def rf_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def rf_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def rf_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "rf host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_rf_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `rf_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def rf_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary draws every bootstrap row from the next Philox
    subsequence on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control; `ensemble/host/rf_oracle.mojo::RF_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(RF_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def rf_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def rf_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def _forest_out(forest: RfHostForest) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::_forest_out_trees`, the same six lists."""
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    rf_host_forest_lists(forest, offsets, colid, quesval, left_child, leaves)
    var meta = Python.list()
    meta.append(PythonObject(forest.n_trees))
    meta.append(PythonObject(forest.num_outputs))
    # The fit's notice bits (`ensemble/rf_notices.mojo`): the public fit
    # raises them as Python warnings, never prints them.
    meta.append(PythonObject(forest.notices))
    var out = Python.list()
    out.append(forest_export_py_list_i64(offsets))
    out.append(forest_export_py_list_i64(colid))
    out.append(forest_export_py_list_f64(quesval))
    out.append(forest_export_py_list_i64(left_child))
    out.append(forest_export_py_list_f64(leaves))
    out.append(meta)
    return out


def _rf_fit[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> Int:
    """`_rf_classifier_fit` / `_rf_regressor_fit` of the GPU binding: the same
    slot checks in the same words, then the host fit. `weights_addr` is the
    weighted fit's Float32 row weights (0: none; the regressor's with
    `weight_max`, their largest, as `rf_regressor_fit_weighted`);
    `tree_start` the GPU binding's global tree ID offset (the shard fits
    below). Returns the export handle (lane calltable-forest: every fit is
    kept in RF_HOST_EXPORTS; `_rf_handle_py` reads the diagnostic lists)."""
    var forest: RfHostForest
    if params.gil_free:
        # a member's worker: no GIL to release (core/forest_entry.mojo)
        forest = _rf_fit_forest[CLASSIFIER, ROWMAJOR](params)
    else:
        with GILReleased(Python()):
            forest = _rf_fit_forest[CLASSIFIER, ROWMAJOR](params)
    return _retain_rf_handle(forest^)


def _kept_forest_out(handle: Int) raises -> PythonObject:
    # The kept fit's diagnostic lists. A helper of its own so the Dict
    # lookup (DictKeyError) and the list build (Error) both raise Error to
    # the caller's try/finally, which cannot mix the two error types.
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(handle)
    var out = PythonObject(None)
    var failed = False
    var words = String("")
    try:
        out = _forest_out(entry.model)
    except problem:
        failed = True
        words = String(problem)
    registry[].put_back(handle, entry^)
    if failed:
        raise Error(words)
    return out


def _rf_handle_py(handle: Int, export: Bool) raises -> PythonObject:
    """PACK of a kept fit for the Python entries: the export descriptor
    (`*_export`), or the diagnostic lists of the kept fit, released (the
    plain entries; the lists `_forest_out` built from the forest)."""
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    if export:
        try:
            return forest_export_descriptor_py(registry[], handle)
        except error:
            registry[].release(handle)
            raise error^
    var out = PythonObject(None)
    try:
        out = _kept_forest_out(handle)
    finally:
        registry[].release(handle)
    return out


def _rf_fit_py[CLASSIFIER: Bool, ROWMAJOR: Bool](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject, criterion: PythonObject,
    export: Bool, weights: Int = 0, tree_start: Int = 0, oob: Int = 0, score: Int = 0, importance: Int = 0,
    weight_max: Float64 = 0,
) raises -> PythonObject:
    """The Python fit entries: UNPACK (`forest_fit_args_py`), the native
    fit, PACK (`_rf_handle_py`)."""
    var args = forest_fit_args_py(x_addr, y_addr, params, criterion)
    args.weights = weights
    args.tree_start = tree_start
    args.oob = oob
    args.score = score
    args.importance = importance
    args.weight_max = weight_max
    return _rf_handle_py(_rf_fit[CLASSIFIER, ROWMAJOR](args), export)


# TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


def rf_classifier_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, True)


def _rf_classifier_fit_weighted(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject, export: Bool,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_classifier_fit_weighted_binding`."""
    var address = _index(weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, export, weights=address)


def rf_classifier_fit_weighted_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    return _rf_classifier_fit_weighted(x_addr, y_addr, params, criterion, weights_addr, False)


def rf_classifier_fit_weighted_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    return _rf_classifier_fit_weighted(x_addr, y_addr, params, criterion, weights_addr, True)


def rf_regressor_fit_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, True)


def rf_regressor_fit_weighted_export_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
    weight_max: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_regressor_fit_weighted_binding` on the
    host column (lane sk-last): Float32 row weights (scikit-learn's
    `sample_weight`) and their largest value (the caller's native max),
    which sizes the label_sum plane."""
    var address = _index(weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, True, weights=address,
                                    weight_max=Float64(py=weight_max))


def rf_classifier_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject,
    weights_addr: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_classifier_fit_shard_binding`: the
    column-major fit of trees `tree_start .. tree_start + n_estimators` of
    the whole forest, weighted when `weights_addr` is nonzero
    (`parallel_ensemble.fit_forest`'s shard; lane/cpu-training-par-wave2,
    2026-09-15)."""
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, False, weights=_index(weights_addr),
                                   tree_start=_index(tree_start))


def rf_regressor_fit_shard_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_regressor_fit_shard_binding`."""
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, False, tree_start=_index(tree_start))


# TOMBSTONE: rf_data_session_open / rf_data_session_close / rf_regressor_fit_session_rows_export / rf_classifier_fit_weighted_session_export (the Python entries of the data session: superseded by the forest entry's typed XF_SESSION_OPEN / XF_SESSION_CLOSE / XF_FIT_SESSION, no caller left) deleted 2026-11-21 by lane dw-trees-boost; code recoverable at cc2c81eb61.
# Restore: git apply experiments/removed/dead_entries_rf_session_ada_calls.patch
def _rf_session_open_native(x_address: Int, n_rows: Int, n_cols: Int, row_major: Bool,
                            share_tables: Bool) raises -> Int:
    """XF_SESSION_OPEN (`bindings/_mojolearn_rf.mojo` `_rf_session_open_native`):
    `x` is float32, ROW-major when `row_major` and COLUMN-major otherwise,
    borrowed through this call only. Returns the session handle."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_data_session_open: invalid shape")
    # A host binding is IDENTICAL only, and an IDENTICAL member draws its
    # own quantile sample (the GPU binding's IDENTICAL refusal).
    if share_tables:
        raise Error(
            "rf_data_session_open: shared quantile tables are a FAST"
            " option; an IDENTICAL member draws its own sample"
        )
    # Lane a8c-device-x-linear-prep-rfe: NaN AND infinity (the GPU binding's
    # device scan of its upload), so the boost meta-estimators' own host
    # scan before the open is deleted; the host column tests the words it
    # reads, once, in parallel SIMD tasks (core/host_finite_scan.mojo).
    if not host_all_finite_f32(f32_ptr(x_address), n_rows * n_cols):
        raise Error("rf_data_session_open: " + RF_NONFINITE_REFUSAL)
    var x = List[Float32]()
    with GILReleased(Python()):
        x = _read_x_col_major(x_address, n_rows, n_cols, row_major)
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var id = reg[].next_id
    reg[].next_id += 1
    reg[].sessions.append(RfHostSession(id, x^, n_rows, n_cols))
    return id


def _rf_session_close_native(handle: Int) raises:
    """XF_SESSION_CLOSE."""
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var si = reg[].find(handle)
    _ = reg[].sessions.pop(si)


def _rf_fit_session_rows_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, regressor: the regressor fit on the rows `params.x`
    (int32 row ids of the session's X, `params[0]` of them, repeats
    allowed), `params.y` their labels in that order: the column-major gather
    of those rows (the bytes the GPU binding's `launch_gather_rows_colmajor`
    writes), then the plain fit. Returns the export handle."""
    comptime entry = "rf_regressor_fit_session_rows"
    var p = _session_params[False](entry, params)
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    if n_rows <= 0:
        raise Error(entry + ": no rows")
    var rp = i32_ptr(params.x)
    var reg = RF_HOST_SESSIONS.get_or_create_ptr()
    var si = reg[].find(params.session)
    var src_rows = reg[].sessions[si].n_rows
    if reg[].sessions[si].n_cols != n_cols:
        raise Error(entry + ": the session holds another column count")
    var x = rf_host_session_rows_x(
        entry, reg[].sessions[si].x, src_rows, rp, n_rows, n_cols
    )
    var y_address = params.y
    var forest: RfHostForest
    with GILReleased(Python()):
        forest = _rf_fit_colmajor[False](
            x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
        )
    return _retain_rf_handle(forest^)


def _rf_fit_weighted_session_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, classifier: the weighted classifier fit on the
    session's X. Returns the export handle."""
    comptime entry = "rf_classifier_fit_weighted_session"
    var p = _session_params[True](entry, params)
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    var w_address = params.weights
    if w_address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    var weights = _read_weights(w_address, n_rows)
    var x = _session_x(entry, params.session, n_rows, n_cols)
    var y_address = params.y
    # sk-weq-trees: the zero-weight rows leave this member's fit
    var y_kept = List[Int32]()
    var n_fit = _host_drop_zero_weight_rows(
        x, weights, y_kept, y_address, n_rows, n_cols, p.bootstrap
    )
    var y_fit = y_address if n_fit == n_rows else Int(y_kept.unsafe_ptr())
    var mwl = _apply_min_weight_fraction(p, _min_weight_fraction(params), weights, n_fit)
    var forest: RfHostForest
    with GILReleased(Python()):
        forest = _rf_fit_colmajor[True](
            x^, y_fit, n_fit, n_cols, n_classes, p, weights, 0,
            min_weight_leaf=mwl,
        )
    _ = y_kept^
    return _retain_rf_handle(forest^)


def rf_regressor_fit_dart_binding(
    dart_handle: PythonObject, cls: PythonObject, rows_addr: PythonObject,
    cols_addr: PythonObject, params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_dart_export` (UNPACK of `_rf_regressor_fit_dart`,
    the forest entry's XF_FIT_DART)."""
    var args = forest_fit_args_py(rows_addr, cols_addr, params, criterion)
    args.session = _index(dart_handle)
    args.cls = _index(cls)
    return _rf_handle_py(_rf_regressor_fit_dart(args), True)


def _rf_regressor_fit_dart(params: ForestFitArgs) raises -> Int:
    """`bindings/_mojolearn_rf.mojo::rf_regressor_fit_dart_binding` on the
    host column (box-run-2-dart-host, 2026-10-05). cpu4-forest moved the
    device DART member fit onto the DART session's buffers (X and the class
    `target` plane `x_trees_dart_step` leaves there), and the Python loop
    (`RandomForestRegressor._fit_in_dart`) calls this entry on both columns;
    the host binding had no twin, so the host DART round could not fit.

    Here the member reads the HOST DART session (xtrees/dart_host.mojo
    `DART_HOST_SESSIONS`, the same `_Global` the host x_trees binding opens):
    X is the caller's row-major n x d array at `x_addr`, the target the
    session's k x n `target` plane. The gathers are
    `ensemble/dart_member.mojo`'s, element for element: X column-major
    `x[j * m + i] = X[rows[i] * d + cols[j]]` (identity rows / columns when
    the address is 0) and `y[i] = target[c * n + rows[i]]` (the whole class
    plane when rows is 0). Then `_rf_fit_colmajor[False]`, the body the plain
    host regressor fit runs: the blocked label magnitude and scale, then
    `rf_host_fit`, the host restatement of the device's
    `fit_forest_prepared` (no prepared tables, no shared tables, as the
    device entry). So the member forest's words are the device member's.
    Same params, checks and export descriptor as the device entry."""
    comptime entry = "rf_regressor_fit_dart"
    comptime if DART_HOST:
        var p = _session_params[False](entry, params)
        var n_rows = params.i(0)
        var n_cols = params.i(1)
        var hid = params.session
        var c = params.cls
        var rows_a = params.x
        var cols_a = params.y
        # held for this member fit (the registry's lock word covers the
        # lookup only; lane fx-gil)
        var dreg = DART_HOST_SESSIONS.get_or_create_ptr()
        var dsess = dreg[].lease(hid)
        var n = dsess[].n
        var d = dsess[].d
        var k = dsess[].k
        if c < 0 or c >= k:
            raise Error(entry + ": class out of range")
        if n_rows <= 0 or n_cols <= 0:
            raise Error(entry + ": invalid shape")
        if (rows_a == 0 and n_rows != n) or (cols_a == 0 and n_cols != d):
            raise Error(
                entry + ": params name a shape the DART session does not hold"
            )
        if cols_a != 0 and n_cols > d:
            raise Error(entry + ": more columns than X holds")
        # the row and column ids, checked against the session (the device
        # entry trusts its callers; a host read out of range is not a word
        # the device writes, so refuse it instead)
        var x = List[Float32]()
        var y = List[Float32]()
        rf_host_dart_member_xy(
            entry, rows_a, cols_a, n_rows, n_cols, n, d, c,
            dsess[].x_addr, dsess[].target, x, y,
        )
        var y_address = Int(y.unsafe_ptr())
        var forest: RfHostForest
        if params.gil_free:
            # a member's worker: no GIL to release (core/forest_entry.mojo)
            forest = _rf_fit_colmajor[False](
                x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
            )
        else:
            with GILReleased(Python()):
                forest = _rf_fit_colmajor[False](
                    x^, y_address, n_rows, n_cols, 0, p, List[Float32](), 0
                )
        # y is read through its address inside the fit: keep it alive past it
        _ = y^
        return _retain_rf_handle(forest^)
    else:
        raise Error(
            entry + ": built without the host DART round"
            " (MOJOLEARN_IDN_DART_DEVICE_OFF)"
        )


def rf_forest_export_binding(
    handle: PythonObject,
    offsets: PythonObject,
    columns: PythonObject,
    thresholds: PythonObject,
    left: PythonObject,
    leaves: PythonObject,
    counts: PythonObject,
) raises -> PythonObject:
    """`bindings/_mojolearn_rf.mojo::rf_forest_export_binding`: the retained
    forest into the caller's five arrays (UNPACK of `_rf_export_into_native`)."""
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var dst: List[Int] = [_index(offsets), _index(columns), _index(thresholds), _index(left), _index(leaves)]
    _rf_export_into_native(_index(handle), dst, _index(counts[0]), _index(counts[1]), _index(counts[2]))
    return PythonObject(None)


def rf_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    return _kept_forest_out(_index(handle))


def rf_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    RF_HOST_EXPORTS.get_or_create_ptr()[].release(_index(handle))
    return PythonObject(None)


def _rf_predict_native(model: List[Int], x_address: Int, out_addr: Int, n_rows: Int, n_cols: Int, n_trees: Int,
                       num_outputs: Int, classifier: Bool, free: Bool = False) raises -> Int:
    """`rf_predict_proba` / `rf_predict_reg`'s contract (XF_PREDICT): ROW-major
    `x`, returns rows written. The walk is `core/forest_host_predict.mojo`'s,
    which rebuilds the trees from the five arrays and refuses a malformed
    node instead of reading past it; `n_nodes` is read off the offsets."""
    var written = 0
    if free:
        # a member's worker: no GIL to release (core/forest_entry.mojo)
        written = _rf_predict_walk(model, x_address, out_addr, n_rows, n_cols, n_trees, num_outputs, classifier)
    else:
        with GILReleased(Python()):
            written = _rf_predict_walk(model, x_address, out_addr, n_rows, n_cols, n_trees, num_outputs, classifier)
    return written


def _rf_predict(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    entry: String,
    classifier: Bool,
) raises -> PythonObject:
    """UNPACK of `_rf_predict_native`: `params` is `[n_rows, n_cols,
    n_trees, num_outputs]`."""
    if len(params) != 4:
        raise Error(
            entry + ": params must hold [n_rows, n_cols, n_trees,"
            " num_outputs], got " + String(len(params))
        )
    var model: List[Int] = [_index(offsets_addr), _index(colid_addr), _index(quesval_addr),
                            _index(left_child_addr), _index(leaves_addr)]
    return PythonObject(_rf_predict_native(model, _index(x_addr), _index(out_addr), _index(params[0]),
                                           _index(params[1]), _index(params[2]), _index(params[3]), classifier))


def rf_predict_proba_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict(
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr,
        x_addr, out_addr, params, String("rf_predict_proba"), True,
    )


def rf_predict_reg_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict(
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr,
        x_addr, out_addr, params, String("rf_predict_reg"), False,
    )


def resident_prepare_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu`: the resident parallel_groves snapshot, on the host."""
    return forest_prepare_host_binding[True](
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, params
    )


def resident_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu`: the grove reduction, on the host."""
    return forest_predict_resident_host_binding[True](handle, x_addr, out_addr, params)


def resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu`."""
    return forest_release_host_binding[True](handle)


struct RfHostForestImage(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest): the bodies the Python entries above unpack into.
    The host column scans X itself (no device scan), keeps its trees in the
    strict snapshot and has no ordered resident word or resident labels."""

    @staticmethod
    def caps() -> Int:
        comptime if DART_HOST:
            return XF_CAP_STRICT | XF_CAP_DART | XF_CAP_SESSION
        else:
            return XF_CAP_STRICT | XF_CAP_SESSION

    @staticmethod
    def dt_default_bins() -> Int:
        # this image exports no `rf_dt_default_bins`: the door's default
        return 128

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return _rf_fit[True, True](args)
            return _rf_fit[True, False](args)
        if row_major:
            return _rf_fit[False, True](args)
        return _rf_fit[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        return _rf_regressor_fit_dart(args)

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        RF_HOST_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        _rf_export_into_native(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return _rf_export_verify_native(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        RF_HOST_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool, free: Bool) raises -> Int:
        return _rf_predict_native(model, x, out_addr, rows, cols, trees, outputs, not regression, free)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_host_native[True](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_host_native[True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: this forest binding has no resident label prediction")

    @staticmethod
    def resident_release(handle: Int) raises:
        forest_release_host_native[True](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        return forest_strict_prepare_host_native[True](model, features, trees, outputs)

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int,
                       free: Bool) raises -> Int:
        return forest_strict_predict_host_native[True](handle, x, out_addr, rows, cols, outputs, free)

    @staticmethod
    def strict_release(handle: Int) raises:
        forest_strict_release_host_native[True](handle)

    @staticmethod
    def session_open(x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
        return _rf_session_open_native(x, rows, cols, row_major, share_tables)

    @staticmethod
    def session_close(handle: Int) raises:
        _rf_session_close_native(handle)

    @staticmethod
    def fit_session(classifier: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            return _rf_fit_weighted_session_native(args)
        return _rf_fit_session_rows_native(args)

    @staticmethod
    def ada_open(handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
        raise Error("RuntimeError: the forest host binding has no AdaBoost device rounds")

    @staticmethod
    def fit_ada_round(classifier: Bool, args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the forest host binding has no AdaBoost device rounds")


def rf_host_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo)."""
    return forest_entry_dispatch[RfHostForestImage](operation, aa, na, pp, np, ff, nf, result, context, sink)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_rf_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_rf_host() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_rf_host")
        # The native forest entry (core/forest_entry_table.mojo) under the cpu key.
        forest_entry_publish["MojolearnForestEntryOwnerRfHostV1"](FOREST_TABLE_RF, "cpu", rf_host_forest_entry_v1)
        # The families' member entries (core/member_abi.mojo) over this forest entry.
        tree_member_publish_image("cpu")
        forest_member_publish_image("cpu")
        module.def_function[rf_host_numeric_mode_binding]("rf_host_numeric_mode")
        module.def_function[rf_host_vendor_binding]("rf_host_vendor")
        module.def_function[rf_host_column_binding]("rf_host_column")
        module.def_function[rf_host_sabotage_binding]("rf_host_sabotage")
        module.def_function[rf_vendor_binding]("rf_vendor")
        module.def_function[rf_numeric_mode_binding]("rf_numeric_mode")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[rf_classifier_fit_export_binding]("rf_classifier_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: TREES_LEGACY_PYTHON_ENTRIES (DELETED: no caller; the door fits through the forest entry table) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
        # Restore: git apply experiments/removed/TREES_LEGACY_PYTHON_ENTRIES.patch
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[rf_regressor_fit_export_binding]("rf_regressor_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[rf_classifier_fit_shard_binding]("rf_classifier_fit_shard")
        module.def_function[rf_forest_export_binding]("forest_export")
        module.def_function[rf_forest_export_legacy_binding]("forest_export_legacy")
        module.def_function[rf_forest_export_release_binding]("forest_export_release")
        module.def_function[rf_predict_proba_binding]("rf_predict_proba")
        module.def_function[rf_predict_reg_binding]("rf_predict_reg")
        module.def_function[rf_classifier_fit_weighted_binding]("rf_classifier_fit_weighted")
        module.def_function[rf_classifier_fit_weighted_export_binding]("rf_classifier_fit_weighted_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[rf_regressor_fit_weighted_export_binding]("rf_regressor_fit_weighted_export")
        module.def_function[resident_prepare_binding]("forest_prepare_gpu")
        module.def_function[resident_predict_binding]("forest_predict_resident_reuse_gpu")
        module.def_function[resident_release_binding]("forest_release_gpu")
        # The strict snapshot of the sequential engine (N044, ne1-a-trees).
        module.def_function[forest_strict_prepare_host_binding[True]]("forest_strict_prepare")
        module.def_function[forest_strict_predict_host_binding[True]]("forest_strict_predict")
        module.def_function[forest_strict_release_host_binding[True]]("forest_strict_release")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # box-run-2-dart-host: the DART member fit on the host DART session
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_rf_host: ", error))
