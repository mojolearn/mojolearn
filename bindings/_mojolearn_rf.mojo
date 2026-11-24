# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the RandomForest estimators (`ensemble/`).

Kept in its OWN extension for the same reason `_mojolearn_trees` is: an
independently changing binding stops being a merge point, and the ensemble
lane changes independently of the extratrees lane. Arrays cross as borrowed
host-buffer addresses; training device buffers and contexts live for one call.
No input pointer is retained. The default fit moves the native tree list into
an export handle; Python allocates Arrays, exports their bytes and releases
the handle in finally. Retained List-returning entries are diagnostics.

THE MODEL CROSSES AS FLAT ARRAYS, exactly the `_mojolearn_trees` protocol
(deviation 146's layout argument): per-node `colid` / `quesval` /
`left_child_id`, the flat `vector_leaf`, and a `tree_offsets` prefix so tree
`t` is the node range `[offsets[t], offsets[t+1])`. The predict bindings
rebuild `RandomForestMetaData` from those arrays and call the IMPLEMENTED
`RandomForest.predict` / `predict_proba` (`randomforest.cuh:382-436`), not a
reimplementation at this boundary. `instance_count` and `best_metric_val`
are not carried: the traversal reads neither.

WHY THIS BINDS `ensemble/` AND NOT `extratrees/impl/randomforest`: the
extratrees surface refuses `bootstrap=True` by name because ITS copy of the
row sampler has no caller; `ensemble/` is the dedicated cuML RandomForest
implementation and its `fit_forest` IS `RandomForest::fit` (`randomforest.cuh:286-370`)
with the with-replacement `RowSampler` wired. This extension is that
sampler's first Python caller.
"""

from core.experiment_guards_trees import TREES_GUARDS_OK
from bindings.device_plan_binding_pack import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from core.hostptr import list_f32
from ensemble.ada_rounds import AdaState
from xtrees.dart_device import DART_SESSIONS
from ensemble.nan_refusal import RF_NONFINITE_REFUSAL

from std.os import abort
from ensemble.device_finite import FOREST_DEVICE_FINITE
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from forest_inference_binding_door import (
    forest_prepare_gpu_binding, forest_release_gpu_binding,
    forest_predict_resident_into_gpu_binding,
    forest_predict_resident_labels_gpu_binding,
    forest_stage_input_gpu_binding,
    forest_unstage_input_gpu_binding,
    forest_ordered_resident_binding,
)
from core.forest_inference_run import (
    forest_prepare_gpu_native, forest_predict_resident_into_gpu_native,
    forest_predict_resident_labels_gpu_native, forest_gpu_resident_caps,
    forest_resident_rows_native,
)
from bindings.forest_fit_boundary_unpack import forest_fit_args_py
from core.forest_fit_request import (
    ForestFitArgs,
    fit_loop_put,
    XF_CAP_DEVICE_SCAN,
    XF_CAP_DART,
    XF_CAP_SESSION,
    XF_CAP_ADA,
    XF_CAP_DEV_ROWS,
)
from core.forest_store_transport import (
    forest_store_publish, forest_store_words, forest_store_admit,
)
from core.mojolearn_rf_run import RF_STORE, rf_store_ctx
from core.gbdt_entry_table import gbdt_vendor_word
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE
from core.tree_member import tree_member_publish_image
from core.forest_member import forest_member_publish_image
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish, forest_session_entry_releases
from core.entry_table import ENTRY_OP_GIL_FREE
from core.native_descriptor_call import native_descriptor_report
from core.forest_entry_table import FOREST_TABLE_RF
from core.native_descriptor_abi import NativeDescriptorError
from core.forest_inference_model import (
    resident_prepare,
    resident_release,
    resident_stage_input,
    resident_unstage_input,
)
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


from ensemble.instruments import StageTimes
from ensemble.decisiontree.decisiontree import (
    ENTROPY,
    GAMMA,
    GINI,
    INVERSE_GAUSSIAN,
    MSE,
    POISSON,
    TreeMetaDataNode,
)
from ensemble.randomforest import (
    RandomForest,
    RandomForestMetaData,
    ForestPrep,
    fit_forest,
)
from core.mojolearn_rf_run import (
    DT,
    CLT,
    RLT,
    N_RF_FIT_PARAMS,
    _f32_ptr,
    _i32_ptr,
    _cls_criteria,
    _reg_criteria,
    _check_criterion,
    _rf_params_len_ok,
    _rf_min_weight_fraction,
    _rf_fold_min_weight_count,
    _rf_params_from,
    RFExportTrees,
    RF_EXPORTS,
    _retain_rf_handle,
    _rf_export_into_native,
    _rf_export_verify_native,
    RfDataSession,
    rf_session_keep,
    rf_session_lease,
    rf_session_set_ada,
    rf_session_rows,
    _session_shape_check,
    _rf_session_restore_prep,
    _u64_ptr,
    _f64_ptr,
    _ada_state_of,
    _rf_predict_native,
    _rf_weighted_session_fit,
    _rf_ada_classifier_round,
    _rf_classifier_fit_forest,
    _rf_regressor_fit_forest,
    _rf_keep_forest,
    _rf_dart_member_forest,
    _rf_session_rows_forest,
    _rf_data_session_stage,
    _rf_ada_open_stage,
    _rf_ada_regressor_round,
    _rf_data_session_close,
)


def _forest_out_trees(trees: List[TreeMetaDataNode[DT]], notices: Int) raises -> PythonObject:
    """Retained same-fit diagnostic; RF label dtype does not affect tree storage.
    `notices` is the fit's notice mask (`ensemble/rf_notices.mojo`), the
    third metadata word."""
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    var num_outputs = 1
    if len(trees) > 0:
        num_outputs = Int(trees[0].num_outputs)
    rf_flatten_trees(trees, offsets, colid, quesval, left_child, leaves)
    var meta = Python.list()
    meta.append(PythonObject(len(trees)))
    meta.append(PythonObject(num_outputs))
    meta.append(PythonObject(notices))
    var out = Python.list()
    out.append(forest_export_py_list_i64(offsets))
    out.append(forest_export_py_list_i64(colid))
    out.append(forest_export_py_list_f64(quesval))
    out.append(forest_export_py_list_i64(left_child))
    out.append(forest_export_py_list_f64(leaves))
    out.append(meta)
    return out


# DEVIATION 2482: both RF label types own the same typed tree list. Moving it
# into the export registry retains existing native storage without flattening.
from forest_export_binding_unpack import (forest_export_py_list_f64, forest_export_py_list_i64,)
from forest_export_binding_pack import (forest_export_descriptor_py,)
from core.mojolearn_rf_impl import rf_flatten_trees


def rf_forest_export_binding(handle: PythonObject, offsets: PythonObject,
    columns: PythonObject, thresholds: PythonObject, left: PythonObject,
    leaves: PythonObject, counts: PythonObject) raises -> PythonObject:
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var dst: List[Int] = [Int(py=offsets), Int(py=columns), Int(py=thresholds), Int(py=left), Int(py=leaves)]
    _rf_export_into_native(Int(py=handle), dst, Int(py=counts[0]), Int(py=counts[1]), Int(py=counts[2]))
    return PythonObject(None)


def rf_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    return _kept_forest_out(Int(py=handle))


def rf_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    RF_EXPORTS.get_or_create_ptr()[].release(Int(py=handle))
    return PythonObject(None)


def _rf_classifier_fit[ROWMAJOR: Bool = False, ROWS: Bool = False](params: ForestFitArgs) raises -> Int:
    """Fit the cuML-implementation RandomForest classifier. `x` is COLUMN-major
    float32 (n_rows * n_cols, the layout `fit_forest`'s default expects);
    `y` is int32 class CODES in [0, n_classes). See `N_RF_FIT_PARAMS`.
    `criterion` is GINI (0) or ENTROPY (1); anything else is refused by
    name (DEVIATION 407). Returns the export handle (lane calltable-forest:
    every fit is kept in RF_EXPORTS; the diagnostic lists are read from the
    kept fit, `_rf_handle_py`)."""
    var bt = StageTimes()
    var t_bind = bt.start()
    var forest: RandomForestMetaData[DT, CLT]
    if params.gil_free:
        # a member's worker: no GIL to release (core/forest_entry.mojo)
        forest = _rf_classifier_fit_forest[ROWMAJOR, ROWS](params, bt)
    else:
        with GILReleased(Python()):
            forest = _rf_classifier_fit_forest[ROWMAJOR, ROWS](params, bt)
    # The fit loop words the door asked for (XF_W_LOOP; lane alm-forest):
    # `fit_forest_prepared`'s tree loop, read once after the fit.
    fit_loop_put(params.loop, forest.fit_loop)
    return _rf_keep_forest(forest, bt, t_bind)


def _rf_fit_native[CLASSIFIER: Bool, ROWMAJOR: Bool](args: ForestFitArgs) raises -> Int:
    """One fit (the forest entry's XF_FIT): the export handle."""
    comptime if CLASSIFIER:
        return _rf_classifier_fit[ROWMAJOR](args)
    else:
        return _rf_regressor_fit[ROWMAJOR](args)


def _kept_forest_out(handle: Int) raises -> PythonObject:
    # The kept fit's diagnostic lists. A helper of its own so the Dict
    # lookup (DictKeyError) and the list build (Error) both raise Error to
    # the caller's try/finally, which cannot mix the two error types.
    var registry = RF_EXPORTS.get_or_create_ptr()
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(handle)
    var out = PythonObject(None)
    var failed = False
    var words = String("")
    try:
        out = _forest_out_trees(entry.model, Int(entry.meta[2]))
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
    plain entries; the same lists `_forest_out` built from the forest)."""
    var registry = RF_EXPORTS.get_or_create_ptr()
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
    return _rf_handle_py(_rf_fit_native[CLASSIFIER, ROWMAJOR](args), export)


def rf_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, EXPORT)


def rf_classifier_fit_weighted_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    var address = Int(py=weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, EXPORT, weights=address)


def rf_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, EXPORT)


def rf_regressor_fit_weighted_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
    weight_max: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit` with Float32 row weights (scikit-learn's
    `sample_weight`; sk-edges). `weight_max` is the largest weight (the
    caller's native max reduction), which sizes the label_sum plane."""
    var address = Int(py=weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, EXPORT, weights=address,
                                    weight_max=Float64(py=weight_max))


# TOMBSTONE: rf_data_session_open / rf_data_session_close / rf_regressor_fit_session_rows_export / rf_classifier_fit_weighted_session_export / rf_ada_open / rf_classifier_fit_ada_round_export / rf_regressor_fit_ada_round_export (the Python entries of the data session and the AdaBoost rounds: superseded by the forest entry's typed XF_SESSION_OPEN / XF_SESSION_CLOSE / XF_FIT_SESSION / XF_ADA_OPEN / XF_FIT_ADA_ROUND, no caller left) deleted 2026-11-21 by lane dw-trees-boost; code recoverable at cc2c81eb61.
# Restore: git apply experiments/removed/dead_entries_rf_session_ada_calls.patch
def _rf_session_open_native(x_address: Int, n_rows: Int, n_cols: Int, row_major: Bool,
                            share_tables: Bool) raises -> Int:
    """XF_SESSION_OPEN: stage X on the device for the member fits of one
    ensemble. `x` is float32, ROW-major when `row_major` and COLUMN-major
    otherwise, borrowed through this call only. Returns the session handle."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_data_session_open: invalid shape")
    comptime if GLOBAL_NUMERIC_MODE == 1:
        if share_tables:
            raise Error(
                "rf_data_session_open: shared quantile tables are a FAST"
                " option; an IDENTICAL member draws its own sample"
            )
    var xp = _f32_ptr(x_address)
    var made = List[RfDataSession]()
    # THE GIL (lane mf-ada2): every data session operation runs with no GIL
    # (`rf_gpu_forest_entry_v1` releases around a door's call; a typed RUN
    # holds none) and the registry has its own lock (core/mojolearn_rf_run.mojo).
    var bad = _rf_data_session_stage(xp, n_rows, n_cols, row_major, share_tables, made)
    if bad:
        raise Error("rf_data_session_open: " + RF_NONFINITE_REFUSAL)
    return rf_session_keep(made.pop())


# TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


def rf_regressor_fit_dart_binding(
    dart_handle: PythonObject, cls: PythonObject, rows_addr: PythonObject,
    cols_addr: PythonObject, params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_dart_export` (UNPACK of `_rf_regressor_fit_dart`,
    the forest entry's XF_FIT_DART)."""
    var args = forest_fit_args_py(rows_addr, cols_addr, params, criterion)
    args.session = Int(py=dart_handle)
    args.cls = Int(py=cls)
    return _rf_handle_py(_rf_regressor_fit_dart(args), True)


def _rf_regressor_fit_dart(params: ForestFitArgs) raises -> Int:
    """cpu4-forest: a DART round's member fit for class `cls` on the DART
    session's device buffers (`x_trees_dart_open`'s X and the `target`
    plane `x_trees_dart_step` wrote), on the DART context, so the K
    n-sized targets never cross to the host. `rows_addr` (int32,
    `params[0]` bag rows, repeats allowed) and `cols_addr` (int32,
    `params[1]` sampled columns) are 0 when the round has none; X is laid
    out column-major at them and the target gathered at the rows on the
    device (`ensemble/dart_member.mojo`). Same params, checks and export
    descriptor as `rf_regressor_fit_session_rows_export`."""
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_regressor_fit_dart: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("rf_regressor_fit_dart: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var crit = params.criterion
    _check_criterion("rf_regressor_fit_dart", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    _rf_fold_min_weight_count(rf_params, _rf_min_weight_fraction(params), n_rows)
    var hid = params.session
    var c = params.cls
    var rows_a = params.x
    var cols_a = params.y
    # The session is held for this member fit by its own reference (the
    # registry's lock word covers the lookup only; lane fx-gil).
    var dreg = DART_SESSIONS.get_or_create_ptr()
    var dsess = dreg[].lease(hid)
    var n = dsess[].n
    var d = dsess[].d
    var k = dsess[].k
    if c < 0 or c >= k:
        raise Error("rf_regressor_fit_dart: class out of range")
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_regressor_fit_dart: invalid shape")
    if (rows_a == 0 and n_rows != n) or (cols_a == 0 and n_cols != d):
        raise Error(
            "rf_regressor_fit_dart: params name a shape the DART session"
            " does not hold"
        )
    if cols_a != 0 and n_cols > d:
        raise Error("rf_regressor_fit_dart: more columns than X holds")

    # Hazard B: the DART session registry is a process-global List that
    # another thread may append to (reallocating it) or pop from while this
    # member fit runs with the GIL released. Take the session's buffer
    # HANDLES (reference counted; the device memory is shared) under the GIL
    # and never index the registry inside the released block.
    var dart_x = dsess[].x.copy()
    var dart_target = dsess[].target.copy()
    var forest: RandomForestMetaData[DT, RLT]
    if params.gil_free:
        # a member's worker: no GIL to release (core/forest_entry.mojo)
        forest = _rf_dart_member_forest(
            dart_x, dart_target, rows_a, cols_a, n_rows, n_cols, n, d, c, rf_params,
        )
    else:
        with GILReleased(Python()):
            forest = _rf_dart_member_forest(
                dart_x, dart_target, rows_a, cols_a, n_rows, n_cols, n, d, c, rf_params,
            )
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_handle(export_trees^, forest.notices)


def _rf_fit_session_rows_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, regressor: `rf_regressor_fit_export` on the rows
    `params.x` (int32 row ids of the session's X, `params[0]` of them,
    repeats allowed) with `params.y` their labels in that order: the member
    fit of a resampling ensemble (AdaBoost.R2). The rows are gathered ON THE
    DEVICE from the staged X into a fresh column-major matrix, the bytes a
    host gather of the same rows stages, and the fit is `fit_forest` on it
    with its own quantile table. Returns the export handle."""
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_regressor_fit_session_rows: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("rf_regressor_fit_session_rows: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    if n_rows <= 0:
        raise Error("rf_regressor_fit_session_rows: no rows")
    var rp = _i32_ptr(params.x)
    var yp = _f32_ptr(params.y)
    var crit = params.criterion
    _check_criterion("rf_regressor_fit_session_rows", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    _rf_fold_min_weight_count(rf_params, _rf_min_weight_fraction(params), n_rows)
    # Hazard B: the session's X handle is taken under the registry's lock
    # (`rf_session_lease`); the registry is not indexed during the fit.
    var lease = rf_session_lease(params.session, False)
    var src_rows = lease.n_rows
    if lease.n_cols != n_cols:
        raise Error("rf_regressor_fit_session_rows: the session holds another column count")
    var forest = _rf_session_rows_forest(
        lease.dx, rp, yp, n_rows, n_cols, src_rows, rf_params,
    )
    _ = lease^  # the fit above read the session's X through its handle
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_handle(export_trees^, forest.notices)


def _rf_fit_weighted_session_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_SESSION, classifier: the weighted classifier fit on a data
    session's X (its COLUMN-major stage): same params and weight checks as
    the weighted XF_FIT. Returns the export handle."""
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_classifier_fit_weighted_session: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values (17 with min_weight_fraction_leaf), got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    if n_classes < 2:
        raise Error("rf_classifier_fit_weighted_session: n_classes must be >= 2")
    var yp = _i32_ptr(params.y)
    var crit = params.criterion
    _check_criterion("rf_classifier_fit_weighted_session", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)
    var mw_frac = _rf_min_weight_fraction(params)
    var weights_address = params.weights
    if weights_address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")

    # cpu3-bindings: one memcpy; the refusals, all-ones test and total run
    # on the device after the upload (`core/weight_scan_device`).
    var weights = list_f32(_f32_ptr(weights_address), n_rows)
    var weight_total = Float64(0)

    var session_id = params.session
    # Hazard B: the session's X handle, scalars and prepared tables are taken
    # under the registry's lock (`rf_session_lease`); the registry is not
    # indexed during the fit (another thread may open or close a session
    # meanwhile). The swapped-out prep goes back on the error path too.
    var lease = rf_session_lease(session_id, True)
    var prep = List[ForestPrep]()
    swap(prep, lease.prep)
    try:
        _session_shape_check(
            "rf_classifier_fit_weighted_session", lease.n_rows, lease.n_cols, n_rows, n_cols,
        )
    except error:
        _rf_session_restore_prep(session_id, prep)
        raise error^
    var forest: RandomForestMetaData[DT, CLT]
    try:
        forest = _rf_weighted_session_fit(
            lease.dx, yp, weights^, n_rows, n_cols, n_classes, rf_params^,
            mw_frac, prep, lease.share_tables, lease.host_x, weight_total,
        )
    except error:
        _rf_session_restore_prep(session_id, prep)
        raise error^
    _rf_session_restore_prep(session_id, prep)
    _ = lease^  # the fit above read the session's X through its handle
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_handle(export_trees^, forest.notices)


# ------------------------------------------------------------ AdaBoost --
# Lane n5-prophet-adaboost-loops (2026-11-09; plan
# plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) section 4, row "AdaBoost").
# The ensemble's boosting state stays on the device for the whole fit
# (ensemble/ada_rounds.mojo): one call per round fits the member on the
# session's X from the resident weights, predicts the training X on the
# device and runs the SAMME / R2 step there. Before, every round rebuilt the
# float32 weights on the host (`_w32` + `trees_weighted_rows`), uploaded them
# into the member fit, downloaded the member's predictions, uploaded weights,
# predictions and labels into the step, downloaded the weights, rescaled
# them on the host and (R2) drew and gathered the rows on the host and
# uploaded them with their targets.


def _rf_ada_open_native(session_id: Int, wa: Int, ya: Int, ca: Int, n: Int, k: Int) raises:
    """XF_ADA_OPEN: keep an AdaBoost fit's state on a data session's device:
    `w` (float64, the normalized starting weights), `y` (int32 class codes
    when k >= 2; float32 targets when k = 0), `cw` (float32 class_weight row
    weights, or 0 for none). n must be the session's row count. Every array
    is borrowed through this call only. Replaces an earlier state of the
    session."""
    if k == 1 or k < 0:
        raise Error("rf_ada_open: k must be 0 (regressor) or >= 2")
    var wp = _u64_ptr(wa)
    if ya == 0:
        raise Error("rf_ada_open: null label buffer address")
    if ca != 0 and k == 0:
        raise Error("rf_ada_open: class weights belong to a classifier")
    if rf_session_rows(session_id) != n or n <= 0:
        raise Error("rf_ada_open: n must be the session's row count")
    var made = List[AdaState]()
    _rf_ada_open_stage(wp, ya, ca, n, k, made)
    # under the registry's lock; a session closed meanwhile refuses
    rf_session_set_ada(session_id, n, made)


def _rf_ada_classifier_round_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_ADA_ROUND, classifier: one AdaBoostClassifier (SAMME) round on
    a data session holding the fit's state (XF_ADA_OPEN): the member's
    weights from the resident boosting weights, the weighted member fit on
    the session's X (same params and checks as the weighted session fit),
    its predictions of the training X and the SAMME step, all on the device.
    `params.rate` is the learning rate, `params.last` the last round;
    `params.stats` (float64, 4) gets [status, alpha, error, total]. The
    weights are rescaled by the total in place when the round moved them and
    the total is positive (the next round's start). Returns the member's
    export handle."""
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_classifier_fit_ada_round: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var n_classes = params.i(2)
    if n_classes < 2:
        raise Error("rf_classifier_fit_ada_round: n_classes must be >= 2")
    var crit = params.criterion
    _check_criterion("rf_classifier_fit_ada_round", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)
    var mw_frac = _rf_min_weight_fraction(params)
    var lr = params.rate
    var last = params.last != 0
    var stats = _f64_ptr(params.stats)
    var session_id = params.session
    # Hazard B: the session's X handle, scalars and prepared tables taken
    # under the registry's lock; the prep goes back on every path.
    var lease = rf_session_lease(session_id, True)
    var prep = List[ForestPrep]()
    swap(prep, lease.prep)
    var state: AdaState
    try:
        _session_shape_check(
            "rf_classifier_fit_ada_round", lease.n_rows, lease.n_cols, n_rows, n_cols,
        )
        state = _ada_state_of(session_id, True, "rf_classifier_fit_ada_round")
        if state.k != n_classes:
            raise Error("rf_classifier_fit_ada_round: n_classes differs from the session's AdaBoost state")
    except error:
        _rf_session_restore_prep(session_id, prep)
        raise error^
    var forest: RandomForestMetaData[DT, CLT]
    try:
        forest = _rf_ada_classifier_round(
            lease.dx, state, n_rows, n_cols, n_classes, rf_params^, mw_frac,
            prep, lease.share_tables, lease.host_x, lr, last, stats,
        )
    except error:
        _rf_session_restore_prep(session_id, prep)
        raise error^
    _rf_session_restore_prep(session_id, prep)
    _ = lease^  # the round above read the session's X through its handle
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_handle(export_trees^, forest.notices)


def _rf_ada_regressor_round_native(params: ForestFitArgs) raises -> Int:
    """XF_FIT_ADA_ROUND, regressor: one AdaBoostRegressor (AdaBoost.R2) round
    on a data session holding the fit's state (XF_ADA_OPEN, k = 0): n rows
    drawn with probability w / sum(w) on the device (`x_trees_weighted_sample`'s
    cdf and draws, stream = the round), the session's X and the targets
    gathered at them, the squared-error member fit (as the session rows
    fit: same params and checks), its predictions of the training X and the
    R2 step, all on the device. `params.round_seed`, `params.round`,
    `params.loss`, `params.rate` and `params.last` are the round's words;
    `params.stats` (float64, 4) gets [status, alpha, error, total]; the
    weights are rescaled by the total in place when the round moved them and
    the total is positive. Returns the member's export handle."""
    if not _rf_params_len_ok(params):
        raise Error(
            "rf_regressor_fit_ada_round: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("rf_regressor_fit_ada_round: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_cols = params.i(1)
    var crit = params.criterion
    _check_criterion("rf_regressor_fit_ada_round", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    _rf_fold_min_weight_count(rf_params, _rf_min_weight_fraction(params), n_rows)
    var seed = params.round_seed
    var round = params.round
    var loss = params.loss
    var lr = params.rate
    var last = params.last != 0
    var stats = _f64_ptr(params.stats)
    var session_id = params.session
    var lease = rf_session_lease(session_id, False)
    _session_shape_check(
        "rf_regressor_fit_ada_round", lease.n_rows, lease.n_cols, n_rows, n_cols,
    )
    var state = _ada_state_of(session_id, False, "rf_regressor_fit_ada_round")
    var forest = _rf_ada_regressor_round(
        lease.dx, state, n_rows, n_cols, rf_params, seed, round, loss, lr, last, stats,
    )
    _ = lease^  # the round above read the session's X through its handle
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_handle(export_trees^, forest.notices)


def _rf_regressor_fit[ROWMAJOR: Bool = False, ROWS: Bool = False](params: ForestFitArgs) raises -> Int:
    """Fit the cuML-implementation RandomForest regressor. Same contract; slot 2
    MUST be 0 and `y` is float32. `criterion` is MSE (2), POISSON (4),
    GAMMA (5) or INVERSE_GAUSSIAN (6); anything else is refused by name
    (DEVIATION 407). The log criteria are DEVIATION 406's: they RUN in
    both numeric modes, comparable to cuML in neither. Returns the export
    handle (as the classifier)."""
    var bt = StageTimes()
    var t_bind = bt.start()
    var forest: RandomForestMetaData[DT, RLT]
    if params.gil_free:
        forest = _rf_regressor_fit_forest[ROWMAJOR, ROWS](params, bt)
    else:
        with GILReleased(Python()):
            forest = _rf_regressor_fit_forest[ROWMAJOR, ROWS](params, bt)
    # The fit loop words the door asked for (XF_W_LOOP; lane alm-forest):
    # `fit_forest_prepared`'s tree loop, read once after the fit.
    fit_loop_put(params.loop, forest.fit_loop)
    return _rf_keep_forest(forest, bt, t_bind)


def _rf_predict_py(offsets_addr: PythonObject, colid_addr: PythonObject, quesval_addr: PythonObject,
                   left_child_addr: PythonObject, leaves_addr: PythonObject, x_addr: PythonObject,
                   out_addr: PythonObject, params: PythonObject, regression: Bool) raises -> PythonObject:
    """`rf_predict_proba` / `rf_predict_reg` (UNPACK of `_rf_predict_native`):
    `params` is `[n_rows, n_cols, n_trees, num_outputs]`."""
    if len(params) != 4:
        if regression:
            raise Error("rf_predict_reg: params must hold [n_rows, n_cols, n_trees, 1], got " + String(len(params)))
        raise Error("rf_predict_proba: params must hold [n_rows, n_cols, n_trees, num_outputs], got "
                    + String(len(params)))
    var model: List[Int] = [Int(py=offsets_addr), Int(py=colid_addr), Int(py=quesval_addr),
                            Int(py=left_child_addr), Int(py=leaves_addr)]
    return PythonObject(_rf_predict_native(model, Int(py=x_addr), Int(py=out_addr), Int(py=params[0]),
                                           Int(py=params[1]), Int(py=params[2]), Int(py=params[3]), regression))


def rf_predict_proba_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject, quesval_addr: PythonObject,
    left_child_addr: PythonObject, leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict_py(offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, x_addr,
                          out_addr, params, False)


def rf_predict_reg_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject, quesval_addr: PythonObject,
    left_child_addr: PythonObject, leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _rf_predict_py(offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, x_addr,
                          out_addr, params, True)


def rf_device_finite_scan_binding() raises -> PythonObject:
    """1 when this build's RF fits refuse a non-finite X cell through a
    device scan (FOREST_DEVICE_FINITE, FAST on Apple), so the Python fit
    skips its host scan of the same cells; 0 otherwise."""
    return PythonObject(1 if FOREST_DEVICE_FINITE else 0)


#: QUALITY FIX (lane apple-fast-q-reg, 2026-10-04), FAST on every vendor.
# TOMBSTONE: MOJOLEARN_DT_BINS_QOLD (old path; the 256-bin FAST default is the recorded winner) deleted 2026-11-15 by lane/expdel2-01-base; code recoverable at b81caf7773.
# Restore: git apply experiments/removed/MOJOLEARN_DT_BINS_QOLD.patch
#: Audit (M3 0.8.34, taxi):
#: decision-tree-reg r2 0.8626 vs scikit-learn 0.8915, bagging-reg 0.9188 vs
#: 0.9388, voting-reg 0.9192 vs 0.9246, stacking-reg 0.9197 vs 0.9325 (all
#: four fit DecisionTreeRegressor members; IDENTICAL has the same numbers, so
#: it is the parameter, not a FAST shortcut). scikit-learn splits on exact
#: thresholds; ours on n_bins quantile bins, 128 by default, which caps a
#: deep tree's resolution on taxi's continuous columns. 256 is the largest
#: count the binned (uint8) path takes, so the kernels stay the same.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab5-dtbins, taxi): decision-tree-reg r2 0.8626 -> 0.8792 at 52.79 -> 53.38
#: ms (+1.1%); bagging-reg r2 0.9170 -> 0.9233 at 202.06 -> 187.32 ms (-7.3%);
#: voting/stacking/adaboost-reg errored in both arms (not measured). KEEP.
comptime RF_DT_DEFAULT_BINS = (
    256 if GLOBAL_NUMERIC_MODE == NUMERIC_FAST else 128
)


def rf_numeric_mode_binding() raises -> PythonObject:
    """Read the numeric policy compiled into this RF binding."""
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def rf_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def rf_classifier_fit_shard_binding(x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject,
    weights_addr: PythonObject) raises -> PythonObject:
    return _rf_fit_py[True, False](x_addr, y_addr, params, criterion, False, weights=Int(py=weights_addr),
                                   tree_start=Int(py=tree_start))


def rf_regressor_fit_shard_binding(x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject) raises -> PythonObject:
    return _rf_fit_py[False, False](x_addr, y_addr, params, criterion, False, tree_start=Int(py=tree_start))


struct RfGpuForest(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest): the bodies the Python entries above unpack into."""

    @staticmethod
    def caps() -> Int:
        var caps = forest_gpu_resident_caps() | XF_CAP_DART | XF_CAP_SESSION | XF_CAP_ADA | XF_CAP_DEV_ROWS
        comptime if FOREST_DEVICE_FINITE:
            caps |= XF_CAP_DEVICE_SCAN
        return caps

    @staticmethod
    def dt_default_bins() -> Int:
        return RF_DT_DEFAULT_BINS

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return _rf_fit_native[True, True](args)
            return _rf_fit_native[True, False](args)
        if row_major:
            return _rf_fit_native[False, True](args)
        return _rf_fit_native[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        return _rf_regressor_fit_dart(args)

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        RF_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        _rf_export_into_native(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return _rf_export_verify_native(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        RF_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool, free: Bool) raises -> Int:
        return _rf_predict_native(model, x, out_addr, rows, cols, trees, outputs, regression)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_gpu_native[True](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_into_gpu_native[True, True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_labels_gpu_native[True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_release(handle: Int) raises:
        resident_release[True](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: rebuild the forest host binding")

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int,
                       free: Bool) raises -> Int:
        raise Error("RuntimeError: rebuild the forest host binding")

    @staticmethod
    def strict_release(handle: Int) raises:
        raise Error("RuntimeError: rebuild the forest host binding")

    @staticmethod
    def session_open(x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
        return _rf_session_open_native(x, rows, cols, row_major, share_tables)

    @staticmethod
    def session_close(handle: Int) raises:
        _rf_data_session_close(handle)

    @staticmethod
    def fit_session(classifier: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            return _rf_fit_weighted_session_native(args)
        return _rf_fit_session_rows_native(args)

    @staticmethod
    def ada_open(handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
        _rf_ada_open_native(handle, w, y, cw, n, k)

    @staticmethod
    def fit_ada_round(classifier: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            return _rf_ada_classifier_round_native(args)
        return _rf_ada_regressor_round_native(args)

    @staticmethod
    def stage_input(x: Int, rows: Int, cols: Int) raises -> Int:
        """XF_STAGE_INPUT (lane mf-meta3c): the body of the Python entry
        `forest_stage_input_gpu` (bindings/forest_inference_binding_door.mojo)."""
        var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x)
        return 1 if resident_stage_input[True](xp.unsafe_origin_cast[MutAnyOrigin](), rows * cols) else 0

    @staticmethod
    def unstage_input() raises:
        resident_unstage_input[True]()

    @staticmethod
    def fit_rows(classifier: Bool, args: ForestFitArgs) raises -> Int:
        # XF_FIT_ROWS (lane vdf-scope-trees): the plain fit bodies with the
        # rows taken out of the store on the device (`rf_store_rows_x`).
        if classifier:
            return _rf_classifier_fit[False, True](args)
        return _rf_regressor_fit[False, True](args)

    @staticmethod
    def resident_rows(mode: Int, handle: Int, store_id: Int, idx: Int, out_addr: Int, rows: Int,
                      cols: Int, outputs: Int) raises -> Int:
        # XF_RESIDENT_ROWS (lane vdf-scope-trees2a): the rows of the store
        # taken on the device and read there by the snapshot (one context).
        var store = RF_STORE.get_or_create_ptr()
        return forest_resident_rows_native[True](store[], rf_store_ctx(), mode, handle, store_id, idx,
                                                 out_addr, rows, cols, outputs)


def _rf_store_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    var store = RF_STORE.get_or_create_ptr()
    if op == XP_DEV_PUT:
        return store[].put(rf_store_ctx(), a[0], p[0])
    if op == XP_DEV_FREE:
        store[].free(rf_store_ctx(), p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return store[].take_rows(rf_store_ctx(), p[0], p[1], a[0], p[2])
    return Int(store[].live)


def rf_gpu_store_transport_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's STORE TRANSPORT entry (core/forest_store_transport.mojo;
    lane vdf-scope-trees): put / free / take rows / live of its device store
    (core/mojolearn_rf_run.mojo `RF_STORE`, on the fits' context), the
    x_prep transport numbering. No CPython: callers hold no GIL or release
    it themselves; the store has its own lock word."""
    try:
        forest_store_admit(operation, na, np, nf, result)
        var value = _rf_store_op(operation, forest_store_words(aa, na), forest_store_words(pp, np))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def rf_gpu_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo).

    THE DATA SESSIONS RUN WITHOUT THE GIL (lane mf-ada2, 2026-11-23): a
    door's data session operation (`forest_session_entry_releases`) runs
    with the GIL released around the whole dispatch and the GIL-free bit set,
    so its upload, fits and launches run bare; a typed RUN that holds no GIL
    sends the bit and runs the same dispatch with no release. The session
    registry has its own lock word (core/mojolearn_rf_run.mojo). Every other
    operation dispatches as before (the fits release around their own body
    unless the bit is set)."""
    if not forest_session_entry_releases(operation):
        return forest_entry_dispatch[RfGpuForest](operation, aa, na, pp, np, ff, nf, result, context, sink)
    var status = 1
    try:
        with GILReleased(Python()):
            status = forest_entry_dispatch[RfGpuForest](
                operation + ENTRY_OP_GIL_FREE, aa, na, pp, np, ff, nf, result, context, sink
            )
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
    return status


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_rf.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_rf() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_rf")
        # The native forest entry (core/forest_entry_table.mojo) under this image's key.
        forest_entry_publish["MojolearnForestEntryOwnerRfGpuV1"](FOREST_TABLE_RF, COMPILED_VENDOR, rf_gpu_forest_entry_v1)
        # Its device store's transport entry (XF_CAP_DEV_ROWS; lane vdf-scope-trees).
        var store_vendor = gbdt_vendor_word(COMPILED_VENDOR)
        if store_vendor >= 0:
            forest_store_publish(FOREST_TABLE_RF, Int(GLOBAL_NUMERIC_MODE), store_vendor,
                                 Int(RF_STORE.get_or_create_ptr()), rf_gpu_store_transport_v1)
        # The families' member entries (core/member_abi.mojo) over this forest entry.
        tree_member_publish_image(COMPILED_VENDOR)
        forest_member_publish_image(COMPILED_VENDOR)
        m.def_function[rf_vendor_binding]("rf_vendor")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[rf_numeric_mode_binding]("rf_numeric_mode")
        m.def_function[rf_device_finite_scan_binding]("rf_device_finite_scan")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_classifier_fit_binding[True]]("rf_classifier_fit_export")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: TREES_LEGACY_PYTHON_ENTRIES (DELETED: no caller; the door fits through the forest entry table) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
        # Restore: git apply experiments/removed/TREES_LEGACY_PYTHON_ENTRIES.patch
        m.def_function[rf_classifier_fit_weighted_binding[False]]("rf_classifier_fit_weighted")
        m.def_function[rf_classifier_fit_weighted_binding[True]]("rf_classifier_fit_weighted_export")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_regressor_fit_binding[True]]("rf_regressor_fit_export")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_regressor_fit_weighted_binding[True]]("rf_regressor_fit_weighted_export")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_predict_proba_binding]("rf_predict_proba")
        m.def_function[rf_predict_reg_binding]("rf_predict_reg")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_ordered_resident_binding]("forest_ordered_resident")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_forest_export_binding]("forest_export")
        m.def_function[rf_forest_export_legacy_binding]("forest_export_legacy")
        m.def_function[rf_forest_export_release_binding]("forest_export_release")
        # every snapshot prepared here runs on the registry's process context and
        # borrows its one I/O workspace (core/forest_inference_model.mojo,
        # FOREST_PER_MODEL_IO; gap-fails2 2026-10-02)
        m.def_function[forest_prepare_gpu_binding[True]]("forest_prepare_gpu")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_release_gpu_binding[True]]("forest_release_gpu")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_predict_resident_into_gpu_binding[True, True]]("forest_predict_resident_reuse_gpu")
        m.def_function[forest_predict_resident_labels_gpu_binding[True]]("forest_predict_resident_labels_gpu")
        m.def_function[forest_stage_input_gpu_binding[True]]("forest_stage_input_gpu")
        m.def_function[forest_unstage_input_gpu_binding[True]]("forest_unstage_input_gpu")
        # TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[rf_classifier_fit_shard_binding]("rf_classifier_fit_shard")
        return m.finalize()
    except e:
        abort(String("failed to initialize _mojolearn_rf: ") + String(e))
        return PythonObject(None)
