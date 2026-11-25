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
    rf_host_dart_member_xy,
    rf_host_forest_lists,
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
)
from core.tree_member import tree_member_publish_image
from core.forest_member import forest_member_publish_image
from core.forest_entry import forest_entry_dispatch, forest_entry_publish, forest_host_entry_releases
from core.entry_table import ENTRY_OP_GIL_FREE
from core.native_descriptor_call import native_descriptor_report
from core.forest_entry_table import FOREST_TABLE_RF
from core.native_descriptor_abi import NativeDescriptorError
from xtrees.dart_host import DART_HOST
from ensemble.host.rf_oracle import (
    RF_ORACLE_HOST_SABOTAGE,
    RfHostForest,
    rf_host_fit,
)
from core.mojolearn_rf_host_run import (
    RF_HOST_EXPORTS,
    _rf_export_into_native,
    _rf_export_verify_native,
    _rf_fit_forest,
    _rf_predict_walk,
    RfHostForestImage,
    rf_host_fit_handle,
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
    """`_rf_classifier_fit` / `_rf_regressor_fit` of the GPU binding for the
    Python entries (which hold the GIL): `rf_host_fit_handle` (core: the same
    slot checks in the same words, the host fit, kept in RF_HOST_EXPORTS)
    with the GIL released around it. Returns the export handle."""
    var handle = 0
    with GILReleased(Python()):
        handle = rf_host_fit_handle[CLASSIFIER, ROWMAJOR](params)
    return handle


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
# The data session bodies (XF_SESSION_OPEN / XF_SESSION_CLOSE / XF_FIT_SESSION) moved to
# core/mojolearn_rf_host_run.mojo on 2026-11-23 (lane mf-ada2): with the registry's own lock
# they need no GIL, so the image serves them itself.


# TOMBSTONE: rf_regressor_fit_dart_binding (the unregistered Python form of XF_FIT_DART: no caller) deleted 2026-11-22 by lane bm-trees; code recoverable at 49f2b12e24. The DART member fit body moved to core/mojolearn_rf_host_run.mojo `rf_host_fit_dart`.


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
                       num_outputs: Int, classifier: Bool) raises -> Int:
    """`_rf_predict_walk` (core) for the Python entries, which hold the GIL:
    released around the whole walk."""
    var written = 0
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


comptime _RfHostImage = RfHostForestImage


def rf_host_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo).

    ONE BODY, TWO ENTRIES (lane bm-trees, 2026-11-22; the gbdt host image's
    pattern): the image's bodies (core) never touch the GIL; a door's long
    operation (`forest_host_entry_releases`: the fits, the DART member fit,
    the predict walk, the strict predict) runs with it released around the
    whole dispatch and the GIL-free bit set; a member's worker sends
    `ENTRY_OP_GIL_FREE` and runs the same dispatch with no release. The data
    session operations are among the released ones (lane mf-ada2: their
    registry has its own lock, `forest_session_entry_releases`)."""
    if not forest_host_entry_releases(operation):
        return forest_entry_dispatch[_RfHostImage](operation, aa, na, pp, np, ff, nf, result, context, sink)
    var status = 1
    try:
        with GILReleased(Python()):
            status = forest_entry_dispatch[_RfHostImage](
                operation + ENTRY_OP_GIL_FREE, aa, na, pp, np, ff, nf, result, context, sink
            )
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
    return status


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
