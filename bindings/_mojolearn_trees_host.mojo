# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_trees` family, the ExtraTrees classifier
and regressor (the CPU training lane, phase 1, et-clf and et-reg,
2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fits are
`extratrees/estimator.mojo::fit_extra_trees_classifier_host_exact` and
`fit_extra_trees_regressor_host_exact`, the DEVICE trainer restated on the
host (`train_tree_exact`, `batched_levelalgo/builder.mojo`: the score
kernel's own sequential oracle per (node, feature), the exact `Int64` keys
the device reduces on, the keyed tie order, the device's rescue, and the
leaf kernel's arithmetic over the device's label plane), so the five model
arrays are the GPU columns' bytes, quantized regression leaves included.
The predict is `core/forest_host_predict.mojo::et_host_predict`, the walk
`bindings/_mojolearn_forest_host.mojo` runs and the forest host gate holds
to the recorded Metal predictions on seven CPUs, exported here under the
GPU binding's `et_predict` name and contract so a loaded-back model
predicts through it on a CPU-only install (that is the model column's
RELOAD check).

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/extratrees.py` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_trees":
"_mojolearn_trees_host"`): `et_classifier_fit`, `et_regressor_fit`, their
`_export` and `_rowmajor` and `_rowmajor_export` forms (the 22-slot params
list of `bindings/_mojolearn_trees.mojo`, mirrored in `extratrees.py`; the
`_export` forms return the owned-handle descriptor `_forest_protocol.py`
consumes through `forest_export` and `forest_export_release`; the
`_rowmajor` forms take a ROW-major X and transpose it on the host, the
same column-major bytes DEVIATION 2637's pinned stage produces),
`forest_export`, `forest_export_legacy`, `forest_export_release`,
`et_predict`, `trees_vendor` answering "cpu" and `trees_numeric_mode`.
Slot 20 (`device`) is accepted at 1, the only value the wrapper sends: the
GPU binding's "Extra Trees training is GPU-only" refusal is the one line
this binding drops (brief section 3.2). `inference_engine='parallel_groves'`
predicts through the resident entries, and since 2026-09-15
(et-reg-bootstrap-parallel) `forest_prepare_gpu`,
`forest_predict_resident_reuse_gpu` and `forest_release_gpu` are exported
here under those names over `core/forest_host_groves.mojo`, the grove
kernels restated on the host (`RF_INPUT=False`: no input flush). The
non-resident `et_predict_gpu_parallel`, the pool entries and the other
resident comparison arms stay absent and refuse BY NAME through
`_HostBinding`. Best-first growth (`max_leaf_nodes`, DEVIATION 466) fits
through `train_tree_exact_bestfirst` (et-clf-entropy-bestfirst, the same
day).

The sabotage arm (`-D MOJOLEARN_HOST_SABOTAGE=1`, `trees_host_sabotage`)
is `extratrees/checks/pcg_rng.mojo::PCG_HOST_SABOTAGE`: one extra draw
before every threshold, so every forest this binary fits differs.
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
from core.mojolearn_trees_host_impl import et_forest_lists
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.forest_host_strict_binding_pack import (
    forest_strict_prepare_host_binding, forest_strict_predict_host_binding, forest_strict_release_host_binding,
)
from bindings.forest_host_groves_binding_pack import (
    forest_predict_resident_host_binding,
    forest_prepare_host_binding,
    forest_release_host_binding,
)
from bindings.forest_fit_boundary_unpack import forest_fit_args_py
from core.entry_table import ENTRY_OP_GIL_FREE
from core.extra_trees_member import extra_trees_member_publish_image
from core.forest_entry import forest_entry_dispatch, forest_entry_publish, forest_host_entry_releases
from core.forest_entry_table import FOREST_TABLE_ET
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from extratrees.checks.pcg_rng import PCG_HOST_SABOTAGE
from extratrees.estimator import FitResult
# RUN (lane bm-trees, 2026-11-22): the fit, the export registry, the predict
# walk and the image's entry operations are typed Mojo in core.
from core.mojolearn_et_host_run import (
    ET_HOST_EXPORTS,
    EtHostForest,
    et_host_export_into,
    et_host_fit_handle,
    et_host_predict_walk,
)


def _index(value: PythonObject) raises -> Int:
    # An exact int (every address, count and slot the wrappers pass) is its
    # own operator.index: read it directly (p3-trees: two attribute reads, a
    # class-name string and a module lookup per integer argument before).
    if _exact_type(value) is _exact_type(PythonObject(0)):
        return Int(py=value)
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("trees host: integers expected, not booleans")
    var operator_module = native_loaded_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def trees_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def trees_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "trees host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_trees_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `trees_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def trees_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary burns one extra RNG draw before every split
    threshold on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative
    control; `extratrees/checks/pcg_rng.mojo::PCG_HOST_SABOTAGE`)."""
    return PythonObject(PCG_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def trees_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def trees_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def _forest_out(result: FitResult) raises -> PythonObject:
    """`bindings/_mojolearn_trees.mojo::_forest_out`, the same six lists."""
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    et_forest_lists(result, offsets, colid, quesval, left_child, leaves)
    var meta = Python.list()
    meta.append(PythonObject(Int(result.forest.n_trees)))
    meta.append(PythonObject(Int(result.forest.num_outputs)))
    meta.append(PythonObject(1 if result.depth_cap_bound else 0))
    meta.append(PythonObject(Int(result.plan.params.max_depth)))
    meta.append(PythonObject(result.plan.max_features_count))
    meta.append(PythonObject(Int(result.plan.n_sampled_rows)))
    var out = Python.list()
    out.append(forest_export_py_list_i64(offsets))
    out.append(forest_export_py_list_i64(colid))
    out.append(forest_export_py_list_f64(quesval))
    out.append(forest_export_py_list_i64(left_child))
    out.append(forest_export_py_list_f64(leaves))
    out.append(meta)
    return out



def _kept_forest_out(handle: Int) raises -> PythonObject:
    # The kept fit's diagnostic lists. A helper of its own so the Dict
    # lookup (DictKeyError) and the list build (Error) both raise Error to
    # the caller's try/finally, which cannot mix the two error types.
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
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


def _et_handle_py(handle: Int, export: Bool) raises -> PythonObject:
    """PACK of a kept fit for the Python entries: the export descriptor
    (`*_export`), or the diagnostic lists of the kept fit, released (the
    plain entries; the lists `_forest_out` built from the fit)."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
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


def _et_fit_py[CLASSIFIER: Bool, ROWMAJOR: Bool](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject, export: Bool,
    weights: Int = 0, tree_start: Int = 0,
) raises -> PythonObject:
    """The Python fit entries: UNPACK (`forest_fit_args_py`), the native
    fit, PACK (`_et_handle_py`)."""
    var args = forest_fit_args_py(x_addr, y_addr, params, PythonObject(None))
    args.weights = weights
    args.tree_start = tree_start
    # The GIL released around the whole fit (its slot checks included: they
    # read plain words and raise in words) and the retain (its registry has
    # its own lock word, core/forest_export_run.mojo).
    var handle = 0
    with GILReleased(Python()):
        handle = et_host_fit_handle[CLASSIFIER, ROWMAJOR](args)
    return _et_handle_py(handle, export)


def et_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[True, False](x_addr, y_addr, params, EXPORT)


def et_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[False, False](x_addr, y_addr, params, EXPORT)


def et_forest_export_binding(
    handle: PythonObject,
    offsets: PythonObject,
    columns: PythonObject,
    thresholds: PythonObject,
    left: PythonObject,
    leaves: PythonObject,
    counts: PythonObject,
) raises -> PythonObject:
    """`forest_export` (UNPACK of `et_host_export_into`)."""
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var dst: List[Int] = [_index(offsets), _index(columns), _index(thresholds), _index(left), _index(leaves)]
    et_host_export_into(_index(handle), dst, _index(counts[0]), _index(counts[1]), _index(counts[2]))
    return PythonObject(None)


def et_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    return _kept_forest_out(_index(handle))


def et_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    ET_HOST_EXPORTS.get_or_create_ptr()[].release(_index(handle))
    return PythonObject(None)


def et_predict_native(model: List[Int], x_address: Int, out_addr: Int, n_rows: Int, n_features: Int,
                      n_trees: Int, num_outputs: Int) raises -> Int:
    """`et_host_predict_walk` (core) for the Python entry, which holds the
    GIL: released around the whole walk, its checks included (they read
    plain words)."""
    var written = 0
    with GILReleased(Python()):
        written = et_host_predict_walk(model, x_address, out_addr, n_rows, n_features, n_trees, num_outputs)
    return written


def et_predict_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`et_predict` (UNPACK of `et_predict_native`): `params` is `[n_rows,
    n_features, n_trees, num_outputs]`."""
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var model: List[Int] = [_index(offsets_addr), _index(colid_addr), _index(quesval_addr),
                            _index(left_child_addr), _index(leaves_addr)]
    return PythonObject(et_predict_native(model, _index(x_addr), _index(out_addr), _index(params[0]),
                                          _index(params[1]), _index(params[2]), _index(params[3])))


def resident_prepare_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu`: the resident parallel_groves snapshot, on the host."""
    return forest_prepare_host_binding[False](
        offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr, params
    )


def resident_predict_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu`: the grove reduction, on the host."""
    return forest_predict_resident_host_binding[False](handle, x_addr, out_addr, params)


def resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu`."""
    return forest_release_host_binding[False](handle)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_trees_host.mojo


def et_host_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo).

    ONE BODY, TWO ENTRIES (lane bm-trees, 2026-11-22; the gbdt host image's
    pattern): the operations of `EtHostForest` (core/mojolearn_et_host_run.mojo)
    never touch the GIL; a door's long operation (`forest_host_entry_releases`:
    the fit, the predict walk, the strict predict) runs with it released
    around the whole dispatch and the GIL-free bit set; a member's worker
    sends `ENTRY_OP_GIL_FREE` and runs the same dispatch with no release."""
    if not forest_host_entry_releases(operation):
        return forest_entry_dispatch[EtHostForest](operation, aa, na, pp, np, ff, nf, result, context, sink)
    var status = 1
    try:
        with GILReleased(Python()):
            status = forest_entry_dispatch[EtHostForest](
                operation + ENTRY_OP_GIL_FREE, aa, na, pp, np, ff, nf, result, context, sink
            )
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
    return status


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_trees_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).

def build__mojolearn_trees_host() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_trees_host")
        # The native forest entry (core/forest_entry_table.mojo) under the cpu key.
        forest_entry_publish["MojolearnForestEntryOwnerEtHostV1"](FOREST_TABLE_ET, "cpu", et_host_forest_entry_v1)
        # The families' member entries (core/member_abi.mojo) over this forest entry.
        extra_trees_member_publish_image("cpu")
        module.def_function[trees_host_numeric_mode_binding]("trees_host_numeric_mode")
        module.def_function[trees_host_vendor_binding]("trees_host_vendor")
        module.def_function[trees_host_column_binding]("trees_host_column")
        module.def_function[trees_host_sabotage_binding]("trees_host_sabotage")
        module.def_function[trees_vendor_binding]("trees_vendor")
        module.def_function[trees_numeric_mode_binding]("trees_numeric_mode")
        module.def_function[et_classifier_fit_binding[False]]("et_classifier_fit")
        module.def_function[et_classifier_fit_binding[True]]("et_classifier_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: TREES_LEGACY_PYTHON_ENTRIES (DELETED: no caller; the door fits through the forest entry table) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
        # Restore: git apply experiments/removed/TREES_LEGACY_PYTHON_ENTRIES.patch
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_regressor_fit_binding[False]]("et_regressor_fit")
        module.def_function[et_regressor_fit_binding[True]]("et_regressor_fit_export")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_forest_export_binding]("forest_export")
        module.def_function[et_forest_export_legacy_binding]("forest_export_legacy")
        module.def_function[et_forest_export_release_binding]("forest_export_release")
        # TOMBSTONE: the legacy-arm rf/et fits and rf_regressor_fit_session_export (dead native entries: reached only by the deleted MOJOLEARN_FOREST_EXPORT=legacy arm / no caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        module.def_function[et_predict_binding]("et_predict")
        module.def_function[resident_prepare_binding]("forest_prepare_gpu")
        module.def_function[resident_predict_binding]("forest_predict_resident_reuse_gpu")
        module.def_function[resident_release_binding]("forest_release_gpu")
        # The strict snapshot of the sequential engine (N044, ne1-a-trees).
        module.def_function[forest_strict_prepare_host_binding[False]]("forest_strict_prepare")
        module.def_function[forest_strict_predict_host_binding[False]]("forest_strict_predict")
        module.def_function[forest_strict_release_host_binding[False]]("forest_strict_release")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_trees_host: ", error))
