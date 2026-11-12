# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the ExtraTrees estimators (`extratrees/`).

Kept in its OWN extension for the same reason `_mojolearn_estimators` and
`_mojolearn_gbdt` are: an independently changing binding stops being a merge
point, and the extratrees lane changes independently of both. Arrays cross as
borrowed host-buffer addresses. Training contexts live for one call; no input
pointer is retained. The default fit returns an owned native export handle and
counts. Python allocates the five model Arrays, exports their bytes and releases
the handle in a finally block. The model then lives in those Python-owned Arrays.

THE MODEL LAYOUT REMAINS FLAT ARRAYS: per-node `colid` / `quesval` /
`left_child_id`, the flat
`vector_leaf`, and a `tree_offsets` prefix so tree `t` is the node range
`[offsets[t], offsets[t+1])`. `et_predict` rebuilds the forest from those
arrays and calls the IMPLEMENTED `forest_vote_host` -- the traversal is
`decisiontree.cuh:394-413` through `flatnode.mojo`, not a reimplementation at
this boundary. `instance_count` and `best_metric_val` are not carried: the
traversal never reads either (`flatnode.mojo` says so of `best_metric_val`
explicitly), and a field the boundary carries but nothing reads is the
present-but-dead state rule 3 forbids.

DEVIATION 2482: caller-buffer export is the default after all-tier same-fit
array/archive gates and a large-data export-only A/B on Metal. Retained List
entrypoints are comparison arms; they do not run on the default fit path.
"""

from core.experiment_guards_trees import TREES_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from ensemble.device_finite import FOREST_DEVICE_FINITE
from ensemble.instruments import StageTimes
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from forest_inference_binding import (
    forest_prepare_gpu_binding, forest_release_gpu_binding,
    forest_predict_resident_into_gpu_binding,
    forest_predict_resident_labels_gpu_binding,
    forest_stage_input_gpu_binding,
    forest_unstage_input_gpu_binding,
    forest_ordered_resident_binding,
    forest_identical_fused_labels_binding,
)
from core.forest_inference_run import (
    forest_prepare_gpu_native, forest_predict_resident_into_gpu_native,
    forest_predict_resident_labels_gpu_native, forest_gpu_resident_caps,
)
from bindings.forest_fit_boundary import forest_fit_args_py
from core.forest_fit_request import ForestFitArgs, XF_CAP_DEVICE_SCAN, forest_export_verify_lists
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_entry_table import FOREST_TABLE_ET
from core.native_descriptor_abi import NativeDescriptorError
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import shared_class_counts_mask
from extratrees.estimator import FitResult
from core.forest_inference_model import resident_release


# DEVIATION 2482: fit/export ownership is separate from inference residency.
from std.sys.compile import is_defined
from forest_export_binding import (
    forest_export_descriptor_py,
    ForestExportRegistry, validate_forest_export_destinations,
    copy_forest_export_leaves, forest_export_py_list_f64,
    forest_export_py_list_i64,
)
from core.mojolearn_trees_host_impl import et_export_into, et_forest_lists
from extratrees.mojolearn_trees_run import (
    _et_classifier_fit_run,
    _et_regressor_fit_run,
    _f32_ptr,
    _i32_ptr,
    et_legacy_meta,
    et_predict_native,
)
from core.shared_slot import SharedSlot
comptime ET_EXPORTS = SharedSlot[ForestExportRegistry[FitResult], ("MojolearnETFitExportIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnETFitExportFastV1")]


# TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


def _retain_et_handle(var result: FitResult) raises -> Int:
    var trees = len(result.forest.trees)
    var nodes = 0
    for t in range(trees):  # small-loop(trees: fitted trees, one node count each): export shape bookkeeping, no row or node data
        var tree_nodes = result.forest.trees[t].num_nodes()
        nodes += tree_nodes
        if len(result.forest.trees[t].vector_leaf) != tree_nodes * Int(result.forest.num_outputs):
            raise Error("fitted ET leaf storage differs from export dimensions")
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    var outputs = Int(result.forest.num_outputs)
    if trees != Int(result.forest.n_trees):
        raise Error("fitted ET tree metadata differs from export dimensions")
    return ET_EXPORTS.get_or_create_ptr()[].insert_native(result^, trees, nodes, outputs, meta^)


def _et_export_into_native(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the kept fit into the caller's five arrays."""
    var registry = ET_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = _i32_ptr(dst[0])
    var cp = _i32_ptr(dst[1])
    var tp = _f32_ptr(dst[2])
    var lp = _i32_ptr(dst[3])
    var times = StageTimes()
    var stamp = times.start()
    et_export_into(
        registry[].entries[id].model, op, cp, tp, lp, dst[4],
        registry[].entries[id].outputs,
    )
    times.stop_host("boundary_export_into", stamp)
    times.report()


def _et_export_verify_native(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = ET_EXPORTS.get_or_create_ptr()
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    ref entry = registry[].entries[id]
    var legacy = et_legacy_meta(entry.model)
    if len(legacy) != len(entry.meta):
        return -1
    for k in range(len(legacy)):  # small-loop(meta: six export metadata words)
        if legacy[k] != entry.meta[k]:
            return -1
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    et_forest_lists(entry.model, offsets, colid, quesval, left_child, leaves)
    return forest_export_verify_lists(offsets, colid, quesval, left_child, leaves, dst)


def et_forest_export_binding(handle: PythonObject, offsets: PythonObject,
    columns: PythonObject, thresholds: PythonObject, left: PythonObject,
    leaves: PythonObject, counts: PythonObject) raises -> PythonObject:
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var dst: List[Int] = [Int(py=offsets), Int(py=columns), Int(py=thresholds), Int(py=left), Int(py=leaves)]
    _et_export_into_native(Int(py=handle), dst, Int(py=counts[0]), Int(py=counts[1]), Int(py=counts[2]))
    return PythonObject(None)


def et_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    var registry = ET_EXPORTS.get_or_create_ptr()
    var id = Int(py=handle)
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    return _forest_out(registry[].entries[id].model)


def et_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    ET_EXPORTS.get_or_create_ptr()[].release(Int(py=handle))
    return PythonObject(None)


def _forest_out(result: FitResult) raises -> PythonObject:
    """The fitted forest as `[offsets, colid, quesval, left_child, leaves,
    meta]`, all Python lists.

    `meta` is `[n_trees, num_outputs, depth_cap_bound, resolved_max_depth,
    max_features_count, n_sampled_rows]` -- the last four are what
    `FitResult`/`FitPlan` exist to report rather than hide (`n_sampled_rows`
    is the bootstrap sample size the fit used, 0 without bootstrap;
    DEVIATION 460)."""
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


def _et_classifier_fit[ROWMAJOR: Bool = False](params: ForestFitArgs) raises -> Int:
    """Fit `ExtraTreesClassifier` (`_et_classifier_fit_run`, with the GIL
    released around it) and keep the fit. Returns the export handle (lane
    calltable-forest: every fit is kept in ET_EXPORTS; `_et_handle_py` reads
    the diagnostic lists)."""
    var times = StageTimes()
    var total_start = 0
    var result: FitResult
    with GILReleased(Python()):
        result = _et_classifier_fit_run[ROWMAJOR](params, times, total_start)
    var stamp = times.start()
    var output = _retain_et_handle(result^)
    times.stop_host("boundary_export_handle", stamp)
    times.stop_host("fit_total", total_start)
    times.report()
    return output


def _et_regressor_fit[ROWMAJOR: Bool = False](params: ForestFitArgs) raises -> Int:
    """Fit `ExtraTreesRegressor` (`_et_regressor_fit_run`, with the GIL
    released around it) and keep the fit. Returns the export handle."""
    var times = StageTimes()
    var total_start = 0
    var result: FitResult
    with GILReleased(Python()):
        result = _et_regressor_fit_run[ROWMAJOR](params, times, total_start)
    var stamp = times.start()
    var output = _retain_et_handle(result^)
    times.stop_host("boundary_export_handle", stamp)
    times.stop_host("fit_total", total_start)
    times.report()
    return output


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
    n_features, n_trees, num_outputs]`. Returns rows written."""
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var model: List[Int] = [Int(py=offsets_addr), Int(py=colid_addr), Int(py=quesval_addr),
                            Int(py=left_child_addr), Int(py=leaves_addr)]
    return PythonObject(et_predict_native(model, Int(py=x_addr), Int(py=out_addr), Int(py=params[0]),
                                          Int(py=params[1]), Int(py=params[2]), Int(py=params[3])))


# TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


def trees_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def trees_stage_copy_policy_binding() raises -> PythonObject:
    comptime if is_defined["MOJOLEARN_ET_SCALAR_STAGE_COMPARE"]():
        return PythonObject("scalar_reference")
    return PythonObject("vector_bytes")


def trees_device_finite_scan_binding() raises -> PythonObject:
    """1 when this build's ExtraTrees fits refuse a non-finite X cell through
    a device scan in `upload_dataset` (FOREST_DEVICE_FINITE, FAST on Apple),
    so the Python fit skips its host scan of the same cells; 0 otherwise."""
    return PythonObject(1 if FOREST_DEVICE_FINITE else 0)


def trees_numeric_mode_binding() raises -> PythonObject:
    """Compiled numeric tier, independent of the package directory label."""
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def trees_shared_counts_mask_binding() raises -> PythonObject:
    """Compiled score policy: bits0..3 correspond to widths4/8/16/32."""
    return PythonObject(shared_class_counts_mask())


def _et_fit_native[CLASSIFIER: Bool, ROWMAJOR: Bool](args: ForestFitArgs) raises -> Int:
    """One fit (the forest entry's XF_FIT): the export handle."""
    comptime if CLASSIFIER:
        return _et_classifier_fit[ROWMAJOR](args)
    else:
        return _et_regressor_fit[ROWMAJOR](args)


def _kept_forest_out(handle: Int) raises -> PythonObject:
    # The kept fit's diagnostic lists. A helper of its own so the Dict
    # lookup (DictKeyError) and the list build (Error) both raise Error to
    # the caller's try/finally, which cannot mix the two error types.
    var registry = ET_EXPORTS.get_or_create_ptr()
    return _forest_out(registry[].entries[handle].model)


def _et_handle_py(handle: Int, export: Bool) raises -> PythonObject:
    """PACK of a kept fit for the Python entries: the export descriptor
    (`*_export`), or the diagnostic lists of the kept fit, released (the
    plain entries; the lists `_forest_out` built from the fit)."""
    var registry = ET_EXPORTS.get_or_create_ptr()
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
    return _et_handle_py(_et_fit_native[CLASSIFIER, ROWMAJOR](args), export)


def et_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[True, False](x_addr, y_addr, params, EXPORT)


def et_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _et_fit_py[False, False](x_addr, y_addr, params, EXPORT)


struct EtGpuForest(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest): the bodies the Python entries above unpack into."""

    @staticmethod
    def caps() -> Int:
        var caps = forest_gpu_resident_caps()
        comptime if FOREST_DEVICE_FINITE:
            caps |= XF_CAP_DEVICE_SCAN
        return caps

    @staticmethod
    def dt_default_bins() -> Int:
        return 128

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return _et_fit_native[True, True](args)
            return _et_fit_native[True, False](args)
        if row_major:
            return _et_fit_native[False, True](args)
        return _et_fit_native[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the extra-trees binding has no DART member fit")

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        ET_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        _et_export_into_native(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return _et_export_verify_native(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        ET_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool) raises -> Int:
        return et_predict_native(model, x, out_addr, rows, cols, trees, outputs)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_gpu_native[False](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_into_gpu_native[False, True](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_labels_gpu_native[False](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_release(handle: Int) raises:
        resident_release[False](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: rebuild the forest host binding")

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: rebuild the forest host binding")

    @staticmethod
    def strict_release(handle: Int) raises:
        raise Error("RuntimeError: rebuild the forest host binding")


def et_gpu_forest_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's published forest entry (core/forest_entry_table.mojo)."""
    return forest_entry_dispatch[EtGpuForest](operation, aa, na, pp, np, ff, nf, result, context, sink)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_trees.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_trees() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_trees")
        # The native forest entry (core/forest_entry_table.mojo) under this image's key.
        forest_entry_publish["MojolearnForestEntryOwnerEtGpuV1"](FOREST_TABLE_ET, COMPILED_VENDOR, et_gpu_forest_entry_v1)
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[trees_vendor_binding]("trees_vendor")
        m.def_function[trees_numeric_mode_binding]("trees_numeric_mode")
        m.def_function[trees_device_finite_scan_binding]("trees_device_finite_scan")
        m.def_function[trees_stage_copy_policy_binding]("trees_stage_copy_policy")
        m.def_function[trees_shared_counts_mask_binding]("trees_shared_counts_mask")
        m.def_function[et_classifier_fit_binding[False]]("et_classifier_fit")
        m.def_function[et_classifier_fit_binding[True]]("et_classifier_fit_export")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: TREES_LEGACY_PYTHON_ENTRIES (DELETED: no caller; the door fits through the forest entry table) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
        # Restore: git apply experiments/removed/TREES_LEGACY_PYTHON_ENTRIES.patch
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[et_regressor_fit_binding[False]]("et_regressor_fit")
        m.def_function[et_regressor_fit_binding[True]]("et_regressor_fit_export")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[et_predict_binding]("et_predict")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_ordered_resident_binding]("forest_ordered_resident")
        # T38/C50 — NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        m.def_function[forest_identical_fused_labels_binding]("forest_identical_fused_labels")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[et_forest_export_binding]("forest_export")
        m.def_function[et_forest_export_legacy_binding]("forest_export_legacy")
        m.def_function[et_forest_export_release_binding]("forest_export_release")
        # every snapshot prepared here runs on the registry's process context and
        # borrows its one I/O workspace (core/forest_inference_model.mojo,
        # FOREST_PER_MODEL_IO; gap-fails2 2026-10-02)
        m.def_function[forest_prepare_gpu_binding[False]]("forest_prepare_gpu")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_release_gpu_binding[False]]("forest_release_gpu")
        # TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[forest_predict_resident_into_gpu_binding[False, True]]("forest_predict_resident_reuse_gpu")
        m.def_function[forest_predict_resident_labels_gpu_binding[False]]("forest_predict_resident_labels_gpu")
        m.def_function[forest_stage_input_gpu_binding[False]]("forest_stage_input_gpu")
        m.def_function[forest_unstage_input_gpu_binding[False]]("forest_unstage_input_gpu")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_trees: ", e))
