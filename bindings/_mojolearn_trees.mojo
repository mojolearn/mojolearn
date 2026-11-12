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
from core.hostptr import list_f32, list_i32
from std.python.bindings import PythonModuleBuilder

from forest_inference_binding import (
    forest_prepare_gpu_binding, forest_release_gpu_binding,
    forest_predict_resident_into_gpu_binding,
    forest_predict_resident_labels_gpu_binding,
    forest_stage_input_gpu_binding,
    forest_unstage_input_gpu_binding,
    forest_ordered_resident_binding,
    forest_identical_fused_labels_binding,
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

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnTreesContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnTreesContextFastV3"


from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor_device,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_GAMMA,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MSE,
    CRITERION_POISSON,
)
from extratrees.impl.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.impl.randomforest.randomforest import Forest, forest_vote_host
from core.forest_inference_model import (
    resident_prepare, resident_predict_into, resident_release,
)


# DEVIATION 2482: fit/export ownership is separate from inference residency.
from std.sys.compile import is_defined
from forest_export_binding import (
    ForestExportRegistry, validate_forest_export_destinations,
    copy_forest_export_leaves, forest_export_py_list_f64,
    forest_export_py_list_i64,
)
from core.mojolearn_trees_host_impl import et_export_into, et_forest_lists
from core.shared_slot import SharedSlot
comptime ET_EXPORTS = SharedSlot[ForestExportRegistry[FitResult], ("MojolearnETFitExportIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnETFitExportFastV1")]


# TOMBSTONE: the legacy-arm et fits, et_predict_gpu_parallel, trees_shap_pair_count, the forest comparison arms and probes (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


def et_legacy_meta(result: FitResult) -> List[Int64]:
    """The six metadata words `_forest_out` reports (and the export keeps)."""
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    return meta^


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


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _copy_f32(addr: Int, n: Int) raises -> List[Float32]:
    """Borrowed host memory into an owned List, read while the GIL-holding
    caller keeps the array alive (the `_arrays.py` contract)."""
    # one memcpy (no per-element host loop)
    return list_f32(_f32_ptr(addr), n)


comptime N_FIT_PARAMS = 22
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words, because a silent reordering here is a
wrong answer rather than a failure:

     0  n_rows
     1  n_features
     2  n_classes        (classifier only; MUST be 0 for the regressor)
     3  n_estimators
     4  max_depth        (-1 = sklearn's None)
     5  min_samples_split
     6  min_samples_leaf
     7  min_weight_fraction_leaf  (float)
     8  max_features_spec         (positive count, 0 = fraction at slot 9,
                                   -1 sqrt, -2 log2, -3 all -- the
                                   estimator.mojo sentinels)
     9  max_features_fraction     (float)
    10  min_impurity_decrease     (float)
    11  bootstrap        (0/1)
    12  oob_score        (0/1)
    13  random_state
    14  warm_start       (0/1)
    15  ccp_alpha        (float)
    16  has_class_weight (0/1)
    17  has_monotonic_cst(0/1)
    18  max_samples      (sklearn's max_samples RESOLVED TO A COUNT by the
                          wrapper: 0 = None; honoured with bootstrap=1,
                          refused by name otherwise -- DEVIATION 460. This
                          slot was a 0/1 `max_samples_set` flag while
                          bootstrap was refused)
    19  max_leaf_nodes   (-1 = sklearn's None)
    20  device           (must be 1: GPU; CPU training retired)
    21  criterion        (the `decisiontree.mojo` CRITERION_* code:
                          0 GINI, 1 ENTROPY for the classifier; 2 MSE for
                          the regressor -- DEVIATION 459)

Every refused sklearn parameter RIDES THROUGH so the refusal fires in
`refuse_unported` by name, in one place, rather than being re-decided at this
boundary. The criterion RIDES AS SLOT 21 since DEVIATION 459 (it did
not while each entry point had exactly one criterion): `et_classifier_fit`
admits GINI and ENTROPY, `et_regressor_fit` admits MSE, and the OTHER
criteria are refused by name in the WRAPPER, which cites the same
NOT_IMPLEMENTED.tsv rows validity_check does.
"""


def _config_from(
    params: ForestFitArgs, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """Slots 3-19 and 21 written OVER `base`. Read under the GIL.

    `base` carries the defaults the slots then overwrite; since DEVIATION
    459 the criterion is slot 21 and `base`'s criterion is only a default
    the slot replaces. `et_classifier_fit` passes `ExtraTreesConfig()` and
    `et_regressor_fit` passes `ExtraTreesConfig().for_regression()`. Every
    slot overwrites its field AFTER the base is taken,
    so no default can shadow what the caller sent -- DEVIATION 458 is what
    happened when the regressor applied `for_regression()` the other way round
    and its `max_features_spec = ALL` default overwrote slots 8-9 of every
    regressor fit.
    """
    var config = base.copy()
    config.n_estimators = Int32(params.i(3))
    config.max_depth = Int32(params.i(4))
    config.min_samples_split = Int32(params.i(5))
    config.min_samples_leaf = Int32(params.i(6))
    config.min_weight_fraction_leaf = params.f(7)
    config.max_features_spec = params.i(8)
    config.max_features_fraction = params.f(9)
    config.min_impurity_decrease = Float32(params.f(10))
    config.bootstrap = params.i(11) != 0
    config.oob_score = params.i(12) != 0
    config.random_state = UInt64(params.i(13))
    config.warm_start = params.i(14) != 0
    config.ccp_alpha = params.f(15)
    config.has_class_weight = params.i(16) != 0
    config.has_monotonic_cst = params.i(17) != 0
    config.max_samples = params.i(18)
    config.max_leaf_nodes = Int32(params.i(19))
    # slot 21, DEVIATION 459: the criterion code, written over the base's
    # default like every other slot.
    config.criterion = Int32(params.i(21))
    return config^


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
    """Fit `ExtraTreesClassifier`. `x` is COLUMN-major float32
    (n_rows * n_features), `y` is float32 class CODES in [0, n_classes).
    See `N_FIT_PARAMS`. Returns the export handle (lane calltable-forest:
    every fit is kept in ET_EXPORTS; `_et_handle_py` reads the diagnostic
    lists)."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_classifier_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_classes = params.i(2)
    if params.f(20) != Float64(1):
        raise Error("Extra Trees training is GPU-only; device (slot 20) must be 1")
    var config = _config_from(params, ExtraTreesConfig())
    if config.criterion != CRITERION_GINI and config.criterion != CRITERION_ENTROPY:
        raise Error(
            "et_classifier_fit: criterion (slot 21) must be GINI (0) or"
            " ENTROPY (1); got " + String(config.criterion)
        )
    # DEVIATION 2480: optional host boundary attribution, no added drains.
    var times = StageTimes()
    var total_start = times.start()
    var stamp = times.start()
    # DEVIATION 2481: caller keeps the immutable feature buffer alive through
    # this synchronous fit; only labels still require owned staging.
    var x_pointer = params.x
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(params.y, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        result = fit_extra_trees_classifier_device(
            ctx, x, y, Int32(n_rows), Int32(n_features), Int32(n_classes),
            config, x_addr=x_pointer, x_row_major=ROWMAJOR, tree_start=params.tree_start,
            weights_addr=params.weights,
        )
        ctx.synchronize()
    times.stop_host("boundary_device_fit_and_context", stamp)
    stamp = times.start()
    var output = _retain_et_handle(result^)
    times.stop_host("boundary_export_handle", stamp)
    times.stop_host("fit_total", total_start)
    times.report()
    return output


def _et_regressor_fit[ROWMAJOR: Bool = False](params: ForestFitArgs) raises -> Int:
    """Fit `ExtraTreesRegressor`. Same contract; slot 2 MUST be 0.
    `weights_addr` (lane sk-final, 0: none) as `et_regressor_fit_weighted`."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_regressor_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if params.i(2) != 0:
        raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = params.i(0)
    var n_features = params.i(1)
    if params.f(20) != Float64(1):
        raise Error("Extra Trees training is GPU-only; device (slot 20) must be 1")
    var config = _config_from(params, ExtraTreesConfig().for_regression())
    if config.criterion != CRITERION_MSE and config.criterion != CRITERION_POISSON and (
        config.criterion != CRITERION_GAMMA and config.criterion != CRITERION_INVERSE_GAUSSIAN
    ):
        raise Error(
            "et_regressor_fit: criterion (slot 21) must be MSE (2), POISSON (4), GAMMA (5) or INVERSE_GAUSSIAN (6); got "
            + String(config.criterion)
        )
    # DEVIATION 2480: optional host boundary attribution, no added drains.
    var times = StageTimes()
    var total_start = times.start()
    var stamp = times.start()
    # DEVIATION 2481: caller keeps the immutable feature buffer alive through
    # this synchronous fit; only labels still require owned staging.
    var x_pointer = params.x
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(params.y, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        result = fit_extra_trees_regressor_device(
            ctx, x, y, Int32(n_rows), Int32(n_features), config,
            x_addr=x_pointer, x_row_major=ROWMAJOR, tree_start=params.tree_start,
            weights_addr=params.weights,
        )
        ctx.synchronize()
    times.stop_host("boundary_device_fit_and_context", stamp)
    stamp = times.start()
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


def et_predict_native(model: List[Int], x_address: Int, out_address: Int, n_rows: Int, n_features: Int,
                      n_trees: Int, num_outputs: Int) raises -> Int:
    """The forest's averaged vote per row, through the IMPLEMENTED traversal
    (XF_PREDICT).

    Model arrays are int32/float32 as `_forest_out` laid them out. `x` here
    is ROW-major (the traversal reads `row[offset + colid]`). `out` is
    n_rows * num_outputs float32 and receives `forest_vote_host`'s average --
    per-class probabilities for the classifier (argmax is the wrapper's,
    exactly as `RandomForest::predict` argmaxes over `predict_proba`), the
    mean prediction for the regressor. Returns rows written.
    """
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")

    var offsets_p = _i32_ptr(model[0])
    var colid_p = _i32_ptr(model[1])
    var quesval_p = _f32_ptr(model[2])
    var left_p = _i32_ptr(model[3])
    var leaves_p = _f32_ptr(model[4])
    var xp = _f32_ptr(x_address)
    var op = _f32_ptr(out_address)

    # cpu-gpu-cleanup t-gbdt: the `sequential` engine runs on the device.
    # The host walk (`core/forest_host_predict.mojo`'s `et_host_predict`,
    # DEVIATION 2900) is the CPU-only install's (`_mojolearn_trees_host`);
    # here `forest_vote_host`'s arithmetic -- zero, add every tree's leaf in
    # increasing tree order, divide by `Float32(n_trees)` -- is
    # `forest_ordered_kernel`, one thread per (row, output), through a
    # one-call ordered snapshot. The GIL stays held for the registry.
    if n_rows == 0:
        return 0
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error("et_predict: tree_offsets must start at 0 and hold at least one node per tree")
    if n_nodes > 2147483647 // num_outputs:
        raise Error("et_predict: node/output count exceeds Int32")
    # one memcpy per array (no per-node host loop); the snapshot uploads once
    var offsets = list_i32(offsets_p, n_trees + 1)
    var columns = list_i32(colid_p, n_nodes)
    var thresholds = list_f32(quesval_p, n_nodes)
    var left = list_i32(left_p, n_nodes)
    var leaves = list_f32(leaves_p, n_nodes * num_outputs)
    var handle = resident_prepare[False](
        offsets, columns, thresholds, left, leaves, n_features, num_outputs, True
    )
    try:
        resident_predict_into[False](
            handle,
            xp.unsafe_origin_cast[MutAnyOrigin](),
            op.unsafe_origin_cast[MutAnyOrigin](),
            n_rows, n_features, num_outputs, True,
        )
    except e:
        resident_release[False](handle)
        raise e
    resident_release[False](handle)
    return n_rows


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
            return registry[].descriptor_py(handle)
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
