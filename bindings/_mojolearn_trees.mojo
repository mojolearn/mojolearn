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
arrays and calls the IMPLEMENTED `forest_vote` -- the traversal is
`decisiontree.cuh:394-413` through `flatnode.mojo`, not a reimplementation at
this boundary. `instance_count` and `best_metric_val` are not carried: the
traversal never reads either (`flatnode.mojo` says so of `best_metric_val`
explicitly), and a field the boundary carries but nothing reads is the
present-but-dead state rule 3 forbids.

DEVIATION 2482: caller-buffer export is the default after all-tier same-fit
array/archive gates and a large-data export-only A/B on Metal. Retained List
entrypoints are comparison arms; they do not run on the default fit path.
"""

from std.os import abort
from ensemble.instruments import StageTimes
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from forest_inference_binding import (
    forest_prepare_gpu_binding, forest_predict_resident_gpu_binding, forest_release_gpu_binding,
    forest_vector_groves_binding, forest_predict_resident_into_gpu_binding,
    forest_resident_layout_binding,
)
from core.forest_inference import forest_predict_gpu
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import shared_class_counts_mask

from max.gpu.host import DeviceContext

from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor_device,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE,
)
from extratrees.impl.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.impl.randomforest.randomforest import Forest, forest_vote


# DEVIATION 2482: fit/export ownership is separate from inference residency.
from std.ffi import _Global
from forest_export_binding import (
    ForestExportRegistry, validate_forest_export_destinations,
    copy_forest_export_leaves,
)
comptime ET_EXPORTS = _Global[StorageType=ForestExportRegistry[FitResult],
    name=("MojoETFitExportIdentical" if GLOBAL_NUMERIC_MODE == 1 else
          "MojoETFitExportDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
          "MojoETFitExportFast"), init_fn=ForestExportRegistry[FitResult].__init__]


def _retain_et_export(var result: FitResult) raises -> PythonObject:
    var trees = len(result.forest.trees)
    var nodes = 0
    for tree in result.forest.trees:
        nodes += tree.num_nodes()
        if len(tree.vector_leaf) != tree.num_nodes() * Int(result.forest.num_outputs):
            raise Error("fitted ET leaf storage differs from export dimensions")
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    var outputs = Int(result.forest.num_outputs)
    if trees != Int(result.forest.n_trees):
        raise Error("fitted ET tree metadata differs from export dimensions")
    return ET_EXPORTS.get_or_create_ptr()[].insert(result^, trees, nodes, outputs, meta^)


def et_forest_export_binding(handle: PythonObject, offsets: PythonObject,
    columns: PythonObject, thresholds: PythonObject, left: PythonObject,
    leaves: PythonObject, counts: PythonObject) raises -> PythonObject:
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var id = Int(py=handle)
    var registry = ET_EXPORTS.get_or_create_ptr()
    registry[].validate(id, Int(py=counts[0]), Int(py=counts[1]), Int(py=counts[2]))
    validate_forest_export_destinations(Int(py=offsets), Int(py=columns),
        Int(py=thresholds), Int(py=left), Int(py=leaves))
    var op = _i32_ptr(Int(py=offsets))
    var cp = _i32_ptr(Int(py=columns))
    var tp = _f32_ptr(Int(py=thresholds))
    var lp = _i32_ptr(Int(py=left))
    var total = 0
    op[0] = 0
    ref model = registry[].entries[id].model
    var times = StageTimes()
    var stamp = times.start()
    for t in range(len(model.forest.trees)):
        ref tree = model.forest.trees[t]
        for i in range(tree.num_nodes()):
            ref node = tree.sparsetree[i]
            cp[total + i] = node.colid
            tp[total + i] = node.quesval
            lp[total + i] = node.left_child_id
        copy_forest_export_leaves(tree.vector_leaf, Int(py=leaves),
                                  total * registry[].entries[id].outputs)
        total += tree.num_nodes()
        op[t + 1] = Int32(total)
    times.stop_host("boundary_export_into", stamp)
    times.report()
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


def _copy_f32(addr: PythonObject, n: Int) raises -> List[Float32]:
    """Borrowed host memory into an owned List, read while the GIL-holding
    caller keeps the array alive (the `_arrays.py` contract)."""
    var p = _f32_ptr(Int(py=addr))
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(p[i])
    return out^


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
boundary. The criterion RIDES AS SLOT 21 since DEVIATION 459 (it did not
while each entry point had exactly one criterion): `et_classifier_fit`
admits GINI and ENTROPY, `et_regressor_fit` admits MSE, and the OTHER
criteria are refused by name in the WRAPPER, which cites the same
NOT_IMPLEMENTED.tsv rows validity_check does.
"""


def _config_from(
    params: PythonObject, base: ExtraTreesConfig
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
    config.n_estimators = Int32(Int(py=params[3]))
    config.max_depth = Int32(Int(py=params[4]))
    config.min_samples_split = Int32(Int(py=params[5]))
    config.min_samples_leaf = Int32(Int(py=params[6]))
    config.min_weight_fraction_leaf = Float64(py=params[7])
    config.max_features_spec = Int(py=params[8])
    config.max_features_fraction = Float64(py=params[9])
    config.min_impurity_decrease = Float32(Float64(py=params[10]))
    config.bootstrap = Int(py=params[11]) != 0
    config.oob_score = Int(py=params[12]) != 0
    config.random_state = UInt64(Int(py=params[13]))
    config.warm_start = Int(py=params[14]) != 0
    config.ccp_alpha = Float64(py=params[15])
    config.has_class_weight = Int(py=params[16]) != 0
    config.has_monotonic_cst = Int(py=params[17]) != 0
    config.max_samples = Int(py=params[18])
    config.max_leaf_nodes = Int32(Int(py=params[19]))
    # slot 21, DEVIATION 459: the criterion code, written over the base's
    # default like every other slot.
    config.criterion = Int32(Int(py=params[21]))
    return config^


def _forest_out(result: FitResult) raises -> PythonObject:
    """The fitted forest as `[offsets, colid, quesval, left_child, leaves,
    meta]`, all Python lists.

    `meta` is `[n_trees, num_outputs, depth_cap_bound, resolved_max_depth,
    max_features_count, n_sampled_rows]` -- the last four are what
    `FitResult`/`FitPlan` exist to report rather than hide (`n_sampled_rows`
    is the bootstrap sample size the fit used, 0 without bootstrap;
    DEVIATION 460)."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(result.forest.trees)):
        var n = result.forest.trees[t].num_nodes()
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            var node = result.forest.trees[t].sparsetree[i]
            colid.append(PythonObject(Int(node.colid)))
            quesval.append(PythonObject(Float64(node.quesval)))
            left_child.append(PythonObject(Int(node.left_child_id)))
        for i in range(len(result.forest.trees[t].vector_leaf)):
            leaves.append(
                PythonObject(Float64(result.forest.trees[t].vector_leaf[i]))
            )
    var meta = Python.list()
    meta.append(PythonObject(Int(result.forest.n_trees)))
    meta.append(PythonObject(Int(result.forest.num_outputs)))
    meta.append(PythonObject(1 if result.depth_cap_bound else 0))
    meta.append(PythonObject(Int(result.plan.params.max_depth)))
    meta.append(PythonObject(result.plan.max_features_count))
    meta.append(PythonObject(Int(result.plan.n_sampled_rows)))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def et_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit `ExtraTreesClassifier`. `x` is COLUMN-major float32
    (n_rows * n_features), `y` is float32 class CODES in [0, n_classes).
    See `N_FIT_PARAMS`; EXPORT selects an owned handle instead of diagnostic lists."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_classifier_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_classes = Int(py=params[2])
    if Float64(py=params[20]) != Float64(1):
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
    var x_pointer = Int(py=x_addr)
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(y_addr, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    with GILReleased(Python()):
        var ctx = DeviceContext()
        result = fit_extra_trees_classifier_device(
            ctx, x, y, Int32(n_rows), Int32(n_features), Int32(n_classes),
            config, x_addr=x_pointer,
        )
    times.stop_host("boundary_device_fit_and_context", stamp)
    stamp = times.start()
    var output: PythonObject
    comptime if EXPORT:
        output = _retain_et_export(result^)
    else:
        output = _forest_out(result)
    comptime if EXPORT:
        times.stop_host("boundary_export_handle", stamp)
    else:
        times.stop_host("boundary_python_objects", stamp)
    times.stop_host("fit_total", total_start)
    times.report()
    return output


def et_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit `ExtraTreesRegressor`. Same contract; slot 2 MUST be 0."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_regressor_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    if Float64(py=params[20]) != Float64(1):
        raise Error("Extra Trees training is GPU-only; device (slot 20) must be 1")
    var config = _config_from(params, ExtraTreesConfig().for_regression())
    if config.criterion != CRITERION_MSE:
        raise Error(
            "et_regressor_fit: criterion (slot 21) must be MSE (2); got "
            + String(config.criterion)
        )
    # DEVIATION 2480: optional host boundary attribution, no added drains.
    var times = StageTimes()
    var total_start = times.start()
    var stamp = times.start()
    # DEVIATION 2481: caller keeps the immutable feature buffer alive through
    # this synchronous fit; only labels still require owned staging.
    var x_pointer = Int(py=x_addr)
    _ = _f32_ptr(x_pointer)
    var x = List[Float32]()
    times.stop_host("boundary_x_borrow", stamp)
    stamp = times.start()
    var y = _copy_f32(y_addr, n_rows)
    times.stop_host("boundary_y_list", stamp)
    stamp = times.start()

    var result: FitResult
    with GILReleased(Python()):
        var ctx = DeviceContext()
        result = fit_extra_trees_regressor_device(
            ctx, x, y, Int32(n_rows), Int32(n_features), config,
            x_addr=x_pointer,
        )
    times.stop_host("boundary_device_fit_and_context", stamp)
    stamp = times.start()
    var output: PythonObject
    comptime if EXPORT:
        output = _retain_et_export(result^)
    else:
        output = _forest_out(result)
    comptime if EXPORT:
        times.stop_host("boundary_export_handle", stamp)
    else:
        times.stop_host("boundary_python_objects", stamp)
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
    """The forest's averaged vote per row, through the IMPLEMENTED traversal.

    Model arrays are int32/float32 as `_forest_out` laid them out (the
    wrapper exports and keeps host Arrays). `x` here is
    ROW-major (the traversal reads `row[offset + colid]`). `out` is
    n_rows * num_outputs float32 and receives `forest_vote`'s average --
    per-class probabilities for the classifier (argmax is the wrapper's,
    exactly as `RandomForest::predict` argmaxes over `predict_proba`), the
    mean prediction for the regressor. `params` is `[n_rows, n_features,
    n_trees, num_outputs]`. Returns rows written.
    """
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")

    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        # Rebuild the forest in `TreeMetaDataNode`'s own layout so the
        # traversal that runs is flatnode.mojo's, not a copy of it here.
        # `instance_count` and `best_metric_val` are zero: the traversal
        # reads neither, and the docstring above says so where a caller can
        # see it.
        var forest = Forest(Int32(num_outputs))
        for t in range(n_trees):
            var lo = Int(offsets_p[t])
            var hi = Int(offsets_p[t + 1])
            if lo < 0 or hi < lo:
                raise Error("et_predict: tree_offsets are not a prefix scan")
            var nodes = List[SparseTreeNode[DType.float32]](
                capacity=hi - lo
            )
            var vleaf = List[Float32](capacity=(hi - lo) * num_outputs)
            for i in range(lo, hi):
                nodes.append(
                    SparseTreeNode[DType.float32](
                        colid_p[i], quesval_p[i], 0.0, left_p[i], 0
                    )
                )
                for k in range(num_outputs):
                    vleaf.append(leaves_p[i * num_outputs + k])
            forest.trees.append(
                TreeMetaDataNode[DType.float32](
                    Int32(t), 0, 0, Int32(num_outputs), vleaf^, nodes^
                )
            )
        forest.n_trees = Int32(n_trees)

        var row = List[Float32](capacity=n_rows * n_features)
        for i in range(n_rows * n_features):
            row.append(xp[i])
        for r in range(n_rows):
            var vote = forest_vote(forest, row, r * n_features)
            for k in range(num_outputs):
                op[r * num_outputs + k] = vote[k]
            wrote += 1
    return PythonObject(wrote)


def et_predict_gpu_parallel_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Opt-in GPU inference: fixed 32-grove reduction graph.

    Borrowed arrays retain the existing prediction ABI; graph/finite validation
    and GPU execution are shared by RF/ET in core.forest_inference.
    """
    if len(params) != 4:
        raise Error("GPU parallel prediction expects rows, features, trees, outputs")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    var trees = Int(py=params[2])
    var outputs = Int(py=params[3])
    if rows < 0 or features < 1 or trees < 1 or outputs < 1:
        raise Error("GPU parallel prediction dimensions are invalid")
    if trees >= 2147483647 or rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("GPU parallel prediction dimensions exceed Int32 indexing")
    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var columns_p = _i32_ptr(Int(py=colid_addr))
    var thresholds_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var x_p = _f32_ptr(Int(py=x_addr))
    var out_p = _f32_ptr(Int(py=out_addr))
    var nodes = Int(offsets_p[trees])
    if nodes < 1 or nodes > 2147483647 // outputs:
        raise Error("GPU parallel prediction node/output count is invalid")
    with GILReleased(Python()):
        var offsets = List[Int32](capacity=trees + 1)
        var columns = List[Int32](capacity=nodes)
        var thresholds = List[Float32](capacity=nodes)
        var left = List[Int32](capacity=nodes)
        var leaves = List[Float32](capacity=nodes * outputs)
        var x = List[Float32](capacity=rows * features)
        for i in range(trees + 1):
            offsets.append(offsets_p[i])
        for i in range(nodes):
            columns.append(columns_p[i])
            thresholds.append(thresholds_p[i])
            left.append(left_p[i])
        for i in range(nodes * outputs):
            leaves.append(leaves_p[i])
        for i in range(rows * features):
            x.append(x_p[i])
        var ctx = DeviceContext()
        var result = forest_predict_gpu[False, True](
            ctx, offsets, columns, thresholds, left, leaves, x,
            rows, features, outputs,
        )
        for i in range(rows * outputs):
            out_p[i] = result[i]
        _ = result^
        _ = ctx^
    return PythonObject(rows)

def trees_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def trees_numeric_mode_binding() raises -> PythonObject:
    """Compiled numeric tier, independent of the package directory label."""
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def trees_shared_counts_mask_binding() raises -> PythonObject:
    """Compiled score policy: bits0..3 correspond to widths4/8/16/32."""
    return PythonObject(shared_class_counts_mask())


@export
def PyInit__mojolearn_trees() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_trees")
        m.def_function[trees_vendor_binding]("trees_vendor")
        m.def_function[trees_numeric_mode_binding]("trees_numeric_mode")
        m.def_function[trees_shared_counts_mask_binding]("trees_shared_counts_mask")
        m.def_function[et_classifier_fit_binding[False]]("et_classifier_fit")
        m.def_function[et_classifier_fit_binding[True]]("et_classifier_fit_export")
        m.def_function[et_regressor_fit_binding[False]]("et_regressor_fit")
        m.def_function[et_regressor_fit_binding[True]]("et_regressor_fit_export")
        m.def_function[et_predict_binding]("et_predict")
        m.def_function[et_predict_gpu_parallel_binding]("et_predict_gpu_parallel")
        m.def_function[forest_resident_layout_binding]("forest_resident_layout")
        m.def_function[et_forest_export_binding]("forest_export")
        m.def_function[et_forest_export_legacy_binding]("forest_export_legacy")
        m.def_function[et_forest_export_release_binding]("forest_export_release")
        m.def_function[forest_prepare_gpu_binding[False]]("forest_prepare_gpu")
        m.def_function[forest_predict_resident_gpu_binding[False]]("forest_predict_resident_gpu")
        m.def_function[forest_release_gpu_binding[False]]("forest_release_gpu")
        m.def_function[forest_vector_groves_binding]("forest_vector_groves")
        m.def_function[forest_predict_resident_into_gpu_binding[False]]("forest_predict_resident_into_gpu")
        m.def_function[forest_predict_resident_into_gpu_binding[False, True]]("forest_predict_resident_reuse_gpu")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_trees: ", e))
