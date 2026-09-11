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

from std.memory import memcpy
from hostptr import copy_f32

from std.os import abort
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

from max.gpu.host import DeviceBuffer, DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    WeightedClassificationBin,
    RegressionBin,
)
from checks.fixed_point import choose_scale
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.instruments import StageTimes
from ensemble.decisiontree.decisiontree import (
    ENTROPY,
    GAMMA,
    GINI,
    INVERSE_GAUSSIAN,
    MSE,
    POISSON,
    DecisionTreeParams,
    TreeMetaDataNode,
    criterion_name,
)
from ensemble.flatnode import SparseTreeNode
from ensemble.randomforest import (
    CLASSIFICATION,
    REGRESSION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    fit_forest,
)

comptime DT = DType.float32
comptime CLT = DType.int32
comptime RLT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLT, ClassificationBin]
comptime WeightedClsObj = ClassificationObjectiveFunction[
    DT, CLT, WeightedClassificationBin
]
comptime RegObj = RegressionObjectiveFunction[DT, RLT, RegressionBin]


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


comptime N_RF_FIT_PARAMS = 16
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words:

     0  n_rows
     1  n_cols
     2  n_classes             (classifier only; MUST be 0 for the regressor)
     3  n_trees
     4  max_depth
     5  max_leaves            (-1 = unlimited, cuML's own sentinel)
     6  max_features_fraction (float in (0, 1]; the WRAPPER resolves
                               'sqrt'/'log2'/counts to a fraction, exactly
                               as cuML's python layer does)
     7  max_n_bins
     8  min_samples_leaf
     9  min_samples_split
    10  min_impurity_decrease (float)
    11  bootstrap             (0/1)
    12  max_samples           (float; ignored and reset to 1.0 by the implementation
                               when bootstrap is 0, `randomforest.cuh:304`)
    13  seed
    14  n_streams             (cuML's python default is 4; DEVIATION 117)
    15  max_batch_size        (cuML default 4096)

The split criterion is NOT a slot: it crosses as its own integer argument
(`criterion`, the `CRITERION` enumerator value from `decisiontree.mojo`)
so the two fit entry points can gate it by LABEL TYPE before the engine
sees it -- see DEVIATION 407 at `_check_criterion` below.
"""

# DEVIATION 407 (2026-08-23, RF lane). THE CRITERION SELECTOR CROSSES THE
# PYTHON BOUNDARY.
# THEIRS: `randomforest_common.pyx:104-151` maps the strings
# 'gini'/'entropy'/'mse'/'poisson'/'gamma'/'inverse_gaussian' (and the
# digits '0'..'7') onto `CRITERION`, refuses MAE by NotImplementedError, and
# passes the enumerator down as `split_criterion`; nothing in their python
# layer checks that the enumerator suits the LABEL TYPE. The C++ `default:`
# arm of `GainPerSplit` (`objectives.cuh:132-136`, `:331-338`) returns
# `-max()` for every candidate, so a regression enumerator handed to the
# classifier (or vice versa) fits a forest of STUMPS without a word.
# OURS: the wrapper maps sklearn-shaped names onto the same enumerators
# (`python/mojolearn/randomforest.py::_CLS_CRITERIA` / `_REG_CRITERIA`)
# and each fit binding REFUSES BY NAME an enumerator its objective has no
# arm for -- a silent stump is the "reached but inert" failure class
# `ensemble/checks/criteria_check.mojo` arm A/B exists to catch, and
# the Python surface must not reopen it. Until this deviation the binding
# hardcoded GINI / MSE and the wrapper refused every other name; that text
# is gone.


def _cls_criteria() -> List[Int]:
    """The enumerators `ClassificationObjectiveFunction.GainPerSplit` has
    an arm for (`objectives.cuh:132-136`)."""
    return [GINI, ENTROPY]


def _reg_criteria() -> List[Int]:
    """The enumerators `RegressionObjectiveFunction.GainPerSplit` has an
    arm for (`objectives.cuh:331-338`); MAE is enumerated but armless,
    theirs and ours."""
    return [MSE, POISSON, GAMMA, INVERSE_GAUSSIAN]


def _check_criterion(
    who: String, criterion: Int, allowed: List[Int]
) raises:
    """DEVIATION 407: refuse an enumerator the objective has no arm for."""
    for i in range(len(allowed)):
        if allowed[i] == criterion:
            return
    var names = String("")
    for i in range(len(allowed)):
        if i > 0:
            names += "/"
        names += criterion_name(allowed[i])
    raise Error(
        who
        + ": split criterion "
        + criterion_name(criterion)
        + " ("
        + String(criterion)
        + ") has no arm in this objective's GainPerSplit"
        + " (objectives.mojo, DEVIATION 407); accepted: "
        + names
    )


def _rf_params_from(params: PythonObject, criterion: Int) raises -> RF_params:
    """Slots 3-15 into `RF_params`, read under the GIL."""
    return RF_params(
        n_trees=Int32(Int(py=params[3])),
        bootstrap=Int(py=params[11]) != 0,
        max_samples=Float32(Float64(py=params[12])),
        seed=UInt64(Int(py=params[13])),
        n_streams=Int32(Int(py=params[14])),
        tree_params=DecisionTreeParams(
            max_depth=Int32(Int(py=params[4])),
            max_leaves=Int32(Int(py=params[5])),
            max_features=Float32(Float64(py=params[6])),
            max_n_bins=Int32(Int(py=params[7])),
            min_samples_leaf=Int32(Int(py=params[8])),
            min_samples_split=Int32(Int(py=params[9])),
            split_criterion=criterion,
            min_impurity_decrease=Float32(Float64(py=params[10])),
            max_batch_size=Int32(Int(py=params[15])),
        ),
    )


def _forest_out_trees(trees: List[TreeMetaDataNode[DT]]) raises -> PythonObject:
    """Retained same-fit diagnostic; RF label dtype does not affect tree storage."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var num_outputs = 1
    if len(trees) > 0:
        num_outputs = Int(trees[0].num_outputs)
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(trees)):
        ref tree = trees[t]
        var n = len(tree.sparsetree)
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            ref node = tree.sparsetree[i]
            colid.append(PythonObject(Int(node.ColumnId())))
            quesval.append(PythonObject(Float64(node.QueryValue())))
            left_child.append(PythonObject(Int(node.LeftChildId())))
        for i in range(len(tree.vector_leaf)):
            leaves.append(PythonObject(Float64(tree.vector_leaf[i])))
    var meta = Python.list()
    meta.append(PythonObject(len(trees)))
    meta.append(PythonObject(num_outputs))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out



def _forest_out(forest: RandomForestMetaData[DT, DT]) raises -> PythonObject:
    return _forest_out_trees(forest.trees)


def _forest_out_i32(forest: RandomForestMetaData[DT, CLT]) raises -> PythonObject:
    return _forest_out_trees(forest.trees)


# DEVIATION 2482: both RF label types own the same typed tree list. Moving it
# into the export registry retains existing native storage without flattening.
from std.ffi import _Global
from forest_export_binding import (
    ForestExportRegistry, validate_forest_export_destinations,
    copy_forest_export_leaves,
)
comptime RFExportTrees = List[TreeMetaDataNode[DT]]
comptime RF_EXPORTS = _Global[StorageType=ForestExportRegistry[RFExportTrees],
    name=("MojoRFFitExportIdentical" if GLOBAL_NUMERIC_MODE == 1 else
          "MojoRFFitExportDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
          "MojoRFFitExportFast"), init_fn=ForestExportRegistry[RFExportTrees].__init__]


def _retain_rf_export(var trees: RFExportTrees) raises -> PythonObject:
    if not len(trees):
        raise Error("cannot export an empty fitted RF")
    var nodes = 0
    var outputs = Int(trees[0].num_outputs)
    for tree in trees:
        nodes += len(tree.sparsetree)
        if Int(tree.num_outputs) != outputs or len(tree.vector_leaf) != len(tree.sparsetree) * outputs:
            raise Error("fitted RF leaf storage differs from export dimensions")
    var count = len(trees)
    var meta: List[Int64] = [Int64(count), Int64(outputs)]
    return RF_EXPORTS.get_or_create_ptr()[].insert(trees^, count, nodes, outputs, meta^)


def rf_forest_export_binding(handle: PythonObject, offsets: PythonObject,
    columns: PythonObject, thresholds: PythonObject, left: PythonObject,
    leaves: PythonObject, counts: PythonObject) raises -> PythonObject:
    if len(counts) != 3:
        raise Error("forest_export requires trees, nodes, outputs capacities")
    var id = Int(py=handle)
    var registry = RF_EXPORTS.get_or_create_ptr()
    registry[].validate(id, Int(py=counts[0]), Int(py=counts[1]), Int(py=counts[2]))
    validate_forest_export_destinations(Int(py=offsets), Int(py=columns),
        Int(py=thresholds), Int(py=left), Int(py=leaves))
    var op = _i32_ptr(Int(py=offsets))
    var cp = _i32_ptr(Int(py=columns))
    var tp = _f32_ptr(Int(py=thresholds))
    var lp = _i32_ptr(Int(py=left))
    var total = 0
    op[0] = 0
    ref trees = registry[].entries[id].model
    for t in range(len(trees)):
        ref tree = trees[t]
        for i in range(len(tree.sparsetree)):
            ref node = tree.sparsetree[i]
            cp[total + i] = Int32(node.ColumnId())
            tp[total + i] = node.QueryValue()
            lp[total + i] = Int32(node.LeftChildId())
        copy_forest_export_leaves(tree.vector_leaf, Int(py=leaves),
                                  total * registry[].entries[id].outputs)
        total += len(tree.sparsetree)
        op[t + 1] = Int32(total)
    return PythonObject(None)


def rf_forest_export_legacy_binding(handle: PythonObject) raises -> PythonObject:
    var registry = RF_EXPORTS.get_or_create_ptr()
    var id = Int(py=handle)
    if id not in registry[].entries:
        raise Error("unknown or released fitted forest export handle")
    return _forest_out_trees(registry[].entries[id].model)


def rf_forest_export_release_binding(handle: PythonObject) raises -> PythonObject:
    RF_EXPORTS.get_or_create_ptr()[].release(Int(py=handle))
    return PythonObject(None)


def _rf_classifier_fit[EXPORT: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
    weights_addr: Int = 0,
) raises -> PythonObject:
    """Fit the cuML-implementation RandomForest classifier. `x` is COLUMN-major
    float32 (n_rows * n_cols, the layout `fit_forest`'s default expects);
    `y` is int32 class CODES in [0, n_classes). See `N_RF_FIT_PARAMS`.
    `criterion` is GINI (0) or ENTROPY (1); anything else is refused by
    name (DEVIATION 407)."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_classifier_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_classes = Int(py=params[2])
    if n_classes < 2:
        raise Error("rf_classifier_fit: n_classes must be >= 2")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _i32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_classifier_fit", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)

    var weights = List[Float32]()
    var weight_total = Float64(0)
    if weights_addr != 0:
        var wp = _f32_ptr(weights_addr)
        var total = Float64(0)
        var all_unit = True
        for i in range(n_rows):
            var w = wp[i]
            if not (w >= 0 and w <= Float32(3.4028234663852886e38)):
                raise Error("class weights must be finite and nonnegative")
            weights.append(w)
            total += Float64(w)
            all_unit = all_unit and w == Float32(1)
        weight_total = total
        if total <= 0:
            raise Error("class weights must have positive total")
        if all_unit:
            weights = List[Float32]()
    var forest: RandomForestMetaData[DT, CLT]
    # DEVIATION 2510 -- the binding's own stage table (MOJOLEARN_STAGE_TIMES=1
    # only): what this entry point spends OUTSIDE `fit_forest`'s fit_total,
    # host-stamped, so the Python-side residual can be split. No arithmetic.
    var bt = StageTimes()
    var t_bind = bt.start()
    with GILReleased(Python()):
        var t_s = bt.start()
        var ctx = DeviceContext()
        bt.stop_host("bind_ctx_create", t_s)
        t_s = bt.start()
        var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
        var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
        ctx.synchronize()
        bt.stop_host("bind_pinned_alloc", t_s)
        t_s = bt.start()
        # DEVIATION 2481: bulk typed copies preserve every input bit while
        # avoiding scalar stores into pinned memory.
        copy_f32(xp, hx.unsafe_ptr(), n_rows * n_cols)
        memcpy(dest=hy.unsafe_ptr(), src=yp, count=n_rows)
        bt.stop_host("bind_host_copy", t_s)
        t_s = bt.start()
        var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
        var dy = ctx.enqueue_create_buffer[CLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        # The host weights drive sampling; non-bootstrap objectives read the
        # device weights at original row IDs. Keep both alive through fitting.
        var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
        ctx.synchronize()
        bt.stop_host("bind_h2d", t_s)
        if len(weights) > 0:
            ctx.enqueue_copy(dst_buf=dsw, src_ptr=weights.unsafe_ptr())
        # cuML randomforest.cuh dispatches weighted objectives only when
        # bootstrap is disabled: sampling already applies bootstrap weights.
        if len(weights) > 0 and not rf_params.bootstrap:
            var scale = choose_scale(weight_total, n_rows)
            if scale < Float64(1.1754943508222875e-38) or scale > Float64(3.4028234663852886e38):
                raise Error("class weights exceed Float32 fixed-point scale range")
            var scales = BinScales(Float32(1), Float32(scale))
            forest = fit_forest[WeightedClsObj](
                ctx, dx, dy, dsw, n_rows, n_cols, n_classes, rf_params,
                scales, sample_weight_host=weights, host_x_addr=Int(xp),
            )
        else:
            forest = fit_forest[ClsObj](
                ctx, dx, dy, dsw, n_rows, n_cols, n_classes, rf_params,
                sample_weight_host=weights, host_x_addr=Int(xp),
            )
        t_s = bt.start()
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
        _ = hx^
        _ = hy^
        bt.stop_host("bind_release_buffers", t_s)
        t_s = bt.start()
        # DEVIATION 1946: THE CONTEXT DIES LAST. Mojo destroys a value at its
        # LAST USE, so without this line `ctx`'s last use is the
        # `synchronize()` above and the five buffers -- two of them PINNED
        # HOST allocations -- are freed against a context that is already
        # gone. `ensemble/checks/rf_ctx_order_probe.mojo` is that ordering
        # in 60 lines with no Python. It is the same class as DEVIATION 1944
        # (a device buffer freed against a context that is not the live one),
        # and it is why the pure-Mojo probe passed where this binding hung:
        # `rf_ctx_probe.mojo::one_fit` takes `ctx` as a BORROWED argument, so
        # the caller's frame keeps it alive past every release.
        _ = ctx^
        bt.stop_host("bind_release_ctx", t_s)
    _ = weights^
    var t_x = bt.start()
    comptime if EXPORT:
        var export_trees = forest.trees^
        forest.trees = RFExportTrees()
        var handle = _retain_rf_export(export_trees^)
        bt.stop_host("bind_export_retain", t_x)
        bt.stop_host("binding_total", t_bind)
        bt.report()
        return handle
    else:
        var out = _forest_out_i32(forest)
        bt.stop_host("bind_export_legacy", t_x)
        bt.stop_host("binding_total", t_bind)
        bt.report()
        return out


def rf_classifier_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_classifier_fit[EXPORT](x_addr, y_addr, params, criterion)


def rf_classifier_fit_weighted_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    var address = Int(py=weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_classifier_fit[EXPORT](x_addr, y_addr, params, criterion, address)


def rf_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
) raises -> PythonObject:
    """Fit the cuML-implementation RandomForest regressor. Same contract; slot 2
    MUST be 0 and `y` is float32. `criterion` is MSE (2), POISSON (4),
    GAMMA (5) or INVERSE_GAUSSIAN (6); anything else is refused by name
    (DEVIATION 407). The log criteria are DEVIATION 406's: they RUN in
    both numeric modes, comparable to cuML in neither."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_regressor_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("rf_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_regressor_fit", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)

    var forest: RandomForestMetaData[DT, RLT]
    # DEVIATION 2510 -- as in the classifier: the binding's own host stamps.
    var bt = StageTimes()
    var t_bind = bt.start()
    with GILReleased(Python()):
        var t_s = bt.start()
        var ctx = DeviceContext()
        bt.stop_host("bind_ctx_create", t_s)
        t_s = bt.start()
        var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
        var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
        ctx.synchronize()
        bt.stop_host("bind_pinned_alloc", t_s)
        t_s = bt.start()
        # DEVIATION 2481: bulk typed copies preserve every input bit while
        # avoiding scalar stores into pinned memory.
        copy_f32(xp, hx.unsafe_ptr(), n_rows * n_cols)
        copy_f32(yp, hy.unsafe_ptr(), n_rows)
        bt.stop_host("bind_host_copy", t_s)
        t_s = bt.start()
        var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
        var dy = ctx.enqueue_create_buffer[RLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](1)
        ctx.synchronize()
        bt.stop_host("bind_h2d", t_s)
        # THE LABEL SCALE IS NOT OPTIONAL. `RegressionBin` accumulates
        # `label_sum` in fixed point through `BinScales.label_scale`
        # (DEVIATION 101b), and the host chooses the scale once per fit
        # from the sum of label magnitudes (`fixed_point.choose_scale`).
        # The unit scale TRUNCATES every |label| < 1 to zero, which fits a
        # forest of zero-leaved stumps -- the build gate caught exactly
        # that. The weight plane stays 1.0: unweighted `Weight()` is a
        # count, which is exact.
        var mag = Float64(0.0)
        for i in range(n_rows):
            var v = Float64(yp[i])
            mag += v if v >= 0.0 else -v
        var scales = BinScales(
            Float32(choose_scale(mag, n_rows)), Float32(1.0)
        )
        # `n_unique_labels` is 1 for regression, exactly what
        # `rf_regressor_fit`'s cuML counterpart passes.
        forest = fit_forest[RegObj](
            ctx, dx, dy, dsw, n_rows, n_cols, 1, rf_params, scales,
            host_x_addr=Int(xp),
        )
        t_s = bt.start()
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
        _ = hx^
        _ = hy^
        bt.stop_host("bind_release_buffers", t_s)
        t_s = bt.start()
        # DEVIATION 1946, as in `rf_classifier_fit_binding`: the context
        # outlives every buffer created on it.
        _ = ctx^
        bt.stop_host("bind_release_ctx", t_s)
    var t_x = bt.start()
    comptime if EXPORT:
        var export_trees = forest.trees^
        forest.trees = RFExportTrees()
        var handle = _retain_rf_export(export_trees^)
        bt.stop_host("bind_export_retain", t_x)
        bt.stop_host("binding_total", t_bind)
        bt.report()
        return handle
    else:
        var out = _forest_out(forest)
        bt.stop_host("bind_export_legacy", t_x)
        bt.stop_host("binding_total", t_bind)
        bt.report()
        return out


def _rebuild_trees(
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int,
    num_outputs: Int,
) raises -> List[TreeMetaDataNode[DT]]:
    """The flat arrays back into `TreeMetaDataNode`'s own layout, so the
    traversal that runs is the implementation's. `instance_count` and
    `best_metric_val` are zero: the traversal reads neither."""
    var trees = List[TreeMetaDataNode[DT]](capacity=n_trees)
    for t in range(n_trees):
        var lo = Int(offsets_p[t])
        var hi = Int(offsets_p[t + 1])
        if lo < 0 or hi < lo:
            raise Error("rf_predict: tree_offsets are not a prefix scan")
        var nodes = List[SparseTreeNode[DT]](capacity=hi - lo)
        var vleaf = List[Float32](capacity=(hi - lo) * num_outputs)
        for i in range(lo, hi):
            nodes.append(
                SparseTreeNode[DT](
                    colid_p[i], quesval_p[i], 0.0, Int64(left_p[i]), 0
                )
            )
            for k in range(num_outputs):
                vleaf.append(leaves_p[i * num_outputs + k])
        trees.append(
            TreeMetaDataNode[DT](
                Int32(t), 0, 0, 0.0, vleaf^, nodes^, Int32(num_outputs)
            )
        )
    return trees^


def _default_rf_params(n_trees: Int) raises -> RF_params:
    """A benign `RF_params` for the predict-side metadata: predict reads the
    forest's trees, not these knobs, but the structs require one."""
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=True,
        max_samples=Float32(1.0),
        seed=UInt64(0),
        n_streams=Int32(4),
        tree_params=DecisionTreeParams(
            max_depth=Int32(16),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(128),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(4096),
        ),
    )


def rf_predict_proba_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Classifier probabilities through the IMPLEMENTED `predict_proba`
    (`predict` stopped one line early, per its docstring). Model arrays are
    int32/float32 as `_forest_out` laid them out; `x` is ROW-major; `out`
    receives n_rows * num_outputs float32. `params` is `[n_rows, n_cols,
    n_trees, num_outputs]`. The wrapper's argmax over these IS cuML's
    `predict`. Returns rows written."""
    if len(params) != 4:
        raise Error(
            "rf_predict_proba: params must hold [n_rows, n_cols, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs < 2:
        raise Error(
            "rf_predict_proba: n_trees must be >= 1 and num_outputs >= 2"
        )
    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        var rf_params = _default_rf_params(n_trees)
        var trees = _rebuild_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, num_outputs,
        )
        var forest = RandomForestMetaData[DT, CLT](
            trees^, rf_params, Int32(n_cols)
        )
        var rf = RandomForest[DT, CLT](
            rf_params=rf_params, rf_type=CLASSIFICATION
        )
        var rows = List[Float32](capacity=n_rows * n_cols)
        for i in range(n_rows * n_cols):
            rows.append(xp[i])
        var probs = List[Float32](capacity=n_rows * num_outputs)
        for _ in range(n_rows * num_outputs):
            probs.append(0.0)
        rf.predict_proba(rows, n_rows, n_cols, probs, forest)
        for i in range(n_rows * num_outputs):
            op[i] = probs[i]
        wrote = n_rows
    return PythonObject(wrote)


def rf_predict_reg_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Regressor predictions through the IMPLEMENTED `RandomForest.predict`.
    Same array contract; `out` is n_rows float32; `params` is `[n_rows,
    n_cols, n_trees, 1]`. Returns rows written."""
    if len(params) != 4:
        raise Error(
            "rf_predict_reg: params must hold [n_rows, n_cols, n_trees, 1],"
            " got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs != 1:
        raise Error(
            "rf_predict_reg: n_trees must be >= 1 and num_outputs must be 1"
        )
    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        var rf_params = _default_rf_params(n_trees)
        var trees = _rebuild_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p, n_trees, 1
        )
        var forest = RandomForestMetaData[DT, RLT](
            trees^, rf_params, Int32(n_cols)
        )
        var rf = RandomForest[DT, RLT](
            rf_params=rf_params, rf_type=REGRESSION
        )
        var rows = List[Float32](capacity=n_rows * n_cols)
        for i in range(n_rows * n_cols):
            rows.append(xp[i])
        var preds = List[Float32](capacity=n_rows)
        for _ in range(n_rows):
            preds.append(0.0)
        rf.predict(rows, n_rows, n_cols, preds, forest)
        for i in range(n_rows):
            op[i] = preds[i]
        wrote = n_rows
    return PythonObject(wrote)


def _rf_predict_gpu_parallel(
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
        var result = forest_predict_gpu[True, True](
            ctx, offsets, columns, thresholds, left, leaves, x,
            rows, features, outputs,
        )
        for i in range(rows * outputs):
            out_p[i] = result[i]
        _ = result^
        _ = ctx^
    return PythonObject(rows)


def rf_predict_proba_gpu_parallel_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    if len(params) != 4:
        raise Error("GPU parallel prediction requires four parameters")
    var outputs = Int(py=params[3])
    if outputs < 2:
        raise Error("GPU parallel prediction output dimension does not match task")
    return _rf_predict_gpu_parallel(
        offsets_addr, colid_addr, quesval_addr, left_child_addr,
        leaves_addr, x_addr, out_addr, params,
    )


def rf_predict_reg_gpu_parallel_binding(
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    if len(params) != 4:
        raise Error("GPU parallel prediction requires four parameters")
    var outputs = Int(py=params[3])
    if outputs != 1:
        raise Error("GPU parallel prediction output dimension does not match task")
    return _rf_predict_gpu_parallel(
        offsets_addr, colid_addr, quesval_addr, left_child_addr,
        leaves_addr, x_addr, out_addr, params,
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


@export
def PyInit__mojolearn_rf() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_rf")
        m.def_function[rf_vendor_binding]("rf_vendor")
        m.def_function[rf_numeric_mode_binding]("rf_numeric_mode")
        m.def_function[rf_classifier_fit_binding[False]]("rf_classifier_fit")
        m.def_function[rf_classifier_fit_binding[True]]("rf_classifier_fit_export")
        m.def_function[rf_classifier_fit_weighted_binding[False]]("rf_classifier_fit_weighted")
        m.def_function[rf_classifier_fit_weighted_binding[True]]("rf_classifier_fit_weighted_export")
        m.def_function[rf_regressor_fit_binding[False]]("rf_regressor_fit")
        m.def_function[rf_regressor_fit_binding[True]]("rf_regressor_fit_export")
        m.def_function[rf_predict_proba_binding]("rf_predict_proba")
        m.def_function[rf_predict_reg_binding]("rf_predict_reg")
        m.def_function[rf_predict_proba_gpu_parallel_binding]("rf_predict_proba_gpu_parallel")
        m.def_function[rf_predict_reg_gpu_parallel_binding]("rf_predict_reg_gpu_parallel")
        m.def_function[forest_resident_layout_binding]("forest_resident_layout")
        m.def_function[rf_forest_export_binding]("forest_export")
        m.def_function[rf_forest_export_legacy_binding]("forest_export_legacy")
        m.def_function[rf_forest_export_release_binding]("forest_export_release")
        m.def_function[forest_prepare_gpu_binding[True]]("forest_prepare_gpu")
        m.def_function[forest_predict_resident_gpu_binding[True]]("forest_predict_resident_gpu")
        m.def_function[forest_release_gpu_binding[True]]("forest_release_gpu")
        m.def_function[forest_vector_groves_binding]("forest_vector_groves")
        m.def_function[forest_predict_resident_into_gpu_binding[True]]("forest_predict_resident_into_gpu")
        m.def_function[forest_predict_resident_into_gpu_binding[True, True]]("forest_predict_resident_reuse_gpu")
        return m.finalize()
    except e:
        abort(String("failed to initialize _mojolearn_rf: ") + String(e))
        return PythonObject(None)
