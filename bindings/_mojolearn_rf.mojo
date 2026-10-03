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
from ensemble.device_layout import device_has_nan_f32, upload_forest_x
from ensemble.nan_refusal import RF_NAN_REFUSAL

from std.os import abort
from ensemble.device_finite import FOREST_DEVICE_FINITE, ForestFiniteScan
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from forest_inference_binding import (
    forest_prepare_gpu_binding, forest_predict_resident_gpu_binding, forest_release_gpu_binding,
    forest_vector_groves_binding, forest_predict_resident_into_gpu_binding,
    forest_predict_resident_labels_gpu_binding,
    forest_resident_layout_binding, forest_ordered_resident_binding,
    forest_pool_available, forest_pool_fault_available,
)
from core.forest_inference import forest_predict_gpu
from core.forest_inference_model import (
    resident_prepare, resident_predict_into, resident_release,
)
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoRfContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoRfContextFast"


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
    FUSED_BOOTSTRAP_GATHER,
    REGRESSION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    ForestPrep,
    fit_forest,
    fit_forest_prepared,
    launch_gather_rows_colmajor,
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


#: FAST on Apple (trees-apple3): a wider node batch. A tree deeper than 12
#: levels has levels of more than 4096 nodes, and every batch of a level is
#: one more round of launches, uploads and a host wait for nodes of a few
#: rows each (M4 Pro, taxi, 100 trees of depth 16: 32.8 node batches and
#: 71.5 histogram rounds per tree). The queue is first in, first out and a
#: node's split reads only its own rows and its own feature sample, so the
#: batch width changes which launch a node rides, not its split. Only the
#: default width (4096) is widened, only for trees without a leaf budget
#: (max_leaves -1) that may grow past 12 levels, and only as far as `RF_FAST_BATCH_BYTES` of histogram
#: workspace per stream allows. OPT-IN until its A/B passes: `-D
#: MOJOLEARN_RF_FAST_BATCH32K` (the 16K arm was DROPPED-noise,
#: lane/apple-fast-trees2 @ bfd1d7cc6).
comptime RF_FAST_BATCH = 0 if not (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
) else (
    32768 if is_defined["MOJOLEARN_RF_FAST_BATCH32K"]() else 0
)
comptime RF_FAST_BATCH_BYTES = 536870912
comptime RF_CUML_DEFAULT_BATCH = 4096


def _rf_batch_size(params: PythonObject) raises -> Int:
    """Slot 15, widened under `RF_FAST_BATCH` (see its note)."""
    var batch = Int(py=params[15])
    comptime if RF_FAST_BATCH > 0:
        var depth = Int(py=params[4])
        if (
            batch == RF_CUML_DEFAULT_BATCH
            and Int(py=params[5]) == -1
            and (depth > 12 or depth < 1)
        ):
            var n_cols = Int(py=params[1])
            var outputs = Int(py=params[2])
            if outputs < 1:
                outputs = 1
            var cols = Int(Float64(py=params[6]) * Float64(n_cols)) + 1
            if cols > n_cols:
                cols = n_cols
            if cols > 40:
                cols = 40
            # 8 bytes covers both bin types
            var per_node = Int(py=params[7]) * outputs * cols * 8
            var wide = RF_FAST_BATCH
            if per_node > 0 and wide * per_node > RF_FAST_BATCH_BYTES:
                wide = RF_FAST_BATCH_BYTES // per_node
            if wide > batch:
                batch = wide
    return batch


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
            max_batch_size=Int32(_rf_batch_size(params)),
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


def _rf_classifier_fit[EXPORT: Bool = False, ROWMAJOR: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
    weights_addr: Int = 0,
    tree_start: Int = 0,
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        bt.stop_host("bind_ctx_create", t_s)
        t_s = bt.start()
        var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
        ctx.synchronize()
        bt.stop_host("bind_pinned_alloc", t_s)
        t_s = bt.start()
        # DEVIATION 2481: bulk typed copies preserve every input bit while
        # avoiding scalar stores into pinned memory.
        memcpy(dest=hy.unsafe_ptr(), src=yp, count=n_rows)
        bt.stop_host("bind_host_copy", t_s)
        t_s = bt.start()
        # cpu-gpu-cleanup t-forest: X goes to the device as the caller's
        # bytes and is transposed there when the caller lent its ROW-major
        # C-order block (ROWMAJOR); the NaN refusal is a device scan
        # (`ensemble/device_layout.mojo`). No host pass over X: the threaded
        # pinned-stage transpose and host NaN scan (DEVIATION 2637) are gone.
        # The builder's host view of X (`host_x_addr`, DEVIATION 2484) is the
        # caller's column-major block when it lent one, else none.
        var dx = upload_forest_x(ctx, xp, n_rows, n_cols, ROWMAJOR)
        # FAST on Apple (lane apple-fast-rfet-scan): the NaN-and-inf device
        # scan below (ensemble/device_finite.mojo) stands in for this NaN scan.
        comptime if not FOREST_DEVICE_FINITE:
            if device_has_nan_f32(ctx, dx, n_rows * n_cols):
                raise Error("rf_classifier_fit: " + RF_NAN_REFUSAL)
        var host_x = 0 if ROWMAJOR else Int(xp)
        var dy = ctx.enqueue_create_buffer[CLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        # The host weights drive sampling; non-bootstrap objectives read the
        # device weights at original row IDs. Keep both alive through fitting.
        var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
        comptime if FOREST_DEVICE_FINITE:
            var fscan = ForestFiniteScan(ctx)
            fscan.enqueue(ctx, dx, n_rows * n_cols)
            ctx.synchronize()
            fscan.refuse_if_bad("rf_classifier_fit: ")
        else:
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
                scales, sample_weight_host=weights, host_x_addr=host_x, tree_start=tree_start,
            )
        else:
            forest = fit_forest[ClsObj](
                ctx, dx, dy, dsw, n_rows, n_cols, n_classes, rf_params,
                sample_weight_host=weights, host_x_addr=host_x, tree_start=tree_start,
            )
        t_s = bt.start()
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
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


def rf_classifier_fit_rowmajor_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_classifier_fit` with `x` ROW-major (C-order) float32, borrowed
    through the call (DEVIATION 2637). Same params, same forest bits."""
    return _rf_classifier_fit[EXPORT, True](x_addr, y_addr, params, criterion)


def rf_classifier_fit_weighted_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    var address = Int(py=weights_addr)
    if address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")
    return _rf_classifier_fit[EXPORT](x_addr, y_addr, params, criterion, address)


def rf_regressor_fit_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    return _rf_regressor_fit[EXPORT](x_addr, y_addr, params, criterion)


# ---------------------------------------------------------------------------
# THE DATA SESSION (trees-apple3). A boosted ensemble (DART, AdaBoost) fits
# one small forest per member on the SAME X: each member's fit scanned X for
# NaN, copied it into a pinned stage and uploaded it again (M4 Pro, 1M x 16:
# about 10 ms of a 30 ms member fit). A session does that once. Its member
# fits are `fit_forest_prepared` on the staged X: the same kernels on the
# same bytes, so a member's forest is the forest `rf_*_fit` returns.
# `share_tables` also keeps the first member's quantile table and binned
# dataset for the later members. Their quantile SAMPLE is then the first
# member's (the sample is drawn with the fit's seed), so the later members'
# bins differ from their own fits': a FAST-only option with a quality check.
# One thread uses a session at a time.
struct RfDataSession(Movable):
    var id: Int
    var dx: DeviceBuffer[DT]
    var n_rows: Int
    var n_cols: Int
    var host_x: Int
    var share_tables: Bool
    var prep: List[ForestPrep]

    def __init__(
        out self,
        id: Int,
        var dx: DeviceBuffer[DT],
        n_rows: Int,
        n_cols: Int,
        host_x: Int,
        share_tables: Bool,
    ):
        self.id = id
        self.dx = dx^
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.host_x = host_x
        self.share_tables = share_tables
        self.prep = List[ForestPrep]()


struct RfSessionRegistry(Defaultable, Movable):
    var sessions: List[RfDataSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[RfDataSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        raise Error("unknown or closed forest data session handle")


comptime RF_SESSIONS = _Global[StorageType=RfSessionRegistry,
    name=("MojoRFDataSessionIdentical" if GLOBAL_NUMERIC_MODE == 1 else
          "MojoRFDataSessionDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
          "MojoRFDataSessionFast"), init_fn=RfSessionRegistry.__init__]


def rf_data_session_open_binding(
    x_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Stage X on the device for the member fits of one ensemble. `params`
    is [n_rows, n_cols, row_major, share_tables]; `x` is float32, ROW-major
    when `row_major` is 1 and COLUMN-major otherwise, borrowed through this
    call only. Returns the session handle."""
    if len(params) != 4:
        raise Error("rf_data_session_open: params must hold 4 values")
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var row_major = Int(py=params[2]) != 0
    var share_tables = Int(py=params[3]) != 0
    if n_rows <= 0 or n_cols <= 0:
        raise Error("rf_data_session_open: invalid shape")
    comptime if GLOBAL_NUMERIC_MODE == 1:
        if share_tables:
            raise Error(
                "rf_data_session_open: shared quantile tables are a FAST"
                " option; an IDENTICAL member draws its own sample"
            )
    var xp = _f32_ptr(Int(py=x_addr))
    var made = List[RfDataSession]()
    var has_nan = False
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        # cpu-gpu-cleanup t-forest: the caller's bytes go to the device as
        # they are, the transpose and the NaN scan run there. The session
        # keeps no host copy of X (`host_x` 0: the borrow ends with this
        # call), so a member's OOB pass reads the device plane.
        var dx = upload_forest_x(ctx, xp, n_rows, n_cols, row_major)
        has_nan = device_has_nan_f32(ctx, dx, n_rows * n_cols)
        if has_nan:
            _ = dx^
        else:
            made.append(
                RfDataSession(0, dx^, n_rows, n_cols, 0, share_tables)
            )
        _ = ctx^
    if has_nan:
        raise Error("rf_data_session_open: " + RF_NAN_REFUSAL)
    var reg = RF_SESSIONS.get_or_create_ptr()
    var id = reg[].next_id
    reg[].next_id += 1
    var session = made.pop()
    session.id = id
    reg[].sessions.append(session^)
    return PythonObject(id)


def rf_data_session_close_binding(handle: PythonObject) raises -> PythonObject:
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(Int(py=handle))
    var session = reg[].sessions.pop(si)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    ctx.synchronize()
    _ = session^
    _ = ctx^
    return PythonObject(None)


def _session_shape_check(
    name: String, session_rows: Int, session_cols: Int,
    n_rows: Int, n_cols: Int,
) raises:
    if session_rows != n_rows or session_cols != n_cols:
        raise Error(
            name
            + ": params name "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ", the session holds "
            + String(session_rows)
            + " x "
            + String(session_cols)
        )


def rf_regressor_fit_session_binding(
    handle: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_export` on a data session's X: same params, same
    checks, same export descriptor."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_regressor_fit_session: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("rf_regressor_fit_session: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var yp = _f32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_regressor_fit_session", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    var session_id = Int(py=handle)
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(session_id)
    _session_shape_check(
        "rf_regressor_fit_session", reg[].sessions[si].n_rows,
        reg[].sessions[si].n_cols, n_rows, n_cols,
    )
    var share_tables = reg[].sessions[si].share_tables
    var host_x = reg[].sessions[si].host_x
    # the kept tables leave the session for the fit and return after it
    var prep = List[ForestPrep]()
    swap(prep, reg[].sessions[si].prep)

    var forest: RandomForestMetaData[DT, RLT]
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var dxv = reg[].sessions[si].dx.create_sub_buffer[DT](
            0, n_rows * n_cols
        )
        var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
        ctx.synchronize()
        copy_f32(yp, hy.unsafe_ptr(), n_rows)
        var dy = ctx.enqueue_create_buffer[RLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](1)
        ctx.synchronize()
        # the label scale, as `_rf_regressor_fit` chooses it
        var mag = Float64(0.0)
        for i in range(n_rows):
            var v = Float64(yp[i])
            mag += v if v >= 0.0 else -v
        var scales = BinScales(
            Float32(choose_scale(mag, n_rows)), Float32(1.0)
        )
        forest = fit_forest_prepared[RegObj](
            ctx, dxv, dy, dsw, n_rows, n_cols, 1, rf_params, prep,
            share_tables, scales, host_x_addr=host_x,
        )
        ctx.synchronize()
        _ = dxv^
        _ = dy^
        _ = dsw^
        _ = hy^
        _ = ctx^
    var back = RF_SESSIONS.get_or_create_ptr()
    var bi = back[].find(session_id)
    swap(prep, back[].sessions[bi].prep)
    _ = prep^
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_export(export_trees^)


def rf_regressor_fit_session_rows_binding(
    handle: PythonObject, rows_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit_export` on the rows `rows` (int32 row ids of the
    session's X, `params[0]` of them, repeats allowed) with `y` their
    labels in that order: the member fit of a resampling ensemble
    (AdaBoost.R2). The rows are gathered ON THE DEVICE from the staged X
    into a fresh column-major matrix, the bytes a host gather of the same
    rows stages, and the fit is `fit_forest` on it with its own quantile
    table."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_regressor_fit_session_rows: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("rf_regressor_fit_session_rows: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    if n_rows <= 0:
        raise Error("rf_regressor_fit_session_rows: no rows")
    var rp = _i32_ptr(Int(py=rows_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_regressor_fit_session_rows", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)
    var session_id = Int(py=handle)
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(session_id)
    var src_rows = reg[].sessions[si].n_rows
    if reg[].sessions[si].n_cols != n_cols:
        raise Error("rf_regressor_fit_session_rows: the session holds another column count")
    for i in range(n_rows):
        var r = Int(rp[i])
        if r < 0 or r >= src_rows:
            raise Error("rf_regressor_fit_session_rows: a row id is outside the session's X")

    var forest: RandomForestMetaData[DT, RLT]
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var hrows = ctx.enqueue_create_host_buffer[CLT](n_rows)
        var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
        ctx.synchronize()
        memcpy(dest=hrows.unsafe_ptr(), src=rp, count=n_rows)
        copy_f32(yp, hy.unsafe_ptr(), n_rows)
        var drows = ctx.enqueue_create_buffer[CLT](n_rows)
        ctx.enqueue_copy(dst_buf=drows, src_ptr=hrows.unsafe_ptr())
        var dy = ctx.enqueue_create_buffer[RLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](1)
        var dxg = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
        launch_gather_rows_colmajor(
            ctx,
            reg[].sessions[si].dx,
            drows,
            dxg,
            src_rows,
            n_rows,
            n_cols,
        )
        var mag = Float64(0.0)
        for i in range(n_rows):
            var v = Float64(yp[i])
            mag += v if v >= 0.0 else -v
        var scales = BinScales(
            Float32(choose_scale(mag, n_rows)), Float32(1.0)
        )
        forest = fit_forest[RegObj](
            ctx, dxg, dy, dsw, n_rows, n_cols, 1, rf_params, scales,
        )
        ctx.synchronize()
        _ = dxg^
        _ = drows^
        _ = dy^
        _ = dsw^
        _ = hrows^
        _ = hy^
        _ = ctx^
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_export(export_trees^)


def rf_classifier_fit_weighted_session_binding(
    handle: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, weights_addr: PythonObject,
) raises -> PythonObject:
    """`rf_classifier_fit_weighted_export` on a data session's X (its
    COLUMN-major stage): same params, same weight checks, same export
    descriptor."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_classifier_fit_weighted_session: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_classes = Int(py=params[2])
    if n_classes < 2:
        raise Error("rf_classifier_fit_weighted_session: n_classes must be >= 2")
    var yp = _i32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_classifier_fit_weighted_session", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)
    var weights_address = Int(py=weights_addr)
    if weights_address == 0:
        raise Error("weighted RF requires a nonzero Float32 weight pointer")

    var weights = List[Float32]()
    var weight_total = Float64(0)
    var wp = _f32_ptr(weights_address)
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

    var session_id = Int(py=handle)
    var reg = RF_SESSIONS.get_or_create_ptr()
    var si = reg[].find(session_id)
    _session_shape_check(
        "rf_classifier_fit_weighted_session", reg[].sessions[si].n_rows,
        reg[].sessions[si].n_cols, n_rows, n_cols,
    )
    var share_tables = reg[].sessions[si].share_tables
    var host_x = reg[].sessions[si].host_x
    var prep = List[ForestPrep]()
    swap(prep, reg[].sessions[si].prep)

    var forest: RandomForestMetaData[DT, CLT]
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var dxv = reg[].sessions[si].dx.create_sub_buffer[DT](
            0, n_rows * n_cols
        )
        var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
        ctx.synchronize()
        memcpy(dest=hy.unsafe_ptr(), src=yp, count=n_rows)
        var dy = ctx.enqueue_create_buffer[CLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](max(1, len(weights)))
        ctx.synchronize()
        if len(weights) > 0:
            ctx.enqueue_copy(dst_buf=dsw, src_ptr=weights.unsafe_ptr())
        if len(weights) > 0 and not rf_params.bootstrap:
            var scale = choose_scale(weight_total, n_rows)
            if scale < Float64(1.1754943508222875e-38) or scale > Float64(3.4028234663852886e38):
                raise Error("class weights exceed Float32 fixed-point scale range")
            var scales = BinScales(Float32(1), Float32(scale))
            forest = fit_forest_prepared[WeightedClsObj](
                ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
                prep, share_tables, scales, sample_weight_host=weights,
                host_x_addr=host_x,
            )
        else:
            forest = fit_forest_prepared[ClsObj](
                ctx, dxv, dy, dsw, n_rows, n_cols, n_classes, rf_params,
                prep, share_tables, sample_weight_host=weights,
                host_x_addr=host_x,
            )
        ctx.synchronize()
        _ = dxv^
        _ = dy^
        _ = dsw^
        _ = hy^
        _ = ctx^
    _ = weights^
    var back = RF_SESSIONS.get_or_create_ptr()
    var bi = back[].find(session_id)
    swap(prep, back[].sessions[bi].prep)
    _ = prep^
    var export_trees = forest.trees^
    forest.trees = RFExportTrees()
    return _retain_rf_export(export_trees^)


def rf_regressor_fit_rowmajor_binding[EXPORT: Bool = False](
    x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject,
) raises -> PythonObject:
    """`rf_regressor_fit` with `x` ROW-major (C-order) float32, borrowed
    through the call (DEVIATION 2637). Same params, same forest bits."""
    return _rf_regressor_fit[EXPORT, True](x_addr, y_addr, params, criterion)


def _rf_regressor_fit[EXPORT: Bool = False, ROWMAJOR: Bool = False](
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
    tree_start: Int = 0,
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        bt.stop_host("bind_ctx_create", t_s)
        t_s = bt.start()
        var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
        ctx.synchronize()
        bt.stop_host("bind_pinned_alloc", t_s)
        t_s = bt.start()
        # DEVIATION 2481: bulk typed copies preserve every input bit while
        # avoiding scalar stores into pinned memory.
        copy_f32(yp, hy.unsafe_ptr(), n_rows)
        bt.stop_host("bind_host_copy", t_s)
        t_s = bt.start()
        # cpu-gpu-cleanup t-forest, as in the classifier: device upload,
        # device transpose, device NaN scan.
        var dx = upload_forest_x(ctx, xp, n_rows, n_cols, ROWMAJOR)
        # FAST on Apple (lane apple-fast-rfet-scan): the NaN-and-inf device
        # scan below (ensemble/device_finite.mojo) stands in for this NaN scan.
        comptime if not FOREST_DEVICE_FINITE:
            if device_has_nan_f32(ctx, dx, n_rows * n_cols):
                raise Error("rf_regressor_fit: " + RF_NAN_REFUSAL)
        var host_x = 0 if ROWMAJOR else Int(xp)
        var dy = ctx.enqueue_create_buffer[RLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](1)
        comptime if FOREST_DEVICE_FINITE:
            var fscan = ForestFiniteScan(ctx)
            fscan.enqueue(ctx, dx, n_rows * n_cols)
            ctx.synchronize()
            fscan.refuse_if_bad("rf_regressor_fit: ")
        else:
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
            host_x_addr=host_x, tree_start=tree_start,
        )
        t_s = bt.start()
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
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

    # cpu-gpu-cleanup t-forest: the `sequential` engine runs on the device.
    # The host walk (`core/forest_host_predict.mojo`'s `rf_host_predict`,
    # DEVIATION 2900) is the CPU-only install's (`_mojolearn_rf_host`); here
    # the same arithmetic -- zero, add every tree's leaf in increasing tree
    # order, divide by `n_trees` -- is `forest_ordered_kernel`, one thread
    # per (row, output), through a one-call ordered snapshot.
    if n_rows == 0:
        return PythonObject(0)
    _ = xp
    _ = op
    return _rf_predict_ordered(
        "rf_predict_proba", offsets_p, colid_p, quesval_p, left_p, leaves_p,
        Int(py=x_addr), Int(py=out_addr), n_rows, n_cols, n_trees, num_outputs,
    )


def _rf_predict_ordered(
    name: String,
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    x_address: Int,
    out_address: Int,
    n_rows: Int,
    n_cols: Int,
    n_trees: Int,
    num_outputs: Int,
) raises -> PythonObject:
    """The `sequential` engine on the device: the flat model goes into an
    ordered resident snapshot (`forest_ordered_kernel`: increasing-tree
    association, the host walk's fold), the borrowed rows are predicted into
    the borrowed output, and the snapshot is released. The GIL stays held,
    as `forest_inference_binding` requires for the registry."""
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error(name + ": tree_offsets must start at 0 and hold at least one node per tree")
    if n_nodes > 2147483647 // num_outputs:
        raise Error(name + ": node/output count exceeds Int32")
    var offsets = List[Int32](capacity=n_trees + 1)
    var columns = List[Int32](capacity=n_nodes)
    var thresholds = List[Float32](capacity=n_nodes)
    var left = List[Int32](capacity=n_nodes)
    var leaves = List[Float32](capacity=n_nodes * num_outputs)
    for i in range(n_trees + 1):
        offsets.append(offsets_p[i])
    for i in range(n_nodes):
        columns.append(colid_p[i])
        thresholds.append(quesval_p[i])
        left.append(left_p[i])
    for i in range(n_nodes * num_outputs):
        leaves.append(leaves_p[i])
    var handle = resident_prepare[True](
        offsets, columns, thresholds, left, leaves, n_cols, num_outputs, True
    )
    try:
        resident_predict_into[True](
            handle,
            _f32_ptr(x_address).unsafe_origin_cast[MutAnyOrigin](),
            _f32_ptr(out_address).unsafe_origin_cast[MutAnyOrigin](),
            n_rows, n_cols, num_outputs, True,
        )
    except e:
        resident_release[True](handle)
        raise e
    resident_release[True](handle)
    return PythonObject(n_rows)


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

    # cpu-gpu-cleanup t-forest: the same device route as `rf_predict_proba`
    # above, read at output 0 of a one-output vote, which is what
    # `RandomForest.predict`'s REGRESSION branch does.
    if n_rows == 0:
        return PythonObject(0)
    _ = xp
    _ = op
    return _rf_predict_ordered(
        "rf_predict_reg", offsets_p, colid_p, quesval_p, left_p, leaves_p,
        Int(py=x_addr), Int(py=out_addr), n_rows, n_cols, n_trees, 1,
    )


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
        var ctx = process_ctx[_DEVCTX_SLOT]()
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

def rf_device_finite_scan_binding() raises -> PythonObject:
    """1 when this build's RF fits refuse a non-finite X cell through a
    device scan (FOREST_DEVICE_FINITE, FAST on Apple), so the Python fit
    skips its host scan of the same cells; 0 otherwise."""
    return PythonObject(1 if FOREST_DEVICE_FINITE else 0)


def rf_numeric_mode_binding() raises -> PythonObject:
    """Read the numeric policy compiled into this RF binding."""
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def rf_fused_bootstrap_gather_binding() raises -> PythonObject:
    """Whether this binary uses the fused bootstrap/gather arm."""
    return PythonObject(Int(FUSED_BOOTSTRAP_GATHER))


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
    return _rf_classifier_fit(x_addr, y_addr, params, criterion,
                             Int(py=weights_addr), Int(py=tree_start))


def rf_regressor_fit_shard_binding(x_addr: PythonObject, y_addr: PythonObject,
    params: PythonObject, criterion: PythonObject, tree_start: PythonObject) raises -> PythonObject:
    return _rf_regressor_fit(x_addr, y_addr, params, criterion, Int(py=tree_start))


@export
def PyInit__mojolearn_rf() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_rf")
        m.def_function[rf_vendor_binding]("rf_vendor")
        m.def_function[rf_numeric_mode_binding]("rf_numeric_mode")
        m.def_function[rf_device_finite_scan_binding]("rf_device_finite_scan")
        m.def_function[rf_fused_bootstrap_gather_binding](
            "rf_fused_bootstrap_gather"
        )
        m.def_function[rf_classifier_fit_binding[False]]("rf_classifier_fit")
        m.def_function[rf_classifier_fit_binding[True]]("rf_classifier_fit_export")
        m.def_function[rf_classifier_fit_rowmajor_binding[False]]("rf_classifier_fit_rowmajor")
        m.def_function[rf_classifier_fit_rowmajor_binding[True]]("rf_classifier_fit_rowmajor_export")
        m.def_function[rf_classifier_fit_weighted_binding[False]]("rf_classifier_fit_weighted")
        m.def_function[rf_classifier_fit_weighted_binding[True]]("rf_classifier_fit_weighted_export")
        m.def_function[rf_regressor_fit_binding[False]]("rf_regressor_fit")
        m.def_function[rf_regressor_fit_binding[True]]("rf_regressor_fit_export")
        m.def_function[rf_regressor_fit_rowmajor_binding[False]]("rf_regressor_fit_rowmajor")
        m.def_function[rf_regressor_fit_rowmajor_binding[True]]("rf_regressor_fit_rowmajor_export")
        m.def_function[rf_predict_proba_binding]("rf_predict_proba")
        m.def_function[rf_predict_reg_binding]("rf_predict_reg")
        m.def_function[rf_predict_proba_gpu_parallel_binding]("rf_predict_proba_gpu_parallel")
        m.def_function[rf_predict_reg_gpu_parallel_binding]("rf_predict_reg_gpu_parallel")
        m.def_function[forest_resident_layout_binding]("forest_resident_layout")
        m.def_function[forest_ordered_resident_binding]("forest_ordered_resident")
        m.def_function[forest_pool_available]("forest_pool_available")
        m.def_function[forest_pool_fault_available]("forest_pool_fault_available")
        m.def_function[rf_forest_export_binding]("forest_export")
        m.def_function[rf_forest_export_legacy_binding]("forest_export_legacy")
        m.def_function[rf_forest_export_release_binding]("forest_export_release")
        # every snapshot prepared here runs on the registry's process context and
        # borrows its one I/O workspace (core/forest_inference_model.mojo,
        # FOREST_PER_MODEL_IO; gap-fails2 2026-10-02)
        m.def_function[forest_prepare_gpu_binding[True]]("forest_prepare_gpu")
        m.def_function[forest_predict_resident_gpu_binding[True]]("forest_predict_resident_gpu")
        m.def_function[forest_release_gpu_binding[True]]("forest_release_gpu")
        m.def_function[forest_vector_groves_binding]("forest_vector_groves")
        m.def_function[forest_predict_resident_into_gpu_binding[True]]("forest_predict_resident_into_gpu")
        m.def_function[forest_predict_resident_into_gpu_binding[True, True]]("forest_predict_resident_reuse_gpu")
        m.def_function[forest_predict_resident_labels_gpu_binding[True]]("forest_predict_resident_labels_gpu")
        m.def_function[rf_data_session_open_binding]("rf_data_session_open")
        m.def_function[rf_data_session_close_binding]("rf_data_session_close")
        m.def_function[rf_regressor_fit_session_binding]("rf_regressor_fit_session_export")
        m.def_function[rf_regressor_fit_session_rows_binding]("rf_regressor_fit_session_rows_export")
        m.def_function[rf_classifier_fit_weighted_session_binding]("rf_classifier_fit_weighted_session_export")
        m.def_function[rf_classifier_fit_shard_binding]("rf_classifier_fit_shard")
        m.def_function[rf_regressor_fit_shard_binding]("rf_regressor_fit_shard")
        return m.finalize()
    except e:
        abort(String("failed to initialize _mojolearn_rf: ") + String(e))
        return PythonObject(None)
