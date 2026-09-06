# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Single-permutation RMSE Ordered loop from dynamic_boosting.h:280-469.

Supported contract: quantized numeric/one-hot columns, explicit permutation,
zero starting point, no bootstrap, one Newton step, one GPU. Each fold owns
its cursor; only EstimateSamples enters its leaf oracle. Quality rows enter
structure scoring, as upstream requires, but never that fold's estimation.
The separately estimated all-row model is the only model exported.

Reuses the existing GPU fold searcher and Newton oracle. The additional
device kernels implement target-at-point gather and cached-bin application.
There is no host gradient/leaf reduction or alternate CPU training path.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, block_dim, thread_idx
from std.math import isfinite
from checks.numerics import identical_mul, identical_mul_add
from checks.fixed_point import choose_scale
from core.identity_trace import IdentityTrace
from gbdt.methods.dynamic_boosting_folds import TFold, EBoostingType, IQueriesGrouping, create_folds
from gbdt.methods.doc_parallel_boosting import TEstimationWorkspace, _estimate_and_apply
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import StageTimes
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import PointwiseTreeWorkspace, fit_oblivious_tree_structure
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import compute_bins_for_model, partition_from_bins
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.models.oblivious_model import TAdditiveModel, TObliviousTreeModel, TObliviousTreeStructure
from gbdt.options.catboost_options import LEAF_ESTIMATION_NEWTON
from gbdt.methods.kernel.pointwise_scores import SCORE_FUNCTION_SOLAR_L2, SCORE_FUNCTION_COSINE, SCORE_FUNCTION_NEWTON_COSINE
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_RMSE


@fieldwise_init
struct OrderedFitResult(Movable):
    var model: TAdditiveModel
    var folds: List[TFold]
    # Cursors use permutation POSITION, not original document id.
    var fold_cursors: List[DeviceBuffer[DType.float32]]
    var estimation_cursor: DeviceBuffer[DType.float32]


def _ordered_target_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    permutation: MutPointer[UInt32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    out_w: MutPointer[Float32, MutAnyOrigin],
    out_g: MutPointer[Float32, MutAnyOrigin],
    size: Int, offset: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var row = Int(permutation.unsafe_load(i))
        var w = weights.unsafe_load(row)
        out_w.unsafe_store(offset + i, w)
        out_g.unsafe_store(offset + i, identical_mul(w, y.unsafe_load(row) - cursor.unsafe_load(i)))


def _ordered_gather_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    permutation: MutPointer[UInt32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    out_y: MutPointer[Float32, MutAnyOrigin],
    out_w: MutPointer[Float32, MutAnyOrigin],
    out_c: MutPointer[Float32, MutAnyOrigin],
    out_b: MutPointer[UInt32, MutAnyOrigin],
    size: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var row = Int(permutation.unsafe_load(i))
        out_y.unsafe_store(i, y.unsafe_load(row))
        out_w.unsafe_store(i, weights.unsafe_load(row))
        out_c.unsafe_store(i, cursor.unsafe_load(i))
        out_b.unsafe_store(i, bins.unsafe_load(row))


def _ordered_apply_kernel(
    permutation: MutPointer[UInt32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    size: Int, rate: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var row = Int(permutation.unsafe_load(i))
        var leaf = Int(bins.unsafe_load(row))
        # Upstream Rescale(step) precedes AppendModels. Round the stored
        # model value before adding, so exported-model prediction and the
        # training cursor use the same arithmetic across multiple trees.
        var scaled = identical_mul(leaves.unsafe_load(leaf), rate)
        cursor.unsafe_store(i, identical_mul_add(scaled, Float32(1), cursor.unsafe_load(i)))


def ordered_estimate_and_apply(
    ctx: DeviceContext, estimate_size: Int, apply_size: Int,
    n_leaves: Int, mut y: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut permutation: DeviceBuffer[DType.uint32],
    mut bins: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    rate: Float32, l2: Float32, sm_count: Int,
    mut trace: IdentityTrace, tag: String,
) raises -> List[Float32]:
    """Estimate on [0,L), apply on [0,R); never alias estimation inputs."""
    if estimate_size < 1 or estimate_size > apply_size:
        raise Error("ordered estimation requires 0 < prefix <= cursor size")
    var gy = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gw = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gc = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gb = ctx.enqueue_create_buffer[DType.uint32](estimate_size)
    ctx.enqueue_function[_ordered_gather_kernel](
        y.unsafe_ptr(), weights.unsafe_ptr(), permutation.unsafe_ptr(),
        cursor.unsafe_ptr(), bins.unsafe_ptr(), gy.unsafe_ptr(),
        gw.unsafe_ptr(), gc.unsafe_ptr(), gb.unsafe_ptr(), estimate_size,
        grid_dim=((estimate_size + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var part = partition_from_bins(ctx, gb, estimate_size, n_leaves)
    var leaves = List[Float32]()
    var not_pd = 0
    var times = StageTimes()
    times.enabled = False
    var ws = List[TEstimationWorkspace]()
    # The estimator updates its private prefix copy. The real fold cursor
    # is updated once below, including the quality-evaluation tail.
    _estimate_and_apply(
        ctx, estimate_size, 1, n_leaves, part.sizes, part.offsets,
        part.row_index, gy, gw, True, gc, OBJECTIVE_RMSE,
        Float32(0), Float32(0.5), Float32(0.5), l2, sm_count,
        LEAF_ESTIMATION_NEWTON, 0, 1, rate, leaves, not_pd,
        trace, times, tag, ws,
    )
    var hl = ctx.enqueue_create_host_buffer[DType.float32](n_leaves)
    var dl = ctx.enqueue_create_buffer[DType.float32](n_leaves)
    for leaf in range(n_leaves):
        hl.unsafe_ptr().unsafe_store(leaf, leaves[leaf])
    ctx.enqueue_copy(dst_buf=dl, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_function[_ordered_apply_kernel](
        permutation.unsafe_ptr(), bins.unsafe_ptr(), dl.unsafe_ptr(),
        cursor.unsafe_ptr(), apply_size, rate,
        grid_dim=((apply_size + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    _ = hl^
    _ = dl^
    _ = gb^
    return leaves^


def fit_ordered_rmse(
    ctx: DeviceContext, layout: CompressedIndexLayout,
    mut cindex: DeviceBuffer[DType.uint32],
    y: List[Float32], sample_weight: List[Float32],
    permutation: List[UInt32], n_estimators: Int, max_depth: Int,
    sm_count: Int, learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    score_function: Int = SCORE_FUNCTION_COSINE,
    one_hot: List[Bool] = List[Bool](),
    growth_rate: Float64 = 2.0, min_fold_size: Int = 100,
) raises -> OrderedFitResult:
    """Native usable Ordered entry; explicit row permutation is mandatory.

    Unsupported objectives, bootstrap and multi-permutation categorical CTR
    options are deliberately absent from this RMSE-only signature.
    """
    var n = len(y)
    if n < 4 or len(permutation) != n or n_estimators < 1:
        raise Error("ordered RMSE needs >=4 rows, a permutation and >=1 tree")
    if max_depth < 1 or max_depth > 8 or sm_count < 1:
        raise Error("ordered RMSE supports depth 1..8 and positive SM count")
    if min_fold_size < 1 or not isfinite(growth_rate) or growth_rate <= Float64(1):
        raise Error("ordered RMSE requires positive min fold size and finite growth >1")
    if len(sample_weight) != 0 and len(sample_weight) != n:
        raise Error("ordered RMSE sample weight shape mismatch")
    if not isfinite(learning_rate) or learning_rate <= Float32(0) or not isfinite(l2_leaf_reg) or l2_leaf_reg < Float32(0):
        raise Error("ordered RMSE requires finite positive rate and nonnegative L2")
    if score_function != SCORE_FUNCTION_SOLAR_L2 and score_function != SCORE_FUNCTION_COSINE and score_function != SCORE_FUNCTION_NEWTON_COSINE:
        raise Error("ordered RMSE supports SolarL2, Cosine and NewtonCosine")
    if len(layout.features) == 0 or (len(one_hot) != 0 and len(one_hot) != len(layout.features)):
        raise Error("ordered RMSE feature shape mismatch")
    var seen = List[Bool]()
    for _ in range(n):
        seen.append(False)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hp = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var weight_sum = Float64(0)
    var max_weight = Float64(0)
    var residual_bound = Float64(0)
    for i in range(n):
        var row = Int(permutation[i])
        if row >= n or seen[row]:
            raise Error("ordered RMSE permutation must be a bijection")
        seen[row] = True
        var w = Float32(1) if len(sample_weight) == 0 else sample_weight[i]
        if not isfinite(y[i]) or not isfinite(w) or w < Float32(0) or not isfinite(identical_mul(w, y[i])):
            raise Error("ordered RMSE targets/weights must be finite; weights nonnegative")
        weight_sum += Float64(w)
        max_weight = max(max_weight, Float64(w))
        residual_bound = max(residual_bound, abs(Float64(y[i])))
        hy.unsafe_ptr().unsafe_store(i, y[i])
        hw.unsafe_ptr().unsafe_store(i, w)
        hp.unsafe_ptr().unsafe_store(i, permutation[i])
    if weight_sum <= Float64(0):
        raise Error("ordered RMSE weights sum to zero")
    var folds = create_folds(n, growth_rate, IQueriesGrouping.without_queries(n), EBoostingType.Ordered, min_fold_size, 1)
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var dw = ctx.enqueue_create_buffer[DType.float32](n)
    var dp = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dw, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dp, src_ptr=hp.unsafe_ptr())
    var cursors = List[DeviceBuffer[DType.float32]]()
    var total = 0
    for f in range(len(folds)):
        var size = folds[f].quality_evaluate_samples.right
        var cursor = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_memset(cursor, Float32(0))
        cursors.append(cursor^)
        total += size
    var estimation = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_memset(estimation, Float32(0))
    var model = TAdditiveModel()
    var pool = List[PointwiseTreeWorkspace]()
    var trace = IdentityTrace()
    ctx.synchronize()
    for iteration in range(n_estimators):
        # A Newton-1 RMSE leaf is a regularized weighted mean residual.
        # Consequently |cursor_next| <= |cursor| + rate * max|residual|.
        # Bound all duplicated fold positions, not merely the original rows.
        # This is conservative scale selection, not a CPU gradient reduction.
        var magnitude_bound = Float64(total) * max_weight * max(Float64(1), residual_bound)
        if not isfinite(magnitude_bound):
            raise Error("ordered RMSE fixed-point magnitude bound overflow")
        var scale = Float32(choose_scale(magnitude_bound, total))
        var sw = ctx.enqueue_create_buffer[DType.float32](total)
        var sg = ctx.enqueue_create_buffer[DType.float32](total)
        var offset = 0
        for f in range(len(folds)):
            var size = folds[f].quality_evaluate_samples.right
            ctx.enqueue_function[_ordered_target_kernel](
                dy.unsafe_ptr(), dw.unsafe_ptr(), dp.unsafe_ptr(),
                cursors[f].unsafe_ptr(), sw.unsafe_ptr(), sg.unsafe_ptr(),
                size, offset,
                grid_dim=((size + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
            offset += size
        var splits = fit_oblivious_tree_structure(
            ctx, layout, n, max_depth, cindex, sw^, sg^, sm_count,
            scale, score_function, pool, l2_leaf_reg,
            one_hot=one_hot, folds=folds, permutation=permutation,
        )
        var bins = ctx.enqueue_create_buffer[DType.uint32](n)
        if len(splits) == 0:
            ctx.enqueue_memset(bins, UInt32(0))
        else:
            compute_bins_for_model(ctx, layout, splits, len(splits), cindex, n, bins)
        var n_leaves = 1 << len(splits)
        for f in range(len(folds)):
            var unused = ordered_estimate_and_apply(
                ctx, folds[f].estimate_samples.right,
                folds[f].quality_evaluate_samples.right,
                n_leaves, dy, dw, dp, bins, cursors[f], learning_rate,
                l2_leaf_reg, sm_count, trace,
                String("ordered.") + String(iteration) + ".fold." + String(f),
            )
            trace.record_device(ctx, String("ordered.") + String(iteration) + ".cursor." + String(f), cursors[f])
        var leaves = ordered_estimate_and_apply(
            ctx, n, n, n_leaves, dy, dw, dp, bins, estimation,
            learning_rate, l2_leaf_reg, sm_count, trace,
            String("ordered.") + String(iteration) + ".estimation",
        )
        var structure = TObliviousTreeStructure()
        trace.record_device(ctx, String("ordered.") + String(iteration) + ".estimation_cursor", estimation)
        structure.splits = splits^
        var weak = TObliviousTreeModel(structure^)
        for leaf in range(n_leaves):
            weak.leaf_values.append(identical_mul(leaves[leaf], learning_rate))
        model.add_weak_model(weak^)
        residual_bound *= Float64(1) + Float64(learning_rate)
    ctx.synchronize()
    _ = hy^
    _ = hw^
    _ = hp^
    return OrderedFitResult(model^, folds^, cursors^, estimation^)
