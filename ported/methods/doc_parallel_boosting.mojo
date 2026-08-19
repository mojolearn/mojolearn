"""The boosting loop: gradients, a tree, leaf values, updated predictions.

PORT OF `catboost/cuda/methods/doc_parallel_boosting.h` at CatBoost
`54a8143a`, the `Fit` body at `:302`. Partial: one objective, no permutations,
no test cursor, no snapshotting, no early stopping.

Their iteration, stripped to what exists here (`:336-400`):

    while (!progressTracker->ShouldStop()) {
        target  = TTargetAtPointTrait::Create(learnTarget, cursor);
        model   = optimizer.Fit(taskDataSet, target);   // structure search
        estimator.Estimate(...);                        // leaf values
        model.Rescale(step);                            // learning rate
        AppendModels(...);                              // cursor += model
        result.AddWeakModel(model);
    }

The ordering is the algorithm and it is kept exactly. In particular the
target is rebuilt from the CURSOR at the top of every iteration, which is
what makes this gradient boosting rather than a forest: each tree fits the
residual left by all the trees before it.

`TTargetAtPointTrait::Create(objective, cursor)` is `mse_kernel` here, since
`RMSE` resolves to `MseImpl` (`pointwise_targets.cu:496`).

## What is NOT ported yet, and matters

- **Permutations.** They keep `PermutationsCount()` cursors and pick one at
  random per iteration for the structure search, which is their ordered
  boosting defence against target leakage. We keep one.
- **Test cursor and early stopping.** `ShouldStop()` is a fixed iteration
  count here.
- **`CalcScoreModelLengthMult`** (`:358`), their model-size regularisation.
- **Snapshotting**, which is `MaybeSaveSnapshot` in their loop.

Each of those changes the ANSWER, not just the plumbing, so none of them is
a detail. They are listed rather than silently skipped.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from ported.gpu_lib.gpu_manager import TCudaManager
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_tree_layout,
)
from ported.models.oblivious_model import (
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from ported.gpu_data.compressed_index_builder import build_layout
from ported.models.kernel.add_bin_values import compute_bins_and_add_kernel
from ported.targets.kernel.pointwise_targets import mse_kernel


def fit(
    mut model: TAdditiveModel,
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    max_depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    n_estimators: Int,
    learning_rate: Float32 = Float32(0.3),
    l2_leaf_reg: Float32 = Float32(3.0),
    use_subtraction: Bool = True,
) raises -> List[Float64]:
    """Their `Fit` (`doc_parallel_boosting.h:302`), one permutation.

    Returns the mean squared error after each iteration, computed on the host
    from the cursor. Their `functionValue` accumulates the NEGATIVE squared
    error inside the target kernel with a float `atomicAdd`; Metal has no
    float atomic, so the reduction moves to the host where it costs one copy
    per iteration and sits on no hot path.

    The loss is returned rather than printed because the thing worth
    asserting is that it DECREASES, and an assertion needs the numbers.
    """
    var stat_count = 2

    # their `cursor`: the running prediction for every row. CatBoost seeds it
    # from the baseline or from zero; zero here.
    var cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var h_zero = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for i in range(n_rows):
        h_zero.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=cursor, src_ptr=h_zero.unsafe_ptr())

    # their der/der2 buffers, in the two-plane layout the histogram kernels
    # already read. See the DEVIATION BLOCK in `pointwise_targets.mojo`.
    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)

    # growth permutes this, so it is reseeded to the identity every iteration
    # exactly as their per-iteration dataset view is.
    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var h_ident = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        h_ident.unsafe_ptr().unsafe_store(i, UInt32(i))

    var h_cursor = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var h_target = ctx.enqueue_create_host_buffer[DType.float32](n_rows)

    # THE FIXED-POINT BOUND, and it is not optional. The histogram flush
    # accumulates through an Int32 because Metal has no float atomic
    # (`mojo_only/fixed_point.mojo`), and the scale that keeps every slot
    # inside Int32 is derived from `sum over all rows of abs(plane)`, one per
    # stat plane. It has to be read back from the plane the target kernel
    # actually wrote, not recomputed from the target here: recomputing it
    # would make this file a second definition of the gradient.
    var h_stats = ctx.enqueue_create_host_buffer[DType.float32](
        stat_count * n_rows
    )

    ctx.enqueue_copy(dst_ptr=h_target.unsafe_ptr(), src_buf=targets)
    ctx.synchronize()

    var losses = List[Float64]()
    # their `TVector<TResultModel>* result`, the ensemble being built

    for _ in range(n_estimators):
        # `TTargetAtPointTrait::Create(learnTarget, cursor)` (`:353`).
        # The gradients are taken AT THE CURRENT PREDICTIONS, which is the
        # whole of what makes this boosting.
        ctx.enqueue_function[mse_kernel](
            targets.unsafe_ptr(), weights.unsafe_ptr(), Int32(n_rows),
            cursor.unsafe_ptr(),
            Int32(1) if has_weights else Int32(0),
            stats.unsafe_ptr(),
            grid_dim=(n_rows + 255) // 256, block_dim=256,
        )

        # The gradients the histogram will accumulate, read back so the
        # fixed-point scale can be bounded by them. It rides on the
        # synchronize the row index already needs, so it costs a copy and no
        # extra drain.
        ctx.enqueue_copy(dst_ptr=h_stats.unsafe_ptr(), src_buf=stats)

        # growth reorders rows in place, so the index restarts each tree
        ctx.enqueue_copy(dst_buf=row_index, src_ptr=h_ident.unsafe_ptr())
        ctx.synchronize()

        # `sum over all rows of abs(...)`, per plane, which is what
        # `choose_scale` is specified against. Plane 0 is `der2` (the
        # weights) and plane 1 is `der`, laid out `stats[s * n_rows + i]`
        # by `mse_kernel`.
        #
        # ===================== WHY THIS EXISTS =====================
        # This call passed `0.0, 0.0` for both. Zero is not "unknown", it is
        # the input for which the scale derivation returns its LARGEST value,
        # 2^28 - 1 -- so `Int32(val * scale)` overflowed on any histogram
        # cell holding more than about eight rows' worth of weight.
        #
        # It was invisible for as long as no replicated kernel reached the
        # Int32 flush on this path: the half-byte kernel took the plain store
        # on both branches, and `boosting_check` is 16 half-byte features and
        # nothing else. The moment `launch_histograms_for_blocks` began
        # launching half-byte with `groups * replicas`, the top two levels of
        # every tree started scoring a wrapped-around histogram. The tree
        # still learns, because the leaf VALUES come from
        # `compute_partition_stats` and never touch the accumulator; only the
        # SPLITS go bad. That is exactly a model that stays monotone, still
        # beats the mean, and is several times worse than it should be.
        # ===========================================================
        var weight_magnitude = Float64(0.0)
        var gradient_magnitude = Float64(0.0)
        for i in range(n_rows):
            var w = Float64(h_stats.unsafe_ptr().unsafe_load(i))
            if w < 0.0:
                w = -w
            weight_magnitude += w
            var g = Float64(
                h_stats.unsafe_ptr().unsafe_load(n_rows + i)
            )
            if g < 0.0:
                g = -g
            gradient_magnitude += g

        # `optimizer.Fit(...)` then `Estimate` then `Rescale` then
        # `AppendModels`, all four inside `run_tree_layout`, in their order.
        var splits = List[TBinarySplit]()
        var leaf_values = List[Float32]()
        var sizes = run_tree_layout(
            ctx, n_rows, fold_counts, max_depth,
            cindex, stats, row_index, cursor,
            Float32(weight_magnitude), Float32(gradient_magnitude),
            splits, leaf_values,
            use_subtraction, True, learning_rate, l2_leaf_reg,
        )
        _ = len(sizes)

        # their `result[i].AddWeakModel(iterationModels[i])` (`:398`).
        # `Rescale(step)` is folded into `add_model_value_kernel`, so the
        # stored values are UNSCALED and the rate is applied on the way out.
        var structure = TObliviousTreeStructure()
        for i in range(len(splits)):
            structure.splits.append(splits[i])
        var weak = TObliviousTreeModel(structure^)
        for i in range(len(leaf_values)):
            weak.leaf_values.append(leaf_values[i] * learning_rate)
        model.add_weak_model(weak^)

        ctx.enqueue_copy(dst_ptr=h_cursor.unsafe_ptr(), src_buf=cursor)
        ctx.synchronize()
        var sse = Float64(0.0)
        for i in range(n_rows):
            var d = Float64(
                h_target.unsafe_ptr().unsafe_load(i)
                - h_cursor.unsafe_ptr().unsafe_load(i)
            )
            sse += d * d
        losses.append(sse / Float64(n_rows))

    return losses^


def predict(
    model: TAdditiveModel,
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    mut cindex: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    """Apply a stored ensemble by EVALUATING every tree.

    Their `TAdditiveModel` prediction: sum over weak models. The learning
    rate is already in the stored leaf values (`Rescale(step)`), so nothing
    here reapplies it.

    This is the path that works on rows the model was never grown on, which
    the partition-based `add_model_value_kernel` cannot do. On the learn set
    the two must agree exactly, and `boosting_check` asserts that.
    """
    var layout = build_layout(fold_counts)

    var h_zero = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for i in range(n_rows):
        h_zero.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=cursor, src_ptr=h_zero.unsafe_ptr())

    for t in range(model.size()):
        ref weak = model.weak_models[t]
        var depth = weak.structure.get_depth()
        if depth == 0:
            continue

        var d_off = ctx.enqueue_create_buffer[DType.uint32](depth)
        var d_shift = ctx.enqueue_create_buffer[DType.uint32](depth)
        var d_mask = ctx.enqueue_create_buffer[DType.uint32](depth)
        var d_bin = ctx.enqueue_create_buffer[DType.uint32](depth)
        var h_off = ctx.enqueue_create_host_buffer[DType.uint32](depth)
        var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](depth)
        var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](depth)
        var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](depth)
        for level in range(depth):
            ref cf = layout.features[Int(weak.structure.splits[level].feature_id)]
            h_off.unsafe_ptr().unsafe_store(
                level, cf.offset * UInt32(n_rows)
            )
            h_shift.unsafe_ptr().unsafe_store(level, cf.shift)
            h_mask.unsafe_ptr().unsafe_store(level, cf.mask)
            h_bin.unsafe_ptr().unsafe_store(
                level, UInt32(Int(weak.structure.splits[level].bin_idx))
            )
        ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())

        var n_leaves = 1 << depth
        var d_vals = ctx.enqueue_create_buffer[DType.float32](n_leaves)
        var h_vals = ctx.enqueue_create_host_buffer[DType.float32](n_leaves)
        for i in range(n_leaves):
            var v = Float32(0.0)
            if i < len(weak.leaf_values):
                v = weak.leaf_values[i]
            h_vals.unsafe_ptr().unsafe_store(i, v)
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())

        var wide = (n_rows + 255) // 256
        if wide > 1024:
            wide = 1024
        ctx.enqueue_function[compute_bins_and_add_kernel](
            cindex.unsafe_ptr(), d_off.unsafe_ptr(), d_shift.unsafe_ptr(),
            d_mask.unsafe_ptr(), d_bin.unsafe_ptr(), Int32(depth),
            d_vals.unsafe_ptr(), Int32(n_rows), cursor.unsafe_ptr(),
            grid_dim=wide, block_dim=256,
        )
        ctx.synchronize()
