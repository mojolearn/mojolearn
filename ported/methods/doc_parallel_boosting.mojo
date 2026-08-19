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

        # growth reorders rows in place, so the index restarts each tree
        ctx.enqueue_copy(dst_buf=row_index, src_ptr=h_ident.unsafe_ptr())
        ctx.synchronize()

        # `optimizer.Fit(...)` then `Estimate` then `Rescale` then
        # `AppendModels`, all four inside `run_tree_layout`, in their order.
        var splits = List[TBinarySplit]()
        var leaf_values = List[Float32]()
        var sizes = run_tree_layout(
            ctx, n_rows, fold_counts, max_depth,
            cindex, stats, row_index, cursor,
            Float32(0.0), Float32(0.0), splits, leaf_values,
            True, True, learning_rate, l2_leaf_reg,
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
