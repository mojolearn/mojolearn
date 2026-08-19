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
from ported.targets.kernel.pointwise_targets import mse_kernel


def fit(
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
        var sizes = run_tree_layout(
            ctx, n_rows, fold_counts, max_depth,
            cindex, stats, row_index, cursor,
            Float32(0.0), Float32(0.0),
            True, True, learning_rate, l2_leaf_reg,
        )
        _ = len(sizes)

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
