"""The boosting loop: gradients, a tree, leaf values, updated predictions.

PORT OF `catboost/cuda/methods/doc_parallel_boosting.h` at CatBoost
`54a8143a`, the `Fit` body at `:302`. Partial: one objective, no test cursor,
no snapshotting, no early stopping, no row sampling. One cursor, which is what
CatBoost itself keeps in this configuration; see the audit below.

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

`TTargetAtPointTrait::Create(objective, cursor)` is `mse_kernel` here. `RMSE`
resolves to `PointwiseTargetImpl<TRmseTarget>`, dispatched at
`pointwise_targets.cu:496-500`, and NOT to `MseImpl`, which nothing in their
tree calls; see the module docstring of `pointwise_targets.mojo`.

**The one step of theirs this file adds is not theirs.** Between the target
kernel and the tree it computes two sums of absolute values and hands them to
`choose_scale`. CatBoost has no counterpart because it flushes histograms
through a float atomic and needs no range bound. That block is load bearing
and its contract audit is written out at the call site.

## Their steps we do NOT take, audited one by one

Two of the four this file used to list turned out not to be divergences at
all in the configuration this port runs, and saying so is part of the audit:

- **Permutations. NOT A DIVERGENCE HERE.** `Config.PermutationCount` defaults
  to 4 (`boosting_options.cpp:14`), but `UpdateGpuSpecificDefaults` sets it
  back to 1 whenever the dataset has no permutation-dependent features and
  boosting is Plain, which is the default (`cuda/train_lib/train.cpp:102-108`,
  "No catFeatures for ctrs found and don't look ahead is disabled. Fallback to
  one permutation"). No CTRs are ported, so stock CatBoost on this input keeps
  exactly one cursor too. One is theirs, not a simplification of theirs.
- **`CalcScoreModelLengthMult` (`:358`). NOT A DIVERGENCE HERE, and it is not
  model-size regularization.** Its only consumer is `ComputeScoreStdDev`,
  which returns 0 unconditionally when `modelLengthMult * randomStrength` is
  zero (`random_score_helper.h:25-32`), and it reaches the searcher solely as
  `options.RandomStrength *= randomStrengthMult`
  (`greedy_subsets_searcher.h:76`). At `random_strength = 0`, which is what
  `CatBoostOptions.check()` requires because score noise is unported, the
  whole quantity is an exact no-op. It becomes a divergence the day random
  strength lands, and not before. (`model_size_reg` is a different option
  entirely and lives on the feature weights.)
- **Test cursor and early stopping. A REAL GAP.** `ShouldStop()` is a fixed
  iteration count here, and `TrackTestErrors` / `IsBestTestIteration` /
  `BestTestCursor` have no counterpart. Nothing selects a best iteration, so
  a fit runs the full budget.
- **Snapshotting. A REAL GAP**, and a cosmetic one: `MaybeSaveSnapshot` and
  `MaybeRestoreFromSnapshot` change no model, only whether a run can resume.

## The estimator, which does NOT run here and does not need to

Their loop calls `estimator.Estimate(...)` (`:371`) whenever
`NeedEstimation()`, and that is `LeavesEstimationMethod != Simple`
(`greedy_subsets_searcher.h:67-69`), so under RMSE's default of Newton it
runs on every iteration and OVERWRITES the leaf values the structure search
produced. This port has no second pass: `run_tree_layout` computes leaf
values once. That is sound only because the two answers coincide for MSE,
and `leaves_estimation.mojo` carries the derivation and the two file:line
citations rather than leaving it to be assumed.
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
from mojo_only.kernel_matrix import TARGET_COLUMN, deterministic_flush_for
from mojo_only.numerics import NUMERIC_IDENTICAL
from ported.gpu_util.kernel.fill import launch_make_sequence
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    BUILD_MODE as HIST_BUILD_MODE,
)
from ported.targets.kernel.pointwise_targets import MSE_BLOCK_SIZE, mse_kernel


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
    learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    use_subtraction: Bool = True,
) raises -> List[Float64]:
    """Their `Fit` (`doc_parallel_boosting.h:302`), one permutation.

    `learning_rate` is their `Config.LearningRate`, whose default is **0.03**
    (`boosting_options.cpp:10`). It stood at 0.3 here, a tenfold larger step
    than stock CatBoost, which is a different model for every caller that did
    not pass one. See `CatBoostOptions.learning_rate` for the data-dependent
    retune (`options_helper.cpp:269-288`) that is deliberately not ported.

    Returns the weighted squared error per row after each iteration, computed
    on the host from the cursor. That is their `functionValue`
    (`pointwise_targets.cu:271`) with the sign flipped and divided by the row
    count: theirs accumulates `-weight * (val - relev)^2` because everything
    downstream of it maximizes, and a caller here wants a number that falls.
    The reduction moves to the host because their kernel-side version needs a
    block reduce plus a float `atomicAdd` into one scalar, and pulling it back
    costs one copy per iteration on no hot path.

    The loss is returned rather than printed because the thing worth
    asserting is that it DECREASES, and an assertion needs the numbers.
    """
    var stat_count = 2

    # their `cursor`: the running prediction for every row. Zeros is THEIR
    # default branch, not a simplification of it: with no baseline column and
    # `boost_from_average` false (`boosting_options.cpp:17`),
    # `cursors->StartingPoint` is unset and `CreateCursors` writes
    # `TVector<float> start(sampleCount, 0.0)`
    # (`doc_parallel_boosting.h:180-186`). `boost_from_average` true would
    # seed it with `CalcOptimumConstApprox` and is refused by
    # `CatBoostOptions.check()`.
    var cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var h_zero = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for i in range(n_rows):
        h_zero.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=cursor, src_ptr=h_zero.unsafe_ptr())

    # their der/der2 buffers, in the two-plane layout the histogram kernels
    # already read. See the DEVIATION BLOCK in `pointwise_targets.mojo`.
    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)

    # growth permutes this, so it is reseeded to the identity every
    # iteration exactly as their per-iteration dataset view is -- ON THE
    # DEVICE, their `MakeSequence` (`fill.cu:47`). This used to upload a
    # host-built identity, 3.2 MB of H2D per tree at 800k rows.
    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)

    # their `functionValue`: ONE float, accumulated by `mse_kernel`'s block
    # reduce + atomicAdd (`pointwise_targets.cu:309-317`). This replaces a
    # HOST loop over every row per iteration (~5 ms/tree at 800k), which
    # was never their design.
    var fv = ctx.enqueue_create_buffer[DType.float32](1)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](1)

    # THE FIXED-POINT BOUND, for the IDENTICAL build only. The integer
    # flush needs a scale bounded by `sum over all rows of abs(plane)`,
    # read back from the plane the target kernel actually wrote. The FAST
    # build flushes through CatBoost's float atomic (which Metal HAS; the
    # sentence that stood here claiming otherwise was the falsified import
    # -path reading), never touches the accumulator, and skips the readback
    # and both host scans at comptime.
    comptime _flush_fixed = deterministic_flush_for[
        TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
    ]()
    var h_stats = ctx.enqueue_create_host_buffer[DType.float32](
        (stat_count * n_rows) if _flush_fixed else 1
    )

    var losses = List[Float64]()
    # their `TVector<TResultModel>* result`, the ensemble being built

    for _ in range(n_estimators):
        # `TTargetAtPointTrait::Create(learnTarget, cursor)` (`:353`).
        # The gradients are taken AT THE CURRENT PREDICTIONS, which is the
        # whole of what makes this boosting.
        # their `functionValue` is accumulated in the SAME launch as the
        # gradients, so the value this iteration reads is the loss of the
        # cursor as it stands -- i.e. after the PREVIOUS tree.
        ctx.enqueue_memset(fv, Float32(0.0))
        ctx.enqueue_function[mse_kernel](
            targets.unsafe_ptr(), weights.unsafe_ptr(), Int32(n_rows),
            cursor.unsafe_ptr(),
            Int32(1) if has_weights else Int32(0),
            stats.unsafe_ptr(),
            fv.unsafe_ptr(), Int32(1),
            grid_dim=(n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE,
            block_dim=MSE_BLOCK_SIZE,
        )
        # stream-ordered behind the kernel; READ after `run_tree_layout`,
        # whose first drain settles it, so it costs no synchronize here.
        ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=fv)

        # growth reorders rows in place, so the index restarts each tree.
        # their `MakeSequence` (`fill.cu:47`), device-side.
        launch_make_sequence(ctx, UInt32(0), row_index, n_rows)

        @parameter
        if _flush_fixed:
            # The gradients the histogram will accumulate, read back so the
            # fixed-point scale can be bounded by them. IDENTICAL only.
            ctx.enqueue_copy(dst_ptr=h_stats.unsafe_ptr(), src_buf=stats)
            ctx.synchronize()

        # `sum over all rows of abs(...)`, per plane, which is what
        # `choose_scale` is specified against. Plane 0 is the WEIGHT plane
        # and plane 1 is `der`, laid out `stats[s * n_rows + i]` by
        # `mse_kernel`, which is their `StatsToAggregate` column order
        # (`pointwise_target_impl.h:188-192`). Plane 0 is the sample weight
        # rather than `der2` under Cosine, their default score function, and
        # the two coincide for RMSE; see `pointwise_targets.mojo`.
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
        #
        # CONTRACT AUDIT, 2026-08-19. Every caller of `choose_scale` on the
        # tree path was re-read against `mojo_only/fixed_point.mojo`'s stated
        # precondition, "sum over all rows of abs(value) for the plane being
        # accumulated":
        #
        #   this file                      sum of |stats| read back from the
        #                                  plane `mse_kernel` wrote, both
        #                                  planes, per iteration -- SOUND
        #   mojo_only/level_check.mojo     abs-accumulates a signed generator
        #                                  at :210-218 and :344-352 -- SOUND
        #   mojo_only/level_bench.mojo     same, three sites -- SOUND
        #   probe_main.check_fixed_point   sums |rows| directly -- SOUND
        #
        # The permutation growth applies to the stats plane does not move the
        # bound: a gather is a bijection on rows, so the sum of magnitudes is
        # invariant, and the sibling subtraction cannot exceed the parent's
        # bound either. The three reserved headroom bits cover the rest,
        # including the one place the bound is not exact: these sums are
        # accumulated in Float64 and narrowed to Float32 at the call below, so
        # a round DOWN yields a scale up to one part in 2^24 too large.
        # `SCALE_HEADROOM_BITS = 3` is a factor of eight against a relative
        # error of 6e-8.
        #
        # If another lane restores the float atomic on the FAST arm, this
        # block does NOT become dead. `DETERMINISM_DEVICE` is the default and
        # it pins the integer flush on every backend, so the contract still
        # governs the shipped configuration; only the `OFF` arm escapes it.
        # ===========================================================
        var weight_magnitude = Float64(0.0)
        var gradient_magnitude = Float64(0.0)

        @parameter
        if _flush_fixed:
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

        # the device `functionValue` read alongside this iteration's
        # gradients: the loss AFTER THE PREVIOUS TREE (their accumulation
        # is `-w * (val - relev)^2`, `pointwise_targets.cu:311`, so the
        # positive loss is its negation). Iteration 0's value is the
        # baseline and is not a tree's loss, so it is dropped, and the
        # LAST tree's loss comes from one extra gradient pass below.
        if len(losses) < n_estimators:
            var v = Float64(h_fv.unsafe_ptr().unsafe_load(0))
            if len(model.weak_models) > 1:
                losses.append(-v / Float64(n_rows))

    # the final tree's loss: one more `functionValue` pass over the settled
    # cursor. The gradient planes it writes feed nothing.
    ctx.enqueue_memset(fv, Float32(0.0))
    ctx.enqueue_function[mse_kernel](
        targets.unsafe_ptr(), weights.unsafe_ptr(), Int32(n_rows),
        cursor.unsafe_ptr(),
        Int32(1) if has_weights else Int32(0),
        stats.unsafe_ptr(),
        fv.unsafe_ptr(), Int32(1),
        grid_dim=(n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE,
        block_dim=MSE_BLOCK_SIZE,
    )
    ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=fv)
    ctx.synchronize()
    losses.append(
        -Float64(h_fv.unsafe_ptr().unsafe_load(0)) / Float64(n_rows)
    )

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
