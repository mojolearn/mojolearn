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

`TTargetAtPointTrait::Create(objective, cursor)` is `launch_approximate`
here -- their `Approximate` fork (`pointwise_target_impl.h:307-358`) between
`CrossEntropyImpl` and the generic `PointwiseTargetImpl<TTarget>`. `RMSE`
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
from max.gpu.host.device_attribute import DeviceAttribute
from gbdt.methods.kernel_add_model_value import add_model_value_kernel

from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.gpu_util.kernel.transform import (
    launch_gather_planes_with_mask_f32,
    launch_gather_with_mask_f32,
)
from gbdt.methods.leaves_estimation.descent_helpers import (
    newton_like_walker_estimate,
)
from gbdt.methods.leaves_estimation.pointwise_oracle import (
    make_bin_optimized_oracle,
)
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_tree_layout,
    TTreeWorkspace,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.models.kernel.add_bin_values import compute_bins_and_add_kernel
from mojo_only.kernel_matrix import TARGET_COLUMN, deterministic_flush_for
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE
from mojo_only.numerics import NUMERIC_IDENTICAL
from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.bootstrap import (
    bootstrap_grid_blocks,
    create_bootstrap_seeds,
    BOOTSTRAP_KERNEL_BAYESIAN,
    BOOTSTRAP_KERNEL_BERNOULLI,
    BOOTSTRAP_KERNEL_POISSON,
    launch_bootstrap,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    BUILD_MODE as HIST_BUILD_MODE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import (
    HIST2_SMEM_IS_I32,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_EXACT,
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
    LEAF_ESTIMATION_SIMPLE,
)
from gbdt.targets.kernel.multilogit import (
    launch_multilogit_value_and_der_search,
    multilogit_blocks,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_RMSE,
    deterministic_sum_lanes_kernel,
    launch_approximate,
)


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
    bootstrap_bayesian: Bool = False,
    bagging_temperature: Float32 = Float32(1.0),
    bootstrap_type: Int = -1,
    bootstrap_param: Float32 = Float32(1.0),
    random_seed: UInt64 = UInt64(0),
    one_hot: List[Bool] = List[Bool](),
    score_function: Int = SCORE_FUNCTION_COSINE,
    objective: Int = OBJECTIVE_RMSE,
    num_classes: Int = 0,
    logloss_border: Float32 = Float32(0.5),
    leaf_estimation_iterations: Int = 1,
    leaf_estimation_method: Int = LEAF_ESTIMATION_NEWTON,
    alpha: Float32 = Float32(0.0),
    estimator_alpha: Float32 = Float32(0.5),
) raises -> List[Float64]:
    """Their `Fit` (`doc_parallel_boosting.h:302`), one permutation.

    `objective` selects the loss and `alpha` is the one float its kernel
    takes; `leaf_estimation_method` and `leaf_estimation_iterations` come
    from `set_leaves_estimation_default`, which is where CatBoost decides
    them per loss (`catboost_options.cpp:273-360`). THE CALLER RESOLVES
    THEM -- this function does not re-derive a default, because a default
    derived in two places drifts in one of them.

    Every loss but one runs their `NeedEstimation` arm
    (`doc_parallel_boosting.h:371-385`): structure only from the searcher,
    then `TDocParallelLeavesEstimator` -- oracle over the bin-sorted rows,
    the walker, then the rescaled model applied to the cursor.

    DEVIATION 64: RMSE AT NEWTON-1 SKIPS THE ESTIMATOR. Their
    `NeedEstimation()` is `LeavesEstimationMethod != Simple`
    (`greedy_subsets_searcher.h:67-69`), which is TRUE for RMSE, so theirs
    runs the estimator and overwrites the searcher's leaf. This port does
    not, because for RMSE alone the two answers are the same number:
    the searcher writes `stats / (w + L2Reg)`
    (`greedy_search_helper.cpp:646-647`) and one Newton step from zero
    writes `Gradient / (Hessian + 1e-20)` with `Hessian = w + L2Reg`,
    since `TRmseTarget::Der2` returns `1.0f` and the kernel stores
    `weight * Der2` -- they agree to every representable digit. The skip
    is conditioned on BOTH the objective AND `Newton` at ONE iteration,
    so a caller who asks RMSE for Gradient leaves, or for more than one
    iteration, gets the estimator like everyone else.

    UNDER BOOTSTRAP THE ESTIMATOR SEES THE PLAIN WEIGHTS: their
    `AddEstimationTask` takes `learnTarget`, not the bootstrapped weak
    target (`doc_parallel_boosting.h:377-383`) -- sampling shapes the
    STRUCTURE only.

    `learning_rate` is their `Config.LearningRate`, whose default is **0.03**
    (`boosting_options.cpp:10`). It stood at 0.3 here, a tenfold larger step
    than stock CatBoost, which is a different model for every caller that did
    not pass one. See `CatBoostOptions.learning_rate` for the data-dependent
    retune (`options_helper.cpp:269-288`) that is deliberately not ported.

    Returns the loss per row after each iteration: their `functionValue`
    (`pointwise_targets.cu:271`, `:363` for the cross-entropy arm) with the
    sign flipped and divided by the row count, because everything
    downstream of theirs maximizes and a caller here wants a number that
    falls. It is computed IN THE SAME LAUNCH as the gradients, as theirs
    is, and folded from per-block partials in one fixed order; the host
    loop this replaced cost about 5 ms/tree at 800k rows.

    The loss is returned rather than printed because the thing worth
    asserting is that it DECREASES, and an assertion needs the numbers.
    """
    # `GetDim()` (`multiclass_targets.h:129-133`): `numClasses - 1` for
    # MultiClass, 1 for everything else. The cursor carries this many
    # planes and the stats carry one more -- their `statCount` is
    # `1 + point.GetColumnCount()` (`pointwise_target_impl.h:186`).
    var approx_dim = 1
    if objective == OBJECTIVE_MULTICLASS:
        if num_classes < 2:
            raise Error(
                "MultiClass needs num_classes >= 2, got "
                + String(num_classes)
            )
        approx_dim = num_classes - 1
    var stat_count = 1 + approx_dim

    # their `cursor`: the running prediction for every row. Zeros is THEIR
    # default branch, not a simplification of it: with no baseline column and
    # `boost_from_average` false (`boosting_options.cpp:17`),
    # `cursors->StartingPoint` is unset and `CreateCursors` writes
    # `TVector<float> start(sampleCount, 0.0)`
    # (`doc_parallel_boosting.h:180-186`). `boost_from_average` true would
    # seed it with `CalcOptimumConstApprox` and is refused by
    # `CatBoostOptions.check()`.
    var cursor = ctx.enqueue_create_buffer[DType.float32](
        approx_dim * n_rows
    )
    # every plane, not only the first
    ctx.enqueue_memset(cursor, Float32(0.0))

    # their der/der2 buffers, in the two-plane layout the histogram kernels
    # already read. See the DEVIATION BLOCK in `pointwise_targets.mojo`.
    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)

    # growth permutes this, so it is reseeded to the identity every
    # iteration exactly as their per-iteration dataset view is -- ON THE
    # DEVICE, their `MakeSequence` (`fill.cu:47`). This used to upload a
    # host-built identity, 3.2 MB of H2D per tree at 800k rows.
    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)

    # their `functionValue`: ONE float, accumulated by `pointwise_target_kernel`'s block
    # reduce + atomicAdd (`pointwise_targets.cu:309-317`). This replaces a
    # HOST loop over every row per iteration (~5 ms/tree at 800k), which
    # was never their design.
    var fv = ctx.enqueue_create_buffer[DType.float32](1)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](1)
    # per-block partials for `pointwise_target_kernel`'s reduces, folded in one fixed
    # order by `deterministic_sum_lanes_kernel`: the float-atomic combine
    # they use made same-seed fits differ in the last bits of
    # `fixed_scale` (bootstrap_check caught it, 2026-08-21)
    var mse_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    # MultiClass launches at `multilogit_blocks`, which is the same 256
    # threads x 1 element shape today; sized from the max so the partials
    # buffers stay right if either constant ever moves.
    var part_blocks = mse_blocks
    if multilogit_blocks(n_rows) > part_blocks:
        part_blocks = multilogit_blocks(n_rows)
    var fv_part = ctx.enqueue_create_buffer[DType.float32](part_blocks)
    var mag_part = ctx.enqueue_create_buffer[DType.float32](
        2 * part_blocks
    )

    # THE FIXED-POINT BOUND, for every build whose histogram quantizes: the
    # IDENTICAL flush, and the Apple column's `HIST_SMEM_SHARED2_I32`
    # shared-memory accumulation, which quantizes at `fixed_scale` even
    # under FAST (`hist_smem_mode_for` in `mojo_only/kernel_matrix.mojo`).
    # The scale must be bounded by `sum over all rows of abs(plane)`, and
    # the two plane magnitudes are computed ON THE DEVICE by `pointwise_target_kernel`'s
    # magnitude reduce -- the same block-reduce-plus-one-atomicAdd shape as
    # its `functionValue` -- into TWO scalars read back below. This replaces
    # a full-stats readback (6.4 MB at 800k rows) plus a `2 * n_rows` host
    # loop per tree. A build that quantizes nothing (NVIDIA/AMD FAST, whose
    # hist_2 arm keeps CatBoost's warp-private float design) skips the
    # launch flag, the copy and the sync at comptime.
    comptime _flush_fixed = deterministic_flush_for[
        TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
    ]()
    comptime _needs_magnitudes = _flush_fixed or HIST2_SMEM_IS_I32
    var mags = ctx.enqueue_create_buffer[DType.float32](2)
    var h_mags = ctx.enqueue_create_host_buffer[DType.float32](2)

    # their `TGpuAwareRandom::GetGpuSeeds`: the per-thread RNG state of
    # the whole fit, created once and advanced in place by every draw
    # (`gbdt/gpu_util/kernel/bootstrap.mojo`). One dummy word when
    # bootstrap is off.
    # `bootstrap_type` is the real switch; `bootstrap_bayesian` predates
    # it and stays as the shorthand every existing caller and check
    # passes. `-1` means "take the bool", so no caller changes meaning.
    var boot_kind = bootstrap_type
    var boot_param = bootstrap_param
    if boot_kind < 0:
        boot_kind = (
            BOOTSTRAP_KERNEL_BAYESIAN if bootstrap_bayesian else -1
        )
        boot_param = bagging_temperature
    var bootstrap_on = boot_kind >= 0

    var boot_seeds: DeviceBuffer[DType.uint64]
    if bootstrap_on:
        boot_seeds = create_bootstrap_seeds(ctx, random_seed)
    else:
        boot_seeds = ctx.enqueue_create_buffer[DType.uint64](1)
    var boot_mag_part = ctx.enqueue_create_buffer[DType.float32](
        2 * bootstrap_grid_blocks(n_rows)
    )

    # ONE device query for the estimator's reduces: `get_attribute` costs
    # 1.26 ms on this Metal device and their `SMCount()` is a cached static.
    # their `weak->NeedEstimation()` (`greedy_subsets_searcher.h:67-69`)
    # is `LeavesEstimationMethod != Simple`; DEVIATION 64 in the docstring
    # is the one subtraction from it, and it is conditioned on the
    # iteration count as well as on the objective.
    var need_estimation = leaf_estimation_method != LEAF_ESTIMATION_SIMPLE
    if (
        objective == OBJECTIVE_RMSE
        and leaf_estimation_method == LEAF_ESTIMATION_NEWTON
        and leaf_estimation_iterations == 1
    ):
        need_estimation = False
    var est_sm = -1
    if need_estimation:
        est_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var losses = List[Float64]()
    var not_pd_blocks = 0
    var not_pd_total = 0
    # their `TVector<TResultModel>* result`, the ensemble being built

    # THE POOL OF ONE, see `TTreeWorkspace`: the tree planes are the FIT's,

    # not the tree's, so growing tree 2 allocates nothing.

    var ws = List[TTreeWorkspace]()

    for _ in range(n_estimators):
        # `TTargetAtPointTrait::Create(learnTarget, cursor)` (`:353`).
        # The gradients are taken AT THE CURRENT PREDICTIONS, which is the
        # whole of what makes this boosting.
        # their `functionValue` is accumulated in the SAME launch as the
        # gradients, so the value this iteration reads is the loss of the
        # cursor as it stands -- i.e. after the PREVIOUS tree.
        var mags_in_mse = _needs_magnitudes and not bootstrap_on
        # under bootstrap the magnitudes must bound the BOOTSTRAPPED
        # planes (a Bayesian weight reaches ~46 at the tail), so the
        # bootstrap kernel computes them AFTER its multiply instead
        if objective == OBJECTIVE_MULTICLASS:
            # `TMultiClassificationTargets::StochasticDer`
            # (`multiclass_targets.cpp:22-45`): column 0 the weights,
            # columns 1.. the ders, one per FREE class.
            launch_multilogit_value_and_der_search(
                ctx, num_classes, n_rows, targets, weights, has_weights,
                cursor, n_rows, row_index, False,
                fv_part, True,
                stats, n_rows,
                mag_part, mags_in_mse,
            )
        else:
            launch_approximate[False](
                ctx, objective, targets, weights, Int32(n_rows), cursor,
                Int32(1) if has_weights else Int32(0),
                alpha, logloss_border,
                stats, fv_part, Int32(1),
                mag_part, Int32(1) if mags_in_mse else Int32(0),
                mse_blocks,
            )
        var fv_blocks = mse_blocks
        if objective == OBJECTIVE_MULTICLASS:
            fv_blocks = multilogit_blocks(n_rows)
        ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
            fv_part.unsafe_ptr(), Int32(fv_blocks), fv.unsafe_ptr(),
            grid_dim=1, block_dim=256,
        )
        if mags_in_mse:
            ctx.enqueue_function[deterministic_sum_lanes_kernel[2]](
                mag_part.unsafe_ptr(), Int32(fv_blocks),
                mags.unsafe_ptr(),
                grid_dim=1, block_dim=256,
            )
        if bootstrap_on:
            # their `BootstrapAndFilter`: one draw per row multiplies
            # BOTH planes. Bayesian never zeroes a weight, so their
            # filter branch cannot fire for it; Bernoulli and Poisson do
            # zero weights and this port still does not filter, which is
            # DEVIATION 69 in `bootstrap.mojo` -- same model, more rows
            # streamed than theirs.
            var compute_mags = False

            @parameter
            if _needs_magnitudes:
                compute_mags = True
            launch_bootstrap(
                ctx, boot_kind, boot_seeds, stats, n_rows, boot_param,
                boot_mag_part, compute_mags,
            )
            if compute_mags:
                ctx.enqueue_function[deterministic_sum_lanes_kernel[2]](
                    boot_mag_part.unsafe_ptr(),
                    Int32(bootstrap_grid_blocks(n_rows)),
                    mags.unsafe_ptr(),
                    grid_dim=1, block_dim=256,
                )
        # stream-ordered behind the kernel; READ after `run_tree_layout`,
        # whose first drain settles it, so it costs no synchronize here.
        ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=fv)

        # growth reorders rows in place, so the index restarts each tree.
        # their `MakeSequence` (`fill.cu:47`), device-side.
        launch_make_sequence(ctx, UInt32(0), row_index, n_rows)

        # `sum over all rows of abs(...)`, per plane, which is what
        # `choose_scale` is specified against. Plane 0 is the WEIGHT plane
        # and plane 1 is `der`, laid out `stats[s * n_rows + i]` by
        # `pointwise_target_kernel`, which is their `StatsToAggregate` column order
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
        # The `HIST_SMEM_SHARED2_I32` matrix row widened who depends on this:
        # the Apple column's hist_2 kernels quantize IN SHARED MEMORY at
        # `fixed_scale` even under `NUMERIC_FAST`, so an unbounded scale now
        # breaks the shipped Apple build, not just the IDENTICAL opt-in.
        # `_needs_magnitudes` above is that union.
        #
        # CONTRACT AUDIT, 2026-08-19, updated for the device reduction.
        # Every caller of `choose_scale` on the tree path against
        # `mojo_only/fixed_point.mojo`'s stated precondition, "sum over all
        # rows of abs(value) for the plane being accumulated":
        #
        #   this file                      sum of |stats| reduced ON DEVICE
        #                                  by `pointwise_target_kernel` from the exact
        #                                  planes it wrote, both planes, per
        #                                  iteration -- SOUND
        #   mojo_only/level_check.mojo     abs-accumulates a signed generator
        #                                  at :210-218 and :344-352 -- SOUND
        #   mojo_only/level_bench.mojo     same, three sites -- SOUND
        #   probe_main.check_fixed_point   sums |rows| directly -- SOUND
        #
        # The permutation growth applies to the stats plane does not move the
        # bound: a gather is a bijection on rows, so the sum of magnitudes is
        # invariant, and the sibling subtraction cannot exceed the parent's
        # bound either. The three reserved headroom bits cover the rest,
        # including the one place the bound is not exact: the device reduce
        # accumulates in Float32 through block sums and a float atomic, so
        # the total can round DOWN by a few parts in 1e6 relative -- a scale
        # at most that much too large. `SCALE_HEADROOM_BITS = 3` is a factor
        # of eight against it, five orders of magnitude of slack.
        #
        # If another lane restores the float atomic on the FAST arm, this
        # block does NOT become dead. `DETERMINISM_DEVICE` is the default and
        # it pins the integer flush on every backend, so the contract still
        # governs the shipped configuration; only the `OFF` arm escapes it.
        # ===========================================================
        var weight_magnitude = Float64(0.0)
        var gradient_magnitude = Float64(0.0)

        @parameter
        if _needs_magnitudes:
            # TWO floats, not 6.4 MB: the one drain the scale still costs.
            ctx.enqueue_copy(dst_ptr=h_mags.unsafe_ptr(), src_buf=mags)
            ctx.synchronize()
            weight_magnitude = Float64(h_mags.unsafe_ptr().unsafe_load(0))
            gradient_magnitude = Float64(h_mags.unsafe_ptr().unsafe_load(1))

        # `optimizer.Fit(...)` then `Estimate` then `Rescale` then
        # `AppendModels`, all four inside `run_tree_layout`, in their order.
        var splits = List[TBinarySplit]()
        var leaf_values = List[Float32]()
        var leaf_offsets = List[Int]()
        var sizes = run_tree_layout(
            ctx, n_rows, fold_counts, max_depth,
            cindex, stats, row_index, cursor,
            Float32(weight_magnitude), Float32(gradient_magnitude),
            splits, leaf_values, leaf_offsets, ws,
            need_estimation,
            use_subtraction, not need_estimation,
            learning_rate, l2_leaf_reg,
            one_hot=one_hot,
            score_function=score_function,
            approx_dim=approx_dim,
            multiclass_optimization=objective == OBJECTIVE_MULTICLASS,
        )

        if need_estimation:
            # their `weak->NeedEstimation()` arm (`doc_parallel_boosting.h:
            # 371-385`): the searcher's leaf values are DISCARDED, the
            # estimator recomputes them at the cursor, and only then does
            # the rescaled model touch the cursor. `row_index` left the
            # searcher bin-sorted, so target/weights/cursor gather straight
            # into the oracle's order (their factory sorts; we inherit).
            var n_leaves = len(sizes)
            var iters = leaf_estimation_iterations
            var g_target = ctx.enqueue_create_buffer[DType.float32](n_rows)
            var g_weights = ctx.enqueue_create_buffer[DType.float32](
                n_rows
            )
            var g_cursor = ctx.enqueue_create_buffer[DType.float32](
                approx_dim * n_rows
            )
            launch_gather_with_mask_f32(
                ctx, g_target, targets, row_index, n_rows,
                UInt32(0xFFFFFFFF),
            )
            if has_weights:
                launch_gather_with_mask_f32(
                    ctx, g_weights, weights, row_index, n_rows,
                    UInt32(0xFFFFFFFF),
                )
            # EVERY PLANE. The oracle owns a COPY of the cursor that it
            # shifts (`AddBinModelValues` inside `MoveTo`), so this cannot
            # be a gather-on-read view; for MultiClass that copy is
            # `numClasses - 1` planes wide.
            launch_gather_planes_with_mask_f32(
                ctx, g_cursor, cursor, row_index, n_rows,
                UInt32(0xFFFFFFFF), approx_dim, n_rows,
            )
            var d_p_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
            var d_p_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
            var h_po = ctx.enqueue_create_host_buffer[DType.uint32](
                n_leaves
            )
            var h_ps = ctx.enqueue_create_host_buffer[DType.uint32](
                n_leaves
            )
            ctx.synchronize()
            # THE DEVICE'S OWN OFFSETS: the final partitions sit in memory
            # in bit-reversed leaf order, so a prefix sum of the sizes is
            # the WRONG segmentation (see run_tree_layout's tail).
            var widest = 1
            for i in range(n_leaves):
                h_po.unsafe_ptr().unsafe_store(
                    i, UInt32(leaf_offsets[i])
                )
                h_ps.unsafe_ptr().unsafe_store(i, UInt32(sizes[i]))
                if sizes[i] > widest:
                    widest = sizes[i]
            ctx.enqueue_copy(dst_buf=d_p_off, src_ptr=h_po.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_p_sz, src_ptr=h_ps.unsafe_ptr())

            var oracle = make_bin_optimized_oracle(
                ctx, n_rows, n_leaves, sizes,
                g_target^, g_weights^, g_cursor^, d_p_off^, d_p_sz^,
                has_weights,
                objective,
                alpha,
                estimator_alpha,
                logloss_border,
                Float64(l2_leaf_reg),
                est_sm,
                leaf_estimation_method,
                num_classes,
            )
            # `TDocParallelLeavesEstimator::Estimate`
            # (`doc_parallel_leaves_estimator.cpp:9-16`): Exact REPLACES
            # the walker, it does not configure it.
            #
            #     if (LeavesEstimationMethod == Exact) {
            #         point = derCalcer->EstimateExact();
            #     } else {
            #         TNewtonLikeWalker walker(*derCalcer, iterations,
            #                                  backtrackingType);
            #         point = walker.Estimate(startPoint);
            #     }
            # DEVIATION 74's counter: how many leaves took their silent
            # gradient fallback because Cholesky found the Hessian not
            # positive definite. Accumulated across the whole fit and
            # reported once, because the number that matters is whether
            # it is ever nonzero.
            var estimated: List[Float32]
            if leaf_estimation_method == LEAF_ESTIMATION_EXACT:
                estimated = oracle.estimate_exact()
            else:
                estimated = newton_like_walker_estimate(
                    oracle, iters, BACKTRACKING_ANY_IMPROVEMENT,
                    List[Float32](), not_pd_blocks,
                )
            not_pd_total += not_pd_blocks
            leaf_values.clear()
            for i in range(len(estimated)):
                leaf_values.append(estimated[i])

            # `AppendModels` for this arm: the ESTIMATED leaves, rescaled,
            # onto the real cursor through the same kernel the RMSE arm
            # uses inside `run_tree_layout`.
            # `estimated` arrives in the CURSOR's gauge -- the walker
            # applied `MakeEstimationResult` on its way out
            # (`descent_helpers.cpp:153`, `:204`) -- so it is
            # `n_leaves * approx_dim` and is BIN-MAJOR, which is the layout
            # `add_model_value_kernel`'s z axis reads.
            var est_len = n_leaves * approx_dim
            var d_est = ctx.enqueue_create_buffer[DType.float32](est_len)
            var h_est = ctx.enqueue_create_host_buffer[DType.float32](
                est_len
            )
            ctx.synchronize()
            if len(estimated) != est_len:
                raise Error(
                    "the estimator returned " + String(len(estimated))
                    + " leaf values for " + String(n_leaves) + " leaves x "
                    + String(approx_dim) + " dims"
                )
            for i in range(est_len):
                h_est.unsafe_ptr().unsafe_store(i, estimated[i])
            ctx.enqueue_copy(dst_buf=d_est, src_ptr=h_est.unsafe_ptr())
            ctx.enqueue_function[add_model_value_kernel](
                oracle.d_p_off.unsafe_ptr(),
                oracle.d_p_sz.unsafe_ptr(),
                row_index.unsafe_ptr(),
                d_est.unsafe_ptr(),
                learning_rate,
                cursor.unsafe_ptr(),
                Int32(approx_dim), Int32(n_rows),
                grid_dim=((widest + 255) // 256, n_leaves, approx_dim),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()
        _ = len(sizes)

        # their `result[i].AddWeakModel(iterationModels[i])` (`:398`).
        # `Rescale(step)` is folded into `add_model_value_kernel`, so the
        # stored values are UNSCALED and the rate is applied on the way out.
        var structure = TObliviousTreeStructure()
        for i in range(len(splits)):
            structure.splits.append(splits[i])
        var weak = TObliviousTreeModel(structure^)
        # their `Dim` / `OutputDim()` (`oblivious_model.h:130-133`): the
        # number of approxes a leaf carries. `MakeEstimationResult` already
        # projected the walker's `numClasses`-wide point down to this, so
        # a MultiClass leaf holds `numClasses - 1` values and the model's
        # dimension matches the CURSOR's, not the walker's.
        weak.dim = approx_dim
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
    # cursor, through the SAME per-block-partials + fixed-order fold as the
    # loop's pass -- this launch still handed the 1-float `fv` where the
    # kernel now stores per-block partials, which under-read the loss by
    # ~n_blocks and wrote past the buffer (caught by the predict repro:
    # replay 36.52 vs a claimed 8.91 final loss, ratio ~= the block count).
    var final_blocks = mse_blocks
    if objective == OBJECTIVE_MULTICLASS:
        final_blocks = multilogit_blocks(n_rows)
        launch_multilogit_value_and_der_search(
            ctx, num_classes, n_rows, targets, weights, has_weights,
            cursor, n_rows, row_index, False,
            fv_part, True,
            stats, n_rows,
            mag_part, False,
        )
    else:
        launch_approximate[False](
            ctx, objective, targets, weights, Int32(n_rows), cursor,
            Int32(1) if has_weights else Int32(0),
            alpha, logloss_border,
            stats, fv_part, Int32(1),
            mag_part, Int32(0),
            mse_blocks,
        )
    ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
        fv_part.unsafe_ptr(), Int32(final_blocks), fv.unsafe_ptr(),
        grid_dim=1, block_dim=256,
    )
    ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=fv)
    ctx.synchronize()
    losses.append(
        -Float64(h_fv.unsafe_ptr().unsafe_load(0)) / Float64(n_rows)
    )
    # DEVIATION 74, MEASURED RATHER THAN ARGUED. A nonzero count means
    # some leaf's Hessian was not positive definite, Cholesky stopped, and
    # that leaf took a gradient step instead of a Newton one -- silently,
    # because their `CB_ENSURE(info >= 0)` passes on exactly that. Printed
    # rather than raised, because the behaviour is THEIRS and the point is
    # to know whether it happens at all.
    if not_pd_total != 0:
        print(
            "  [deviation 74] Cholesky found a non-positive-definite"
            " Hessian in", not_pd_total,
            "leaf-blocks over this fit; those leaves took their gradient"
            " fallback, as CatBoost's own dposv does",
        )

    return losses^


def model_approx_dim(model: TAdditiveModel) raises -> Int:
    """`OutputDim()` of the ensemble (`oblivious_model.h:130-133`).

    Every weak model in one ensemble has the same `Dim`; disagreement is a
    corrupted model rather than a case to handle, so it raises. An empty
    ensemble is one-dimensional, which is what a zero-tree predict returns.
    """
    if model.size() == 0:
        return 1
    var d = model.weak_models[0].dim
    for t in range(1, model.size()):
        if model.weak_models[t].dim != d:
            raise Error(
                "tree " + String(t) + " has dim "
                + String(model.weak_models[t].dim)
                + " but tree 0 has " + String(d)
            )
    if d < 1:
        raise Error("model dim is " + String(d))
    return d


def predict(
    model: TAdditiveModel,
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    mut cindex: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    one_hot: List[Bool] = List[Bool](),
) raises:
    """Apply a stored ensemble by EVALUATING every tree.

    Their `TAdditiveModel` prediction: sum over weak models. The learning
    rate is already in the stored leaf values (`Rescale(step)`), so nothing
    here reapplies it.

    This is the path that works on rows the model was never grown on, which
    the partition-based `add_model_value_kernel` cannot do. On the learn set
    the two must agree exactly, and `boosting_check` asserts that.

    THE ENSEMBLE IS PACKED ONCE. This used to create eight buffers, fill
    them and `ctx.synchronize()` PER TREE, which at ~191 us per drain plus
    the allocations made the host loop cost as much as the kernels: our
    artifact, not theirs -- their evaluator holds the whole model resident
    and their `AddObliviousTreeImpl` launches back to back on one stream.
    Now the per-level split records of every tree and every tree's leaf
    values go up in five copies total, the launches are enqueued back to
    back exactly as their stream takes them, and the ONE drain at the end
    is what makes `cursor` readable."""
    var layout = build_layout(fold_counts, one_hot)
    var approx_dim = model_approx_dim(model)

    ctx.enqueue_memset(cursor, Float32(0.0))

    # pack every tree's per-level records and leaf values, flat.
    # `total_leaves` counts VALUES, so it carries the approx dimension.
    var total_levels = 0
    var total_leaves = 0
    for t in range(model.size()):
        total_levels += model.weak_models[t].structure.get_depth()
        total_leaves += (
            (1 << model.weak_models[t].structure.get_depth()) * approx_dim
        )
    if total_levels == 0:
        ctx.synchronize()
        return

    var d_off = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_shift = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_mask = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_eq = ctx.enqueue_create_buffer[DType.uint8](total_levels)
    var d_vals = ctx.enqueue_create_buffer[DType.float32](total_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_eq = ctx.enqueue_create_host_buffer[DType.uint8](total_levels)
    var h_vals = ctx.enqueue_create_host_buffer[DType.float32](total_leaves)

    var lvl = 0
    var leaf = 0
    for t in range(model.size()):
        ref weak = model.weak_models[t]
        var depth = weak.structure.get_depth()
        for level in range(depth):
            ref cf = layout.features[
                Int(weak.structure.splits[level].feature_id)
            ]
            h_off.unsafe_ptr().unsafe_store(
                lvl, cf.offset * UInt32(n_rows)
            )
            h_shift.unsafe_ptr().unsafe_store(lvl, cf.shift)
            h_mask.unsafe_ptr().unsafe_store(lvl, cf.mask)
            h_bin.unsafe_ptr().unsafe_store(
                lvl, UInt32(Int(weak.structure.splits[level].bin_idx))
            )
            # THE PREDICATE COMES OFF THE MODEL, and the layout only
            # confirms it. Their `TAddModelDocParallel::AddTask` switches
            # on `split.SplitType` and asserts the dataset agrees --
            # `if (split.SplitType == EBinSplitType::TakeBin)
            #  CB_ENSURE(dataSet.IsOneHot(split.FeatureId)); else
            #  CB_ENSURE(!dataSet.IsOneHot(...))`
            # (`add_oblivious_tree_model_doc_parallel.cpp:139-144`). This
            # used to read the layout alone, which is fine while the layout
            # that grew the tree is still in hand and not fine for a model
            # read back from a file: the file, not the caller's fold
            # counts, has to say what the predicate is.
            var take_bin = (
                Int(weak.structure.splits[level].split_type)
                == BIN_SPLIT_TAKE_BIN
            )
            if take_bin != cf.one_hot_feature:
                raise Error(
                    "tree " + String(t) + " level " + String(level)
                    + " is a "
                    + String("TakeBin" if take_bin else "TakeGreater")
                    + " split on feature "
                    + String(Int(weak.structure.splits[level].feature_id))
                    + ", which the layout says is "
                    + String("one-hot" if cf.one_hot_feature else "ordered")
                )
            h_eq.unsafe_ptr().unsafe_store(
                lvl, UInt8(1) if take_bin else UInt8(0)
            )
            lvl += 1
        var n_values = (1 << depth) * approx_dim
        for i in range(n_values):
            var v = Float32(0.0)
            if i < len(weak.leaf_values):
                v = weak.leaf_values[i]
            h_vals.unsafe_ptr().unsafe_store(leaf + i, v)
        leaf += n_values
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_eq, src_ptr=h_eq.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())

    # `AddObliviousTreeImpl` per tree, back to back on one stream, their
    # own launch shape (`add_model_value.cu`); no drain between trees.
    var wide = (n_rows + 255) // 256
    if wide > 1024:
        wide = 1024
    lvl = 0
    leaf = 0
    for t in range(model.size()):
        ref weak = model.weak_models[t]
        var depth = weak.structure.get_depth()
        # a depth-0 tree still packed one leaf slot above, so the offsets
        # advance whether or not a kernel launches
        if depth > 0:
            ctx.enqueue_function[compute_bins_and_add_kernel](
                cindex.unsafe_ptr(),
                d_off.unsafe_ptr() + lvl,
                d_shift.unsafe_ptr() + lvl,
                d_mask.unsafe_ptr() + lvl,
                d_bin.unsafe_ptr() + lvl,
                d_eq.unsafe_ptr() + lvl,
                Int32(depth),
                d_vals.unsafe_ptr() + leaf,
                Int32(n_rows),
                cursor.unsafe_ptr(),
                Int32(approx_dim),
                Int32(n_rows),
                grid_dim=(wide, approx_dim, 1),
                block_dim=(256, 1, 1),
            )
        lvl += depth
        leaf += (1 << depth) * approx_dim
    ctx.synchronize()
