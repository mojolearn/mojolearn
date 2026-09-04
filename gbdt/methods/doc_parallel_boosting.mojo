# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

- **Permutations. A DIVERGENCE THE MOMENT THE FIT IS CATEGORICAL, and this
  entry used to deny it.** `Config.PermutationCount` defaults to 4
  (`boosting_options.cpp:14`), and `UpdateGpuSpecificDefaults` sets it back to
  1 only when `HasPermutationFeatures` is FALSE and boosting is Plain
  (`cuda/train_lib/train.cpp:100-108`, "No catFeatures for ctrs found and
  don't look ahead is disabled. Fallback to one permutation").
  `HasPermutationFeatures` is true as soon as a cat feature is used for a CTR
  (`:86-98`).

  **CTRs are ported now**, so on a categorical fit stock CatBoost keeps FOUR
  permutations where this keeps one. Four permutations means four learn
  cursors, four ensembles, leaf values estimated separately on each, the
  structure searched on a RANDOM non-estimation permutation (`:349-351`), and
  the exported model taken from permutation `count - 1` (`:527`). On a
  numeric-only fit the old sentence is still true and one cursor is still
  theirs.

  **PORTED 2026-08-21**, and the sentence that called it the largest
  remaining gap is deleted rather than annotated. `fit_with_test` takes
  `perm_cindexes` and `est_permutation`; it keeps one cursor per
  permutation, draws the structure permutation the way their `:349-351`
  draws it, estimates leaf values separately on every permutation
  (`:371-385`), and exports the estimation permutation's ensemble
  (`:526-528`). `pixi run check-permutation-count` gates it, and gate 5 is
  the one that isolates the LOOP from the data: count 3 and count 4 read at
  est 2 estimate identical leaves on identical rows and differ only in
  which permutation the structure was searched on.

  Two things of theirs are still absent and neither is read. Their
  `result` is a `TVector<TResultModel>`, one ENSEMBLE per permutation; this
  keeps only the estimation permutation's, because the others are written
  and never read -- their only consumers are snapshot restore and
  `GetL1LeavesSum()` for a bootstrap arm this port does not take. And their
  `AppendEnsembles` replay on restore has no counterpart, because there is
  no snapshotting.

  It is not ordered boosting -- see archive/reference/PORTING.md 88 for why that is a
  different learner entirely -- but it is the machinery ordered boosting is
  built on.
- **`CalcScoreModelLengthMult` (`:358`). PORTED, and it is not model-size
  regularization.** Its only consumer is `ComputeScoreStdDev`, which returns
  0 unconditionally when `modelLengthMult * randomStrength` is zero
  (`random_score_helper.h:25-32`). It reaches the greedy searcher as
  `options.RandomStrength *= randomStrengthMult`
  (`greedy_subsets_searcher.h:76`) and the doc-parallel one as
  `ModelLengthMultiplier`
  (`oblivious_tree_doc_parallel_structure_searcher.h:25,31`); both are wired
  in the loop below. At `random_strength = 0`, this port's default, the whole
  quantity is still an exact no-op and the reduce it would need is never
  launched. (`model_size_reg` is a different option entirely and lives on the
  feature weights.) This entry used to say the quantity was unported.
- **Test cursor and early stopping. PORTED 2026-08-21**, and this entry used
  to say otherwise. `fit_with_test` carries their `testCursor`,
  `TrackTestErrors` is `_test_loss` through the same target kernel the learn
  loss uses, `ShouldStop()` is the overfitting detector, and
  `ShrinkToBestIteration` runs in `gbdt/train.mojo` where their
  `train_template.h:127-137` runs it. **`BestTestCursor` is still absent**
  (`:220-222`, `:441-443`): it exists to EXPORT the test predictions taken at
  the best iteration, and nothing here exports test approxes at all.
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

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute
from gbdt.methods.kernel_add_model_value import add_model_value_kernel
from gbdt.metrics.optimal_const_for_loss import (
    calc_one_dimensional_optimum_const_approx,
)

from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.gpu_util.kernel.transform import (
    launch_gather_planes_with_mask_f32,
    launch_gather_with_mask_f32,
)
from core.device_liveness import (
    DEAD_DEVICE_POISON,
    SAB_2002_DEAD_DEVICE,
)
from core.identity_trace import IdentityTrace
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.methods.leaves_estimation.descent_helpers import (
    newton_like_walker_estimate,
)
from gbdt.methods.leaves_estimation.pointwise_oracle import (
    make_bin_optimized_oracle,
    merge_stage_times,
)
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
)
from checks.fixed_point import choose_scale
from gbdt.methods.random_score_helper import (
    calc_score_model_length_mult,
    compute_score_std_dev,
)
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import (
    PointwiseTreeWorkspace,
    fit_oblivious_tree_structure,
    fit_oblivious_tree_structure_traced,
    split_stat_planes,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_tree_layout,
    run_tree_layout_traced,
    TTreeWorkspace,
)
# ---- the NON-SYMMETRIC arm (DEVIATION 259): CatBoost's
# `TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>`, the searcher their
# `pointwise_non_symmetric.cpp:7-29` registers for every single-target
# pointwise loss under `EGrowPolicy::Depthwise` and `Lossguide`
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    TDepthwiseWorkspace,
    fit_non_symmetric_tree,
)
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.models.non_symmetric_tree import TNonSymmetricTree
from gbdt.models.add_non_symmetric_tree_doc_parallel import (
    add_non_symmetric_tree_to_cursor,
    compute_non_symmetric_bins_for_model,
)
from gbdt.options.catboost_options import (
    GROW_DEPTHWISE,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    grow_policy_name,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.models.kernel.add_bin_values import compute_bins_and_add_kernel
from checks.kernel_matrix import (
    TARGET_COLUMN,
    deterministic_flush_for,
    greedy_one_byte_fixed_for,
)
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE
from checks.numerics import PIN_DETERMINISM
from checks.numerics import NUMERIC_IDENTICAL
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
    HIST2_SMEM_MODE,
)
from gbdt.data.permutation import TRandom
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    compute_bins_for_model,
    partition_from_bins,
)
from gbdt.overfitting_detector.overfitting_detector import (
    OD_NONE,
    make_overfitting_detector,
)
from gbdt.options.overfitting_detector_options import (
    OD_DEFAULT_STOP_PVALUE,
    OD_DEFAULT_WAIT_ITERATIONS,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_EXACT,
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
    LEAF_ESTIMATION_SIMPLE,
    is_second_order_score_function,
)
from gbdt.targets.kernel.multilogit import (
    launch_multilogit_value_and_der_search,
    launch_one_vs_all_value_and_der,
    multilogit_blocks,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
    OBJECTIVE_RMSE,
    deterministic_sum_lanes_kernel,
    launch_approximate,
)


@fieldwise_init
struct TestArm(Movable):
    """Their `testCursor` and the detector that reads it.

    `AppendModels(dataSet, iterationModels, ..., learnCursors, testCursor)`
    (`doc_parallel_boosting.h:391-396`) applies each new weak model to the
    TEST cursor as well as the learn one, and their training loop feeds the
    resulting error to `DetectOverfitting`
    (`overfitting_detector.h:25-34`). This struct is that pair.

    `n_rows == 0` means NO TEST SET, and every buffer here is then unread
    -- the same contract `n_weights == 0` has for the weight column. A
    detector built without a test set is inert by construction
    (`overfitting_detector.cpp:122-124`), so the two agree.

    THE TEST ROWS MUST BE QUANTIZED AGAINST THE MODEL'S OWN BORDERS.
    `gbdt/train.mojo` does that, because it is the only place that owns
    them; handing this struct a `cindex` built against different borders
    would score every split against the wrong bins and nothing would
    assert.
    """

    var n_rows: Int
    var cindex: DeviceBuffer[DType.uint32]
    var targets: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var cursor: DeviceBuffer[DType.float32]
    var stats: DeviceBuffer[DType.float32]
    var fv_part: DeviceBuffer[DType.float32]
    var fv: DeviceBuffer[DType.float32]
    var h_fv: HostBuffer[DType.float32]
    var mag_dummy: DeviceBuffer[DType.float32]
    #: the per-tree apply's packed records, one tree at a time
    var d_off: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.uint32]
    var d_mask: DeviceBuffer[DType.uint32]
    var d_bin: DeviceBuffer[DType.uint32]
    var d_eq: DeviceBuffer[DType.uint8]
    var d_vals: DeviceBuffer[DType.float32]
    var h_off: HostBuffer[DType.uint32]
    var h_shift: HostBuffer[DType.uint32]
    var h_mask: HostBuffer[DType.uint32]
    var h_bin: HostBuffer[DType.uint32]
    var h_eq: HostBuffer[DType.uint8]
    var h_vals: HostBuffer[DType.float32]


def make_test_arm(
    ctx: DeviceContext,
    n_rows: Int,
    var cindex: DeviceBuffer[DType.uint32],
    var targets: DeviceBuffer[DType.float32],
    approx_dim: Int,
    stat_count: Int,
    max_depth: Int,
) raises -> TestArm:
    """Size the held-out arm. `n_rows == 0` still allocates one word per
    buffer, because a struct field cannot be absent and the contract is
    that nothing reads them."""
    var n = n_rows if n_rows > 0 else 1
    var blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    if multilogit_blocks(n) > blocks:
        blocks = multilogit_blocks(n)
    var max_leaves = 1 << max_depth
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_memset(w, Float32(1.0))
    return TestArm(
        n_rows,
        cindex^, targets^, w^,
        ctx.enqueue_create_buffer[DType.float32](approx_dim * n),
        ctx.enqueue_create_buffer[DType.float32](stat_count * n),
        ctx.enqueue_create_buffer[DType.float32](blocks),
        ctx.enqueue_create_buffer[DType.float32](1),
        ctx.enqueue_create_host_buffer[DType.float32](blocks),
        ctx.enqueue_create_buffer[DType.float32](2),
        ctx.enqueue_create_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_buffer[DType.uint8](max_depth),
        ctx.enqueue_create_buffer[DType.float32](max_leaves * approx_dim),
        ctx.enqueue_create_host_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_host_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_host_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_host_buffer[DType.uint32](max_depth),
        ctx.enqueue_create_host_buffer[DType.uint8](max_depth),
        ctx.enqueue_create_host_buffer[DType.float32](
            max_leaves * approx_dim
        ),
    )


def _apply_last_tree_to_test(
    ctx: DeviceContext,
    model: TAdditiveModel,
    layout: CompressedIndexLayout,
    mut test: TestArm,
    approx_dim: Int,
    learning_rate: Float32,
) raises:
    """One tree onto the held-out cursor, their `AddObliviousTree`.

    The LAST weak model only. `predict` packs the whole ensemble because
    it is called once; this runs per iteration, so it packs one tree --
    `depth` level records and `2^depth * dim` values -- and launches once.

    THE LEAF VALUES ALREADY CARRY THE LEARNING RATE. `fit` stores
    `leaf_values[i] * learning_rate` on the weak model (their
    `Rescale(step)` folded in), so this passes 1.0 and must NOT reapply
    it. Reapplying would make the held-out curve fall at a different rate
    from the learn curve, and the detector would stop on the wrong shape.
    """
    var t = model.size() - 1
    if t < 0:
        return
    var n = test.n_rows
    if not model.is_oblivious():
        # their `AppendModels` through `TAddModelDocParallel<
        # TNonSymmetricTree>` (`add_non_symmetric_tree_doc_parallel.cpp:
        # 182-206`): bins off the model, then `AddBinModelValues`. The
        # stored values already carry the rate, as above.
        add_non_symmetric_tree_to_cursor(
            ctx, layout, model.non_symmetric_models[t], test.cindex, n,
            test.cursor,
        )
        return
    ref weak = model.weak_models[t]
    var depth = weak.structure.get_depth()
    if depth == 0:
        return

    for level in range(depth):
        ref cf = layout.features[
            Int(weak.structure.splits[level].feature_id)
        ]
        test.h_off.unsafe_ptr().unsafe_store(
            level, cf.offset * UInt32(n)
        )
        test.h_shift.unsafe_ptr().unsafe_store(level, cf.shift)
        test.h_mask.unsafe_ptr().unsafe_store(level, cf.mask)
        test.h_bin.unsafe_ptr().unsafe_store(
            level, UInt32(Int(weak.structure.splits[level].bin_idx))
        )
        var take_bin = (
            Int(weak.structure.splits[level].split_type)
            == BIN_SPLIT_TAKE_BIN
        )
        test.h_eq.unsafe_ptr().unsafe_store(
            level, UInt8(1) if take_bin else UInt8(0)
        )
    var n_values = (1 << depth) * approx_dim
    for i in range(n_values):
        var v = Float32(0.0)
        if i < len(weak.leaf_values):
            v = weak.leaf_values[i]
        test.h_vals.unsafe_ptr().unsafe_store(i, v)

    ctx.enqueue_copy(dst_buf=test.d_off, src_ptr=test.h_off.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=test.d_shift, src_ptr=test.h_shift.unsafe_ptr()
    )
    ctx.enqueue_copy(dst_buf=test.d_mask, src_ptr=test.h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=test.d_bin, src_ptr=test.h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=test.d_eq, src_ptr=test.h_eq.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=test.d_vals, src_ptr=test.h_vals.unsafe_ptr())

    var wide = (n + 255) // 256
    if wide > 1024:
        wide = 1024
    ctx.enqueue_function[compute_bins_and_add_kernel](
        test.cindex.unsafe_ptr(),
        test.d_off.unsafe_ptr(),
        test.d_shift.unsafe_ptr(),
        test.d_mask.unsafe_ptr(),
        test.d_bin.unsafe_ptr(),
        test.d_eq.unsafe_ptr(),
        Int32(depth),
        test.d_vals.unsafe_ptr(),
        Int32(n),
        test.cursor.unsafe_ptr(),
        Int32(approx_dim),
        Int32(n),
        # 1.0: the rate is already inside the stored leaf values
        grid_dim=(wide, approx_dim, 1),
        block_dim=(256, 1, 1),
    )


def _test_loss(
    ctx: DeviceContext,
    mut test: TestArm,
    objective: Int,
    num_classes: Int,
    alpha: Float32,
    logloss_border: Float32,
    approx_dim: Int,
) raises -> Float64:
    """The held-out loss, through the SAME target kernel the learn loss
    uses, so the two curves are the same quantity.

    Their `functionValue` with the sign flipped and divided by the row
    count, exactly as the learn loss is -- which is what makes the
    detector's comparison meaningful.
    """
    var n = test.n_rows
    var blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    if objective == OBJECTIVE_MULTICLASS:
        blocks = multilogit_blocks(n)
        launch_multilogit_value_and_der_search(
            ctx, num_classes, n, test.targets, test.weights, False,
            test.cursor, n, test.cindex, False,
            test.fv_part, True, test.stats, n, test.mag_dummy, False,
        )
    elif objective == OBJECTIVE_MULTICLASS_OVA:
        blocks = multilogit_blocks(n)
        launch_one_vs_all_value_and_der[True](
            ctx, num_classes, n, test.targets, test.weights, False,
            test.cursor, n, test.cindex, False,
            test.fv_part, True, test.stats, n, test.mag_dummy, False,
        )
    else:
        launch_approximate[False](
            ctx, objective, test.targets, test.weights, Int32(n),
            test.cursor, Int32(0), alpha, logloss_border,
            test.stats, test.fv_part, Int32(1),
            test.mag_dummy, Int32(0), blocks,
        )
    ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
        test.fv_part.unsafe_ptr(), Int32(blocks), test.fv.unsafe_ptr(),
        grid_dim=1, block_dim=256,
    )
    ctx.enqueue_copy(dst_ptr=test.h_fv.unsafe_ptr(), src_buf=test.fv)
    ctx.synchronize()
    return -Float64(test.h_fv.unsafe_ptr().unsafe_load(0)) / Float64(n)


@fieldwise_init
struct FitResult(Movable):
    """What a fit reports beyond the model itself.

    `fit` returns only `learn_losses`, which is what its nine existing
    call sites expect; `fit_with_test` returns this.
    """

    var learn_losses: List[Float64]
    #: empty when there is no held-out set
    var test_losses: List[Float64]
    #: the index of the lowest TEST loss, or of the lowest learn loss when
    #: there is no test set. Their `TLearnProgress`'s best-iteration
    #: bookkeeping, kept in the detector here because it is the only
    #: object that already knows when a new best arrived.
    var best_iteration: Int
    #: True when the detector fired before `n_estimators` was reached
    var stopped_early: Bool



def _tree_tag(iteration: Int) -> String:
    """`treeNNN` prefix for identity-trace tags, zero-padded to sort in
    fit order."""
    var s = String(iteration)
    while s.byte_length() < 3:
        s = String("0") + s
    return String("tree") + s


struct TEstimationWorkspace(Movable):
    """The leaf-estimation stage's per-shape buffers, owned by the FIT.

    ================= DEVIATION 1890 =================
    CatBoost does not allocate the estimator's gathers per tree and
    neither should this: their `TCudaManager` hands every estimator
    buffer out of the per-device memory pool (`cuda_lib/memory_pool.h`),
    so tree 2's `TDocParallelLeavesEstimator` reuses tree 1's device
    memory. This port called `enqueue_create_buffer` inside
    `_estimate_and_apply` -- THREE fresh `n_rows`-sized buffers (target,
    weights, `approx_dim * n_rows` of cursor), the two partition-offset
    buffers plus their host staging, and the estimate's device/host pair,
    for EVERY estimation task, and Logloss runs one task per permutation
    per tree. Same repair as `TTreeWorkspace`'s (its DEVIATION BLOCK is
    the precedent, POOL OF ONE and all): the fit holds the list across
    trees, keyed by the dataset shape, and a shape change rebuilds it.

    `n_leaves_cap` is a CAPACITY, not an exact key: a non-symmetric tree
    that stopped early has fewer leaves, and every consumer walks exactly
    `n_leaves` entries (grid y, `bin_count` loops), so a wider buffer is
    bit-inert. Growing reallocates; shrinking reuses.

    THE LIFETIME RULE RIDES ALONG [[mojo-buffer-freed-at-last-use]]: the
    workspace OWNS these buffers across trees, so nothing in
    `_estimate_and_apply` dies under a queued copy any more -- the
    staging halves (`h_po`/`h_ps`/`h_est`) are rewritten by the NEXT call
    only after this call's tail drain, which is what makes the reuse
    race-free (see DEVIATION 1891 at that drain).
    ==================================================
    """

    var n_rows_key: Int
    var approx_dim_key: Int
    var n_leaves_cap: Int
    var g_target: DeviceBuffer[DType.float32]
    var g_weights: DeviceBuffer[DType.float32]
    var g_cursor: DeviceBuffer[DType.float32]
    var d_p_off: DeviceBuffer[DType.uint32]
    var d_p_sz: DeviceBuffer[DType.uint32]
    var h_po: HostBuffer[DType.uint32]
    var h_ps: HostBuffer[DType.uint32]
    var d_est: DeviceBuffer[DType.float32]
    var h_est: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        approx_dim: Int,
        n_leaves: Int,
    ) raises:
        self.n_rows_key = n_rows
        self.approx_dim_key = approx_dim
        self.n_leaves_cap = n_leaves
        self.g_target = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.g_weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.g_cursor = ctx.enqueue_create_buffer[DType.float32](
            approx_dim * n_rows
        )
        self.d_p_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
        self.d_p_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
        self.h_po = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
        self.h_ps = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
        self.d_est = ctx.enqueue_create_buffer[DType.float32](
            n_leaves * approx_dim
        )
        self.h_est = ctx.enqueue_create_host_buffer[DType.float32](
            n_leaves * approx_dim
        )


def _estimate_and_apply(
    ctx: DeviceContext,
    n_rows: Int,
    approx_dim: Int,
    n_leaves: Int,
    sizes: List[Int],
    leaf_offsets: List[Int],
    mut row_index: DeviceBuffer[DType.uint32],
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut cursor: DeviceBuffer[DType.float32],
    objective: Int,
    alpha: Float32,
    estimator_alpha: Float32,
    logloss_border: Float32,
    l2_leaf_reg: Float32,
    est_sm: Int,
    leaf_estimation_method: Int,
    num_classes: Int,
    iters: Int,
    learning_rate: Float32,
    mut leaf_values: List[Float32],
    mut not_pd_total: Int,
    mut trace: IdentityTrace,
    mut stage_times: StageTimes,
    leaf_tag: String,
    # DEVIATION 1890: the fit-owned estimation workspace (pool of one)
    mut est_ws: List[TEstimationWorkspace],
) raises:
    """One estimation task: their `TDocParallelLeavesEstimator::Estimate`
    plus the `AppendModels` that follows it, for ONE (dataset, cursor).

    `row_index`, `leaf_offsets` and `sizes` are the partition -- rows
    grouped by leaf. Which partition depends on which permutation this task
    is for: the one the tree was grown on inherits the searcher's, and
    every other one gets `partition_from_bins`. Nothing else in here
    differs between permutations, which is why this is one function called
    `permutation_count` times rather than two paths.

    `leaf_values` is OVERWRITTEN with the estimate, and the caller keeps
    only the estimation permutation's -- that is the ensemble their `Run()`
    exports (`doc_parallel_boosting.h:526-528`).
    """
    var not_pd_blocks = 0
    # their `weak->NeedEstimation()` arm (`doc_parallel_boosting.h:
    # 371-385`): the searcher's leaf values are DISCARDED, the
    # estimator recomputes them at the cursor, and only then does
    # the rescaled model touch the cursor. `row_index` left the
    # searcher bin-sorted, so target/weights/cursor gather straight
    # into the oracle's order (their factory sorts; we inherit).
    #
    # DEVIATION 1890: the buffers come from the fit's pool of one, not
    # from per-tree `enqueue_create_buffer` (see `TEstimationWorkspace`'s
    # block). Reuse is safe because every consumer of the previous task's
    # contents drained at that task's tail (DEVIATION 1891), and the
    # gathers below overwrite every cell this task reads.
    if (
        len(est_ws) == 0
        or est_ws[0].n_rows_key != n_rows
        or est_ws[0].approx_dim_key != approx_dim
        or est_ws[0].n_leaves_cap < n_leaves
    ):
        est_ws.clear()
        est_ws.append(
            TEstimationWorkspace(ctx, n_rows, approx_dim, n_leaves)
        )
    ref g_target = est_ws[0].g_target
    ref g_weights = est_ws[0].g_weights
    ref g_cursor = est_ws[0].g_cursor
    ref d_p_off = est_ws[0].d_p_off
    ref d_p_sz = est_ws[0].d_p_sz
    ref h_po = est_ws[0].h_po
    ref h_ps = est_ws[0].h_ps
    ref d_est = est_ws[0].d_est
    ref h_est = est_ws[0].h_est
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
    # ================= DEVIATION 1891 =================
    # A `ctx.synchronize()` stood here and it settled NOTHING a later op
    # needs: every enqueue above it (the pool's creates on a rebuild, the
    # three gathers) is stream-ordered before every device op below, and
    # writing a host buffer straight after `enqueue_create_host_buffer`
    # is this repo's settled practice (`TTreeWorkspace.__init__`'s
    # constant fills, `make_bin_optimized_oracle`'s `h_leaves`). The host
    # writes below reuse `h_po`/`h_ps` across tasks, and THAT hazard is
    # held to the tail drain: the previous task's `enqueue_copy` of these
    # buffers became a RUN at its own tail `ctx.synchronize()`, which
    # returned before this call began. Removed 2026-08-26; one full
    # device drain per estimation task, and Logloss at 10 walker
    # iterations ran ~34 full-n_rows passes between drains 1 and 2, so
    # the drain was pure serialization. Bit-inert: a drain reorders
    # nothing on one stream.
    # ==================================================
    # THE DEVICE'S OWN OFFSETS: the final partitions sit in memory
    # in bit-reversed leaf order, so a prefix sum of the sizes is
    # the WRONG segmentation (see run_tree_layout's tail).
    for i in range(n_leaves):
        h_po.unsafe_ptr().unsafe_store(
            i, UInt32(leaf_offsets[i])
        )
        h_ps.unsafe_ptr().unsafe_store(i, UInt32(sizes[i]))
    ctx.enqueue_copy(dst_buf=d_p_off, src_ptr=h_po.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p_sz, src_ptr=h_ps.unsafe_ptr())
    # THE BUFFERS MUST OUTLIVE THE COPY -- and now they structurally do.
    # When `h_po`/`h_ps` were locals, their last use was the two
    # `enqueue_copy`s above, so Mojo could free them there -- and the very
    # next host allocation was `h_leaves` inside
    # `make_bin_optimized_oracle`, SAME POOL, SAME DTYPE, SAME LENGTH,
    # filled `[i] = i` immediately. That was DEVIATION 134's pointwise
    # half (measured: 8 of 12 splits reproduced, first divergence at tree
    # 8), [[mojo-buffer-freed-at-last-use]], same defect as DEVIATION
    # 125a, and it was held to a mid-function sync. DEVIATION 1890 closes
    # it at the OWNERSHIP level instead: the workspace owns the staging
    # across trees, so no copy in this function ever outlives its source.

    # The oracle receives HANDLE COPIES (`DeviceBuffer.copy()` copies the
    # handle, not the bytes -- see the `cursors` note in `fit_with_test`),
    # so the workspace keeps the memory alive across trees while the
    # oracle's handles die with it at this task's tail drain.
    var oracle = make_bin_optimized_oracle(
        ctx, n_rows, n_leaves, sizes,
        g_target.copy(), g_weights.copy(), g_cursor.copy(),
        d_p_off.copy(), d_p_sz.copy(),
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
        trace.record_list_f32(leaf_tag, estimated)
    else:
        estimated = newton_like_walker_estimate(
            oracle, iters, BACKTRACKING_ANY_IMPROVEMENT,
            List[Float32](), not_pd_blocks,
            stage_times, trace, leaf_tag,
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
    # DEVIATION 1891: a second full drain stood here, whose only real work
    # was the lifetime hold for `h_po`/`h_ps` ("the far end of the hold").
    # The workspace owns them now (DEVIATION 1890), so the hold has no
    # buffers to protect and the drain folded into the tail one below.
    # `h_est` is safe to write without a drain for the same reason
    # `h_po`/`h_ps` are: the previous task's copy of it ran before that
    # task's tail drain returned.
    if len(estimated) != est_len:
        raise Error(
            "the estimator returned " + String(len(estimated))
            + " leaf values for " + String(n_leaves) + " leaves x "
            + String(approx_dim) + " dims"
        )
    for i in range(est_len):
        h_est.unsafe_ptr().unsafe_store(i, estimated[i])
    ctx.enqueue_copy(dst_buf=d_est, src_ptr=h_est.unsafe_ptr())
    # MACHINE-SIZED x (the kernel strides): the widest-leaf grid priced
    # every leaf at the largest leaf's block count -- on a skewed depth-8
    # tree, tens of millions of empty threads per launch. Same repair as
    # DEVIATION 210b in `kernel_add_model_value.mojo`; the launch runs
    # once per tree, so the fix here is the grid, not the kernel.
    var amv_gx = 2 * oracle.sm_count
    if amv_gx < 1:
        amv_gx = 1
    ctx.enqueue_function[add_model_value_kernel](
        oracle.d_p_off.unsafe_ptr(),
        oracle.d_p_sz.unsafe_ptr(),
        row_index.unsafe_ptr(),
        d_est.unsafe_ptr(),
        learning_rate,
        cursor.unsafe_ptr(),
        Int32(approx_dim), Int32(n_rows),
        grid_dim=(amv_gx, n_leaves, approx_dim),
        block_dim=(256, 1, 1),
    )
    # THE ONE SETTLE POINT of the task (DEVIATION 1891). This drain is
    # KEPT, and it now carries every hold the removed drains carried:
    # (a) the workspace staging (`h_po`/`h_ps`/`h_est`) may be host-
    #     rewritten by the NEXT task only because their copies became
    #     runs here;
    # (b) the oracle dies PAST it, pinned by the explicit discard below.
    #     Its last implicit use is the launch above, and dying there
    #     would free its exclusive buffers (`d_bins`, `d_shift`, the
    #     Exact path's trailing `move_to` operands) under queued work --
    #     the step-33 race class, device side.
    ctx.synchronize()
    _ = oracle^  # past the drain (step-33 race class, device side)


def fit_with_test(
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
    mut trace: IdentityTrace,
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
    # `boost_from_average`, PORTED 2026-08-22 and RESOLVED BY THE CALLER:
    # their layering resolves the data-dependent default in
    # `options_helper.cpp::AdjustBoostFromAverageDefaultValue` before the
    # trainer runs, and this port keeps that shape -- `train`/the python
    # surface resolve, this function only obeys. False is every existing
    # caller and every oracle fixture (the generators pin it off).
    boost_from_average: Bool = False,
    # THE HELD-OUT ARM IS OPTIONAL AND EVERY EXISTING CALLER IS
    # UNCHANGED. `fit` has nine call sites across checks, benches and
    # `train`, one of which belongs to another session, so these are
    # additive with defaults rather than a new required argument.
    var test: Optional[TestArm] = None,
    # THE OTHER PERMUTATIONS' COMPRESSED INDICES, in permutation order and
    # INCLUDING the estimation one, which must be `cindex` itself. Empty
    # means one permutation, which is what every caller but `train` passes
    # and what CatBoost itself uses without a CTR-bearing categorical
    # feature (`cuda/train_lib/train.cpp:99-108`).
    var perm_cindexes: List[DeviceBuffer[DType.uint32]] = List[
        DeviceBuffer[DType.uint32]
    ](),
    est_permutation: Int = -1,
    # `TObliviousTreeLearnerOptions::RandomStrength`
    # (`oblivious_tree_options.cpp:17`). **THEIR DEFAULT IS 1.0 AND THIS
    # ONE IS 0.0**; see `CatBoostOptions.random_strength` for why the
    # default did not move with the port.
    #
    # It reaches BOTH searchers and does something on only one of them:
    # the doc-parallel arm's noise survives into the gain
    # (`kernel/pointwise_scores.cu:396-402`), the greedy arm's cancels
    # against its own before-calcer (`compute_scores.cu:84-134`). That is
    # CatBoost's behaviour, not a gap here.
    random_strength: Float32 = Float32(0.0),
    # WHICH STRUCTURE SEARCHER GROWS THE TREE. False is
    # `TGreedySubsetsSearcher`, which is what this repository has always
    # run and what CatBoost runs for MULTICLASS symmetric trees. True is
    # `TDocParallelObliviousTreeSearcher`, which is what CatBoost runs for
    # SINGLE-TARGET symmetric trees at `boosting_type=Plain`
    # (`archive/reference/PORTING.md` 91 F) -- the arm every matched benchmark pins CatBoost
    # to.
    #
    # Additive with a default, like `test` above and for the same reason:
    # `fit` has nine call sites and one belongs to another session.
    #
    # DEFAULT IS FALSE, and that is not timidity. The two searchers pick
    # IDENTICAL splits (`pixi run check-pointwise-vs-greedy`), so this is
    # not a correctness switch. The arm's old per-tree overheads are
    # falling -- the host round trip in `split_stat_planes` became a
    # device launch, DEVIATION 143 pools its per-tree construction --
    # but the arm still trails the greedy searcher at the measured
    # shapes, and flipping the default is a MEASUREMENT's job in a
    # settled window: bit-identity plus a speed win, or it stays False.
    use_pointwise_searcher: Bool = False,
    od_type: Int = OD_NONE,
    od_pvalue: Float64 = OD_DEFAULT_STOP_PVALUE,
    od_wait: Int = OD_DEFAULT_WAIT_ITERATIONS,
    # ============================ DEVIATION 259 ============================
    # `grow_policy`, `max_leaves`, `min_data_in_leaf` -- their
    # `TObliviousTreeLearnerOptions::{GrowPolicy, MaxLeaves, MinDataInLeaf}`
    # (`oblivious_tree_options.cpp:23-25`). Additive with defaults, like
    # every argument after `trace`, so the nine `fit` call sites and every
    # oracle fixture run the code they ran before, to the bit.
    #
    # SymmetricTree (the default) is the loop this file has always been.
    # Depthwise and Lossguide grow `TNonSymmetricTree` through
    # `fit_non_symmetric_tree` -- ONE searcher for both, as theirs is one
    # `TGreedySearchHelper` with `switch (Options.Policy)` -- and then take
    # their `NeedEstimation` arm UNCONDITIONALLY: `ComputeBins` off the
    # model (`doc_parallel_leaves_estimator.cpp:48`), the estimator, and
    # `AddBinModelValues` for the apply. DEVIATION 64's RMSE shortcut is
    # the GREEDY OBLIVIOUS arm's (its leaf is applied inside
    # `run_tree_layout`); this arm has no such in-searcher apply and runs
    # the estimator exactly as the pointwise arm does, so for RMSE the
    # numbers are the same Newton step from zero by DEVIATION 64's own
    # derivation.
    #
    # THE MODEL IS A `TNonSymmetricTree` PER WEAK MODEL. Their trainer
    # returns `TAdditiveModel<TNonSymmetricTree>` for these policies
    # (`train.cpp:436-455`); here the same `model` carries them in
    # `non_symmetric_models`, and `predict` / the held-out apply / the
    # model text dispatch on the shape.
    #
    # `max_leaves` is READ ONLY UNDER LOSSGUIDE (`catboost_options.cpp:
    # 993-1001` pins every other policy to `1 << depth`, which is what the
    # -1 resolves to); `min_data_in_leaf` is `MinLeafSize`, live under the
    # non-symmetric policies only (`greedy_search_helper.cpp:685` guards
    # the size test with `Policy != SymmetricTree`). The CALLER refuses
    # the pairs CatBoost refuses; this function takes resolved numbers.
    # =======================================================================
    grow_policy: Int = GROW_SYMMETRIC,
    max_leaves: Int = -1,
    min_data_in_leaf: Int = 1,
) raises -> FitResult:
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
    elif objective == OBJECTIVE_MULTICLASS_OVA:
        # `GetDim()` is `NumClasses` -- no pinned class
        if num_classes < 2:
            raise Error(
                "MultiClassOneVsAll needs num_classes >= 2, got "
                + String(num_classes)
            )
        approx_dim = num_classes
    var stat_count = 1 + approx_dim

    # ---- the non-symmetric policies' own refusals (DEVIATION 259) ----
    var non_symmetric = grow_policy != GROW_SYMMETRIC
    if non_symmetric:
        if grow_policy != GROW_DEPTHWISE and grow_policy != GROW_LOSSGUIDE:
            raise Error(
                "grow_policy " + String(grow_policy) + " is not"
                " SymmetricTree, Depthwise or Lossguide; EGrowPolicy::Region"
                " is unported (structure_searcher_options.check)"
            )
        if use_pointwise_searcher:
            # `TDocParallelObliviousTreeSearcher` grows OBLIVIOUS trees and
            # nothing else: their non-symmetric trainers are
            # `TGpuTrainer<TPointwiseTargetsImpl, TNonSymmetricTree>`
            # through `train_template_pointwise_greedy_subsets_searcher.h`
            # (`pointwise_non_symmetric.cpp:5`), never the doc-parallel
            # oblivious searcher. Refused by name rather than silently
            # growing a symmetric tree under a non-symmetric name.
            raise Error(
                "use_pointwise_searcher is TDocParallelObliviousTreeSearcher,"
                " an OBLIVIOUS searcher; grow_policy="
                + grow_policy_name(grow_policy)
                + " is grown by TGreedySubsetsSearcher<TNonSymmetricTree>"
                " only (pointwise_non_symmetric.cpp:5-29)"
            )
        if objective == OBJECTIVE_MULTICLASS or objective == OBJECTIVE_MULTICLASS_OVA:
            # their GPU trainer registry has NO (MultiClass, Depthwise) or
            # (MultiClass, Lossguide) entry -- `multiclass.cpp:5-14`
            # registers the multiclass targets at the default grow policy
            # only -- so `TGpuTrainerFactory::Has` fails with "optimization
            # scheme is not supported for GPU learning" (`train.cpp:279`).
            raise Error(
                "Error: optimization scheme is not supported for GPU learning"
                " Loss=MultiClass;OptimizationScheme="
                + grow_policy_name(grow_policy)
                + " (their TGpuTrainerFactory has no non-symmetric"
                " multiclass trainer, multiclass.cpp:5-14, train.cpp:279)"
            )
    # `catboost_options.cpp:993-1001`: every policy but Lossguide pins
    # MaxLeaves to `1 << MaxDepth`; -1 here is their IsDefault() arm
    var ns_max_leaves = max_leaves
    if grow_policy != GROW_LOSSGUIDE:
        if max_leaves >= 0 and max_leaves != (1 << max_depth):
            raise Error(
                "max_leaves option works only with lossguide tree growing"
                " (catboost_options.cpp:998); for "
                + grow_policy_name(grow_policy)
                + " it is 1 << depth == " + String(1 << max_depth)
                + ", got " + String(max_leaves)
            )
        ns_max_leaves = 1 << max_depth
    elif ns_max_leaves < 0:
        # their constructed default, `MaxLeaves("max_leaves", 31)`
        # (`oblivious_tree_options.cpp:24`), which only Lossguide keeps
        ns_max_leaves = 31
    if non_symmetric and ns_max_leaves < 2:
        raise Error(
            "max_leaves must be at least 2 under "
            + grow_policy_name(grow_policy) + ", got "
            + String(ns_max_leaves)
        )
    if non_symmetric and min_data_in_leaf < 0:
        raise Error(
            "min_data_in_leaf must be non-negative, got "
            + String(min_data_in_leaf)
        )

    # their `cursor`: the running prediction for every row. With
    # `boost_from_average` false, `cursors->StartingPoint` is unset and
    # `CreateCursors` writes `TVector<float> start(sampleCount, 0.0)`
    # (`doc_parallel_boosting.h:180-186`); with it TRUE (PORTED
    # 2026-08-22), `StartingPoint = NCB::CalcOptimumConstApprox(loss,
    # target, weights)` (`:174-182`) and every cursor plane is written
    # with `const float value = (*cursors->StartingPoint)[dim]` -- the
    # double narrowed to float32 exactly once, here as there. The same
    # value goes on the model as its bias
    # (`modelToExport.SetBias(cursors->StartingPoint)`, `:434`).
    var starting_approx = Float64(0.0)
    if boost_from_average:
        if approx_dim != 1:
            raise Error(
                "boost_from_average is one-dimensional here; their ENSURE"
                " list has no MultiClass either"
            )
        var h_t = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_ptr=h_t.unsafe_ptr(), src_buf=targets)
        var h_w = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        if has_weights:
            ctx.enqueue_copy(dst_ptr=h_w.unsafe_ptr(), src_buf=weights)
        ctx.synchronize()
        var t_host = List[Float32](capacity=n_rows)
        var w_host = List[Float32](capacity=n_rows)
        for i in range(n_rows):
            t_host.append(h_t.unsafe_ptr().unsafe_load(i))
        if has_weights:
            for i in range(n_rows):
                w_host.append(h_w.unsafe_ptr().unsafe_load(i))
        starting_approx = calc_one_dimensional_optimum_const_approx(
            objective, t_host, w_host, has_weights
        )
        # their `modelToExport.SetBias` (`:434`), set here so an early
        # stop or a raise mid-fit cannot produce a seeded-cursor model
        # that forgot to say so.
        model.bias = starting_approx
        _ = h_t^
        _ = h_w^
    var start_value = Float32(starting_approx)
    var cursor = ctx.enqueue_create_buffer[DType.float32](
        approx_dim * n_rows
    )
    # every plane, not only the first
    ctx.enqueue_memset(cursor, start_value)

    # ---- the permutations -------------------------------------------
    #
    # `cursors->Cursors.resize(PermutationsCount())`
    # (`doc_parallel_boosting.h:141`), all written to the same starting
    # value (`:180-186`). `cursor` above IS the estimation permutation's
    # and keeps its own variable, so a one-permutation fit runs the code
    # it ran before this block existed, to the bit.
    var perm_count = len(perm_cindexes)
    if perm_count == 0:
        perm_count = 1
    var est_p = est_permutation
    if est_p == -1:
        est_p = perm_count - 1
    if est_p < 0 or est_p >= perm_count:
        raise Error(
            "est_permutation " + String(est_p) + " is outside the "
            + String(perm_count) + " permutations given"
        )
    #: one cursor per permutation, in permutation order.
    #: `cursors[est_p]` IS `cursor` -- `DeviceBuffer.copy()` copies the
    #: handle, not the bytes, so the two names are the same memory and a
    #: write through either is seen by both. That is what keeps the
    #: one-permutation path identical: it allocates one cursor, memsets it
    #: once, and every kernel below writes the same buffer it always did.
    var cursors = List[DeviceBuffer[DType.float32]]()
    for p in range(perm_count):
        if p == est_p:
            cursors.append(cursor.copy())
        else:
            var c = ctx.enqueue_create_buffer[DType.float32](
                approx_dim * n_rows
            )
            # `cursors->Cursors[i]`, ALL written to the same starting
            # value (`doc_parallel_boosting.h:180-186`)
            ctx.enqueue_memset(c, start_value)
            cursors.append(c^)

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
    # under FAST (`hist_smem_mode_for` in `checks/kernel_matrix.mojo`).
    # The scale must be bounded by `sum over all rows of abs(plane)`, and
    # the two plane magnitudes are computed ON THE DEVICE by `pointwise_target_kernel`'s
    # magnitude reduce -- the same block-reduce-plus-one-atomicAdd shape as
    # its `functionValue` -- into TWO scalars read back below. This replaces
    # a full-stats readback (6.4 MB at 800k rows) plus a `2 * n_rows` host
    # loop per tree. A build that quantizes nothing (NVIDIA and 32-lane AMD
    # FAST, whose hist_2 arm keeps CatBoost's warp-private float design)
    # skips the launch flag, the copy and the sync at comptime. A 64-lane
    # column's FAST build quantizes: its one-byte blocks route through the
    # fused 8-bit fixed-point kernel (`greedy_one_byte_fixed_for`,
    # DEVIATION 1906), so it needs the bound exactly as the Apple column
    # does -- an unbounded scale here is the WHY THIS EXISTS failure below.
    comptime _flush_fixed = deterministic_flush_for[
        TARGET_COLUMN, PIN_DETERMINISM
    ]()
    comptime _one_byte_fixed = greedy_one_byte_fixed_for[
        TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
    ]()
    comptime _needs_magnitudes = (
        _flush_fixed or HIST2_SMEM_IS_I32 or _one_byte_fixed
    )
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
        # ...AND ONLY AT ONE PERMUTATION. DEVIATION 64 rests on the
        # searcher's leaf being the same number a Newton step from zero
        # gives, which it is -- for the rows the searcher partitioned. The
        # other permutations partition the same tree differently, so their
        # leaves are different numbers and there is nothing to reuse.
        if perm_count > 1:
            need_estimation = True
    var est_sm = -1
    # `use_pointwise_searcher` ALSO needs it, and that is DEVIATION 64
    # meeting DEVIATION 104. Item 64 skips estimation for RMSE + Newton at
    # one iteration and one permutation, because the GREEDY searcher's leaf
    # already IS the Newton step. `TDocParallelObliviousTreeSearcher`
    # returns the STRUCTURE ONLY -- theirs calls a separate
    # `ReadAndEstimateLeaves` -- so the pointwise arm has no leaf to reuse
    # and must run the estimator whatever item 64 says. Leaving `est_sm` at
    # -1 hands the estimator's reduce an invalid block count.
    # ...and the NON-SYMMETRIC arm too, for the same reason as the pointwise
    # one: `fit_non_symmetric_tree` returns the searcher's own leaf values
    # but never applies them, so the estimator runs on every tree whatever
    # DEVIATION 64 says (the numbers coincide for RMSE by its derivation).
    if need_estimation or use_pointwise_searcher or non_symmetric:
        est_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    # the `random_strength` reduces are machine-sized like every strided
    # grid of theirs, and neither of the two conditions above implies the
    # option is set.
    var noise_sm = est_sm
    if random_strength != Float32(0.0) and noise_sm <= 0:
        noise_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    # their `TGpuAwareRandom` for the fit, the one both searchers draw
    # their per-level `GlobalSeed` from (`oblivious_tree_doc_parallel_
    # structure_searcher.cpp:15`, `greedy_search_helper.h:15`). One draw
    # per TREE here; see DEVIATION 139.
    var noise_rand = TRandom(random_seed)

    var losses = List[Float64]()
    var test_losses = List[Float64]()
    # `CreateOverfittingDetector(options, maxIsOptimal, hasTest)`
    # (`overfitting_detector.cpp:205-207`). EVERY loss this port trains is
    # MINIMIZED, so `maxIsOptimal` is False; a detector built without a
    # test set is inert whatever was asked (`:122-124`).
    var has_test = test.__bool__() and test.value().n_rows > 0
    if boost_from_average and has_test:
        # their `CreateCursors` seeds the TEST cursor with the same
        # `StartingPoint` (the CB_ENSURE at `:174-182` names
        # TestDataProvider precisely because the seed reaches it)
        ref t_arm = test.value()
        ctx.enqueue_memset(t_arm.cursor, start_value)
    var detector = make_overfitting_detector(
        od_type, False, od_pvalue, od_wait, has_test
    )
    if od_type != OD_NONE and not has_test:
        raise Error(
            "od_type is set but there is no held-out set. Stopping on the"
            " LEARN loss would stop on a curve that falls almost by"
            " construction; their own detector is inert without a test"
            " set (overfitting_detector.cpp:122-124) and this refuses"
            " rather than silently never firing."
        )
    var stopped_early = False
    # the SAME layout the searcher uses; the test rows were quantized
    # against the same borders, so the same feature offsets apply
    var layout_for_test = build_layout(fold_counts, one_hot)
    var not_pd_blocks = 0
    var not_pd_total = 0
    # their `TVector<TResultModel>* result`, the ensemble being built

    # THE POOL OF ONE, see `TTreeWorkspace`: the tree planes are the FIT's,

    # not the tree's, so growing tree 2 allocates nothing.

    var ws = List[TTreeWorkspace]()
    # the pointwise arm's pool of one (DEVIATION 143), same span: the FIT's
    var pw_pool = List[PointwiseTreeWorkspace]()
    # the non-symmetric arm's per-leaf pool (`TDepthwiseWorkspace`), same
    # span; `fit_non_symmetric_tree` re-keys it on the shape and reuses it
    var dws = List[TDepthwiseWorkspace]()
    # the ESTIMATOR's pool (DEVIATION 1890), same span: every
    # `_estimate_and_apply` task of every permutation of every tree reuses
    # one set of gather/staging buffers, keyed by (n_rows, approx_dim) with
    # a leaf-count capacity
    var est_ws = List[TEstimationWorkspace]()

    # `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`.
    # BOTH their searchers key the stat planes' content on it -- the greedy
    # arm through `ComputeTarget` -> `StochasticDer`
    # (`greedy_search_helper.cpp:286-296`), the doc-parallel arm through
    # `NewtonAtZero` vs `GradientAtZero`
    # (`oblivious_tree_doc_parallel_structure_searcher.cpp:195-207`) -- and
    # both arms here read the ONE stats buffer the launch below fills, so
    # this is the one place the flag applies.
    var second_order = is_second_order_score_function(score_function)
    if second_order and (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
    ):
        # their `CB_ENSURE(!secondDerAsWeights, ...)`
        # (`multiclass_targets.cpp:27`), raised at fit entry rather than
        # inside the first tree's der launch -- same first observable
        # effect, no half-grown tree.
        raise Error(
            "MultiClass loss doesn't support second derivatives in tree"
            " structure search currently (their CB_ENSURE,"
            " `multiclass_targets.cpp:27`)"
        )

    # one stage clock per fit -- env read ONCE (MOJOLEARN_STAGE_TIMES=1);
    # the identity trace arrives from the caller so border records and
    # tree records share one seq space.
    var stage_times = StageTimes()

    for iteration in range(n_estimators):
        # ---- which permutation the STRUCTURE is searched on ----------
        #
        # `TRandom rand(iteration + BaseIterationSeed); rand.Advance(10);`
        # (`doc_parallel_boosting.h:343-345`) and the draw (`:349-351`):
        #
        #     learnPermutationCount = estimationPermutation
        #                             ? permutationCount - 1 : 1
        #     learnPermutationId = learnPermutationCount > 1
        #         ? rand.NextUniformL() % (learnPermutationCount - 1) : 0
        #
        # **THE MODULUS IS `learnPermutationCount - 1`, NOT
        # `learnPermutationCount`**, so at their default of four
        # permutations the structure comes from permutation 0 or 1 and
        # permutation 2 is never searched on. That reads like an
        # off-by-one in their code and it is transcribed rather than
        # corrected: COPY, DO NOT IMPROVE, and a fit that searched on a
        # permutation theirs never searches on is not this port's call to
        # make. It is the same expression in their feature-parallel
        # learner (`dynamic_boosting.h:286-289`), which is evidence it is
        # deliberate or at least old.
        var learn_perm_count = perm_count - 1 if est_p != 0 else 1
        var learn_p = est_p
        if learn_perm_count > 1:
            var rnd = TRandom(UInt64(iteration) + random_seed)
            rnd.advance(10)
            learn_p = Int(
                rnd.next_uniform_l() % UInt64(learn_perm_count - 1)
            )
        # the learn permutation's (index, cursor) pair, as HANDLES onto the
        # real buffers -- `copy()` on a `DeviceBuffer` copies the handle,
        # so the searcher writes the permutation's own cursor.
        var lc = (
            cindex.copy() if learn_p == est_p
            else perm_cindexes[learn_p].copy()
        )
        var lcur = cursors[learn_p].copy()
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
                lcur, n_rows, row_index, False,
                fv_part, True,
                stats, n_rows,
                mag_part, mags_in_mse,
            )
        elif objective == OBJECTIVE_MULTICLASS_OVA:
            # the same `StochasticDer`, its `MultiClassOneVsAll` arm
            # (`:46-49`), where `statCount` keeps the full
            # `1 + NumClasses` because there is no pinned class to drop.
            launch_one_vs_all_value_and_der[True](
                ctx, num_classes, n_rows, targets, weights, has_weights,
                lcur, n_rows, row_index, False,
                fv_part, True,
                stats, n_rows,
                mag_part, mags_in_mse,
            )
        elif second_order:
            # `secondDerAsWeights=true`: plane 0 becomes `weight * der2`
            # (`pointwise_target_impl.h:193-201`); plane 1 stays
            # `weight * der`. Runtime flag to comptime arm, the same shape
            # as the objective dispatch one call down.
            launch_approximate[False, True](
                ctx, objective, targets, weights, Int32(n_rows), lcur,
                Int32(1) if has_weights else Int32(0),
                alpha, logloss_border,
                stats, fv_part, Int32(1),
                mag_part, Int32(1) if mags_in_mse else Int32(0),
                mse_blocks,
            )
        else:
            launch_approximate[False](
                ctx, objective, targets, weights, Int32(n_rows), lcur,
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
        # ================ THE `random_strength` MAGNITUDE ==============
        # `auto mult = CalcScoreModelLengthMult(objectCount,
        #                                       iteration * step);`
        # (`doc_parallel_boosting.h:358-359`), where `step` is the learning
        # rate. It is handed to `CreateStructureSearcher(mult, ...)` and
        # from there either multiplies the greedy searcher's option
        # (`greedy_subsets_searcher.h:73-76`) or becomes the doc-parallel
        # searcher's `ModelLengthMultiplier` (`oblivious_tree_doc_parallel_
        # structure_searcher.h:25,31`).
        var noise_mult = Float64(0.0)
        if random_strength != Float32(0.0):
            noise_mult = calc_score_model_length_mult(
                Float64(n_rows), Float64(iteration) * Float64(learning_rate)
            )
        var tree_seed = noise_rand.next_uniform_l()

        # `ComputeWeakTarget` computes the doc-parallel arm's std dev
        # BETWEEN the gradient and the bootstrap
        # (`oblivious_tree_doc_parallel_structure_searcher.cpp:200-218`),
        # so it is taken from the UNBOOTSTRAPPED planes. The greedy arm's
        # is taken from the bootstrapped ones instead
        # (`greedy_search_helper.cpp:381-385`) and is computed inside
        # `run_tree_layout`. That ordering difference is CatBoost's, and it
        # is a second reason the two arms' noise magnitudes differ; see
        # `gbdt/methods/random_score_helper.mojo` for the first two.
        var pointwise_score_std_dev = Float32(0.0)
        if use_pointwise_searcher and random_strength != Float32(0.0):
            pointwise_score_std_dev = Float32(
                compute_score_std_dev(
                    ctx,
                    noise_mult,
                    Float64(random_strength),
                    stats,
                    n_rows,
                    n_rows,
                    noise_sm if noise_sm > 0 else 1,
                )
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
                boot_mag_part, compute_mags, stat_count,
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
        # under Cosine/L2 and `weight * der2` under NewtonCosine/NewtonL2
        # (`second_order` above); the two coincide for RMSE, whose Der2 is
        # 1.0. Either way the kernel's magnitudes bound the plane AS
        # STORED; see `pointwise_targets.mojo`.
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
        # `checks/fixed_point.mojo`'s stated precondition, "sum over all
        # rows of abs(value) for the plane being accumulated":
        #
        #   this file                      sum of |stats| reduced ON DEVICE
        #                                  by `pointwise_target_kernel` from the exact
        #                                  planes it wrote, both planes, per
        #                                  iteration -- SOUND
        #   checks/level_check.mojo     abs-accumulates a signed generator
        #                                  at :210-218 and :344-352 -- SOUND
        #   checks/level_bench.mojo     same, three sites -- SOUND
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
        # THE SHIPPED BUILD IS `NUMERIC_FAST` (numerics.mojo:74) -- the
        # runtime `determinism` option is validated but wired to nothing,
        # and a comment here used to claim it pinned the integer flush,
        # which was false (caught 2026-08-21 when Andrew asked why the
        # nondeterministic mirror was not the default: it already was).
        # This block still governs the shipped APPLE configuration because
        # Metal's threadgroup atomics are integer-only, so the hist_2
        # shared stage quantizes at `fixed_scale` even under FAST
        # (`HIST2_SMEM_IS_I32`), and `_needs_magnitudes` is that union.
        # ===========================================================
        # DEVIATION 95: the scale is derived on the device inside
        # `run_tree_layout` from the magnitudes buffer, so the per-tree
        # drain that used to read two floats back is gone -- the tree's
        # own drain is now the loop's only one.
        var mags_opt = Optional[DeviceBuffer[DType.float32]]()

        @parameter
        if _needs_magnitudes:
            mags_opt = Optional(mags.copy())

        # `optimizer.Fit(...)` then `Estimate` then `Rescale` then
        # `AppendModels`, all four inside `run_tree_layout`, in their order.
        var splits = List[TBinarySplit]()
        var leaf_values = List[Float32]()
        var leaf_offsets = List[Int]()
        var sizes = List[Int]()
        # the non-symmetric arm's tree, empty on the oblivious arms
        var ns_trees = List[TNonSymmetricTree]()

        if non_symmetric:
            # ---- CatBoost's NON-SYMMETRIC learner (DEVIATION 259) -------
            # `TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>::FitImpl`
            # through the merged Depthwise/Lossguide driver, then their
            # estimation loop over every permutation, exactly as the
            # pointwise arm below: the searcher's partition is NOT reused,
            # the bins are recomputed from the MODEL on each permutation's
            # compressed index (`task.Model->ComputeBins(*task.DataSet,
            # &bins)`, `doc_parallel_leaves_estimator.cpp:48`), grouped by
            # `partition_from_bins`, estimated, and applied. That is also
            # what re-indexes the leaves: the searcher hands back values
            # in LEAF-ID order and the model's bins are VISIT order
            # (`model_builder.mojo`), and the estimator writes the values
            # in the model's own bin order, which is the order the apply
            # kernels read.
            var opts = TTreeStructureSearcherOptions()
            opts.policy = grow_policy
            opts.max_depth = max_depth
            opts.max_leaves = ns_max_leaves
            opts.l2_reg = l2_leaf_reg
            opts.score_function = score_function
            opts.min_leaf_size = Float64(min_data_in_leaf)
            # `options.RandomStrength *= randomStrengthMult`
            # (`greedy_subsets_searcher.h:76`), the same multiply the
            # greedy oblivious arm receives below
            opts.random_strength = Float32(
                noise_mult * Float64(random_strength)
            )
            # THE FIXED-POINT MAGNITUDES, read back like the pointwise arm's
            # scale: the non-symmetric driver derives `choose_scale` on the
            # host from the two plane magnitudes (DEVIATION 95's device
            # derivation is wired to `run_tree_layout` only). One drain per
            # tree to read two floats, under the builds that quantize;
            # PRICED, same bill as the pointwise arm, and the same fix
            # applies when someone measures it.
            var wmag = Float32(0.0)
            var gmag = Float32(0.0)

            @parameter
            if _needs_magnitudes:
                var hm = ctx.enqueue_create_host_buffer[DType.float32](2)
                ctx.enqueue_copy(dst_buf=hm, src_buf=mags)
                ctx.synchronize()
                wmag = hm[0]
                gmag = hm[1]
                _ = hm^  # past the drain
            # ====================== DEVIATION 260 ======================
            # THE hist_2 ACCUMULATION MODE IS THE KERNEL MATRIX'S
            # `HIST2_SMEM_MODE` ROW, the same one the oblivious
            # `run_tree_layout` defaults to: shared-Int32 on Apple and under
            # IDENTICAL, CatBoost's warp-private float on NVIDIA/AMD FAST.
            # The lane's own gates (`check-depthwise`, `check-lossguide`,
            # the E2 growth cards) drive `fit_non_symmetric_tree` at mode 0
            # (warp-private float), so this is the first caller to run the
            # non-symmetric driver on the Int32 arms, and it is recorded
            # because the first measurement said the arm was BROKEN here
            # and that reading was wrong. One Depthwise tree, 20,000 rows
            # x 24 features, three runs each, 2026-08-23 on this M4:
            #
            #   borders   mode 0 (loss)        HIST2_SMEM_MODE, before 261
            #   15        38.508 deterministic 38.508 deterministic
            #   64        38.323 deterministic 41.898 deterministic
            #   128       38.299 deterministic 43.30 / 43.30 / 43.32
            #
            # The divergence began at depth 4 and was a HOST RACE in the
            # non-symmetric driver (DEVIATION 261, one staging pair reused
            # for three id lists per level) that the float arms happened to
            # land on the right side of; after 261 the two modes agree BIT
            # FOR BIT on this fixture (0x8cedc2e5a5fc1ae5 at depth 6, 128
            # borders, FAST; IDENTICAL deterministic at 38.299009), exactly
            # as the oblivious driver's two modes do. `checks/
            # grow_policy_check.mojo` claim 7 is the run-to-run control at
            # this shape. The matrix row stands; an inline mode 0 here would
            # have been a vendor/arm fork hiding a race.
            # ============================================================
            var tree = fit_non_symmetric_tree[HIST2_SMEM_MODE](
                ctx, n_rows, fold_counts, opts,
                lc, stats, row_index,
                wmag, gmag,
                ws, dws, trace,
                one_hot=one_hot,
                approx_dim=approx_dim,
                multiclass_optimization=objective == OBJECTIVE_MULTICLASS,
                random_seed=tree_seed,
                tag_prefix=_tree_tag(iteration) + ".",
            )
            var n_bins = tree.bin_count()
            for p in range(perm_count):
                var pv = List[Float32]()
                var d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
                if p == learn_p:
                    compute_non_symmetric_bins_for_model(
                        ctx, layout_for_test, tree.model_structure, lc,
                        n_rows, d_bins,
                    )
                else:
                    compute_non_symmetric_bins_for_model(
                        ctx, layout_for_test, tree.model_structure,
                        perm_cindexes[p], n_rows, d_bins,
                    )
                var part = partition_from_bins(ctx, d_bins, n_rows, n_bins)
                _estimate_and_apply(
                    ctx, n_rows, approx_dim, len(part.sizes),
                    part.sizes, part.offsets,
                    part.row_index, targets, weights, has_weights,
                    cursors[p],
                    objective, alpha, estimator_alpha, logloss_border,
                    l2_leaf_reg, est_sm, leaf_estimation_method,
                    num_classes, leaf_estimation_iterations,
                    learning_rate,
                    pv, not_pd_total,
                    trace, stage_times,
                    _tree_tag(iteration) + ".perm" + String(p)
                    + ".leaves.estimated",
                    est_ws,
                )
                if p == est_p:
                    leaf_values.clear()
                    for i in range(len(pv)):
                        leaf_values.append(pv[i])
            # their `UpdateLeaves(std::move(point))` (`doc_parallel_leaves_
            # estimator.cpp:39`), then `Rescale(step)` folded in as the
            # oblivious arm folds it (DEVIATION 256 applies unchanged)
            if len(leaf_values) != n_bins * approx_dim:
                raise Error(
                    "the estimator returned " + String(len(leaf_values))
                    + " values for a non-symmetric tree of " + String(n_bins)
                    + " bins x " + String(approx_dim)
                )
            tree.leaf_values.clear()
            for i in range(len(leaf_values)):
                tree.leaf_values.append(leaf_values[i] * learning_rate)
            tree.dim = approx_dim
            ns_trees.append(tree^)
        elif use_pointwise_searcher:
            # ---- CatBoost's SINGLE-TARGET symmetric learner -----------
            # `TDocParallelObliviousTreeSearcher::FitImpl` returns the
            # STRUCTURE only (DEVIATION 104). The partition it grew is not
            # reused: the leaves are re-derived from the structure through
            # `compute_bins_for_model` + `partition_from_bins`, which is
            # the SAME path the non-estimation permutations below already
            # take. That is deliberate -- it means the pointwise arm shares
            # every line of leaf estimation with the greedy arm, so a
            # difference between the two arms can only come from the
            # structure.
            # ===================== THE FIXED-POINT SCALE ==============
            # The 8-bit accumulator holds Int32 fixed point (DEVIATION 93,
            # Metal has no threadgroup float atomics), so it needs the same
            # scale the greedy path derives: `choose_scale` over the LARGER
            # of the two planes' sums of magnitudes.
            #
            # This passed a hardcoded 1.0 and the symptom is the one
            # DEVIATION 95's block warns about in as many words -- "the tree
            # still learns, because the leaf VALUES come from
            # `compute_partition_stats` and never touch the accumulator;
            # only the SPLITS go bad". It reproduced CatBoost 48/48 at 15
            # and 100 borders and 7/48 at 254, because 254 borders is the
            # only fixture that reaches the 8-bit kernel at all.
            #
            # PRICED: this drains once per tree to read two floats, which
            # the greedy path stopped doing in DEVIATION 95 by deriving the
            # scale on the device. It cannot do the same yet because
            # `compute_hist2` takes `fixed_scale` as a HOST scalar
            # (`pointwise_kernels.mojo:1318`); making it a device pointer is
            # the fix and is not attempted here.
            var scale = Float32(1.0)
            @parameter
            if _needs_magnitudes:
                var hm = ctx.enqueue_create_host_buffer[DType.float32](2)
                ctx.enqueue_copy(dst_buf=hm, src_buf=mags)
                ctx.synchronize()
                _ = boot_mag_part^  # past the drain (step-33 race class, device side)
                _ = mags^  # past the drain (step-33 race class, device side)
                var m0 = Float64(hm[0])
                if m0 < 0.0:
                    m0 = -m0
                var m1 = Float64(hm[1])
                if m1 < 0.0:
                    m1 = -m1
                scale = Float32(
                    choose_scale(m1 if m1 > m0 else m0, n_rows)
                )

            var planes = split_stat_planes(ctx, stats, n_rows)
            splits = fit_oblivious_tree_structure_traced(
                ctx, layout_for_test, n_rows, max_depth, lc,
                planes[0], planes[1],
                est_sm if est_sm > 0 else 1,
                scale,
                score_function,
                pw_pool,
                trace, stage_times, _tree_tag(iteration),
                l2_leaf_reg,
                score_std_dev=pointwise_score_std_dev,
                seed=tree_seed,
                one_hot=one_hot,
            )
            var d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
            compute_bins_for_model(
                ctx, layout_for_test, splits, len(splits), lc, n_rows,
                d_bins,
            )
            var part = partition_from_bins(
                ctx, d_bins, n_rows, 1 << len(splits)
            )
            sizes = part.sizes.copy()
            leaf_offsets = part.offsets.copy()
            row_index = part.row_index.copy()
            if not need_estimation:
                # the greedy arm estimates and applies INSIDE
                # `run_tree_layout` when the method is Simple; this arm has
                # to do it explicitly, through the same estimator.
                _estimate_and_apply(
                    ctx, n_rows, approx_dim, len(sizes), sizes,
                    leaf_offsets, row_index, targets, weights, has_weights,
                    lcur, objective, alpha, estimator_alpha,
                    logloss_border, l2_leaf_reg, est_sm,
                    leaf_estimation_method, num_classes,
                    leaf_estimation_iterations, learning_rate,
                    leaf_values, not_pd_total,
                    trace, stage_times,
                    _tree_tag(iteration) + ".leaves.estimated",
                    est_ws,
                )
        else:
            sizes = run_tree_layout_traced(
                ctx, n_rows, fold_counts, max_depth,
                lc, stats, row_index, lcur,
                Float32(0.0), Float32(0.0),
                splits, leaf_values, leaf_offsets, ws,
                trace, stage_times, _tree_tag(iteration),
                need_estimation,
                use_subtraction, not need_estimation,
                learning_rate, l2_leaf_reg,
                one_hot=one_hot,
                score_function=score_function,
                approx_dim=approx_dim,
                mags_dev=mags_opt^,
                # `MultiLogitOptimization` is set ONLY for MultiClass
                # (`multiclass_targets.cpp:32-35`); OneVsAll has no pinned
                # class, so its histogram carries every plane and the
                # score needs no extra leaf contribution.
                multiclass_optimization=objective
                == OBJECTIVE_MULTICLASS,
                # `options.RandomStrength *= randomStrengthMult`
                # (`greedy_subsets_searcher.h:76`). The multiply happens
                # HERE, in the boosting loop, exactly as their
                # `CreateStructureSearcher` does it -- the searcher
                # receives one number and never sees `mult`.
                random_strength=Float32(
                    noise_mult * Float64(random_strength)
                ),
                random_seed=tree_seed,
            )

        if need_estimation and not non_symmetric:
            # ---- their estimation loop (`doc_parallel_boosting.h:
            # 371-385`): ONE TASK PER PERMUTATION, each on its own dataset
            # and its own cursor, all estimating the SAME structure.
            #
            #     for (permutation = 0; permutation < permutationCount; ++permutation) {
            #         estimator.AddEstimationTask(*(learnTarget[permutation]),
            #                                     dataSet.GetDataSetForPermutation(permutation),
            #                                     (*learnCursors)[permutation],
            #                                     &iterationModels[permutation]);
            #     }
            #
            # The permutation the tree was GROWN on inherits the searcher's
            # partition; the others compute their own, because a row's leaf
            # depends on that permutation's CTR columns and the searcher
            # never looked at them.
            for p in range(perm_count):
                var pv = List[Float32]()
                if p == learn_p:
                    _estimate_and_apply(
                        ctx, n_rows, approx_dim, len(sizes), sizes,
                        leaf_offsets,
                        row_index, targets, weights, has_weights,
                        cursors[p],
                        objective, alpha, estimator_alpha, logloss_border,
                        l2_leaf_reg, est_sm, leaf_estimation_method,
                        num_classes, leaf_estimation_iterations,
                        learning_rate,
                        pv, not_pd_total,
                        trace, stage_times,
                        _tree_tag(iteration) + ".perm" + String(p)
                        + ".leaves.estimated",
                        est_ws,
                    )
                else:
                    var d_bins = ctx.enqueue_create_buffer[DType.uint32](
                        n_rows
                    )
                    # DEPTH IS `len(splits)`, NOT `max_depth`: a tree that
                    # stopped growing early has fewer splits, and
                    # `len(sizes)` is `1 << len(splits)` either way.
                    compute_bins_for_model(
                        ctx, layout_for_test, splits, len(splits),
                        perm_cindexes[p], n_rows, d_bins,
                    )
                    var part = partition_from_bins(
                        ctx, d_bins, n_rows, len(sizes)
                    )
                    _estimate_and_apply(
                        ctx, n_rows, approx_dim, len(part.sizes),
                        part.sizes, part.offsets,
                        part.row_index, targets, weights, has_weights,
                        cursors[p],
                        objective, alpha, estimator_alpha, logloss_border,
                        l2_leaf_reg, est_sm, leaf_estimation_method,
                        num_classes, leaf_estimation_iterations,
                        learning_rate,
                        pv, not_pd_total,
                        trace, stage_times,
                        _tree_tag(iteration) + ".perm" + String(p)
                        + ".leaves.estimated",
                        est_ws,
                    )
                # the EXPORTED ensemble is the estimation permutation's
                # (`doc_parallel_boosting.h:526-528`); the others exist to
                # carry their own cursors forward and their weak models are
                # never read, so they are not accumulated.
                if p == est_p:
                    leaf_values.clear()
                    for i in range(len(pv)):
                        leaf_values.append(pv[i])
        _ = len(sizes)

        # their `result[i].AddWeakModel(iterationModels[i])` (`:398`).
        # `Rescale(step)` is folded into `add_model_value_kernel`, so the
        # stored values are UNSCALED and the rate is applied on the way out.
        if non_symmetric:
            # the `TAdditiveModel<TNonSymmetricTree>` instantiation; the
            # rate was folded into the tree's values above
            model.add_non_symmetric_model(ns_trees.pop())
        else:
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
            # DEVIATION 256, justified UNPINNED: IDENTITY_PATHS row 9 names
            # the LEAF RESCALE, and this multiply is it -- the device half
            # was already closed as the cursor-update fma
            # (`add_model_value_kernel`, `39a0d88`), and what remains is one
            # correctly-rounded HOST Float32 multiply with no chain to
            # contract and no device flush policy in play, so its bits are
            # the same on every host.
            for i in range(len(leaf_values)):
                weak.leaf_values.append(leaf_values[i] * learning_rate)
            model.add_weak_model(weak^)

        # ---- their `AppendModels(..., learnCursors, testCursor)` -----
        # (`doc_parallel_boosting.h:391-396`): the SAME weak model goes on
        # to the held-out cursor. One tree at a time, through the
        # ROW-WISE apply, because the partition-wise one only knows rows
        # the tree was grown on.
        if has_test:
            _apply_last_tree_to_test(
                ctx, model, layout_for_test, test.value(), approx_dim,
                learning_rate,
            )
            var t_loss = _test_loss(
                ctx, test.value(), objective, num_classes, alpha,
                logloss_border, approx_dim,
            )
            test_losses.append(t_loss)
            # `DetectOverfitting(testError, detector, ...)`
            # (`overfitting_detector.h:25-34`)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                stopped_early = True
                break

        # the device `functionValue` read alongside this iteration's
        # gradients: the loss AFTER THE PREVIOUS TREE (their accumulation
        # is `-w * (val - relev)^2`, `pointwise_targets.cu:311`, so the
        # positive loss is its negation). Iteration 0's value is the
        # baseline and is not a tree's loss, so it is dropped, and the
        # LAST tree's loss comes from one extra gradient pass below.
        if len(losses) < n_estimators:
            var v = Float64(h_fv.unsafe_ptr().unsafe_load(0))
            # `size()` counts either shape (the non-symmetric ensemble's
            # trees are in `non_symmetric_models`)
            if model.size() > 1:
                losses.append(-v / Float64(n_rows))

    # the final tree's loss: one more `functionValue` pass over the settled
    # cursor, through the SAME per-block-partials + fixed-order fold as the
    # loop's pass -- this launch still handed the 1-float `fv` where the
    # kernel now stores per-block partials, which under-read the loss by
    # ~n_blocks and wrote past the buffer (caught by the predict repro:
    # replay 36.52 vs a claimed 8.91 final loss, ratio ~= the block count).
    var final_blocks = mse_blocks
    if objective == OBJECTIVE_MULTICLASS_OVA:
        final_blocks = multilogit_blocks(n_rows)
        launch_one_vs_all_value_and_der[True](
            ctx, num_classes, n_rows, targets, weights, has_weights,
            cursor, n_rows, row_index, False,
            fv_part, True,
            stats, n_rows,
            mag_part, False,
        )
    elif objective == OBJECTIVE_MULTICLASS:
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
    # ================= DEVIATION 2002, LEG 2 =================
    # THE POISONED FINAL-LOSS WORD, the whole-fit backstop across every
    # grow policy this loop drives. The 134 loaded window measured fits
    # returning with `mse -0.0` because a dead Metal context silently
    # stopped delivering readbacks: `h_fv` kept its stale +0.0 and
    # `-0.0 / n` became the reported train loss. The host plants a
    # poison BIT PATTERN in the word before the copy is enqueued; on a
    # live device the drain overwrites it with the accumulated
    # `functionValue`, and the exact poison bits surviving means the
    # loss was NEVER DELIVERED -- so the fit raises rather than record
    # a loss it never computed. Bit-compared, not value-compared, so a
    # legitimate loss of ANY value (including a constant-target fit's
    # exact -0.0) passes: only non-delivery fails, which cannot happen
    # on a live in-order queue with the synchronize below.
    # `-D MOJOLEARN_2002_SABOTAGE=1` (REQUIRED-RED, see
    # `core/device_liveness.mojo`) compiles the copy out to fake
    # exactly that signature.
    # =========================================================
    h_fv.unsafe_ptr().bitcast[UInt32]().unsafe_store(
        0, DEAD_DEVICE_POISON
    )
    comptime if not SAB_2002_DEAD_DEVICE:
        ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=fv)
    ctx.synchronize()
    _ = fv_part^  # past the drain (step-33 race class, device side)
    _ = fv^  # past the drain (step-33 race class, device side)
    if (
        h_fv.unsafe_ptr().bitcast[UInt32]().unsafe_load(0)
        == DEAD_DEVICE_POISON
    ):
        raise Error(
            "DEVIATION 2002: refusing to return this fit -- the final"
            " train loss was never delivered (the host-side poison"
            " survived the drain). This is the dead/saturated-device"
            " signature (Metal context death under load): fits in this"
            " state return empty models and a stale mse of -0.0 with no"
            " error, so the fit raises instead. The process's GPU"
            " context is not trustworthy; restart the process on a"
            " quieter box."
            + (
                " [SABOTAGE ARM -D MOJOLEARN_2002_SABOTAGE=1 IS ARMED:"
                " this failure is the REQUIRED-RED verdict.]"
                if SAB_2002_DEAD_DEVICE else ""
            )
        )
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

    stage_times.report(
        grow_policy_name(grow_policy) + " fit (doc-parallel boosting)"
    )

    return FitResult(
        losses^, test_losses^, detector.best_iteration, stopped_early
    )


def model_approx_dim(model: TAdditiveModel) raises -> Int:
    """`OutputDim()` of the ensemble (`oblivious_model.h:130-133`).

    Every weak model in one ensemble has the same `Dim`; disagreement is a
    corrupted model rather than a case to handle, so it raises. An empty
    ensemble is one-dimensional, which is what a zero-tree predict returns.
    """
    if model.size() == 0:
        return 1
    if not model.is_oblivious():
        # `TNonSymmetricTree::OutputDim()` (`non_symmetric_tree.h:179-182`),
        # the same one-dim-per-ensemble rule
        var nd = model.non_symmetric_models[0].dim
        for t in range(1, model.size()):
            if model.non_symmetric_models[t].dim != nd:
                raise Error(
                    "non-symmetric tree " + String(t) + " has dim "
                    + String(model.non_symmetric_models[t].dim)
                    + " but tree 0 has " + String(nd)
                )
        if nd < 1:
            raise Error("model dim is " + String(nd))
        return nd
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

    # the model's bias IS the cursor's starting value
    # (`modelToExport.SetBias(cursors->StartingPoint)`,
    # `doc_parallel_boosting.h:434`): a fit under `boost_from_average`
    # grew every tree against a cursor seeded there, so an apply that
    # started at zero would return the residual, not the target.
    ctx.enqueue_memset(cursor, Float32(model.bias))

    if not model.is_oblivious():
        # THE NON-SYMMETRIC SHAPE (DEVIATION 259): their
        # `TAddModelDocParallel<TNonSymmetricTree>`, one tree at a time --
        # bins off the model, `AddBinModelValues` onto the cursor
        # (`add_non_symmetric_tree_doc_parallel.cpp:182-206`). Each tree
        # drains once (its bins buffer is per tree); that is a fixed
        # per-tree cost on a path that runs once per predict, stated
        # rather than hidden, and the packed-once form the oblivious arm
        # below takes is the fix if anyone measures a need.
        for t in range(model.size()):
            add_non_symmetric_tree_to_cursor(
                ctx, layout, model.non_symmetric_models[t], cindex, n_rows,
                cursor,
            )
        ctx.synchronize()
        return

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
    _ = d_vals^  # past the drain (step-33 race class, device side)
    _ = d_eq^  # past the drain (step-33 race class, device side)
    _ = d_bin^  # past the drain (step-33 race class, device side)
    _ = d_mask^  # past the drain (step-33 race class, device side)
    _ = d_shift^  # past the drain (step-33 race class, device side)
    _ = d_off^  # past the drain (step-33 race class, device side)
    # past the drain: the six pack buffers' last uses were their
    # enqueues, which freed them under queued copies -- and THIS is the
    # AUC path, so the step-33 race class here corrupts the metric even
    # when the fit was clean
    _ = h_off^
    _ = h_shift^
    _ = h_mask^
    _ = h_bin^
    _ = h_eq^
    _ = h_vals^

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
    # forwarded; see `fit_with_test` for what it selects and why the
    # default is False
    use_pointwise_searcher: Bool = False,
    # forwarded; `oblivious_tree_options.cpp:17`
    random_strength: Float32 = Float32(0.0),
    # forwarded; resolved by the caller (see `fit_with_test`'s docstring
    # on this parameter)
    boost_from_average: Bool = False,) raises -> List[Float64]:
    """`fit_with_test` with no held-out set and no detector.

    THIS WRAPPER EXISTS SO NINE CALL SITES DID NOT HAVE TO CHANGE when the
    test arm landed -- checks, benches, and another session's interleaved
    harness. Its signature and return are exactly what they were.
    """
    var trace = IdentityTrace()
    var r = fit_with_test(
        model=model,
        ctx=ctx,
        trace=trace,
        n_rows=n_rows,
        fold_counts=fold_counts,
        max_depth=max_depth,
        cindex=cindex,
        targets=targets,
        weights=weights,
        has_weights=has_weights,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        use_subtraction=use_subtraction,
        bootstrap_bayesian=bootstrap_bayesian,
        bagging_temperature=bagging_temperature,
        bootstrap_type=bootstrap_type,
        bootstrap_param=bootstrap_param,
        random_seed=random_seed,
        one_hot=one_hot,
        score_function=score_function,
        objective=objective,
        num_classes=num_classes,
        logloss_border=logloss_border,
        leaf_estimation_iterations=leaf_estimation_iterations,
        leaf_estimation_method=leaf_estimation_method,
        alpha=alpha,
        estimator_alpha=estimator_alpha,
        use_pointwise_searcher=use_pointwise_searcher,
        random_strength=random_strength,
        boost_from_average=boost_from_average,
    )
    var out = r.learn_losses.copy()
    return out^
