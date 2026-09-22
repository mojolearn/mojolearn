# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's GPU Ordered boosting for `GradientBoosting(boosting_type=
'Ordered')` (lane/catboost-parity, 2026-09-19).

Reference: `catboost/cuda/methods/dynamic_boosting.h` (`TDynamicBoosting`,
CatBoost `54a8143a`), with the weak learner it runs,
`TFeatureParallelPointwiseObliviousTree` (`feature_parallel_pointwise_
oblivious_tree.h`) and its fold searcher `TFeatureParallelObliviousTreeSearcher`
(`oblivious_tree_structure_searcher.cpp`). Their GPU learner takes this arm for
every Ordered fit: `DataPartitionType` is FeatureParallel by default
(`boosting_options.cpp:25`) and only a Plain fit is moved to DocParallel
(`cuda/train_lib/train.cpp:73-84`), where Ordered is refused
(`boosting_options.cpp:71-74`).

## What one fit does, in their order

`CreateState` (`dynamic_boosting.h:555-640`):

  * `permutation_count` datasets (default 4, `boosting_options.cpp:14`), each
    `GetPermutation(dataProvider, id, blockSize)` (`data/permutation.h:98-104`)
    of the learn pool. Permutation 0 is the IDENTITY of the pool, and the pool
    their Ordered fit sees was SHUFFLED AT LOAD (`ShuffleLearnDataIfNeeded`,
    `private/libs/algo/preprocess.cpp:161-199`: `NeedShuffle` is true for an
    Ordered fit with no time column). Both are restated: `ordered_permutations`
    composes the load shuffle with each `TDataPermutation`.
  * `blockSize` is `GetPermutationBlockSize` (`dynamic_boosting.h:115-128`):
    1 below 50,000 rows, else `fold_permutation_block` (64 when unset or 0,
    `cuda/train_lib/train.cpp:115-118`) rounded to a power of two and halved
    while `block * 128 > rows`.
  * the LAST permutation is the ESTIMATION permutation; the others are LEARN
    permutations (all of them when there is more than one, else the one).
  * per learn permutation, `CreateFolds` (`:189-223`, `dynamic_boosting_folds.
    create_folds`): folds whose estimate slice is a prefix `[0, L)` of the
    permutation and whose quality slice is `[L, R)`, growing by
    `fold_len_multiplier` (default 2, `boosting_options.cpp:11`) from
    `MinEstimationSize` (`min_fold_size` 100, `:24`). One cursor per fold over
    `[0, R)` in permutation order, and one estimation cursor over every row,
    all written with the starting point.

`Fit` (`:233-469`), per iteration:

  1. the learn permutation the STRUCTURE is searched on:
     `Random.NextUniformL() % (learnPermutationCount - 1)` when there is more
     than one (`:282-289`) -- the modulus is `count - 1`, so at the default four
     permutations the structure comes from permutation 0 or 1, and permutation 2
     is estimated but never searched on, as theirs.
  2. per fold, the derivatives at the fold cursor on `[0, R)`
     (`TTargetAtPointTrait::Create`, `:307-321`), learn slice and quality slice
     concatenated fold after fold (`ComputeWeakTarget`, `oblivious_tree_
     structure_searcher.cpp:381-445`): plane 0 the weight (`weight * der2`
     under NewtonCosine), plane 1 `weight * der`.
  3. `ScoreStdDev = ModelLengthMultiplier * sqrt(sum2 / count) *
     random_strength` over the QUALITY slices only (`:430-442`), with
     `ModelLengthMultiplier = CalcScoreModelLengthMult(rows, iteration * lr)`
     (`dynamic_boosting.h:299-301`).
  4. the bootstrap (`:197-212`): one draw per concatenated position, then the
     LEARN slices' draws reset to 1 (`ObservationsToBootstrap` TestOnly, their
     default, `oblivious_tree_options.cpp:31`), both planes multiplied.
  5. the fold-based oblivious structure search (`fit_oblivious_tree_structure`
     with `folds` and the learn permutation, the dynamic Cosine / NewtonCosine
     scorer).
  6. leaves estimated per (learn permutation, fold) on the estimate slice at
     that fold's cursor, and for the estimation permutation on every row at the
     estimation cursor (`:369-411`), with the loss's own leaf estimator and the
     ORIGINAL weights (the bootstrap reaches the search only); each model
     rescaled by the learning rate and added to its cursor over `[0, R)`
     (`:413-446`); the estimation permutation's model is the one exported
     (`:448`).
  7. the learn loss at the estimation cursor (`metricCalcer.SetPoint(cursor.
     Estimation)`, `:452-455`).
  8. with an eval set, the exported model onto the test cursor (`:423-430`),
     the held-out loss and the overfitting detector, through the Plain fit's
     own `_apply_last_tree_to_test`, `_test_loss` and detector.

## What is refused, by name, and why (the caller, `gbdt/train.mojo`)

  * Depthwise / Lossguide: "Ordered boosting is not supported for nonsymmetric
    trees" (`catboost_options.cpp:757-759`).
  * MultiClass / MultiClassOneVsAll: their GPU forces Plain for these losses
    and refuses an explicit Ordered (`catboost_options.cpp:949-967`).
  * L2 / NewtonL2 scores: "can't be used with ordered boosting"
    (`catboost_options.cpp:972-978`; `FindOptimalSplitDynamic` has no arm).
  * the Exact leaf estimator: "Exact leaf estimation method don't work with
    ordered boosting on GPU" (`catboost_options.cpp:346-350`). Unset, MAE,
    Quantile and MAPE take their Gradient default under Ordered, as theirs
    (`useExact` needs Plain on GPU, `:290-293`).
  * NOT IMPLEMENTED here (CatBoost has them): categorical CTR features (their
    permutation-dependent CTR datasets), the ranking losses (query-aware folds),
    the pointwise doc-parallel searcher flag, feature_fraction below 1.

An eval set, the overfitting detector and use_best_model are theirs: step 8
below, their test cursor.

## The identity contract

Every reduction a decision reads is a fixed-order device fold
(`deterministic_sum_lanes_kernel`), every stored float product is flushed BY
BITS (`ftz`), the fixed-point histogram scale comes from those folds, and the
host random streams are `TRandom`, so the bits do not depend on the vendor.
`gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit` restates this file
on the host for the CPU column.

## Deviations from theirs (the model is theirs; the random streams are not)

  * the load shuffle is `permutation.shuffle` seeded from `random_seed` rather
    than their `TRestorableFastRng64`, and the learn-permutation and per-tree
    noise draws come from their own `TRandom` stream rather than the fit's
    shared `TGpuAwareRandom`: the same distributions, not the same draws;
  * the score noise multiplier uses the portable log/exp (the same formula);
  * a quality row of weight 0 contributes nothing to `sum2` (theirs divides by
    it, `DivideVector`, and poisons the standard deviation with a NaN);
  * fold boundaries are the ONE-device ones on every device count: their
    `CreateFolds` widens the first fold on several GPUs (`:194-198`), and a
    multi-GPU fit here must equal the one-GPU fit bit for bit.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_idx, block_dim, thread_idx
from std.math import isfinite, sqrt
from std.sys.compile import is_defined

from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from checks.fixed_point import choose_scale
from checks.numerics import ftz, identical_mul
from gbdt.data.ordered_plan import (
    ORDERED_BOOTSTRAP_SALT,
    ORDERED_MIN_FOLD_SIZE,
    ORDERED_STREAM_SALT,
    ordered_model_length_mult,
    ordered_permutation_block_size,
    ordered_permutations,
)
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.gpu_util.kernel.bootstrap import (
    bootstrap_grid_blocks,
    create_bootstrap_seeds,
    launch_bootstrap,
)
from gbdt.methods.doc_parallel_boosting import (
    TEstimationWorkspace,
    TestArm,
    _apply_last_tree_to_test,
    _estimate_and_apply,
    _test_loss,
)
from gbdt.overfitting_detector.overfitting_detector import (
    make_overfitting_detector,
)
from gbdt.methods.dynamic_boosting import (
    _ordered_apply_kernel,
    _ordered_gather_kernel,
)
from gbdt.methods.dynamic_boosting_folds import (
    EBoostingType,
    IQueriesGrouping,
    TFold,
    create_folds,
)
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import StageTimes
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    compute_bins_for_model,
    LeafPartition,
)
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import (
    PointwiseTreeWorkspace,
    fit_oblivious_tree_structure_traced,
)
from gbdt.models.oblivious_model import (
    TAdditiveModel,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.methods.kernel.pointwise_scores import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_NEWTON_COSINE,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    deterministic_sum_lanes_kernel,
    launch_approximate,
)

comptime ORDERED_BLOCK = 256

#: THE NEGATIVE CONTROL of this arm: `-D MOJOLEARN_ORDERED_SABOTAGE=1` makes
#: every FOLD estimate its leaves on the WHOLE fold, quality slice included --
#: the look-ahead ordered boosting exists to prevent, i.e. Plain boosting's
#: leaves on the fold cursors. The estimation permutation's leaves are
#: untouched, so only the fold cursors, and through their gradients the next
#: trees' structures, move. (A one-row leak was tried first and moved no cell
#: of gbdt-ordered: 20 trees chose the same splits.) The host oracle carries
#: the same arm (`GBDT_ORDERED_SABOTAGE`).
comptime ORDERED_SABOTAGE = is_defined["MOJOLEARN_ORDERED_SABOTAGE"]()


@fieldwise_init
struct OrderedFitOutput(Movable):
    """The learn curve, the held-out curve (empty without an eval set), the
    error tracker's best iteration and whether the detector stopped the
    fit -- what `fit_with_test`'s `FitResult` reports for a Plain fit."""

    var learn_losses: List[Float64]
    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool


@fieldwise_init
struct OrderedBoostingOptions(Copyable, Movable):
    """Everything the Ordered loop reads that is not an array, resolved by
    `train` the way their options resolve it."""

    var objective: Int
    #: the target kernel's alpha (`TLossDescription.kernel_alpha`)
    var kernel_alpha: Float32
    #: the leaf estimator's alpha (`get_alpha`)
    var estimator_alpha: Float32
    var logloss_border: Float32
    var leaf_method: Int
    var leaf_iterations: Int
    var score_function: Int
    var learning_rate: Float32
    var l2_leaf_reg: Float32
    var random_strength: Float32
    var random_seed: UInt64
    #: `BOOTSTRAP_KERNEL_*`, -1 for none
    var bootstrap_kind: Int
    var bootstrap_param: Float32
    var permutation_count: Int
    var fold_len_multiplier: Float64
    #: the RESOLVED permutation block size (`ordered_permutation_block_size`)
    var permutation_block: Int
    var min_fold_size: Int
    #: the starting point (`boost_from_average`), 0 without it
    var start_value: Float32


def ordered_folds(n_rows: Int, fold_len_multiplier: Float64, min_fold_size: Int) raises -> List[TFold]:
    """`CreateFolds` at ONE device (see the module docstring)."""
    return create_folds(
        n_rows, fold_len_multiplier, IQueriesGrouping.without_queries(n_rows),
        EBoostingType.Ordered, min_fold_size, 1,
    )


# ===========================================================================
# KERNELS
# ===========================================================================


def _ord_gather_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """`dst[i] = src[perm[i]]`: a row value into permutation order."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        dst.unsafe_store(i, src.unsafe_load(Int(perm.unsafe_load(i))))


def _ord_scatter_planes_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    offset_in: Int32,
):
    """A fold's two search planes (`stats[i]`, `stats[size + i]`) into its
    slot of the concatenated weight and weighted-target planes."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var off = Int(offset_in)
        sw.unsafe_store(off + i, stats.unsafe_load(i))
        sg.unsafe_store(off + i, stats.unsafe_load(size + i))


def _ord_std_terms_kernel(
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    quality: MutPointer[UInt32, MutAnyOrigin],
    terms: MutPointer[Float32, MutAnyOrigin],
    total_in: Int32,
):
    """Their `DivideVector(testTarget, testWeights)` then `DotProduct(t, t,
    &testWeights)` (`oblivious_tree_structure_searcher.cpp:430-438`), one term
    per QUALITY position: `((g / w)^2) * w`, 0 elsewhere and at w <= 0 (the
    DEVIATION in the module docstring)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total_in):
        var term = Float32(0.0)
        if quality.unsafe_load(i) != UInt32(0):
            var w = sw.unsafe_load(i)
            if w > Float32(0.0):
                var q = ftz(sg.unsafe_load(i) / w)
                term = ftz(ftz(q * q) * w)
        terms.unsafe_store(i, term)


def _ord_bootstrap_apply_kernel(
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    draws: MutPointer[Float32, MutAnyOrigin],
    quality: MutPointer[UInt32, MutAnyOrigin],
    total_in: Int32,
):
    """`FillBuffer(learnWeights, 1.0f)` then the two `MultiplyVector`s
    (`oblivious_tree_structure_searcher.cpp:199-211`): a quality position
    takes its draw, a learn position keeps its planes."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total_in):
        if quality.unsafe_load(i) != UInt32(0):
            var b = draws.unsafe_load(i)
            sw.unsafe_store(i, ftz(sw.unsafe_load(i) * b))
            sg.unsafe_store(i, ftz(sg.unsafe_load(i) * b))


def _ord_abs_planes_kernel(
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    total_in: Int32,
):
    """`[|sw[i]|, |sg[i]|]` interleaved, the input of the fixed-order fold
    that gives `choose_scale` its sums of magnitudes."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total_in):
        dst.unsafe_store(2 * i, abs(sw.unsafe_load(i)))
        dst.unsafe_store(2 * i + 1, abs(sg.unsafe_load(i)))


def _grid(n: Int) -> Int:
    return (n + ORDERED_BLOCK - 1) // ORDERED_BLOCK


# ===========================================================================
# ONE ESTIMATION TASK
# ===========================================================================


def _partition_from_host_bins(
    ctx: DeviceContext,
    tree_bins: HostBuffer[DType.uint32],
    permutation: List[UInt32],
    n_rows: Int,
    n_leaves: Int,
    h_rows: HostBuffer[DType.uint32],
) raises -> LeafPartition:
    """`partition_from_bins` (`doc_parallel_leaves_estimator.mojo`,
    DEVIATION 90) over the permutation's prefix `[0, n_rows)` of a tree's
    bins that are ALREADY ON THE HOST: row `i`'s leaf is
    `tree_bins[permutation[i]]`, exactly the value `_ordered_gather_kernel`
    writes to the gathered bins buffer that function would download, and
    the same STABLE counting sort over it, so the partition is the same
    list of integers.

    Why: that function downloads the gathered bins and drains twice per
    call, and an Ordered tree calls it once per (learn permutation, fold)
    plus once for the estimation permutation -- 28 calls on a default fit
    of fewer than 500 rows, all reading ONE tree's bins. The fit now
    downloads those bins once per tree and partitions every task here.

    `h_rows` is the caller's staging (at least `n_rows` long); its upload
    is enqueued and NOT drained here, so the caller keeps it alive until
    its task's tail drain."""
    if n_leaves <= 0:
        raise Error("partition_from_bins: n_leaves must be positive")
    var sizes = List[Int]()
    for _ in range(n_leaves):
        sizes.append(0)
    var bins_p = tree_bins.unsafe_ptr()
    for r in range(n_rows):
        var b = Int(bins_p.unsafe_load(Int(permutation[r])))
        if b < 0 or b >= n_leaves:
            raise Error(
                "partition_from_bins: row " + String(r) + " fell in leaf "
                + String(b) + " of " + String(n_leaves)
            )
        sizes[b] += 1

    var offsets = List[Int]()
    var running = 0
    for i in range(n_leaves):
        offsets.append(running)
        running += sizes[i]

    var fill = offsets.copy()
    var rows_p = h_rows.unsafe_ptr()
    for r in range(n_rows):
        var b = Int(bins_p.unsafe_load(Int(permutation[r])))
        rows_p.unsafe_store(fill[b], UInt32(r))
        fill[b] += 1

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=h_rows.unsafe_ptr())
    return LeafPartition(row_index^, offsets^, sizes^)


def _ordered_estimate_task(
    ctx: DeviceContext,
    estimate_size: Int,
    apply_size: Int,
    n_leaves: Int,
    mut y: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut permutation: DeviceBuffer[DType.uint32],
    host_permutation: List[UInt32],
    mut bins: DeviceBuffer[DType.uint32],
    tree_bins: HostBuffer[DType.uint32],
    h_rows: HostBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    opts: OrderedBoostingOptions,
    sm_count: Int,
    mut est_ws: List[TEstimationWorkspace],
    mut trace: IdentityTrace,
    tag: String,
    mut est_times: StageTimes,
    mut walker_times: StageTimes,
) raises -> List[Float32]:
    """Their `AddEstimationTask(targetSlice(estimate), cursorSlice)` then
    `AddTask(model, [0, apply))` (`dynamic_boosting.h:377-385`,
    `:431-444`): estimate on permutation positions `[0, estimate_size)` at the
    cursor, add `leaf * rate` to cursor positions `[0, apply_size)`.
    The estimator (`_estimate_and_apply`) moves a PRIVATE gathered copy of the
    cursor (its walker's `MoveTo`), never the real one."""
    if estimate_size < 1 or estimate_size > apply_size:
        raise Error("ordered estimation requires 0 < prefix <= cursor size")
    est_times.begin(ctx)
    var gy = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gw = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gc = ctx.enqueue_create_buffer[DType.float32](estimate_size)
    var gb = ctx.enqueue_create_buffer[DType.uint32](estimate_size)
    ctx.enqueue_function[_ordered_gather_kernel](
        y.unsafe_ptr(), weights.unsafe_ptr(), permutation.unsafe_ptr(),
        cursor.unsafe_ptr(), bins.unsafe_ptr(), gy.unsafe_ptr(),
        gw.unsafe_ptr(), gc.unsafe_ptr(), gb.unsafe_ptr(), Int32(estimate_size),
        grid_dim=(_grid(estimate_size), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
    )
    est_times.end(ctx, "est.gather")
    est_times.begin(ctx)
    var part = _partition_from_host_bins(
        ctx, tree_bins, host_permutation, estimate_size, n_leaves, h_rows
    )
    est_times.end(ctx, "est.partition")
    var leaves = List[Float32]()
    var not_pd = 0
    est_times.begin(ctx)
    _estimate_and_apply(
        ctx, estimate_size, 1, n_leaves, part.sizes, part.offsets,
        part.row_index, gy, gw, True, gc, opts.objective,
        opts.kernel_alpha, opts.estimator_alpha, opts.logloss_border,
        opts.l2_leaf_reg, sm_count, opts.leaf_method, 0,
        opts.leaf_iterations, opts.learning_rate, leaves, not_pd,
        trace, walker_times, tag, est_ws,
    )
    est_times.end(ctx, "est.estimate_and_apply")
    est_times.begin(ctx)
    var hl = ctx.enqueue_create_host_buffer[DType.float32](n_leaves)
    var dl = ctx.enqueue_create_buffer[DType.float32](n_leaves)
    for leaf in range(n_leaves):
        hl.unsafe_ptr().unsafe_store(leaf, leaves[leaf])
    ctx.enqueue_copy(dst_buf=dl, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_function[_ordered_apply_kernel](
        permutation.unsafe_ptr(), bins.unsafe_ptr(), dl.unsafe_ptr(),
        cursor.unsafe_ptr(), Int32(apply_size), opts.learning_rate,
        grid_dim=(_grid(apply_size), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
    )
    ctx.synchronize()
    est_times.end(ctx, "est.apply")
    _ = hl^
    _ = dl^
    _ = gb^
    _ = gy^
    _ = gw^
    _ = gc^
    return leaves^


# ===========================================================================
# THE FIT
# ===========================================================================


def fit_ordered(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    mut cindex: DeviceBuffer[DType.uint32],
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_estimators: Int,
    max_depth: Int,
    sm_count: Int,
    one_hot: List[Bool],
    opts: OrderedBoostingOptions,
    mut model: TAdditiveModel,
    mut trace: IdentityTrace,
    mut test: TestArm,
    od_type: Int,
    od_pvalue: Float64,
    od_wait: Int,
) raises -> OrderedFitOutput:
    """Train `n_estimators` oblivious trees by their GPU Ordered boosting
    (see the module docstring) into `model`. The learn loss after each tree
    is `-functionValue / rows` at the estimation cursor, the plain fit's
    convention. `targets` and `weights` are per ORIGINAL row (weights all
    ones without sample or class weights).

    THE HELD-OUT SET (`test.n_rows > 0`): their test cursor, started at the
    starting point (`dynamic_boosting.h:627-635`) and moved by the exported
    estimation model every iteration (`:423-430`), through the Plain fit's
    own `_apply_last_tree_to_test` and `_test_loss`, and the overfitting
    detector over the held-out curve, as the Plain loop feeds it."""
    if n_rows < 4:
        # `CB_ENSURE(queryCount >= 4 * devCount)` (`dynamic_boosting.h:200`)
        raise Error(
            "Error: pool has just " + String(n_rows) + " groups or docs,"
            " can't use #1 GPUs to learn on such small pool"
        )
    if n_estimators < 1 or max_depth < 1 or max_depth > 16:
        raise Error("ordered boosting needs n_estimators >= 1 and depth 1..16")
    if not (
        opts.score_function == SCORE_FUNCTION_COSINE
        or opts.score_function == SCORE_FUNCTION_NEWTON_COSINE
    ):
        raise Error(
            "Score function can't be used with ordered boosting"
            " (catboost_options.cpp:972-978): Cosine and NewtonCosine only"
        )
    if opts.permutation_count < 1:
        raise Error("Permutation count should be positive (boosting_options.cpp:67)")
    if not (opts.fold_len_multiplier > 1.0):
        raise Error(
            "fold len multiplier should be greater than 1"
            " (boosting_options.cpp:64)"
        )
    var second_order = opts.score_function == SCORE_FUNCTION_NEWTON_COSINE

    # ---- CreateState --------------------------------------------------
    var perms = ordered_permutations(
        n_rows, opts.permutation_count, opts.permutation_block,
        opts.random_seed,
    )
    var perm_count = len(perms)
    var est_p = perm_count - 1
    var learn_count = est_p if est_p > 0 else 1
    var folds = ordered_folds(n_rows, opts.fold_len_multiplier, opts.min_fold_size)
    var n_folds = len(folds)
    var dperms = List[DeviceBuffer[DType.uint32]]()
    for p in range(perm_count):
        var h = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
        for i in range(n_rows):
            h.unsafe_ptr().unsafe_store(i, perms[p][i])
        var d = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        _ = h^
        dperms.append(d^)

    # the concatenated fold layout: fold f at [offsets[f], + R_f), its learn
    # slice first; `quality` marks the quality slices
    var offsets = List[Int]()
    var total = 0
    for f in range(n_folds):
        offsets.append(total)
        total += folds[f].quality_evaluate_samples.right
    var hq = ctx.enqueue_create_host_buffer[DType.uint32](total)
    for f in range(n_folds):
        var left = folds[f].estimate_samples.right
        var right = folds[f].quality_evaluate_samples.right
        for i in range(right):
            hq.unsafe_ptr().unsafe_store(
                offsets[f] + i, UInt32(1) if i >= left else UInt32(0)
            )
    var quality = ctx.enqueue_create_buffer[DType.uint32](total)
    ctx.enqueue_copy(dst_buf=quality, src_ptr=hq.unsafe_ptr())

    # cursors: [learn permutation][fold], each over [0, R_f); the
    # estimation cursor over every row in the estimation permutation's order
    var cursors = List[List[DeviceBuffer[DType.float32]]]()
    for _ in range(learn_count):
        var per = List[DeviceBuffer[DType.float32]]()
        for f in range(n_folds):
            var c = ctx.enqueue_create_buffer[DType.float32](
                folds[f].quality_evaluate_samples.right
            )
            enqueue_fill(ctx, c, opts.start_value)
            per.append(c^)
        cursors.append(per^)
    var est_cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    enqueue_fill(ctx, est_cursor, opts.start_value)

    # the estimation permutation's targets and weights, for the learn loss
    var est_y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var est_w = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_function[_ord_gather_kernel](
        targets.unsafe_ptr(), dperms[est_p].unsafe_ptr(), est_y.unsafe_ptr(),
        Int32(n_rows), grid_dim=(_grid(n_rows), 1, 1),
        block_dim=(ORDERED_BLOCK, 1, 1),
    )
    ctx.enqueue_function[_ord_gather_kernel](
        weights.unsafe_ptr(), dperms[est_p].unsafe_ptr(), est_w.unsafe_ptr(),
        Int32(n_rows), grid_dim=(_grid(n_rows), 1, 1),
        block_dim=(ORDERED_BLOCK, 1, 1),
    )

    var bootstrap_on = opts.bootstrap_kind >= 0
    var boot_seeds: DeviceBuffer[DType.uint64]
    if bootstrap_on:
        boot_seeds = create_bootstrap_seeds(
            ctx, opts.random_seed ^ ORDERED_BOOTSTRAP_SALT
        )
    else:
        boot_seeds = ctx.enqueue_create_buffer[DType.uint64](1)
    var boot_mags = ctx.enqueue_create_buffer[DType.float32](
        2 * bootstrap_grid_blocks(total)
    )
    var rng = TRandom(opts.random_seed ^ ORDERED_STREAM_SALT)
    var pool = List[PointwiseTreeWorkspace]()
    # ONE ESTIMATION WORKSPACE PER ESTIMATE SIZE, not one for the fit.
    # `_estimate_and_apply`'s pool of one (DEVIATION 1890) is keyed on
    # the task's row count, and an Ordered tree runs tasks at every fold's
    # prefix length and then at `n_rows`, so a single pool rebuilt its
    # buffers (and dropped its oracle scratch) on EVERY task -- measured
    # as most of an Ordered tree's wall at 320 rows. Slot `f` serves fold
    # `f`'s tasks (the same size for every learn permutation, and under
    # the sabotage arm too) and slot `n_folds` the estimation task, so
    # each slot sees one key for the whole fit: exactly the reuse the
    # Plain fit already makes of its one slot, under the same contract
    # (every cell a task reads is written by that task's gathers; the
    # previous task's consumers drained at its tail).
    # the per-tree host copy of the bins and the partition's upload staging
    # (every task drains at its tail, so one staging buffer serves them all)
    var h_tree_bins = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    var h_part_rows = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    var est_pools = List[List[TEstimationWorkspace]]()
    for _ in range(n_folds + 1):
        est_pools.append(List[TEstimationWorkspace]())
    var losses = List[Float64]()
    var fv_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var fv_part = ctx.enqueue_create_buffer[DType.float32](fv_blocks)
    var fv = ctx.enqueue_create_buffer[DType.float32](1)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](1)
    var dummy_mag = ctx.enqueue_create_buffer[DType.float32](2)
    var loss_stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var has_test = test.n_rows > 0
    if has_test:
        enqueue_fill(ctx, test.cursor, opts.start_value)
    var detector = make_overfitting_detector(
        od_type, False, od_pvalue, od_wait, has_test
    )
    var test_losses = List[Float64]()
    var stopped_early = False
    ctx.synchronize()
    _ = hq^
    # `MOJOLEARN_STAGE_TIMES=1`: the per-stage triage table (drains per
    # stage, NOT a benchmark); one Bool test per stage when unset
    var times = StageTimes()
    # the structure search's own per-level table rides the same clock only
    # when it is on; otherwise the searcher gets a disabled one
    var no_trace = IdentityTrace.disabled()
    var est_times = StageTimes()
    var walker_times = StageTimes()

    for iteration in range(n_estimators):
        var tag = String("ordered.") + String(iteration)
        # 1. the learn permutation the structure is searched on
        var learn_p = 0
        if learn_count > 1:
            learn_p = Int(rng.next_uniform_l() % UInt64(learn_count - 1))
        var tree_seed = rng.next_uniform_l()

        # 2. the fold derivatives, concatenated
        times.begin(ctx)
        var sw = ctx.enqueue_create_buffer[DType.float32](total)
        var sg = ctx.enqueue_create_buffer[DType.float32](total)
        for f in range(n_folds):
            var r = folds[f].quality_evaluate_samples.right
            var gy = ctx.enqueue_create_buffer[DType.float32](r)
            var gw = ctx.enqueue_create_buffer[DType.float32](r)
            ctx.enqueue_function[_ord_gather_kernel](
                targets.unsafe_ptr(), dperms[learn_p].unsafe_ptr(),
                gy.unsafe_ptr(), Int32(r), grid_dim=(_grid(r), 1, 1),
                block_dim=(ORDERED_BLOCK, 1, 1),
            )
            ctx.enqueue_function[_ord_gather_kernel](
                weights.unsafe_ptr(), dperms[learn_p].unsafe_ptr(),
                gw.unsafe_ptr(), Int32(r), grid_dim=(_grid(r), 1, 1),
                block_dim=(ORDERED_BLOCK, 1, 1),
            )
            var stats = ctx.enqueue_create_buffer[DType.float32](2 * r)
            var blocks = (r + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
            var part = ctx.enqueue_create_buffer[DType.float32](blocks)
            if second_order:
                launch_approximate[False, True](
                    ctx, opts.objective, gy, gw, Int32(r),
                    cursors[learn_p][f], Int32(1), opts.kernel_alpha,
                    opts.logloss_border, stats, part, Int32(0),
                    dummy_mag, Int32(0), blocks,
                )
            else:
                launch_approximate[False](
                    ctx, opts.objective, gy, gw, Int32(r),
                    cursors[learn_p][f], Int32(1), opts.kernel_alpha,
                    opts.logloss_border, stats, part, Int32(0),
                    dummy_mag, Int32(0), blocks,
                )
            ctx.enqueue_function[_ord_scatter_planes_kernel](
                stats.unsafe_ptr(), sw.unsafe_ptr(), sg.unsafe_ptr(),
                Int32(r), Int32(offsets[f]), grid_dim=(_grid(r), 1, 1),
                block_dim=(ORDERED_BLOCK, 1, 1),
            )
            ctx.synchronize()
            _ = gy^
            _ = gw^
            _ = stats^
            _ = part^

        times.end(ctx, "ord.derivatives")
        # 3. the score noise, from the UNBOOTSTRAPPED quality slices
        times.begin(ctx)
        var score_std = Float32(0.0)
        if opts.random_strength != Float32(0.0):
            var terms = ctx.enqueue_create_buffer[DType.float32](total)
            ctx.enqueue_function[_ord_std_terms_kernel](
                sw.unsafe_ptr(), sg.unsafe_ptr(), quality.unsafe_ptr(),
                terms.unsafe_ptr(), Int32(total),
                grid_dim=(_grid(total), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
            )
            var s2 = ctx.enqueue_create_buffer[DType.float32](1)
            ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
                terms.unsafe_ptr(), Int32(total), s2.unsafe_ptr(),
                grid_dim=1, block_dim=256,
            )
            var hs = ctx.enqueue_create_host_buffer[DType.float32](1)
            ctx.enqueue_copy(dst_buf=hs, src_buf=s2)
            ctx.synchronize()
            var count = 0
            for f in range(n_folds):
                count += (
                    folds[f].quality_evaluate_samples.right
                    - folds[f].estimate_samples.right
                )
            var mult = ordered_model_length_mult(
                n_rows, Float64(iteration) * Float64(opts.learning_rate)
            )
            score_std = Float32(
                mult
                * sqrt(Float64(hs[0]) / (Float64(count) + 1e-100))
                * Float64(opts.random_strength)
            )
            _ = terms^
            _ = s2^
            _ = hs^

        times.end(ctx, "ord.score_std")
        # 4. the bootstrap, quality slices only
        times.begin(ctx)
        if bootstrap_on:
            var draws = ctx.enqueue_create_buffer[DType.float32](total)
            enqueue_fill(ctx, draws, Float32(1.0))
            launch_bootstrap(
                ctx, opts.bootstrap_kind, boot_seeds, draws, total,
                opts.bootstrap_param, boot_mags, False, 1,
            )
            ctx.enqueue_function[_ord_bootstrap_apply_kernel](
                sw.unsafe_ptr(), sg.unsafe_ptr(), draws.unsafe_ptr(),
                quality.unsafe_ptr(), Int32(total),
                grid_dim=(_grid(total), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
            )
            ctx.synchronize()
            _ = draws^

        times.end(ctx, "ord.bootstrap")
        # the fixed-point scale from the planes as the searcher reads them
        times.begin(ctx)
        var absv = ctx.enqueue_create_buffer[DType.float32](2 * total)
        ctx.enqueue_function[_ord_abs_planes_kernel](
            sw.unsafe_ptr(), sg.unsafe_ptr(), absv.unsafe_ptr(), Int32(total),
            grid_dim=(_grid(total), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
        )
        var mags = ctx.enqueue_create_buffer[DType.float32](2)
        ctx.enqueue_function[deterministic_sum_lanes_kernel[2]](
            absv.unsafe_ptr(), Int32(total), mags.unsafe_ptr(),
            grid_dim=1, block_dim=256,
        )
        var hm = ctx.enqueue_create_host_buffer[DType.float32](2)
        ctx.enqueue_copy(dst_buf=hm, src_buf=mags)
        ctx.synchronize()
        var m0 = Float64(hm[0])
        var m1 = Float64(hm[1])
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, total))
        trace.record_scalar_f32(tag + ".scale", scale)
        trace.record_scalar_f32(tag + ".score_std", score_std)
        _ = absv^
        _ = mags^
        _ = hm^

        times.end(ctx, "ord.scale")
        # 5. the structure, on the learn permutation's folds
        times.begin(ctx)
        var splits = fit_oblivious_tree_structure_traced(
            ctx, layout, n_rows, max_depth, cindex, sw^, sg^, sm_count,
            scale, opts.score_function, pool, no_trace, times,
            String("tree"), opts.l2_leaf_reg,
            score_std_dev=score_std, seed=tree_seed, one_hot=one_hot,
            folds=folds, permutation=perms[learn_p],
        )
        times.end(ctx, "ord.structure")
        times.begin(ctx)
        var bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        if len(splits) == 0:
            enqueue_fill(ctx, bins, UInt32(0))
        else:
            compute_bins_for_model(
                ctx, layout, splits, len(splits), cindex, n_rows, bins
            )
        var n_leaves = 1 << len(splits)
        # the tree's bins on the host ONCE, for every task's partition
        # (`_partition_from_host_bins`)
        ctx.enqueue_copy(dst_buf=h_tree_bins, src_buf=bins)
        ctx.synchronize()

        times.end(ctx, "ord.bins")
        # 6. the fold models, then the estimation model
        times.begin(ctx)
        for lp in range(learn_count):
            for f in range(n_folds):
                var est = folds[f].estimate_samples.right
                comptime if ORDERED_SABOTAGE:
                    est = folds[f].quality_evaluate_samples.right
                _ = _ordered_estimate_task(
                    ctx, est, folds[f].quality_evaluate_samples.right,
                    n_leaves, targets, weights, dperms[lp], perms[lp], bins,
                    h_tree_bins, h_part_rows, cursors[lp][f], opts, sm_count, est_pools[f], trace,
                    tag + ".perm." + String(lp) + ".fold." + String(f),
                    est_times, walker_times,
                )
        times.end(ctx, "ord.fold_estimates")
        times.begin(ctx)
        var leaves = _ordered_estimate_task(
            ctx, n_rows, n_rows, n_leaves, targets, weights, dperms[est_p],
            perms[est_p], bins, h_tree_bins, h_part_rows, est_cursor, opts, sm_count, est_pools[n_folds], trace,
            tag + ".estimation", est_times, walker_times,
        )
        var structure = TObliviousTreeStructure()
        structure.splits = splits^
        var weak = TObliviousTreeModel(structure^)
        for leaf in range(n_leaves):
            weak.leaf_values.append(identical_mul(leaves[leaf], opts.learning_rate))
        model.add_weak_model(weak^)
        trace.record_device(ctx, tag + ".estimation_cursor", est_cursor)

        times.end(ctx, "ord.estimation_estimate")
        # 7. the learn loss at the estimation cursor
        times.begin(ctx)
        launch_approximate[False](
            ctx, opts.objective, est_y, est_w, Int32(n_rows), est_cursor,
            Int32(1), opts.kernel_alpha, opts.logloss_border, loss_stats,
            fv_part, Int32(1), dummy_mag, Int32(0), fv_blocks,
        )
        ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
            fv_part.unsafe_ptr(), Int32(fv_blocks), fv.unsafe_ptr(),
            grid_dim=1, block_dim=256,
        )
        ctx.enqueue_copy(dst_buf=h_fv, src_buf=fv)
        ctx.synchronize()
        losses.append(-Float64(h_fv[0]) / Float64(n_rows))
        times.end(ctx, "ord.learn_loss")
        _ = bins^
        # 8. the held-out cursor, its loss, the detector
        if has_test:
            _apply_last_tree_to_test(
                ctx, model, layout, test, 1, opts.learning_rate
            )
            var t_loss = _test_loss(
                ctx, test, opts.objective, 0, opts.kernel_alpha,
                opts.logloss_border, 1,
            )
            test_losses.append(t_loss)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                stopped_early = True
                break
    ctx.synchronize()
    times.report("ordered boosting")
    est_times.report("ordered estimation tasks")
    walker_times.report("ordered estimate_and_apply internals")
    _ = quality^
    _ = h_tree_bins^
    _ = h_part_rows^
    _ = boot_seeds^
    _ = boot_mags^
    _ = est_y^
    _ = est_w^
    _ = fv_part^
    _ = fv^
    _ = h_fv^
    _ = dummy_mag^
    _ = loss_stats^
    # the ERROR tracker's best: the held-out curve's with a test set, else
    # the first strict minimum of the learn curve (`error_tracker.h:58-64`)
    var best = 0
    if has_test:
        best = detector.best_iteration
    else:
        for i in range(1, len(losses)):
            if losses[i] < losses[best]:
                best = i
    return OrderedFitOutput(losses^, test_losses^, best, stopped_early)
