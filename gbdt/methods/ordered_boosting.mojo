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
from std.os import getenv
from std.gpu import block_idx, block_dim, thread_idx
from std.math import isfinite, sqrt
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from checks.fixed_point import choose_scale
from checks.numerics import ftz, identical_mul, identical_mul64
from gbdt.data.ordered_plan import (
    ORDERED_BOOTSTRAP_SALT,
    ORDERED_MIN_FOLD_SIZE,
    ORDERED_STREAM_SALT,
    ordered_model_length_mult,
    ordered_permutation_block_size,
    ordered_permutations,
)
from gbdt.data.permutation import TRandom
from gbdt.gpu_util.arena import BufferArena
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.gpu_util.kernel.bootstrap import (
    bootstrap_grid_blocks,
    create_bootstrap_seeds,
    launch_bootstrap,
)
from gbdt.methods.doc_parallel_boosting import (
    PendingEstimation,
    TEstimationWorkspace,
    _estimate_complete,
    _estimate_prepare,
    estimate_advance,
    estimate_can_batch,
    estimate_workspace,
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
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import enqueue_snap_plane
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
    REDUCE_LANES_BLOCK,
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


def _ord_std_and_mags_kernel(
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    quality: MutPointer[UInt32, MutAnyOrigin],
    total_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """ONE block of `REDUCE_LANES_BLOCK`: `dst[0]` the score-noise sum of
    `_ord_std_terms_kernel`'s terms, `dst[1]`/`dst[2]` the sums of `|sw|` and
    `|sg|` (`_ord_abs_planes_kernel`), each folded EXACTLY as
    `deterministic_sum_lanes_kernel` folds it -- thread `t` adds positions
    `t, t + 256, ...` in ascending order, then the same shared tree -- so the
    three sums are the bits of the two separate passes, from one read of the
    planes and without their two scratch arrays (lane/ordered-speed). Only
    valid when the planes are not bootstrapped between the two (the noise is
    taken BEFORE the bootstrap and the scale AFTER it)."""
    var tid = Int(thread_idx.x)
    var total = Int(total_in)
    var a0 = Float32(0.0)
    var a1 = Float32(0.0)
    var a2 = Float32(0.0)
    var i = tid
    # four strides' loads issued together, then added IN ORDER: the same
    # sequence of adds as the one-stride loop, more loads in flight (the
    # block is one threadgroup, so its latency is the whole kernel's)
    comptime STEP = 4 * REDUCE_LANES_BLOCK
    while i + 3 * REDUCE_LANES_BLOCK < total:
        var w = SIMD[DType.float32, 4]()
        var g = SIMD[DType.float32, 4]()
        var qm = SIMD[DType.uint32, 4]()
        comptime for k in range(4):
            w[k] = sw.unsafe_load(i + k * REDUCE_LANES_BLOCK)
            g[k] = sg.unsafe_load(i + k * REDUCE_LANES_BLOCK)
            qm[k] = quality.unsafe_load(i + k * REDUCE_LANES_BLOCK)
        comptime for k in range(4):
            var term = Float32(0.0)
            if qm[k] != UInt32(0):
                if w[k] > Float32(0.0):
                    var q = ftz(g[k] / w[k])
                    term = ftz(ftz(q * q) * w[k])
            a0 += term
            a1 += abs(w[k])
            a2 += abs(g[k])
        i += STEP
    while i < total:
        var w = sw.unsafe_load(i)
        var g = sg.unsafe_load(i)
        var term = Float32(0.0)
        if quality.unsafe_load(i) != UInt32(0):
            if w > Float32(0.0):
                var q = ftz(g / w)
                term = ftz(ftz(q * q) * w)
        a0 += term
        a1 += abs(w)
        a2 += abs(g)
        i += REDUCE_LANES_BLOCK
    var red = stack_allocation[
        3 * REDUCE_LANES_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = a0
    red[REDUCE_LANES_BLOCK + tid] = a1
    red[2 * REDUCE_LANES_BLOCK + tid] = a2
    barrier()
    var step = REDUCE_LANES_BLOCK // 2
    while step > 0:
        if tid < step:
            comptime for lane in range(3):
                red[lane * REDUCE_LANES_BLOCK + tid] = (
                    red[lane * REDUCE_LANES_BLOCK + tid]
                    + red[lane * REDUCE_LANES_BLOCK + tid + step]
                )
        barrier()
        step //= 2
    if tid == 0:
        comptime for lane in range(3):
            dst.unsafe_store(lane, red[lane * REDUCE_LANES_BLOCK])


def _ord_std_lanes_kernel(
    sw: MutPointer[Float32, MutAnyOrigin],
    sg: MutPointer[Float32, MutAnyOrigin],
    quality: MutPointer[UInt32, MutAnyOrigin],
    total_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`_ord_std_and_mags_kernel`'s lane chains, spread over blocks
    (lane/neural-pass122): lane t = block * block_dim + thread runs the same
    loop over positions t, t + 256, ... and stores its three sums at
    dst[t], dst[256 + t], dst[512 + t]; `_ord_std_combine_kernel` folds
    them with the same shared tree. The one block streamed every plane
    through one SM (L40S taxi 4.1M: 13 ms a tree, the largest kernel)."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = Int(total_in)
    var a0 = Float32(0.0)
    var a1 = Float32(0.0)
    var a2 = Float32(0.0)
    var i = tid
    # ORD_STD_LANE_STRIDES strides' loads in flight, the NEXT batch issued
    # before this one is folded (software pipelining; the adds keep the
    # one-stride loop's order). One warp per SM: its chain waited on DRAM
    # every few positions (peer's nsys on #122: 13 ms a tree unchanged).
    comptime SU = ORD_STD_LANE_STRIDES
    comptime STEP = SU * REDUCE_LANES_BLOCK
    var w = SIMD[DType.float32, SU]()
    var g = SIMD[DType.float32, SU]()
    var qm = SIMD[DType.uint32, SU]()
    var have = i + (SU - 1) * REDUCE_LANES_BLOCK < total
    if have:
        comptime for k in range(SU):
            w[k] = sw.unsafe_load(i + k * REDUCE_LANES_BLOCK)
            g[k] = sg.unsafe_load(i + k * REDUCE_LANES_BLOCK)
            qm[k] = quality.unsafe_load(i + k * REDUCE_LANES_BLOCK)
    while have:
        var nxt = i + STEP
        var have_next = nxt + (SU - 1) * REDUCE_LANES_BLOCK < total
        var nw = SIMD[DType.float32, SU]()
        var ng = SIMD[DType.float32, SU]()
        var nq = SIMD[DType.uint32, SU]()
        if have_next:
            comptime for k in range(SU):
                nw[k] = sw.unsafe_load(nxt + k * REDUCE_LANES_BLOCK)
                ng[k] = sg.unsafe_load(nxt + k * REDUCE_LANES_BLOCK)
                nq[k] = quality.unsafe_load(nxt + k * REDUCE_LANES_BLOCK)
        comptime for k in range(SU):
            var term = Float32(0.0)
            if qm[k] != UInt32(0):
                if w[k] > Float32(0.0):
                    var q = ftz(g[k] / w[k])
                    term = ftz(ftz(q * q) * w[k])
            a0 += term
            a1 += abs(w[k])
            a2 += abs(g[k])
        w = nw
        g = ng
        qm = nq
        i = nxt
        have = have_next
    while i < total:
        var w = sw.unsafe_load(i)
        var g = sg.unsafe_load(i)
        var term = Float32(0.0)
        if quality.unsafe_load(i) != UInt32(0):
            if w > Float32(0.0):
                var q = ftz(g / w)
                term = ftz(ftz(q * q) * w)
        a0 += term
        a1 += abs(w)
        a2 += abs(g)
        i += REDUCE_LANES_BLOCK
    dst.unsafe_store(tid, a0)
    dst.unsafe_store(REDUCE_LANES_BLOCK + tid, a1)
    dst.unsafe_store(2 * REDUCE_LANES_BLOCK + tid, a2)


def _ord_std_combine_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """ONE block of `REDUCE_LANES_BLOCK`: `_ord_std_and_mags_kernel`'s
    shared tree over the lanes' sums from `_ord_std_lanes_kernel`."""
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        3 * REDUCE_LANES_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    comptime for lane in range(3):
        red[lane * REDUCE_LANES_BLOCK + tid] = part.unsafe_load(lane * REDUCE_LANES_BLOCK + tid)
    barrier()
    var step = REDUCE_LANES_BLOCK // 2
    while step > 0:
        if tid < step:
            comptime for lane in range(3):
                red[lane * REDUCE_LANES_BLOCK + tid] = (
                    red[lane * REDUCE_LANES_BLOCK + tid]
                    + red[lane * REDUCE_LANES_BLOCK + tid + step]
                )
        barrier()
        step //= 2
    if tid == 0:
        comptime for lane in range(3):
            dst.unsafe_store(lane, red[lane * REDUCE_LANES_BLOCK])


#: Lanes per block of `_ord_std_lanes_kernel` (one warp: its loads of
#: positions t .. t + 31 are one contiguous 128-byte line per plane).
comptime ORD_STD_LANES = 32
#: Strides a lane of `_ord_std_lanes_kernel` keeps in flight.
comptime ORD_STD_LANE_STRIDES = 16


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


def _ord_stage_in_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    permutation: MutPointer[UInt32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    out_y: MutPointer[Float32, MutAnyOrigin],
    out_w: MutPointer[Float32, MutAnyOrigin],
    out_c: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """`_ordered_gather_kernel` into permutation order, then the
    estimator's three bin-order gathers through `row_index`
    (`_estimate_and_apply`'s stage-in), in one pass: position `i` of the
    estimator's arrays is permutation position `j = row_index[i]`, so
    `out_y[i] = y[perm[j]]`, `out_w[i] = weights[perm[j]]`,
    `out_c[i] = cursor[j]` -- the same loads and stores of the same values,
    without the intermediate arrays."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        var j = Int(row_index.unsafe_load(i))
        var row = Int(permutation.unsafe_load(j))
        out_y.unsafe_store(i, y.unsafe_load(row))
        out_w.unsafe_store(i, weights.unsafe_load(row))
        out_c.unsafe_store(i, cursor.unsafe_load(j))


def _ord_leaf_gather_kernel(
    bins: MutPointer[UInt32, MutAnyOrigin],
    perm: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`dst[r] = bins[perm[r]]`: permutation position `r`'s leaf, the value
    `_partition_into` read on the host through `tree_bins[permutation[r]]`
    (scalar u32 loads and stores; no vector load)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        dst.unsafe_store(i, bins.unsafe_load(Int(perm.unsafe_load(i))))


def _ord_segment_rows_kernel(
    sorted_rows: MutPointer[UInt32, MutAnyOrigin],
    table: MutPointer[UInt32, MutAnyOrigin],
    n_leaves_in: Int32,
    row_index: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """One task's partition from its permutation's FULL stable partition
    (lane/ordered-speed). `table` holds, per leaf `b`, the task's offset
    `table[3b]`, its size `table[3b + 1]` and the leaf's segment start in
    `sorted_rows` `table[3b + 2]`. Output position `i` lies in the leaf `c`
    that is the LAST with `offset <= i` (an empty leaf before it shares its
    offset; every leaf after it starts past `i`), and takes the
    `(i - offset_c)`-th row of `c`'s segment: the rows of a leaf are in
    ascending permutation position there, so a task over the prefix
    `[0, E)` owns exactly the first `size_c` of them, in the order the
    stable counting sort of `[0, E)` wrote them. Scalar u32 loads only."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        var lo = 0
        var hi = Int(n_leaves_in) - 1
        while lo < hi:
            var mid = (lo + hi + 1) >> 1
            if Int(table.unsafe_load(3 * mid)) <= i:
                lo = mid
            else:
                hi = mid - 1
        var off = Int(table.unsafe_load(3 * lo))
        var seg = Int(table.unsafe_load(3 * lo + 2))
        row_index.unsafe_store(i, sorted_rows.unsafe_load(seg + i - off))


def _ord_stage_in_p_kernel(
    y_p: MutPointer[Float32, MutAnyOrigin],
    w_p: MutPointer[Float32, MutAnyOrigin],
    cursor: MutPointer[Float32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    out_y: MutPointer[Float32, MutAnyOrigin],
    out_w: MutPointer[Float32, MutAnyOrigin],
    out_c: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """`_ord_stage_in_kernel` from the permutation's targets and weights
    already in permutation order (`y_p[j] = y[perm[j]]`, gathered once per
    fit): the same values, one indirection fewer."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        var j = Int(row_index.unsafe_load(i))
        out_y.unsafe_store(i, y_p.unsafe_load(j))
        out_w.unsafe_store(i, w_p.unsafe_load(j))
        out_c.unsafe_store(i, cursor.unsafe_load(j))


def _grid(n: Int) -> Int:
    return (n + ORDERED_BLOCK - 1) // ORDERED_BLOCK


# ===========================================================================
# ONE ESTIMATION TASK
# ===========================================================================


def _ord_chunk_count_kernel(
    leaf: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[UInt32, MutAnyOrigin],
    starts: MutPointer[UInt32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_leaves_in: Int32,
):
    """Chunk `t`'s per-leaf row counts, `counts[b * n_chunks + t]`, one
    thread per chunk walking its rows in order (no atomics: a column is
    one thread's). A leaf id out of range is skipped here and caught by
    the host's total check."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nc = Int(n_chunks_in)
    if t < nc:
        var nl = Int(n_leaves_in)
        for b in range(nl):
            counts.unsafe_store(b * nc + t, UInt32(0))
        var lo = Int(starts.unsafe_load(t))
        var hi = Int(starts.unsafe_load(t + 1))
        for r in range(lo, hi):
            var b = Int(leaf.unsafe_load(r))
            if b < nl:
                var at = b * nc + t
                counts.unsafe_store(at, counts.unsafe_load(at) + 1)


def _ord_leaf_scan_kernel(
    counts: MutPointer[UInt32, MutAnyOrigin],
    prefix: MutPointer[UInt32, MutAnyOrigin],
    totals: MutPointer[UInt32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_leaves_in: Int32,
):
    """Per leaf `b` (one thread), the exclusive prefix of its chunk counts
    in chunk order, and its total. Integer sums: order-free."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b < Int(n_leaves_in):
        var nc = Int(n_chunks_in)
        var run = UInt32(0)
        for t in range(nc):
            prefix.unsafe_store(b * nc + t, run)
            run += counts.unsafe_load(b * nc + t)
        totals.unsafe_store(b, run)



#: Threads of `_ord_leaf_scan_block_kernel`'s block (one block a leaf).
comptime ORD_SCAN_TPB = 256


def _ord_leaf_scan_block_kernel(
    counts: MutPointer[UInt32, MutAnyOrigin],
    prefix: MutPointer[UInt32, MutAnyOrigin],
    totals: MutPointer[UInt32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_leaves_in: Int32,
):
    """`_ord_leaf_scan_kernel` with ONE BLOCK per leaf (lane/neural-pass125):
    thread i scans a contiguous run of the leaf's chunk counts, a shared
    scan of the runs' totals gives each run its start, and the run writes
    its exclusive prefixes. Integer sums: the same prefixes. The one-thread
    form walked ~3,200 chunks per leaf on ONE block of 64 threads (L40S
    taxi 4.1M: 0.75 ms a launch, 4 a tree)."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nc = Int(n_chunks_in)
    if b >= Int(n_leaves_in):
        return
    var per = (nc + ORD_SCAN_TPB - 1) // ORD_SCAN_TPB
    var lo = min(nc, tid * per)
    var hi = min(nc, lo + per)
    var row = b * nc
    var s = UInt32(0)
    for t in range(lo, hi):
        s += counts.unsafe_load(row + t)
    var sh = stack_allocation[ORD_SCAN_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    sh[tid] = s
    barrier()
    var off = 1
    while off < ORD_SCAN_TPB:
        var v = sh[tid] + (sh[tid - off] if tid >= off else UInt32(0))
        barrier()
        sh[tid] = v
        barrier()
        off *= 2
    var run = sh[tid] - s
    for t in range(lo, hi):
        prefix.unsafe_store(row + t, run)
        run += counts.unsafe_load(row + t)
    if tid == ORD_SCAN_TPB - 1:
        totals.unsafe_store(b, sh[tid])


def _ord_seg_start_kernel(
    totals: MutPointer[UInt32, MutAnyOrigin],
    seg_start: MutPointer[UInt32, MutAnyOrigin],
    n_leaves_in: Int32,
):
    """One thread: the leaves' run starts, the exclusive scan of totals."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var run = UInt32(0)
        for b in range(Int(n_leaves_in)):
            seg_start.unsafe_store(b, run)
            run += totals.unsafe_load(b)


def _ord_snap_kernel(
    prefix: MutPointer[UInt32, MutAnyOrigin],
    totals: MutPointer[UInt32, MutAnyOrigin],
    bound_chunk: MutPointer[UInt32, MutAnyOrigin],
    snap: MutPointer[UInt32, MutAnyOrigin],
    n_bounds_in: Int32,
    n_chunks_in: Int32,
    n_leaves_in: Int32,
):
    """`snap[k * n_leaves + b]`: rows of leaf `b` below the `k`-th task
    boundary. Every boundary is a chunk start (`_PermPartition`), chunk
    `bound_chunk[k]`, so it is that chunk's exclusive prefix, or the
    leaf's total at `n_chunks` (the boundary `need`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nl = Int(n_leaves_in)
    if i < Int(n_bounds_in) * nl:
        var k = i // nl
        var b = i - k * nl
        var nc = Int(n_chunks_in)
        var c = Int(bound_chunk.unsafe_load(k))
        if c >= nc:
            snap.unsafe_store(i, totals.unsafe_load(b))
        else:
            snap.unsafe_store(i, prefix.unsafe_load(b * nc + c))


def _ord_chunk_scatter_kernel(
    leaf: MutPointer[UInt32, MutAnyOrigin],
    prefix: MutPointer[UInt32, MutAnyOrigin],
    seg_start: MutPointer[UInt32, MutAnyOrigin],
    sorted_rows: MutPointer[UInt32, MutAnyOrigin],
    starts: MutPointer[UInt32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_leaves_in: Int32,
):
    """The stable scatter: chunk `t`'s rows, in order, to
    `seg_start[b] + prefix[b][t]` and on (the column is advanced in place;
    `_ord_snap_kernel` has read it first). Chunks are in row order and a
    chunk's rows are walked in order, so a leaf's run lists its rows
    ascending: the stable counting sort, the same list of integers."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nc = Int(n_chunks_in)
    if t < nc:
        var nl = Int(n_leaves_in)
        var lo = Int(starts.unsafe_load(t))
        var hi = Int(starts.unsafe_load(t + 1))
        for r in range(lo, hi):
            var b = Int(leaf.unsafe_load(r))
            if b < nl:
                var at = b * nc + t
                var slot = prefix.unsafe_load(at)
                prefix.unsafe_store(at, slot + 1)
                sorted_rows.unsafe_store(
                    Int(seg_start.unsafe_load(b) + slot), UInt32(r)
                )


#: the per-(leaf, chunk) count matrix is capped at this many cells; the
#: chunk grows to fit (depth 16 has 65,536 leaves)
comptime ORDERED_PART_CELLS = 1 << 22
# (lane/neural-pass125: 256, was 1024 -- the count and scatter kernels
# run one thread a chunk; the stable sort's output does not depend on it)
comptime ORDERED_PART_CHUNK = 256


struct _PermPartition(Movable):
    """One permutation's STABLE partition of its positions `[0, need)` by
    the tree's leaf, rebuilt each tree on the device (lane/ordered-speed):
    `d_sorted` lists the positions leaf by leaf, ascending within a leaf;
    `seg_start[b]` is where leaf `b`'s run begins; `snap[k * n_leaves + b]`
    counts leaf `b` among the positions below the `k`-th task boundary
    `bounds[k]` (a task over the prefix `[0, E)` reads the snapshot of
    `E`). Replaces a host counting sort per TASK (two random-access host
    passes over each prefix: 6.8 s of 13.5 s for 10 trees at 4.1M rows on
    the M2 Pro) with one device counting sort per permutation; only the
    counts come home."""

    var need: Int
    var bounds: List[Int]
    var n_chunks: Int
    #: chunk `t` is rows `[starts[t], starts[t + 1])`; every boundary is a start
    var d_starts: DeviceBuffer[DType.uint32]
    #: boundary `k`'s chunk index (`n_chunks` for `need`)
    var d_bound_chunk: DeviceBuffer[DType.uint32]
    var d_leaf: DeviceBuffer[DType.uint32]
    var d_sorted: DeviceBuffer[DType.uint32]
    var d_seg: DeviceBuffer[DType.uint32]
    var d_tot: DeviceBuffer[DType.uint32]
    var d_snap: DeviceBuffer[DType.uint32]
    var h_seg: HostBuffer[DType.uint32]
    var h_tot: HostBuffer[DType.uint32]
    var h_snap: HostBuffer[DType.uint32]
    var seg_start: List[Int]
    var snap: List[Int]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut arena: BufferArena,
        var bounds: List[Int],
        leaf_capacity: Int,
    ) raises:
        var need = 0
        for i in range(len(bounds)):
            if bounds[i] > need:
                need = bounds[i]
        # ascending, unique (a handful of fold prefixes)
        var uniq = List[Int]()
        for i in range(len(bounds)):
            var seen = False
            for j in range(len(uniq)):
                if uniq[j] == bounds[i]:
                    seen = True
            if not seen:
                var at = len(uniq)
                uniq.append(bounds[i])
                while at > 0 and uniq[at - 1] > uniq[at]:
                    var t = uniq[at - 1]
                    uniq[at - 1] = uniq[at]
                    uniq[at] = t
                    at -= 1
        self.need = need
        var nb = len(uniq)
        # chunks of ORDERED_PART_CHUNK rows (grown so the (leaf, chunk)
        # matrix fits ORDERED_PART_CELLS at the fit's leaf capacity), cut
        # again at every boundary so a snapshot is a chunk prefix
        var chunk = ORDERED_PART_CHUNK
        var cap = ORDERED_PART_CELLS // leaf_capacity - nb
        if cap < 1:
            cap = 1
        if (need + chunk - 1) // chunk > cap:
            chunk = (need + cap - 1) // cap
        var starts = List[Int]()
        var bound_chunk = List[Int]()
        var r = 0
        var k = 0
        while r < need:
            while k < nb and uniq[k] <= r:
                bound_chunk.append(len(starts))
                k += 1
            starts.append(r)
            var nxt = r + chunk
            if k < nb and uniq[k] < nxt:
                nxt = uniq[k]
            if nxt > need:
                nxt = need
            r = nxt
        while k < nb:
            bound_chunk.append(len(starts))
            k += 1
        self.n_chunks = len(starts)
        starts.append(need)
        self.d_starts = arena.device[DType.uint32](ctx, len(starts))
        self.d_bound_chunk = arena.device[DType.uint32](ctx, nb)
        var hs = ctx.enqueue_create_host_buffer[DType.uint32](len(starts))
        for i in range(len(starts)):
            hs.unsafe_ptr().unsafe_store(i, UInt32(starts[i]))
        var hb = ctx.enqueue_create_host_buffer[DType.uint32](nb)
        for i in range(nb):
            hb.unsafe_ptr().unsafe_store(i, UInt32(bound_chunk[i]))
        ctx.enqueue_copy(dst_buf=self.d_starts, src_ptr=hs.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_bound_chunk, src_ptr=hb.unsafe_ptr())
        ctx.synchronize()
        _ = hs^
        _ = hb^
        self.bounds = uniq^
        self.d_leaf = arena.device[DType.uint32](ctx, need)
        self.d_sorted = arena.device[DType.uint32](ctx, need)
        self.d_seg = arena.device[DType.uint32](ctx, leaf_capacity)
        self.d_tot = arena.device[DType.uint32](ctx, leaf_capacity)
        self.d_snap = arena.device[DType.uint32](ctx, nb * leaf_capacity)
        self.h_seg = arena.host_buffer[DType.uint32](ctx, leaf_capacity)
        self.h_tot = arena.host_buffer[DType.uint32](ctx, leaf_capacity)
        self.h_snap = arena.host_buffer[DType.uint32](ctx, nb * leaf_capacity)
        self.seg_start = List[Int]()
        self.snap = List[Int]()

    def enqueue_sort(
        mut self,
        ctx: DeviceContext,
        mut bins: DeviceBuffer[DType.uint32],
        mut dperm: DeviceBuffer[DType.uint32],
        n_leaves: Int,
        mut counts: DeviceBuffer[DType.uint32],
        mut prefix: DeviceBuffer[DType.uint32],
    ) raises:
        """The leaves in permutation order, the chunked counting sort, the
        boundary snapshots, and the counts home, all enqueued (the caller
        drains once for every permutation, then calls `settle`). `counts`
        and `prefix` are the caller's scratch of `ORDERED_PART_CELLS`,
        shared by the permutations: the queue orders one permutation's
        reads before the next one's writes."""
        var need = self.need
        ctx.enqueue_function[_ord_leaf_gather_kernel](
            bins.unsafe_ptr(), dperm.unsafe_ptr(), self.d_leaf.unsafe_ptr(),
            Int32(need), grid_dim=(_grid(need), 1, 1),
            block_dim=(ORDERED_BLOCK, 1, 1),
        )
        var n_chunks = self.n_chunks
        if n_chunks * n_leaves > ORDERED_PART_CELLS:
            raise Error("ordered partition: chunk matrix over its capacity")
        ctx.enqueue_function[_ord_chunk_count_kernel](
            self.d_leaf.unsafe_ptr(), counts.unsafe_ptr(),
            self.d_starts.unsafe_ptr(), Int32(n_chunks), Int32(n_leaves),
            grid_dim=(_grid(n_chunks), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
        )
        ctx.enqueue_function[_ord_leaf_scan_block_kernel](
            counts.unsafe_ptr(), prefix.unsafe_ptr(), self.d_tot.unsafe_ptr(),
            Int32(n_chunks), Int32(n_leaves),
            grid_dim=(n_leaves, 1, 1), block_dim=(ORD_SCAN_TPB, 1, 1),
        )
        ctx.enqueue_function[_ord_seg_start_kernel](
            self.d_tot.unsafe_ptr(), self.d_seg.unsafe_ptr(), Int32(n_leaves),
            grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
        )
        var nb = len(self.bounds)
        ctx.enqueue_function[_ord_snap_kernel](
            prefix.unsafe_ptr(), self.d_tot.unsafe_ptr(),
            self.d_bound_chunk.unsafe_ptr(), self.d_snap.unsafe_ptr(),
            Int32(nb), Int32(n_chunks), Int32(n_leaves),
            grid_dim=(_grid(nb * n_leaves), 1, 1),
            block_dim=(ORDERED_BLOCK, 1, 1),
        )
        ctx.enqueue_function[_ord_chunk_scatter_kernel](
            self.d_leaf.unsafe_ptr(), prefix.unsafe_ptr(),
            self.d_seg.unsafe_ptr(), self.d_sorted.unsafe_ptr(),
            self.d_starts.unsafe_ptr(), Int32(n_chunks), Int32(n_leaves),
            grid_dim=(_grid(n_chunks), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=self.h_seg, src_buf=self.d_seg)
        ctx.enqueue_copy(dst_buf=self.h_tot, src_buf=self.d_tot)
        ctx.enqueue_copy(dst_buf=self.h_snap, src_buf=self.d_snap)

    def settle(mut self, n_leaves: Int) raises:
        """After the caller's drain: the counts onto the host lists, and
        the check the host sort made row by row (every row in a leaf)."""
        var total = 0
        for b in range(n_leaves):
            total += Int(self.h_tot[b])
        if total != self.need:
            raise Error(
                "partition_from_bins: " + String(self.need - total)
                + " rows fell outside the tree's " + String(n_leaves)
                + " leaves"
            )
        self.seg_start = List[Int](length=n_leaves, fill=0)
        for b in range(n_leaves):
            self.seg_start[b] = Int(self.h_seg[b])
        var nb = len(self.bounds)
        self.snap = List[Int](length=nb * n_leaves, fill=0)
        for i in range(nb * n_leaves):
            self.snap[i] = Int(self.h_snap[i])

    def task_partition(
        mut self,
        ctx: DeviceContext,
        estimate_size: Int,
        n_leaves: Int,
        mut htab: HostBuffer[DType.uint32],
        mut dtab: DeviceBuffer[DType.uint32],
        mut row_index: DeviceBuffer[DType.uint32],
        mut sizes: List[Int],
        mut offsets: List[Int],
    ) raises:
        """The partition `_partition_into` built for the prefix
        `[0, estimate_size)`: the same sizes and offsets on the host, the
        same `row_index` on the device (`_ord_segment_rows_kernel`)."""
        var k = -1
        for i in range(len(self.bounds)):
            if self.bounds[i] == estimate_size:
                k = i
        if k < 0:
            raise Error("ordered partition: no snapshot at the task's prefix")
        sizes.clear()
        offsets.clear()
        var running = 0
        var tp = htab.unsafe_ptr()
        for b in range(n_leaves):
            var c = self.snap[k * n_leaves + b]
            sizes.append(c)
            offsets.append(running)
            tp.unsafe_store(3 * b, UInt32(running))
            tp.unsafe_store(3 * b + 1, UInt32(c))
            tp.unsafe_store(3 * b + 2, UInt32(self.seg_start[b]))
            running += c
        ctx.enqueue_copy(dst_buf=dtab, src_ptr=htab.unsafe_ptr())
        ctx.enqueue_function[_ord_segment_rows_kernel](
            self.d_sorted.unsafe_ptr(), dtab.unsafe_ptr(), Int32(n_leaves),
            row_index.unsafe_ptr(), Int32(estimate_size),
            grid_dim=(_grid(estimate_size), 1, 1),
            block_dim=(ORDERED_BLOCK, 1, 1),
        )


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


struct _OrderedSlot(Movable):
    """One batched estimation task's own buffers, for the whole fit: the
    partition's row order and its leaf table (`_PermPartition.
    task_partition`) with its upload staging, and the leaf
    upload pair (at `1 << max_depth`, of which a tree reads its
    `n_leaves`). Every cell a task reads it writes first; the batch's
    closing drain orders one tree's reads before the next tree's writes."""

    var row_index: DeviceBuffer[DType.uint32]
    var htab: HostBuffer[DType.uint32]
    var dtab: DeviceBuffer[DType.uint32]
    var dl: DeviceBuffer[DType.float32]
    var hl: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut arena: BufferArena,
        estimate_size: Int,
        leaf_capacity: Int,
    ) raises:
        self.row_index = arena.device[DType.uint32](ctx, estimate_size)
        self.htab = arena.host_buffer[DType.uint32](ctx, 3 * leaf_capacity)
        self.dtab = arena.device[DType.uint32](ctx, 3 * leaf_capacity)
        self.dl = arena.device[DType.float32](ctx, leaf_capacity)
        self.hl = arena.host_buffer[DType.float32](ctx, leaf_capacity)


struct _OrderedPending(Movable):
    """One `_ordered_estimate_task` stopped at its estimator's drain
    (`_ordered_estimate_prepare`); the estimator's state lives here until
    the batch's closing drain."""

    var est: PendingEstimation
    var apply_size: Int

    def __init__(out self, var est: PendingEstimation, apply_size: Int):
        self.est = est^
        self.apply_size = apply_size


def _ordered_estimate_prepare(
    ctx: DeviceContext,
    estimate_size: Int,
    apply_size: Int,
    n_leaves: Int,
    mut y: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut permutation: DeviceBuffer[DType.uint32],
    mut part: _PermPartition,
    mut slot: _OrderedSlot,
    mut cursor: DeviceBuffer[DType.float32],
    opts: OrderedBoostingOptions,
    sm_count: Int,
    mut est_ws: List[TEstimationWorkspace],
    mut arena: BufferArena,
    mut est_times: StageTimes,
    mut walker_times: StageTimes,
) raises -> _OrderedPending:
    """`_ordered_estimate_task` up to its estimator's evaluation readback,
    nothing drained (`_estimate_prepare`): the same gather, the same
    partition, the same estimator launches in the same order, into the
    task's own `slot` and `est_ws` -- the batch keeps every task's buffers
    in flight at once."""
    if estimate_size < 1 or estimate_size > apply_size:
        raise Error("ordered estimation requires 0 < prefix <= cursor size")
    est_times.begin(ctx)
    var sizes = List[Int]()
    var offsets = List[Int]()
    part.task_partition(
        ctx, estimate_size, n_leaves, slot.htab, slot.dtab, slot.row_index,
        sizes, offsets,
    )
    est_times.end(ctx, "est.partition")
    est_times.begin(ctx)
    # the gather into permutation order and the estimator's stage-in, as
    # one pass straight into this task's estimation workspace
    estimate_workspace(ctx, est_ws, arena, estimate_size, n_leaves)
    # `y` and `weights` arrive in THIS permutation's order (the fit's
    # per-permutation copies), so the stage-in is one gather through the
    # partition
    ctx.enqueue_function[_ord_stage_in_p_kernel](
        y.unsafe_ptr(), weights.unsafe_ptr(),
        cursor.unsafe_ptr(), slot.row_index.unsafe_ptr(),
        est_ws[0].g_target.unsafe_ptr(), est_ws[0].g_weights.unsafe_ptr(),
        est_ws[0].g_cursor.unsafe_ptr(), Int32(estimate_size),
        grid_dim=(_grid(estimate_size), 1, 1), block_dim=(ORDERED_BLOCK, 1, 1),
    )
    est_times.end(ctx, "est.gather")
    est_times.begin(ctx)
    var est = _estimate_prepare(
        ctx, estimate_size, n_leaves, sizes, offsets,
        slot.row_index, y, weights, True, cursor, opts.objective,
        opts.kernel_alpha, opts.estimator_alpha, opts.logloss_border,
        opts.l2_leaf_reg, sm_count, opts.leaf_method, est_ws, arena,
        walker_times, staged=True, iterations=opts.leaf_iterations,
    )
    est_times.end(ctx, "est.estimate_and_apply")
    return _OrderedPending(est^, apply_size)


def _ordered_estimate_complete(
    ctx: DeviceContext,
    mut pending: _OrderedPending,
    mut slot: _OrderedSlot,
    n_leaves: Int,
    mut permutation: DeviceBuffer[DType.uint32],
    mut bins: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    opts: OrderedBoostingOptions,
    mut est_ws: List[TEstimationWorkspace],
    mut trace: IdentityTrace,
    tag: String,
    mut est_times: StageTimes,
    mut walker_times: StageTimes,
) raises -> List[Float32]:
    """The rest of `_ordered_estimate_task` after the batch's drain: the
    estimator's host half and its `AppendModels` (`_estimate_complete`),
    then `leaf * rate` onto the real cursor, ENQUEUED; the buffers stay in
    `pending` for the batch's closing drain."""
    var leaves = List[Float32]()
    var not_pd = 0
    est_times.begin(ctx)
    _estimate_complete(
        ctx, pending.est, slot.row_index, cursor, opts.learning_rate,
        leaves, not_pd, trace, walker_times, tag, est_ws,
        append_to_cursor=False,
    )
    est_times.end(ctx, "est.estimate_and_apply")
    est_times.begin(ctx)
    ref hl = slot.hl
    ref dl = slot.dl
    for leaf in range(n_leaves):
        hl.unsafe_ptr().unsafe_store(leaf, leaves[leaf])
    ctx.enqueue_copy(dst_buf=dl, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_function[_ordered_apply_kernel](
        permutation.unsafe_ptr(), bins.unsafe_ptr(), dl.unsafe_ptr(),
        cursor.unsafe_ptr(), Int32(pending.apply_size), opts.learning_rate,
        grid_dim=(_grid(pending.apply_size), 1, 1),
        block_dim=(ORDERED_BLOCK, 1, 1),
    )
    est_times.end(ctx, "est.apply")
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
    # THE FIT'S ARENA (`gbdt/gpu_util/arena.mojo`): the long-lived per-fold
    # and per-task buffers below are carved from a few parents, because on
    # Metal every live allocation is bound to every launch
    var arena = BufferArena()
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
            var c = arena.device[DType.float32](
                ctx, folds[f].quality_evaluate_samples.right
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

    # every permutation's targets and weights in its own order, ONCE for the
    # fit: the fold derivatives read prefixes of the learn permutation's, the
    # batched estimation's stage-in reads them through its partition
    var dys = List[DeviceBuffer[DType.float32]]()
    var dws = List[DeviceBuffer[DType.float32]]()
    for p in range(perm_count):
        var dy = arena.device[DType.float32](ctx, n_rows)
        var dw = arena.device[DType.float32](ctx, n_rows)
        ctx.enqueue_function[_ord_gather_kernel](
            targets.unsafe_ptr(), dperms[p].unsafe_ptr(), dy.unsafe_ptr(),
            Int32(n_rows), grid_dim=(_grid(n_rows), 1, 1),
            block_dim=(ORDERED_BLOCK, 1, 1),
        )
        ctx.enqueue_function[_ord_gather_kernel](
            weights.unsafe_ptr(), dperms[p].unsafe_ptr(), dw.unsafe_ptr(),
            Int32(n_rows), grid_dim=(_grid(n_rows), 1, 1),
            block_dim=(ORDERED_BLOCK, 1, 1),
        )
        dys.append(dy^)
        dws.append(dw^)

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
    # the fold derivatives' gathers and planes, one set per fold for the
    # whole fit: every cell is rewritten each tree before it is read, and
    # the fit's queue orders a tree's rewrite after the last tree's reads,
    # so the per-fold drain that used to free them is gone (it was a drain
    # per fold per tree, nine a tree below 500 rows)
    var der_stats = List[DeviceBuffer[DType.float32]]()
    var der_part = List[DeviceBuffer[DType.float32]]()
    for f in range(n_folds):
        var r = folds[f].quality_evaluate_samples.right
        der_stats.append(arena.device[DType.float32](ctx, 2 * r))
        der_part.append(
            arena.device[DType.float32](
                ctx, (r + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
            )
        )
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
    #
    # THE BATCHED ESTIMATION (`estimate_can_batch`: a single-dimensional
    # pointwise loss under the Newton or Gradient walker, which every
    # default Ordered fit is). Each task's only drains were its estimator's
    # (the oracle's weight fold, one readback per walker evaluation, the
    # estimator's tail) and its own tail; the tasks of a tree are
    # independent -- each reads the tree's bins and its OWN cursor and
    # writes only that cursor -- so the tree now enqueues every task up to
    # its first readback, then walks every task in LOCK STEP (one drain per
    # round, `estimate_advance`), finishes them in task order (the same
    # trace records in the same order), and drains once more: at the
    # default four permutations, (walker rounds + 1) drains for the tree's
    # 28 tasks instead of several per task. Each task's oracle calls run in
    # the order its own walk makes them, on its own buffers. It needs one
    # workspace and one partition staging per TASK (slot `lp * n_folds +
    # f`, the estimation task last), since all of them are in flight
    # together.
    var batch = estimate_can_batch(
        opts.objective, opts.leaf_method, opts.leaf_iterations
    )
    var n_slots = learn_count * n_folds + 1 if batch else n_folds + 1
    for _ in range(n_slots):
        est_pools.append(List[TEstimationWorkspace]())
    var slots = List[_OrderedSlot]()
    if batch:
        for _ in range(learn_count):
            for f in range(n_folds):
                var est = folds[f].estimate_samples.right
                comptime if ORDERED_SABOTAGE:
                    est = folds[f].quality_evaluate_samples.right
                slots.append(_OrderedSlot(ctx, arena, est, 1 << max_depth))
        slots.append(_OrderedSlot(ctx, arena, n_rows, 1 << max_depth))
    # the batch's partitions, one per permutation a task reads
    # (`_PermPartition`): the learn permutations at every fold's estimate
    # prefix, and the estimation permutation at every row. With one
    # permutation the two are the same list and one partition serves both.
    var parts = List[_PermPartition]()
    if batch:
        for p in range(perm_count):
            var bounds = List[Int]()
            if p < learn_count:
                for f in range(n_folds):
                    var est = folds[f].estimate_samples.right
                    comptime if ORDERED_SABOTAGE:
                        est = folds[f].quality_evaluate_samples.right
                    bounds.append(est)
            if p == est_p:
                bounds.append(n_rows)
            parts.append(
                _PermPartition(ctx, arena, bounds^, 1 << max_depth)
            )
    var part_counts = arena.device[DType.uint32](ctx, ORDERED_PART_CELLS if batch else 1)
    var part_prefix = arena.device[DType.uint32](ctx, ORDERED_PART_CELLS if batch else 1)
    var losses = List[Float64]()
    var fv_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var fv_part = ctx.enqueue_create_buffer[DType.float32](fv_blocks)
    var fv = ctx.enqueue_create_buffer[DType.float32](1)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](1)
    var dummy_mag = ctx.enqueue_create_buffer[DType.float32](2)
    var d_sums = ctx.enqueue_create_buffer[DType.float32](3)
    var h_sums = ctx.enqueue_create_host_buffer[DType.float32](3)
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
            # the prefix [0, r) of the learn permutation's own-order copies
            ref gy = dys[learn_p]
            ref gw = dws[learn_p]
            ref stats = der_stats[f]
            ref part = der_part[f]
            var blocks = (r + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
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

        times.end(ctx, "ord.derivatives")
        # 3. the score noise, from the UNBOOTSTRAPPED quality slices
        times.begin(ctx)
        var score_std = Float32(0.0)
        # the noise sum and the scale's two magnitudes in ONE pass when no
        # bootstrap lies between them (`_ord_std_and_mags_kernel`)
        var fused_sums = opts.random_strength != Float32(0.0) and not bootstrap_on
        var m0 = Float64(0.0)
        var m1 = Float64(0.0)
        if fused_sums:
            var part = ctx.enqueue_create_buffer[DType.float32](3 * REDUCE_LANES_BLOCK)
            if String(getenv("MOJOLEARN_ORD_STD_SPLIT")) != "0":
                ctx.enqueue_function[_ord_std_lanes_kernel](
                    sw.unsafe_ptr(), sg.unsafe_ptr(), quality.unsafe_ptr(),
                    Int32(total), part.unsafe_ptr(),
                    grid_dim=REDUCE_LANES_BLOCK // ORD_STD_LANES, block_dim=ORD_STD_LANES,
                )
                ctx.enqueue_function[_ord_std_combine_kernel](
                    part.unsafe_ptr(), d_sums.unsafe_ptr(),
                    grid_dim=1, block_dim=REDUCE_LANES_BLOCK,
                )
            else:
                ctx.enqueue_function[_ord_std_and_mags_kernel](
                    sw.unsafe_ptr(), sg.unsafe_ptr(), quality.unsafe_ptr(),
                    Int32(total), d_sums.unsafe_ptr(),
                    grid_dim=1, block_dim=REDUCE_LANES_BLOCK,
                )
            ctx.enqueue_copy(dst_buf=h_sums, src_buf=d_sums)
            ctx.synchronize()
            _ = part^
            m0 = Float64(h_sums[1])
            m1 = Float64(h_sums[2])
            var count = 0
            for f in range(n_folds):
                count += (
                    folds[f].quality_evaluate_samples.right
                    - folds[f].estimate_samples.right
                )
            var mult = ordered_model_length_mult(
                n_rows, identical_mul64(Float64(iteration), Float64(opts.learning_rate))
            )
            score_std = Float32(
                mult
                * sqrt(Float64(h_sums[0]) / (Float64(count) + 1e-100))
                * Float64(opts.random_strength)
            )
        elif opts.random_strength != Float32(0.0):
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
            # the product pinned: inlined, the default build fused it into
            # `log(n) - model_size` (lane/pinned-mul-contract-free)
            var mult = ordered_model_length_mult(
                n_rows, identical_mul64(Float64(iteration), Float64(opts.learning_rate))
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
        if not fused_sums:
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
            m0 = Float64(hm[0])
            m1 = Float64(hm[1])
            _ = absv^
            _ = mags^
            _ = hm^
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, total))
        trace.record_scalar_f32(tag + ".scale", scale)
        trace.record_scalar_f32(tag + ".score_std", score_std)

        # lane/sym-quality: the gradient plane onto the tree's grid before
        # the search (`enqueue_snap_plane`), after the score std dev, the
        # bootstrap and the scale, as gbdt_oracle_ordered restates it
        enqueue_snap_plane(ctx, sg, total, scale)
        times.end(ctx, "ord.scale")
        # 5. the structure, on the learn permutation's folds
        times.begin(ctx)
        var splits = fit_oblivious_tree_structure_traced(
            ctx, layout, n_rows, max_depth, cindex, sw^, sg^, sm_count,
            scale, opts.score_function, pool, no_trace, times,
            String("tree"), opts.l2_leaf_reg,
            score_std_dev=score_std, seed=tree_seed, one_hot=one_hot,
            folds=folds, permutation=perms[learn_p], permutation_id=learn_p,
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
        if batch:
            # one device counting sort per permutation (`_PermPartition`),
            # one drain for all of them, their counts home
            for p in range(perm_count):
                parts[p].enqueue_sort(
                    ctx, bins, dperms[p], n_leaves, part_counts, part_prefix
                )
            ctx.synchronize()
            for p in range(perm_count):
                parts[p].settle(n_leaves)
        else:
            # the tree's bins on the host ONCE, for every task's partition
            # (`_partition_from_host_bins`)
            ctx.enqueue_copy(dst_buf=h_tree_bins, src_buf=bins)
            ctx.synchronize()

        times.end(ctx, "ord.bins")
        # 6. the fold models, then the estimation model
        times.begin(ctx)
        var leaves = List[Float32]()
        if batch:
            var pend = List[_OrderedPending]()
            for lp in range(learn_count):
                for f in range(n_folds):
                    var est = folds[f].estimate_samples.right
                    comptime if ORDERED_SABOTAGE:
                        est = folds[f].quality_evaluate_samples.right
                    var slot = lp * n_folds + f
                    pend.append(
                        _ordered_estimate_prepare(
                            ctx, est, folds[f].quality_evaluate_samples.right,
                            n_leaves, dys[lp], dws[lp], dperms[lp], parts[lp],
                            slots[slot],
                            cursors[lp][f], opts, sm_count, est_pools[slot],
                            arena, est_times, walker_times,
                        )
                    )
            var est_slot = learn_count * n_folds
            pend.append(
                _ordered_estimate_prepare(
                    ctx, n_rows, n_rows, n_leaves, dys[est_p], dws[est_p],
                    dperms[est_p], parts[est_p],
                    slots[est_slot], est_cursor, opts, sm_count,
                    est_pools[est_slot], arena, est_times, walker_times,
                )
            )
            # the walks in lock step: one drain per round for every task
            # still walking (a one-iteration walk is one round)
            var walking = True
            while walking:
                ctx.synchronize()
                walking = False
                for t in range(len(pend)):
                    if pend[t].est.phase != 2:
                        if estimate_advance(pend[t].est):
                            walking = True
            for lp in range(learn_count):
                for f in range(n_folds):
                    var slot = lp * n_folds + f
                    _ = _ordered_estimate_complete(
                        ctx, pend[slot], slots[slot], n_leaves, dperms[lp],
                        bins,
                        cursors[lp][f], opts, est_pools[slot], trace,
                        tag + ".perm." + String(lp) + ".fold." + String(f),
                        est_times, walker_times,
                    )
            leaves = _ordered_estimate_complete(
                ctx, pend[est_slot], slots[est_slot], n_leaves,
                dperms[est_p], bins,
                est_cursor, opts, est_pools[est_slot], trace,
                tag + ".estimation", est_times, walker_times,
            )
            ctx.synchronize()
            _ = pend^
        for lp in range(0 if batch else learn_count):
            for f in range(n_folds):
                var est = folds[f].estimate_samples.right
                comptime if ORDERED_SABOTAGE:
                    est = folds[f].quality_evaluate_samples.right
                _ = _ordered_estimate_task(
                    ctx, est, folds[f].quality_evaluate_samples.right,
                    n_leaves, targets, weights, dperms[lp], perms[lp], bins,
                    h_tree_bins, h_part_rows, cursors[lp][f], opts, sm_count,
                    est_pools[f], trace,
                    tag + ".perm." + String(lp) + ".fold." + String(f),
                    est_times, walker_times,
                )
        times.end(ctx, "ord.fold_estimates")
        times.begin(ctx)
        if not batch:
            leaves = _ordered_estimate_task(
                ctx, n_rows, n_rows, n_leaves, targets, weights,
                dperms[est_p], perms[est_p], bins, h_tree_bins, h_part_rows,
                est_cursor, opts, sm_count, est_pools[n_folds], trace,
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
    _ = slots^
    _ = parts^
    _ = part_counts^
    _ = part_prefix^
    _ = dys^
    _ = dws^
    _ = der_stats^
    _ = der_part^
    _ = cursors^
    _ = arena^
    _ = boot_seeds^
    _ = boot_mags^
    _ = est_y^
    _ = est_w^
    _ = fv_part^
    _ = fv^
    _ = h_fv^
    _ = dummy_mag^
    _ = d_sums^
    _ = h_sums^
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
