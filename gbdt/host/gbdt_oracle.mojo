# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host, a SECOND spelling of the device
trainer on the gbdt-symmetric lane (workstream E batch 3, 2026-09-14; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 1.1 gbdt and the batch 3
section).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The library imports are the `checks/numerics`
seams and the GPU-free host modules the device fit ITSELF runs on the host,
unchanged: the pieces of the GreedyLogSum border search and the NaN mode
(`gbdt/data/quantization.mojo`, `gbdt/grid_creator/binarization.mojo`; the
two entry functions are restated, because the device calls them on a worker
thread that reads subnormals as zero and the host oracle calls them on its
own thread, `_calc_quantization_phase_b`), the
compressed-index layout and the policy blocks
(`gbdt/gpu_data/compressed_index_builder.mojo`,
`gbdt/gpu_data/feature_blocks.mojo`, `gbdt/gpu_data/grid_policy.mojo`) and
CatBoost's host `TRandom` (`gbdt/data/permutation.mojo`). Reusing them is not
a shared-bug risk the rf oracle's rule guards against: they are host code on
the device path too, so the host binding runs the SAME function the GPU
binding runs (the SAME function is not the SAME bits when the calling thread
differs, which is what the border search taught; see
`_calc_quantization_phase_b`). Every device KERNEL the fit reaches is
RESTATED below, with the file and line it MIRRORS.

THE CONFIGURATION THIS COVERS, and it is the lane's, by name
(tools/identity_break.py `gbdt-symmetric`: 20 trees, depth 6, Logloss, every
other option at its default). The binding refuses by name every value
outside it (bindings/_mojolearn_gbdt_host.mojo, `_refuse`):

  grow_policy SymmetricTree, loss Logloss (any loss_border),
  score_function Cosine, leaf_estimation_method Newton (any iteration count),
  no bootstrap, no sample_weight, no class_weights, no CTR categorical
  column (one-hot columns are carried, `one_hot_in` below and
  gbdt/host/gbdt_oracle_onehot.mojo), no eval_set and no overfitting detector,
  random_strength 0, the greedy searcher (use_pointwise_searcher False),
  boost_from_average unset or False, feature_fraction 1, and no
  feature with exactly one border (the BINARY histogram policy, whose
  nibble-combination decode is not restated). border_count, n_estimators,
  max_depth, learning_rate, l2_leaf_reg, random_state, nan_mode (Min and
  Max on an X carrying NaN since 2026-09-15, the gbdt-nan-modes lane) and
  border_build_max_samples (both border paths) are carried.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build, the
default flags: DEVIATION 2550 borrow, 2031 ridx splits OFF, 2580 level
quantize OFF, 2581 group width ON, 2030 fused move OFF)

  1. `_quantize_training_columns` (`gbdt/train.mojo:1898-2241`): the shared
     border subsample of `sample_indices_for_borders` (`:595-664`) when the
     row count exceeds `border_build_max_samples`, the device radix sort
     (`gbdt/gpu_util/kernel/radix_sort.mojo:299-395`, the twiddled key
     order) otherwise, then `calc_quantization` per float column, RESTATED
     as `_calc_quantization_phase_b` with the subnormal flush the device's
     phase B workers apply (measured on the `denormal` fixture; see there).
  2. `_build_cindex_from_columns` + `binarize_float_feature_kernel`
     (`gbdt/train.mojo:549-592`, `gbdt/gpu_data/kernel/binarize.mojo:83-160`):
     the bin is the count of borders the value exceeds, OR-ed into the word.
  3. Per tree (`gbdt/methods/doc_parallel_boosting.mojo:1427-2169`): the
     search pass of `cross_entropy_kernel[True, False]`
     (`gbdt/targets/kernel/pointwise_targets.mojo:866-1054`) with its
     per-block `two_phase_halving_sum` partials (`core/pinned_reduce.mojo:
     103-147`) folded by `deterministic_sum_lanes_kernel` (`:808-863`), and
     the scale of `choose_scale_kernel` (`kernel/histogram_utils.mojo:
     808-867`).
  4. `run_tree_layout_traced` (`gbdt/methods/greedy_subsets_searcher/
     greedy_search_helper.mojo:4677-5835`), every level of it, including the
     levels the host gate later discards, because the device splits them:
       - the histograms of `launch_histograms_for_blocks` (`:2644-3238`).
         ONE-BYTE blocks take the 2581 width arms, whose value is an Int32
         sum of `hist2_quantize(stat, scale, hist2_dither(position))` per
         (leaf, stat, bin), dequantized by `write_reduces_from_fixed_kernel
         [False]` (`histogram_utils.mojo:33-97`, `:712-805`;
         `hist_2_one_byte_base.mojo:409-573`, `:919-1251`; the per-width
         slot rule of `hist_2_one_byte_{5,6,7,8}bit.mojo`). Integer sums, so
         the thread layout does not reach the bits and is not simulated.
         HALF-BYTE blocks accumulate FLOAT per replica slice, so their
         thread layout IS the arithmetic and is simulated thread for thread:
         `half_byte_hist_gather_kernel` (`hist_half_byte.mojo:555-1040`),
         the slice geometry of `point_hist_half_byte_template.mojo:182-357`
         (512-float replicas, `& 24` sub-copies, the rotation slot), the
         pinned replication of `replication_for` (`greedy_search_helper.mojo:
         2547-2641`, sm 32), the head and tail peel with DEVIATION 2600's
         bound, the striped loop, the two reduce stages, and the flush
         (`active_block_count > 1` quantizes each block partial
         `Int32(val * scale)`, otherwise the float is stored).
       - `scan_histograms_kernel` (`histogram_utils.mojo:257-326`) over the
         computed leaves, `substract_histograms_kernel` (`:206-254`) for the
         derived siblings.
       - `compute_partition_stats` (`gbdt/gpu_util/partitions_reduce.mojo:
         104-227`) at the pinned 32 chunks.
       - `compute_optimal_splits_kernel[COSINE]` (`kernel/compute_scores.mojo:
         46-86`, `:89-266`) and `resolve_and_pack_kernel`
         (`kernel/split_resolve.mojo:90-163`): the level winner is the
         largest gain, ties to the smaller bin-feature, a gain of
         -FLOAT32_MAX or NaN never taken.
       - `split_and_make_sequence_kernel`, the three-phase stable partition
         and `launch_reorder_in_leaves` (`kernel/split_points.mojo:55-153`,
         `:1083-1338`, `:753-933`): each leaf's rows, zeros then ones, in
         order; `copy_histograms_kernel` (`histogram_utils.mojo:473-506`);
         `update_partitions_and_plan_kernel` (`split_points.mojo:250-340`,
         the tie computes the right child).
       - the post-tree gates and the rollback of `accept_symmetric_level_
         winner` (`greedy_search_helper.mojo:3835-3880`, `:5656-5682`).
  5. The estimation task (`doc_parallel_boosting.mojo:622-850`): the gathers
     by the row index, `make_bin_optimized_oracle`
     (`gbdt/methods/leaves_estimation/pointwise_oracle.mojo:968-1212`,
     unweighted `WeightsCpu` from the leaf sizes), `move_to` and
     `write_value_and_first_derivatives` (`:331-573`, the host Float32 fold
     of the value partials), `regularize` (`:954-965`), the Newton walker
     (`descent_helpers.mojo:81-100`, `:186-285`) with AnyImprovement
     (`step_estimator.mojo:50-72`), and `add_model_value_kernel`
     (`gbdt/methods/kernel_add_model_value.mojo:45-104`, the cursor update
     through `identical_mul_add`).
  6. The weak model's rescale (`doc_parallel_boosting.mojo:2131-2133`), the
     learn loss bookkeeping (`:2158-2169`, `:2180-2259`) and the model text
     (`gbdt/models/model_text.mojo:268-312`, `:374-487`, `:601-670`).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` adds 1.0 to the Newton
walker's Hessian regularizer (`lambda_reg`, `pointwise_oracle.mojo:547-555`),
so every estimated leaf of every tree moves and every fixture's model and
predictions move with it.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-symmetric lane is the measurement.
"""
from std.math import exp, floor, isfinite, log, log2, sqrt
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
    identical_sqrt,
)
from gbdt.data.permutation import TRandom
from gbdt.gpu_util.kernel.random_gen import (
    advance_seed_k,
    next_normal_f,
    next_poisson_f,
    next_uniform_f,
)
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_FALSE,
    NAN_TREATMENT_AS_IS,
    NAN_TREATMENT_AS_TRUE,
    compute_nan_mode,
    has_nans,
    nan_substitution,
    nan_value_treatment,
)
from gbdt.grid_creator.binarization import (
    TFeatureBin,
    _heap_pop,
    _heap_push,
    _sort_ascending,
    _update_best_split,
    BORDER_TYPE_GREEDY_LOG_SUM,
    select_borders,
)
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    NAN_MODE_MAX,
    NAN_MODE_MIN,
    nan_mode_from_name,
)
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)


#: `BOOTSTRAP_KERNEL_*` (`bootstrap.mojo:106-108`); -1 is no bootstrap.
comptime GBDT_BOOT_BAYESIAN = 0
comptime GBDT_BOOT_BERNOULLI = 1
comptime GBDT_BOOT_POISSON = 2
#: `BOOTSTRAP_BLOCK_SIZE`, `BOOTSTRAP_SEED_COUNT` (`bootstrap.mojo:112-115`).
comptime GBDT_BOOT_BLOCK = 256
comptime GBDT_BOOT_SEEDS = 65536


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime GBDT_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()
comptime GBDT_HOST_BINARIZE_LINEAR = is_defined[
    "MOJOLEARN_GBDT_HOST_BINARIZE_LINEAR"
]()
"""A/B receipt only: restore the pre-optimization linear border scan."""

#: `MSE_BLOCK_SIZE` (`pointwise_targets.mojo:143`), the target kernel's block.
comptime GBDT_MSE_BLOCK = 256
#: `REDUCE_LANES_BLOCK` (`pointwise_targets.mojo:805`).
comptime GBDT_REDUCE_LANES_BLOCK = 256
#: `STATS_BLOCK` (`partitions_reduce.mojo:101`).
comptime GBDT_STATS_BLOCK = 512
#: `PINNED_PARTITION_CHUNKS_SM` (`checks/kernel_matrix.mojo:668`), the sm
#: count every IDENTICAL chunk and replication formula reads.
comptime GBDT_PINNED_SM = 32
#: the half-byte family under IDENTICAL: `block_size_for[K_HIST_HALF_BYTE]`
#: (`checks/kernel_matrix.mojo:472-488`, 32 KB over 16 floats x 4 bytes),
#: `replication_lanes_for` 32, `LOAD_SIZE` 4, `UNROLL` 1
#: (`hist_half_byte.mojo:76-105`), 16 floats per thread.
comptime GBDT_HB_BLOCK = 512
comptime GBDT_HB_LANES = 32
comptime GBDT_HB_LOAD = 4
comptime GBDT_HB_FLOATS = 16
comptime GBDT_HB_REDUCE_WIDTH = 512
#: `FLOAT32_MAX` (`compute_scores.mojo:32`).
comptime GBDT_FLOAT32_MAX = Float32(3.4028234663852886e38)
#: `WINNER_SENTINEL` (`split_resolve.mojo:60`).
comptime GBDT_SENTINEL = UInt32(0xFFFFFFFF)
#: Logloss's Newton iteration default (`catboost_options.mojo:1304-1308`).
comptime GBDT_LOGLOSS_NEWTON_ITERATIONS = 10


# ===========================================================================
# THE PARAMETERS AND THE MODEL
# ===========================================================================


@fieldwise_init
struct GbdtHostParams(ImplicitlyCopyable, Movable):
    """What the lane's fit reads, resolved the way `gbdt_fit` and `train`
    resolve it (`bindings/_mojolearn_gbdt.mojo:345-383`,
    `gbdt/train.mojo:1560-1649`)."""

    var border_count: Int
    var border_build_max_samples: Int
    var n_estimators: Int
    var max_depth: Int
    var learning_rate: Float32
    var l2_leaf_reg: Float32
    var random_seed: UInt64
    var nan_mode: Int
    var logloss_border: Float32
    var leaf_estimation_iterations: Int
    #: `feature_border_type` (`binarization.mojo` BORDER_TYPE_*),
    #: GreedyLogSum (0) unless the fit named another
    var border_type: Int


@fieldwise_init
struct GbdtHostModel(Movable):
    """`TrainedModel` for one oblivious, one-dimensional, float-only ensemble
    (`gbdt/train.mojo:209-258`), flat: tree `t`'s splits are
    `split_*[tree_split_offsets[t] .. tree_split_offsets[t + 1])` and its
    `1 << depth` leaves `leaf_values[tree_leaf_offsets[t] .. ]`."""

    var fold_counts: List[Int]
    var borders: List[List[Float32]]
    var nan_treatment: List[Int]
    var tree_split_offsets: List[Int]
    var split_features: List[Int]
    var split_bins: List[Int]
    var tree_leaf_offsets: List[Int]
    var leaf_values: List[Float32]
    var losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool

    def n_trees(self) -> Int:
        return len(self.tree_split_offsets) - 1


# ===========================================================================
# THE FOLDS: `two_phase_halving_sum`, `deterministic_sum_lanes_kernel`
# ===========================================================================


def _halving_fold(mut slab: List[Float32]) -> Float32:
    """The halving tree `red[t] += red[t + step]`, `step = N/2 .. 1`
    (`core/pinned_reduce.mojo:103-147`, whose three barriers fold the same
    additions in the same order; `deterministic_sum_lanes_kernel`'s stage at
    `pointwise_targets.mojo:847-858` is the same tree). `len(slab)` is a
    power of two; threads with no data hold 0.0."""
    var step = len(slab) // 2
    while step > 0:
        for t in range(step):
            slab[t] = slab[t] + slab[t + step]
        step //= 2
    return slab[0]


def _deterministic_sum_lanes(
    partials: List[Float32], lanes: Int, count: Int
) -> List[Float32]:
    """`deterministic_sum_lanes_kernel[lanes]` (`pointwise_targets.mojo:
    808-863`): thread `t` folds slots `t, t + 256, ...` ascending from 0.0,
    then the 256-lane halving tree, per lane."""
    var out = List[Float32](length=lanes, fill=Float32(0.0))
    for lane in range(lanes):
        var slab = List[Float32](
            length=GBDT_REDUCE_LANES_BLOCK, fill=Float32(0.0)
        )
        for tid in range(GBDT_REDUCE_LANES_BLOCK):
            var acc = Float32(0.0)
            var i = tid
            while i < count:
                acc += partials[i * lanes + lane]
                i += GBDT_REDUCE_LANES_BLOCK
            slab[tid] = acc
        out[lane] = _halving_fold(slab)
    return out^


# ===========================================================================
# THE TARGET KERNEL: `cross_entropy_kernel[has_border=True]`
# ===========================================================================


@fieldwise_init
struct _CeRow(ImplicitlyCopyable, Movable):
    var weighted_direction: Float32
    var weighted_direction_raw: Float32
    var weighted_scale: Float32
    var score: Float32


def _cross_entropy_row(
    target_class: Float32, val: Float32, border: Float32, weight: Float32
) -> _CeRow:
    """One in-range thread of `cross_entropy_kernel`
    (`pointwise_targets.mojo:947-1054`), term for term: the routed `exp`,
    the clamped probability, the bordered class, the flushed direction and
    scale, and the score with its isfinite fallback."""
    var exp_val = identical_exp(val)
    var p = Float32(1.0)
    if isfinite(exp_val):
        p = exp_val / (Float32(1.0) + exp_val)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
    var c: Float32
    if target_class > border:
        c = Float32(1.0)
    else:
        c = Float32(0.0)
    var direction = ftz(c - p)
    var scale = ftz(p * (Float32(1.0) - p))
    var log_exp_val_plus_one = val
    if isfinite(exp_val):
        log_exp_val_plus_one = identical_log(Float32(1.0) + exp_val)
    var score = weight * (c * val - log_exp_val_plus_one)
    return _CeRow(
        ftz(weight * direction), weight * direction, ftz(weight * scale), score
    )


def _logloss_search_pass(
    targets: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    border: Float32,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_approximate[False]` on Logloss with `compute_fv` and
    `compute_magnitudes` set (`doc_parallel_boosting.mojo:1544-1552`): the
    SEARCH planes (plane 0 the unit weight, plane 1 `ftz(weight * (c - p))`),
    one score partial and two magnitude partials per 256-thread block
    through the halving tree. Out-of-range threads add 0.0."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var weight = Float32(1.0)
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var r = _cross_entropy_row(targets[i], cursor[i], border, weight)
                var plane0 = weight
                stats[i] = plane0
                stats[n_rows + i] = r.weighted_direction
                s_score[t] = r.score
                s_w[t] = abs(plane0)
                s_g[t] = abs(r.weighted_direction_raw)
        fv_partials[b] = _halving_fold(s_score)
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def _logloss_value(
    targets: List[Float32], cursor: List[Float32], n_rows: Int, border: Float32
) -> Float32:
    """The final learn loss pass (`doc_parallel_boosting.mojo:2199-2211`):
    the same kernel's score partials, folded by
    `deterministic_sum_lanes_kernel[1]`."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_partials = List[Float32](length=blocks, fill=Float32(0.0))
    var weight = Float32(1.0)
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                s_score[t] = _cross_entropy_row(
                    targets[i], cursor[i], border, weight
                ).score
        fv_partials[b] = _halving_fold(s_score)
    return _deterministic_sum_lanes(fv_partials, 1, blocks)[0]


# ===========================================================================
# THE FIXED-POINT SCALE AND THE DITHERED QUANTIZER
# ===========================================================================


def _choose_scale_from_magnitudes(
    w: Float32, g: Float32, row_count: Int
) -> Float32:
    """`choose_scale_kernel` (`histogram_utils.mojo:808-867`), line for line."""
    var m = w
    if g > m:
        m = g
    if m == Float32(0.0):
        return Float32(1.0)
    var limit = Int64((1 << 30) - 1) - Int64(row_count)
    var floor_limit = Int64((1 << 28) - 1)
    if limit < floor_limit:
        limit = floor_limit
    var mbits = bitcast[DType.uint32](m)
    var exp_field = Int((mbits >> 23) & UInt32(0xFF))
    var mant = Int64(Int(mbits & UInt32(0x7FFFFF)))
    var s_int: Int64
    var e2: Int
    if exp_field == 0:
        s_int = mant
        e2 = -149
    else:
        s_int = mant + Int64(1 << 23)
        e2 = exp_field - 150
    var t = 30
    while (s_int << Int64(t)) > limit:
        t -= 1
    var k = t - e2
    var scale = Float32(1.0)
    if k >= 0:
        for _ in range(k):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-k):
            scale = scale * Float32(0.5)
    return scale


def _hist2_dither(pos: Int) -> Float32:
    """`hist2_dither` (`histogram_utils.mojo:33-46`)."""
    var h = UInt32(pos) * UInt32(2654435761)
    h ^= h >> 16
    h = h * UInt32(2246822519)
    h ^= h >> 13
    return Float32(Int(h >> 8)) * Float32(5.9604645e-08)


def _hist2_quantize(val: Float32, fixed_scale: Float32, u: Float32) -> Int32:
    """`hist2_quantize` (`histogram_utils.mojo:49-97`), the flushed operand
    and product, `floor`, and the fraction compare."""
    var scaled = ftz(ftz(val) * fixed_scale)
    var base = floor(scaled)
    var q = Int32(base)
    if ftz(scaled - base) + u >= Float32(1.0):
        q += Int32(1)
    return q


# ===========================================================================
# THE GRID: borders, NaN treatment, the compressed index
# ===========================================================================


@fieldwise_init
struct GbdtHostGrid(Movable):
    var borders: List[List[Float32]]
    var fold_counts: List[Int]
    var nan_treatment: List[Int]


def _generate_seed_for_borders(from_seed: UInt64) -> UInt64:
    """`generate_seed_for_borders` (`gbdt/train.mojo:595-602`)."""
    var sd = from_seed
    for _ in range(5):
        sd = 6364136223846793005 * sd + 1442695040888963407
    return sd


def _sample_indices_for_borders(nrr: Int, sn: Int, sd0: UInt64) -> List[UInt32]:
    """`sample_indices_for_borders` (`gbdt/train.mojo:605-664`), both
    branches, over the same `TRandom`."""
    var sample_idx = List[UInt32]()
    var rnd0 = TRandom(_generate_seed_for_borders(sd0))
    if sn >= nrr:
        for i in range(nrr):
            sample_idx.append(UInt32(i))
    elif sn > 1 and Float64(sn) > Float64(nrr) / log2(Float64(sn)):
        sample_idx.resize(nrr, UInt32(0))
        for i in range(nrr):
            sample_idx[i] = UInt32(i)
        for i in range(sn):
            var j = i + Int(rnd0.next_uniform_l() % UInt64(nrr - i))
            var t = sample_idx[i]
            sample_idx[i] = sample_idx[j]
            sample_idx[j] = t
        sample_idx.resize(sn, UInt32(0))
    else:
        var seen = List[Bool]()
        seen.resize(nrr, False)
        while len(sample_idx) < sn:
            var c = Int(rnd0.next_uniform_l() % UInt64(nrr))
            if not seen[c]:
                seen[c] = True
                sample_idx.append(UInt32(c))
    return sample_idx^


def _sorted_by_twiddled_key(values: List[Float32]) -> List[Float32]:
    """The device border sort's order (`radix_sort.mojo:299-395`): ascending
    by the monotone twiddle, negatives `~bits`, the rest `bits | 0x80000000`,
    so the output depends on the value bits alone."""
    var keys = List[UInt32](capacity=len(values))
    for i in range(len(values)):
        var bits = bitcast[DType.uint32](values[i])
        if (bits & UInt32(0x80000000)) != UInt32(0):
            keys.append(~bits)
        else:
            keys.append(bits | UInt32(0x80000000))
    sort(keys)
    var out = List[Float32](capacity=len(values))
    for i in range(len(keys)):
        var k = keys[i]
        var bits: UInt32
        if (k & UInt32(0x80000000)) != UInt32(0):
            bits = k ^ UInt32(0x80000000)
        else:
            bits = ~k
        out.append(bitcast[DType.float32](bits))
    return out^


def _best_split_phase_b(
    var values: List[Float32], max_borders_count: Int
) raises -> List[Float32]:
    """`best_split` (`gbdt/grid_creator/binarization.mojo:203-289`) as it
    computes inside the device fit's phase B worker, statement for
    statement, with every subnormal flushed to its signed zero BY BITS: the
    values as they enter, both halves of the midpoint and the midpoint.

    The comparisons of `_update_best_split` and the prelude sort then see
    every subnormal as the signed zero a denormals-as-zero reader sees, the
    radix sort keeps the sign classes in the positions the unflushed bit
    order gives them, and the heap scores are Float64 over integer bin sizes
    that no flush reaches. See `_calc_quantization_phase_b` for why."""
    # their `filterNans`, then the flush
    var clean = List[Float32]()
    for i in range(len(values)):
        if values[i] == values[i]:
            clean.append(ftz(values[i]))
    if len(clean) == 0:
        return List[Float32]()

    _sort_ascending(clean)

    var root = TFeatureBin()
    root.bin_start = 0
    root.bin_end = len(clean)
    root.best_split = 0
    root.best_score = 0.0
    _update_best_split(root, clean)

    var bins = List[TFeatureBin]()
    _heap_push(bins, root)

    while len(bins) <= max_borders_count and bins[0].can_split():
        var top = bins[0].copy()
        _heap_pop(bins)

        var left = TFeatureBin()
        left.bin_start = top.bin_start
        left.bin_end = top.best_split
        _update_best_split(left, clean)

        top.bin_start = top.best_split
        _update_best_split(top, clean)

        _heap_push(bins, left)
        _heap_push(bins, top)

    var borders = List[Float32]()
    for i in range(len(bins)):
        if bins[i].is_first():
            continue
        var s = bins[i].bin_start
        var half_below = ftz(Float32(0.5) * clean[s - 1])
        var half_above = ftz(Float32(0.5) * clean[s])
        borders.append(ftz(half_below + half_above))
    _sort_ascending(borders)

    var out = List[Float32]()
    for i in range(len(borders)):
        if i == 0 or borders[i] != borders[i - 1]:
            out.append(borders[i])
    return out^


def _calc_quantization_phase_b(
    var values: List[Float32], border_count: Int, nan_mode_option: Int,
    border_type: Int = BORDER_TYPE_GREEDY_LOG_SUM,
) raises -> Tuple[List[Float32], Int]:
    """`calc_quantization` (`gbdt/data/quantization.mojo:136-177`) as the
    device fit's PHASE B computes it: inside `_dp_task` on a
    `sync_parallelize` worker (`gbdt/train.mojo:2189-2208`), where every
    subnormal reads as a signed zero.

    MEASURED, NOT DESIGNED. CPU identity gate run 34893018288 at 9e3a04d70:
    the `denormal` fixture's saved gbdt-symmetric model hashes
    4f2c8b24bbe2eb42 on the Apple M4, the H100 and the MI325X, which is the
    `denormal_ftz` model on all four columns, while every one of the seven
    CPU runners hashed c41cbc306cf7f609 through the imported
    `calc_quantization`. The column's 5000 subnormals are distinct values,
    so an unflushed GreedyLogSum search spends 32 of column 0's 128 borders
    inside them and moves the normal borders beside them (a float32
    simulation of `best_split` reproduces this; flushing the INPUTS gives
    the `denormal_ftz` grid exactly, flushing only the midpoints does not).
    Training and held-out predictions agreed, so the extra borders never
    decided a split on this lane; the saved `feature` records did not.

    No statement on the device path flushes the column: the radix sort is
    an integer sort of the twiddled bits (`radix_sort.mojo:337-395`), the
    staging and `_dp_task` copies are loads and stores, and `best_split`
    has no `ftz`. The one border build that runs `best_split` on the
    CALLING thread, `train_ordered_rmse` (`gbdt/train.mojo:2274-2283`),
    keeps the subnormals on the same three columns (gbdt-ordered-rmse
    `denormal` and `denormal_ftz` differ in the train, infer and model
    columns). The worker thread's floating point mode is therefore the
    reading the records support, and this restatement models it BY BITS
    (`ftz`, `checks/numerics.mojo:73`; the host family is IDENTICAL only)
    so the CPU column does not depend on any runner's MXCSR or FPCR. The
    flushed midpoint halves are that model's consequence for a normal
    below 2^-125 whose half is subnormal; no fixture plants one, so that
    arm is unmeasured."""
    # Upstream CalcQuantizationAndNanMode checks this BEFORE BestSplit
    # filters NaNs (54a8143a, libs/data/quantization.cpp:315-320).
    if nan_mode_option == NAN_MODE_FORBIDDEN and has_nans(values):
        raise Error(
            "There are nan factors and nan values for float features are"
            " not allowed. Set nan_mode != Forbidden."
        )
    var nan_mode = compute_nan_mode(values, nan_mode_option)

    var non_nan_border_count = border_count
    if nan_mode != NAN_MODE_FORBIDDEN:
        non_nan_border_count -= 1

    var borders = List[Float32]()
    if non_nan_border_count > 0:
        if border_type == BORDER_TYPE_GREEDY_LOG_SUM:
            borders = _best_split_phase_b(values^, non_nan_border_count)
        else:
            # the six other border types flush BY BITS themselves, so the
            # device fit's phase B and this restatement call the SAME
            # function on the same column (`select_borders`)
            borders = select_borders(values^, non_nan_border_count, border_type)

    if nan_mode == NAN_MODE_MIN:
        var with_nan = List[Float32]()
        with_nan.append(Float32(-3.4028234663852886e38))
        for i in range(len(borders)):
            with_nan.append(borders[i])
        borders = with_nan^
    elif nan_mode == NAN_MODE_MAX:
        borders.append(Float32(3.4028234663852886e38))

    return (borders^, nan_mode)


def gbdt_host_grid(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_count: Int,
    border_build_max_samples: Int,
    random_seed: UInt64,
    nan_mode: Int,
    border_type: Int = BORDER_TYPE_GREEDY_LOG_SUM,
) raises -> GbdtHostGrid:
    """`_quantize_training_columns` for an all-float, one-permutation fit
    (`gbdt/train.mojo:1966-2241`). The full-data path hands
    `calc_quantization` the device-sorted column; the sampled path hands it
    the shared subsample, with the device's NaN seed into the sample
    (`:2019-2034`). The per-column border search is
    `_calc_quantization_phase_b`, not the imported `calc_quantization`: see
    its docstring for the subnormal flush the device fit's phase B applies."""
    var border_sample_n = n_rows
    if border_build_max_samples > 0 and border_build_max_samples < n_rows:
        border_sample_n = border_build_max_samples
    var sample_idx = List[UInt32]()
    if border_sample_n < n_rows:
        sample_idx = _sample_indices_for_borders(
            n_rows, border_sample_n, random_seed
        )
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    var nan_treatment = List[Int]()
    for f in range(n_features):
        var col = List[Float32](capacity=border_sample_n)
        if border_sample_n == n_rows:
            var raw = List[Float32](capacity=n_rows)
            for r in range(n_rows):
                raw.append(x_colmajor[f * n_rows + r])
            col = _sorted_by_twiddled_key(raw)
        else:
            for i in range(border_sample_n):
                col.append(x_colmajor[f * n_rows + Int(sample_idx[i])])
            var has_nan = False
            for r in range(n_rows):
                var v = x_colmajor[f * n_rows + r]
                if v != v:
                    has_nan = True
                    break
            if has_nan:
                var sample_has = False
                for i in range(border_sample_n):
                    var v2 = col[i]
                    if v2 != v2:
                        sample_has = True
                        break
                if not sample_has:
                    col[0] = Float32(0.0) / Float32(0.0)
        var q = _calc_quantization_phase_b(col^, border_count, nan_mode, border_type)
        var nb = len(q[0])
        if nb > border_count + 1:
            raise Error(
                "parallel border build failed on float column " + String(f)
            )
        borders.append(q[0].copy())
        fold_counts.append(nb)
        nan_treatment.append(nan_value_treatment(q[1]))
    return GbdtHostGrid(borders^, fold_counts^, nan_treatment^)


def _binarize_columns(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    grid: GbdtHostGrid,
    layout: CompressedIndexLayout,
) raises -> List[UInt32]:
    """`_build_cindex_from_columns` and `binarize_float_feature_kernel`
    (`gbdt/train.mojo:549-592`, `binarize.mojo:83-160`): a feature with no
    borders keeps zero bits; a NaN under `AsIs` is refused, otherwise
    substituted; the bin is the count of borders the value exceeds."""
    var cindex = List[UInt32](length=n_rows * layout.columns, fill=UInt32(0))
    for f in range(n_features):
        var nb = len(grid.borders[f])
        if nb == 0:
            continue
        ref cf = layout.features[f]
        var treat = grid.nan_treatment[f]
        var sub = nan_substitution(treat)
        var base = Int(cf.offset) * n_rows
        for r in range(n_rows):
            var v = x_colmajor[f * n_rows + r]
            if v != v:
                if treat == NAN_TREATMENT_AS_IS:
                    raise Error(
                        "There are NaNs in feature number " + String(f)
                        + " but there were no NaNs in the learn dataset"
                    )
                v = sub
            # `grid.borders[f]` is ascending.  The device kernel's answer is
            # the number of borders STRICTLY below `v`, i.e. lower_bound(v).
            # A linear walk used to repeat as many as 128 comparisons for
            # every row of every host fit; binary search returns the same
            # insertion point in ceil(log2(nb)) comparisons.  Keep the
            # comparison written as `border < v` so signed zero and equality
            # retain the exact `v > border` semantics above.
            var index = UInt32(0)
            comptime if GBDT_HOST_BINARIZE_LINEAR:
                for b in range(nb):
                    if v > grid.borders[f][b]:
                        index += 1
            else:
                var lo = 0
                var hi = nb
                while lo < hi:
                    var mid = lo + (hi - lo) // 2
                    if grid.borders[f][mid] < v:
                        lo = mid + 1
                    else:
                        hi = mid
                index = UInt32(lo)
            cindex[base + r] = cindex[base + r] | ((index & cf.mask) << cf.shift)
    return cindex^


# ===========================================================================
# THE HISTOGRAMS
# ===========================================================================


def _one_byte_block(
    blk: PolicyBlock,
    block_first_bin: Int,
    hist_cells: Int,
    compute_ids: List[Int],
    p_off: List[Int],
    p_sz: List[Int],
    row_index: List[Int],
    stats: List[Float32],
    cindex: List[UInt32],
    layout: CompressedIndexLayout,
    n_rows: Int,
    fixed_scale: Float32,
    mut hist: List[Float32],
):
    """A ONE-BYTE block on the IDENTICAL shared-Int32 route
    (`launch_one_byte_arms[False, True, ...]`, `greedy_search_helper.mojo:
    2013-2107`, `hist2_one_byte_gather_kernel` at 5/6/7 bits and
    `hist2_8bit_gather_kernel` at 8): every (row, stat) is
    `hist2_quantize(stat, scale, hist2_dither(position))`, added to its
    (stat, feature, bin) cell when the bin is one the writeback reads
    (`bin < folds`; a `bin == 1 << bits` skip mark is never below `folds`),
    and every add is an Int32 add, so the replica and group layout moves no
    bit. The cell is `write_reduces_from_fixed_kernel[False]`'s
    (`histogram_utils.mojo:774-805`): `ftz(Float32(Int(q)) / scale)` when
    `q != 0`, else 0.0."""
    var n_f = blk.count()
    var total = 0
    for k in range(n_f):
        total += Int(blk.folds[k])
    for j in range(len(compute_ids)):
        var slot = compute_ids[j]
        var off = p_off[slot]
        var sz = p_sz[slot]
        var acc = List[Int32](length=2 * total, fill=Int32(0))
        for pos in range(off, off + sz):
            var row = row_index[pos]
            var u = _hist2_dither(pos)
            var q0 = _hist2_quantize(stats[pos], fixed_scale, u)
            var q1 = _hist2_quantize(stats[n_rows + pos], fixed_scale, u)
            for k in range(n_f):
                ref cf = layout.features[blk.feature_ids[k]]
                var word = cindex[Int(cf.offset) * n_rows + row]
                var bin = Int((word >> cf.shift) & cf.mask)
                if bin < Int(blk.folds[k]):
                    var at = Int(blk.fold_offset[k]) + bin
                    acc[at] = acc[at] + q0
                    acc[total + at] = acc[total + at] + q1
        for z in range(2):
            for c in range(total):
                var q = acc[z * total + c]
                var val = Float32(0.0)
                if q != Int32(0):
                    val = ftz(Float32(Int(q)) / fixed_scale)
                hist[slot * 2 * hist_cells + z * hist_cells + block_first_bin + c] = val


def _half_byte_slot(ci: UInt32, tid: Int, i: Int) -> Int:
    """`slice_offset(tid) + add_point_slot(ci, tid, i)`
    (`point_hist_half_byte_template.mojo:182-269`)."""
    var f = (tid + i) & 7
    var bin = Int((ci >> UInt32(28 - 4 * f)) & UInt32(15))
    bin <<= 5
    bin += f
    return 512 * (tid // 32) + (tid & 24) + bin


def _half_byte_one_block(
    local_block_idx: Int,
    active_block_count: Int,
    p_offset: Int,
    p_size: Int,
    stat_id: Int,
    column: Int,
    row_index: List[Int],
    stats: List[Float32],
    cindex: List[UInt32],
    n_rows: Int,
) -> List[Float32]:
    """ONE grid block of `half_byte_hist_gather_kernel` (`hist_half_byte.mojo:
    592-930`; the direct kernel at depth 0 is the same arithmetic over the
    identity index), up to its `Reduce()`: returns the 128 stage-2 cells,
    `[feature + 8 * fold]`.

    THE ORDER OF THE FLOAT ADDS IS THE ONE THE BARRIERS FIX. Every thread of
    the block makes the same number of turn syncs (DEVIATION 2600's peel
    bound, the uniform striped count), so sync window `w` of every thread
    coincides: head point turns 0..7, tail point turns 8..15, then turns
    `16 + 8 * it + i`. Within one window the 8 lanes of a tile write 8
    distinct features, and different tiles write different replicas, so a
    cell is written by at most one thread per window, in that thread's
    point order. Looping window, then thread, then point reproduces every
    cell's add sequence, the 0.0 adds of rows a thread does not own
    included."""
    var smem = List[Float32](
        length=GBDT_HB_BLOCK * GBDT_HB_FLOATS, fill=Float32(0.0)
    )
    comptime ALIGN_SIZE = GBDT_HB_LOAD * GBDT_HB_LANES * 1
    var head_len = p_size
    var to_align = ALIGN_SIZE - (p_offset % ALIGN_SIZE)
    if to_align < head_len:
        head_len = to_align
    if head_len < 0:
        head_len = 0
    var body_size = p_size - head_len
    if body_size < 0:
        body_size = 0
    var tail_len = body_size % ALIGN_SIZE
    var tail_start = p_offset + head_len + (body_size - tail_len)

    # the peel (`hist_half_byte.mojo:698-736`): one trip per thread, since
    # PEEL_END is 512 and every thread starts at its own index
    for phase in range(2):
        var plen = head_len if phase == 0 else tail_len
        var pstart = p_offset if phase == 0 else tail_start
        for i in range(8):
            for t in range(GBDT_HB_BLOCK):
                var ci = UInt32(0)
                var st = Float32(0.0)
                if local_block_idx == 0 and t < plen:
                    var pos = pstart + t
                    ci = cindex[column * n_rows + row_index[pos]]
                    st = stats[stat_id * n_rows + pos]
                var s = _half_byte_slot(ci, t, i)
                smem[s] = ftz(smem[s] + st)

    # the striped loop (`:738-900`)
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len
    var warps_per_block = GBDT_HB_BLOCK // GBDT_HB_LANES
    var entries_per_warp = GBDT_HB_LANES * 1 * GBDT_HB_LOAD
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1
    var bases = List[Int](length=GBDT_HB_BLOCK, fill=0)
    var iter_counts = List[Int](length=GBDT_HB_BLOCK, fill=0)
    for t in range(GBDT_HB_BLOCK):
        var global_warp_id = local_block_idx * warps_per_block + (
            t // GBDT_HB_LANES
        )
        var remaining = aligned_size - global_warp_id * entries_per_warp
        if remaining < 0:
            remaining = 0
        var local_idx = (t & (GBDT_HB_LANES - 1)) * GBDT_HB_LOAD
        bases[t] = aligned_offset + global_warp_id * entries_per_warp + local_idx
        iter_counts[t] = (remaining - local_idx + stripe_size - 1) // stripe_size
    for it in range(max_iters):
        for i in range(8):
            for t in range(GBDT_HB_BLOCK):
                var active = it < iter_counts[t]
                for e in range(GBDT_HB_LOAD):
                    var ci = UInt32(0)
                    var st = Float32(0.0)
                    if active:
                        var pos = bases[t] + it * stripe_size + e
                        ci = cindex[column * n_rows + row_index[pos]]
                        st = stats[stat_id * n_rows + pos]
                    var s = _half_byte_slot(ci, t, i)
                    smem[s] = ftz(smem[s] + st)

    # `Reduce()` stage 1 (`:902-918`): each of 512 slots folds its 16
    # replicas ascending from 0.0; no thread reads a slot another writes
    for s in range(GBDT_HB_REDUCE_WIDTH):
        var acc = Float32(0.0)
        var i2 = s
        while i2 < GBDT_HB_BLOCK * GBDT_HB_FLOATS:
            acc = ftz(acc + smem[i2])
            i2 += GBDT_HB_REDUCE_WIDTH
        smem[s] = acc
    # stage 2 (`:920-930`): every read precedes every write
    var stage2 = List[Float32](length=128, fill=Float32(0.0))
    for t in range(128):
        var acc2 = Float32(0.0)
        for group in range(4):
            acc2 = ftz(acc2 + smem[32 * ((t >> 3) & 15) + (t & 7) + 8 * group])
        stage2[t] = acc2
    return stage2^


def _half_byte_block(
    blk: PolicyBlock,
    block_first_bin: Int,
    hist_cells: Int,
    compute_ids: List[Int],
    depth: Int,
    p_off: List[Int],
    p_sz: List[Int],
    row_index: List[Int],
    stats: List[Float32],
    cindex: List[UInt32],
    n_rows: Int,
    fixed_scale: Float32,
    mut hist: List[Float32],
):
    """A HALF-BYTE block: `launch_histograms_for_blocks`' half-byte arm
    (`greedy_search_helper.mojo:2907-2990`) over `replication_for`'s pinned
    grid (`:2547-2641`), then the flush of `AddToGlobalMemory`
    (`hist_half_byte.mojo:932-1040`) and the bridge
    `write_reduces_from_fixed_kernel[True]` (`histogram_utils.mojo:774-805`):

      active blocks > 1   q = sum over blocks with |val| > 1e-20 of
                          Int32(val * scale); the cell is
                          ftz(Float32(Int(q)) / scale) when q != 0, else the
                          memset scratch's 0.0
      one active block    val when |val| > 1e-20, else 0.0
      none (empty leaf)   0.0
    """
    var n_f = blk.count()
    var n_compute = len(compute_ids)
    var groups = (n_f + 7) // 8
    var max_active_blocks = 2 * GBDT_PINNED_SM
    if depth > 0:
        max_active_blocks = 2 * max_active_blocks
    var base_count = groups * n_compute * 2
    if base_count < 1:
        base_count = 1
    var replicas = (max_active_blocks + base_count - 1) // base_count
    if replicas < 1:
        replicas = 1
    var min_docs_per_block = GBDT_HB_LANES * 1 * GBDT_HB_LOAD * (
        GBDT_HB_BLOCK // GBDT_HB_LANES
    )
    for j in range(n_compute):
        var slot = compute_ids[j]
        var p_offset = p_off[slot]
        var p_size = p_sz[slot]
        var active_block_count = (p_size + min_docs_per_block - 1) // min_docs_per_block
        if active_block_count > replicas:
            active_block_count = replicas
        for z in range(2):
            for g in range(groups):
                var feature_offset = g * 8
                var f_count = n_f - feature_offset
                if f_count > 8:
                    f_count = 8
                var column = blk.first_column + g
                var vals = List[Float32](
                    length=128 * active_block_count, fill=Float32(0.0)
                )
                for lb in range(active_block_count):
                    var stage2 = _half_byte_one_block(
                        lb, active_block_count, p_offset, p_size, z, column,
                        row_index, stats, cindex, n_rows,
                    )
                    for t in range(128):
                        vals[lb * 128 + t] = stage2[t]
                for fid in range(f_count):
                    var folds = Int(blk.folds[feature_offset + fid])
                    var fold_off = Int(blk.fold_offset[feature_offset + fid])
                    for fold in range(folds):
                        var cell = Float32(0.0)
                        if active_block_count == 1:
                            var v = vals[fid + 8 * fold]
                            if abs(v) > Float32(1e-20):
                                cell = v
                        elif active_block_count > 1:
                            var q = Int32(0)
                            for lb in range(active_block_count):
                                var v = vals[lb * 128 + fid + 8 * fold]
                                if abs(v) > Float32(1e-20):
                                    q = q + Int32(v * fixed_scale)
                            if q != Int32(0):
                                cell = ftz(Float32(Int(q)) / fixed_scale)
                        hist[
                            slot * 2 * hist_cells + z * hist_cells
                            + block_first_bin + fold_off + fold
                        ] = cell


def _partition_stat(
    stats: List[Float32], line_size: Int, stat_id: Int, offset: Int, size: Int
) -> Float32:
    """`compute_partition_stats` for one (leaf, stat)
    (`partitions_reduce.mojo:104-227`, `partition_stats_chunks` pinned to
    (2 * 32 + n_stats - 1) // n_stats = 32 chunks at 2 stats): phase 1 strides
    each 512-thread block over the leaf, phase 2 folds one partial per chunk,
    each through the 512-lane halving tree."""
    comptime MAX_CHUNKS = (2 * GBDT_PINNED_SM + 2 - 1) // 2
    var partials = List[Float32](length=MAX_CHUNKS, fill=Float32(0.0))
    var stride = MAX_CHUNKS * GBDT_STATS_BLOCK
    for chunk in range(MAX_CHUNKS):
        var slab = List[Float32](length=GBDT_STATS_BLOCK, fill=Float32(0.0))
        for tid in range(GBDT_STATS_BLOCK):
            var v = Float32(0.0)
            var i = chunk * GBDT_STATS_BLOCK + tid
            while i < size:
                v += stats[stat_id * line_size + offset + i]
                i += stride
            slab[tid] = v
        partials[chunk] = _halving_fold(slab)
    var slab2 = List[Float32](length=GBDT_STATS_BLOCK, fill=Float32(0.0))
    for tid in range(GBDT_STATS_BLOCK):
        var acc = Float32(0.0)
        var c = tid
        while c < MAX_CHUNKS:
            acc += partials[c]
            c += GBDT_STATS_BLOCK
        slab2[tid] = acc
    return _halving_fold(slab2)


def _add_leaf_cosine(
    sum: Float32,
    weight: Float32,
    lambda_l2: Float32,
    mut score: Float32,
    mut denum_sqr: Float32,
):
    """`_add_leaf[SCORE_FUNCTION_COSINE, normalize=False, pin_mul_add=True]`
    (`compute_scores.mojo:46-86`)."""
    var lam = lambda_l2
    var mu = Float32(0.0)
    if weight > Float32(0.0):
        mu = sum / (weight + lam)
    mu = ftz(mu)
    score = ftz(identical_mul_add(sum, mu, score))
    denum_sqr = ftz(identical_mul_add(ftz(weight * mu), mu, denum_sqr))


def _cosine_gain(
    hist: List[Float32],
    hist_cells: Int,
    part_stats: List[Float32],
    n_live: Int,
    bin_feature_id: Int,
    lambda_l2: Float32,
    score_std_dev: Float32 = Float32(0.0),
    level_seed: UInt64 = 0,
    feature_id: Int = 0,
) -> Float32:
    """One bin-feature of `compute_optimal_splits_kernel[COSINE]`
    (`compute_scores.mojo:140-224`): the leaves in dense order, the clamped
    weights, the two `AddLeaf`s per leaf, the sqrt normalization, the score
    noise when `score_std_dev` is not zero (one normal draw per FEATURE off
    `advance_seed_k(level_seed + feature, 4)`, subtracted from the score and
    from the zero score-before by the pinned fma, `:204-219`; lane/catboost-
    parity), feature weight 1.0."""
    var score = Float32(0.0)
    var denum_sqr = Float32(1e-10)
    for i in range(n_live):
        var leaf_base = i * 2 * hist_cells
        var weight_left = max(hist[leaf_base + bin_feature_id], Float32(0.0))
        var weight_right = ftz(max(part_stats[i * 2] - weight_left, Float32(0.0)))
        var sum_left = hist[leaf_base + hist_cells + bin_feature_id]
        var part_stat = part_stats[i * 2 + 1]
        var sum_right = ftz(part_stat - sum_left)
        _add_leaf_cosine(sum_left, weight_left, lambda_l2, score, denum_sqr)
        _add_leaf_cosine(sum_right, weight_right, lambda_l2, score, denum_sqr)
    var final_score = score
    var score_before = Float32(0.0)
    if denum_sqr > Float32(1e-15):
        final_score = ftz(score / identical_sqrt(denum_sqr))
    else:
        final_score = -GBDT_FLOAT32_MAX
    if score_std_dev != Float32(0.0):
        var seed = advance_seed_k(level_seed + UInt64(feature_id), 4)
        var draw = next_normal_f(seed)
        var neg_draw = -draw[0]
        final_score = ftz(identical_mul_add(neg_draw, score_std_dev, final_score))
        score_before = ftz(identical_mul_add(neg_draw, score_std_dev, score_before))
    return ftz(ftz(final_score - score_before) * Float32(1.0))


# ===========================================================================
# THE BOOTSTRAP (`gbdt/gpu_util/kernel/bootstrap.mojo`)
# ===========================================================================


def gbdt_bootstrap_seeds(base_seed: UInt64) -> List[UInt64]:
    """`create_bootstrap_seeds` (`bootstrap.mojo:249-275`): splitmix64."""
    var seeds = List[UInt64](capacity=GBDT_BOOT_SEEDS)
    var x = base_seed
    for _ in range(GBDT_BOOT_SEEDS):
        x += UInt64(0x9E3779B97F4A7C15)
        var z = x
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        seeds.append(z)
    return seeds^


def _bootstrap_pass(
    kind: Int,
    mut seeds: List[UInt64],
    mut stats: List[Float32],
    n_rows: Int,
    param: Float32,
) raises -> Tuple[Float32, Float32]:
    """`launch_bootstrap` + `bootstrap_kernel` (`bootstrap.mojo:118-246`,
    `:292-348`) at two stat planes, then `deterministic_sum_lanes_kernel[2]`
    over the block magnitudes. Returns the two folded magnitudes."""
    var by_rows = (n_rows + GBDT_BOOT_BLOCK - 1) // GBDT_BOOT_BLOCK
    var blocks = GBDT_BOOT_SEEDS // GBDT_BOOT_BLOCK
    if by_rows < blocks:
        blocks = by_rows
    if blocks < 1:
        blocks = 1
    var stride = blocks * GBDT_BOOT_BLOCK
    var mag_part = List[Float32](length=2 * blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_w = List[Float32](length=GBDT_BOOT_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_BOOT_BLOCK, fill=Float32(0.0))
        for tid in range(GBDT_BOOT_BLOCK):
            var gid = b * GBDT_BOOT_BLOCK + tid
            var s = seeds[gid]
            var mag_w = Float32(0.0)
            var mag_g = Float32(0.0)
            var i = gid
            while i < n_rows:
                var bw: Float32
                if kind == GBDT_BOOT_BAYESIAN:
                    # `bootstrap_kernel[BAYESIAN]` (lane/catboost-parity):
                    # `-log(u + 1e-20)`, raised to the temperature unless 1
                    var draw = next_uniform_f(s)
                    s = draw[1]
                    var tmp = -identical_log(draw[0] + Float32(1e-20))
                    bw = tmp
                    if param != Float32(1.0):
                        bw = identical_pow(tmp, param)
                elif kind == GBDT_BOOT_BERNOULLI:
                    var draw = next_uniform_f(s)
                    s = draw[1]
                    bw = Float32(1.0) if draw[0] < param else Float32(0.0)
                elif kind == GBDT_BOOT_POISSON:
                    var draw = next_poisson_f(s, param)
                    s = draw[1]
                    bw = draw[0]
                else:
                    raise Error("gbdt host: bootstrap kind " + String(kind) + " is not restated")
                var w = stats[i] * bw
                stats[i] = w
                mag_w += abs(w)
                var g = stats[n_rows + i] * bw
                stats[n_rows + i] = g
                mag_g += abs(g)
                i += stride
            seeds[gid] = s
            s_w[tid] = mag_w
            s_g[tid] = mag_g
        mag_part[2 * b] = _halving_fold(s_w)
        mag_part[2 * b + 1] = _halving_fold(s_g)
    var mags = _deterministic_sum_lanes(mag_part, 2, blocks)
    return (mags[0], mags[1])


# ===========================================================================
# THE SCORE NOISE MAGNITUDE (`compute_target_std_dev`), shared by the
# symmetric, Depthwise and Lossguide host arms (moved here from
# gbdt_oracle_depthwise.mojo, lane/catboost-parity)
# ===========================================================================


def _target_std_dev(stats: List[Float32], n_rows: Int) -> Float64:
    """`compute_target_std_dev` (`greedy_search_helper.mojo:224-280`) over
    `compute_target_variance_kernel` (`compute_scores.mojo:274-318`) at stat
    count 2: `min(4 * 32, ceil(n / 512))` blocks of 512 threads striding the
    rows, the flushed per-thread accumulations of rows with weight above
    1e-15, the per-block halving folds stored flushed, the three-lane fold,
    then `sqrt(sum2 / (weight + 1e-100))` in double."""
    comptime B = 512
    var n_blocks = (n_rows + B - 1) // B
    if 4 * 32 < n_blocks:
        n_blocks = 4 * 32
    if n_blocks < 1:
        n_blocks = 1
    var stride = n_blocks * B
    var partials = List[Float32](length=3 * n_blocks, fill=Float32(0.0))
    for b in range(n_blocks):
        var s0 = List[Float32](length=B, fill=Float32(0.0))
        var s1 = List[Float32](length=B, fill=Float32(0.0))
        var s2 = List[Float32](length=B, fill=Float32(0.0))
        for tid in range(B):
            var weighted_sum = Float32(0.0)
            var weighted_sum2 = Float32(0.0)
            var total_weight = Float32(0.0)
            var i = B * b + tid
            while i < n_rows:
                var w = stats[i]
                if w > Float32(1e-15):
                    var wt = stats[n_rows + i]
                    weighted_sum = ftz(weighted_sum + wt)
                    weighted_sum2 = ftz(weighted_sum2 + ftz(ftz(wt * wt) / w))
                    total_weight = ftz(total_weight + w)
                i += stride
            s0[tid] = weighted_sum
            s1[tid] = weighted_sum2
            s2[tid] = total_weight
        partials[3 * b] = ftz(_halving_fold_512(s0))
        partials[3 * b + 1] = ftz(_halving_fold_512(s1))
        partials[3 * b + 2] = ftz(_halving_fold_512(s2))
    var l2 = _deterministic_sum_lanes(partials, 3, n_blocks)
    var sum2 = Float64(l2[1])
    var weight = Float64(l2[2])
    return sqrt(sum2 / (weight + 1e-100))


def _halving_fold_512(mut slab: List[Float32]) -> Float32:
    var step = len(slab) // 2
    while step > 0:
        for t in range(step):
            slab[t] = slab[t] + slab[t + step]
        step //= 2
    return slab[0]


# ===========================================================================
# THE LEAF ESTIMATOR: the oracle and the Newton walker
# ===========================================================================


def _oracle_move_to(
    new_point: List[Float32],
    mut current_point: List[Float32],
    bins: List[Int],
    mut g_cursor: List[Float32],
    n_rows: Int,
):
    """`BinOptimizedOracle.move_to` (`pointwise_oracle.mojo:331-403`) for a
    single-dimensional loss: the Float32 shift against the current point,
    then `add_bin_model_value_kernel` (`kernel_add_model_value.mojo:
    114-183`), a plain add per row."""
    var shift = List[Float32](length=len(new_point), fill=Float32(0.0))
    for i in range(len(new_point)):
        shift[i] = new_point[i] - current_point[i]
    for pos in range(n_rows):
        g_cursor[pos] = g_cursor[pos] + shift[bins[pos]]
    for i in range(len(new_point)):
        current_point[i] = new_point[i]


def _oracle_eval(
    g_target: List[Float32],
    g_cursor: List[Float32],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    border: Float32,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
):
    """`write_value_and_first_derivatives`' single-dimensional arm
    (`pointwise_oracle.mojo:458-573`): `cross_entropy_kernel[True, True]`
    (plane 0 `ftz(weight * (c - p))`, plane 1 `ftz(weight * p * (1 - p))`,
    one score partial per 256-thread block), `compute_partition_stats` per
    leaf over the gathered order, and the value as the HOST Float32 fold of
    the partials in block order."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    var weight = Float32(1.0)
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var r = _cross_entropy_row(g_target[i], g_cursor[i], border, weight)
                stats[i] = r.weighted_direction
                stats[n_rows + i] = r.weighted_scale
                s_score[t] = r.score
        fv[b] = _halving_fold(s_score)
    gradient.clear()
    cached_der2.clear()
    for leaf in range(len(sizes)):
        gradient.append(
            Float64(_partition_stat(stats, n_rows, 0, offsets[leaf], sizes[leaf]))
        )
        cached_der2.append(
            Float64(_partition_stat(stats, n_rows, 1, offsets[leaf], sizes[leaf]))
            + lambda_reg
        )
    var fv32 = Float32(0.0)
    for b in range(blocks):
        fv32 += fv[b]
    value = Float64(fv32)


def _diagonal_direction(
    gradient: List[Float64], hessian: List[Float64]
) -> List[Float32]:
    """`_diagonal_direction` (`descent_helpers.mojo:81-100`)."""
    comptime EPS_1E20F = Float64(Float32(1e-20))
    var direction = List[Float32]()
    for i in range(len(gradient)):
        if hessian[i] > 0:
            direction.append(Float32(gradient[i] / (hessian[i] + EPS_1E20F)))
        else:
            direction.append(Float32(0.0))
    return direction^


def _walker_move(
    point: List[Float32], direction: List[Float32], step: Float64
) -> List[Float32]:
    """`_move` (`descent_helpers.mojo:186-195`)."""
    var moved = List[Float32]()
    for i in range(len(point)):
        moved.append(Float32(Float64(point[i]) + step * Float64(direction[i])))
    return moved^


def _regularize(weights_cpu: List[Float64], mut point: List[Float32]):
    """`regularize` (`pointwise_oracle.mojo:954-965`), MinLeafWeight 1e-20."""
    for leaf in range(len(weights_cpu)):
        if weights_cpu[leaf] < 1e-20:
            point[leaf] = Float32(0.0)


def _estimate_leaves(
    targets: List[Float32],
    cursor: List[Float32],
    row_index: List[Int],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    border: Float32,
    l2_leaf_reg: Float32,
    iterations: Int,
) raises -> List[Float32]:
    """`_estimate_and_apply`'s estimate (`doc_parallel_boosting.mojo:
    698-797`): the gathers by the row index, the oracle, and
    `newton_like_walker_estimate` with AnyImprovement
    (`descent_helpers.mojo:198-285`, `step_estimator.mojo:50-72`)."""
    var n_leaves = len(sizes)
    var g_target = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    for pos in range(n_rows):
        g_target[pos] = targets[row_index[pos]]
        g_cursor[pos] = cursor[row_index[pos]]
    # `fill_bins_from_partition_kernel` (`kernel_add_model_value.mojo:186-204`)
    var bins = List[Int](length=n_rows, fill=0)
    for leaf in range(n_leaves):
        for k in range(sizes[leaf]):
            bins[offsets[leaf] + k] = leaf
    var weights_cpu = List[Float64]()
    for leaf in range(n_leaves):
        weights_cpu.append(Float64(sizes[leaf]))
    var lambda_reg = Float64(l2_leaf_reg)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0
    var current_point = List[Float32](length=n_leaves, fill=Float32(0.0))

    var cur_point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var cur_value = Float64(0.0)
    var cur_grad = List[Float64]()
    var cached_der2 = List[Float64]()
    _oracle_move_to(cur_point, current_point, bins, g_cursor, n_rows)
    _oracle_eval(
        g_target, g_cursor, offsets, sizes, n_rows, border, lambda_reg,
        cur_value, cur_grad, cached_der2,
    )
    var cur_hess = cached_der2.copy()
    var direction = _diagonal_direction(cur_grad, cur_hess)

    if iterations == 1:
        var result = _walker_move(cur_point, direction, 1.0)
        _regularize(weights_cpu, result)
        return result^

    var updated = False
    var iteration = 0
    while iteration < iterations:
        var function_value = cur_value
        var step = Float64(1.0)
        var accepted = False
        var next_value = Float64(0.0)
        var next_grad = List[Float64]()
        while iteration < iterations or ((not updated) and iteration < 100):
            var next_point = _walker_move(cur_point, direction, step)
            _regularize(weights_cpu, next_point)
            _oracle_move_to(next_point, current_point, bins, g_cursor, n_rows)
            _oracle_eval(
                g_target, g_cursor, offsets, sizes, n_rows, border, lambda_reg,
                next_value, next_grad, cached_der2,
            )
            if function_value <= next_value:
                cur_hess = cached_der2.copy()
                cur_point = next_point.copy()
                cur_value = next_value
                cur_grad = next_grad.copy()
                direction = _diagonal_direction(cur_grad, cur_hess)
                iteration += 1
                updated = True
                accepted = True
                break
            iteration += 1
            step /= 2
        if not accepted:
            break
    return cur_point^


# ===========================================================================
# THE FIT
# ===========================================================================


def gbdt_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    one_hot_in: List[Bool] = List[Bool](),
    bootstrap_kind: Int = -1,
    bootstrap_param: Float32 = Float32(1.0),
    random_strength: Float32 = Float32(0.0),
) raises -> GbdtHostModel:
    """`train` then `fit_with_test` on the covered configuration (see the
    module docstring for what that is and what mirrors what).

    `bootstrap_kind` (`GBDT_BOOT_*`, -1 none) and `random_strength` are the
    stochastic arm CatBoost's GPU defaults select (Bayesian at temperature
    1, strength 1; lane/catboost-parity, the gbdt-catboost-defaults lane):
    the per-tree `noise_rand` draw, `_bootstrap_pass` on both planes (whose
    magnitudes then set the fixed-point scale), the tree's
    `compute_target_std_dev` over the bootstrapped planes times the
    strength and `calc_score_model_length_mult`, one `level_rand` draw per
    level and the per-feature noise in `_cosine_gain`. The leaves are
    estimated on the learn target, not the bootstrapped planes, as the
    device's `_estimate_and_apply` does.

    `one_hot_in` (empty for none) names the ONE-HOT columns
    (gbdt/host/gbdt_oracle_onehot.mojo resolves them from the flags): their
    grid is `_quantize_training_columns`' one-hot arm (`gbdt/train.mojo:
    1990-2002`, borders `code + 0.5` below the largest code, folds `maxc + 1`
    or 0, AsIs), the layout carries the flag, the scan skips them and the
    split is an equality, as the device fit does."""
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if params.max_depth < 0:
        raise Error("max_depth must not be negative")

    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, params.border_count,
        params.border_build_max_samples, params.random_seed, params.nan_mode,
        params.border_type,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
    if len(one_hot_in) == n_features:
        for f in range(n_features):
            if not one_hot_in[f]:
                continue
            one_hot[f] = True
            var maxc = 0
            for r in range(n_rows):
                var c = Int(x_colmajor[f * n_rows + r])
                if c > maxc:
                    maxc = c
            if maxc > 254:
                raise Error("one-hot feature " + String(f) + " has more than 255 categories")
            var bs = List[Float32]()
            for c in range(maxc):
                bs.append(Float32(c) + Float32(0.5))
            grid.fold_counts[f] = len(bs) + 1 if len(bs) > 0 else 0
            grid.borders[f] = bs^
            grid.nan_treatment[f] = NAN_TREATMENT_AS_IS
    elif len(one_hot_in) != 0:
        raise Error("one_hot flags must be empty or one per feature")
    var layout = build_layout(grid.fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    for b in range(len(blocks)):
        if blocks[b].policy == POLICY_BINARY:
            raise Error(
                "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a"
                " feature with exactly one border (the BinaryFeatures"
                " histogram policy, feature "
                + String(blocks[b].feature_ids[0])
                + "); the gbdt host binding restates the half-byte and"
                " one-byte policies only (gbdt/host/gbdt_oracle.mojo)"
            )
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var hist_cells = layout.hist_cells

    # the flat bin-feature tables (`TTreeWorkspace.refresh_layout_metadata`,
    # `greedy_search_helper.mojo:3742-3798`)
    var bf_feature = List[Int](length=hist_cells, fill=0)
    var bf_bin = List[Int](length=hist_cells, fill=0)
    for f in range(n_features):
        ref lf = layout.features[f]
        for b in range(Int(lf.folds)):
            bf_feature[Int(lf.first_fold_index) + b] = f
            bf_bin[Int(lf.first_fold_index) + b] = b

    var max_depth = params.max_depth
    var max_leaves = 1 << max_depth
    var lr = params.learning_rate
    var border = params.logloss_border

    # `enqueue_fill(ctx, cursor, start_value)` with no boost from average
    # (`doc_parallel_boosting.mojo:1159-1164`)
    var cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mse_blocks, fill=Float32(0.0))

    var losses = List[Float64]()
    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()

    # the stochastic arm (`doc_parallel_boosting.mojo:1470-1523`): the
    # bootstrap seeds and the per-tree noise stream, both off `random_seed`
    var bootstrap_on = bootstrap_kind >= 0
    var boot_seeds = List[UInt64]()
    if bootstrap_on:
        boot_seeds = gbdt_bootstrap_seeds(params.random_seed)
    var noise_rand = TRandom(params.random_seed)

    for iteration in range(params.n_estimators):
        # ---- the gradients, the learn loss and the magnitudes ----
        _logloss_search_pass(y, cursor, n_rows, border, stats, fv_part, mag_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
        # `calc_score_model_length_mult` (`random_score_helper.mojo:
        # 219-243`, host libm as theirs) and the per-tree seed, drawn every
        # tree whether or not the noise is on
        var noise_mult = Float64(0.0)
        if random_strength != Float32(0.0):
            var model_left = exp(
                log(Float64(n_rows))
                - Float64(iteration) * Float64(params.learning_rate)
            )
            noise_mult = model_left / (1.0 + model_left)
        var tree_seed = noise_rand.next_uniform_l()
        if bootstrap_on:
            var bm = _bootstrap_pass(
                bootstrap_kind, boot_seeds, stats, n_rows, bootstrap_param
            )
            mags[0] = bm[0]
            mags[1] = bm[1]
        var fixed_scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        # `run_tree_layout`'s ScoreStdDev (`greedy_search_helper.mojo:
        # 5038-5051`) over the bootstrapped planes, and its level stream
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            score_std_dev = Float32(
                Float64(Float32(noise_mult * Float64(random_strength)))
                * _target_std_dev(stats, n_rows)
            )
        var level_rand = TRandom(tree_seed)

        # ---- `run_tree_layout_traced`, every level (TWIN: the loop in
        # `gbdt_oracle_rmse.mojo::gbdt_rmse_host_fit`; edit both) ----
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[r] = r
        var p_off = List[Int](length=max_leaves, fill=0)
        var p_sz = List[Int](length=max_leaves, fill=0)
        p_sz[0] = n_rows
        var hist = List[Float32](
            length=max_leaves * 2 * hist_cells, fill=Float32(0.0)
        )
        var ids_compute = List[Int](length=max_leaves, fill=0)
        var sub_from = List[Int](length=max_leaves, fill=0)
        var sub_what = List[Int](length=max_leaves, fill=0)
        var winners_score = List[Float32]()
        var winners_bf = List[UInt32]()
        var n_live = 1
        for depth in range(max_depth):
            # `Random.NextUniformL()`, one draw per level before the launch
            var level_seed = level_rand.next_uniform_l()
            var half = n_live // 2
            var planned = depth > 0
            var compute = List[Int]()
            if planned:
                for j in range(half):
                    compute.append(ids_compute[j])
            else:
                for j in range(n_live):
                    compute.append(j)

            var block_first_bin = 0
            for b in range(len(blocks)):
                ref blk = blocks[b]
                var total = 0
                for k in range(blk.count()):
                    total += Int(blk.folds[k])
                if blk.policy == POLICY_HALF_BYTE:
                    _half_byte_block(
                        blk, block_first_bin, hist_cells, compute, depth,
                        p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_ONE_BYTE:
                    _one_byte_block(
                        blk, block_first_bin, hist_cells, compute, p_off, p_sz,
                        row_index, stats, cindex, layout, n_rows, fixed_scale,
                        hist,
                    )
                block_first_bin += total

            # `scan_histograms_kernel` over the computed leaves
            for j in range(len(compute)):
                var slot = compute[j]
                for z in range(2):
                    for f in range(n_features):
                        ref cf = layout.features[f]
                        var folds = Int(cf.folds)
                        if cf.one_hot_feature or folds <= 1:
                            continue
                        var base = (
                            slot * 2 * hist_cells + z * hist_cells
                            + Int(cf.first_fold_index)
                        )
                        var running = Float32(0.0)
                        for i in range(folds):
                            running = ftz(running + hist[base + i])
                            hist[base + i] = running

            # `substract_histograms_kernel`: the larger sibling, in place
            if planned and half > 0:
                for j in range(half):
                    var from_slot = sub_from[j]
                    var what_slot = sub_what[j]
                    for z in range(2):
                        var from_base = from_slot * 2 * hist_cells + z * hist_cells
                        var what_base = what_slot * 2 * hist_cells + z * hist_cells
                        for bf in range(hist_cells):
                            var new_val = ftz(hist[from_base + bf] - hist[what_base + bf])
                            if z == 0:
                                new_val = max(new_val, Float32(0.0))
                            hist[from_base + bf] = new_val

            # `compute_partition_stats` over the live leaves
            var part_stats = List[Float32](length=2 * n_live, fill=Float32(0.0))
            for i in range(n_live):
                part_stats[2 * i] = _partition_stat(stats, n_rows, 0, p_off[i], p_sz[i])
                part_stats[2 * i + 1] = _partition_stat(stats, n_rows, 1, p_off[i], p_sz[i])

            # the score and the device winner
            var best_gain = -GBDT_FLOAT32_MAX
            var best_bin = GBDT_SENTINEL
            for bf in range(hist_cells):
                var gain = _cosine_gain(
                    hist, hist_cells, part_stats, n_live, bf, params.l2_leaf_reg,
                    score_std_dev, level_seed, bf_feature[bf],
                )
                if gain > best_gain:
                    best_gain = gain
                    best_bin = UInt32(bf)
            if best_bin != GBDT_SENTINEL:
                winners_score.append(ftz(best_gain))
            else:
                winners_score.append(ftz(-GBDT_FLOAT32_MAX))
            winners_bf.append(best_bin)
            var bf_split = 0
            if best_bin != GBDT_SENTINEL:
                bf_split = Int(best_bin)

            # the split chain, every live leaf
            var split_f = bf_feature[bf_split] if hist_cells > 0 else 0
            var split_b = bf_bin[bf_split] if hist_cells > 0 else 0
            ref sfeat = layout.features[split_f]
            var new_rows = row_index.copy()
            var new_stats = stats.copy()
            for i in range(n_live):
                var off = p_off[i]
                var sz = p_sz[i]
                var zeros = List[Int]()
                var ones = List[Int]()
                for k in range(sz):
                    var row = row_index[off + k]
                    var word = cindex[Int(sfeat.offset) * n_rows + row]
                    var feature_val = word & (sfeat.mask << sfeat.shift)
                    var value = UInt32(split_b) << sfeat.shift
                    var goes_right: Bool
                    if sfeat.one_hot_feature:
                        goes_right = feature_val == value
                    else:
                        goes_right = feature_val > value
                    if goes_right:
                        ones.append(k)
                    else:
                        zeros.append(k)
                var dst = 0
                for k in range(len(zeros)):
                    var src = off + zeros[k]
                    new_rows[off + dst] = row_index[src]
                    new_stats[off + dst] = stats[src]
                    new_stats[n_rows + off + dst] = stats[n_rows + src]
                    dst += 1
                for k in range(len(ones)):
                    var src = off + ones[k]
                    new_rows[off + dst] = row_index[src]
                    new_stats[off + dst] = stats[src]
                    new_stats[n_rows + off + dst] = stats[n_rows + src]
                    dst += 1
                # `copy_histograms_kernel`: parent into the right slot
                var src_base = i * 2 * hist_cells
                var dst_base = (n_live + i) * 2 * hist_cells
                for c in range(2 * hist_cells):
                    hist[dst_base + c] = hist[src_base + c]
                # `update_partitions_and_plan_kernel`
                var left_sz = len(zeros)
                var right_sz = sz - left_sz
                p_sz[i] = left_sz
                p_off[n_live + i] = off + left_sz
                p_sz[n_live + i] = right_sz
                var small = n_live + i
                var big = i
                if left_sz < right_sz:
                    small = i
                    big = n_live + i
                ids_compute[i] = small
                sub_from[i] = big
                sub_what[i] = small
            row_index = new_rows^
            stats = new_stats^
            n_live = n_live * 2

        # ---- the gates, post-tree, and the rollback ----
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        var grown = 0
        for d in range(max_depth):
            var best_score = winners_score[d]
            var best_bin_u = winners_bf[d]
            if best_bin_u == GBDT_SENTINEL or Int(best_bin_u) >= hist_cells:
                raise Error(
                    "All splits have infinite score. Probably, numerical"
                    " overflow occurs in loss function and/or split score"
                    " calculation. Try increasing l2_leaf_reg, and/or"
                    " decreasing learning_rate, etc."
                    " [level " + String(d) + ", live leaves "
                    + String(1 << d) + "]"
                )
            var cf_ = bf_feature[Int(best_bin_u)]
            var cb_ = bf_bin[Int(best_bin_u)]
            if not (best_score > Float32(0.0)):
                break
            var repeated = False
            for i in range(len(tree_features)):
                if tree_features[i] == cf_ and tree_bins[i] == cb_:
                    repeated = True
            if repeated:
                break
            tree_features.append(cf_)
            tree_bins.append(cb_)
            grown += 1
        if grown < max_depth:
            var live = n_live
            for _ in range(max_depth - grown):
                var h2 = live // 2
                for i in range(h2):
                    p_sz[i] = p_sz[i] + p_sz[h2 + i]
                live = h2
            n_live = live
        var sizes = List[Int]()
        var offsets = List[Int]()
        var covered_rows = 0
        for i in range(n_live):
            sizes.append(p_sz[i])
            offsets.append(p_off[i])
            covered_rows += p_sz[i]
        if covered_rows != n_rows:
            raise Error(
                "gbdt host: the final leaf partitions cover "
                + String(covered_rows) + " of " + String(n_rows) + " rows"
            )

        # ---- the estimation task and `AppendModels` ----
        var estimated = _estimate_leaves(
            y, cursor, row_index, offsets, sizes, n_rows, border,
            params.l2_leaf_reg, params.leaf_estimation_iterations,
        )
        for leaf in range(n_live):
            for k in range(sizes[leaf]):
                var row = row_index[offsets[leaf] + k]
                cursor[row] = identical_mul_add(estimated[leaf], lr, cursor[row])
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(estimated)):
            model_leaves.append(estimated[i] * lr)
        tree_leaf_offsets.append(len(model_leaves))

        # the learn loss read alongside this iteration's gradients
        if len(losses) < params.n_estimators:
            var v = Float64(fv)
            if iteration + 1 > 1:
                losses.append(-v / Float64(n_rows))

    losses.append(-Float64(_logloss_value(y, cursor, n_rows, border)) / Float64(n_rows))
    return GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, 0, False,
    )


# ===========================================================================
# THE MODEL TEXT (`gbdt/models/model_text.mojo`)
# ===========================================================================


def _hex_fixed(v: UInt64, digits: Int) -> String:
    """`_hex_fixed` (`model_text.mojo:268-274`)."""
    var table = String("0123456789abcdef")
    var out = String("")
    for i in range(digits):
        var nib = Int((v >> UInt64((digits - 1 - i) * 4)) & UInt64(0xF))
        out += String(table[byte=nib])
    return out^


def gbdt_f32_token(v: Float32) -> String:
    """`f32_token` (`model_text.mojo:303-307`)."""
    return String(v) + "/" + _hex_fixed(UInt64(bitcast[DType.uint32](v)), 8)


def gbdt_f64_token(v: Float64) -> String:
    """`f64_token` (`model_text.mojo:310-312`)."""
    return String(v) + "/" + _hex_fixed(bitcast[DType.uint64](v), 16)


def _nan_token(treatment: Int) raises -> String:
    """`nan_treatment_token` (`model_text.mojo:232-244`)."""
    if treatment == NAN_TREATMENT_AS_IS:
        return String("as_is")
    if treatment == NAN_TREATMENT_AS_FALSE:
        return String("as_false")
    if treatment == NAN_TREATMENT_AS_TRUE:
        return String("as_true")
    raise Error("unknown nan treatment " + String(treatment))


def gbdt_host_model_text(m: GbdtHostModel) raises -> String:
    """`model_text` (`model_text.mojo:374-670`) for a float-only oblivious
    one-dimensional model with zero bias: the header comment, the four
    header records, one `feature` record per column (one-hot flags carried,
    all 0, as `train` hands `column_one_hot`), then `tree`/`split`/`leaf`
    records and the `loss` records."""
    var n_features = len(m.fold_counts)
    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += String("format ") + String("mojolearn-model") + " " + String(2) + "\n"
    out += (
        String("features ") + String(n_features) + " "
        + String(n_features) + "\n"
    )
    out += String("trees ") + String(m.n_trees()) + "\n"
    out += String("losses ") + String(len(m.losses)) + "\n"
    for f in range(n_features):
        var line = (
            String("feature ") + String(f) + " folds "
            + String(m.fold_counts[f]) + " one_hot " + String(0)
            + " type " + String("float")
            + " nan " + _nan_token(m.nan_treatment[f])
            + " borders " + String(len(m.borders[f]))
        )
        for b in range(len(m.borders[f])):
            line += " " + gbdt_f32_token(m.borders[f][b])
        out += line + "\n"
    for t in range(m.n_trees()):
        var lo = m.tree_split_offsets[t]
        var depth = m.tree_split_offsets[t + 1] - lo
        var leaf_lo = m.tree_leaf_offsets[t]
        var n_values = (1 << depth) * 1
        if m.tree_leaf_offsets[t + 1] - leaf_lo != n_values:
            raise Error(
                "tree " + String(t) + " has depth " + String(depth)
                + ", dim 1 and "
                + String(m.tree_leaf_offsets[t + 1] - leaf_lo)
                + " leaf values, not " + String(n_values)
            )
        out += (
            String("tree ") + String(t) + " depth " + String(depth)
            + " dim " + String(1) + " weights " + String(0) + "\n"
        )
        for level in range(depth):
            out += (
                String("split ") + String(t) + " " + String(level) + " "
                + String(m.split_features[lo + level]) + " "
                + String(m.split_bins[lo + level]) + "\n"
            )
        for i in range(n_values):
            out += (
                String("leaf ") + String(t) + " " + String(i) + " "
                + gbdt_f32_token(m.leaf_values[leaf_lo + i]) + "\n"
            )
    for i in range(len(m.losses)):
        out += String("loss ") + String(i) + " " + gbdt_f64_token(m.losses[i]) + "\n"
    return out^
