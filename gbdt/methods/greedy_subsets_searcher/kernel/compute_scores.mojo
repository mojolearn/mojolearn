# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Score every candidate split of a level and pick the best. A kernel that re-derives the running prefix per candidate must walk bins in order, which forces the bin loop innermost, which pushes the leaf loop outward."""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.targets.kernel.pointwise_targets import pinned_block_sum
from checks.kernel_matrix import partition_chunks_sm_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,  # DEVIATION 258: row 10, NVIDIA sqrt is approximate
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
)


comptime SCORE_BLOCK_SIZE = 128

comptime FLOAT32_MAX = Float32(3.4028234663852886e38)


@always_inline
def _add_leaf[
    score_function: Int,
    normalize: Bool,
    pin_mul_add: Bool = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL),
](
    sum: Float32,
    weight: Float32,
    lambda_l2: Float32,
    mut score: Float32,
    mut denum_sqr: Float32,
):
    """`calcer.AddLeaf(sum, weight)` -- one score calcer's accumulation."""
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )

    comptime if cosine:
        var lam = lambda_l2
        comptime if normalize:
            lam = ftz(lambda_l2 * weight)
        var mu = Float32(0.0)
        if weight > Float32(0.0):
            mu = sum / (weight + lam)

        comptime if pin_mul_add:
            mu = ftz(mu)
            score = ftz(identical_mul_add(sum, mu, score))
            denum_sqr = ftz(
                identical_mul_add(ftz(weight * mu), mu, denum_sqr)
            )
        else:
            score += sum * mu
            denum_sqr += weight * mu * mu
    else:
        if weight > Float32(1e-20):
            comptime if pin_mul_add:
                var num = ftz(sum * sum)
                score = ftz(score + ftz(num / (weight + lambda_l2)))
            else:
                score += (sum * sum) / (weight + lambda_l2)


def compute_optimal_splits_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    p_count_in: Int32,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplits`, with the block argmax that follows it."""
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )
    comptime assert (
        cosine
        or score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ), (
        "score_function has no calcer here; only Cosine, NewtonCosine, L2"
        " and NewtonL2 are ported"
    )

    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var p_count = Int(p_count_in)
    var tid = Int(thread_idx.x)

    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    var offset = Int(block_idx.x) * SCORE_BLOCK_SIZE
    while offset < bin_feature_count:
        var bin_feature_id = offset + tid
        if bin_feature_id >= bin_feature_count:
            break
        if bf_skip.unsafe_load(bin_feature_id) != 0:
            offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)
            continue

        var score = Float32(0.0)
        var denum_sqr = Float32(1e-10)


        for i in range(p_count):
            var leaf_id = Int(ldg(part_ids + i))
            var leaf_base = leaf_id * stat_count * bin_feature_count

            var weight_left = max(
                ldg(histograms + (leaf_base + bin_feature_id)),
                Float32(0.0),
            )
            var weight_right = ftz(
                max(
                    ldg(part_stats + leaf_id * stat_count) - weight_left,
                    Float32(0.0),
                )
            )

            var total_sum_left = Float32(0.0)
            var total_sum_part = Float32(0.0)

            for stat_id in range(1, stat_count):
                var stat_slot = (
                    leaf_base
                    + stat_id * bin_feature_count
                    + bin_feature_id
                )
                var sum_left = ldg(histograms + stat_slot)
                var part_stat = ldg(
                    part_stats + (leaf_id * stat_count + stat_id)
                )
                var sum_right = ftz(part_stat - sum_left)

                _add_leaf[score_function, normalize](
                    sum_left, weight_left, lambda_l2, score, denum_sqr
                )
                _add_leaf[score_function, normalize](
                    sum_right, weight_right, lambda_l2, score, denum_sqr
                )
                total_sum_left = ftz(total_sum_left + sum_left)
                total_sum_part = ftz(total_sum_part + part_stat)

            if multiclass_optimization != Int32(0):
                var total_sum_right = ftz(total_sum_part - total_sum_left)
                _add_leaf[score_function, normalize](
                    -total_sum_left, weight_left, lambda_l2,
                    score, denum_sqr,
                )
                _add_leaf[score_function, normalize](
                    -total_sum_right, weight_right, lambda_l2,
                    score, denum_sqr,
                )

        var feature_id = Int(bf_feature_id.unsafe_load(bin_feature_id))

        var final_score = score
        var score_before = Float32(0.0)
        comptime if cosine:
            if denum_sqr > Float32(1e-15):
                final_score = ftz(score / identical_sqrt(denum_sqr))
            else:
                final_score = -FLOAT32_MAX

            if score_std_dev != Float32(0.0):
                var seed = advance_seed_k(
                    global_seed + UInt64(feature_id), 4
                )
                var draw = next_normal_f(seed)
                var neg_draw = -draw[0]
                final_score = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, final_score
                    )
                )
                score_before = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, score_before
                    )
                )

        var gain = ftz(
            ftz(final_score - score_before)
            * ldg(feature_weights + feature_id)
        )

        if gain > best_gain:
            best_gain = gain
            best_bin = UInt32(bin_feature_id)

        offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)

    var s_score = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    s_score[tid] = best_gain
    s_bin[tid] = best_bin
    barrier()

    var stride = SCORE_BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            var take = s_score[tid + stride] > s_score[tid]
            if s_score[tid + stride] == s_score[tid]:
                if s_bin[tid + stride] < s_bin[tid]:
                    take = True
            if take:
                s_score[tid] = s_score[tid + stride]
                s_bin[tid] = s_bin[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        var index = s_bin[0]
        if index != UInt32(0xFFFFFFFF) and Int(index) < bin_feature_count:
            out_score.unsafe_store(Int(block_idx.x), ftz(s_score[0]))
            out_bin.unsafe_store(Int(block_idx.x), index)
        else:
            out_score.unsafe_store(Int(block_idx.x), ftz(-FLOAT32_MAX))
            out_bin.unsafe_store(Int(block_idx.x), UInt32(0xFFFFFFFF))


comptime TARGET_VARIANCE_BLOCK = 512


def compute_target_variance_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    stat_count_in: Int32,
    stat_line_size_in: Int32,
    is_multiclass: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """`ComputeTargetVarianceImpl` (`compute_scores.cu:226-283`). Metal has no fp64 and this repository does not accept a float atomic on the fit path (`deterministic_sum_lanes_kernel`'s docstring), so each block STORES three Float32 partials at its own slot and the fold is deterministic."""
    var size = Int(size_in)
    var stat_count = Int(stat_count_in)
    var stat_line_size = Int(stat_line_size_in)

    var i = TARGET_VARIANCE_BLOCK * Int(block_idx.x) + Int(thread_idx.x)

    var weighted_sum = Float32(0.0)
    var weighted_sum2 = Float32(0.0)
    var total_weight = Float32(0.0)

    while i < size:
        var w = stats.unsafe_load(i)
        if w > Float32(1e-15):
            var stat_sum = Float32(0.0)
            for stat_id in range(1, stat_count):
                var wt = stats.unsafe_load(i + stat_line_size * stat_id)
                weighted_sum = ftz(weighted_sum + wt)
                weighted_sum2 = ftz(
                    weighted_sum2 + ftz(ftz(wt * wt) / w)
                )
                stat_sum = ftz(stat_sum + wt)
            if is_multiclass != Int32(0):
                weighted_sum = ftz(weighted_sum + -stat_sum)
                weighted_sum2 = ftz(
                    weighted_sum2 + ftz(ftz(stat_sum * stat_sum) / w)
                )
            total_weight = ftz(total_weight + w)
        i += Int(grid_dim.x) * TARGET_VARIANCE_BLOCK

    var b_sum = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        weighted_sum
    )
    var b_sum2 = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        weighted_sum2
    )
    var b_weight = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        total_weight
    )

    if thread_idx.x == 0:
        var slot = 3 * Int(block_idx.x)
        partials.unsafe_store(slot, ftz(b_sum))
        partials.unsafe_store(slot + 1, ftz(b_sum2))
        partials.unsafe_store(slot + 2, ftz(b_weight))


def target_variance_blocks(size: Int, sm_count: Int) -> Int:
    """`min(4 * TArchProps::SMCount(), CeilDivide(size, blockSize))` (`compute_scores.cu:291`). Inert at this port's default `random_strength = 0`; CatBoost's default is 1.0, so the row must hold before that default is wired."""
    comptime _identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var sm = partition_chunks_sm_for[_identical](sm_count)
    var by_data = (size + TARGET_VARIANCE_BLOCK - 1) // TARGET_VARIANCE_BLOCK
    var by_machine = 4 * sm
    if by_machine < 1:
        by_machine = 1
    if by_data < by_machine:
        return by_data if by_data > 0 else 1
    return by_machine




comptime LEAFWISE_SCORE_BLOCK_SIZE = 256


@always_inline
def _leafwise_scan_part[
    score_function: Int, normalize: Bool, block_size: Int
](
    this_part_id: Int,
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count: Int,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count: Int,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    mut best_gain: Float32,
    mut best_bin: UInt32,
):
    """One leaf's candidate scan -- `compute_scores.cu:406-472`, both copies."""
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )
    comptime assert (
        cosine
        or score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ), (
        "score_function has no calcer here; only Cosine, NewtonCosine, L2"
        " and NewtonL2 are ported"
    )

    var tid = Int(thread_idx.x)
    var leaf_base = this_part_id * stat_count * bin_feature_count

    var offset = Int(block_idx.x) * block_size
    while offset < bin_feature_count:
        var bin_feature_id = offset + tid
        if bin_feature_id >= bin_feature_count:
            break
        if bf_skip.unsafe_load(bin_feature_id) != 0:
            offset += block_size * Int(grid_dim.x)
            continue

        var score = Float32(0.0)
        var denum_sqr = Float32(1e-10)
        var score_b = Float32(0.0)
        var denum_sqr_b = Float32(1e-10)

        var part_weight = ldg(part_stats + this_part_id * stat_count)
        var weight_left = max(
            ldg(histograms + (leaf_base + bin_feature_id)), Float32(0.0)
        )
        var weight_right = ftz(max(part_weight - weight_left, Float32(0.0)))

        var to_zero_part_split = (
            weight_left < Float32(1e-20) or weight_right < Float32(1e-20)
        )

        var total_sum_left = Float32(0.0)
        var total_sum_part = Float32(0.0)

        for stat_id in range(1, stat_count):
            var stat_slot = (
                leaf_base + stat_id * bin_feature_count + bin_feature_id
            )
            var sum_left = ldg(histograms + stat_slot)
            var part_stat = ldg(
                part_stats + (this_part_id * stat_count + stat_id)
            )
            total_sum_part = ftz(total_sum_part + part_stat)
            var sum_right = ftz(part_stat - sum_left)

            _add_leaf[score_function, normalize, True](
                sum_left, weight_left, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                sum_right, weight_right, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                part_stat, part_weight, lambda_l2, score_b, denum_sqr_b
            )
            total_sum_left = ftz(total_sum_left + sum_left)

        if multiclass_optimization != Int32(0):
            var total_sum_right = ftz(total_sum_part - total_sum_left)
            _add_leaf[score_function, normalize, True](
                -total_sum_left, weight_left, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                -total_sum_right, weight_right, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                -total_sum_part, part_weight, lambda_l2,
                score_b, denum_sqr_b,
            )

        var feature_id = Int(bf_feature_id.unsafe_load(bin_feature_id))

        var final_score = score
        var score_before = score_b
        comptime if cosine:
            if denum_sqr > Float32(1e-15):
                final_score = ftz(score / identical_sqrt(denum_sqr))
            else:
                final_score = -FLOAT32_MAX
            if denum_sqr_b > Float32(1e-15):
                score_before = ftz(score_b / identical_sqrt(denum_sqr_b))
            else:
                score_before = -FLOAT32_MAX

            if score_std_dev != Float32(0.0):
                var seed = advance_seed_k(
                    global_seed + UInt64(feature_id), 4
                )
                var draw = next_normal_f(seed)
                var neg_draw = -draw[0]
                final_score = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, final_score
                    )
                )
                score_before = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, score_before
                    )
                )

        if to_zero_part_split:
            final_score = -FLOAT32_MAX
            score_before = -FLOAT32_MAX

        var gain = Float32(0.0)
        if not to_zero_part_split:
            gain = ftz(final_score - score_before)
        gain = ftz(gain * ldg(feature_weights + feature_id))

        if gain > best_gain:
            best_gain = gain
            best_bin = UInt32(bin_feature_id)

        offset += block_size * Int(grid_dim.x)


@always_inline
def _leafwise_argmax_write[
    block_size: Int
](
    best_gain: Float32,
    best_bin: UInt32,
    bin_feature_count: Int,
    out_slot: Int,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """Their `ARGMAX()` macro (`compute_scores.cu:20-51`) for the leafwise kernels, with the same tie rule and the same poison record."""
    var tid = Int(thread_idx.x)
    var s_score = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        block_size,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    s_score[tid] = best_gain
    s_bin[tid] = best_bin
    barrier()

    var stride = block_size // 2
    while stride > 0:
        if tid < stride:
            var take = s_score[tid + stride] > s_score[tid]
            if s_score[tid + stride] == s_score[tid]:
                if s_bin[tid + stride] < s_bin[tid]:
                    take = True
            if take:
                s_score[tid] = s_score[tid + stride]
                s_bin[tid] = s_bin[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        var index = s_bin[0]
        if index != UInt32(0xFFFFFFFF) and Int(index) < bin_feature_count:
            out_score.unsafe_store(out_slot, ftz(s_score[0]))
            out_bin.unsafe_store(out_slot, index)
        else:
            out_score.unsafe_store(out_slot, ftz(-FLOAT32_MAX))
            out_bin.unsafe_store(out_slot, UInt32(0xFFFFFFFF))


def compute_optimal_split_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_id_in: Int32,
    maybe_second_part_id_in: Int32,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplit` (`compute_scores.cu:393-475`) -- LOSSGUIDE."""
    var this_part_id = Int(part_id_in)
    if Int(block_idx.y) != 0:
        this_part_id = Int(maybe_second_part_id_in)

    var bin_feature_count = Int(bin_feature_count_in)
    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    _leafwise_scan_part[
        score_function, normalize, LEAFWISE_SCORE_BLOCK_SIZE
    ](
        this_part_id,
        bf_skip,
        bin_feature_count,
        bf_feature_id,
        feature_weights,
        histograms,
        part_stats,
        Int(stat_count_in),
        multiclass_optimization,
        lambda_l2,
        score_std_dev,
        global_seed,
        best_gain,
        best_bin,
    )

    _leafwise_argmax_write[LEAFWISE_SCORE_BLOCK_SIZE](
        best_gain,
        best_bin,
        bin_feature_count,
        Int(block_idx.x) + Int(block_idx.y) * Int(grid_dim.x),
        out_score,
        out_bin,
    )


def compute_optimal_splits_region_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplitsRegion` (`compute_scores.cu:303-385`) -- DEPTHWISE."""
    var this_part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))

    var bin_feature_count = Int(bin_feature_count_in)
    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    _leafwise_scan_part[
        score_function, normalize, LEAFWISE_SCORE_BLOCK_SIZE
    ](
        this_part_id,
        bf_skip,
        bin_feature_count,
        bf_feature_id,
        feature_weights,
        histograms,
        part_stats,
        Int(stat_count_in),
        multiclass_optimization,
        lambda_l2,
        score_std_dev,
        global_seed,
        best_gain,
        best_bin,
    )

    _leafwise_argmax_write[LEAFWISE_SCORE_BLOCK_SIZE](
        best_gain,
        best_bin,
        bin_feature_count,
        Int(block_idx.x) + Int(block_idx.y) * Int(grid_dim.x),
        out_score,
        out_bin,
    )
