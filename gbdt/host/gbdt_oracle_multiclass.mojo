# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the two multi-output losses,
MultiClass and MultiClassOneVsAll, on symmetric trees
(lane/cpu-training-gbdt-losses, 2026-09-15): the gbdt-multiclass and
gbdt-onevsall lanes of tools/identity_break.py.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
the GPU-free host modules the symmetric oracle already reuses, CatBoost's
dense Cholesky solve (`gbdt/lapack/linear_system.mojo`, which the device
walker runs on the host too) and the symmetric oracle itself.

THE CONFIGURATIONS THIS COVERS, by name. `gbdt-multiclass`: 20 depth-6
trees, MultiClass on the three-class target with class weights
[1, 2, 0.5]. `gbdt-onevsall`: 20 depth-6 trees, MultiClassOneVsAll on the
same target. The public stochastic defaults are also supported: Bayesian,
Bernoulli and Poisson bootstrap scale every derivative plane; the score
noise variance reconstructs MultiClass's pinned derivative plane. Leaves are Newton at
one iteration (`catboost_options.mojo:1314-1321`); the binding refuses by
name what is outside.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build defaults)

  1. `train`'s class count and class weights (`gbdt/train.mojo:1512-1552`):
     the weight plane `class_weights[class]`, `has_weights` when given.
  2. The grid, the layout and the binarize: the symmetric oracle's.
  3. Per tree, the search pass: `multilogit_val_and_first_der_kernel
     [1, search=True]` for MultiClass and `one_vs_all_val_and_first_der_
     kernel[1, True]` for OneVsAll (`gbdt/targets/kernel/multilogit.mojo`):
     plane 0 the weight, planes 1.. the class ders, the score partials, and
     the magnitudes (`sum |w|`, `sum max_k |der_k|`, DEVIATION 79) through
     the 256-lane halving tree and the two-lane fold.
  4. `run_tree_layout_traced` at `stat_count = 1 + dim` (a TWIN of the loop
     in `gbdt_oracle.mojo::gbdt_host_fit`, every plane): the one-byte
     blocks' per-row dithered Int32 sums per plane (the pair kernels and
     the odd prelude alike, `greedy_search_helper.mojo:2388-2520`), the
     half-byte blocks with `replication_for` over `stat_count`, the scan
     and the subtraction per plane, `compute_partition_stats` at
     `(64 + stat_count - 1) // stat_count` chunks, and
     `compute_optimal_splits_kernel[COSINE]` with the per-class loop and,
     for MultiClass, the multiclass optimization terms
     (`kernel/compute_scores.mojo:89-266`).
  5. The estimation (`doc_parallel_boosting.mojo:622-850`,
     `pointwise_oracle.mojo`): the gathers, `WeightsCpu` (the leaf sizes,
     or the one-stat partition reduce of the weights), the value and der
     planes at the pinned cursor, the MultiClass gradient reconstruction,
     the blocked lower-triangular Hessian row by row with lambda on the
     diagonal and the per-leaf Cholesky (or the OneVsAll diagonal Hessian),
     `RegularizeImpl`, `MakeEstimationResult`'s gauge fix, and
     `add_model_value_kernel`'s plane-major cursor update through
     `identical_mul_add`.
  6. The rescale on append, the learn losses and the model text with
     `dim` leaves per bin.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to lambda, so every leaf moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the two lanes is the measurement.
"""
from std.math import exp, fma, isfinite, log
from gbdt.data.permutation import TRandom
from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_sqrt,
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
from gbdt.host.gbdt_oracle import (
    GBDT_FLOAT32_MAX,
    GBDT_HB_LANES,
    GBDT_HB_BLOCK,
    GBDT_HB_LOAD,
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_PINNED_SM,
    GBDT_SENTINEL,
    GBDT_STATS_BLOCK,
    GbdtHostParams,
    _add_leaf_cosine,
    _bootstrap_pass,
    _target_std_dev,
    gbdt_bootstrap_seeds,
    _binarize_columns,
    _choose_scale_from_magnitudes,
    _deterministic_sum_lanes,
    _half_byte_one_block,
    _halving_fold,
    _pinned_partition_stat,
    _hist2_dither,
    _hist2_quantize,
    gbdt_f32_token,
    gbdt_f64_token,
    gbdt_host_grid,
    _nan_token,
)
from gbdt.lapack.linear_system import solve_linear_system_cholesky


#: `OBJECTIVE_MULTICLASS`, `OBJECTIVE_MULTICLASS_OVA`
#: (`pointwise_targets.mojo:66-72`).
comptime GBDT_OBJ_MULTICLASS = 12
comptime GBDT_OBJ_MULTICLASS_OVA = 13


@fieldwise_init
struct GbdtHostMultiModel(Movable):
    """A one-shape oblivious float-only ensemble with `dim` values per leaf
    (bin-major, `[leaf * dim + d]`)."""

    var fold_counts: List[Int]
    var borders: List[List[Float32]]
    var nan_treatment: List[Int]
    var tree_split_offsets: List[Int]
    var split_features: List[Int]
    var split_bins: List[Int]
    var tree_leaf_offsets: List[Int]
    var leaf_values: List[Float32]
    var losses: List[Float64]
    var dim: Int

    def n_trees(self) -> Int:
        return len(self.tree_split_offsets) - 1


def _partition_stat_n(
    stats: List[Float32], line_size: Int, stat_id: Int, offset: Int,
    size: Int, n_stats: Int,
) -> Float32:
    """`compute_partition_stats` for one (leaf, stat) at `n_stats` planes
    (`partitions_reduce.mojo`): `(2 * 32 + n_stats - 1) // n_stats` chunks,
    phase 1 striding each 512-thread block, phase 2 folding the partials."""
    var max_chunks = (2 * GBDT_PINNED_SM + n_stats - 1) // n_stats
    if max_chunks < 1:
        max_chunks = 1
    return _pinned_partition_stat(
        stats, stat_id * line_size + offset, size, max_chunks
    )


# ===========================================================================
# THE TARGET KERNELS (`gbdt/targets/kernel/multilogit.mojo`)
# ===========================================================================


def _clip_prob(p: Float32) -> Float32:
    """`clip_prob` (`multilogit.mojo`): `[1e-7, 1 - 1e-7]`."""
    return max(min(p, Float32(1.0) - Float32(1e-7)), Float32(1e-7))


def _multi_pass(
    objective: Int,
    num_classes: Int,
    targets: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    cursor: List[Float32],
    n_rows: Int,
    search: Bool,
    compute_mags: Bool,
    mut der: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """One launch of the value-and-der kernel over rows in `cursor`'s order
    (no load indices): `search` writes plane 0 the weight and the classes at
    planes 1.. (their `StatsToAggregate`), else the classes at planes 0..;
    one score partial per 256-thread block, and under `compute_mags` the two
    magnitude partials."""
    var eff = num_classes - 1
    var dim = eff if objective == GBDT_OBJ_MULTICLASS else num_classes
    var plane_base = 1 if search else 0
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var idx = b * GBDT_MSE_BLOCK + t
            var in_range = idx < n_rows
            var weight = Float32(1.0)
            if has_weights and in_range:
                weight = weights[idx]
            var tmp_score = Float32(0.0)
            var mag_der = Float32(0.0)
            var mag_weight = Float32(0.0)
            if objective == GBDT_OBJ_MULTICLASS:
                var target_class = 0
                if in_range:
                    target_class = Int(targets[idx])
                var mx = Float32(0.0)
                if in_range:
                    for k in range(eff):
                        var v = cursor[k * n_rows + idx]
                        if v > mx:
                            mx = v
                var tmp = Float32(0.0)
                if in_range and target_class < eff:
                    tmp = cursor[target_class * n_rows + idx]
                var class_approx = tmp - mx
                var se = Float32(0.0)
                if in_range:
                    for k in range(eff):
                        se += identical_exp(cursor[k * n_rows + idx] - mx)
                se += identical_exp(Float32(0.0) - mx)
                if in_range:
                    if search:
                        der[idx] = weight
                    var max_abs_der = Float32(0.0)
                    for k in range(eff):
                        var pk = identical_exp(cursor[k * n_rows + idx] - mx) / se
                        var indicator = Float32(1.0) if target_class == k else Float32(0.0)
                        var d = weight * (indicator - pk)
                        der[(plane_base + k) * n_rows + idx] = d
                        var ad = abs(d)
                        if ad > max_abs_der:
                            max_abs_der = ad
                    mag_der += max_abs_der
                    mag_weight += abs(weight)
                var log_denum = identical_log(se)
                if in_range:
                    # one rounding, as the default (contract=fast) build fused it
                    tmp_score = identical_mul_add(weight, class_approx - log_denum, tmp_score)
            else:
                var target_class = 0
                if in_range:
                    target_class = Int(targets[idx])
                if search and in_range:
                    der[idx] = weight
                    mag_weight += abs(weight)
                var max_abs = Float32(0.0)
                for clazz in range(num_classes):
                    var val = Float32(0.0)
                    if in_range:
                        val = cursor[clazz * n_rows + idx]
                    var exp_val = identical_exp(val)
                    var p = _clip_prob(exp_val / (Float32(1.0) + exp_val))
                    var c = Float32(1.0) if target_class == clazz else Float32(0.0)
                    var direction = c - p
                    if in_range:
                        var d = weight * direction
                        der[(plane_base + clazz) * n_rows + idx] = d
                        var ad = abs(d)
                        if ad > max_abs:
                            max_abs = ad
                    var log_term = val
                    if isfinite(exp_val):
                        log_term = identical_log(Float32(1.0) + exp_val)
                    if in_range:
                        tmp_score += weight * (c * val - log_term) / Float32(num_classes)
                mag_der += max_abs
            s_score[t] = tmp_score
            s_w[t] = mag_weight
            s_g[t] = mag_der
        fv_partials[b] = _halving_fold(s_score)
        if compute_mags:
            mag_partials[2 * b] = _halving_fold(s_w)
            mag_partials[2 * b + 1] = _halving_fold(s_g)
    _ = dim


# ===========================================================================
# THE HISTOGRAMS AT `stat_count` PLANES
# ===========================================================================


def _one_byte_block_n(
    blk: PolicyBlock,
    block_first_bin: Int,
    hist_cells: Int,
    stat_count: Int,
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
    """`gbdt_oracle.mojo::_one_byte_block` at `stat_count` planes: every
    (row, stat) quantized with the row position's dither and summed in Int32
    per (stat, feature, bin), whichever kernel (the pair ladder or the odd
    prelude) covers the plane; the fixed bridge writes the cells."""
    var n_f = blk.count()
    var total = 0
    for k in range(n_f):
        total += Int(blk.folds[k])
    for j in range(len(compute_ids)):
        var slot = compute_ids[j]
        var off = p_off[slot]
        var sz = p_sz[slot]
        var acc = List[Int32](length=stat_count * total, fill=Int32(0))
        var q = List[Int32](length=stat_count, fill=Int32(0))
        for pos in range(off, off + sz):
            var row = row_index[pos]
            var u = _hist2_dither(pos)
            for s in range(stat_count):
                q[s] = _hist2_quantize(stats[s * n_rows + pos], fixed_scale, u)
            for k in range(n_f):
                ref cf = layout.features[blk.feature_ids[k]]
                var word = cindex[Int(cf.offset) * n_rows + row]
                var bin = Int((word >> cf.shift) & cf.mask)
                if bin < Int(blk.folds[k]):
                    var at = Int(blk.fold_offset[k]) + bin
                    for s in range(stat_count):
                        acc[s * total + at] = acc[s * total + at] + q[s]
        for z in range(stat_count):
            for c in range(total):
                var qq = acc[z * total + c]
                var val = Float32(0.0)
                if qq != Int32(0):
                    val = ftz(Float32(Int(qq)) / fixed_scale)
                hist[slot * stat_count * hist_cells + z * hist_cells + block_first_bin + c] = val


def _half_byte_block_n(
    blk: PolicyBlock,
    block_first_bin: Int,
    hist_cells: Int,
    stat_count: Int,
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
    """`gbdt_oracle.mojo::_half_byte_block` at `stat_count` planes: the
    replication base is `groups * n_compute * stat_count`
    (`replication_for`), every plane its own grid z."""
    var n_f = blk.count()
    var n_compute = len(compute_ids)
    var groups = (n_f + 7) // 8
    var max_active_blocks = 2 * GBDT_PINNED_SM
    if depth > 0:
        max_active_blocks = 2 * max_active_blocks
    var base_count = groups * n_compute * stat_count
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
        for z in range(stat_count):
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
                            slot * stat_count * hist_cells + z * hist_cells
                            + block_first_bin + fold_off + fold
                        ] = cell


def _cosine_gain_n(
    hist: List[Float32],
    hist_cells: Int,
    stat_count: Int,
    part_stats: List[Float32],
    n_live: Int,
    bin_feature_id: Int,
    lambda_l2: Float32,
    multiclass_optimization: Bool,
    score_std_dev: Float32 = Float32(0.0),
    level_seed: UInt64 = 0,
    feature_id: Int = 0,
) -> Float32:
    """One bin-feature of `compute_optimal_splits_kernel[COSINE]`
    (`compute_scores.mojo:140-224`) at `stat_count` planes: the per-class
    `AddLeaf`s, the running totals, the multiclass optimization terms, the
    sqrt normalization, per-feature noise, feature weight 1.0."""
    var score = Float32(0.0)
    var denum_sqr = Float32(1e-10)
    for i in range(n_live):
        var leaf_base = i * stat_count * hist_cells
        var weight_left = max(hist[leaf_base + bin_feature_id], Float32(0.0))
        var weight_right = ftz(max(part_stats[i * stat_count] - weight_left, Float32(0.0)))
        var total_sum_left = Float32(0.0)
        var total_sum_part = Float32(0.0)
        for stat_id in range(1, stat_count):
            var sum_left = hist[leaf_base + stat_id * hist_cells + bin_feature_id]
            var part_stat = part_stats[i * stat_count + stat_id]
            var sum_right = ftz(part_stat - sum_left)
            _add_leaf_cosine(sum_left, weight_left, lambda_l2, score, denum_sqr)
            _add_leaf_cosine(sum_right, weight_right, lambda_l2, score, denum_sqr)
            total_sum_left = ftz(total_sum_left + sum_left)
            total_sum_part = ftz(total_sum_part + part_stat)
        if multiclass_optimization:
            var total_sum_right = ftz(total_sum_part - total_sum_left)
            _add_leaf_cosine(-total_sum_left, weight_left, lambda_l2, score, denum_sqr)
            _add_leaf_cosine(-total_sum_right, weight_right, lambda_l2, score, denum_sqr)
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
# THE ESTIMATION (`pointwise_oracle.mojo`, `descent_helpers.mojo`)
# ===========================================================================


def _estimate_multi(
    objective: Int,
    num_classes: Int,
    targets: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    cursor: List[Float32],
    row_index: List[Int],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    l2_leaf_reg: Float32,
) raises -> List[Float32]:
    """`_estimate_and_apply`'s estimate at one Newton iteration: the gathers,
    `WeightsCpu`, the walker's first evaluation (move to zero, value and der
    planes, the MultiClass reconstruction), `write_second_derivatives`
    (blocked for MultiClass, diagonal for OneVsAll), the direction, the one
    step, `RegularizeImpl` and `MakeEstimationResult`. Returns
    `n_leaves * cursor_dim` values, bin-major."""
    var n_leaves = len(sizes)
    var is_mc = objective == GBDT_OBJ_MULTICLASS
    var cursor_dim = num_classes - 1 if is_mc else num_classes
    var sbd = num_classes
    var g_target = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_weights = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=cursor_dim * n_rows, fill=Float32(0.0))
    for pos in range(n_rows):
        g_target[pos] = targets[row_index[pos]]
        if has_weights:
            g_weights[pos] = weights[row_index[pos]]
        for k in range(cursor_dim):
            g_cursor[k * n_rows + pos] = cursor[k * n_rows + row_index[pos]]
    var weights_cpu = List[Float64]()
    for leaf in range(n_leaves):
        if has_weights:
            weights_cpu.append(
                Float64(_partition_stat_n(g_weights, n_rows, 0, offsets[leaf], sizes[leaf], 1))
            )
        else:
            weights_cpu.append(Float64(sizes[leaf]))
    var lambda_reg = Float64(l2_leaf_reg)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0

    # `move_to(zeros)`: the shift is 0.0 per (bin, dim), added per row
    var bins = List[Int](length=n_rows, fill=0)
    for leaf in range(n_leaves):
        for k in range(sizes[leaf]):
            bins[offsets[leaf] + k] = leaf
    var shift = List[Float32](length=n_leaves * cursor_dim, fill=Float32(0.0))
    for pos in range(n_rows):
        for d in range(cursor_dim):
            g_cursor[d * n_rows + pos] = g_cursor[d * n_rows + pos] + shift[bins[pos] * cursor_dim + d]

    # `write_value_and_first_derivatives`' multi-dimensional arm
    var der = List[Float32](length=sbd * n_rows, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_part = List[Float32](length=blocks, fill=Float32(0.0))
    var mag_dummy = List[Float32](length=2, fill=Float32(0.0))
    _multi_pass(
        objective, num_classes, g_target, g_weights, has_weights, g_cursor,
        n_rows, False, False, der, fv_part, mag_dummy,
    )
    var der_at_point = List[Float64]()
    for leaf in range(n_leaves):
        for d in range(cursor_dim):
            der_at_point.append(
                Float64(_partition_stat_n(der, n_rows, d, offsets[leaf], sizes[leaf], cursor_dim))
            )
    var gradient = List[Float64](length=n_leaves * sbd, fill=Float64(0.0))
    if not is_mc:
        for i in range(n_leaves * cursor_dim):
            gradient[i] = der_at_point[i]
    else:
        for bin in range(n_leaves):
            var total = Float64(0.0)
            for d in range(cursor_dim):
                var val = der_at_point[bin * cursor_dim + d]
                gradient[bin * sbd + d] = val
                total += val
            gradient[bin * sbd + cursor_dim] = -total

    # `write_second_derivatives` and the direction
    var direction = List[Float32](length=n_leaves * sbd, fill=Float32(0.0))
    if is_mc:
        var hbs = sbd
        var matrix_size = hbs * hbs
        var second_der = List[Float64](length=matrix_size * n_leaves, fill=Float64(0.0))
        for row in range(hbs):
            var column_count = row + 1
            var der2 = List[Float32](length=sbd * n_rows, fill=Float32(0.0))
            var eff = num_classes - 1
            for idx in range(n_rows):
                var mx = Float32(0.0)
                for k in range(eff):
                    var v = g_cursor[k * n_rows + idx]
                    if v > mx:
                        mx = v
                var se = Float32(0.0)
                for k in range(eff):
                    se += identical_exp(g_cursor[k * n_rows + idx] - mx)
                se += identical_exp(Float32(0.0) - mx)
                var weight = g_weights[idx] if has_weights else Float32(1.0)
                var p_row: Float32
                if row < eff:
                    p_row = identical_exp(g_cursor[row * n_rows + idx] - mx) / se
                else:
                    p_row = identical_exp(-mx) / se
                for k in range(row):
                    var pk = identical_exp(g_cursor[k * n_rows + idx] - mx) / se
                    der2[k * n_rows + idx] = -weight * pk * p_row
                der2[row * n_rows + idx] = weight * (Float32(1.0) - p_row) * p_row
            for bin in range(n_leaves):
                var base = bin * matrix_size
                for col in range(column_count):
                    var val = Float64(
                        _partition_stat_n(der2, n_rows, col, offsets[bin], sizes[bin], column_count)
                    )
                    if col == row:
                        second_der[base + row * hbs + row] = val + lambda_reg
                    else:
                        second_der[base + row * hbs + col] = val
                        second_der[base + col * hbs + row] = val
        for block_id in range(n_leaves):
            var sigma = List[Float64]()
            for i in range(hbs * hbs):
                sigma.append(second_der[block_id * hbs * hbs + i])
            var solution = List[Float64]()
            for i in range(hbs):
                solution.append(gradient[block_id * hbs + i])
            _ = solve_linear_system_cholesky(sigma, solution)
            for i in range(hbs):
                direction[block_id * hbs + i] = Float32(solution[i])
    else:
        var der2 = List[Float32](length=cursor_dim * n_rows, fill=Float32(0.0))
        for clazz in range(num_classes):
            for idx in range(n_rows):
                var weight = g_weights[idx] if has_weights else Float32(1.0)
                var val = g_cursor[clazz * n_rows + idx]
                var exp_val = identical_exp(val)
                var p = _clip_prob(exp_val / (Float32(1.0) + exp_val))
                der2[clazz * n_rows + idx] = weight * p * (Float32(1.0) - p)
        comptime EPS_1E20F = Float64(Float32(1e-20))
        for bin in range(n_leaves):
            for d in range(cursor_dim):
                var h = Float64(
                    _partition_stat_n(der2, n_rows, d, offsets[bin], sizes[bin], cursor_dim)
                ) + lambda_reg
                var g = gradient[bin * sbd + d]
                if h > 0:
                    direction[bin * sbd + d] = Float32(g / (h + EPS_1E20F))
                else:
                    direction[bin * sbd + d] = Float32(0.0)

    # `_move(cur_point, direction, 1.0)`, `regularize`, `make_estimation_result`
    var point = List[Float32](length=n_leaves * sbd, fill=Float32(0.0))
    for i in range(n_leaves * sbd):
        point[i] = Float32(Float64(Float32(0.0)) + 1.0 * Float64(direction[i]))
    for bin in range(n_leaves):
        if weights_cpu[bin] < 1e-20:
            for d in range(sbd):
                point[bin * sbd + d] = Float32(0.0)
    if not is_mc:
        return point^
    var out = List[Float32]()
    for bin in range(n_leaves):
        for d in range(cursor_dim):
            out.append(point[bin * sbd + d] - point[bin * sbd + cursor_dim])
    return out^


# ===========================================================================
# THE FIT
# ===========================================================================


def gbdt_multi_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    objective: Int,
    class_weights: List[Float32],
    bootstrap_kind: Int = -1,
    bootstrap_param: Float32 = Float32(1.0),
    random_strength: Float32 = Float32(0.0),
) raises -> GbdtHostMultiModel:
    """`train` then `fit_with_test` on the two covered configurations."""
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    var mx = -1
    for r in range(n_rows):
        var v = y[r]
        if v < Float32(0.0):
            raise Error(
                "MultiClass label at row " + String(r)
                + " is negative; labels are dense class codes 0..k-1"
            )
        var iv = Int(v)
        if Float32(iv) != v:
            raise Error(
                "MultiClass label at row " + String(r)
                + " is not an integer; labels are dense class codes 0..k-1"
            )
        if iv > mx:
            mx = iv
    var num_classes = mx + 1
    if num_classes < 2:
        raise Error(
            "the multiclass family needs at least two classes; the labels"
            " reach only " + String(num_classes)
        )
    var has_weights = len(class_weights) > 0
    if has_weights and len(class_weights) != num_classes:
        raise Error(
            "class_weights takes " + String(num_classes)
            + " entries, got " + String(len(class_weights))
        )
    var weights = List[Float32](length=n_rows, fill=Float32(1.0))
    if has_weights:
        for r in range(n_rows):
            weights[r] = Float32(1.0) * class_weights[Int(y[r])]
    var is_mc = objective == GBDT_OBJ_MULTICLASS
    var dim = num_classes - 1 if is_mc else num_classes
    var stat_count = 1 + dim

    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, params.border_count,
        params.border_build_max_samples, params.random_seed, params.nan_mode,
        params.border_type,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
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

    var cursor = List[Float32](length=dim * n_rows, fill=Float32(0.0))
    var stats = List[Float32](length=stat_count * n_rows, fill=Float32(0.0))
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

    var boot_seeds = List[UInt64]()
    if bootstrap_kind >= 0:
        boot_seeds = gbdt_bootstrap_seeds(params.random_seed)
    var noise_rand = TRandom(params.random_seed)

    for iteration in range(params.n_estimators):
        _multi_pass(
            objective, num_classes, y, weights, has_weights, cursor, n_rows,
            True, True, stats, fv_part, mag_part,
        )
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
        var noise_mult = Float64(0.0)
        if random_strength != Float32(0.0):
            # one rounding, as the default (contract=fast) build fused it
            var model_left = exp(fma(-Float64(iteration), Float64(params.learning_rate), log(Float64(n_rows))))
            noise_mult = model_left / (1.0 + model_left)
        var tree_seed = noise_rand.next_uniform_l()
        if bootstrap_kind >= 0:
            var bm = _bootstrap_pass(bootstrap_kind, boot_seeds, stats, n_rows, bootstrap_param, stat_count)
            mags[0] = bm[0]
            mags[1] = bm[1]
        var fixed_scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            score_std_dev = Float32(Float64(Float32(noise_mult * Float64(random_strength))) * _target_std_dev(stats, n_rows, stat_count, is_mc))
        var level_rand = TRandom(tree_seed)

        # ---- `run_tree_layout_traced` at `stat_count` planes (TWIN of the
        # loop in `gbdt_oracle.mojo::gbdt_host_fit`) ----
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[r] = r
        var p_off = List[Int](length=max_leaves, fill=0)
        var p_sz = List[Int](length=max_leaves, fill=0)
        p_sz[0] = n_rows
        var hist = List[Float32](
            length=max_leaves * stat_count * hist_cells, fill=Float32(0.0)
        )
        var ids_compute = List[Int](length=max_leaves, fill=0)
        var sub_from = List[Int](length=max_leaves, fill=0)
        var sub_what = List[Int](length=max_leaves, fill=0)
        var winners_score = List[Float32]()
        var winners_bf = List[UInt32]()
        var n_live = 1
        for depth in range(max_depth):
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
                    _half_byte_block_n(
                        blk, block_first_bin, hist_cells, stat_count, compute,
                        depth, p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_ONE_BYTE:
                    _one_byte_block_n(
                        blk, block_first_bin, hist_cells, stat_count, compute,
                        p_off, p_sz, row_index, stats, cindex, layout, n_rows,
                        fixed_scale, hist,
                    )
                block_first_bin += total

            for j in range(len(compute)):
                var slot = compute[j]
                for z in range(stat_count):
                    for f in range(n_features):
                        ref cf = layout.features[f]
                        var folds = Int(cf.folds)
                        if cf.one_hot_feature or folds <= 1:
                            continue
                        var base = (
                            slot * stat_count * hist_cells + z * hist_cells
                            + Int(cf.first_fold_index)
                        )
                        var running = Float32(0.0)
                        for i in range(folds):
                            running = ftz(running + hist[base + i])
                            hist[base + i] = running

            if planned and half > 0:
                for j in range(half):
                    var from_slot = sub_from[j]
                    var what_slot = sub_what[j]
                    for z in range(stat_count):
                        var from_base = from_slot * stat_count * hist_cells + z * hist_cells
                        var what_base = what_slot * stat_count * hist_cells + z * hist_cells
                        for bf in range(hist_cells):
                            var new_val = ftz(hist[from_base + bf] - hist[what_base + bf])
                            if z == 0:
                                new_val = max(new_val, Float32(0.0))
                            hist[from_base + bf] = new_val

            var part_stats = List[Float32](length=stat_count * n_live, fill=Float32(0.0))
            for i in range(n_live):
                for s in range(stat_count):
                    part_stats[stat_count * i + s] = _partition_stat_n(
                        stats, n_rows, s, p_off[i], p_sz[i], stat_count
                    )

            var best_gain = -GBDT_FLOAT32_MAX
            var best_bin = GBDT_SENTINEL
            for bf in range(hist_cells):
                var gain = _cosine_gain_n(
                    hist, hist_cells, stat_count, part_stats, n_live, bf,
                    params.l2_leaf_reg, is_mc,
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
                    for s in range(stat_count):
                        new_stats[s * n_rows + off + dst] = stats[s * n_rows + src]
                    dst += 1
                for k in range(len(ones)):
                    var src = off + ones[k]
                    new_rows[off + dst] = row_index[src]
                    for s in range(stat_count):
                        new_stats[s * n_rows + off + dst] = stats[s * n_rows + src]
                    dst += 1
                var src_base = i * stat_count * hist_cells
                var dst_base = (n_live + i) * stat_count * hist_cells
                for c in range(stat_count * hist_cells):
                    hist[dst_base + c] = hist[src_base + c]
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

        var estimated = _estimate_multi(
            objective, num_classes, y, weights, has_weights, cursor,
            row_index, offsets, sizes, n_rows, params.l2_leaf_reg,
        )
        # `add_model_value_kernel`: `[leaf * dim + d]` into `[d * n + row]`
        for leaf in range(n_live):
            for d in range(dim):
                var raw = estimated[leaf * dim + d]
                for k in range(sizes[leaf]):
                    var at = d * n_rows + row_index[offsets[leaf] + k]
                    cursor[at] = identical_mul_add(raw, lr, cursor[at])
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(estimated)):
            model_leaves.append(estimated[i] * lr)
        tree_leaf_offsets.append(len(model_leaves))

        if len(losses) < params.n_estimators:
            var v = Float64(fv)
            if iteration + 1 > 1:
                losses.append(-v / Float64(n_rows))

    _multi_pass(
        objective, num_classes, y, weights, has_weights, cursor, n_rows,
        True, False, stats, fv_part, mag_part,
    )
    var final_fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
    losses.append(-Float64(final_fv) / Float64(n_rows))
    return GbdtHostMultiModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, dim,
    )


def gbdt_multi_host_model_text(m: GbdtHostMultiModel) raises -> String:
    """`gbdt_oracle.mojo::gbdt_host_model_text` with `dim` values per leaf
    (`model_text.mojo:601-670`)."""
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
        var n_values = (1 << depth) * m.dim
        if m.tree_leaf_offsets[t + 1] - leaf_lo != n_values:
            raise Error(
                "tree " + String(t) + " has depth " + String(depth)
                + ", dim " + String(m.dim) + " and "
                + String(m.tree_leaf_offsets[t + 1] - leaf_lo)
                + " leaf values, not " + String(n_values)
            )
        out += (
            String("tree ") + String(t) + " depth " + String(depth)
            + " dim " + String(m.dim) + " weights " + String(0) + "\n"
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
