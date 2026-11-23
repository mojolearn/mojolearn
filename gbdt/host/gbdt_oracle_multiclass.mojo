# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the multi-output losses,
MultiClass and MultiClassOneVsAll, on symmetric trees
(lane/cpu-training-gbdt-losses, 2026-09-15): the gbdt-multiclass and
gbdt-onevsall lanes of tools/identity_break.py; and MultiRMSE
(lane/algos-trees, 2026-09-27), the trees-gbdt-multirmse lane: the target is
`dim` dim-major planes, the der kernel is `multi_rmse_val_and_first_der_
kernel`, and the Hessian is MultiClass's BLOCKED arm with
`multi_rmse_second_der_kernel`'s rows (zeros left of the diagonal, the
weight on it), solved by the same per-leaf Cholesky.

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
from gbdt.data.class_weight_body import class_weight_slot, class_weighted

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_sqrt,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, host_predict_chunk
from gbdt.methods.cursor_rescale import rescaled_leaves
from gbdt.methods.random_score_bodies import score_model_length_mult
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
    apply_scaled_leaves_host,
    _identity_rows,
    GBDT_FLOAT32_MAX,
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_PINNED_SM,
    GBDT_SENTINEL,
    GBDT_STATS_BLOCK,
    GbdtHostParams,
    _add_leaf_cosine,
    _bootstrap_pass,
    _snap_gradients,
    _target_std_dev,
    gbdt_bootstrap_seeds,
    _binarize_columns,
    _choose_scale_from_magnitudes,
    _deterministic_sum_lanes,
    _binary_block_n,
    _half_byte_block_n,
    _one_byte_block_n,
    _halving_fold,
    _pinned_partition_stats,
    GbdtIntPtr,
    _gain_winner,
    _gbdt_units,
    _partition_stats_level,
    _scan_histograms,
    _split_level,
    _subtract_histograms,
    gbdt_f32_token,
    gbdt_f64_token,
    gbdt_host_grid,
    _nan_token,
    _ns_project_index,
    _sample_tree_folds,
    # moved to gbdt_oracle (lane gbdt-cpu-rest-4-cat, 2026-11-23): the
    # tree-CTR level driver scores at `stat_count` planes with them too
    _cosine_gain_n,
    _cosine_gains_n,
    _stat_chunks,
)
from gbdt.host.gbdt_oracle_eval import (
    GbdtHostEval,
    GbdtHostEvalFit,
    gbdt_eval_add_tree,
    gbdt_eval_detector,
    gbdt_eval_shrink_point,
    gbdt_eval_test_bins,
)
from gbdt.lapack.linear_system import solve_linear_system_cholesky


#: `OBJECTIVE_MULTICLASS`, `OBJECTIVE_MULTICLASS_OVA`, `OBJECTIVE_MULTIRMSE`
#: (`pointwise_targets.mojo`).
comptime GBDT_OBJ_MULTICLASS = 12
comptime GBDT_OBJ_MULTICLASS_OVA = 13
comptime GBDT_OBJ_MULTIRMSE = 17


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


@fieldwise_init
struct GbdtHostMultiFit(Movable):
    """`gbdt_multi_host_fit`'s pair: the ensemble and what its HELD-OUT arm
    reports (`GbdtHostEvalFit`), as `GbdtHostFitWithEval` for one output."""

    var model: GbdtHostMultiModel
    var eval: GbdtHostEvalFit


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
    # THE SPLIT (lane gbdt-host-par), the unit of `_logloss_search_pass`:
    # one 256-thread device block owns its rows' planes and its partials,
    # and its halving trees are the kernel's (a thread's class loops run in
    # the kernel's order inside it). Blocks go round robin to `_gbdt_units`
    # tasks; a task holds three block slabs, every lane rewritten per block.
    var units = _gbdt_units(blocks, 16 * num_classes * n_rows)
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    # an empty `weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var der_p = rebind[HostF32Ptr](der.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())
    # without `compute_mags` the magnitude partials are never written
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm objective, imm num_classes, imm eff, imm plane_base, imm has_weights, imm search, imm compute_mags, imm targets_p, imm weights_p, imm cursor_p, imm der_p, imm fv_p, imm mag_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var idx = b * GBDT_MSE_BLOCK + t
                var in_range = idx < n_rows
                var weight = Float32(1.0)
                if has_weights and in_range:
                    weight = weights_p[idx]
                var tmp_score = Float32(0.0)
                var mag_der = Float32(0.0)
                var mag_weight = Float32(0.0)
                if objective == GBDT_OBJ_MULTICLASS:
                    var target_class = 0
                    if in_range:
                        target_class = Int(targets_p[idx])
                    var mx = Float32(0.0)
                    if in_range:
                        for k in range(eff):
                            var v = cursor_p[k * n_rows + idx]
                            if v > mx:
                                mx = v
                    var tmp = Float32(0.0)
                    if in_range and target_class < eff:
                        tmp = cursor_p[target_class * n_rows + idx]
                    var class_approx = tmp - mx
                    var se = Float32(0.0)
                    if in_range:
                        for k in range(eff):
                            se += identical_exp(cursor_p[k * n_rows + idx] - mx)
                    se += identical_exp(Float32(0.0) - mx)
                    if in_range:
                        if search:
                            der_p[idx] = weight
                        var max_abs_der = Float32(0.0)
                        for k in range(eff):
                            var pk = identical_exp(cursor_p[k * n_rows + idx] - mx) / se
                            var indicator = Float32(1.0) if target_class == k else Float32(0.0)
                            var d = weight * (indicator - pk)
                            der_p[(plane_base + k) * n_rows + idx] = d
                            var ad = abs(d)
                            if ad > max_abs_der:
                                max_abs_der = ad
                        mag_der += max_abs_der
                        mag_weight += abs(weight)
                    var log_denum = identical_log(se)
                    if in_range:
                        # one rounding, as the default (contract=fast) build fused it
                        tmp_score = identical_mul_add(weight, class_approx - log_denum, tmp_score)
                elif objective == GBDT_OBJ_MULTIRMSE:
                    # `multi_rmse_val_and_first_der_kernel[1, search]`: the
                    # target is `num_classes` dim-major planes; dimension
                    # OUTSIDE, the one element inside; the score is the negated
                    # per-thread `fma` accumulation
                    if search and in_range:
                        der_p[idx] = weight
                        mag_weight += abs(weight)
                    var max_abs = Float32(0.0)
                    var sum_dim_errors = Float32(0.0)
                    for d in range(num_classes):
                        if in_range:
                            var target = targets_p[d * n_rows + idx]
                            var approx = cursor_p[d * n_rows + idx]
                            var diff = target - approx
                            sum_dim_errors = fma(diff * diff, weight, sum_dim_errors)
                            var dd = diff * weight
                            der_p[(plane_base + d) * n_rows + idx] = dd
                            var ad = abs(dd)
                            if ad > max_abs:
                                max_abs = ad
                    mag_der += max_abs
                    tmp_score = -sum_dim_errors
                else:
                    var target_class = 0
                    if in_range:
                        target_class = Int(targets_p[idx])
                    if search and in_range:
                        der_p[idx] = weight
                        mag_weight += abs(weight)
                    var max_abs = Float32(0.0)
                    for clazz in range(num_classes):
                        var val = Float32(0.0)
                        if in_range:
                            val = cursor_p[clazz * n_rows + idx]
                        var exp_val = identical_exp(val)
                        var p = _clip_prob(exp_val / (Float32(1.0) + exp_val))
                        var c = Float32(1.0) if target_class == clazz else Float32(0.0)
                        var direction = c - p
                        if in_range:
                            var d = weight * direction
                            der_p[(plane_base + clazz) * n_rows + idx] = d
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
            fv_p[b] = _halving_fold(s_score)
            if compute_mags:
                mag_p[2 * b] = _halving_fold(s_w)
                mag_p[2 * b + 1] = _halving_fold(s_g)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(targets)
    _ = len(weights)
    _ = len(cursor)
    _ = len(der)
    _ = len(fv_partials)
    _ = len(mag_partials)
    _ = dim


# ===========================================================================
# THE HISTOGRAMS AT `stat_count` PLANES
# ===========================================================================


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
    var is_mrmse = objective == GBDT_OBJ_MULTIRMSE
    var cursor_dim = num_classes - 1 if is_mc else num_classes
    var sbd = num_classes
    # MultiRMSE's target is `num_classes` dim-major planes, gathered plane
    # by plane like the cursor (`launch_gather_planes_with_mask_f32`)
    var target_planes = num_classes if is_mrmse else 1
    var g_target = List[Float32](length=target_planes * n_rows, fill=Float32(0.0))
    var g_weights = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=cursor_dim * n_rows, fill=Float32(0.0))
    # `move_to(zeros)`: the shift is 0.0 per (bin, dim), added per row
    var bins = List[Int](length=n_rows, fill=0)
    var shift = List[Float32](length=n_leaves * cursor_dim, fill=Float32(0.0))
    # THE SPLIT (lane gbdt-host-par): the gathers, the bins and the shift
    # are one statement per (position, plane), a device thread a position:
    # a task owns a run of positions (a leaf's range clipped to it)
    var g_units = _gbdt_units(n_rows, (2 + target_planes + 3 * cursor_dim) * n_rows)
    var g_chunk = host_predict_chunk(n_rows, g_units)
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    # an empty `weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var row_p = rebind[GbdtIntPtr](row_index.unsafe_ptr())
    var off_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var sz_p = rebind[GbdtIntPtr](sizes.unsafe_ptr())
    var gt_p = rebind[HostF32Ptr](g_target.unsafe_ptr())
    var gw_p = rebind[HostF32Ptr](g_weights.unsafe_ptr())
    var gc_p = rebind[HostF32Ptr](g_cursor.unsafe_ptr())
    var bins_p = rebind[GbdtIntPtr](bins.unsafe_ptr())
    var shift_p = rebind[HostF32Ptr](shift.unsafe_ptr())

    def gather(unit: Int) {imm g_chunk, imm n_rows, imm n_leaves, imm target_planes, imm cursor_dim, imm has_weights, imm targets_p, imm weights_p, imm cursor_p, imm row_p, imm off_p, imm sz_p, imm gt_p, imm gw_p, imm gc_p, imm bins_p, imm shift_p}:
        var lo = unit * g_chunk
        var hi = min(lo + g_chunk, n_rows)
        for pos in range(lo, hi):
            var row = row_p[pos]
            for t in range(target_planes):
                gt_p[t * n_rows + pos] = targets_p[t * n_rows + row]
            if has_weights:
                gw_p[pos] = weights_p[row]
            for k in range(cursor_dim):
                gc_p[k * n_rows + pos] = cursor_p[k * n_rows + row]
        for leaf in range(n_leaves):
            for pos in range(max(off_p[leaf], lo), min(off_p[leaf] + sz_p[leaf], hi)):
                bins_p[pos] = leaf
        for pos in range(lo, hi):
            for d in range(cursor_dim):
                gc_p[d * n_rows + pos] = gc_p[d * n_rows + pos] + shift_p[bins_p[pos] * cursor_dim + d]

    host_parallelize(gather, g_units)
    _ = len(targets)
    _ = len(weights)
    _ = len(cursor)
    _ = len(row_index)
    _ = len(bins)
    _ = len(shift)
    var weights_cpu = List[Float64]()
    if has_weights:
        # `compute_partition_stats` of the gathered weights at ONE stat (64
        # chunks), the leaves' launches across the cores
        var weight_sums = List[Float32](length=n_leaves, fill=Float32(0.0))
        _pinned_partition_stats(
            g_weights, offsets, sizes, 2 * GBDT_PINNED_SM, weight_sums
        )
        for leaf in range(n_leaves):  # driver-loop: over the leaves
            weights_cpu.append(Float64(weight_sums[leaf]))
    else:
        for leaf in range(n_leaves):  # driver-loop: over the leaves
            weights_cpu.append(Float64(sizes[leaf]))
    var lambda_reg = Float64(l2_leaf_reg)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0

    # `write_value_and_first_derivatives`' multi-dimensional arm
    var der = List[Float32](length=sbd * n_rows, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_part = List[Float32](length=blocks, fill=Float32(0.0))
    var mag_dummy = List[Float32](length=2, fill=Float32(0.0))
    _multi_pass(
        objective, num_classes, g_target, g_weights, has_weights, g_cursor,
        n_rows, False, False, der, fv_part, mag_dummy,
    )
    # the (leaf, dim) sums: `compute_partition_stats` at `cursor_dim` planes
    # (`_stat_chunks`), the launches across the cores
    var der_sums = _partition_stats_level(
        der, n_rows, cursor_dim, _stat_chunks(cursor_dim), offsets, sizes, n_leaves
    )
    var der_at_point = List[Float64]()
    for i in range(n_leaves * cursor_dim):  # driver-loop: over (leaf, dim)
        der_at_point.append(Float64(der_sums[i]))
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
    if is_mc or is_mrmse:
        # MultiRMSE is BLOCKED too: `GetHessianType()` names only OneVsAll
        # Diagonal (`multiclass_targets.h:118-123`)
        var hbs = sbd
        var matrix_size = hbs * hbs
        var second_der = List[Float64](length=matrix_size * n_leaves, fill=Float64(0.0))
        for row in range(hbs):
            var column_count = row + 1
            var der2 = List[Float32](length=sbd * n_rows, fill=Float32(0.0))
            var eff = num_classes - 1
            # THE SPLIT (lane gbdt-host-par): one Hessian row is one device
            # launch, a thread a data row writing its own `row + 1` plane
            # elements (the class loops in the kernel's order inside it);
            # the data rows run in `_gbdt_units` contiguous ranges
            var h_units = _gbdt_units(n_rows, 24 * num_classes * n_rows)
            var h_chunk = host_predict_chunk(n_rows, h_units)
            var der2_p = rebind[HostF32Ptr](der2.unsafe_ptr())

            def second_rows(unit: Int) {imm h_chunk, imm n_rows, imm row, imm eff, imm is_mrmse, imm has_weights, imm gw_p, imm gc_p, imm der2_p}:
                for idx in range(unit * h_chunk, min(n_rows, (unit + 1) * h_chunk)):
                    if is_mrmse:
                        # `multi_rmse_second_der_kernel`: zeros left of the
                        # diagonal, the weight on it
                        for k in range(row):
                            der2_p[k * n_rows + idx] = Float32(0.0)
                        der2_p[row * n_rows + idx] = (
                            gw_p[idx] if has_weights else Float32(1.0)
                        )
                        continue
                    var mx = Float32(0.0)
                    for k in range(eff):
                        var v = gc_p[k * n_rows + idx]
                        if v > mx:
                            mx = v
                    var se = Float32(0.0)
                    for k in range(eff):
                        se += identical_exp(gc_p[k * n_rows + idx] - mx)
                    se += identical_exp(Float32(0.0) - mx)
                    var weight = gw_p[idx] if has_weights else Float32(1.0)
                    var p_row: Float32
                    if row < eff:
                        p_row = identical_exp(gc_p[row * n_rows + idx] - mx) / se
                    else:
                        p_row = identical_exp(-mx) / se
                    for k in range(row):
                        var pk = identical_exp(gc_p[k * n_rows + idx] - mx) / se
                        der2_p[k * n_rows + idx] = -weight * pk * p_row
                    der2_p[row * n_rows + idx] = weight * (Float32(1.0) - p_row) * p_row

            host_parallelize(second_rows, h_units)
            # the (leaf, column) sums at `column_count` planes, the launches
            # across the cores
            var row_sums = _partition_stats_level(
                der2, n_rows, column_count, _stat_chunks(column_count),
                offsets, sizes, n_leaves,
            )
            for bin in range(n_leaves):  # driver-loop: over (leaf, column)
                var base = bin * matrix_size
                for col in range(column_count):
                    var val = Float64(row_sums[bin * column_count + col])
                    if col == row:
                        second_der[base + row * hbs + row] = val + lambda_reg
                    else:
                        second_der[base + row * hbs + col] = val
                        second_der[base + col * hbs + row] = val
        # KEPT SERIAL: one `hbs x hbs` Cholesky solve per leaf. The solver
        # raises (a task of a host split cannot), and a tree's leaves times
        # `hbs^3` is below one task's work at any class count the planes
        # above can hold.
        for block_id in range(n_leaves):  # driver-loop: over the leaves
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
        # THE SPLIT (lane gbdt-host-par): one statement per (class, row),
        # the rows in `_gbdt_units` contiguous ranges
        var d_units = _gbdt_units(n_rows, 16 * num_classes * n_rows)
        var d_chunk = host_predict_chunk(n_rows, d_units)
        var der2_p = rebind[HostF32Ptr](der2.unsafe_ptr())

        def diagonal_rows(unit: Int) {imm d_chunk, imm n_rows, imm num_classes, imm has_weights, imm gw_p, imm gc_p, imm der2_p}:
            for clazz in range(num_classes):
                for idx in range(unit * d_chunk, min(n_rows, (unit + 1) * d_chunk)):
                    var weight = gw_p[idx] if has_weights else Float32(1.0)
                    var val = gc_p[clazz * n_rows + idx]
                    var exp_val = identical_exp(val)
                    var p = _clip_prob(exp_val / (Float32(1.0) + exp_val))
                    der2_p[clazz * n_rows + idx] = weight * p * (Float32(1.0) - p)

        host_parallelize(diagonal_rows, d_units)
        comptime EPS_1E20F = Float64(Float32(1e-20))
        var diag_sums = _partition_stats_level(
            der2, n_rows, cursor_dim, _stat_chunks(cursor_dim), offsets, sizes, n_leaves
        )
        for bin in range(n_leaves):  # driver-loop: over (leaf, dim)
            for d in range(cursor_dim):
                var h = Float64(diag_sums[bin * cursor_dim + d]) + lambda_reg
                var g = gradient[bin * sbd + d]
                if h > 0:
                    direction[bin * sbd + d] = Float32(g / (h + EPS_1E20F))
                else:
                    direction[bin * sbd + d] = Float32(0.0)

    # the gathered planes the Hessian tasks read through pointers live to here
    _ = len(g_cursor)
    _ = len(g_weights)
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


def _class_weight_rows(
    y: List[Float32], class_weights: List[Float32], n_rows: Int,
    mut weights: List[Float32],
):
    """The weight plane of a class-weighted fit, `class_weight_fold_kernel`
    (`gbdt/data/target_prep.mojo`): `weights[r] = weights[r] *
    class_weights[label]` over the sample weight (1.0 without one), the
    labels admitted by the caller (dense integer codes below the class
    count). One statement per row, the rows in `_gbdt_units` contiguous
    ranges."""
    var units = _gbdt_units(n_rows, 2 * n_rows)
    var chunk = host_predict_chunk(n_rows, units)
    var y_p = rebind[HostF32Ptr](y.unsafe_ptr())
    var cw_p = rebind[HostF32Ptr](class_weights.unsafe_ptr())
    var w_p = rebind[HostF32Ptr](weights.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n_rows, imm y_p, imm cw_p, imm w_p}:
        for r in range(unit * chunk, min(n_rows, (unit + 1) * chunk)):
            # the device kernel's two statements, one body
            # (`gbdt/data/class_weight_body.mojo`)
            w_p[r] = class_weighted(w_p[r], cw_p[class_weight_slot(y_p[r], True)])

    host_parallelize(rows, units)
    _ = len(y)
    _ = len(class_weights)
    _ = len(weights)


def gbdt_multi_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    objective: Int,
    class_weights: List[Float32],
    # the held-out pool (`GbdtHostEval.none()` for none; lane
    # gbdt-cpu-rest-2, 2026-11-23)
    eval: GbdtHostEval,
    bootstrap_kind: Int = -1,
    bootstrap_param: Float32 = Float32(1.0),
    random_strength: Float32 = Float32(0.0),
    # MultiRMSE's target dimension; `y` then holds that many dim-major planes
    target_dim: Int = 1,
    # the sample weights in row order, empty for none (lane gbdt-cpu-rest,
    # 2026-11-23): the device's one `weights` column, which the class
    # weights multiply (`train`'s product, data_providers.cpp:168)
    sample_weights: List[Float32] = List[Float32](),
    # the per-tree feature sample (lane gbdt-cpu-rest-2, 2026-11-23)
    feature_fraction: Float64 = 1.0,
) raises -> GbdtHostMultiFit:
    """`train` then `fit_with_test` on the covered configurations.

    `eval` is the HELD-OUT arm (lane gbdt-cpu-rest-2), the statements of
    `gbdt_oracle.mojo::gbdt_host_boost` at `dim` values per leaf: the test
    rows binarized against the fit's grid, the test cursor `dim` planes of
    `n_eval` from zero (`CreateCursors`; no boost from average here), the
    last tree's `2^depth * dim` stored values added `[d * n + row] +=
    [leaf * dim + d]` (`compute_bins_and_add_kernel` at the cursor stride),
    the held-out loss through the SAME value-and-der launch the learn loss
    uses at unit weight (`_test_loss_enqueue`), the detector and
    `ShrinkToBestIteration`. MultiRMSE refuses an eval set on both columns
    (`train`'s words).

    `feature_fraction` below 1 is the device's per-tree feature sample
    through `_sample_tree_folds` and `_ns_project_index`, as
    `gbdt_host_boost` runs it."""
    if len(sample_weights) != 0 and len(sample_weights) != n_rows:
        raise Error("sample_weight size mismatch")
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    var is_mrmse = objective == GBDT_OBJ_MULTIRMSE
    if target_dim != 1 and not is_mrmse:
        raise Error("a multi-dimensional target is read only by MultiRMSE")
    if len(y) != n_rows * target_dim:
        raise Error("y size mismatch")
    if is_mrmse and target_dim < 2:
        # `CB_ENSURE(NumClasses > 1, ...)` (`multiclass_targets.h:167`)
        raise Error(
            "Only one class found, can't learn multiclass objective"
            " (MultiRMSE needs a target dimension >= 2)"
        )
    if is_mrmse and len(class_weights) > 0:
        raise Error("class_weights do not apply to loss='MultiRMSE'")
    var mx = -1
    if is_mrmse:
        # `NumClasses = GetTargetDimension()` (`multiclass_targets.h:155-156`)
        mx = target_dim - 1
    for r in range(0 if is_mrmse else n_rows):
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
    var has_class_weights = len(class_weights) > 0
    if has_class_weights and len(class_weights) != num_classes:
        raise Error(
            "class_weights takes " + String(num_classes)
            + " entries, got " + String(len(class_weights))
        )
    var has_weights = has_class_weights or len(sample_weights) > 0
    var weights: List[Float32]
    if len(sample_weights) > 0:
        weights = sample_weights.copy()
    else:
        weights = List[Float32](length=n_rows, fill=Float32(1.0))
    if has_class_weights:
        _class_weight_rows(y, class_weights, n_rows, weights)
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
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var hist_cells = layout.hist_cells
    var bf_feature = List[Int](length=hist_cells, fill=0)
    var bf_bin = List[Int](length=hist_cells, fill=0)
    for f in range(n_features):
        ref lf = layout.features[f]
        for b in range(Int(lf.folds)):
            bf_feature[Int(lf.first_fold_index) + b] = f
            bf_bin[Int(lf.first_fold_index) + b] = b

    # ---- the HELD-OUT arm (gbdt/host/gbdt_oracle_eval.mojo) ----
    ref ev = eval
    var has_test = ev.n_rows > 0
    if has_test and is_mrmse:
        # `train`'s words (gbdt/train.mojo)
        raise Error(
            "loss='MultiRMSE' with eval_set is not carried here: the"
            " held-out arm's target is one value per row"
        )
    if has_test:
        if len(ev.x_colmajor) != ev.n_rows * n_features:
            raise Error("eval_set x_colmajor size mismatch")
        if len(ev.y) != ev.n_rows:
            raise Error("eval_set y size mismatch")
    var n_eval = ev.n_rows if has_test else 1
    var test_cindex = List[UInt32]()
    if has_test:
        test_cindex = _binarize_columns(
            ev.x_colmajor, ev.n_rows, n_features, grid, layout
        )
    var test_cursor = List[Float32](length=dim * n_eval, fill=Float32(0.0))
    var test_stats = List[Float32](length=stat_count * n_eval, fill=Float32(0.0))
    var test_blocks = (n_eval + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var test_fv_part = List[Float32](length=test_blocks, fill=Float32(0.0))
    var test_mag_part = List[Float32](length=2 * test_blocks, fill=Float32(0.0))
    var no_weights = List[Float32]()
    var detector = gbdt_eval_detector(ev)
    var test_losses = List[Float64]()
    var stopped_early = False
    var feat_offset = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_one_hot = List[Bool](length=n_features, fill=False)
    for f in range(n_features):  # driver-loop: over the features (one record each)
        feat_offset[f] = layout.features[f].offset
        feat_mask[f] = layout.features[f].mask
        feat_shift[f] = layout.features[f].shift
        feat_one_hot[f] = layout.features[f].one_hot_feature

    # ---- the per-tree feature sample's state (`gbdt_host_boost`'s) ----
    var ff_on = feature_fraction < 1.0
    var s_layout = layout.copy()
    var s_blocks = blocks.copy()
    var s_hist_cells = hist_cells
    var s_bf_feature = bf_feature.copy()
    var s_bf_bin = bf_bin.copy()
    var full_cindex = List[UInt32]()
    if ff_on:
        full_cindex = cindex.copy()
    var ff_one_hot = List[Bool](length=n_features, fill=False)
    var feature_random = TRandom(params.random_seed ^ UInt64(0x4645415455524553))

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
            # the device's one body (gbdt/methods/random_score_bodies.mojo;
            # this was a host libm restatement, lane gbdt-cpu-rest-2)
            noise_mult = score_model_length_mult(
                Float64(n_rows), iteration, params.learning_rate
            )
        var tree_seed = noise_rand.next_uniform_l()
        if bootstrap_kind >= 0:
            var bm = _bootstrap_pass(bootstrap_kind, boot_seeds, stats, n_rows, bootstrap_param, stat_count)
            mags[0] = bm[0]
            mags[1] = bm[1]
        var fixed_scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        _snap_gradients(stats, n_rows, stat_count, fixed_scale)  # lane/sym-quality
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            score_std_dev = Float32(Float64(Float32(noise_mult * Float64(random_strength))) * _target_std_dev(stats, n_rows, stat_count, is_mc))
        var level_rand = TRandom(tree_seed)

        # ---- the per-tree feature sample (`feature_fraction < 1`) ----
        if ff_on:
            var tree_folds = _sample_tree_folds(
                grid.fold_counts, feature_fraction, feature_random
            )
            s_layout = build_layout(tree_folds, ff_one_hot)
            s_blocks = blocks_for(s_layout, n_rows)
            cindex = _ns_project_index(
                full_cindex, layout, s_layout, tree_folds, n_rows, n_features
            )
            s_hist_cells = s_layout.hist_cells
            s_bf_feature = List[Int](length=s_hist_cells, fill=0)
            s_bf_bin = List[Int](length=s_hist_cells, fill=0)
            for f in range(n_features):  # driver-loop: over the features and their borders (the tree layout)
                ref lf = s_layout.features[f]
                for b in range(Int(lf.folds)):
                    s_bf_feature[Int(lf.first_fold_index) + b] = f
                    s_bf_bin[Int(lf.first_fold_index) + b] = b

        # ---- `run_tree_layout_traced` at `stat_count` planes (TWIN of the
        # loop in `gbdt_oracle.mojo::gbdt_host_fit`) ----
        var row_index = List[Int](length=n_rows, fill=0)
        _identity_rows(row_index, n_rows)
        var p_off = List[Int](length=max_leaves, fill=0)
        var p_sz = List[Int](length=max_leaves, fill=0)
        p_sz[0] = n_rows
        var hist = List[Float32](
            length=max_leaves * stat_count * s_hist_cells, fill=Float32(0.0)
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
            for b in range(len(s_blocks)):
                ref blk = s_blocks[b]
                var total = 0
                for k in range(blk.count()):
                    total += Int(blk.folds[k])
                if blk.policy == POLICY_BINARY:
                    _binary_block_n(
                        blk, block_first_bin, s_hist_cells, stat_count, compute,
                        depth, p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_HALF_BYTE:
                    _half_byte_block_n(
                        blk, block_first_bin, s_hist_cells, stat_count, compute,
                        depth, p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_ONE_BYTE:
                    _one_byte_block_n(
                        blk, block_first_bin, s_hist_cells, stat_count, compute,
                        p_off, p_sz, row_index, stats, cindex, s_layout, n_rows,
                        fixed_scale, hist,
                    )
                block_first_bin += total

            # `scan_histograms_kernel` over the computed leaves, every plane
            _scan_histograms(hist, s_hist_cells, stat_count, compute, s_layout, n_features)

            # `substract_histograms_kernel`: the larger sibling, in place
            if planned and half > 0:
                _subtract_histograms(hist, s_hist_cells, sub_from, sub_what, half, stat_count)

            # `compute_partition_stats` over the live leaves, at the chunk
            # count of `stat_count` planes (`_stat_chunks`)
            var part_stats = _partition_stats_level(
                stats, n_rows, stat_count,
                _stat_chunks(stat_count),
                p_off, p_sz, n_live,
            )

            # the score and the device winner
            var gains = _cosine_gains_n(
                hist, s_hist_cells, stat_count, part_stats, n_live,
                params.l2_leaf_reg, is_mc, score_std_dev, level_seed, s_bf_feature,
            )
            var best_bin = _gain_winner(gains, s_hist_cells)
            var best_gain = -GBDT_FLOAT32_MAX
            if best_bin != GBDT_SENTINEL:
                best_gain = gains[Int(best_bin)]
            if best_bin != GBDT_SENTINEL:
                winners_score.append(ftz(best_gain))
            else:
                winners_score.append(ftz(-GBDT_FLOAT32_MAX))
            winners_bf.append(best_bin)
            var bf_split = 0
            if best_bin != GBDT_SENTINEL:
                bf_split = Int(best_bin)

            var split_f = s_bf_feature[bf_split] if s_hist_cells > 0 else 0
            var split_b = s_bf_bin[bf_split] if s_hist_cells > 0 else 0
            ref sfeat = s_layout.features[split_f]
            # the split chain, every live leaf, every plane
            _split_level(
                cindex, Int(sfeat.offset) * n_rows, sfeat.mask << sfeat.shift,
                UInt32(split_b) << sfeat.shift, sfeat.one_hot_feature, n_rows,
                n_live, s_hist_cells, row_index, stats, p_off, p_sz, hist,
                ids_compute, sub_from, sub_what, stat_count,
            )
            n_live = n_live * 2

        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        var grown = 0
        for d in range(max_depth):
            var best_score = winners_score[d]
            var best_bin_u = winners_bf[d]
            if best_bin_u == GBDT_SENTINEL or Int(best_bin_u) >= s_hist_cells:
                raise Error(
                    "All splits have infinite score. Probably, numerical"
                    " overflow occurs in loss function and/or split score"
                    " calculation. Try increasing l2_leaf_reg, and/or"
                    " decreasing learning_rate, etc."
                    " [level " + String(d) + ", live leaves "
                    + String(1 << d) + "]"
                )
            var cf_ = s_bf_feature[Int(best_bin_u)]
            var cb_ = s_bf_bin[Int(best_bin_u)]
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
        var scaled = rescaled_leaves(estimated, lr)
        apply_scaled_leaves_host(cursor, scaled, row_index, offsets, sizes, n_live, dim, n_rows)
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(estimated)):
            model_leaves.append(scaled[i])
        tree_leaf_offsets.append(len(model_leaves))

        # ---- `AppendModels(..., testCursor)` and the detector ----
        if has_test:
            if grown > 0:
                var test_bins = gbdt_eval_test_bins(
                    test_cindex, n_eval, tree_features, tree_bins,
                    feat_offset, feat_mask, feat_shift, feat_one_hot,
                )
                gbdt_eval_add_tree(test_cursor, test_bins, scaled, dim)
            # `_test_loss_enqueue`: the learn loss's own launch, unit weight
            _multi_pass(
                objective, num_classes, ev.y, no_weights, False, test_cursor,
                n_eval, True, False, test_stats, test_fv_part, test_mag_part,
            )
            var t_fv = _deterministic_sum_lanes(test_fv_part, 1, test_blocks)[0]
            var t_loss = -Float64(t_fv) / Float64(n_eval)
            test_losses.append(t_loss)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                # their break is BEFORE the learn loss append
                stopped_early = True
                break

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
    # ---- `ShrinkToBestIteration` (`gbdt/train.mojo:2241-2254`) ----
    var shrink_to = gbdt_eval_shrink_point(
        test_losses, ev.want_best_model, ev.best_model_min_trees
    )
    var n_trees = len(tree_split_offsets) - 1
    if 0 < shrink_to and shrink_to < n_trees:
        while len(tree_split_offsets) - 1 > shrink_to:
            _ = tree_split_offsets.pop()
            _ = tree_leaf_offsets.pop()
        split_features.resize(
            tree_split_offsets[len(tree_split_offsets) - 1], 0
        )
        split_bins.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
        model_leaves.resize(
            tree_leaf_offsets[len(tree_leaf_offsets) - 1], Float32(0.0)
        )
    var model = GbdtHostMultiModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, dim,
    )
    return GbdtHostMultiFit(
        model^,
        GbdtHostEvalFit(test_losses^, detector.best_iteration, stopped_early),
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
