# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the
gbdt-pointwise-l2-bayesian-eval lane: the doc-parallel (pointwise) searcher
with L2 scores, the Bayesian bootstrap, boost from average on Logloss, row
weights, an eval set with the Iter overfitting detector and best-model
truncation (lane/cpu-training-gbdt-ordered, 2026-09-15).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
`checks/fixed_point.choose_scale`, the GPU-free host modules the device fit
itself runs on the host (`gbdt/gpu_util/kernel/random_gen.mojo`'s
`next_uniform_f`, the overfitting detector, `build_layout`, `blocks_for`),
the symmetric oracle (the grid, the binarize, the cross-entropy row, the
halving folds, the pinned partition stats, the model text), the RMSE
oracle's bias record, and the ordered oracle's pointwise structure search,
whose single-task arm with the plain L2 scorer this fit runs.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-pointwise-l2-bayesian-eval`): SymmetricTree, `loss="Logloss"`,
`score_function="L2"`, `use_pointwise_searcher=True`,
`bootstrap_type="Bayesian"` at any temperature, `boost_from_average=True`,
row weights, an eval set, `od_type="Iter"` at any wait, `use_best_model`,
Newton leaves at the loss default, `random_strength` 0, no class weights,
no categoricals, no NaN, `feature_fraction` 1. The binding refuses by name
every other value of those options on this arm.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build)

  1. `train` (`gbdt/train.mojo:944-1897`): the grid and the binarize of the
     symmetric oracle, the eval rows binarized against the same borders,
     `use_best_model` resolved (unset is on with a non-constant eval
     target), the weights buffer.
  2. `fit_with_test` (`gbdt/methods/doc_parallel_boosting.mojo:853-2280`):
     `calc_one_dimensional_optimum_const_approx`'s Logloss arm
     (`gbdt/metrics/optimal_const_for_loss.mojo`, the weighted Float64 sums,
     the Float32 average, `-portable_log64(1 / p - 1)`), the learn and test
     cursors at its Float32, `create_bootstrap_seeds` (splitmix64).
  3. Per tree: `cross_entropy_kernel[True, False]` over the weighted rows
     (plane 0 the weight, plane 1 `ftz(w * (c - p))`, the score partials);
     `bootstrap_kernel[BAYESIAN]` (`gbdt/gpu_util/kernel/bootstrap.mojo`):
     one draw per row from the row's own seed, `-identical_log(u + 1e-20)`
     raised by `identical_pow` to the temperature, both planes multiplied,
     the per-block magnitude halving and the lane fold; `choose_scale` of
     the larger magnitude; the single-task pointwise structure search;
     `compute_bins_for_model` and `partition_from_bins`; the Newton walker
     over the weighted Logloss oracle (`WeightsCpu` from the one-stat
     partition reduce); `add_model_value_kernel`'s fused cursor update; the
     rescaled leaves; `compute_bins_and_add_kernel` onto the test cursor and
     the unweighted test loss; the detector.
  4. The final learn loss pass, `model.shrink` to the best iteration, and
     the model text with its `bias` record.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the Newton walker's Hessian
regularizer, so every leaf of every tree moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-pointwise-l2-bayesian-eval lane is the
measurement.
"""
from std.math import isfinite

from checks.fixed_point import choose_scale
from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
    portable_log64,
)
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_HALF_BYTE, POLICY_ONE_BYTE
from gbdt.gpu_util.kernel.random_gen import next_uniform_f
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostModel,
    _binarize_columns,
    _deterministic_sum_lanes,
    _halving_fold,
    _partition_stat,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_ordered import (
    _PwHelper,
    _ordered_tree_structure,
    _partition_stat_n,
)
from gbdt.host.gbdt_oracle_rmse import GbdtRmseHostFit, gbdt_rmse_host_model_text
from gbdt.overfitting_detector.overfitting_detector import (
    OD_ITER,
    make_overfitting_detector,
)


#: `BOOTSTRAP_SEED_COUNT` and `BOOTSTRAP_BLOCK_SIZE` (`bootstrap.mojo`).
comptime GBDT_PW_SEED_COUNT = 65536
comptime GBDT_PW_BOOT_BLOCK = 256


@fieldwise_init
struct GbdtPointwiseFit(Movable):
    var text: String
    var losses: List[Float64]
    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool


def _ce(target: Float32, val: Float32, border: Float32, weight: Float32) -> Tuple[Float32, Float32, Float32]:
    """One in-range thread of `cross_entropy_kernel[has_border=True]`:
    `(ftz(w * direction), ftz(w * scale), score)`."""
    var exp_val = identical_exp(val)
    var p = Float32(1.0)
    if isfinite(exp_val):
        p = exp_val / (Float32(1.0) + exp_val)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
    var c = Float32(1.0) if target > border else Float32(0.0)
    var direction = ftz(c - p)
    var scale = ftz(p * (Float32(1.0) - p))
    var log_exp_val_plus_one = val
    if isfinite(exp_val):
        log_exp_val_plus_one = identical_log(Float32(1.0) + exp_val)
    var score = weight * (c * val - log_exp_val_plus_one)
    return (ftz(weight * direction), ftz(weight * scale), score)


def _logloss_value(
    targets: List[Float32], weights: List[Float32], cursor: List[Float32],
    n: Int, border: Float32,
) -> Float32:
    """The score partials per 256-thread block, folded by
    `deterministic_sum_lanes_kernel[1]`."""
    var blocks = (n + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var part = List[Float32](length=blocks, fill=Float32(0.0))
    for b in range(blocks):
        var slab = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n:
                slab[t] = _ce(targets[i], cursor[i], border, weights[i])[2]
        part[b] = _halving_fold(slab)
    return _deterministic_sum_lanes(part, 1, blocks)[0]


def _estimate_logloss_leaves(
    targets: List[Float32],
    weights: List[Float32],
    cursor: List[Float32],
    row_index: List[Int],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    border: Float32,
    l2_leaf_reg: Float32,
    iterations: Int,
) raises -> List[Float32]:
    """`_estimate_and_apply`'s estimate with row weights: the gathers, the
    weighted oracle and `newton_like_walker_estimate` with AnyImprovement
    (the symmetric oracle's `_estimate_leaves`, weighted)."""
    var n_leaves = len(sizes)
    var g_target = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_weight = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    for pos in range(n_rows):
        g_target[pos] = targets[row_index[pos]]
        g_weight[pos] = weights[row_index[pos]]
        g_cursor[pos] = cursor[row_index[pos]]
    var bins = List[Int](length=n_rows, fill=0)
    for leaf in range(n_leaves):
        for k in range(sizes[leaf]):
            bins[offsets[leaf] + k] = leaf
    var weights_cpu = List[Float64]()
    for leaf in range(n_leaves):
        weights_cpu.append(Float64(_partition_stat_n(g_weight, n_rows, 0, offsets[leaf], sizes[leaf], 1)))
    var lambda_reg = Float64(l2_leaf_reg)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0
    var current_point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK

    comptime EPS_1E20F = Float64(Float32(1e-20))
    var cur_point = List[Float32](length=n_leaves, fill=Float32(0.0))

    var cur_value = Float64(0.0)
    var cur_grad = List[Float64]()
    var cached_der2 = List[Float64]()
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    # `move_to(zero)`: a zero shift added onto every gathered cursor row
    for pos in range(n_rows):
        g_cursor[pos] = g_cursor[pos] + (cur_point[bins[pos]] - current_point[bins[pos]])
    _oracle_eval_weighted(
        g_target, g_weight, g_cursor, offsets, sizes, n_rows, border,
        lambda_reg, cur_value, cur_grad, cached_der2, stats, fv,
    )
    var cur_hess = cached_der2.copy()
    var direction = List[Float32]()
    for i in range(n_leaves):
        direction.append(Float32(cur_grad[i] / (cur_hess[i] + EPS_1E20F)) if cur_hess[i] > 0 else Float32(0.0))

    if iterations == 1:
        var result = List[Float32]()
        for i in range(n_leaves):
            result.append(Float32(Float64(cur_point[i]) + 1.0 * Float64(direction[i])))
            if weights_cpu[i] < 1e-20:
                result[i] = Float32(0.0)
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
            var next_point = List[Float32]()
            for i in range(n_leaves):
                next_point.append(Float32(Float64(cur_point[i]) + step * Float64(direction[i])))
                if weights_cpu[i] < 1e-20:
                    next_point[i] = Float32(0.0)
            var shift = List[Float32](length=n_leaves, fill=Float32(0.0))
            for i in range(n_leaves):
                shift[i] = next_point[i] - current_point[i]
            for pos in range(n_rows):
                g_cursor[pos] = g_cursor[pos] + shift[bins[pos]]
            for i in range(n_leaves):
                current_point[i] = next_point[i]
            _oracle_eval_weighted(
                g_target, g_weight, g_cursor, offsets, sizes, n_rows, border,
                lambda_reg, next_value, next_grad, cached_der2, stats, fv,
            )
            if function_value <= next_value:
                cur_hess = cached_der2.copy()
                cur_point = next_point.copy()
                cur_value = next_value
                cur_grad = next_grad.copy()
                direction.clear()
                for i in range(n_leaves):
                    direction.append(Float32(cur_grad[i] / (cur_hess[i] + EPS_1E20F)) if cur_hess[i] > 0 else Float32(0.0))
                iteration += 1
                updated = True
                accepted = True
                break
            iteration += 1
            step /= 2
        if not accepted:
            break
    return cur_point^


def _oracle_eval_weighted(
    g_target: List[Float32],
    g_weight: List[Float32],
    g_cursor: List[Float32],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    border: Float32,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
    mut stats: List[Float32],
    mut fv: List[Float32],
):
    """`write_value_and_first_derivatives`' single-dimensional arm with row
    weights (`pointwise_oracle.mojo:458-573`)."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(blocks):
        var slab = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var r = _ce(g_target[i], g_cursor[i], border, g_weight[i])
                stats[i] = r[0]
                stats[n_rows + i] = r[1]
                slab[t] = r[2]
        fv[b] = _halving_fold(slab)
    gradient.clear()
    cached_der2.clear()
    for leaf in range(len(sizes)):
        gradient.append(Float64(_partition_stat(stats, n_rows, 0, offsets[leaf], sizes[leaf])))
        cached_der2.append(Float64(_partition_stat(stats, n_rows, 1, offsets[leaf], sizes[leaf])) + lambda_reg)
    var fv32 = Float32(0.0)
    for b in range(blocks):
        fv32 += fv[b]
    value = Float64(fv32)


def gbdt_pointwise_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    w: List[Float32],
    n_rows: Int,
    n_features: Int,
    eval_x_colmajor: List[Float32],
    eval_y: List[Float32],
    n_eval: Int,
    border_count: Int,
    border_build_max_samples: Int,
    n_estimators: Int,
    max_depth: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    random_seed: UInt64,
    nan_mode: Int,
    border: Float32,
    leaf_iterations: Int,
    bagging_temperature: Float32,
    od_wait: Int,
    use_best_model: Int,
    best_model_min_trees: Int,
) raises -> GbdtPointwiseFit:
    """`train` then `fit_with_test` on the covered configuration."""
    if n_rows < 1 or n_features < 1 or n_eval < 1:
        raise Error("the pointwise eval fit needs rows, features and an eval set")
    # ---- the grid, the binarize, the eval rows ----
    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, border_count,
        border_build_max_samples, random_seed, nan_mode,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
    var layout = build_layout(grid.fold_counts, one_hot)
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var test_cindex = _binarize_columns(eval_x_colmajor, n_eval, n_features, grid, layout)
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        if blk.policy != POLICY_ONE_BYTE and blk.policy != POLICY_HALF_BYTE:
            raise Error(
                "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a"
                " feature with exactly one border under"
                " use_pointwise_searcher=True (the BinaryFeatures histogram"
                " policy, feature " + String(blk.feature_ids[0]) + ")"
            )
        var gids = List[Int]()
        var offs = List[Int]()
        var firsts = List[Int]()
        var folds = List[Int]()
        var hist_line = 0
        for k in range(blk.count()):
            var f = blk.feature_ids[k]
            gids.append(f)
            offs.append(Int(layout.features[f].offset) * n_rows)
            firsts.append(Int(blk.fold_offset[k]))
            folds.append(Int(blk.folds[k]))
            hist_line += Int(blk.folds[k])
        helpers.append(_PwHelper(blk.policy, gids^, offs^, firsts^, folds^, hist_line, List[Float32](), List[Int](), List[Bool]()))
    var feat_offset = List[Int](length=n_features, fill=0)
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    for f in range(n_features):
        feat_offset[f] = Int(layout.features[f].offset) * n_rows
        feat_shift[f] = layout.features[f].shift
        feat_mask[f] = layout.features[f].mask

    var eval_const = True
    for r in range(1, n_eval):
        if eval_y[r] != eval_y[0]:
            eval_const = False
            break
    var want_best_model = use_best_model
    if want_best_model == -1:
        want_best_model = 0 if eval_const else 1

    # ---- boost from average (`calc_one_dimensional_optimum_const_approx`) ----
    var summary_weight = Float64(0.0)
    for i in range(n_rows):
        summary_weight += Float64(w[i])
    var target_sum = Float64(0.0)
    for i in range(n_rows):
        target_sum += Float64(y[i]) * Float64(w[i])
    var best_probability = Float64(Float32(target_sum / summary_weight))
    if best_probability <= 0.0 or best_probability >= 1.0:
        raise Error(
            "boost_from_average: the weighted mean target is "
            + String(best_probability) + ", outside (0, 1); a one-class pool"
            " has no finite log-odds"
        )
    var starting_approx = -portable_log64(1.0 / best_probability - 1.0)
    var start_value = Float32(starting_approx)
    var cursor = List[Float32](length=n_rows, fill=start_value)
    var test_cursor = List[Float32](length=n_eval, fill=start_value)
    var test_weights = List[Float32](length=n_eval, fill=Float32(1.0))

    # ---- `create_bootstrap_seeds` ----
    var seeds = List[UInt64](length=GBDT_PW_SEED_COUNT, fill=UInt64(0))
    var sx = random_seed
    for i in range(GBDT_PW_SEED_COUNT):
        sx += UInt64(0x9E3779B97F4A7C15)
        var z = sx
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        seeds[i] = z
    var boot_blocks = min(GBDT_PW_SEED_COUNT // GBDT_PW_BOOT_BLOCK, (n_rows + GBDT_PW_BOOT_BLOCK - 1) // GBDT_PW_BOOT_BLOCK)
    if boot_blocks < 1:
        boot_blocks = 1

    var detector = make_overfitting_detector(OD_ITER, False, Float64(-1.0), od_wait, True)
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var losses = List[Float64]()
    var test_losses = List[Float64]()
    var stopped_early = False
    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()
    var doc_ids = List[Int](length=n_rows, fill=0)
    for i in range(n_rows):
        doc_ids[i] = i
    var part_bounds = List[Int]()
    part_bounds.append(0)
    part_bounds.append(n_rows)

    for iteration in range(n_estimators):
        # ---- the search planes and the learn score ----
        var sw = List[Float32](length=n_rows, fill=Float32(0.0))
        var sg = List[Float32](length=n_rows, fill=Float32(0.0))
        var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
        for b in range(mse_blocks):
            var slab = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                if i < n_rows:
                    var r = _ce(y[i], cursor[i], border, w[i])
                    sw[i] = w[i]
                    sg[i] = r[0]
                    slab[t] = r[2]
            fv_part[b] = _halving_fold(slab)
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        # ---- `bootstrap_kernel[BAYESIAN]` ----
        var mag_part = List[Float32](length=2 * boot_blocks, fill=Float32(0.0))
        for blk in range(boot_blocks):
            var red_w = List[Float32](length=GBDT_PW_BOOT_BLOCK, fill=Float32(0.0))
            var red_g = List[Float32](length=GBDT_PW_BOOT_BLOCK, fill=Float32(0.0))
            for tid in range(GBDT_PW_BOOT_BLOCK):
                var gid = blk * GBDT_PW_BOOT_BLOCK + tid
                var s = seeds[gid]
                var mw = Float32(0.0)
                var mg = Float32(0.0)
                var i = gid
                while i < n_rows:
                    var draw = next_uniform_f(s)
                    s = draw[1]
                    var tmp = -identical_log(draw[0] + Float32(1e-20))
                    var bw = tmp
                    if bagging_temperature != Float32(1.0):
                        bw = identical_pow(tmp, bagging_temperature)
                    var wv = sw[i] * bw
                    sw[i] = wv
                    mw += abs(wv)
                    var gv = sg[i] * bw
                    sg[i] = gv
                    var gmax = Float32(0.0)
                    var a = abs(gv)
                    if a > gmax:
                        gmax = a
                    mg += gmax
                    i += boot_blocks * GBDT_PW_BOOT_BLOCK
                seeds[gid] = s
                red_w[tid] = mw
                red_g[tid] = mg
            mag_part[2 * blk] = _halving_fold(red_w)
            mag_part[2 * blk + 1] = _halving_fold(red_g)
        var mags = _deterministic_sum_lanes(mag_part, 2, boot_blocks)
        var m0 = Float64(mags[0])
        if m0 < 0.0:
            m0 = -m0
        var m1 = Float64(mags[1])
        if m1 < 0.0:
            m1 = -m1
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, n_rows))

        # ---- the single-task pointwise structure search ----
        var splits = _ordered_tree_structure(
            cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
            1, 0, scale, l2_leaf_reg, feat_offset, feat_shift, feat_mask,
            True,
        )
        # ---- `compute_bins_for_model`, `partition_from_bins` ----
        var n_leaves = 1 << len(splits)
        var bins = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            var leaf = 0
            for level in range(len(splits)):
                var fid = splits[level].feature
                var mask = feat_mask[fid] << feat_shift[fid]
                var value = UInt32(splits[level].bin) << feat_shift[fid]
                if (cindex[feat_offset[fid] + r] & mask) > value:
                    leaf += 1 << level
            bins[r] = leaf
        var sizes = List[Int](length=n_leaves, fill=0)
        for r in range(n_rows):
            sizes[bins[r]] += 1
        var offsets = List[Int](length=n_leaves, fill=0)
        var running = 0
        for i in range(n_leaves):
            offsets[i] = running
            running += sizes[i]
        var fill = offsets.copy()
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[fill[bins[r]]] = r
            fill[bins[r]] += 1
        # ---- the estimator and `add_model_value_kernel` ----
        var estimated = _estimate_logloss_leaves(
            y, w, cursor, row_index, offsets, sizes, n_rows, border,
            l2_leaf_reg, leaf_iterations,
        )
        for leaf in range(n_leaves):
            for k in range(sizes[leaf]):
                var row = row_index[offsets[leaf] + k]
                cursor[row] = identical_mul_add(estimated[leaf], learning_rate, cursor[row])
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        var tree_leaves = List[Float32]()
        for leaf in range(n_leaves):
            tree_leaves.append(estimated[leaf] * learning_rate)
            model_leaves.append(estimated[leaf] * learning_rate)
        tree_leaf_offsets.append(len(model_leaves))
        # ---- the test cursor, the test loss, the detector ----
        if len(splits) > 0:
            for r in range(n_eval):
                var leaf = 0
                for level in range(len(splits)):
                    var fid = splits[level].feature
                    var mask = feat_mask[fid] << feat_shift[fid]
                    var value = UInt32(splits[level].bin) << feat_shift[fid]
                    if (test_cindex[Int(layout.features[fid].offset) * n_eval + r] & mask) > value:
                        leaf += 1 << level
                test_cursor[r] = test_cursor[r] + tree_leaves[leaf]
        var t_loss = -Float64(_logloss_value(eval_y, test_weights, test_cursor, n_eval, border)) / Float64(n_eval)
        test_losses.append(t_loss)
        detector.add_error(t_loss)
        if detector.is_need_stop():
            stopped_early = True
            break
        if len(losses) < n_estimators:
            if len(tree_split_offsets) - 1 > 1:
                losses.append(-Float64(fv) / Float64(n_rows))

    losses.append(-Float64(_logloss_value(y, w, cursor, n_rows, border)) / Float64(n_rows))

    # ---- `use_best_model`: `model.shrink(best_iter)` ----
    if want_best_model == 1 and len(test_losses) > 0:
        var min_trees_best = -1
        var min_trees_err = Float64(0.0)
        for i in range(len(test_losses)):
            if i + 1 < best_model_min_trees:
                continue
            if min_trees_best < 0 or test_losses[i] < min_trees_err:
                min_trees_err = test_losses[i]
                min_trees_best = i
        var best_iter = min_trees_best + 1
        var n_trees = len(tree_split_offsets) - 1
        if 0 < best_iter and best_iter < n_trees:
            while len(tree_split_offsets) - 1 > best_iter:
                _ = tree_split_offsets.pop()
                _ = tree_leaf_offsets.pop()
            split_features.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
            split_bins.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
            model_leaves.resize(tree_leaf_offsets[len(tree_leaf_offsets) - 1], Float32(0.0))

    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses.copy(), detector.best_iteration, stopped_early,
    )
    var text = gbdt_rmse_host_model_text(GbdtRmseHostFit(model^, starting_approx))
    return GbdtPointwiseFit(text^, losses^, test_losses^, detector.best_iteration, stopped_early)
