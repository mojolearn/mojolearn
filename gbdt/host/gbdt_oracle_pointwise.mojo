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
from std.math import fma, isfinite

from checks.fixed_point import choose_scale
from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
    portable_log64,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, HostU32Ptr, host_predict_chunk
from gbdt.methods.cursor_rescale import rescaled_leaves
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_HALF_BYTE, POLICY_ONE_BYTE
from gbdt.gpu_util.kernel.random_gen import next_uniform_f
from gbdt.host.gbdt_oracle import (
    apply_scaled_leaves_host,
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostModel,
    _binarize_columns,
    _deterministic_sum_lanes,
    _halving_fold,
    _snap_plane,
    _partition_stats_level,
    _pinned_partition_stats,
    _partition_for_tree,
    _oracle_move_to,
    _logloss_value,
    _gather_rows,
    _identity_rows,
    _leaf_bins,
    _gbdt_units,
    GbdtIntPtr,
    GbdtU64Ptr,
    GBDT_PARTITION_CHUNKS,
    GBDT_PINNED_SM,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_ordered import (
    _PwHelper,
    _ordered_tree_structure,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_LOGLOSS,
    GBDT_OBJ_RMSE,
    GbdtHostLoss,
    _estimate_leaves_for_loss,
    _loss_row,
    _loss_value,
)
from gbdt.host.gbdt_oracle_rmse import GbdtRmseHostFit, gbdt_rmse_host_model_text
from gbdt.metrics.sample_quantile import calculate_weighted_target_average
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


def _pw_search_pass(
    loss: GbdtHostLoss,
    targets: List[Float32],
    weights: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    mut sw: List[Float32],
    mut sg: List[Float32],
    mut fv_partials: List[Float32],
):
    """The search planes and the learn score of one tree (the loss's
    target kernel with row weights, `_loss_row` of gbdt_oracle_losses.mojo:
    `cross_entropy_kernel[has_border=True]` for Logloss, the pointwise
    target kernel for RMSE): `sw` the row's weight, `sg` the flushed
    weighted derivative, one score partial per 256-thread block through the
    halving tree. Lane gbdt-cpu-rest-2 (2026-11-23): this file's own
    Logloss row (`_ce`, a third copy of that kernel's math) is deleted.

    The unit is one 256-thread device block (its rows' two planes, its own
    partial and halving tree), as `_logloss_search_pass`; blocks go round
    robin to `_gbdt_units` tasks."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var units = _gbdt_units(blocks, 16 * n_rows)
    var objective = loss.objective
    var alpha = loss.kernel_alpha
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm objective, imm alpha, imm targets_p, imm weights_p, imm cursor_p, imm sw_p, imm sg_p, imm fv_p}:
        var slab = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slab is reused)
                slab[t] = Float32(0.0)
                if i < n_rows:
                    var r = _loss_row(objective, targets_p[i], cursor_p[i], alpha, weights_p[i])
                    sw_p[i] = weights_p[i]
                    sg_p[i] = r.der
                    slab[t] = r.score
            fv_p[b] = _halving_fold(slab)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(targets)
    _ = len(weights)
    _ = len(cursor)
    _ = len(sw)
    _ = len(sg)
    _ = len(fv_partials)


def _pw_bootstrap(
    mut seeds: List[UInt64],
    mut sw: List[Float32],
    mut sg: List[Float32],
    n_rows: Int,
    boot_blocks: Int,
    bagging_temperature: Float32,
    mut mag_part: List[Float32],
):
    """`bootstrap_kernel[BAYESIAN]` over the two search planes: thread `gid`
    of block `blk` owns seed `gid` and the rows `gid, gid + stride, ...`,
    its draws and two magnitudes ascending in its rows, then the block's
    halving folds.

    The unit is one device block, as `_bootstrap_pass`: no block reads or
    writes another's seeds, rows or magnitude pair. Blocks go round robin to
    `_gbdt_units` tasks; a task holds two block slabs, every lane rewritten
    per block."""
    var stride = boot_blocks * GBDT_PW_BOOT_BLOCK
    var units = _gbdt_units(boot_blocks, 16 * n_rows)
    var seeds_p = rebind[GbdtU64Ptr](seeds.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_part.unsafe_ptr())

    def block_draws(unit: Int) {imm units, imm boot_blocks, imm stride, imm n_rows, imm bagging_temperature, imm seeds_p, imm sw_p, imm sg_p, imm mag_p}:
        var red_w = List[Float32](length=GBDT_PW_BOOT_BLOCK, fill=Float32(0.0))
        var red_g = List[Float32](length=GBDT_PW_BOOT_BLOCK, fill=Float32(0.0))
        var blk = unit
        while blk < boot_blocks:
            for tid in range(GBDT_PW_BOOT_BLOCK):
                var gid = blk * GBDT_PW_BOOT_BLOCK + tid
                var s = seeds_p[gid]
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
                    var wv = sw_p[i] * bw
                    sw_p[i] = wv
                    mw += abs(wv)
                    var gv = sg_p[i] * bw
                    sg_p[i] = gv
                    var gmax = Float32(0.0)
                    var a = abs(gv)
                    if a > gmax:
                        gmax = a
                    mg += gmax
                    i += stride
                seeds_p[gid] = s
                red_w[tid] = mw
                red_g[tid] = mg
            mag_p[2 * blk] = _halving_fold(red_w)
            mag_p[2 * blk + 1] = _halving_fold(red_g)
            blk += units

    host_parallelize(block_draws, units)
    _ = len(seeds)
    _ = len(sw)
    _ = len(sg)
    _ = len(mag_part)


def _pw_tree_onto_rows(
    cindex: List[UInt32],
    level_offsets: List[Int],
    level_masks: List[UInt32],
    level_values: List[UInt32],
    leaves: List[Float32],
    n: Int,
    mut cursor: List[Float32],
):
    """One oblivious tree onto a pool's cursor (`compute_bins_kernel` then
    `add_bin_model_value_kernel`): a row's leaf is the sum of `1 << level`
    over the levels whose TakeGreater split it passes, then one add. A row
    is one device thread; the rows run in `_gbdt_units` contiguous ranges."""
    var depth = len(level_offsets)
    var units = _gbdt_units(n, (2 + 3 * depth) * n)
    var chunk = host_predict_chunk(n, units)
    var cindex_p = rebind[HostU32Ptr](cindex.unsafe_ptr())
    var off_p = rebind[GbdtIntPtr](level_offsets.unsafe_ptr())
    var mask_p = rebind[HostU32Ptr](level_masks.unsafe_ptr())
    var value_p = rebind[HostU32Ptr](level_values.unsafe_ptr())
    var leaves_p = rebind[HostF32Ptr](leaves.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm depth, imm cindex_p, imm off_p, imm mask_p, imm value_p, imm leaves_p, imm cursor_p}:
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var leaf = 0
            for level in range(depth):
                if (cindex_p[off_p[level] + r] & mask_p[level]) > value_p[level]:
                    leaf += 1 << level
            cursor_p[r] = cursor_p[r] + leaves_p[leaf]

    host_parallelize(rows, units)
    _ = len(cindex)
    _ = len(level_offsets)
    _ = len(level_masks)
    _ = len(level_values)
    _ = len(leaves)
    _ = len(cursor)


# Lane gbdt-cpu-rest-2 (2026-11-23): `_estimate_logloss_leaves` and
# `_oracle_eval_weighted`, this file's copy of the weighted Newton walker and
# its Logloss oracle, are deleted: the fit calls the shared estimator
# (gbdt_oracle_losses.mojo `_estimate_leaves_for_loss`, the walker of every
# pointwise loss). Unweighted fits pass no weights there: a leaf's weight is
# then its row count, which the Float32 sum of ones it replaces equals
# exactly below 2^24 rows, and `w * x == x` at w = 1.


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
    loss: GbdtHostLoss,
    bagging_temperature: Float32,
    od_wait: Int,
    use_best_model: Int,
    best_model_min_trees: Int,
    has_weights: Bool = True,
) raises -> GbdtPointwiseFit:
    """`train` then `fit_with_test` on the covered configuration.

    Lane gbdt-cpu-rest (2026-11-23): `has_weights` False is the device's
    unweighted arm, `w` then a ones column (every weighted product below is
    then exact, `w * x == x`); the one place the arms differ is the
    starting constant's `summaryWeight`, the integer row count on the
    unweighted branch of `calculate_weighted_target_average`. `n_eval` 0
    is a fit without a held-out set: no test cursor, no detector, no
    best-model shrink, as `fit_with_test` without a test pool.

    Lane gbdt-cpu-rest-2 (2026-11-23): `loss` is Logloss (its target
    border in `kernel_alpha`) or RMSE, Newton leaves, as the binding
    resolves them; the device's pointwise arm runs the loss's own target
    kernel for the search planes and the shared leaf estimator on the
    structure (`doc_parallel_boosting.mojo`, DEVIATION 104: the searcher
    returns no leaves, so RMSE estimates too). Every pass reads the loss
    through `_loss_row` / `_loss_value` / `_estimate_leaves_for_loss`
    (gbdt_oracle_losses.mojo), the bodies the greedy pointwise fit calls.
    Boost from average is the loss's optimum constant: Logloss's log-odds,
    RMSE's weighted mean (`_rmse_starting_approx`'s arm)."""
    var is_logloss = loss.objective == GBDT_OBJ_LOGLOSS
    if not is_logloss and loss.objective != GBDT_OBJ_RMSE:
        raise Error("the pointwise host fit takes Logloss or RMSE")
    if loss.method != GBDT_LEAF_NEWTON:
        raise Error("the pointwise host fit takes Newton leaves")
    if n_rows < 1 or n_features < 1 or n_eval < 0:
        raise Error("the pointwise fit needs rows and features")
    var has_test = n_eval > 0
    # ---- the grid, the binarize, the eval rows ----
    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, border_count,
        border_build_max_samples, random_seed, nan_mode,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
    var layout = build_layout(grid.fold_counts, one_hot)
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var test_cindex = List[UInt32]()
    if has_test:
        test_cindex = _binarize_columns(eval_x_colmajor, n_eval, n_features, grid, layout)
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        # every policy, BinaryFeatures included: `_ordered_tree_structure`
        # builds the binary cells (`_pw_binary_cells`, the
        # `pw_hb_binary_sum` writeback) for a one-border feature
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
    # the shared weighted arm: `bfa_tree_sum`'s fold order, the device's
    var best_probability = Float64(
        calculate_weighted_target_average(y, w, has_weights)
    )
    # RMSE: the weighted mean itself (the RMSE arm of
    # `calc_one_dimensional_optimum_const_approx`)
    var starting_approx = best_probability
    if is_logloss:
        if best_probability <= 0.0 or best_probability >= 1.0:
            raise Error(
                "boost_from_average: the weighted mean target is "
                + String(best_probability) + ", outside (0, 1); a one-class pool"
                " has no finite log-odds"
            )
        starting_approx = -portable_log64(1.0 / best_probability - 1.0)
    var start_value = Float32(starting_approx)
    var cursor = List[Float32](length=n_rows, fill=start_value)
    var test_cursor = List[Float32](length=n_eval, fill=start_value)
    # the estimator's row weights: none on the unweighted arm (once per fit)
    var est_weights = List[Float32]()
    if has_weights:
        est_weights = w.copy()

    # ---- `create_bootstrap_seeds` ----
    var seeds = List[UInt64](length=GBDT_PW_SEED_COUNT, fill=UInt64(0))
    var sx = random_seed
    # KEPT SERIAL: once per fit, a fixed 65536 seeds off one running
    # counter (a few integer operations each)
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
    _identity_rows(doc_ids, n_rows)
    var part_bounds = List[Int]()
    part_bounds.append(0)
    part_bounds.append(n_rows)

    for iteration in range(n_estimators):
        # ---- the search planes and the learn score ----
        var sw = List[Float32](length=n_rows, fill=Float32(0.0))
        var sg = List[Float32](length=n_rows, fill=Float32(0.0))
        var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
        _pw_search_pass(loss, y, w, cursor, n_rows, sw, sg, fv_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        # ---- `bootstrap_kernel[BAYESIAN]` ----
        var mag_part = List[Float32](length=2 * boot_blocks, fill=Float32(0.0))
        _pw_bootstrap(
            seeds, sw, sg, n_rows, boot_blocks, bagging_temperature, mag_part
        )
        var mags = _deterministic_sum_lanes(mag_part, 2, boot_blocks)
        var m0 = Float64(mags[0])
        if m0 < 0.0:
            m0 = -m0
        var m1 = Float64(mags[1])
        if m1 < 0.0:
            m1 = -m1
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, n_rows))
        # lane/sym-quality: `enqueue_snap_plane` (doc_parallel_boosting's
        # pointwise arm), the gradient plane onto the tree's grid
        _snap_plane(sg, n_rows, scale)

        # ---- the single-task pointwise structure search ----
        var splits = _ordered_tree_structure(
            cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
            1, 0, scale, l2_leaf_reg, feat_offset, feat_shift, feat_mask,
            List[Bool](), True,
        )
        # ---- `compute_bins_for_model`, `partition_from_bins` ----
        # (`_partition_for_model`: the rows' leaves and the stable counting
        # sort, rows ascending within a leaf, across the cores)
        var n_leaves = 1 << len(splits)
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        for level in range(len(splits)):  # driver-loop: over the tree's levels
            tree_features.append(splits[level].feature)
            tree_bins.append(splits[level].bin)
        var partition = _partition_for_tree(
            cindex, layout, tree_features, tree_bins, n_rows
        )
        # ---- the estimator and `add_model_value_kernel` ----
        # the shared estimator; unweighted, no weights (see the note above)
        var estimated = _estimate_leaves_for_loss(
            loss, y, cursor, partition.row_index, partition.offsets,
            partition.sizes, n_rows, l2_leaf_reg,
            weights=est_weights,
        )
        var scaled = rescaled_leaves(estimated, learning_rate)
        apply_scaled_leaves_host(
            cursor, scaled, partition.row_index, partition.offsets,
            partition.sizes, n_leaves,
        )
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        var tree_leaves = List[Float32]()
        for leaf in range(n_leaves):
            tree_leaves.append(scaled[leaf])
            model_leaves.append(scaled[leaf])
        tree_leaf_offsets.append(len(model_leaves))
        # ---- the test cursor, the test loss, the detector ----
        if not has_test:
            if len(losses) < n_estimators:
                if len(tree_split_offsets) - 1 > 1:
                    losses.append(-Float64(fv) / Float64(n_rows))
            continue
        if len(splits) > 0:
            var level_offsets = List[Int]()
            var level_masks = List[UInt32]()
            var level_values = List[UInt32]()
            for level in range(len(splits)):  # driver-loop: over the tree's levels
                var fid = splits[level].feature
                level_offsets.append(Int(layout.features[fid].offset) * n_eval)
                level_masks.append(feat_mask[fid] << feat_shift[fid])
                level_values.append(UInt32(splits[level].bin) << feat_shift[fid])
            _pw_tree_onto_rows(
                test_cindex, level_offsets, level_masks, level_values,
                tree_leaves, n_eval, test_cursor,
            )
        var t_loss = -Float64(_loss_value(loss, eval_y, test_cursor, n_eval)) / Float64(n_eval)
        test_losses.append(t_loss)
        detector.add_error(t_loss)
        if detector.is_need_stop():
            stopped_early = True
            break
        if len(losses) < n_estimators:
            if len(tree_split_offsets) - 1 > 1:
                losses.append(-Float64(fv) / Float64(n_rows))

    losses.append(-Float64(_loss_value(loss, y, cursor, n_rows, w)) / Float64(n_rows))

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
