# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the gbdt-rmse lane, the RMSE arm
of `gbdt/host/gbdt_oracle.mojo` (workstream E batch 3, 2026-09-14; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md, the batch 3 gbdt-rmse section).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
the GPU-free host modules the symmetric oracle already reuses (the layout and
the policy blocks), and the symmetric oracle itself, whose restatements of
the device kernels this fit reaches unchanged (the grid and the binarize,
both histogram families, the pinned partition stats, the Cosine score, the
lane folds, `choose_scale_kernel`, the model text).

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py `gbdt-rmse`:
20 trees, depth 6, `loss="RMSE"`, every other option at its default). It is
the gbdt-symmetric configuration with the loss swapped, and the binding
refuses by name what it refuses there, plus `leaf_estimation_iterations`
other than 1 under RMSE (bindings/_mojolearn_gbdt_host.mojo, `_refuse`).
`boost_from_average` unset or True resolves to the seeded cursor and False
to the zero cursor; both are carried.

WHY THE LEAVES COME FROM THE SEARCHER. `fit_with_test` clears
`need_estimation` for RMSE with Newton leaves at one iteration and one
permutation (DEVIATION 64, `gbdt/methods/doc_parallel_boosting.mojo:
1294-1305`), and `train` resolves every lane option into that arm: RMSE's
default method is Newton at one iteration
(`gbdt/options/catboost_options.mojo:1270-1274`), `use_exact_leaves` does
not list RMSE, and a fit with no CTR feature has one permutation
(`gbdt/train.mojo:1126-1136`). So the greedy arm runs
`run_tree_layout_traced` with `export_offsets=False` and
`apply_to_cursor=True` (`doc_parallel_boosting.mojo:1997-2028`), and the
walker never runs.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build, the
default flags the symmetric oracle names)

  1. `AdjustBoostFromAverageDefaultValue` (`gbdt/train.mojo:1580-1605`):
     unset resolves TRUE for RMSE. `_rmse_starting_approx` below.
  2. The grid, the layout, the binary policy refusal and the binarize: the
     symmetric oracle's `gbdt_host_grid` and `_binarize_columns`, called.
  3. `calc_one_dimensional_optimum_const_approx`'s RMSE arm
     (`gbdt/metrics/optimal_const_for_loss.mojo:52-111`, unweighted branch:
     the Float64 row-order sum over `Float64(n)`, narrowed to Float32 at
     the return, widened back), the bias it sets and the Float32 cursor
     fill (`doc_parallel_boosting.mojo:1131-1164`). Restated, not imported:
     that module imports a kernel module.
  4. Per tree, the search pass `launch_approximate[False]` on RMSE
     (`doc_parallel_boosting.mojo:1544-1552`), which is
     `pointwise_target_kernel[OBJECTIVE_RMSE, False, False]`
     (`gbdt/targets/kernel/pointwise_targets.mojo:560-802`): plane 0 the
     unit weight, plane 1 `ftz(weight * (t - p))`, the score partial
     `-weight * ((t - p) * (t - p))` and the magnitude partials `|plane 0|`,
     `|der|` per 256-thread block through the halving tree
     (`pinned_block_sum`, `:147-179`), folded by
     `deterministic_sum_lanes_kernel` (`:808-863`), then
     `choose_scale_kernel`. `_rmse_search_pass` below.
  5. `run_tree_layout_traced`, every level, the gates and the rollback
     (`greedy_search_helper.mojo:4677-5682`). The level loop below is a
     TWIN of the one in `gbdt_host_fit` (`gbdt_oracle.mojo`); an edit to
     either must visit the other.
  6. The tail under `apply_to_cursor` (`greedy_search_helper.mojo:
     5692-5739`): `compute_partition_stats` over the final (rolled back)
     partitions at the pinned 32 chunks, `compute_leaf_values_kernel`
     (`gbdt/methods/leaves_estimation/leaves_estimation.mojo:75-181`: the
     Hessian `w + l2`, its `<= 0` guard, `g / (hessian + 1e-20)`, and
     `RegularizeImpl`'s `w < 1e-20` zero), and `add_model_value_kernel`
     (`gbdt/methods/kernel_add_model_value.mojo:45-104`, the cursor update
     through `identical_mul_add`). `_rmse_leaf_value` below.
  7. The rescale on append (`doc_parallel_boosting.mojo:2131-2133`), the
     learn losses (`:2158-2169`, the final pass `:2180-2259`) and the model
     text with its `bias` record, written after `losses` when the bias is
     not zero (`gbdt/models/model_text.mojo:405-410`).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`, the define the
symmetric oracle reads (`GBDT_ORACLE_HOST_SABOTAGE`), adds 1.0 to the l2
regularizer of `_rmse_leaf_value`, so every leaf with a nonzero gradient sum
of every tree moves, the cursor moves with it, and every fixture's model and
predictions move. The symmetric arm's walker sabotage never runs here.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-rmse lane is the measurement.
"""
from std.math import exp, log
from std.memory import bitcast
from checks.numerics import ftz, identical_mul_add
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.host.gbdt_oracle import (
    GBDT_FLOAT32_MAX,
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_SENTINEL,
    GbdtHostModel,
    GbdtHostParams,
    _binarize_columns,
    _binary_block,
    _bootstrap_pass,
    _choose_scale_from_magnitudes,
    _cosine_gain,
    _cosine_gains,
    _deterministic_sum_lanes,
    _half_byte_block,
    _halving_fold,
    _one_byte_block,
    _partition_stat,
    _target_std_dev,
    gbdt_bootstrap_seeds,
    gbdt_f64_token,
    gbdt_host_grid,
    gbdt_host_model_text,
)


#: `MIN_LEAF_WEIGHT` (`leaves_estimation.mojo:72`).
comptime GBDT_RMSE_MIN_LEAF_WEIGHT = Float32(1e-20)


@fieldwise_init
struct GbdtRmseHostFit(Movable):
    """The fitted ensemble and its bias (`TAdditiveModel.bias`, the starting
    approx `boost_from_average` sets; 0.0 without it)."""

    var model: GbdtHostModel
    var bias: Float64


# ===========================================================================
# THE STARTING APPROX
# ===========================================================================


def _rmse_starting_approx(targets: List[Float32], n_rows: Int) raises -> Float64:
    """`calculate_weighted_target_average` on the unweighted branch, through
    the RMSE arm of `calc_one_dimensional_optimum_const_approx`
    (`optimal_const_for_loss.mojo:52-111`): `summary_weight` is the exact
    row count, the target sum accumulates in Float64 in row order, the
    quotient narrows to Float32 at the return and widens back."""
    if n_rows == 0:
        raise Error("optimal const approx: empty target")
    var summary_weight = Float64(n_rows)
    var target_sum = Float64(0.0)
    for i in range(n_rows):
        target_sum += Float64(targets[i])
    return Float64(Float32(target_sum / summary_weight))


# ===========================================================================
# THE TARGET KERNEL: `pointwise_target_kernel[OBJECTIVE_RMSE]`
# ===========================================================================


@fieldwise_init
struct _RmseRow(ImplicitlyCopyable, Movable):
    var der: Float32
    var score: Float32


def _rmse_row(relev: Float32, val: Float32, weight: Float32) -> _RmseRow:
    """One in-range thread of `pointwise_target_kernel[OBJECTIVE_RMSE,
    estimation=False]` (`pointwise_targets.mojo:696-775`): `target_der` is
    `t - p` (`:421-423`), stored flushed as `ftz(weight * der)`;
    `target_score` is `(t - p) * (t - p)` (`:348-350`), accumulated as
    `-weight * score`."""
    var der = ftz(weight * (relev - val))
    var score = -weight * ((relev - val) * (relev - val))
    return _RmseRow(der, score)


def _rmse_search_pass(
    targets: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_approximate[False]` on RMSE with `compute_fv` and
    `compute_magnitudes` set: plane 0 the unit weight, plane 1 the flushed
    der, one score partial and two magnitude partials per 256-thread block
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
                var r = _rmse_row(targets[i], cursor[i], weight)
                var plane0 = weight
                stats[i] = plane0
                stats[n_rows + i] = r.der
                s_score[t] = r.score
                s_w[t] = abs(plane0)
                s_g[t] = abs(r.der)
        fv_partials[b] = _halving_fold(s_score)
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def _rmse_value(targets: List[Float32], cursor: List[Float32], n_rows: Int) -> Float32:
    """The final learn loss pass (`doc_parallel_boosting.mojo:2180-2211`):
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
                s_score[t] = _rmse_row(targets[i], cursor[i], weight).score
        fv_partials[b] = _halving_fold(s_score)
    return _deterministic_sum_lanes(fv_partials, 1, blocks)[0]


# ===========================================================================
# THE SEARCHER'S LEAF: `compute_leaf_values_kernel`
# ===========================================================================


def _rmse_leaf_value(w: Float32, g: Float32, l2: Float32) -> Float32:
    """`compute_leaf_values_kernel` (`leaves_estimation.mojo:75-181`) for
    one leaf, term for term. The sabotage arm adds 1.0 to the regularizer
    (see THE NEGATIVE CONTROL)."""
    var reg = l2
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        reg = reg + Float32(1.0)
    var hessian = w + reg
    if hessian <= Float32(0.0):
        return Float32(0.0)
    var v = g / (hessian + Float32(1e-20))
    if w < GBDT_RMSE_MIN_LEAF_WEIGHT:
        v = Float32(0.0)
    return v


# ===========================================================================
# THE FIT
# ===========================================================================


def gbdt_rmse_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    boost_from_average: Bool,
    bootstrap_kind: Int = -1,
    bootstrap_param: Float32 = Float32(1.0),
    random_strength: Float32 = Float32(0.0),
) raises -> GbdtRmseHostFit:
    """`train` then `fit_with_test` on RMSE with the searcher's leaves (see
    the module docstring for what that covers and what mirrors what).
    `boost_from_average` is the RESOLVED flag (`train.mojo:1580-1605`).

    `bootstrap_kind` (`GBDT_BOOT_*`, -1 none) and `random_strength` are the
    stochastic arm of `gbdt_oracle.mojo::gbdt_host_fit_eval`, the same draws
    in the same order. The searcher's leaves are read off the stats planes
    (`need_estimation` is False, `doc_parallel_boosting.mojo:1484-1497`), so
    under a bootstrap they are the BOOTSTRAPPED planes' leaves, as the
    device's `apply_to_cursor` tail computes them."""
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
    var layout = build_layout(grid.fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
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

    # `starting_approx`, `model.bias` and `enqueue_fill(ctx, cursor,
    # start_value)` (`doc_parallel_boosting.mojo:1131-1164`)
    var starting_approx = Float64(0.0)
    if boost_from_average:
        starting_approx = _rmse_starting_approx(y, n_rows)
    var start_value = Float32(starting_approx)
    var cursor = List[Float32](length=n_rows, fill=start_value)
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

    # the stochastic arm (`doc_parallel_boosting.mojo:1470-1523`)
    var bootstrap_on = bootstrap_kind >= 0
    var boot_seeds = List[UInt64]()
    if bootstrap_on:
        boot_seeds = gbdt_bootstrap_seeds(params.random_seed)
    var noise_rand = TRandom(params.random_seed)

    for iteration in range(params.n_estimators):
        # ---- the gradients, the learn loss and the magnitudes ----
        _rmse_search_pass(y, cursor, n_rows, stats, fv_part, mag_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
        # `calc_score_model_length_mult` and the per-tree seed, drawn every
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
        # `run_tree_layout`'s ScoreStdDev over the bootstrapped planes
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            score_std_dev = Float32(
                Float64(Float32(noise_mult * Float64(random_strength)))
                * _target_std_dev(stats, n_rows)
            )
        var level_rand = TRandom(tree_seed)

        # ---- `run_tree_layout_traced`, every level (TWIN of the loop in
        # `gbdt_oracle.mojo::gbdt_host_fit`) ----
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
                if blk.policy == POLICY_BINARY:
                    _binary_block(
                        blk, block_first_bin, hist_cells, compute, depth,
                        p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_HALF_BYTE:
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
            var gains = _cosine_gains(
                hist, hist_cells, part_stats, n_live, params.l2_leaf_reg,
                score_std_dev, level_seed, bf_feature,
            )
            for bf in range(hist_cells):
                var gain = gains[bf]
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

        # ---- the tail under `apply_to_cursor`: the final partition stats
        # over the reordered stats planes, the searcher's leaf values, the
        # cursor update ----
        var leaf_values = List[Float32](length=n_live, fill=Float32(0.0))
        for i in range(n_live):
            var w = _partition_stat(stats, n_rows, 0, offsets[i], sizes[i])
            var g = _partition_stat(stats, n_rows, 1, offsets[i], sizes[i])
            leaf_values[i] = _rmse_leaf_value(w, g, params.l2_leaf_reg)
        for leaf in range(n_live):
            for k in range(sizes[leaf]):
                var row = row_index[offsets[leaf] + k]
                cursor[row] = identical_mul_add(leaf_values[leaf], lr, cursor[row])

        # ---- `AppendModels`: the structure and the rescaled leaves ----
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(leaf_values)):
            model_leaves.append(leaf_values[i] * lr)
        tree_leaf_offsets.append(len(model_leaves))

        # the learn loss read alongside this iteration's gradients
        if len(losses) < params.n_estimators:
            var v = Float64(fv)
            if iteration + 1 > 1:
                losses.append(-v / Float64(n_rows))

    losses.append(-Float64(_rmse_value(y, cursor, n_rows)) / Float64(n_rows))
    return GbdtRmseHostFit(
        GbdtHostModel(
            grid.fold_counts.copy(), grid.borders.copy(),
            grid.nan_treatment.copy(), tree_split_offsets^, split_features^,
            split_bins^, tree_leaf_offsets^, model_leaves^, losses^, 0, False,
        ),
        starting_approx,
    )


def gbdt_text_with_bias(base: String, n_losses: Int, bias: Float64) raises -> String:
    """`model_text`'s `bias` record (`model_text.mojo:405-410`), written
    after the `losses` header record when the bias is not zero BY BITS (a
    -0.0 bias is written). Shared by the symmetric and non-symmetric text."""
    if bitcast[DType.uint64](bias) == UInt64(0):
        return base.copy()
    var header = String("losses ") + String(n_losses)
    var out = String("")
    var inserted = False
    var pieces = base.split("\n")
    for p in range(len(pieces)):
        var line = String(pieces[p])
        if p + 1 == len(pieces):
            # the text ends in a newline, so the last piece is the empty tail
            out += line
            break
        out += line + "\n"
        if not inserted and line == header:
            out += String("bias ") + gbdt_f64_token(bias) + "\n"
            inserted = True
    if not inserted:
        raise Error("gbdt host: the model text has no `losses` header record")
    return out^


def gbdt_rmse_host_model_text(fit: GbdtRmseHostFit) raises -> String:
    """The symmetric oracle's text with the `bias` record a seeded fit adds."""
    return gbdt_text_with_bias(
        gbdt_host_model_text(fit.model), len(fit.model.losses), fit.bias
    )
