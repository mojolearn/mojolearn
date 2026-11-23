# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the doc-parallel (pointwise)
searcher on Logloss and RMSE: the gbdt-pointwise-l2-bayesian-eval lane
(lane/cpu-training-gbdt-ordered, 2026-09-15) and, since lane gbdt-rest-3
(2026-11-23), every score function, bootstrap, leaf method and
`random_strength` the device's pointwise arm trains on those two losses.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
the GPU-free host modules the device fit itself runs on the host
(`gbdt/gpu_util/kernel/random_gen.mojo`'s generators through the bootstrap,
`TRandom`, the overfitting detector, `build_layout`,
`random_score_bodies.score_model_length_mult`), the symmetric oracle (the
grid, the binarize, the bootstrap pass, the scale, the partition, the model
text), the loss oracle (the target pass, the loss value, the shared leaf
estimator), the grouped pointwise fit's noise magnitude and plane split,
and the ordered oracle's single-fold structure search.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-pointwise-l2-bayesian-eval`, `gbdt-pointwise-arms`): SymmetricTree,
`loss` Logloss or RMSE, `use_pointwise_searcher=True`, `score_function` L2,
NewtonL2, Cosine or NewtonCosine, `bootstrap_type` Bayesian (any
temperature), Bernoulli, Poisson (any subsample) or No, Newton or Gradient
leaves at any iteration count, any `random_strength`,
`boost_from_average=True`, with or without row weights, with or without an
eval set under the Iter detector and `use_best_model`, at any
`feature_fraction` (the device's per-tree sample); no class weights, no
categoricals. The binding refuses by name every other value of those options
on this arm.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build)

  1. `train` (`gbdt/train.mojo`): the grid and the binarize of the
     symmetric oracle, the eval rows binarized against the same borders,
     `use_best_model` resolved (unset is on with a non-constant eval
     target), the weights buffer.
  2. `fit_with_test` (`gbdt/methods/doc_parallel_boosting.mojo`):
     `calc_one_dimensional_optimum_const_approx` (Logloss's log-odds,
     RMSE's weighted mean), the learn and test cursors at its Float32,
     `create_bootstrap_seeds` (splitmix64), `noise_rand = TRandom(seed)`.
  3. Per tree: the loss's target kernel (`launch_approximate[False,
     second_order]`, `_loss_search_pass`): plane 0 the row weight, or the
     weighted second derivative under NewtonL2 / NewtonCosine, plane 1 the
     flushed weighted derivative, the score and magnitude partials; the
     tree's noise seed (one `noise_rand` draw per tree); under
     `random_strength` the score-noise deviation from the UNBOOTSTRAPPED
     planes (`compute_score_std_dev`, `ComputeWeakTarget`'s order) times
     `score_model_length_mult`; `bootstrap_kernel` of the bootstrap kind
     over both planes and its magnitudes, else the target kernel's;
     `choose_scale_kernel`; the snap of the gradient plane; the single-fold
     structure search (`find_optimal_split_single_fold_kernel` with
     `TL2ScoreCalcer` or `TCosineScoreCalcer`, the noise drawn per feature
     on the cosine arm); `compute_bins_for_model` and
     `partition_from_bins`; the shared leaf estimator at the loss's method
     (Newton or Gradient); `add_model_value_kernel`'s fused cursor update;
     the rescaled leaves; `compute_bins_and_add_kernel` onto the test
     cursor and the unweighted test loss; the detector.
  4. The final learn loss pass, `model.shrink` to the best iteration, and
     the model text with its `bias` record.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the Newton walker's Hessian
regularizer, so every leaf of every tree moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on those lanes is the measurement.
"""
from checks.numerics import portable_log64
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, HostU32Ptr, host_predict_chunk
from gbdt.data.permutation import TRandom
from gbdt.methods.cursor_rescale import rescaled_leaves
from gbdt.methods.random_score_bodies import score_model_length_mult
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.host.gbdt_oracle import (
    apply_scaled_leaves_host,
    GBDT_MSE_BLOCK,
    GbdtHostModel,
    _binarize_columns,
    _bootstrap_pass,
    _choose_scale_from_magnitudes,
    _deterministic_sum_lanes,
    _snap_plane,
    _partition_for_tree,
    _identity_rows,
    _gbdt_units,
    GbdtIntPtr,
    gbdt_bootstrap_seeds,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_ordered import (
    _OrdSplit,
    _ordered_tree_structure,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_GRADIENT,
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_LOGLOSS,
    GBDT_OBJ_RMSE,
    GbdtHostLoss,
    _estimate_leaves_for_loss,
    _loss_search_pass,
    _loss_value,
)
from gbdt.host.gbdt_oracle_pointwise_rank import (
    _pointwise_target_std_dev,
    _pw_feature_words,
    _pw_helpers,
    _pw_sample_tree,
    _split_planes,
)
from gbdt.host.gbdt_oracle_rmse import GbdtRmseHostFit, gbdt_rmse_host_model_text
from gbdt.metrics.sample_quantile import calculate_weighted_target_average
from gbdt.overfitting_detector.overfitting_detector import (
    OD_ITER,
    make_overfitting_detector,
)


@fieldwise_init
struct GbdtPointwiseFit(Movable):
    var text: String
    var losses: List[Float64]
    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool


# Lane gbdt-rest-3 (2026-11-23): `_pw_search_pass` (this file's target pass)
# and `_pw_bootstrap` (its Bayesian-only copy of `bootstrap_kernel`) are
# deleted with the constants `GBDT_PW_SEED_COUNT` / `GBDT_PW_BOOT_BLOCK` and
# the inline splitmix64 seed loop: the fit calls the loss oracle's target
# pass (`_loss_search_pass`, which now takes the second-order planes), the
# symmetric oracle's bootstrap (`_bootstrap_pass`, every kind) and its seeds
# (`gbdt_bootstrap_seeds`), as the grouped pointwise fit does.


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
    od_wait: Int,
    use_best_model: Int,
    best_model_min_trees: Int,
    has_weights: Bool = True,
    second_order: Bool = False,
    cosine_score: Bool = False,
    random_strength: Float32 = Float32(0.0),
    feature_fraction: Float64 = 1.0,
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
    RMSE's weighted mean (`_rmse_starting_approx`'s arm).

    Lane gbdt-rest-3 (2026-11-23): the rest of the device's pointwise arm
    on these two losses. `second_order` is NewtonL2 / NewtonCosine (plane 0
    the weighted second derivative, `secondDerAsWeights`), `cosine_score`
    Cosine / NewtonCosine (`TCosineScoreCalcer`, else `TL2ScoreCalcer`);
    the bootstrap is `loss.bootstrap_kind` at `loss.bootstrap_param` (-1:
    none, the target kernel's magnitudes then set the scale); the leaves are
    the loss's method (Newton or Gradient) at its iteration count;
    `random_strength` scales the per-tree noise of the cosine scorer, its
    deviation taken before the bootstrap. The seeds and the per-tree noise
    draw are the device fit's (`create_bootstrap_seeds`, `noise_rand`) and
    are drawn whatever the options, as there. `feature_fraction` below 1
    is the device's per-tree feature sample on this searcher
    (`_pw_sample_tree`, shared with the grouped pointwise fit)."""
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    var is_logloss = loss.objective == GBDT_OBJ_LOGLOSS
    if not is_logloss and loss.objective != GBDT_OBJ_RMSE:
        raise Error("the pointwise host fit takes Logloss or RMSE")
    if loss.method != GBDT_LEAF_NEWTON and loss.method != GBDT_LEAF_GRADIENT:
        raise Error("the pointwise host fit takes Newton or Gradient leaves")
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
    # every policy, BinaryFeatures included: `_ordered_tree_structure`
    # builds the binary cells for a one-border feature
    var helpers = _pw_helpers(layout, n_rows)
    var feat_offset = List[Int]()
    var feat_shift = List[UInt32]()
    var feat_mask = List[UInt32]()
    _pw_feature_words(layout, n_rows, n_features, feat_offset, feat_shift, feat_mask)
    # the per-tree feature sample (lane gbdt-rest-3): the searcher's words
    # for the tree's layout; the full layout's above address the held-out
    # rows and the partition
    var ff_on = feature_fraction < 1.0
    var feature_random = TRandom(random_seed ^ UInt64(0x4645415455524553))
    var tree_cindex = List[UInt32]()
    var s_offset = feat_offset.copy()
    var s_shift = feat_shift.copy()
    var s_mask = feat_mask.copy()

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

    # ---- `create_bootstrap_seeds`, the noise stream ----
    var bootstrap_on = loss.bootstrap_kind >= 0
    var seeds = gbdt_bootstrap_seeds(random_seed)
    var noise_rand = TRandom(random_seed)

    var detector = make_overfitting_detector(OD_ITER, False, Float64(-1.0), od_wait, True)
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    # the planes and the partials, reused by every tree (every word is
    # written again by the target pass)
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mse_blocks, fill=Float32(0.0))
    var sw = List[Float32](length=n_rows, fill=Float32(0.0))
    var sg = List[Float32](length=n_rows, fill=Float32(0.0))
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
        _loss_search_pass(
            loss, y, cursor, n_rows, stats, fv_part, mag_part, est_weights,
            second_order,
        )
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        # ---- the tree's noise: its seed, and its deviation from the
        # unbootstrapped planes (`ComputeWeakTarget`'s order) ----
        var tree_seed = noise_rand.next_uniform_l()
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            var mult = score_model_length_mult(
                Float64(n_rows), iteration, learning_rate
            )
            if mult * Float64(random_strength) != 0.0:
                score_std_dev = Float32(
                    mult * _pointwise_target_std_dev(stats, n_rows)
                    * Float64(random_strength)
                )
        # ---- `bootstrap_kernel` and `choose_scale_kernel` ----
        var scale: Float32
        if bootstrap_on:
            var bm = _bootstrap_pass(
                loss.bootstrap_kind, seeds, stats, n_rows, loss.bootstrap_param
            )
            scale = _choose_scale_from_magnitudes(bm[0], bm[1], n_rows)
        else:
            var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
            scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        _split_planes(stats, n_rows, sw, sg)
        # lane/sym-quality: `enqueue_snap_plane` (doc_parallel_boosting's
        # pointwise arm), the gradient plane onto the tree's grid
        _snap_plane(sg, n_rows, scale)

        # ---- the single-task pointwise structure search ----
        var splits: List[_OrdSplit]
        if ff_on:
            _pw_sample_tree(
                cindex, layout, grid.fold_counts, feature_fraction,
                feature_random, n_rows, n_features, tree_cindex, helpers,
                s_offset, s_shift, s_mask,
            )
            splits = _ordered_tree_structure(
                tree_cindex, helpers, max_depth, sw, sg, doc_ids,
                part_bounds, 1, 0, scale, l2_leaf_reg, s_offset, s_shift,
                s_mask, List[Bool](), not cosine_score, score_std_dev,
                tree_seed, cosine_score,
            )
        else:
            splits = _ordered_tree_structure(
                cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
                1, 0, scale, l2_leaf_reg, feat_offset, feat_shift, feat_mask,
                List[Bool](), not cosine_score, score_std_dev, tree_seed,
                cosine_score,
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
