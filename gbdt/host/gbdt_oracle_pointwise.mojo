# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the doc-parallel (pointwise)
searcher on every one-output loss the device's pointwise arm trains: the
gbdt-pointwise-l2-bayesian-eval lane (lane/cpu-training-gbdt-ordered,
2026-09-15), since lane gbdt-rest-3 (2026-11-23) every score function,
bootstrap, leaf method and `random_strength` on Logloss and RMSE, and since
lane gbdt-cpu-rest-4-pwl (2026-11-23) the rest of that arm: CrossEntropy,
MAE, Quantile, MAPE, LogLinQuantile, Poisson, Lq, Expectile, Tweedie and
Huber (Exact leaves where `train` resolves them), `boost_from_average`
resolved as `train` resolves it, every `feature_border_type`, every
overfitting detector, one-hot and CTR categorical columns (every
permutation). The grouped losses run `gbdt_oracle_pointwise_rank.mojo`.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the GPU-free host modules the
device fit itself runs on the host (`gbdt/gpu_util/kernel/random_gen.mojo`'s
generators through the bootstrap, `TRandom`, the overfitting detector,
`build_layout`, `random_score_bodies.score_model_length_mult`), the
symmetric oracle (the grid, the one-hot grid, the binarize, the bootstrap
pass, the scale, the partition, the learn permutation, the model text), the
loss oracle (the target pass, the loss value, the shared leaf estimator,
Exact included), the held-out arm (`gbdt_oracle_eval.mojo`), the CTR
prelude (`gbdt_oracle_ctr.mojo`), the grouped pointwise fit's noise
magnitude and feature sample, and the ordered oracle's single-fold
structure search.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-pointwise-l2-bayesian-eval`, `gbdt-pointwise-arms`,
`gbdt-pointwise-rest`): SymmetricTree, `use_pointwise_searcher=True`, a
one-output loss (Logloss, RMSE, CrossEntropy and the pointwise losses),
`score_function` L2, NewtonL2, Cosine or NewtonCosine, `bootstrap_type`
Bayesian, Bernoulli, Poisson or No, the loss's leaf methods at any
iteration count, any `random_strength`, `boost_from_average` on, off or
unset, sample and class weights, an eval set under any detector and
`use_best_model`, any `feature_fraction` on numeric columns, numeric,
one-hot and CTR columns. The binding refuses in `train`'s words what the
device refuses (Depthwise and Lossguide, feature combinations,
`boost_from_average` outside its list). MultiClass, MultiClassOneVsAll and
MultiRMSE on
this searcher run `gbdt_oracle_multiclass.gbdt_multi_host_fit(
use_pointwise_searcher=True)` (lane gbdt-pw-multiclass, 2026-11-23).

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build)

  1. `train` (`gbdt/train.mojo`): the grid (the fit's border type) and the
     one-hot grid and the binarize of the symmetric oracle, or the CTR
     prelude (the columns over each permutation, the tables, one index per
     permutation), the eval rows binarized against the same borders, the
     weights buffer.
  2. `fit_with_test` (`gbdt/methods/doc_parallel_boosting.mojo`): the
     starting constant the binding computed
     (`calc_one_dimensional_optimum_const_approx`), every cursor and the
     test cursor at its Float32, `create_bootstrap_seeds` (splitmix64),
     `noise_rand = TRandom(seed)`, the detector.
  3. Per tree: the learn permutation (`gbdt_learn_permutation`); the loss's
     target kernel on that permutation's cursor (`launch_approximate[False,
     second_order]`, `_loss_search_pass`): plane 0 the row weight, or the
     weighted second derivative under NewtonL2 / NewtonCosine, plane 1 the
     flushed weighted derivative, the score and magnitude partials; the
     tree's noise seed (one `noise_rand` draw per tree); under
     `random_strength` the score-noise deviation from the UNBOOTSTRAPPED
     planes (`compute_score_std_dev`, `ComputeWeakTarget`'s order) times
     `score_model_length_mult`; `bootstrap_kernel` of the bootstrap kind
     over both planes and its magnitudes, else the target kernel's;
     `choose_scale_kernel`; the snap of the gradient plane; the single-fold
     structure search on the learn permutation's index
     (`find_optimal_split_single_fold_kernel` with `TL2ScoreCalcer` or
     `TCosineScoreCalcer`, the noise drawn per feature on the cosine arm,
     one-hot columns as equality tests); then EVERY permutation estimates
     the structure on its own cursor (`compute_bins_for_model` and
     `partition_from_bins` over its index, the shared leaf estimator at the
     loss's method, `add_model_value_kernel`'s cursor update), the
     estimation permutation's rescaled leaves exported;
     `compute_bins_and_add_kernel` onto the test cursor and the unweighted
     test loss; the detector.
  4. The final learn loss pass on the estimation permutation's cursor,
     `ShrinkToBestIteration`, and the model text with its `bias` record
     (one-hot and CTR records where the columns have them).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the Newton walker's Hessian
regularizer, so every leaf of every tree moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on those lanes is the measurement.
"""
from gbdt.data.permutation import TRandom
from gbdt.methods.random_score_bodies import score_model_length_mult
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.host.gbdt_oracle import (
    rescale_apply_leaves_host,
    GBDT_MSE_BLOCK,
    GbdtHostGrid,
    GbdtHostModel,
    GbdtHostParams,
    _binarize_columns,
    _bootstrap_pass,
    _choose_scale_from_magnitudes,
    _deterministic_sum_lanes,
    _snap_plane,
    _partition_for_tree,
    _identity_rows,
    gbdt_bootstrap_seeds,
    gbdt_host_grid,
    gbdt_host_model_text,
    gbdt_host_one_hot_grid,
    gbdt_learn_permutation,
)
from gbdt.host.gbdt_oracle_ctr import (
    gbdt_ctr_column_kinds,
    gbdt_ctr_host_model_text_of,
    gbdt_ctr_host_prelude,
    gbdt_has_ctr_columns,
)
from gbdt.host.gbdt_oracle_eval import (
    GbdtHostEval,
    gbdt_eval_add_tree,
    gbdt_eval_detector,
    gbdt_eval_shrink_point,
    gbdt_eval_test_bins,
)
from gbdt.host.gbdt_oracle_onehot import (
    gbdt_host_model_text_one_hot,
    gbdt_resolve_one_hot,
)
from gbdt.host.gbdt_oracle_ordered import (
    _OrdSplit,
    _ordered_tree_structure,
)
from gbdt.targets.ranking_options import uses_grouped_rank_workspace
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_EXACT,
    GBDT_LEAF_GRADIENT,
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_PAIR_LOGIT,
    GBDT_OBJ_QUERY_RMSE,
    GBDT_OBJ_YETI_RANK,
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
)
from gbdt.host.gbdt_oracle_rmse import gbdt_text_with_bias
from gbdt.host.gbdt_oracle_tree_ctr import GbdtTreeCtrHost
from gbdt.models.combination_map import TCombinationCtrs
from gbdt.models.ctr_value_table import TCtrValueTable
from core.fit_loop_clock import FitLoop, fit_loop_start, fit_loop_stop


@fieldwise_init
struct GbdtPointwiseFit(Movable):
    var text: String
    var losses: List[Float64]
    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool
    #: the boosting loop's executed rounds and its time: the estimator's
    #: `fit_loop_` (core/fit_loop_clock.mojo; lane ab-loop-mode)
    var fit_loop: FitLoop


# Lane gbdt-rest-3 (2026-11-23): `_pw_search_pass` (this file's target pass)
# and `_pw_bootstrap` (its Bayesian-only copy of `bootstrap_kernel`) are
# deleted with the constants `GBDT_PW_SEED_COUNT` / `GBDT_PW_BOOT_BLOCK` and
# the inline splitmix64 seed loop: the fit calls the loss oracle's target
# pass (`_loss_search_pass`, which now takes the second-order planes), the
# symmetric oracle's bootstrap (`_bootstrap_pass`, every kind) and its seeds
# (`gbdt_bootstrap_seeds`), as the grouped pointwise fit does.


# Lane gbdt-cpu-rest-4-pwl (2026-11-23): `_pw_tree_onto_rows`, this file's
# copy of `compute_bins_kernel` + `add_bin_model_value_kernel` for the test
# cursor (TakeGreater only), is deleted: the held-out apply is the shared
# arm's (`gbdt_eval_test_bins`, which also takes a one-hot column's equality
# test, then `gbdt_eval_add_tree`), the same per-row compares and one add of
# the stored value, as the greedy loop applies it.


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
    weights: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    loss: GbdtHostLoss,
    eval: GbdtHostEval,
    starting_approx: Float64,
    flags: List[UInt32] = List[UInt32](),
    permutation_count: Int = -1,
    ctr_estimation_permutation_id: Int = -1,
    second_order: Bool = False,
    cosine_score: Bool = False,
    random_strength: Float32 = Float32(0.0),
    feature_fraction: Float64 = 1.0,
    max_ctr_complexity: Int = 1,
) raises -> GbdtPointwiseFit:
    """`train` then `fit_with_test` with `use_pointwise_searcher=True` on a
    one-output loss (see the module docstring).

    `weights` (row order, empty: unit) are the fit's one weight column, the
    sample weights times the class weights as `train` folds them (the
    binding builds it); empty is the device's unweighted arm, whose
    starting constant's `summaryWeight` is the row count. `loss` is the loss
    as `train` resolves it (the binding resolves it): its objective, the
    target kernel's alpha (Logloss's border), the estimator's alpha, the
    leaf method (Newton, Gradient, or Exact for MAE, MAPE and Quantile) at
    its iteration count, the bootstrap kind and parameter. `params` carries
    the grid options (border type included), the tree options and the
    learning rate. `starting_approx` is the binding's
    `calc_one_dimensional_optimum_const_approx` when `boost_from_average`
    resolves True, else 0: every cursor and the test cursor start at its
    Float32 and the model records it as its `bias`. `eval` is the held-out
    arm (`eval.n_rows == 0`: none) with the detector's resolved options and
    `use_best_model` resolved.

    `flags` (empty: numeric) are the categorical flags as `gbdt_fit`
    decodes them (bit 0 categorical, bit 1 one-hot). With a CTR column the
    columns, tables and per-permutation indexes are the CTR prelude's
    (`gbdt_ctr_host_prelude`, `permutation_count` and
    `ctr_estimation_permutation_id` resolved there, no feature
    combinations); otherwise the one-hot columns take the one-hot grid
    (`gbdt_host_one_hot_grid`) and the fit runs one permutation.

    `second_order` is NewtonL2 / NewtonCosine (plane 0 the weighted second
    derivative, `secondDerAsWeights`), `cosine_score` Cosine / NewtonCosine
    (`TCosineScoreCalcer`, else `TL2ScoreCalcer`); `random_strength` scales
    the per-tree noise of the cosine scorer, its deviation taken before the
    bootstrap. The seeds and the per-tree noise draw are the device fit's
    (`create_bootstrap_seeds`, `noise_rand`) and are drawn whatever the
    options, as there. `feature_fraction` below 1 is the device's per-tree
    feature sample on this searcher (`_pw_sample_tree`, shared with the
    grouped pointwise fit), numeric columns only (the binding refuses the
    rest in `train`'s words). `max_ctr_complexity` above 1 with a CTR
    column is refused in `fit_with_test`'s words: this searcher mints no
    feature combinations."""
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    if (
        loss.objective == GBDT_OBJ_QUERY_RMSE
        or loss.objective == GBDT_OBJ_PAIR_LOGIT
        or loss.objective == GBDT_OBJ_YETI_RANK
        or uses_grouped_rank_workspace(loss.objective)
    ):
        raise Error(
            "the pointwise host fit takes the one-output losses; the grouped"
            " ones run gbdt_ranking_pointwise_host_fit"
        )
    if (
        loss.method != GBDT_LEAF_NEWTON and loss.method != GBDT_LEAF_GRADIENT
        and loss.method != GBDT_LEAF_EXACT
    ):
        raise Error("the pointwise host fit takes Newton, Gradient or Exact leaves")
    if n_rows < 1 or n_features < 1:
        raise Error("the pointwise fit needs rows and features")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if len(weights) != 0 and len(weights) != n_rows:
        raise Error("weights size mismatch")
    if len(flags) != 0 and len(flags) != n_features:
        raise Error("cat flags must be empty or one per feature")
    if params.max_depth < 0:
        raise Error("max_depth must not be negative")
    var has_test = eval.has_test()
    var n_eval = eval.n_rows
    if has_test:
        if len(eval.x_colmajor) != n_eval * n_features:
            raise Error("eval_set x_colmajor size mismatch")
        if len(eval.y) != n_eval:
            raise Error("eval_set y size mismatch")

    # ---- the columns: the grid, the binarize, the eval rows ----
    # (`train`: numeric and one-hot columns over one index; CTR columns
    # over one index per permutation, the prelude's)
    var grid = GbdtHostGrid(List[List[Float32]](), List[Int](), List[Int]())
    var cindexes = List[List[UInt32]]()
    var test_cindex = List[UInt32]()
    var one_hot = List[Bool]()
    var tables = List[TCtrValueTable]()
    var ctr_column_count = 0
    var est_p = 0
    var kind = 0  # 0 numeric, 1 one-hot, 2 CTR
    var layout: CompressedIndexLayout
    if len(flags) != 0 and gbdt_has_ctr_columns(
        gbdt_ctr_column_kinds(flags, x_colmajor, n_rows, n_features)
    ):
        kind = 2
        if max_ctr_complexity > 1:
            # `fit_with_test`'s words (gbdt/methods/doc_parallel_boosting.mojo)
            raise Error(
                "max_ctr_complexity above 1 (feature combinations) is grown"
                " by the greedy searcher; use_pointwise_searcher does not"
                " mint combinations"
            )
        var no_tree_ctr = List[GbdtTreeCtrHost]()
        var shape = gbdt_ctr_host_prelude(
            x_colmajor, y, n_rows, n_features, params, flags,
            permutation_count, ctr_estimation_permutation_id, 1,
            eval.x_colmajor, n_eval, 1, grid, cindexes, test_cindex, one_hot,
            tables, no_tree_ctr,
        )
        est_p = shape.est_perm
        ctr_column_count = shape.ctr_column_count
        layout = shape.layout.copy()
    else:
        grid = gbdt_host_grid(
            x_colmajor, n_rows, n_features, params.border_count,
            params.border_build_max_samples, params.random_seed,
            params.nan_mode, params.border_type,
        )
        # one permutation without a CTR column (`train`'s Plain collapse),
        # the estimation permutation checked against it in `train`'s words
        var one_est = ctr_estimation_permutation_id
        if one_est == -1:
            one_est = 0
        if one_est != 0:
            raise Error(
                "ctr_estimation_permutation_id " + String(one_est)
                + " is outside the 1 permutations this fit builds"
            )
        var one_hot_in = List[Bool]()
        if len(flags) != 0:
            kind = 1
            one_hot_in = gbdt_resolve_one_hot(flags, x_colmajor, n_rows, n_features)
        one_hot = gbdt_host_one_hot_grid(
            grid, x_colmajor, n_rows, n_features, one_hot_in
        )
        layout = build_layout(grid.fold_counts, one_hot)
        cindexes.append(_binarize_columns(x_colmajor, n_rows, n_features, grid, layout))
        if has_test:
            test_cindex = _binarize_columns(
                eval.x_colmajor, n_eval, n_features, grid, layout
            )
    var perm_count = len(cindexes)
    # the layout's columns (the base columns, then the CTR columns)
    var n_columns = len(grid.fold_counts)
    # every policy, BinaryFeatures included: `_ordered_tree_structure`
    # builds the binary cells for a one-border feature
    var helpers = _pw_helpers(layout, n_rows)
    var feat_offset = List[Int]()
    var feat_shift = List[UInt32]()
    var feat_mask = List[UInt32]()
    _pw_feature_words(layout, n_rows, n_columns, feat_offset, feat_shift, feat_mask)
    # the held-out apply's words (`gbdt_eval_test_bins`) and the searcher's
    # one-hot flags (empty: no one-hot column, the threshold arm only)
    var eval_offset = List[UInt32](length=n_columns, fill=UInt32(0))
    var eval_mask = List[UInt32](length=n_columns, fill=UInt32(0))
    var eval_shift = List[UInt32](length=n_columns, fill=UInt32(0))
    var eval_one_hot = List[Bool](length=n_columns, fill=False)
    var any_one_hot = False
    for f in range(n_columns):  # driver-loop: over the layout's columns
        eval_offset[f] = layout.features[f].offset
        eval_mask[f] = layout.features[f].mask
        eval_shift[f] = layout.features[f].shift
        eval_one_hot[f] = layout.features[f].one_hot_feature
        if eval_one_hot[f]:
            any_one_hot = True
    var search_one_hot = List[Bool]()
    if any_one_hot:
        search_one_hot = eval_one_hot.copy()
    # the per-tree feature sample (lane gbdt-rest-3): the searcher's words
    # for the tree's layout; the full layout's above address the held-out
    # rows and the partition. Numeric columns only (one permutation).
    var ff_on = feature_fraction < 1.0
    if ff_on and kind != 0:
        raise Error("feature_fraction<1 supports numeric features only; categorical/CTR input refused")
    var feature_random = TRandom(params.random_seed ^ UInt64(0x4645415455524553))
    var tree_cindex = List[UInt32]()
    var s_offset = feat_offset.copy()
    var s_shift = feat_shift.copy()
    var s_mask = feat_mask.copy()

    # ---- every cursor at the starting constant's Float32 ----
    var start_value = Float32(starting_approx)
    var cursors = List[List[Float32]]()
    for _ in range(perm_count):  # driver-loop: over the permutations (one cursor each)
        cursors.append(List[Float32](length=n_rows, fill=start_value))
    var test_cursor = List[Float32](length=n_eval, fill=start_value)

    # ---- `create_bootstrap_seeds`, the noise stream, the detector ----
    var bootstrap_on = loss.bootstrap_kind >= 0
    var seeds = gbdt_bootstrap_seeds(params.random_seed)
    var noise_rand = TRandom(params.random_seed)
    var detector = gbdt_eval_detector(eval)
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    # the planes and the partials, reused by every tree (every word is
    # written again by the target pass)
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mse_blocks, fill=Float32(0.0))
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
    var max_depth = params.max_depth
    var learning_rate = params.learning_rate
    var l2_leaf_reg = params.l2_leaf_reg

    # `fit_loop_` (lane ab-loop-mode): the host clock read once before the
    # first round and once after the last (nothing inside the loop); the
    # rounds executed are the trees the loop added.
    var fit_loop_t0 = fit_loop_start()
    for iteration in range(params.n_estimators):
        # ---- which permutation the STRUCTURE is searched on ----
        var learn_p = gbdt_learn_permutation(
            iteration, params.random_seed, perm_count, est_p
        )
        # ---- the search planes and the learn score ----
        _loss_search_pass(
            loss, y, cursors[learn_p], n_rows, stats, fv_part, mag_part,
            weights, second_order,
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
        # lane/sym-quality: `enqueue_snap_plane` (doc_parallel_boosting's
        # pointwise arm), the gradient plane onto the tree's grid, in place
        # in `stats` (lane gbdt-one-body: the search reads both planes from
        # `stats` at offsets 0 and n_rows, no per-tree copy)
        _snap_plane(stats, n_rows, scale, n_rows)

        # ---- the single-task pointwise structure search, on the learn
        # permutation's index ----
        var splits: List[_OrdSplit]
        if ff_on:
            _pw_sample_tree(
                cindexes[0], layout, grid.fold_counts, feature_fraction,
                feature_random, n_rows, n_features, tree_cindex, helpers,
                s_offset, s_shift, s_mask,
            )
            splits = _ordered_tree_structure(
                tree_cindex, helpers, max_depth, stats, stats, doc_ids,
                part_bounds, 1, 0, scale, l2_leaf_reg, s_offset, s_shift,
                s_mask, List[Bool](), not cosine_score, score_std_dev,
                tree_seed, cosine_score, sw_off=0, sg_off=n_rows,
            )
        else:
            splits = _ordered_tree_structure(
                cindexes[learn_p], helpers, max_depth, stats, stats, doc_ids,
                part_bounds, 1, 0, scale, l2_leaf_reg, feat_offset,
                feat_shift, feat_mask, search_one_hot, not cosine_score,
                score_std_dev, tree_seed, cosine_score, sw_off=0,
                sg_off=n_rows,
            )
        var n_leaves = 1 << len(splits)
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        for level in range(len(splits)):  # driver-loop: over the tree's levels
            tree_features.append(splits[level].feature)
            tree_bins.append(splits[level].bin)
        # ---- every permutation estimates the structure on its own cursor
        # (`doc_parallel_boosting.mojo`'s estimation loop): its own
        # `compute_bins_for_model` + `partition_from_bins` (the stable
        # counting sort, rows ascending within a leaf, across the cores),
        # the shared estimator (unweighted: no weights, see the note above)
        # and `add_model_value_kernel`; the estimation permutation's leaves
        # are the model's ----
        var est_scaled = List[Float32]()
        for p in range(perm_count):  # driver-loop: over the permutations
            var partition = _partition_for_tree(
                cindexes[p], layout, tree_features, tree_bins, n_rows
            )
            var estimated = _estimate_leaves_for_loss(
                loss, y, cursors[p], partition.row_index, partition.offsets,
                partition.sizes, n_rows, l2_leaf_reg,
                weights=weights,
            )
            var scaled = rescale_apply_leaves_host(
                cursors[p], estimated, learning_rate, partition.row_index,
                partition.offsets, partition.sizes, n_leaves,
            )
            if p == est_p:
                est_scaled = scaled.copy()
        for level in range(len(splits)):  # driver-loop: over the tree's levels
            split_features.append(tree_features[level])
            split_bins.append(tree_bins[level])
        tree_split_offsets.append(len(split_features))
        for leaf in range(n_leaves):  # driver-loop: over the tree's leaves
            model_leaves.append(est_scaled[leaf])
        tree_leaf_offsets.append(len(model_leaves))
        # ---- the test cursor, the test loss, the detector ----
        if has_test:
            # a depth-0 tree adds nothing, as their apply returns early
            if len(splits) > 0:
                var test_bins = gbdt_eval_test_bins(
                    test_cindex, n_eval, tree_features, tree_bins,
                    eval_offset, eval_mask, eval_shift, eval_one_hot,
                )
                gbdt_eval_add_tree(test_cursor, test_bins, est_scaled)
            # the held-out rows at unit weight, through the learn loss's
            # target kernel (`_test_loss`)
            var t_loss = -Float64(_loss_value(loss, eval.y, test_cursor, n_eval)) / Float64(n_eval)
            test_losses.append(t_loss)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                # their break is BEFORE the learn loss append
                stopped_early = True
                break
        if len(losses) < params.n_estimators:
            if len(tree_split_offsets) - 1 > 1:
                losses.append(-Float64(fv) / Float64(n_rows))
    var fit_loop = fit_loop_stop(fit_loop_t0, len(tree_split_offsets) - 1)

    # the final learn loss, on the ESTIMATION permutation's cursor
    losses.append(
        -Float64(_loss_value(loss, y, cursors[est_p], n_rows, weights))
        / Float64(n_rows)
    )

    # ---- `ShrinkToBestIteration`: `model.shrink(best_iter)` ----
    var shrink_to = gbdt_eval_shrink_point(
        test_losses, eval.want_best_model, eval.best_model_min_trees
    )
    var n_trees = len(tree_split_offsets) - 1
    if 0 < shrink_to and shrink_to < n_trees:
        while len(tree_split_offsets) - 1 > shrink_to:
            _ = tree_split_offsets.pop()
            _ = tree_leaf_offsets.pop()
        split_features.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
        split_bins.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
        model_leaves.resize(tree_leaf_offsets[len(tree_leaf_offsets) - 1], Float32(0.0))

    var n_losses = len(losses)
    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses.copy(), detector.best_iteration, stopped_early,
    )
    var base: String
    if kind == 2:
        base = gbdt_ctr_host_model_text_of(
            model, one_hot, tables, ctr_column_count, TCombinationCtrs()
        )
    elif kind == 1:
        base = gbdt_host_model_text_one_hot(model, one_hot)
    else:
        base = gbdt_host_model_text(model)
    var text = gbdt_text_with_bias(base, n_losses, starting_approx)
    _ = len(cindexes)
    _ = len(test_cindex)
    return GbdtPointwiseFit(text^, losses^, test_losses^, detector.best_iteration, stopped_early, fit_loop^)
