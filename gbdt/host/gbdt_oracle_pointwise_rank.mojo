# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Grouped objectives with the host doc-parallel symmetric-tree searcher."""

# CPU binding compilation passed on Apple M1 at O0 (2026-10-28).
# This includes the pointwise owner and actual gbdt_fit binding. Runtime,
# identity and negative-control execution remain pending; verifier lane
# registration is not evidence of passing execution.
# See docs/GBDT_CPU_RANKING_DRAFT_20261027.md for exact scope and pending work.
#
# PairLogit/YetiRank derivatives and grouped leaf estimation already exist in
# the CPU greedy owner. This file supplies the distinct pointwise orchestration:
# Newton-vs-gradient search planes, noise BEFORE bootstrap using row count,
# single-fold scoring, stable bin partition, and the shared grouped estimator.
# It never substitutes the greedy searcher or calls Python/native GPU kernels.

from gbdt.methods.random_score_bodies import std_dev_of_sums, std_dev_row_term
from std.math import exp, fma, isfinite, log, sqrt

from checks.numerics import ftz, identical_mul_add
from gbdt.methods.random_score_bodies import score_model_length_mult
from gbdt.data.pairs import (
    IDN_PAIRLOGIT_GROUP,
    PairList,
    generate_pairs,
    order_pairs_by_winner,
)
from gbdt.data.permutation import TRandom
from gbdt.data.yeti_rank_tasks import YetiRankTasks, yeti_rank_tasks
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout, build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from core.host_parallel import host_parallelize
from core.fit_loop_clock import fit_loop_start, fit_loop_stop
from core.host_predict_threads import HostF32Ptr, host_predict_chunk
from gbdt.host.gbdt_oracle import (
    _ns_project_index,
    _sample_tree_folds,
    rescale_apply_leaves_host,
    _gbdt_units,
    _identity_rows,
    _partition_for_tree,
    GBDT_MSE_BLOCK,
    GbdtHostModel, GbdtHostFitWithEval, GbdtHostGrid,
    GbdtHostParams,
    _binarize_columns,
    _bootstrap_pass,
    _choose_scale_from_magnitudes,
    _deterministic_sum_lanes,
    _halving_fold,
    _snap_plane,
    gbdt_bootstrap_seeds,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_GRADIENT,
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_PAIR_LOGIT,
    GBDT_OBJ_QUERY_RMSE,
    GBDT_OBJ_YETI_RANK,
    GbdtHostLoss,
    _estimate_leaves_for_loss,
    _zero_average_host, GroupedTestLoss,
)
from gbdt.host.gbdt_oracle_eval import (
    GbdtHostEval, GbdtHostEvalFit, gbdt_eval_add_tree, gbdt_eval_detector,
    gbdt_eval_test_bins, gbdt_eval_shrink_symmetric,
)
from gbdt.host.gbdt_oracle_ctr import GbdtCtrHostFit, gbdt_ctr_host_prelude
from gbdt.host.gbdt_oracle_tree_ctr import GbdtTreeCtrHost
from gbdt.models.combination_map import TCombinationCtrs
from gbdt.models.ctr_value_table import TCtrValueTable
from gbdt.host.gbdt_oracle_ordered import _OrdSplit, _PwHelper, _ordered_tree_structure
from gbdt.host.gbdt_oracle_pair import (
    HostPairs,
    host_pair_groups,
    host_pairs,
    pair_logit_search_pass,
    pair_logit_value,
)
from gbdt.host.gbdt_oracle_query import query_rmse_search_pass, query_rmse_value
from gbdt.host.ranking_fit import HostRankingFit
from gbdt.targets.ranking_options import is_lightgbm_ranking, ranking_seed
from gbdt.host.gbdt_oracle_yeti import yeti_rank_search_pass


def _pointwise_target_std_dev(
    stats: List[Float32], n_rows: Int, object_count: Int = -1
) -> Float64:
    """Compute the doc-parallel score-noise deviation before bootstrap.

    `stats` holds `n_rows` weights then `n_rows` weighted targets. A
    multi-output target (lane gbdt-pw-multiclass, 2026-11-23) passes its
    `_dims_planes` with `n_rows = dims * rows` and `object_count = rows`:
    the device's `compute_score_std_dev_planes`, which divides by the
    object count; -1 divides by `n_rows`."""
    # random_score_helper.mojo pins 4 * 32 blocks of 512 threads in identical
    # mode. This is a reduction topology, not a hardware-utilization cap.
    # Its row contribution is w * (g/w)^2 without a zero-weight guard; the
    # greedy helper's weight denominator and guard are different algorithms.
    comptime BLOCK = 512
    var blocks = min(128, max(1, (n_rows + BLOCK - 1) // BLOCK))
    var partials = List[Float32](length=blocks, fill=Float32(0))
    # THE SPLIT (lane gbdt-host-par): the unit is one 512-thread device
    # block: each thread's strided `ftz` chain ascending in its rows and the
    # block's halving tree. Blocks go round robin to `_gbdt_units` tasks; a
    # task holds one slab, every lane rewritten per block.
    var stride = blocks * BLOCK
    var units = _gbdt_units(blocks, 8 * n_rows)
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var part_p = rebind[HostF32Ptr](partials.unsafe_ptr())

    def block_sums(unit: Int) {imm units, imm blocks, imm stride, imm n_rows, imm stats_p, imm part_p}:
        var slab = List[Float32](length=BLOCK, fill=Float32(0))
        var b = unit
        while b < blocks:
            for t in range(BLOCK):
                var r = b * BLOCK + t
                var acc = Float32(0)
                while r < n_rows:
                    # the device kernel's row body (random_score_bodies)
                    acc = std_dev_row_term(stats_p[r], stats_p[n_rows + r], acc)
                    r += stride
                slab[t] = acc
            part_p[b] = _halving_fold(slab)
            b += units

    host_parallelize(block_sums, units)
    _ = len(stats)
    var sum2 = _deterministic_sum_lanes(partials, 1, blocks)[0]
    var objects = n_rows if object_count < 0 else object_count
    return std_dev_of_sums(Float64(sum2), Float64(objects))


def _pw_helpers(
    layout: CompressedIndexLayout, n_rows: Int
) raises -> List[_PwHelper]:
    """The single-fold searcher's policy helpers of `layout` (the device's
    `pw_pool`, one per policy block, BinaryFeatures included: the searcher
    builds the binary cells for a one-border feature). A fit at
    feature_fraction 1 builds them once; a sampled fit builds them for each
    tree's layout, as the device clears `pw_pool` per sampled tree."""
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):  # driver-loop: over the layout's policy blocks
        ref blk = blocks[b]
        var gids = List[Int]()
        var offs = List[Int]()
        var firsts = List[Int]()
        var folds = List[Int]()
        var hist_line = 0
        for k in range(blk.count()):  # driver-loop: over the block's features
            var f = blk.feature_ids[k]
            gids.append(f)
            offs.append(Int(layout.features[f].offset) * n_rows)
            firsts.append(Int(blk.fold_offset[k]))
            folds.append(Int(blk.folds[k]))
            hist_line += Int(blk.folds[k])
        helpers.append(_PwHelper(
            blk.policy, gids^, offs^, firsts^, folds^, hist_line,
            List[Float32](), List[Int](), List[Bool](),
        ))
    return helpers^


def _pw_feature_words(
    layout: CompressedIndexLayout,
    n_rows: Int,
    n_features: Int,
    mut feat_offset: List[Int],
    mut feat_shift: List[UInt32],
    mut feat_mask: List[UInt32],
):
    """Each feature's word offset, shift and mask in `layout` (the
    partition update's addressing)."""
    feat_offset = List[Int](length=n_features, fill=0)
    feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    for f in range(n_features):  # driver-loop: over the layout's features
        feat_offset[f] = Int(layout.features[f].offset) * n_rows
        feat_shift[f] = layout.features[f].shift
        feat_mask[f] = layout.features[f].mask


def _pw_sample_tree(
    cindex: List[UInt32],
    layout: CompressedIndexLayout,
    fold_counts: List[Int],
    fraction: Float64,
    mut random: TRandom,
    n_rows: Int,
    n_features: Int,
    mut tree_cindex: List[UInt32],
    mut helpers: List[_PwHelper],
    mut feat_offset: List[Int],
    mut feat_shift: List[UInt32],
    mut feat_mask: List[UInt32],
) raises:
    """One tree's feature sample for the single-fold searcher
    (`feature_fraction < 1`, `doc_parallel_boosting.mojo`): the device's
    `sample_tree_folds` on its own stream, the tree layout, the learn index
    projected onto it (`_ns_project_index`, the helpers the symmetric
    losses fit calls), and the searcher's helpers and feature words rebuilt
    for it, as the device clears `pw_pool` for a sampled tree. The splits
    the searcher returns name the fit's features and bins, so the
    partition, the estimate and the held-out apply read the full layout."""
    var tree_folds = _sample_tree_folds(fold_counts, fraction, random)
    var one_hot = List[Bool](length=n_features, fill=False)
    var t_layout = build_layout(tree_folds, one_hot)
    tree_cindex = _ns_project_index(
        cindex, layout, t_layout, tree_folds, n_rows, n_features
    )
    helpers = _pw_helpers(t_layout, n_rows)
    _pw_feature_words(
        t_layout, n_rows, n_features, feat_offset, feat_shift, feat_mask
    )


def gbdt_ranking_pointwise_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    loss: GbdtHostLoss,
    group_sizes: List[Int],
    pair_winners: List[UInt32],
    pair_losers: List[UInt32],
    pair_weights: List[Float32],
    second_order: Bool,
    cosine_score: Bool,
    random_strength: Float32,
    weights: List[Float32] = List[Float32](),
    feature_fraction: Float64 = 1.0,
    eval: GbdtHostEval = GbdtHostEval.none(),
    eval_group_sizes: List[Int] = List[Int](),
    flags: List[UInt32] = List[UInt32](),
    permutation_count: Int = -1, ctr_estimation_permutation_id: Int = -1,
    max_ctr_complexity: Int = 1,
) raises -> GbdtCtrHostFit:
    """Fit grouped targets on numeric columns using pointwise tree search.

    `weights` (row order, empty: unit) are the sample weights, the device's
    one `weights` column (lane gbdt-cpu-rest, 2026-11-23), read exactly as
    the greedy grouped fit (`gbdt_oracle_losses.mojo::
    gbdt_losses_host_fit`) reads them: YetiRank's `has_weights` arm,
    PairLogit's generated pairs' weights (a caller's pairs keep their own),
    QueryRMSE's weighted query means, and the leaf weights of the shared
    grouped estimator.

    `feature_fraction` below 1 is the device's per-tree feature sample on
    this searcher (`_pw_sample_tree`; lane gbdt-rest-3, 2026-11-23)."""
    if not (feature_fraction > 0.0) or feature_fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    var has_weights = len(weights) > 0
    if has_weights and len(weights) != n_rows:
        raise Error("weights size mismatch")
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features or len(y) != n_rows:
        raise Error("pointwise ranking input dimensions do not match")
    # (lane gbdt-cpu-rest-4-tail, 2026-11-23: the depth cap is `train`'s,
    # `check_tree_limits`, before this fit; a negative depth is the
    # searcher's `create_subsets` refusal)
    if params.max_depth < 0:
        raise Error(
            String("maxDepth must be non-negative, got ") + String(params.max_depth)
        )
    if loss.objective != GBDT_OBJ_PAIR_LOGIT and loss.objective != GBDT_OBJ_YETI_RANK and loss.objective != GBDT_OBJ_QUERY_RMSE and not is_lightgbm_ranking(loss.objective):
        raise Error("pointwise ranking needs a grouped objective")
    if loss.method != GBDT_LEAF_NEWTON and loss.method != GBDT_LEAF_GRADIENT:
        raise Error("pointwise grouped targets require Newton or Gradient leaves")
    if loss.objective == GBDT_OBJ_YETI_RANK and loss.method != GBDT_LEAF_NEWTON:
        raise Error("YetiRank leaf estimation method cannot be changed")
    var grouped_rows = 0
    for g in range(len(group_sizes)):
        if group_sizes[g] < 1:
            raise Error("pointwise ranking query sizes must be positive")
        grouped_rows += group_sizes[g]
    if grouped_rows != n_rows:
        raise Error("pointwise ranking queries must cover every row")
    # KEPT SERIAL: admission, once per fit, a compare per row and the
    # refusal at the first bad row
    if not is_lightgbm_ranking(loss.objective):
        for r in range(n_rows):
            if not isfinite(y[r]):
                raise Error("pointwise ranking targets must be finite")

    var sizes_u32 = List[UInt32](capacity=len(group_sizes))
    for g in range(len(group_sizes)):
        sizes_u32.append(UInt32(group_sizes[g]))
    var pairs = Optional[HostPairs]()
    var loss_norm = Float64(n_rows)
    if loss.objective == GBDT_OBJ_PAIR_LOGIT:
        if IDN_PAIRLOGIT_GROUP and len(pair_winners) == 0:
            var hp = host_pair_groups(sizes_u32, y, weights, n_rows)
            loss_norm = hp.total_weight()
            pairs = Optional(hp^)
        else:
            var ordered: PairList
            if len(pair_winners) > 0:
                ordered = order_pairs_by_winner(
                    PairList(pair_winners.copy(), pair_losers.copy(), pair_weights.copy()),
                    sizes_u32, n_rows,
                )
            else:
                ordered = order_pairs_by_winner(
                    generate_pairs(sizes_u32, y, weights), sizes_u32, n_rows
                )
            var hp = host_pairs(ordered.winners, ordered.losers, ordered.weights, n_rows)
            loss_norm = hp.total_weight()
            pairs = Optional(hp^)
    var rank = Optional[HostRankingFit]()
    if is_lightgbm_ranking(loss.objective):
        rank = Optional(HostRankingFit(group_sizes, y))
    var yeti_tasks = Optional[YetiRankTasks]()
    var yeti_rand = TRandom(params.random_seed ^ UInt64(0x5945544952414E4B))
    if loss.objective == GBDT_OBJ_YETI_RANK:
        yeti_tasks = Optional(yeti_rank_tasks(sizes_u32, n_rows))

    if len(eval.x_colmajor) != eval.n_rows * n_features or len(eval.y) != eval.n_rows:
        raise Error("eval_set size mismatch")
    var grid = GbdtHostGrid(List[List[Float32]](), List[Int](), List[Int]())
    var one_hot = List[Bool]()
    var cindex = List[UInt32]()
    var test_cindex = List[UInt32]()
    var tables = List[TCtrValueTable]()
    var tree_ctr = List[GbdtTreeCtrHost]()
    var layout: CompressedIndexLayout
    var n_cols = n_features
    var ctr_column_count = 0
    if len(flags) != 0:
        if feature_fraction < 1:
            raise Error("feature_fraction<1 supports numeric features only; categorical/CTR input refused")
        var indexes = List[List[UInt32]]()
        var shape = gbdt_ctr_host_prelude(
            x_colmajor, y, n_rows, n_features, params, flags, permutation_count,
            ctr_estimation_permutation_id, 1, eval.x_colmajor, eval.n_rows, max_ctr_complexity,
            grid, indexes, test_cindex, one_hot, tables, tree_ctr, query_sizes=group_sizes,
        )
        layout = shape.layout.copy()
        n_cols = shape.n_base_columns
        ctr_column_count = shape.ctr_column_count
        cindex = indexes.pop()
    else:
        grid = gbdt_host_grid(x_colmajor, n_rows, n_features, params.border_count,
                              params.border_build_max_samples, params.random_seed, params.nan_mode, params.border_type)
        one_hot = List[Bool](length=n_features, fill=False)
        layout = build_layout(grid.fold_counts, one_hot)
        cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
        if eval.n_rows > 0:
            test_cindex = _binarize_columns(eval.x_colmajor, eval.n_rows, n_features, grid, layout)
    var rank_test = Optional[GroupedTestLoss]()
    var search_hot = List[Bool]()
    if len(flags) != 0:
        search_hot = one_hot.copy()
    var test_offsets = List[UInt32]()
    var test_masks = List[UInt32]()
    var test_shifts = List[UInt32]()
    var test_cursor = List[Float32](length=eval.n_rows, fill=0.0)
    var detector = gbdt_eval_detector(eval)
    var test_losses = List[Float64]()
    var stopped_early = False
    if eval.n_rows > 0:
        rank_test = Optional(GroupedTestLoss(loss.objective, eval_group_sizes, eval.y))
        for f in range(n_cols):
            test_offsets.append(layout.features[f].offset)
            test_masks.append(layout.features[f].mask)
            test_shifts.append(layout.features[f].shift)
    var helpers = _pw_helpers(layout, n_rows)
    var feat_offset = List[Int]()
    var feat_shift = List[UInt32]()
    var feat_mask = List[UInt32]()
    _pw_feature_words(layout, n_rows, n_cols, feat_offset, feat_shift, feat_mask)
    # the per-tree feature sample's state (unread at feature_fraction 1)
    var ff_on = feature_fraction < 1.0
    var feature_random = TRandom(params.random_seed ^ UInt64(0x4645415455524553))
    var tree_cindex = List[UInt32]()
    var doc_ids = List[Int](length=n_rows, fill=0)
    _identity_rows(doc_ids, n_rows)
    var part_bounds = List[Int]()
    part_bounds.append(0)
    part_bounds.append(n_rows)

    var cursor = List[Float32](length=n_rows, fill=Float32(0))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0))
    var row_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_blocks = pairs.value().blocks() if pairs.__bool__() else row_blocks
    var mag_blocks = row_blocks
    if pairs.__bool__() and pairs.value().group_layout:
        mag_blocks = fv_blocks
    if rank.__bool__():
        fv_blocks = 1
    var fv_part = List[Float32](length=fv_blocks, fill=Float32(0))
    var mag_part = List[Float32](length=2 * mag_blocks, fill=Float32(0))
    var bootstrap_on = loss.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(params.random_seed)
    var noise_rand = TRandom(params.random_seed)
    var losses = List[Float64]()
    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()

    # `fit_loop_` (lane ab-loop-mode): the host clock read once before the
    # first round and once after the last (nothing inside the loop); the
    # rounds executed are the trees the loop added.
    var fit_loop_t0 = fit_loop_start()
    for iteration in range(params.n_estimators):
        if loss.objective == GBDT_OBJ_YETI_RANK:
            yeti_rank_search_pass(
                y, weights, has_weights, cursor, group_sizes,
                yeti_tasks.value(), yeti_rand.next_uniform_l(), 10,
                Float32(0.85), n_rows, stats, fv_part, mag_part,
            )
        elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
            pair_logit_search_pass(
                pairs.value(), cursor, n_rows, stats, fv_part, mag_part,
                second_order,
            )
        elif rank.__bool__():
            rank.value().search(loss.objective, y, cursor, weights, ranking_seed(params.random_seed),
                                iteration, second_order, stats, fv_part, mag_part)
        else:
            query_rmse_search_pass(y, cursor, group_sizes, n_rows, stats, fv_part, mag_part, weights)
        var fv = _deterministic_sum_lanes(fv_part, 1, fv_blocks)[0]
        var tree_seed = noise_rand.next_uniform_l()
        var score_std_dev = Float32(0)
        if random_strength != Float32(0):
            # the device's one body (gbdt/methods/random_score_bodies.mojo;
            # this was a host libm restatement, lane gbdt-cpu-rest-2)
            var mult = score_model_length_mult(
                Float64(n_rows), iteration, params.learning_rate
            )
            if mult * Float64(random_strength) != 0.0:
                score_std_dev = Float32(mult * _pointwise_target_std_dev(stats, n_rows) * Float64(random_strength))
        var scale: Float32
        if bootstrap_on:
            var bm = _bootstrap_pass(loss.bootstrap_kind, seeds, stats, n_rows, loss.bootstrap_param)
            scale = _choose_scale_from_magnitudes(bm[0], bm[1], n_rows)
        else:
            var mags = _deterministic_sum_lanes(mag_part, 2, mag_blocks)
            scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        # the gradient plane snapped in place in `stats`; the search reads
        # both planes from `stats` at offsets 0 and n_rows (lane
        # gbdt-one-body: no per-tree plane lists and copy, `_split_planes`)
        _snap_plane(stats, n_rows, scale, n_rows)
        var splits: List[_OrdSplit]
        if ff_on:
            # the tree's sample: its helpers and feature words replace the
            # fit's for this tree (`_pw_sample_tree`)
            _pw_sample_tree(
                cindex, layout, grid.fold_counts, feature_fraction,
                feature_random, n_rows, n_features, tree_cindex, helpers,
                feat_offset, feat_shift, feat_mask,
            )
            splits = _ordered_tree_structure(
                tree_cindex, helpers, params.max_depth, stats, stats, doc_ids,
                part_bounds, 1, 0, scale, params.l2_leaf_reg, feat_offset,
                feat_shift, feat_mask, search_hot, not cosine_score,
                score_std_dev, tree_seed, cosine_score, sw_off=0, sg_off=n_rows,
            )
        else:
            splits = _ordered_tree_structure(
                cindex, helpers, params.max_depth, stats, stats, doc_ids,
                part_bounds, 1, 0, scale, params.l2_leaf_reg, feat_offset,
                feat_shift, feat_mask, search_hot, not cosine_score,
                score_std_dev, tree_seed, cosine_score, sw_off=0, sg_off=n_rows,
            )
        var n_leaves = 1 << len(splits)
        # `compute_bins_for_model` + `partition_from_bins`: the rows' leaves
        # and the stable counting sort across the cores (lane gbdt-host-par)
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        for level in range(len(splits)):  # driver-loop: over the tree's levels
            tree_features.append(splits[level].feature)
            tree_bins.append(splits[level].bin)
        var partition = _partition_for_tree(
            cindex, layout, tree_features, tree_bins, n_rows
        )
        var yeti_seed = UInt64(0)
        if loss.objective == GBDT_OBJ_YETI_RANK:
            yeti_seed = yeti_rand.next_uniform_l()
        var estimated = _estimate_leaves_for_loss(
            loss, y, cursor, partition.row_index, partition.offsets,
            partition.sizes, n_rows, params.l2_leaf_reg, group_sizes, pairs,
            yeti_tasks, yeti_seed, weights, rank,
        )
        if loss.objective == GBDT_OBJ_PAIR_LOGIT or loss.objective == GBDT_OBJ_YETI_RANK:
            _zero_average_host(estimated)
        var scaled = rescale_apply_leaves_host(
            cursor, estimated, params.learning_rate, partition.row_index,
            partition.offsets, partition.sizes, n_leaves,
        )
        # one computation feeds the cursor and the stored model
        for leaf in range(n_leaves):
            model_leaves.append(scaled[leaf])
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        tree_leaf_offsets.append(len(model_leaves))
        if rank_test:
            if len(splits) > 0:
                var bins = gbdt_eval_test_bins(test_cindex, eval.n_rows, tree_features, tree_bins,
                                               test_offsets, test_masks, test_shifts, one_hot)
                gbdt_eval_add_tree(test_cursor, bins, scaled)
            var held_out = rank_test.value().value(eval.y, test_cursor)
            test_losses.append(held_out)
            detector.add_error(held_out)
            if detector.is_need_stop():
                stopped_early = True
                break
        if iteration > 0:
            losses.append(1.0 - Float64(fv) if rank.__bool__() else -Float64(fv) / loss_norm)
    var fit_loop = fit_loop_stop(fit_loop_t0, len(tree_split_offsets) - 1)

    if rank.__bool__():
        losses.append(1.0 - Float64(rank.value().workspace.ndcg(y, cursor)))
    elif loss.objective == GBDT_OBJ_YETI_RANK:
        # No extra RNG draw: the GPU final loss pass writes zero partials.
        losses.append(-Float64(Float32(0)) / Float64(n_rows))
    elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
        var last = pair_logit_value(pairs.value(), cursor)
        losses.append(-Float64(_deterministic_sum_lanes(last, 1, len(last))[0]) / loss_norm)
    else:
        var last = query_rmse_value(y, cursor, group_sizes, n_rows, weights)
        losses.append(-Float64(_deterministic_sum_lanes(last, 1, len(last))[0]) / loss_norm)
    gbdt_eval_shrink_symmetric(test_losses, eval.want_best_model, eval.best_model_min_trees,
                              tree_split_offsets, tree_leaf_offsets, split_features, split_bins, model_leaves)
    # Query-aware CTRs are single-feature columns; the model uses the fitted grid directly.
    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, detector.best_iteration, stopped_early,
    )
    model.fit_loop = fit_loop^
    return GbdtCtrHostFit(
        GbdtHostFitWithEval(model^, GbdtHostEvalFit(test_losses^, detector.best_iteration, stopped_early)),
        one_hot^, tables^, ctr_column_count, TCombinationCtrs(),
    )
