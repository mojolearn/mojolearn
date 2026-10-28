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

from std.math import exp, fma, isfinite, log, sqrt

from checks.numerics import ftz, identical_mul_add
from gbdt.data.pairs import (
    IDN_PAIRLOGIT_GROUP,
    PairList,
    generate_pairs,
    order_pairs_by_winner,
)
from gbdt.data.permutation import TRandom
from gbdt.data.yeti_rank_tasks import YetiRankTasks, yeti_rank_tasks
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    GbdtHostModel,
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
    _zero_average_host,
)
from gbdt.host.gbdt_oracle_ordered import _PwHelper, _ordered_tree_structure
from gbdt.host.gbdt_oracle_pair import (
    HostPairs,
    host_pair_groups,
    host_pairs,
    pair_logit_search_pass,
    pair_logit_value,
)
from gbdt.host.gbdt_oracle_query import query_rmse_search_pass, query_rmse_value
from gbdt.host.gbdt_oracle_yeti import yeti_rank_search_pass


def _pointwise_target_std_dev(stats: List[Float32], n_rows: Int) -> Float64:
    """Compute the doc-parallel score-noise deviation before bootstrap."""
    # random_score_helper.mojo pins 4 * 32 blocks of 512 threads in identical
    # mode. This is a reduction topology, not a hardware-utilization cap.
    # Its row contribution is w * (g/w)^2 without a zero-weight guard; the
    # greedy helper's weight denominator and guard are different algorithms.
    comptime BLOCK = 512
    var blocks = min(128, max(1, (n_rows + BLOCK - 1) // BLOCK))
    var partials = List[Float32](length=blocks, fill=Float32(0))
    for b in range(blocks):
        var slab = List[Float32](length=BLOCK, fill=Float32(0))
        for t in range(BLOCK):
            var r = b * BLOCK + t
            var acc = Float32(0)
            while r < n_rows:
                var w = stats[r]
                var q = ftz(stats[n_rows + r] / w)
                acc = ftz(identical_mul_add(ftz(w * q), q, acc))
                r += blocks * BLOCK
            slab[t] = acc
        partials[b] = _halving_fold(slab)
    var sum2 = _deterministic_sum_lanes(partials, 1, blocks)[0]
    return sqrt(Float64(sum2) / (Float64(n_rows) + 1e-100))


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
) raises -> GbdtHostModel:
    """Fit grouped targets on numeric columns using pointwise tree search."""
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features or len(y) != n_rows:
        raise Error("pointwise ranking input dimensions do not match")
    if params.max_depth < 0 or params.max_depth > 16:
        raise Error("pointwise ranking depth must be in [0, 16]")
    if loss.objective != GBDT_OBJ_PAIR_LOGIT and loss.objective != GBDT_OBJ_YETI_RANK and loss.objective != GBDT_OBJ_QUERY_RMSE:
        raise Error("pointwise ranking needs QueryRMSE, PairLogit or YetiRank")
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
            var hp = host_pair_groups(sizes_u32, y, List[Float32](), n_rows)
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
                    generate_pairs(sizes_u32, y, List[Float32]()), sizes_u32, n_rows
                )
            var hp = host_pairs(ordered.winners, ordered.losers, ordered.weights, n_rows)
            loss_norm = hp.total_weight()
            pairs = Optional(hp^)
    var yeti_tasks = Optional[YetiRankTasks]()
    var yeti_rand = TRandom(params.random_seed ^ UInt64(0x5945544952414E4B))
    if loss.objective == GBDT_OBJ_YETI_RANK:
        yeti_tasks = Optional(yeti_rank_tasks(sizes_u32, n_rows))

    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, params.border_count,
        params.border_build_max_samples, params.random_seed, params.nan_mode,
        params.border_type,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
    var layout = build_layout(grid.fold_counts, one_hot)
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
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
        helpers.append(_PwHelper(
            blk.policy, gids^, offs^, firsts^, folds^, hist_line,
            List[Float32](), List[Int](), List[Bool](),
        ))
    var feat_offset = List[Int](length=n_features, fill=0)
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    for f in range(n_features):
        feat_offset[f] = Int(layout.features[f].offset) * n_rows
        feat_shift[f] = layout.features[f].shift
        feat_mask[f] = layout.features[f].mask
    var doc_ids = List[Int](length=n_rows, fill=0)
    for r in range(n_rows):
        doc_ids[r] = r
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

    for iteration in range(params.n_estimators):
        if loss.objective == GBDT_OBJ_YETI_RANK:
            yeti_rank_search_pass(
                y, List[Float32](), False, cursor, group_sizes,
                yeti_tasks.value(), yeti_rand.next_uniform_l(), 10,
                Float32(0.85), n_rows, stats, fv_part, mag_part,
            )
        elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
            pair_logit_search_pass(
                pairs.value(), cursor, n_rows, stats, fv_part, mag_part,
                second_order,
            )
        else:
            query_rmse_search_pass(y, cursor, group_sizes, n_rows, stats, fv_part, mag_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, fv_blocks)[0]
        var tree_seed = noise_rand.next_uniform_l()
        var score_std_dev = Float32(0)
        if random_strength != Float32(0):
            var left = exp(fma(-Float64(iteration), Float64(params.learning_rate), log(Float64(n_rows))))
            var mult = left / (1.0 + left)
            if mult * Float64(random_strength) != 0.0:
                score_std_dev = Float32(mult * _pointwise_target_std_dev(stats, n_rows) * Float64(random_strength))
        var scale: Float32
        if bootstrap_on:
            var bm = _bootstrap_pass(loss.bootstrap_kind, seeds, stats, n_rows, loss.bootstrap_param)
            scale = _choose_scale_from_magnitudes(bm[0], bm[1], n_rows)
        else:
            var mags = _deterministic_sum_lanes(mag_part, 2, mag_blocks)
            scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        var sw = List[Float32](length=n_rows, fill=Float32(0))
        var sg = List[Float32](length=n_rows, fill=Float32(0))
        for r in range(n_rows):
            sw[r] = stats[r]
            sg[r] = stats[n_rows + r]
        _snap_plane(sg, n_rows, scale)
        var splits = _ordered_tree_structure(
            cindex, helpers, params.max_depth, sw, sg, doc_ids, part_bounds,
            1, 0, scale, params.l2_leaf_reg, feat_offset, feat_shift,
            feat_mask, not cosine_score, score_std_dev, tree_seed, cosine_score,
        )
        var n_leaves = 1 << len(splits)
        var bins = List[Int](length=n_rows, fill=0)
        var sizes = List[Int](length=n_leaves, fill=0)
        for r in range(n_rows):
            var leaf = 0
            for level in range(len(splits)):
                var fid = splits[level].feature
                var mask = feat_mask[fid] << feat_shift[fid]
                var value = UInt32(splits[level].bin) << feat_shift[fid]
                if (cindex[feat_offset[fid] + r] & mask) > value:
                    leaf += 1 << level
            bins[r] = leaf
            sizes[leaf] += 1
        var offsets = List[Int](length=n_leaves, fill=0)
        var running = 0
        for leaf in range(n_leaves):
            offsets[leaf] = running
            running += sizes[leaf]
        var fill = offsets.copy()
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[fill[bins[r]]] = r
            fill[bins[r]] += 1
        var yeti_seed = UInt64(0)
        if loss.objective == GBDT_OBJ_YETI_RANK:
            yeti_seed = yeti_rand.next_uniform_l()
        var estimated = _estimate_leaves_for_loss(
            loss, y, cursor, row_index, offsets, sizes, n_rows,
            params.l2_leaf_reg, group_sizes, pairs, yeti_tasks, yeti_seed,
        )
        if loss.objective == GBDT_OBJ_PAIR_LOGIT or loss.objective == GBDT_OBJ_YETI_RANK:
            _zero_average_host(estimated)
        for leaf in range(n_leaves):
            for k in range(sizes[leaf]):
                var r = row_index[offsets[leaf] + k]
                cursor[r] = identical_mul_add(estimated[leaf], params.learning_rate, cursor[r])
            model_leaves.append(estimated[leaf] * params.learning_rate)
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        tree_leaf_offsets.append(len(model_leaves))
        if iteration > 0:
            losses.append(-Float64(fv) / loss_norm)

    if loss.objective == GBDT_OBJ_YETI_RANK:
        # No extra RNG draw: the GPU final loss pass writes zero partials.
        losses.append(-Float64(Float32(0)) / Float64(n_rows))
    elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
        var last = pair_logit_value(pairs.value(), cursor)
        losses.append(-Float64(_deterministic_sum_lanes(last, 1, len(last))[0]) / loss_norm)
    else:
        var last = query_rmse_value(y, cursor, group_sizes, n_rows)
        losses.append(-Float64(_deterministic_sum_lanes(last, 1, len(last))[0]) / loss_norm)
    return GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, 0, False,
    )
