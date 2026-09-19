# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the NON-SYMMETRIC grow policies,
a second spelling of the device trainer on the gbdt-depthwise and
gbdt-lossguide lanes (workstream E batch 3, 2026-09-14; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md, the batch 3 section).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. It builds on `gbdt/host/gbdt_oracle.mojo` (the
gbdt-symmetric oracle: the grid, the binarize, the Logloss search pass, the
histogram blocks, the partition stats, the Cosine calcer, the Newton leaf
estimator and the float tokens), on `gbdt/host/gbdt_oracle_lossguide.mojo`
(the Lossguide selection, the L2 calcer and the NewtonL2 search planes), and
on host code the device fit ITSELF runs, imported unchanged:
`checks/fixed_point.mojo::choose_scale` (the non-symmetric driver derives
the fixed-point scale on the host, `greedy_search_helper_depthwise.mojo:
1075-1095`) and `greedy_subsets_searcher/split_properties_helper.mojo`
(`build_necessary_histograms`, `non_zero_leaves`; that file imports nothing).

ONE DRIVER FOR BOTH POLICIES, as on the device
(`greedy_search_helper_depthwise.mojo:725-2552`, `fit_non_symmetric_tree`,
one function with four policy branches). THE CONFIGURATIONS THIS COVERS are
the lanes' (tools/identity_break.py): `gbdt-depthwise` is 20 trees, depth 6,
Depthwise, Logloss, Cosine; `gbdt-lossguide` is 20 trees, max_leaves 32,
Lossguide, Logloss, NewtonL2 (the policy's default score function), depth 6.
Since 2026-09-15 (lane/cpu-training-gbdt-losses) also
`gbdt-lossguide-newtoncosine`: 20 trees, max_leaves 32, Lossguide, Logloss,
NewtonCosine (the Cosine calcer over the NewtonL2 planes),
`min_child_hessian` 1 (`child_hessian_below` by bits, a rejected leaf made
terminal), `min_split_gain` 0.01, `min_data_in_leaf` 8, `feature_fraction`
0.5 (`sample_tree_folds` over its own TRandom stream, the index repacked per
tree), `random_strength` 1 (the per-tree `CalcScoreModelLengthMult`, the
per-tree seed, `compute_target_std_dev` and the per-launch level seed with
the per-feature normal draw on both scores), the Bernoulli bootstrap at 0.7
and Gradient leaves at three iterations (`gbdt_oracle_losses.mojo`). Every
other option is at its default and the binding refuses by name what is
outside (bindings/_mojolearn_gbdt_host.mojo, `_refuse`).

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build defaults:
DEVIATION 1902 ridx-only splits OFF, 2661 group width OFF, 1901 incremental
partition stats ON, 1903 and 1904 FAST-only, 2551 device leaf partition ON)

  1. The grid, the binarize and the per-tree search pass: the symmetric
     oracle's, unchanged (`doc_parallel_boosting.mojo:1531-1565`), with the
     NewtonL2 planes for Lossguide.
  2. The scale: `choose_scale(max(|w|, |g|), n_rows)` on the host over the
     two folded magnitudes (`doc_parallel_boosting.mojo:1760-1772`,
     `greedy_search_helper_depthwise.mojo:1075-1095`).
  3. `fit_non_symmetric_tree` (`greedy_search_helper_depthwise.mojo:
     1061-2501`), per iteration:
       - `CreateInitialSubsets` (`:1057-1100`): one root leaf over every row,
         zeroed histograms.
       - `BuildNecessaryHistograms` (`:1249-1269`, imported): the zero pass
         over every compute slot (`:1372-1416`, the IDENTICAL arm), the
         histograms of the non-empty compute leaves through the SAME
         launcher the symmetric oracle restates with `depth = iteration - 1`
         and `n_live = len(non_zero)` (`:1418-1479`; without 2661 the
         one-byte blocks take the block-widest ladder, whose Int32 sums of
         the dithered quantizer are the width arms' sums), the scan over the
         built set (`:1481-1505`), the subtraction over the pairs
         (`:1507-1556`), and the `CurrentPath` reset (`:1558-1572`).
       - `SelectLeavesToVisit` (`:549-570`), then the partition stats over
         every leaf (`:1617-1652`; the 1901 cache recomputes only dirty
         leaves and a clean leaf's range, rows and fold are unchanged, so a
         full recompute is the same bits).
       - The score (`:1665-1803`): `compute_optimal_splits_region_kernel`
         for Depthwise, `compute_optimal_split_kernel` for Lossguide, both
         the leafwise scan of `kernel/compute_scores.mojo:346-500` (the
         clamped weights, the zero-part split scored 0.0, the `score_before`
         of the parent, the gain times feature weight 1.0) and the 256-thread
         argmax of `:504-550` (largest gain, ties to the smaller bin, a
         result at or below -FLOAT32_MAX never taken), one record per
         `argmax_blocks` score block.
       - The host reduce (`:1853-1904`): per visited leaf, the blocks in
         order through `best_split_properties_less` (`gbdt/methods/helpers.
         mojo:114-162`: gain, then feature as ui32, then bin) on the NEGATED
         kernel gain, the `ToSplit` clamp (`:229-264`).
       - `SelectLeavesToSplit` (`:2034-2045`): Depthwise `gain < 0`
         (`:573-621`), Lossguide the argmin (the lossguide file).
       - `MakeSplit` (`:2058-2346`): left keeps the id and offset, right is
         `leavesCount + i`; each split leaf's rows, zeros then ones, in
         order, with both stat planes (`split_and_make_sequence_kernel`,
         `launch_stable_partition_routed`, `launch_reorder_in_leaves`, the
         symmetric oracle's restatement of the same kernels); the parent
         histogram copied into the right slot (`:2299-2329`);
         `update_partitions_after_split_kernel` (`kernel/split_points.mojo:
         156-248`); `SplitLeaf` (`:624-654`) with the `CurrentPath` to
         `PreviousPath` transition; `RebuildLeavesSizes` (`:2352-2357`);
         `MarkTerminal` over both children (`:2401-2412`, `IsTerminalLeaf`
         at `:504-527`, the `<=` size test).
       - `ShouldTerminate` (`:530-546`) and the end-of-tree partition stats,
         whose weight plane is the model's leaf weights (`:2418-2461`).
  4. `BuildTreeLikeModel<TNonSymmetricTree>` (`greedy_subsets_searcher/
     model_builder.mojo:138-347`): the paths folded into a node arena in
     leaf-id order, flattened pre-order, subtree LEAF counts written back,
     leaves numbered left to right.
  5. The estimation (`doc_parallel_boosting.mojo:1832-1904`): the bins off
     the MODEL (`compute_non_symmetric_decision_tree_bins_kernel`, the walk
     `core/gbdt_host_predict.mojo:323-347` restates), the stable grouping
     of `DeviceLeafPartitioner.partition` / `partition_from_bins`
     (`leaves_estimation/doc_parallel_leaves_estimator.mojo:196-259`,
     `:338-398`: rows ascending within a leaf), then the symmetric oracle's
     Newton estimator and fused cursor update, and the rescale folded into
     the stored leaves.
  6. The learn losses (`doc_parallel_boosting.mojo:2158-2169`) and the model
     text's non-symmetric shape (`gbdt/models/model_text.mojo:535-600`:
     `ntree`, `node`, `leaf`, and `weight` records with 64-bit tokens,
     since the searcher's leaf weights ride on the tree).

THE NEGATIVE CONTROL is the symmetric oracle's: `-D MOJOLEARN_HOST_SABOTAGE=1`
adds 1.0 to the Newton walker's Hessian regularizer inside `_estimate_leaves`,
which both policies call for every tree, so every leaf value, every cursor
and therefore every later tree's structure and every prediction move.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the two lanes is the measurement.
"""
from std.math import exp, log, sqrt
from std.memory import bitcast

from checks.fixed_point import choose_scale
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from gbdt.data.permutation import TRandom
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
    _bootstrap_pass,
    _target_std_dev,
    gbdt_bootstrap_seeds,
    GBDT_FLOAT32_MAX,
    GBDT_MSE_BLOCK,
    GbdtHostParams,
    _add_leaf_cosine,
    _binarize_columns,
    _deterministic_sum_lanes,
    _estimate_leaves,
    _half_byte_block,
    _logloss_search_pass,
    _logloss_value,
    _nan_token,
    _one_byte_block,
    _partition_stat,
    gbdt_f32_token,
    gbdt_f64_token,
    gbdt_host_grid,
)
from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_NEWTON,
    GbdtHostLoss,
    _estimate_leaves_for_loss,
)
from gbdt.host.gbdt_oracle_lossguide import (
    add_leaf_l2,
    logloss_search_pass_newton,
    lossguide_select_leaves_to_split,
)
from gbdt.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_CURRENT_PATH,
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    build_necessary_histograms,
    non_zero_leaves,
)


#: `GROW_DEPTHWISE` / `GROW_LOSSGUIDE` (`gbdt/options/catboost_options.mojo:
#: 80-81`), the codes the wrapper sends in slot 31.
comptime GBDT_HOST_GROW_DEPTHWISE = 1
comptime GBDT_HOST_GROW_LOSSGUIDE = 2
#: `SCORE_FUNCTION_COSINE` / `SCORE_FUNCTION_NEWTON_L2`
#: (`catboost_options.mojo:130-131`).
comptime GBDT_HOST_SCORE_COSINE_NS = 1
comptime GBDT_HOST_SCORE_NEWTON_L2 = 2
comptime GBDT_HOST_SCORE_NEWTON_COSINE = 3
#: `LEAFWISE_SCORE_BLOCK_SIZE` (`kernel/compute_scores.mojo:342`).
comptime GBDT_LEAFWISE_BLOCK = 256
#: `ESplitValue` (`gbdt/methods/helpers.mojo:110-111`).
comptime GBDT_SPLIT_ZERO = 0
comptime GBDT_SPLIT_ONE = 1


@fieldwise_init
struct GbdtHostTreeParams(ImplicitlyCopyable, Movable):
    """The symmetric oracle's parameters plus what the non-symmetric fit
    reads (`doc_parallel_boosting.mojo:1738-1745`): the policy, the resolved
    `ns_max_leaves` (`:1093-1107`), `min_data_in_leaf` as the searcher's
    `min_leaf_size`, and the score function code."""

    var base: GbdtHostParams
    var policy: Int
    var max_leaves: Int
    var min_leaf_size: Float64
    var score_function: Int
    #: `child_hessian_threshold` (`gbdt/options/child_hessian.mojo:8-25`),
    #: -1 disabled
    var min_child_hessian: Float32
    #: `min_split_gain`, -1 disabled
    var min_split_gain: Float64
    #: the user's `random_strength` (the fit multiplies it per tree)
    var random_strength: Float32
    var feature_fraction: Float64
    #: the leaf estimator and the bootstrap (`gbdt_oracle_losses.mojo`)
    var loss: GbdtHostLoss


@fieldwise_init
struct GbdtHostNsModel(Movable):
    """`TrainedModel` for a non-symmetric, one-dimensional, float-only
    ensemble, flat: tree `t`'s nodes are `node_*[tree_node_offsets[t] ..
    tree_node_offsets[t + 1])`, its `nodes + 1` leaves
    `leaf_values[tree_leaf_offsets[t] ..]` and `leaf_weights` the same span."""

    var fold_counts: List[Int]
    var borders: List[List[Float32]]
    var nan_treatment: List[Int]
    var tree_node_offsets: List[Int]
    var node_feature: List[Int]
    var node_bin: List[Int]
    var node_left: List[Int]
    var node_right: List[Int]
    var tree_leaf_offsets: List[Int]
    var leaf_values: List[Float32]
    var leaf_weights: List[Float64]
    var losses: List[Float64]

    def n_trees(self) -> Int:
        return len(self.tree_node_offsets) - 1


@fieldwise_init
struct _NsLeaf(Copyable, Movable):
    """`TLeaf` (`points_subsets.mojo:351-402`) with its `TBestSplitProperties`
    (`:285-317`, defaults feature -1, bin 0, gain `Float32.MAX`, undefined)
    and its `TLeafPath` (`gbdt/data/leaf_path.mojo`) flattened. All splits
    are TakeGreater: the binding refuses categoricals."""

    var size: Int
    var histograms_type: Int
    var best_feature: Int32
    var best_bin: Int32
    var best_gain: Float32
    var best_defined: Bool
    var is_terminal: Bool
    var path_features: List[Int]
    var path_bins: List[Int]
    var path_directions: List[Int]

    def depth(self) -> Int:
        return len(self.path_features)

    def reset_best(mut self):
        self.best_feature = Int32(-1)
        self.best_bin = Int32(0)
        self.best_gain = Float32.MAX
        self.best_defined = False


def _fresh_leaf(size: Int) -> _NsLeaf:
    return _NsLeaf(
        size, HISTOGRAMS_ZEROES, Int32(-1), Int32(0), Float32.MAX, False,
        False, List[Int](), List[Int](), List[Int](),
    )


@fieldwise_init
struct _NsTree(Movable):
    """One `TNonSymmetricTree` without its values: the pre-order nodes (bin,
    feature, LEAF-COUNT subtrees) and the leaf weights in bin order."""

    var node_feature: List[Int]
    var node_bin: List[Int]
    var node_left: List[Int]
    var node_right: List[Int]
    var leaf_weights: List[Float64]


# ===========================================================================
# THE SCORE: the leafwise scan and argmax (`kernel/compute_scores.mojo`)
# ===========================================================================


def _leafwise_gain(
    hist: List[Float32],
    hist_cells: Int,
    part_stats: List[Float32],
    leaf: Int,
    bin_feature_id: Int,
    lambda_l2: Float32,
    cosine: Bool,
    min_child_hessian: Float32,
    score_std_dev: Float32,
    level_seed: UInt64,
    feature_id: Int,
    mut rejected: Bool,
) -> Float32:
    """One candidate of `_leafwise_scan_part` (`compute_scores.mojo:396-494`)
    at stat count 2, no multiclass, feature weight 1.0: the child-Hessian
    rejection (`rejected`, the candidate is not scored), the calcer over
    (left, right) and over the parent, the Cosine normalization, the noise
    draw per feature (`advance_seed_k(seed + feature, 4)`, one normal, the
    pinned mul-add on both scores), the zero-part rule, the gain."""
    var score = Float32(0.0)
    var denum_sqr = Float32(1e-10)
    var score_b = Float32(0.0)
    var denum_sqr_b = Float32(1e-10)
    var leaf_base = leaf * 2 * hist_cells
    var part_weight = part_stats[leaf * 2]
    var weight_left = max(hist[leaf_base + bin_feature_id], Float32(0.0))
    var weight_right = ftz(max(part_weight - weight_left, Float32(0.0)))
    rejected = False
    if min_child_hessian >= Float32(0.0):
        # `child_hessian_below` (`compute_scores.mojo:36-42`), by bits
        var th = bitcast[DType.uint32](min_child_hessian) & UInt32(0x7FFFFFFF)
        if (bitcast[DType.uint32](weight_left) & UInt32(0x7FFFFFFF)) < th or (
            bitcast[DType.uint32](weight_right) & UInt32(0x7FFFFFFF)
        ) < th:
            rejected = True
            return Float32(0.0)
    var to_zero_part_split = (
        weight_left < Float32(1e-20) or weight_right < Float32(1e-20)
    )
    var sum_left = hist[leaf_base + hist_cells + bin_feature_id]
    var part_stat = part_stats[leaf * 2 + 1]
    var sum_right = ftz(part_stat - sum_left)
    if cosine:
        _add_leaf_cosine(sum_left, weight_left, lambda_l2, score, denum_sqr)
        _add_leaf_cosine(sum_right, weight_right, lambda_l2, score, denum_sqr)
        _add_leaf_cosine(part_stat, part_weight, lambda_l2, score_b, denum_sqr_b)
    else:
        add_leaf_l2(sum_left, weight_left, lambda_l2, score)
        add_leaf_l2(sum_right, weight_right, lambda_l2, score)
        add_leaf_l2(part_stat, part_weight, lambda_l2, score_b)
    var final_score = score
    var score_before = score_b
    if cosine:
        if denum_sqr > Float32(1e-15):
            final_score = ftz(score / identical_sqrt(denum_sqr))
        else:
            final_score = -GBDT_FLOAT32_MAX
        if denum_sqr_b > Float32(1e-15):
            score_before = ftz(score_b / identical_sqrt(denum_sqr_b))
        else:
            score_before = -GBDT_FLOAT32_MAX
        if score_std_dev != Float32(0.0):
            var seed = advance_seed_k(level_seed + UInt64(feature_id), 4)
            var draw = next_normal_f(seed)
            var neg_draw = -draw[0]
            final_score = ftz(identical_mul_add(neg_draw, score_std_dev, final_score))
            score_before = ftz(identical_mul_add(neg_draw, score_std_dev, score_before))
    if to_zero_part_split:
        final_score = -GBDT_FLOAT32_MAX
        score_before = -GBDT_FLOAT32_MAX
    var gain = Float32(0.0)
    if not to_zero_part_split:
        gain = ftz(final_score - score_before)
    return ftz(gain * Float32(1.0))


def _best_split_less(
    g1: Float32, f1: Int32, b1: Int32, g2: Float32, f2: Int32, b2: Int32
) -> Bool:
    """`best_split_properties_less` (`gbdt/methods/helpers.mojo:149-162`):
    gain, then feature as ui32, then bin as ui32, strict."""
    if g1 < g2:
        return True
    elif g1 == g2:
        var u1 = UInt32(f1)
        var u2 = UInt32(f2)
        if u1 < u2:
            return True
        elif u1 == u2:
            return UInt32(b1) < UInt32(b2)
        return False
    return False


def _score_leaf(
    hist: List[Float32],
    hist_cells: Int,
    part_stats: List[Float32],
    leaf: Int,
    lambda_l2: Float32,
    cosine: Bool,
    min_child_hessian: Float32,
    score_std_dev: Float32,
    level_seed: UInt64,
    argmax_blocks: Int,
    bf_feature: List[Int],
    bf_bin: List[Int],
    layout: CompressedIndexLayout,
    mut out: _NsLeaf,
):
    """The score kernel's records for one leaf, then the host reduce.

    DEVICE, per score block `bx` (`compute_scores.mojo:382-500`, `:504-550`):
    thread `t` walks bins `bx * 256 + t + k * 256 * argmax_blocks`
    ascending under a strict `>` from -FLOAT32_MAX, and the 256-thread tree
    keeps the larger gain, a tie to the smaller bin. So block `bx` holds the
    smallest bin of largest gain among the bins whose `bin // 256` is `bx`
    modulo `argmax_blocks`, stored `ftz`, or the poison record when no gain
    exceeded -FLOAT32_MAX.

    HOST (`greedy_search_helper_depthwise.mojo:1854-1903`): blocks in order,
    poison skipped, `TBestSplitProperties(feature, clamp(bin), -gain, -gain)`
    folded through `best_split_properties_less` from the default record; the
    fold REPLACES the leaf's record, defined or not."""
    out.reset_best()
    for bx in range(argmax_blocks):
        var blk_gain = -GBDT_FLOAT32_MAX
        var blk_bin = -1
        for bf in range(hist_cells):
            if (bf // GBDT_LEAFWISE_BLOCK) % argmax_blocks != bx:
                continue
            var rejected = False
            var gain = _leafwise_gain(
                hist, hist_cells, part_stats, leaf, bf, lambda_l2, cosine,
                min_child_hessian, score_std_dev, level_seed, bf_feature[bf],
                rejected,
            )
            if rejected:
                continue
            if gain > blk_gain:
                blk_gain = gain
                blk_bin = bf
        if blk_bin < 0:
            continue
        var our_gain = ftz(blk_gain)
        var feature = bf_feature[blk_bin]
        var bin = bf_bin[blk_bin]
        # `TBinFeatureTable.to_split`'s clamp (`:252-257`), TakeGreater arm
        var max_bin = Int(layout.features[feature].folds) - 1
        if bin > max_bin:
            bin = max_bin
        var cand_gain = -our_gain
        # `if (blockProps[i] < bestSplits[scoreBlockId])`, the incumbent
        # starting as the default record (`reset_best` above holds exactly
        # its fields: gain Float32.MAX, feature (ui32)-1, bin 0)
        if _best_split_less(
            cand_gain, Int32(feature), Int32(bin),
            out.best_gain, out.best_feature, out.best_bin,
        ):
            out.best_feature = Int32(feature)
            out.best_bin = Int32(bin)
            out.best_gain = cand_gain
            out.best_defined = True


# ===========================================================================
# THE DRIVER: `fit_non_symmetric_tree`
# ===========================================================================


def _is_terminal_leaf(
    leaf: _NsLeaf, min_leaf_size: Float64, max_depth: Int
) -> Bool:
    """`is_terminal_leaf` (`greedy_search_helper_depthwise.mojo:504-527`),
    non-symmetric arm: `size <= min_leaf_size` or `depth >= max_depth`."""
    if Float64(leaf.size) <= min_leaf_size:
        return True
    return leaf.depth() >= max_depth


def _grow_non_symmetric_tree(
    n_rows: Int,
    n_features: Int,
    layout: CompressedIndexLayout,
    blocks: List[PolicyBlock],
    cindex: List[UInt32],
    mut stats: List[Float32],
    fixed_scale: Float32,
    bf_feature: List[Int],
    bf_bin: List[Int],
    params: GbdtHostTreeParams,
    random_strength: Float32,
    tree_seed: UInt64,
) raises -> _NsTree:
    """`fit_non_symmetric_tree` (see the module docstring, stage 3). `stats`
    arrives in document order and leaves permuted by the splits, as the
    device plane does; the caller rewrites it before the next tree."""
    var lossguide = params.policy == GBDT_HOST_GROW_LOSSGUIDE
    var cosine = (
        params.score_function == GBDT_HOST_SCORE_COSINE_NS
        or params.score_function == GBDT_HOST_SCORE_NEWTON_COSINE
    )
    # `level_rand = TRandom(random_seed)` (`:1152`) and `CreateInitialSubsets`'
    # ScoreStdDev (`:1179-1187`): the strength times `compute_target_std_dev`
    # over the (bootstrapped) planes in document order
    var level_rand = TRandom(tree_seed)
    var score_std_dev = Float32(0.0)
    if random_strength != Float32(0.0):
        score_std_dev = Float32(
            Float64(random_strength) * _target_std_dev(stats, n_rows)
        )
    var max_leaves = params.max_leaves
    var max_depth = params.base.max_depth
    var lambda_l2 = params.base.l2_leaf_reg
    var hist_cells = layout.hist_cells

    # `argmaxBlockCount = Min(CeilDivide(binFeatureCount, 256), 64)` (`:853`)
    var argmax_blocks = (hist_cells + 255) // 256
    if argmax_blocks > 64:
        argmax_blocks = 64
    if argmax_blocks < 1:
        argmax_blocks = 1

    # ---- CreateInitialSubsets (`:1057-1100`) ----
    var row_index = List[Int](length=n_rows, fill=0)
    for r in range(n_rows):
        row_index[r] = r
    var p_off = List[Int](length=max_leaves, fill=0)
    var p_sz = List[Int](length=max_leaves, fill=0)
    p_sz[0] = n_rows
    var hist = List[Float32](length=max_leaves * 2 * hist_cells, fill=Float32(0.0))
    var part_stats = List[Float32](length=2 * max_leaves, fill=Float32(0.0))
    var leaves = List[_NsLeaf]()
    leaves.append(_fresh_leaf(n_rows))
    var parent_of = List[Int]()
    parent_of.append(0)

    var max_iterations = max_depth + 2
    if lossguide:
        max_iterations = max_leaves + 1
    var iteration = 0
    var leaf_weights_by_id = List[Float64]()
    while True:
        iteration += 1
        if iteration > max_iterations:
            raise Error(
                String("lossguide" if lossguide else "depthwise")
                + " growth loop did not terminate in "
                + String(max_iterations) + " iterations; leaves="
                + String(len(leaves))
            )

        # ---- BuildNecessaryHistograms (`:1249-1572`) ----
        var records = List[LeafRecord]()
        for i in range(len(leaves)):
            records.append(
                LeafRecord(
                    UInt32(leaves[i].size), leaves[i].histograms_type,
                    parent_of[i], leaves[i].is_terminal,
                )
            )
        var plan = build_necessary_histograms(records)
        var non_zero = non_zero_leaves(records, plan.compute_ids)

        # the IDENTICAL zero pass over every compute slot (`:1384-1416`)
        for i in range(len(plan.compute_ids)):
            var slot = Int(plan.compute_ids[i])
            for c in range(2 * hist_cells):
                hist[slot * 2 * hist_cells + c] = Float32(0.0)

        if len(non_zero) > 0:
            var compute = List[Int]()
            for i in range(len(non_zero)):
                compute.append(Int(non_zero[i]))
            var depth_arg = iteration - 1
            var block_first_bin = 0
            for b in range(len(blocks)):
                ref blk = blocks[b]
                var total = 0
                for k in range(blk.count()):
                    total += Int(blk.folds[k])
                if blk.policy == POLICY_HALF_BYTE:
                    _half_byte_block(
                        blk, block_first_bin, hist_cells, compute, depth_arg,
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
            # `scan_histograms_kernel` over the built set (`:1485-1500`)
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

        # `substract_histograms_kernel` over the pairs (`:1507-1556`)
        for j in range(len(plan.subtract_from)):
            var from_slot = Int(plan.subtract_from[j])
            var what_slot = Int(plan.subtract_what[j])
            for z in range(2):
                var from_base = from_slot * 2 * hist_cells + z * hist_cells
                var what_base = what_slot * 2 * hist_cells + z * hist_cells
                for bf in range(hist_cells):
                    var new_val = ftz(hist[from_base + bf] - hist[what_base + bf])
                    if z == 0:
                        new_val = max(new_val, Float32(0.0))
                    hist[from_base + bf] = new_val

        # `allUpdatedLeaves` (`:1562-1566`)
        var updated = plan.updated_ids()
        for i in range(len(updated)):
            var id = Int(updated[i])
            leaves[id].histograms_type = HISTOGRAMS_CURRENT_PATH
            leaves[id].reset_best()

        # ---- SelectLeavesToVisit (`:549-570`) ----
        var visit = List[Int]()
        for i in range(len(leaves)):
            if not leaves[i].is_terminal and not leaves[i].best_defined:
                visit.append(i)
        if len(visit) > 0:
            # the partition stats, every leaf (`:1617-1652`)
            for i in range(len(leaves)):
                part_stats[2 * i] = _partition_stat(stats, n_rows, 0, p_off[i], p_sz[i])
                part_stats[2 * i + 1] = _partition_stat(stats, n_rows, 1, p_off[i], p_sz[i])
            if lossguide and len(visit) > 2:
                raise Error(
                    "Lossguide scored " + String(len(visit))
                    + " leaves; their CB_ENSURE allows at most 2"
                    " (greedy_search_helper.cpp:511)"
                )
            # `Random.NextUniformL()`, one draw per launch (`:1656`)
            var level_seed = level_rand.next_uniform_l()
            var noise = score_std_dev if cosine else Float32(0.0)
            for i in range(len(visit)):
                _score_leaf(
                    hist, hist_cells, part_stats, visit[i], lambda_l2, cosine,
                    params.min_child_hessian, noise, level_seed,
                    argmax_blocks, bf_feature, bf_bin, layout, leaves[visit[i]],
                )
            # a rejected leaf cannot become eligible later in this tree
            # (`:2001-2004`)
            if params.min_child_hessian >= Float32(0.0):
                for i in range(len(visit)):
                    if not leaves[visit[i]].best_defined:
                        leaves[visit[i]].is_terminal = True

        # ---- SelectLeavesToSplit (`:2034-2045`) ----
        var to_split = List[Int]()
        if lossguide:
            var defined = List[Bool]()
            var gains = List[Float32]()
            for i in range(len(leaves)):
                defined.append(leaves[i].best_defined)
                gains.append(leaves[i].best_gain)
            to_split = lossguide_select_leaves_to_split(defined, gains)
        else:
            for i in range(len(leaves)):
                if leaves[i].best_defined and leaves[i].best_gain < Float32(0.0):
                    to_split.append(i)
        # the opt-in split-gain threshold (`:2050-2056`)
        if params.min_split_gain >= 0.0:
            var accepted = List[Int]()
            for k in range(len(to_split)):
                if Float64(-leaves[to_split[k]].best_gain) > params.min_split_gain:
                    accepted.append(to_split[k])
            to_split = accepted^

        if len(to_split) > 0:
            # ---- MakeSplit's multi-leaf arm (`:2058-2346`) ----
            var leaves_count = len(leaves)
            var new_rows = row_index.copy()
            var new_stats = stats.copy()
            for i in range(len(to_split)):
                var left_id = to_split[i]
                var right_id = leaves_count + i
                if not leaves[left_id].best_defined:
                    raise Error("Best split is undefined for leaf " + String(left_id))
                var split_f = Int(leaves[left_id].best_feature)
                var split_b = Int(leaves[left_id].best_bin)
                ref sfeat = layout.features[split_f]
                # the split chain over the parent's range, zeros then ones
                var off = p_off[left_id]
                var sz = p_sz[left_id]
                var zeros = List[Int]()
                var ones = List[Int]()
                for k in range(sz):
                    var row = row_index[off + k]
                    var word = cindex[Int(sfeat.offset) * n_rows + row]
                    var feature_val = word & (sfeat.mask << sfeat.shift)
                    var value = UInt32(split_b) << sfeat.shift
                    if feature_val > value:
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
                # `copy_histograms_kernel`: the parent into the right slot
                var src_base = left_id * 2 * hist_cells
                var dst_base = right_id * 2 * hist_cells
                for c in range(2 * hist_cells):
                    hist[dst_base + c] = hist[src_base + c]
                # `update_partitions_after_split_kernel`
                p_sz[left_id] = len(zeros)
                p_off[right_id] = off + len(zeros)
                p_sz[right_id] = sz - len(zeros)
                # `SplitLeaf` over the parent snapshot, both children
                var parent = leaves[left_id].copy()
                var child_type = HISTOGRAMS_ZEROES
                if parent.histograms_type == HISTOGRAMS_CURRENT_PATH:
                    child_type = HISTOGRAMS_PREVIOUS_PATH
                var left = _fresh_leaf(0)
                left.histograms_type = child_type
                left.path_features = parent.path_features.copy()
                left.path_bins = parent.path_bins.copy()
                left.path_directions = parent.path_directions.copy()
                var right = left.copy()
                left.path_features.append(split_f)
                left.path_bins.append(split_b)
                left.path_directions.append(GBDT_SPLIT_ZERO)
                right.path_features.append(split_f)
                right.path_bins.append(split_b)
                right.path_directions.append(GBDT_SPLIT_ONE)
                leaves[left_id] = left^
                leaves.append(right^)
                parent_of[left_id] = left_id
                parent_of.append(left_id)
            row_index = new_rows^
            stats = new_stats^
            # `RebuildLeavesSizes` (`:2352-2357`)
            for i in range(len(leaves)):
                leaves[i].size = p_sz[i]
            # `MarkTerminal(leftIds)`, `MarkTerminal(rightIds)` (`:2404-2412`)
            for i in range(len(to_split)):
                var left_id = to_split[i]
                var right_id = leaves_count + i
                leaves[left_id].is_terminal = _is_terminal_leaf(
                    leaves[left_id], params.min_leaf_size, max_depth
                )
                leaves[right_id].is_terminal = _is_terminal_leaf(
                    leaves[right_id], params.min_leaf_size, max_depth
                )
        else:
            for i in range(len(leaves)):
                leaves[i].is_terminal = True

        # ---- ShouldTerminate (`:530-546`) and the leaf weights ----
        var terminate = len(leaves) >= max_leaves
        if not terminate:
            terminate = True
            for i in range(len(leaves)):
                if not leaves[i].is_terminal:
                    terminate = False
                    break
        if terminate:
            for i in range(len(leaves)):
                leaf_weights_by_id.append(
                    Float64(_partition_stat(stats, n_rows, 0, p_off[i], p_sz[i]))
                )
            break

    return _build_flat_tree(leaves, leaf_weights_by_id)


# ===========================================================================
# THE MODEL BUILDER (`greedy_subsets_searcher/model_builder.mojo`)
# ===========================================================================


@fieldwise_init
struct _Arena(Movable):
    """`TFlatTreeBuilder`'s node arena (`model_builder.mojo:59-107`): a child
    is an index, -1 their `nullptr`."""

    var is_terminal: List[Bool]
    var feature: List[Int]
    var bin: List[Int]
    var leaf_id: List[Int]
    var left: List[Int]
    var right: List[Int]
    var root: Int

    def child(self, parent: Int, is_right: Bool) -> Int:
        if parent < 0:
            return self.root
        if is_right:
            return self.right[parent]
        return self.left[parent]

    def set_child(mut self, parent: Int, is_right: Bool, child: Int):
        if parent < 0:
            self.root = child
        elif is_right:
            self.right[parent] = child
        else:
            self.left[parent] = child

    def push(mut self, terminal: Bool, feature: Int, bin: Int, leaf: Int) -> Int:
        self.is_terminal.append(terminal)
        self.feature.append(feature)
        self.bin.append(bin)
        self.leaf_id.append(leaf)
        self.left.append(-1)
        self.right.append(-1)
        return len(self.is_terminal) - 1


def _visit(
    arena: _Arena,
    cursor: Int,
    mut out_feature: List[Int],
    mut out_bin: List[Int],
    mut out_left: List[Int],
    mut out_right: List[Int],
    mut out_leaf_order: List[Int],
) raises -> Int:
    """`TFlatTreeBuilder._visit` (`model_builder.mojo:251-296`): the node is
    pushed BEFORE either recursion, both subtree leaf counts written back
    after; a terminal node appends its leaf in visit order."""
    if cursor < 0:
        raise Error("Tree is empty (cursor is nullptr)")
    if arena.is_terminal[cursor]:
        out_leaf_order.append(arena.leaf_id[cursor])
        return 1
    out_feature.append(arena.feature[cursor])
    out_bin.append(arena.bin[cursor])
    out_left.append(0)
    out_right.append(0)
    var idx = len(out_feature) - 1
    var left_subtree = _visit(
        arena, arena.left[cursor], out_feature, out_bin, out_left, out_right,
        out_leaf_order,
    )
    var right_subtree = _visit(
        arena, arena.right[cursor], out_feature, out_bin, out_left, out_right,
        out_leaf_order,
    )
    if left_subtree > 65535 or right_subtree > 65535:
        raise Error("TTreeNode subtree does not fit in ui16")
    out_left[idx] = left_subtree
    out_right[idx] = right_subtree
    return left_subtree + right_subtree


def _build_flat_tree(
    leaves: List[_NsLeaf], weights_by_id: List[Float64]
) raises -> _NsTree:
    """`build_non_symmetric_tree` (`model_builder.mojo:299-347`) with the
    `Exception` duplicate policy: `add` per leaf in leaf-id order
    (`:153-231`), then the pre-order flatten. The weights follow the leaves
    into visit order."""
    if len(leaves) == 0:
        raise Error("Error: empty region")
    var arena = _Arena(
        List[Bool](), List[Int](), List[Int](), List[Int](), List[Int](),
        List[Int](), -1,
    )
    for leaf in range(len(leaves)):
        var parent = -1
        var is_right = False
        for i in range(leaves[leaf].depth()):
            var f = leaves[leaf].path_features[i]
            var b = leaves[leaf].path_bins[i]
            var cursor = arena.child(parent, is_right)
            if cursor < 0:
                cursor = arena.push(False, f, b, -1)
                arena.set_child(parent, is_right, cursor)
            elif arena.is_terminal[cursor] or arena.feature[cursor] != f or arena.bin[cursor] != b:
                raise Error("Error: path is not from current tree.")
            parent = cursor
            is_right = leaves[leaf].path_directions[i] == GBDT_SPLIT_ONE
        var end = arena.child(parent, is_right)
        if end >= 0:
            raise Error("Can't add terminal leaf twice")
        var node = arena.push(True, 0, 0, leaf)
        arena.set_child(parent, is_right, node)
    var nf = List[Int]()
    var nb = List[Int]()
    var nl = List[Int]()
    var nr = List[Int]()
    var order = List[Int]()
    _ = _visit(arena, arena.root, nf, nb, nl, nr, order)
    var weights = List[Float64]()
    for i in range(len(order)):
        weights.append(weights_by_id[order[i]])
    return _NsTree(nf^, nb^, nl^, nr^, weights^)


def _non_symmetric_bins(
    tree: _NsTree,
    layout: CompressedIndexLayout,
    cindex: List[UInt32],
    n_rows: Int,
) raises -> List[Int]:
    """`compute_non_symmetric_decision_tree_bins_kernel` over the training
    index (`add_non_symmetric_tree_doc_parallel.mojo:63-188`,
    `add_bin_values.mojo:277-320`; the walk `core/gbdt_host_predict.mojo:
    323-347` restates): shift then mask, `>` against the node's bin, a right
    turn adds the left subtree's leaf count. No nodes: every row in bin 0."""
    var n_nodes = len(tree.node_feature)
    var bins = List[Int](length=n_rows, fill=0)
    if n_nodes == 0:
        return bins^
    for r in range(n_rows):
        var bin = 0
        var node = 0
        var stop = False
        while not stop:
            ref cf = layout.features[tree.node_feature[node]]
            var feature_val = (cindex[Int(cf.offset) * n_rows + r] >> cf.shift) & cf.mask
            if feature_val > UInt32(tree.node_bin[node]):
                bin += tree.node_left[node]
                stop = tree.node_right[node] == 1
                if not stop:
                    node += tree.node_left[node]
            else:
                stop = tree.node_left[node] == 1
                if not stop:
                    node += 1
        bins[r] = bin
    return bins^


# ===========================================================================
# THE FIT
# ===========================================================================


def _sample_tree_folds(
    folds: List[Int], fraction: Float64, mut random: TRandom
) raises -> List[Int]:
    """`sample_tree_folds` (`gbdt/gpu_data/feature_sampling.mojo:31-54`),
    restated because that module defines a kernel: the eligible features,
    `max(1, Int(eligible * fraction + 0.5))`, the partial Fisher-Yates over
    `TRandom.uniform`, the full-length fold vector."""
    if not (fraction > 0.0) or fraction > 1.0:
        raise Error("feature_fraction must be finite and in (0, 1]")
    var eligible = List[Int]()
    for f in range(len(folds)):
        if folds[f] > 0:
            eligible.append(f)
    var count = max(1, Int(Float64(len(eligible)) * fraction + 0.5))
    if count >= len(eligible):
        return folds.copy()
    var result = List[Int](length=len(folds), fill=0)
    for i in range(count):
        var j = i + Int(random.uniform(UInt64(len(eligible) - i)))
        var selected = eligible[j]
        eligible[j] = eligible[i]
        eligible[i] = selected
        result[selected] = folds[selected]
    return result^


def gbdt_host_fit_non_symmetric(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostTreeParams,
) raises -> GbdtHostNsModel:
    """`train` then `fit_with_test`'s non-symmetric arm on the covered
    configuration (see the module docstring)."""
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if params.base.max_depth < 0:
        raise Error("max_depth must not be negative")
    if params.policy != GBDT_HOST_GROW_DEPTHWISE and params.policy != GBDT_HOST_GROW_LOSSGUIDE:
        raise Error("gbdt_host_fit_non_symmetric is Depthwise or Lossguide")
    if params.max_leaves < 2:
        raise Error("max_leaves must be at least 2, got " + String(params.max_leaves))
    var newton = (
        params.score_function == GBDT_HOST_SCORE_NEWTON_L2
        or params.score_function == GBDT_HOST_SCORE_NEWTON_COSINE
    )
    if not newton and params.score_function != GBDT_HOST_SCORE_COSINE_NS:
        raise Error("the non-symmetric host fit restates Cosine, NewtonL2 and NewtonCosine only")
    var base = params.base

    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, base.border_count,
        base.border_build_max_samples, base.random_seed, base.nan_mode,
        base.border_type,
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
                " one-byte policies only (gbdt/host/gbdt_oracle_depthwise.mojo)"
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

    var lr = base.learning_rate
    var border = base.logloss_border
    var cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_part = List[Float32](length=mse_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mse_blocks, fill=Float32(0.0))

    var bootstrap_on = params.loss.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(base.random_seed)
    # `noise_rand = TRandom(random_seed)` (`doc_parallel_boosting.mojo:
    # 1339`), one draw per tree; the feature stream (`:1418-1421`)
    var noise_rand = TRandom(base.random_seed)
    var feature_random = TRandom(base.random_seed ^ UInt64(0x4645415455524553))
    # the Newton walker the four covered Logloss lanes were measured on
    var plain_newton = (
        params.loss.method == GBDT_LEAF_NEWTON and not bootstrap_on
    )

    var losses = List[Float64]()
    var tree_node_offsets = List[Int]()
    tree_node_offsets.append(0)
    var node_feature = List[Int]()
    var node_bin = List[Int]()
    var node_left = List[Int]()
    var node_right = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()
    var model_weights = List[Float64]()

    for iteration in range(base.n_estimators):
        # ---- the per-tree feature sample (`doc_parallel_boosting.mojo:
        # 1462-1480`, `gbdt/gpu_data/feature_sampling.mojo:31-54`) and the
        # projected index ----
        var t_layout = layout.copy()
        var t_blocks = blocks.copy()
        var t_cindex = cindex.copy()
        var t_bf_feature = bf_feature.copy()
        var t_bf_bin = bf_bin.copy()
        if params.feature_fraction < 1.0:
            var tree_folds = _sample_tree_folds(
                grid.fold_counts, params.feature_fraction, feature_random
            )
            t_layout = build_layout(tree_folds, one_hot)
            t_blocks = blocks_for(t_layout, n_rows)
            for b in range(len(t_blocks)):
                if t_blocks[b].policy == POLICY_BINARY:
                    raise Error(
                        "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a"
                        " sampled feature with exactly one border"
                    )
            t_cindex = List[UInt32](length=n_rows * t_layout.columns, fill=UInt32(0))
            for f in range(n_features):
                if tree_folds[f] <= 0:
                    continue
                ref src = layout.features[f]
                ref dstf = t_layout.features[f]
                for r in range(n_rows):
                    var value = (cindex[Int(src.offset) * n_rows + r] >> src.shift) & src.mask
                    t_cindex[Int(dstf.offset) * n_rows + r] = (
                        t_cindex[Int(dstf.offset) * n_rows + r] | (value << dstf.shift)
                    )
            t_bf_feature = List[Int](length=t_layout.hist_cells, fill=0)
            t_bf_bin = List[Int](length=t_layout.hist_cells, fill=0)
            for f in range(n_features):
                ref lf = t_layout.features[f]
                for b in range(Int(lf.folds)):
                    t_bf_feature[Int(lf.first_fold_index) + b] = f
                    t_bf_bin[Int(lf.first_fold_index) + b] = b

        # ---- the gradients, the learn loss and the magnitudes ----
        if newton:
            logloss_search_pass_newton(y, cursor, n_rows, border, stats, fv_part, mag_part)
        else:
            _logloss_search_pass(y, cursor, n_rows, border, stats, fv_part, mag_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, mse_blocks)[0]
        var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
        # `calc_score_model_length_mult` (`random_score_helper.mojo:
        # 219-243`) and the per-tree seed, drawn every tree
        var noise_mult = Float64(0.0)
        if params.random_strength != Float32(0.0):
            var model_exp_length = log(Float64(n_rows))
            var model_left = exp(
                model_exp_length - Float64(iteration) * Float64(base.learning_rate)
            )
            noise_mult = model_left / (1.0 + model_left)
        var tree_seed = noise_rand.next_uniform_l()
        if bootstrap_on:
            var bm = _bootstrap_pass(
                params.loss.bootstrap_kind, seeds, stats, n_rows,
                params.loss.bootstrap_param,
            )
            mags[0] = bm[0]
            mags[1] = bm[1]
        # the host scale (`greedy_search_helper_depthwise.mojo:1079-1089`)
        var mag = Float64(mags[0])
        if mag < 0.0:
            mag = -mag
        var gmag = Float64(mags[1])
        if gmag < 0.0:
            gmag = -gmag
        if gmag > mag:
            mag = gmag
        var fixed_scale = Float32(choose_scale(mag, n_rows))

        var tree = _grow_non_symmetric_tree(
            n_rows, n_features, t_layout, t_blocks, t_cindex, stats, fixed_scale,
            t_bf_feature, t_bf_bin, params,
            Float32(noise_mult * Float64(params.random_strength)), tree_seed,
        )
        var n_bins = len(tree.node_feature) + 1

        # ---- the bins off the model, the stable grouping ----
        var bins = _non_symmetric_bins(tree, layout, cindex, n_rows)
        var sizes = List[Int](length=n_bins, fill=0)
        for r in range(n_rows):
            if bins[r] < 0 or bins[r] >= n_bins:
                raise Error(
                    "partition_from_bins: row " + String(r) + " fell in leaf "
                    + String(bins[r]) + " of " + String(n_bins)
                )
            sizes[bins[r]] += 1
        var offsets = List[Int]()
        var running = 0
        for i in range(n_bins):
            offsets.append(running)
            running += sizes[i]
        var fill = offsets.copy()
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[fill[bins[r]]] = r
            fill[bins[r]] += 1

        # ---- the estimation task and `AppendModels` ----
        var estimated: List[Float32]
        if plain_newton:
            estimated = _estimate_leaves(
                y, cursor, row_index, offsets, sizes, n_rows, border,
                base.l2_leaf_reg, base.leaf_estimation_iterations,
            )
        else:
            estimated = _estimate_leaves_for_loss(
                params.loss, y, cursor, row_index, offsets, sizes, n_rows,
                base.l2_leaf_reg,
            )
        if len(estimated) != n_bins:
            raise Error(
                "the estimator returned " + String(len(estimated))
                + " values for a non-symmetric tree of " + String(n_bins) + " bins"
            )
        for leaf in range(n_bins):
            for k in range(sizes[leaf]):
                var row = row_index[offsets[leaf] + k]
                cursor[row] = identical_mul_add(estimated[leaf], lr, cursor[row])
        for i in range(len(tree.node_feature)):
            node_feature.append(tree.node_feature[i])
            node_bin.append(tree.node_bin[i])
            node_left.append(tree.node_left[i])
            node_right.append(tree.node_right[i])
        tree_node_offsets.append(len(node_feature))
        for i in range(n_bins):
            model_leaves.append(estimated[i] * lr)
            model_weights.append(tree.leaf_weights[i])
        tree_leaf_offsets.append(len(model_leaves))

        # the learn loss read alongside this iteration's gradients
        if len(losses) < base.n_estimators:
            if iteration + 1 > 1:
                losses.append(-Float64(fv) / Float64(n_rows))

    losses.append(-Float64(_logloss_value(y, cursor, n_rows, border)) / Float64(n_rows))
    return GbdtHostNsModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_node_offsets^, node_feature^, node_bin^, node_left^, node_right^,
        tree_leaf_offsets^, model_leaves^, model_weights^, losses^,
    )


# ===========================================================================
# THE MODEL TEXT (`gbdt/models/model_text.mojo:374-670`, non-symmetric shape)
# ===========================================================================


def gbdt_host_ns_model_text(m: GbdtHostNsModel) raises -> String:
    """`model_text` for a float-only non-symmetric one-dimensional model with
    zero bias: the header comment and the four header records, one `feature`
    record per column, then per tree `ntree`, its pre-order `node` records,
    `leaf` records in bin order and `weight` records with 64-bit tokens
    (`model_text.mojo:535-600`, the searcher's weights ride on
    `TNonSymmetricTree`), then the `loss` records."""
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
        var lo = m.tree_node_offsets[t]
        var n_nodes = m.tree_node_offsets[t + 1] - lo
        var n_bins = n_nodes + 1
        var leaf_lo = m.tree_leaf_offsets[t]
        if m.tree_leaf_offsets[t + 1] - leaf_lo != n_bins:
            raise Error(
                "non-symmetric tree " + String(t) + " has " + String(n_bins)
                + " bins, dim 1 and "
                + String(m.tree_leaf_offsets[t + 1] - leaf_lo) + " leaf values"
            )
        out += (
            String("ntree ") + String(t) + " nodes " + String(n_nodes)
            + " dim " + String(1) + " weights " + String(1) + "\n"
        )
        for i in range(n_nodes):
            out += (
                String("node ") + String(t) + " " + String(i) + " "
                + String(m.node_feature[lo + i]) + " " + String(m.node_bin[lo + i])
                + " " + String(m.node_left[lo + i]) + " "
                + String(m.node_right[lo + i]) + "\n"
            )
        for i in range(n_bins):
            out += (
                String("leaf ") + String(t) + " " + String(i) + " "
                + gbdt_f32_token(m.leaf_values[leaf_lo + i]) + "\n"
            )
        for i in range(n_bins):
            out += (
                String("weight ") + String(t) + " " + String(i) + " "
                + gbdt_f64_token(m.leaf_weights[leaf_lo + i]) + "\n"
            )
    for i in range(len(m.losses)):
        out += String("loss ") + String(i) + " " + gbdt_f64_token(m.losses[i]) + "\n"
    return out^
