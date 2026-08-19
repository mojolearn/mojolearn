"""Leaf values for a grown tree.

MIRRORS `catboost/cuda/methods/leaves_estimation/`, which in CatBoost is a
whole subsystem: `TLeavesEstimation` with a descent loop, an ordered variant,
exact estimation for some objectives, and per-objective backtracking.

**Only the pointwise Newton step is ported**, which is what their descent
reduces to in one iteration for a pointwise objective with a diagonal
Hessian:

    value = -sum(gradient) / (sum(weight) + l2)

clipped by `max_leaf_value`. NOT ported, and named so nobody assumes
otherwise: `leaf_estimation_iterations > 1`, ordered boosting's separate
estimation, exact estimation for MAE and quantile, and the backtracking line
search. Those change the VALUE a leaf gets and none of them change the tree
structure, which is what this repository has been building.

Their `leaf_estimation_backtracking` default is also not here. A port of the
descent loop belongs beside this the day a non-Newton objective does.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


comptime LEAF_BLOCK = 256


def compute_leaf_values_kernel(
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    n_leaves_in: Int32,
    l2: Float32,
    max_leaf_value: Float32,
    out_values: MutPointer[Float32, MutAnyOrigin],
):
    """The Newton step per leaf, from the stats the level already computed.

    `part_stats` is `[leaf][stat]` with stat 0 the weight plane and stat 1
    the gradient, the same layout `compute_optimal_splits_kernel` reads, so
    this needs no new reduction: `compute_partition_stats` has already
    produced it for the final level.

    An EMPTY leaf gets 0.0 rather than a division by `l2`. A tree at depth 8
    over 4,096 rows has empty leaves as a matter of course (46 of 64 were
    populated in this repository's own check), so this is the common case and
    not an edge case.

    The clip is CatBoost's `max_leaf_value`. It is applied here rather than
    at prediction because a stored model should not contain a value the
    trainer would refuse.
    """
    var stat_count = Int(stat_count_in)
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return

    var w = part_stats.unsafe_load(leaf * stat_count)
    var g = part_stats.unsafe_load(leaf * stat_count + 1)

    if w <= Float32(0.0):
        out_values.unsafe_store(leaf, Float32(0.0))
        return

    # ==================== SIGN CONVENTION ====================
    # `+g`, NOT `-g`. This is CatBoost's convention and it is easy to get
    # backwards, so it is written down rather than inferred.
    #
    # Their `MseImpl` sets `der[i] = weight * (relev - val)`
    # (`pointwise_targets.cu`), which they name `direction`: it is the
    # NEGATIVE loss gradient, already pointing downhill. Their Newton step
    # then solves `Hessian * x = Gradient` and uses `x` UNNEGATED as the move
    # direction (`descent_helpers.cpp:104-116`).
    #
    # So a leaf's value is `+sum(der) / (sum(der2) + l2)`, the weighted mean
    # residual, and adding it to the prediction moves toward the target.
    #
    # It was `-g` here, ported before any target existed to fix the
    # convention. With CatBoost's `der` that inverts every step: measured on
    # `boosting_check`, the loss GREW by about 1.68x per iteration, from 231
    # to 55839 over twelve trees, instead of falling.
    # =========================================================
    var v = g / (w + l2)
    if v > max_leaf_value:
        v = max_leaf_value
    if v < -max_leaf_value:
        v = -max_leaf_value
    out_values.unsafe_store(leaf, v)
