"""Leaf values for a grown tree.

MIRRORS `catboost/cuda/methods/leaves_estimation/`, which in CatBoost is a
whole subsystem: `TLeavesEstimation` with a descent loop, an ordered variant,
exact estimation for some objectives, and per-objective backtracking.

**Only the pointwise Newton step is ported**, which is what their descent
reduces to in one iteration for a pointwise objective with a diagonal
Hessian:

    value = +sum(der) / (sum(weight) + l2)

followed by their `RegularizeImpl`. The sign is `+`, not `-`; see the SIGN
CONVENTION block in the kernel, which this line used to contradict. NOT ported, and named so nobody assumes
otherwise: `leaf_estimation_iterations > 1`, ordered boosting's separate
estimation, exact estimation for MAE and quantile, and the backtracking line
search. Those change the VALUE a leaf gets and none of them change the tree
structure, which is what this repository has been building.

Their `leaf_estimation_backtracking` default is also not here. A port of the
descent loop belongs beside this the day a non-Newton objective does.

NOT PORTED, and it is a real one: `MakeZeroAverage`
(`doc_parallel_leaves_estimator.cpp:25-37`) shifts every leaf by
`-sum(point) / count` after estimation, so the tree's leaf values average to
zero. It is a CROSS-LEAF reduction and this kernel is one thread per leaf,
so it cannot go here; it needs a second pass. It is off by default
(`MakeZeroAverage = false`, `leaves_estimation_config.h:14`) and CatBoost
turns it on only for the loss functions that need a bias-free tree.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


comptime LEAF_BLOCK = 256

#: `TLeavesEstimationConfig::MinLeafWeight`, `leaves_estimation_config.h:11`.
#: NOT a user option: `CreateLeavesEstimationConfig` passes the literal
#: `1e-20` for it on every path (`leaves_estimation_config.h:60`), so it is a
#: constant here rather than a kernel argument.
comptime MIN_LEAF_WEIGHT = Float32(1e-20)


def compute_leaf_values_kernel(
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    n_leaves_in: Int32,
    l2: Float32,
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
    """
    var stat_count = Int(stat_count_in)
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return

    var w = part_stats.unsafe_load(leaf * stat_count)
    var g = part_stats.unsafe_load(leaf * stat_count + 1)

    # `w > 1e-20 ? ... : 0.0` -- `greedy_search_helper.cpp:646`. Ours guarded
    # on `w <= 0`, which divides in the window `0 < w <= 1e-20` where theirs
    # returns 0.
    if w <= Float32(1e-20):
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
    # `Gradient[i] / (Hessian[i] + 1e-20f)` -- `descent_helpers.cpp:87`.
    var v = g / (w + l2 + Float32(1e-20))

    # ======================= RegularizeImpl =======================
    #     for (size_t bin = 0; bin < binWeights.size(); ++bin) {
    #         if (binWeights[bin] < config.MinLeafWeight) {
    #             for (ui32 dim = 0; dim < approxDim; ++dim) {
    #                 (*point)[bin * approxDim + dim] = 0;
    #             }
    #         }
    #     }
    # -- `leaves_estimation/oracle_interface.h:43-53`.
    #
    # THIS IS THE ONLY THING CATBOOST DOES TO A LEAF VALUE AFTER THE STEP.
    # It ZEROES an underweight leaf; it does not bound a well-supported one.
    # What stood here instead was a symmetric clamp to a `max_leaf_value`
    # parameter, which CatBoost does not have --
    # `grep -rn "max_leaf_value\|MaxLeafValue" catboost/` returns nothing --
    # and the callers were passing 1e6 and 1e30 for it, two different
    # invented values for the same invented knob. A clamp changes the value
    # of exactly the leaves whose step is largest, which are the leaves that
    # carry the tree.
    #
    # Note the strict `<`: a leaf at exactly `MinLeafWeight` survives here
    # and is zeroed by the `w <= 1e-20` guard above, which is theirs too and
    # uses the other comparison. Both are copied as written.
    # ==============================================================
    if w < MIN_LEAF_WEIGHT:
        v = Float32(0.0)

    out_values.unsafe_store(leaf, v)
