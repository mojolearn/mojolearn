# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Leaf values for a grown tree.

MIRRORS `catboost/cuda/methods/leaves_estimation/`, which in CatBoost is a
whole subsystem: `TLeavesEstimation` with a descent loop, an ordered variant,
exact estimation for some objectives, and per-objective backtracking.

**Only the pointwise Newton step is ported**, which is what their whole
subsystem reduces to at `leaf_estimation_iterations = 1`, which is RMSE's
default (`catboost_options.cpp:61`).

## Which of their two leaf values this kernel is

CatBoost computes a leaf value TWICE and the second one wins.

1. The structure search fills one in as it terminates:
   `w > 1e-20 ? stats[...] / (w + Options.L2Reg) : 0.0`
   -- `greedy_search_helper.cpp:646-647`. No epsilon in the denominator.
2. The boosting loop then calls the estimator whenever
   `NeedEstimation()` is true, and that is
   `LeavesEstimationMethod != Simple` (`greedy_subsets_searcher.h:67-69`),
   which under RMSE's default of Newton is TRUE. `UpdateLeaves` overwrites
   what the search produced (`doc_parallel_leaves_estimator.cpp:39`).

**This kernel computes the second one**, even though it is wired where the
first one sits. The estimator starts from a zero point
(`doc_parallel_leaves_estimator.cpp:10`), takes the `Iterations == 1`
shortcut (`descent_helpers.cpp:149-154`), and moves one full step along

    MoveDirection[i] = Hessian[i] > 0 ? Gradient[i] / (Hessian[i] + 1e-20f) : 0

-- `descent_helpers.cpp:87`, where `Gradient` is `sum(der)` over the leaf and
`Hessian` is `sum(der2)` over the leaf with `lambda` already added
(`pointwise_oracle.cpp:86-89`). Then `Oracle.Regularize` runs
`RegularizeImpl` (`descent_helpers.cpp:152`).

The two formulas differ only by `1e-20` in the denominator and by which
quantity guards the division. For MSE they agree to every representable
digit, because `der2` IS the weight there: `TRmseTarget::Der2` returns
`1.0f` (`pointwise_targets.cu:188-190`) and the kernel stores
`weight * Der2` (`:265-267`).

That agreement is why ONE kernel can stand for both, and it is a fact about
MSE rather than a general one. The day an objective arrives whose `der2` is
not its weight, the estimator has to become a second pass over the final
partition, and this note is where the reader should start.

NOT ported, and named so nobody assumes otherwise: `leaf_estimation_iterations
> 1` and with it the whole backtracking walker, ordered boosting's separate
estimation, and exact estimation for MAE and quantile. Those change the VALUE
a leaf gets and none of them change the tree structure.

NOT PORTED, and it is a real one: `MakeZeroAverage`
(`doc_parallel_leaves_estimator.cpp:25-37`) shifts every leaf by
`-sum(point) / count` after estimation, so the tree's leaf values average to
zero. It is a CROSS-LEAF reduction and this kernel is one thread per leaf,
so it cannot go here; it needs a second pass. CatBoost turns it on only for
PairLogit and YetiRank (`NeedZeroAverage`, `train_template.h:29-40`), so it
is off for every objective this port can reach.
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

    An EMPTY leaf reaches 0.0 the way theirs does, by dividing `0` by
    `0 + l2 + 1e-20` and then being zeroed again by `RegularizeImpl`, rather
    than by an early return of our own. A tree at depth 8 over 4,096 rows has
    empty leaves as a matter of course (46 of 64 were populated in this
    repository's own check), so this is the common case and not an edge case,
    and it is worth taking the branch they take.
    """
    var stat_count = Int(stat_count_in)
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return

    var w = part_stats.unsafe_load(leaf * stat_count)
    var g = part_stats.unsafe_load(leaf * stat_count + 1)

    # `(*Der2AtPoint)[i] += lambda` -- `pointwise_oracle.cpp:86-89`. The
    # regularizer is folded into the HESSIAN before the direction is taken,
    # not into the denominator at the point of division, and that is what the
    # guard below then tests.
    var hessian = w + l2

    # `MoveDirection[i] = Hessian[i] > 0 ? ... : 0` --
    # `descent_helpers.cpp:87`. THE GUARD IS ON THE HESSIAN, NOT THE WEIGHT.
    # What stood here was `w <= 1e-20 -> 0`, borrowed from the structure
    # searcher's fallback (`greedy_search_helper.cpp:646`), which is the
    # branch that does NOT run under RMSE's default of Newton. The two
    # disagree exactly where `w <= 1e-20 < w + l2`: theirs divides and then
    # lets `RegularizeImpl` decide, ours returned early and never reached it.
    # With `l2 = 3.0` that window is every leaf whose weight underflows, and
    # both paths end at 0.0 for those, which is why this was invisible.
    if hessian <= Float32(0.0):
        out_values.unsafe_store(leaf, Float32(0.0))
        return

    # ==================== SIGN CONVENTION ====================
    # `+g`, NOT `-g`. This is CatBoost's convention and it is easy to get
    # backwards, so it is written down rather than inferred.
    #
    # `TRmseTarget::Der(t, p)` is `t - p` (`pointwise_targets.cu:184-186`)
    # and the kernel stores `weight * Der` (`:263`), which CatBoost names
    # `direction` in the MSE twin. It is the
    # NEGATIVE loss gradient, already pointing downhill. Their walker then
    # takes `MoveDirection` UNNEGATED and adds one full step of it to a point
    # that starts at zero (`descent_helpers.cpp:69-71`, `:151`).
    #
    # So a leaf's value is `+sum(der) / (sum(der2) + l2)`, the weighted mean
    # residual, and adding it to the prediction moves toward the target.
    #
    # It was `-g` here, ported before any target existed to fix the
    # convention. With CatBoost's `der` that inverts every step: measured on
    # `boosting_check`, the loss GREW by about 1.68x per iteration, from 231
    # to 55839 over twelve trees, instead of falling.
    # =========================================================
    # `Gradient[i] / (Hessian[i] + 1e-20f)` -- `descent_helpers.cpp:87`. The
    # `1e-20` is THEIRS and it is added to the already-regularized Hessian,
    # so it is not a second copy of `MIN_LEAF_WEIGHT` and not a guard: at
    # `l2 = 0` CatBoost has already substituted `1e-20` for the regularizer
    # itself (`catboost_options.cpp:357-359`) and this term is what keeps the
    # division finite in the window below that.
    var v = g / (hessian + Float32(1e-20))

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
    # Note the strict `<`, and note that `binWeights` is the leaf's WEIGHT
    # (`DerCalcer->GetWeights(0)` reduced by `ComputePartitionStats`,
    # `pointwise_oracle.cpp:241-244`), not its Hessian. So this test reads
    # `w`, while the division above is guarded on `w + l2`. The two guards
    # look redundant and are not: theirs is the only one that can zero a leaf
    # whose weight underflows while its Hessian is a healthy `l2`.
    # ==============================================================
    if w < MIN_LEAF_WEIGHT:
        v = Float32(0.0)

    out_values.unsafe_store(leaf, v)
