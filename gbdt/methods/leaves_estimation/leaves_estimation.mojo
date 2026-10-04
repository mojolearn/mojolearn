# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Leaf values for a grown tree.

Reference: `catboost/cuda/methods/leaves_estimation/`, which in CatBoost is a
whole subsystem: `TLeavesEstimation` with a descent loop, an ordered variant,
exact estimation for some objectives, and per-objective backtracking.

**Only the pointwise Newton step is implemented**, which is what their whole
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

NOT implemented, and named so nobody assumes otherwise: `leaf_estimation_iterations
> 1` and with it the whole backtracking walker, ordered boosting's separate
estimation, and exact estimation for MAE and quantile. Those change the VALUE
a leaf gets and none of them change the tree structure.

NOT IN THIS KERNEL: `MakeZeroAverage`
(`doc_parallel_leaves_estimator.cpp:25-37`) shifts every leaf by
`-sum(point) / count` after estimation, so the tree's leaf values average to
zero. It is a CROSS-LEAF reduction and this kernel is one thread per leaf,
so it cannot go here. CatBoost turns it on for PairLogit, PairLogitPairwise,
YetiRank and YetiRankPairwise (`NeedZeroAverage`, `train_template.h:29-40`).
Of those this implementation trains PairLogit, and
`gbdt/methods/doc_parallel_boosting.mojo::_estimate_and_apply` applies the
shift on the host after the walker; the host oracle
(`gbdt/host/gbdt_oracle_losses.mojo`) restates it.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.soft_f64 import (
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_neg,
    sf64_sqrt,
    sf64_sub,
    sf64_to_f32,
)

#: lane/fam2-gbdt F5 (IDENTICAL, every vendor; default ON): the ONE-STEP
#: Newton estimation (`leaf_estimation_iterations == 1`, the default of
#: Logloss and the other single-dimensional pointwise losses) is finished ON
#: THE DEVICE. `_estimate_and_apply` (`gbdt/methods/doc_parallel_boosting.mojo`)
#: used to drain after the evaluation, read the per-leaf sums back, take the
#: walker's one step on the host (`descent_helpers.mojo`: Float64 divide,
#: regularize), upload the leaf values and drain again. Now
#: `newton_one_step_kernel` takes that step from the device sums in
#: soft-float64 (`checks/soft_f64.mojo`, IEEE double in integer arithmetic,
#: the same on every vendor), the cursor add reads its output, and the task
#: drains ONCE. The values are the host walker's bit for bit (correctly
#: rounded double add, divide and narrowing), so the host column is
#: untouched. The regularizer's per-leaf weight is the leaf's row count on
#: an unweighted fit (already on the device); on a weighted fit the host
#: uploads its keep/zero decision per leaf (one float per leaf, from the
#: weight sums the oracle's constructor already read).
#: `-D MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF` (or the master
#: `-D MOJOLEARN_IDN_ALL_OFF`) restores the host walker.
comptime IDN_EST_ONE_STEP_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


#: lane fix-g1-gbdt (audit G1 / F7 "GBDT multiclass leaf solve"; IDENTICAL,
#: every vendor; default ON): the ONE-STEP MultiClass Newton estimation
#: (`leaf_estimation_iterations == 1`, Newton, the blocked Hessian) is
#: finished ON THE DEVICE. The walker used to drain after the first
#: evaluation (gradient readback), drain again after the `numClasses`
#: Hessian rows, build the per-leaf `numClasses x numClasses` system on the
#: host, solve it there (`descent_helpers._blocked_hessian_direction`,
#: `gbdt/lapack/linear_system.mojo`'s Cholesky), then upload the leaf
#: values and drain a third time. Now every row's per-leaf reduction is
#: stashed on the device (`f32_stash_kernel`) and
#: `multiclass_one_step_kernel` (one thread per leaf) does the gradient
#: reconstruction, the lambda diagonal, the Cholesky factor and the two
#: triangular solves, the not-PD fallback (gradient untouched, DEVIATION
#: 74), the narrowing, the walker's fused move, Regularize and the gauge
#: projection, in soft-float64 (`checks/soft_f64.mojo`: correctly rounded
#: add/sub/fma/div/sqrt, the same integer code on every vendor). Each step
#: is the host walker's operation for operation (the host `fma` is the
#: soft `fma`, the host `sqrt` the soft correctly rounded `sqrt`), so the
#: leaf values are the host walker's bit for bit and the host column
#: (`gbdt/host/gbdt_oracle_multiclass.mojo::_estimate_multi`) is untouched.
#: The task drains ONCE. Classes above `MC_ONE_STEP_MAX_CLASSES` (the
#: per-thread system is a local array) keep the host walker.
#: `-D MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF` (or the master
#: `-D MOJOLEARN_IDN_ALL_OFF`) restores the host walker.
comptime IDN_GBDT_MC_ONE_STEP_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: the widest class count `multiclass_one_step_kernel` solves (its
#: per-thread system is `MC_ONE_STEP_MAX_CLASSES^2` doubles of local memory)
comptime MC_ONE_STEP_MAX_CLASSES = 16


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
    # It was `-g` here, implemented before any target existed to fix the
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


def newton_one_step_kernel(
    part_stats: MutPointer[Float32, MutAnyOrigin],
    leaf_sizes: MutPointer[UInt32, MutAnyOrigin],
    lambda_bits: UInt64,
    min_leaf_weight_bits: UInt64,
    n_leaves_in: Int32,
    out_values: MutPointer[Float32, MutAnyOrigin],
    has_mask: Int32,
):
    """`IDN_EST_ONE_STEP_DEVICE`: `TNewtonLikeWalker::Estimate` at
    `Iterations == 1` on the diagonal arm, one thread per leaf, from the
    evaluation's per-leaf sums `part_stats[2 * leaf] = sum(der)`,
    `part_stats[2 * leaf + 1] = sum(der2)`:

        gradient = double(der);  hessian = double(der2) + lambda
        direction = hessian > 0 ? float(gradient / (hessian + 1e-20f)) : 0
        point = float(1.0 * direction + 0.0)       (the walker's fused move)
        point = 0 when double(leaf row count) < MinLeafWeight  (Regularize)

    With `has_mask` (a weighted fit) `out_values[leaf]` arrives holding the
    host's Regularize decision for the leaf, 0.0 = zero it, and is
    overwritten with the value (one buffer in and out, so no two launch
    arguments alias).

    The doubles are their bit patterns in `UInt64`; `lambda_bits` and
    `min_leaf_weight_bits` are the host doubles' bits."""
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return
    var g = sf64_from_f32(part_stats.unsafe_load(2 * leaf))
    var h = sf64_add(
        sf64_from_f32(part_stats.unsafe_load(2 * leaf + 1)), lambda_bits
    )
    var v = Float32(0.0)
    if (not sf64_is_nan(h)) and sf64_gt(h, SF64_ZERO):
        var eps = sf64_from_f32(Float32(1e-20))
        v = sf64_to_f32(sf64_div(g, sf64_add(h, eps)))
    # `fma(1.0, direction, +0.0)`: a zero of either sign lands on +0.0
    if v == Float32(0.0):
        v = Float32(0.0)
    if has_mask != Int32(0):
        if out_values.unsafe_load(leaf) == Float32(0.0):
            v = Float32(0.0)
    else:
        var weight = sf64_from_int(Int(leaf_sizes.unsafe_load(leaf)))
        if sf64_lt(weight, min_leaf_weight_bits):
            v = Float32(0.0)
    out_values.unsafe_store(leaf, v)


def f32_stash_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    dst_offset_in: Int32,
    n_in: Int32,
):
    """`IDN_GBDT_MC_ONE_STEP_DEVICE`: copy `n` floats of a per-leaf reduction
    (the oracle's reused `d_multi_stats`) into their slot of a stash, so the
    next row's reduction can overwrite the source while the stash keeps the
    row for `multiclass_one_step_kernel`. A byte copy: no value changes."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    dst.unsafe_store(Int(dst_offset_in) + i, src.unsafe_load(i))


def multiclass_one_step_kernel(
    grad_stats: MutPointer[Float32, MutAnyOrigin],
    hess_stats: MutPointer[Float32, MutAnyOrigin],
    leaf_sizes: MutPointer[UInt32, MutAnyOrigin],
    lambda_bits: UInt64,
    min_leaf_weight_bits: UInt64,
    n_leaves_in: Int32,
    cursor_dim_in: Int32,
    out_values: MutPointer[Float32, MutAnyOrigin],
    not_pd: MutPointer[UInt32, MutAnyOrigin],
    has_mask: Int32,
):
    """`IDN_GBDT_MC_ONE_STEP_DEVICE`: `TNewtonLikeWalker::Estimate` at
    `Iterations == 1` on the BLOCKED arm (MultiClass, Newton), one thread
    per leaf. `k = cursor_dim + 1` (`SingleBinDim()`).

    Inputs, the oracle's per-leaf float32 reductions at the zero point:
      `grad_stats[leaf * cursor_dim + d]`: `sum(der)` of class plane `d`;
      `hess_stats`: Hessian row `r` (columns `0..r`) at offset
      `n_leaves * r (r + 1) / 2`, `[leaf * (r + 1) + c]`.

    The host walker, step for step (doubles are `UInt64` bit patterns):
      gradient[d] = double(der[d]);  gradient[k-1] = -(0 + der[0] + ...)
                    (`_write_multi_dim_value_and_first_derivatives`)
      sigma[r][c] = double(h[r][c]), + lambda on the diagonal
                    (`_write_blocked_second_derivatives`; only the lower
                    triangle is read by the solve)
      Cholesky + two triangular solves with every `x -= a * b` one `fma`
                    (`solve_linear_system_cholesky`); a leading minor with
                    `s <= 0` stops it and leaves the GRADIENT as the
                    solution (DEVIATION 74), counted in `not_pd[leaf]`
      direction = float(solution);  point = float(fma(1.0, dir, +0.0))
                    (`_move`: a zero of either sign lands on +0.0)
      Regularize: every component 0 when the leaf weight < MinLeafWeight
      out[leaf * cursor_dim + d] = point[d] - point[k-1]  (float32,
                    `make_estimation_result`; computed as the double
                    difference narrowed once, which is the correctly
                    rounded float difference)

    With `has_mask` (a weighted fit) `out_values[leaf * cursor_dim]` arrives
    holding the host's Regularize decision, 0.0 = zero the leaf; each thread
    reads only its own leaf's slot before writing it."""
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return
    var cd = Int(cursor_dim_in)
    var k = cd + 1
    var keep = True
    if has_mask != Int32(0):
        if out_values.unsafe_load(leaf * cd) == Float32(0.0):
            keep = False
    else:
        var weight = sf64_from_int(Int(leaf_sizes.unsafe_load(leaf)))
        if sf64_lt(weight, min_leaf_weight_bits):
            keep = False

    var sol = InlineArray[UInt64, MC_ONE_STEP_MAX_CLASSES](fill=UInt64(0))
    var sigma = InlineArray[
        UInt64, MC_ONE_STEP_MAX_CLASSES * MC_ONE_STEP_MAX_CLASSES
    ](fill=UInt64(0))

    # the gradient, with MultiClass's reconstructed pinned component
    var total = SF64_ZERO
    for d in range(cd):
        var v = sf64_from_f32(grad_stats.unsafe_load(leaf * cd + d))
        sol[d] = v
        total = sf64_add(total, v)
    sol[cd] = sf64_neg(total)

    # the lower triangle of the blocked Hessian, lambda on the diagonal
    for r in range(k):
        var row_base = n_leaves * ((r * (r + 1)) // 2) + leaf * (r + 1)
        for c in range(r + 1):
            var hv = sf64_from_f32(hess_stats.unsafe_load(row_base + c))
            if c == r:
                hv = sf64_add(hv, lambda_bits)
            sigma[r * k + c] = hv

    var info = 0
    if k == 1:
        # `if (target->size() == 1) { (*target)[0] /= (*matrix)[0]; }`
        sol[0] = sf64_div(sol[0], sigma[0])
    else:
        # dpotrf, lower triangle in place
        for j in range(k):
            var s = sigma[j * k + j]
            for kk in range(j):
                var ljk = sigma[j * k + kk]
                s = sf64_fma(sf64_neg(ljk), ljk, s)
            # the host's `s <= 0.0` (false for NaN)
            if (not sf64_is_nan(s)) and (not sf64_gt(s, SF64_ZERO)):
                info = j + 1
                break
            var ljj = sf64_sqrt(s)
            sigma[j * k + j] = ljj
            for i in range(j + 1, k):
                var t = sigma[i * k + j]
                for kk in range(j):
                    t = sf64_fma(sf64_neg(sigma[i * k + kk]), sigma[j * k + kk], t)
                sigma[i * k + j] = sf64_div(t, ljj)
        if info == 0:
            # dpotrs: L y = b, then L^T x = y
            for i in range(k):
                var t = sol[i]
                for kk in range(i):
                    t = sf64_fma(sf64_neg(sigma[i * k + kk]), sol[kk], t)
                sol[i] = sf64_div(t, sigma[i * k + i])
            for ii in range(k):
                var i = k - 1 - ii
                var t = sol[i]
                for kk in range(i + 1, k):
                    t = sf64_fma(sf64_neg(sigma[kk * k + i]), sol[kk], t)
                sol[i] = sf64_div(t, sigma[i * k + i])
    not_pd.unsafe_store(leaf, UInt32(1) if info != 0 else UInt32(0))

    # the pinned component's point (`_move`'s +0.0 normalisation, Regularize)
    var p_last = sf64_to_f32(sol[cd])
    if p_last == Float32(0.0) or not keep:
        p_last = Float32(0.0)
    var p_last_bits = sf64_from_f32(p_last)
    for d in range(cd):
        var p = sf64_to_f32(sol[d])
        if p == Float32(0.0) or not keep:
            p = Float32(0.0)
        out_values.unsafe_store(
            leaf * cd + d,
            sf64_to_f32(sf64_sub(sf64_from_f32(p), p_last_bits)),
        )
