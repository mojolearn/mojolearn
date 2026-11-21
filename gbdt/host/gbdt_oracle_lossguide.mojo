# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Lossguide arms of the GradientBoosting fit on the host, for the
gbdt-lossguide lane (workstream E batch 3, 2026-09-14).

HOST ONLY. No `max.gpu`, `std.gpu`, `DeviceContext` or kernel module is
imported. The GPU keeps ONE non-symmetric driver for Depthwise and Lossguide
(`gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`,
`fit_non_symmetric_tree`) and a separate file for the Lossguide selection
(`greedy_search_helper_lossguide.mojo`); the host keeps the same shape. The
shared driver is `gbdt/host/gbdt_oracle_depthwise.mojo`, which imports this
file, and this file imports nothing from it.

WHAT LOSSGUIDE ADDS OVER DEPTHWISE, AND WHERE EACH PIECE IS MIRRORED

  1. The selection, `find_best_leaf_to_split` and `select_leaves_to_split`
     (`greedy_search_helper_lossguide.mojo:551-669`): an ARGMIN over every
     leaf with a defined best split, seeded at `Float32.MAX`, strict `<` so
     the lowest leaf id keeps a tie, and NO sign test.
  2. The score, `compute_optimal_split_kernel[SCORE_FUNCTION_L2]`
     (`kernel/compute_scores.mojo:553-610`, the scan at `:346-500`), which
     the driver launches for NewtonL2 as for L2
     (`greedy_search_helper_depthwise.mojo:1694-1728`): `_add_leaf[L2]`
     (`compute_scores.mojo:80-86`) is `add_leaf_l2` below. The leafwise scan
     and argmax are shared with Depthwise and live in the driver.
  3. The search planes under NewtonL2, `cross_entropy_kernel[True, False,
     second_der_as_weights=True]` (`gbdt/targets/kernel/pointwise_targets.
     mojo:947-1054`, launched at `doc_parallel_boosting.mojo:1531-1543`):
     plane 0 is `ftz(weight * p * (1 - p))`, plane 1 `ftz(weight * (c - p))`,
     and the weight magnitude bounds plane 0 AS STORED.

The score function default is the policy's: `GradientBoosting` sends
NewtonL2 for Lossguide (`python/mojolearn/ensemble.py`, their
`catboost_options.cpp:980-991`), which is the lane's configuration.
"""
from checks.numerics import ftz
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    _gbdt_units,
    _cross_entropy_row,
    _halving_fold,
)


def lossguide_find_best_leaf(
    defined: List[Bool], gains: List[Float32]
) -> Int:
    """`find_best_leaf_to_split` (`greedy_search_helper_lossguide.mojo:
    551-640`): the lowest-gain leaf among the defined ones (the stored gain
    is THEIR sign, lower is better), strict `<` from `Float32.MAX`, so the
    first leaf keeps an exact tie and a NaN gain never wins. -1 when no leaf
    is defined."""
    var best_leaf = -1
    var best_gain = Float32.MAX
    for i in range(len(defined)):  # driver-loop: over one tree's leaves (a compare each)
        if defined[i]:
            if gains[i] < best_gain:
                best_gain = gains[i]
                best_leaf = i
    return best_leaf


def lossguide_select_leaves_to_split(
    defined: List[Bool], gains: List[Float32]
) -> List[Int]:
    """`select_leaves_to_split` (`greedy_search_helper_lossguide.mojo:
    643-669`): `[best]` or `[]`."""
    var out = List[Int]()
    var best = lossguide_find_best_leaf(defined, gains)
    if best >= 0:
        out.append(best)
    return out^


def add_leaf_l2(
    sum: Float32, weight: Float32, lambda_l2: Float32, mut score: Float32
):
    """`_add_leaf[SCORE_FUNCTION_L2, normalize=False, pin_mul_add=True]`
    (`compute_scores.mojo:80-84`): a leaf below the 1e-20 weight adds
    nothing; otherwise the flushed square over the regularized weight."""
    if weight > Float32(1e-20):
        var num = ftz(sum * sum)
        score = ftz(score + ftz(num / (weight + lambda_l2)))


def logloss_search_pass_newton(
    targets: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    border: Float32,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
    weights: List[Float32] = List[Float32](),
):
    """`launch_approximate[False, True]` on Logloss with `compute_fv` and
    `compute_magnitudes` set (`doc_parallel_boosting.mojo:1531-1543`,
    `pointwise_targets.mojo:1003-1054`): plane 0 `ftz(weight * scale)`,
    plane 1 `ftz(weight * direction)`, one score partial and two magnitude
    partials (|plane 0| and |weight * direction| unflushed) per 256-thread
    block through the halving tree. Out-of-range threads add 0.0. `weights`
    empty is the unit weight, else one weight per row (binary class
    weights)."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var has_weights = len(weights) > 0
    # THE SPLIT (lane gbdt-host-par), the unit of `_logloss_search_pass`:
    # one 256-thread device block owns its rows' two planes and its three
    # partials, and its halving trees are the kernel's. Blocks go round
    # robin to `_gbdt_units` tasks; a task holds three block slabs.
    var units = _gbdt_units(blocks, 16 * n_rows)
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    # an empty `weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm border, imm has_weights, imm targets_p, imm cursor_p, imm weights_p, imm stats_p, imm fv_p, imm mag_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var weight = Float32(1.0)
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slabs are reused)
                s_score[t] = Float32(0.0)
                s_w[t] = Float32(0.0)
                s_g[t] = Float32(0.0)
                if i < n_rows:
                    if has_weights:
                        weight = weights_p[i]
                    var r = _cross_entropy_row(targets_p[i], cursor_p[i], border, weight)
                    var plane0 = r.weighted_scale
                    stats_p[i] = plane0
                    stats_p[n_rows + i] = r.weighted_direction
                    s_score[t] = r.score
                    s_w[t] = abs(plane0)
                    s_g[t] = abs(r.weighted_direction_raw)
            fv_p[b] = _halving_fold(s_score)
            mag_p[2 * b] = _halving_fold(s_w)
            mag_p[2 * b + 1] = _halving_fold(s_g)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(targets)
    _ = len(cursor)
    _ = len(weights)
    _ = len(stats)
    _ = len(fv_partials)
    _ = len(mag_partials)
