# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The PairLogit target on the host (lane/gbdt-learning-to-rank, stage 3): the
device sequence of `gbdt/targets/kernel/pair_logit.mojo`, restated statement for
statement in plain loops, for `gbdt/host/gbdt_oracle_losses.mojo`'s symmetric
fit.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. `prepare_pairs` comes from the GPU-free
`gbdt/data/pairs.mojo`, which the device target imports too, so the endpoint
order and the per-row pair weights are one piece of code.

WHAT IS RESTATED, IN THE DEVICE'S ORDER

  1. The point in row order: `cursor[r]` on the search, `g_cursor[inverse[r]]`
     in the estimator, `inverse` the inverse of the tree's bin order.
  2. Per pair (`pair_logit_pair_kernel`): `diff`, `identical_exp`, the clamped
     `p`, `ftz(w * (1 - p))`, `ftz(w * ftz(p * (1 - p)))`, and the score
     `w * (diff - log(1 + exp))` with the infinity fallback, one value partial
     per 256-pair block through `_halving_fold`.
  3. Per row (`pair_logit_row_kernel`): the endpoint list folded from 0.0 in
     increasing pair index, the loser adding `-(w * direction)`; SEARCH planes
     `[row pair weight, ftz(der)]`, ESTIMATION planes `[ftz(der), ftz(der2)]` at
     the row's bin position; the magnitudes `|plane 0|`, `|plane 1|` per
     256-row block.

The host binding refuses every score function but Cosine under SymmetricTree,
so the search plane 0 here is always the per-row pair weight (the device's
`second_order=False` arm).

THE NEGATIVE CONTROL is the fit's: `GBDT_ORACLE_HOST_SABOTAGE` adds 1.0 to the
walker's lambda in `gbdt_oracle_losses.mojo`, so every PairLogit leaf moves.
"""

from std.math import isfinite

from checks.numerics import ftz, identical_exp, identical_log, identical_mul
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    _deterministic_sum_lanes,
    _halving_fold,
    _partition_stat,
)
from gbdt.data.pairs import MAX_PAIR_COUNT_ON_GPU, PairPrep, prepare_pairs


@fieldwise_init
struct HostPairs(Movable):
    """The fit's pairs and their host layout."""

    var winners: List[Int]
    var losers: List[Int]
    var weights: List[Float32]
    var prep: PairPrep
    #: lane/fam2-gbdt F2 (`IDN_PAIRLOGIT_GROUP`): the GROUP LAYOUT of
    #: `gbdt/targets/kernel/pair_logit_group.mojo`, built by
    #: `host_pair_groups`. `group_layout` false leaves the four lists empty.
    var group_layout: Bool
    #: `n_groups + 1` row offsets
    var group_offsets: List[Int]
    #: the grades in row order
    var grades: List[Float32]
    #: one weight per group (the first row's weight; 1 without weights)
    var group_w: List[Float32]
    #: the per-row pair weights `ftz(w * count)`
    var group_row_weights: List[Float32]
    #: `PairsTotalWeight`: the Float64 sum over groups of the Float32
    #: `w * pairs` the setup kernel stores
    var group_total: Float64

    def n_pairs(self) -> Int:
        return len(self.winners)

    def blocks(self) -> Int:
        if self.group_layout:
            # one value partial (and two magnitudes) per group
            return len(self.group_w)
        var b = (len(self.winners) + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
        return b if b > 0 else 1

    def total_weight(self) -> Float64:
        if self.group_layout:
            return self.group_total
        return self.prep.total


def host_pairs(
    winners: List[UInt32], losers: List[UInt32], weights: List[Float32], n_rows: Int
) raises -> HostPairs:
    var prep = prepare_pairs(winners, losers, weights, n_rows)
    var w = List[Int](capacity=len(winners))
    var l = List[Int](capacity=len(losers))
    for p in range(len(winners)):
        w.append(Int(winners[p]))
        l.append(Int(losers[p]))
    return HostPairs(
        w^, l^, weights.copy(), prep^, False, List[Int](), List[Float32](),
        List[Float32](), List[Float32](), Float64(0.0),
    )


def host_pair_groups(
    group_sizes: List[UInt32],
    targets: List[Float32],
    row_weights: List[Float32],
    n_rows: Int,
) raises -> HostPairs:
    """`make_pairwise_group_buffers` + `pair_logit_group_setup_kernel`
    (`IDN_PAIRLOGIT_GROUP`): the group offsets, the grades, the group weight
    (the first row's weight; `row_weights` empty means 1), the per-row pair
    weights `ftz(w * count)` and the Float64 total of the per-group Float32
    `w * pairs`, with the device setup's refusals in its order."""
    var n_groups = len(group_sizes)
    if n_groups < 1:
        raise Error("Cannot generate pairs for data without groups")
    var offsets = List[Int](capacity=n_groups + 1)
    offsets.append(0)
    var at = 0
    for q in range(n_groups):
        var size = Int(group_sizes[q])
        if size < 1:
            raise Error("PairLogit: query " + String(q) + " has no rows")
        at += size
        offsets.append(at)
    if at != n_rows:
        raise Error(
            "PairLogit: the query sizes cover " + String(at) + " rows of "
            + String(n_rows)
        )
    var grades = List[Float32](capacity=n_rows)
    for r in range(n_rows):
        grades.append(targets[r])
    var group_w = List[Float32](capacity=n_groups)
    var rw = List[Float32](length=n_rows, fill=Float32(0.0))
    var total = Float64(0.0)
    var pair_count = Float64(0.0)
    for g in range(n_groups):
        var begin = offsets[g]
        var end = offsets[g + 1]
        var w = Float32(1.0)
        if len(row_weights) > 0:
            w = row_weights[begin]
        # the kernel's per-thread `endpoints` are integer-valued floats far
        # below 2^24, so their block sum is exact in any order
        var endpoints = 0
        for i in range(begin, end):
            var count = 0
            for j in range(begin, end):
                if targets[j] != targets[i]:
                    count += 1
            rw[i] = ftz(identical_mul(w, Float32(count)))
            endpoints += count
        var pairs_q = Float32(endpoints) * Float32(0.5)
        if pairs_q > Float32(MAX_PAIR_COUNT_ON_GPU):
            raise Error(
                "Too many pairs should be generated for group: "
                + String(Int(pairs_q))
                + " , use max_pairs option to limit generated pair count"
            )
        pair_count += Float64(pairs_q)
        total += Float64(identical_mul(w, pairs_q))
        group_w.append(w)
    if pair_count < 1.0:
        raise Error("Target data is constant. Cannot generate pairs.")
    if not (total > 0.0):
        raise Error(
            "Observation weights should be greater or equal zero. Total"
            " weight should be greater, than zero"
        )
    return HostPairs(
        List[Int](), List[Int](), List[Float32](),
        PairPrep(List[UInt32](), List[UInt32](), List[Float32](), Float64(0.0)),
        True, offsets^, grades^, group_w^, rw^, total,
    )


def _group_values(
    pairs: HostPairs,
    point: List[Float32],
    mut der: List[Float32],
    mut der2: List[Float32],
    mut fv_partials: List[Float32],
):
    """`pair_logit_group_kernel` over every group: per document `i` (thread
    `(i - begin) % 256`, chunks ascending) the fold over the group's
    documents `j` ascending, `der[i] = ftz(sum)`, `der2[i] = ftz(sum)` in
    ROW order, and one value partial per group: each thread's `fv_local`
    accumulated over its documents in chunk order, then the 256-lane
    halving tree (`pinned_block_sum`)."""
    for g in range(len(pairs.group_w)):
        var begin = pairs.group_offsets[g]
        var end = pairs.group_offsets[g + 1]
        var w = pairs.group_w[g]
        var s_fv = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for i in range(begin, end):
            var tid = (i - begin) % GBDT_MSE_BLOCK
            var p_i = point[i]
            var g_i = pairs.grades[i]
            var acc_der = Float32(0.0)
            var acc_der2 = Float32(0.0)
            var fv_local = s_fv[tid]
            for j in range(begin, end):
                var g_j = pairs.grades[j]
                if g_j != g_i:
                    var winner_side = g_i > g_j
                    var diff = p_i - point[j]
                    if not winner_side:
                        diff = point[j] - p_i
                    var exp_diff = identical_exp(diff)
                    var p = Float32(1.0)
                    if isfinite(Float32(1.0) + exp_diff):
                        p = exp_diff / (Float32(1.0) + exp_diff)
                    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
                    var direction = Float32(1.0) - p
                    var scale = ftz(identical_mul(p, Float32(1.0) - p))
                    var wd = ftz(identical_mul(w, direction))
                    if winner_side:
                        acc_der = acc_der + wd
                        var log_exp_val_plus_one = diff
                        if isfinite(Float32(1.0) + exp_diff):
                            log_exp_val_plus_one = identical_log(
                                Float32(1.0) + exp_diff
                            )
                        fv_local = fv_local + identical_mul(
                            w, diff - log_exp_val_plus_one
                        )
                    else:
                        acc_der = acc_der + (-wd)
                    acc_der2 = acc_der2 + ftz(identical_mul(w, scale))
            s_fv[tid] = fv_local
            der[i] = ftz(acc_der)
            der2[i] = ftz(acc_der2)
        fv_partials[g] = _halving_fold(s_fv)


def _group_search_pass(
    pairs: HostPairs,
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_pair_logit_group[False, False, ...]`: planes `[row pair
    weight, der]` in row order, one value partial and two magnitudes per
    GROUP (each thread's `|plane|` accumulated over its documents in chunk
    order, then the halving tree)."""
    var der = List[Float32](length=n_rows, fill=Float32(0.0))
    var der2 = List[Float32](length=n_rows, fill=Float32(0.0))
    _group_values(pairs, cursor, der, der2, fv_partials)
    for g in range(len(pairs.group_w)):
        var begin = pairs.group_offsets[g]
        var end = pairs.group_offsets[g + 1]
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for i in range(begin, end):
            var tid = (i - begin) % GBDT_MSE_BLOCK
            var weight = pairs.group_row_weights[i]
            stats[i] = weight
            stats[n_rows + i] = der[i]
            s_w[tid] = s_w[tid] + abs(weight)
            s_g[tid] = s_g[tid] + abs(der[i])
        mag_partials[2 * g] = _halving_fold(s_w)
        mag_partials[2 * g + 1] = _halving_fold(s_g)


def _pair_values(
    pairs: HostPairs,
    point: List[Float32],
    mut pair_dir: List[Float32],
    mut pair_scale: List[Float32],
    mut fv_partials: List[Float32],
):
    """`pair_logit_pair_kernel` over every pair, in block order."""
    var n_pairs = pairs.n_pairs()
    for b in range(pairs.blocks()):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i >= n_pairs:
                continue
            var w = pairs.weights[i]
            var diff = point[pairs.winners[i]] - point[pairs.losers[i]]
            var exp_diff = identical_exp(diff)
            var p = Float32(1.0)
            if isfinite(Float32(1.0) + exp_diff):
                p = exp_diff / (Float32(1.0) + exp_diff)
            p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
            var direction = Float32(1.0) - p
            var scale = ftz(p * (Float32(1.0) - p))
            pair_dir[i] = ftz(w * direction)
            pair_scale[i] = ftz(w * scale)
            var log_exp_val_plus_one = diff
            if isfinite(Float32(1.0) + exp_diff):
                log_exp_val_plus_one = identical_log(Float32(1.0) + exp_diff)
            s_score[t] = w * (diff - log_exp_val_plus_one)
        fv_partials[b] = _halving_fold(s_score)


def _row_sums(
    pairs: HostPairs, row: Int, pair_dir: List[Float32], pair_scale: List[Float32]
) -> Tuple[Float32, Float32]:
    """`pair_logit_row_kernel`'s fold for one row."""
    var acc_der = Float32(0.0)
    var acc_der2 = Float32(0.0)
    for k in range(Int(pairs.prep.offsets[row]), Int(pairs.prep.offsets[row + 1])):
        var code = pairs.prep.codes[k]
        var p = Int(code >> UInt32(1))
        if (code & UInt32(1)) != UInt32(0):
            acc_der = acc_der + (-pair_dir[p])
        else:
            acc_der = acc_der + pair_dir[p]
        acc_der2 = acc_der2 + pair_scale[p]
    return (ftz(acc_der), ftz(acc_der2))


def pair_logit_search_pass(
    pairs: HostPairs,
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_pair_logit_with[False, False]` with the value and the
    magnitudes: planes `[row pair weight, der]` in row order."""
    if pairs.group_layout:
        _group_search_pass(pairs, cursor, n_rows, stats, fv_partials, mag_partials)
        return
    var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    _pair_values(pairs, cursor, pair_dir, pair_scale, fv_partials)
    var row_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(row_blocks):
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var r = b * GBDT_MSE_BLOCK + t
            if r >= n_rows:
                continue
            var sums = _row_sums(pairs, r, pair_dir, pair_scale)
            var weight = pairs.prep.row_weights[r]
            stats[r] = weight
            stats[n_rows + r] = sums[0]
            s_w[t] = abs(weight)
            s_g[t] = abs(sums[0])
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def pair_logit_value(pairs: HostPairs, cursor: List[Float32]) -> List[Float32]:
    """The final learn-loss pass's per-256-pair value partials (the caller
    folds them with `_deterministic_sum_lanes`). On the group layout: one
    partial per group."""
    if pairs.group_layout:
        var n = len(cursor)
        var g_der = List[Float32](length=n, fill=Float32(0.0))
        var g_der2 = List[Float32](length=n, fill=Float32(0.0))
        var g_fv = List[Float32](length=pairs.blocks(), fill=Float32(0.0))
        _group_values(pairs, cursor, g_der, g_der2, g_fv)
        return g_fv^
    var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var fv = List[Float32](length=pairs.blocks(), fill=Float32(0.0))
    _pair_values(pairs, cursor, pair_dir, pair_scale, fv)
    return fv^


def pair_logit_eval(
    pairs: HostPairs,
    g_cursor: List[Float32],
    row_index: List[Int],
    offsets_leaf: List[Int],
    sizes_leaf: List[Int],
    n_rows: Int,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
):
    """`BinOptimizedOracle.write_value_and_first_derivatives`' pairwise arm:
    the point read back to row order through the inverse bin order, the planes
    `[der, der2]` at each row's bin position, the per-leaf partition stats, the
    Hessian plus lambda, and the host Float32 fold of the pair-block value
    partials."""
    var inverse = List[Int](length=n_rows, fill=0)
    for pos in range(n_rows):
        inverse[row_index[pos]] = pos
    var point = List[Float32](length=n_rows, fill=Float32(0.0))
    for r in range(n_rows):
        point[r] = g_cursor[inverse[r]]
    var fv = List[Float32](length=pairs.blocks(), fill=Float32(0.0))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    if pairs.group_layout:
        # `launch_pair_logit_group[True, False, ...]` (or the
        # `PAIRLOGIT_EST_REUSE` scatter of the same values): `[der, der2]`
        # at each row's bin position, one value partial per group
        var g_der = List[Float32](length=n_rows, fill=Float32(0.0))
        var g_der2 = List[Float32](length=n_rows, fill=Float32(0.0))
        _group_values(pairs, point, g_der, g_der2, fv)
        for r in range(n_rows):
            var g_dst = inverse[r]
            stats[g_dst] = g_der[r]
            stats[n_rows + g_dst] = g_der2[r]
    else:
        var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
        var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
        _pair_values(pairs, point, pair_dir, pair_scale, fv)
        for r in range(n_rows):
            var sums = _row_sums(pairs, r, pair_dir, pair_scale)
            var dst = inverse[r]
            stats[dst] = sums[0]
            stats[n_rows + dst] = sums[1]
    gradient.clear()
    cached_der2.clear()
    for leaf in range(len(sizes_leaf)):
        gradient.append(
            Float64(_partition_stat(stats, n_rows, 0, offsets_leaf[leaf], sizes_leaf[leaf]))
        )
        cached_der2.append(
            Float64(_partition_stat(stats, n_rows, 1, offsets_leaf[leaf], sizes_leaf[leaf]))
            + lambda_reg
        )
    # lane cpu4-gbdt: the oracle's value fold is the device's
    # `deterministic_sum_lanes_kernel[1]` order (was an ascending chain)
    value = Float64(_deterministic_sum_lanes(fv, 1, pairs.blocks())[0])
