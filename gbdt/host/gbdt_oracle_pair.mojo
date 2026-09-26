# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The PairLogit target on the host (lane/gbdt-learning-to-rank, stage 3): the
device sequence of `gbdt/targets/kernel/pair_logit.mojo`, restated statement for
statement in plain loops, for `gbdt/host/gbdt_oracle_losses.mojo`'s symmetric
fit.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu`, a `DeviceContext` or a
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

from checks.numerics import ftz, identical_exp, identical_log
from gbdt.host.gbdt_oracle import GBDT_MSE_BLOCK, _halving_fold, _partition_stat
from gbdt.data.pairs import PairPrep, prepare_pairs


@fieldwise_init
struct HostPairs(Movable):
    """The fit's pairs and their host layout."""

    var winners: List[Int]
    var losers: List[Int]
    var weights: List[Float32]
    var prep: PairPrep

    def n_pairs(self) -> Int:
        return len(self.winners)

    def blocks(self) -> Int:
        var b = (len(self.winners) + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
        return b if b > 0 else 1


def host_pairs(
    winners: List[UInt32], losers: List[UInt32], weights: List[Float32], n_rows: Int
) raises -> HostPairs:
    var prep = prepare_pairs(winners, losers, weights, n_rows)
    var w = List[Int](capacity=len(winners))
    var l = List[Int](capacity=len(losers))
    for p in range(len(winners)):
        w.append(Int(winners[p]))
        l.append(Int(losers[p]))
    return HostPairs(w^, l^, weights.copy(), prep^)


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
    folds them with `_deterministic_sum_lanes`)."""
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
    var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var fv = List[Float32](length=pairs.blocks(), fill=Float32(0.0))
    _pair_values(pairs, point, pair_dir, pair_scale, fv)
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
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
    var fv32 = Float32(0.0)
    for b in range(pairs.blocks()):
        fv32 += fv[b]
    value = Float64(fv32)
