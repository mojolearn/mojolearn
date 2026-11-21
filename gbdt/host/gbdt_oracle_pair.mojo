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
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, HostU32Ptr, host_predict_chunk
from gbdt.host.gbdt_oracle import (
    GBDT_MSE_BLOCK,
    GbdtIntPtr,
    _gather_rows,
    _gbdt_units,
    _deterministic_sum_lanes,
    _halving_fold,
    _partition_stats_level,
    GBDT_PARTITION_CHUNKS,
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
    # once per fit: the pair ends widened, one statement per pair, the
    # pairs in `_gbdt_units` contiguous ranges
    var n_pairs = len(winners)
    var w = List[Int](length=n_pairs, fill=0)
    var l = List[Int](length=n_pairs, fill=0)
    var units = _gbdt_units(n_pairs, 2 * n_pairs)
    var chunk = host_predict_chunk(n_pairs, units)
    var win_p = rebind[HostU32Ptr](winners.unsafe_ptr())
    var lose_p = rebind[HostU32Ptr](losers.unsafe_ptr())
    var w_p = rebind[GbdtIntPtr](w.unsafe_ptr())
    var l_p = rebind[GbdtIntPtr](l.unsafe_ptr())

    def widen(unit: Int) {imm chunk, imm n_pairs, imm win_p, imm lose_p, imm w_p, imm l_p}:
        for p in range(unit * chunk, min(n_pairs, (unit + 1) * chunk)):
            w_p[p] = Int(win_p[p])
            l_p[p] = Int(lose_p[p])

    host_parallelize(widen, units)
    _ = len(winners)
    _ = len(losers)
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
    for r in range(n_rows):  # once per fit: the grades' copy
        grades.append(targets[r])
    var group_w = List[Float32](capacity=n_groups)
    var rw = List[Float32](length=n_rows, fill=Float32(0.0))
    var total = Float64(0.0)
    var pair_count = Float64(0.0)
    # THE SPLIT (lane gbdt-host-par): the unit is one GROUP, as the setup
    # kernel's launch: its documents' counts (integers), their `ftz(w *
    # count)` and the group's endpoint count read and write the group's own
    # rows. Groups go round robin to `_gbdt_units` tasks over the documents'
    # squared group sizes. The refusals and the two Float64 totals stay one
    # chain over the groups ascending, after the join.
    var has_row_weights = len(row_weights) > 0
    var group_endpoints = List[Int](length=n_groups, fill=0)
    var units = _gbdt_units(n_groups, 2 * n_rows * max(1, n_rows // n_groups))
    var off_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    # an empty `row_weights` is never read (`has_row_weights` guards it)
    var roww_p = rebind[HostF32Ptr](row_weights.unsafe_ptr())
    var rw_p = rebind[HostF32Ptr](rw.unsafe_ptr())
    var ends_p = rebind[GbdtIntPtr](group_endpoints.unsafe_ptr())

    def count_groups(unit: Int) {imm units, imm n_groups, imm has_row_weights, imm off_p, imm targets_p, imm roww_p, imm rw_p, imm ends_p}:
        var g = unit
        while g < n_groups:
            var begin = off_p[g]
            var end = off_p[g + 1]
            var w = Float32(1.0)
            if has_row_weights:
                w = roww_p[begin]
            var endpoints = 0
            for i in range(begin, end):
                var count = 0
                for j in range(begin, end):
                    if targets_p[j] != targets_p[i]:
                        count += 1
                rw_p[i] = ftz(identical_mul(w, Float32(count)))
                endpoints += count
            ends_p[g] = endpoints
            g += units

    host_parallelize(count_groups, units)
    _ = len(targets)
    _ = len(rw)
    for g in range(n_groups):  # driver-loop: over the groups (refusals and totals in order)
        var begin = offsets[g]
        var w = Float32(1.0)
        if has_row_weights:
            w = row_weights[begin]
        # the kernel's per-thread `endpoints` are integer-valued floats far
        # below 2^24, so their block sum is exact in any order
        var endpoints = group_endpoints[g]
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


# Tried 2026-10-08 (MOJOLEARN_TREES_T29_VERSIONED, the 'versioned' arm of T29, run ge123e6f9): the generated V1 PairLogit
# objective (gbdt/targets/kernel/tree_t29_pair.mojo + tree_t29_units.mojo, host _group_values_t29); NV/AMD gbdt-rank-pairlogit
# istella 3.43x/3.76x SLOWER -> deleted (T29 'on' stays). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.


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
    # Tried 2026-10-08 (MOJOLEARN_TREES_T29_VERSIONED, the 'versioned' arm of T29, run ge123e6f9): the generated V1 PairLogit
    # objective (gbdt/targets/kernel/tree_t29_pair.mojo + tree_t29_units.mojo, host _group_values_t29); NV/AMD gbdt-rank-pairlogit
    # istella 3.43x/3.76x SLOWER -> deleted (T29 'on' stays). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
    # THE SPLIT (lane gbdt-host-par): the unit is one GROUP, one launch of
    # the kernel: its documents' folds over the group's documents ascending,
    # their `der` / `der2` rows and the group's 256 value lanes and halving
    # tree are its own. Groups go round robin to `_gbdt_units` tasks over
    # the documents' squared group sizes; a task holds one lane slab.
    var n_groups = len(pairs.group_w)
    var n_docs = len(pairs.grades)
    var units = _gbdt_units(n_groups, 40 * n_docs * max(1, n_docs // max(n_groups, 1)))
    var goff_p = rebind[GbdtIntPtr](pairs.group_offsets.unsafe_ptr())
    var gw_p = rebind[HostF32Ptr](pairs.group_w.unsafe_ptr())
    var grades_p = rebind[HostF32Ptr](pairs.grades.unsafe_ptr())
    var point_p = rebind[HostF32Ptr](point.unsafe_ptr())
    var der_p = rebind[HostF32Ptr](der.unsafe_ptr())
    var der2_p = rebind[HostF32Ptr](der2.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())

    def group_folds(unit: Int) {imm units, imm n_groups, imm goff_p, imm gw_p, imm grades_p, imm point_p, imm der_p, imm der2_p, imm fv_p}:
        var s_fv = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var g = unit
        while g < n_groups:
            var begin = goff_p[g]
            var end = goff_p[g + 1]
            var w = gw_p[g]
            # every lane starts a group at 0.0 (the slab is reused)
            for t in range(GBDT_MSE_BLOCK):
                s_fv[t] = Float32(0.0)
            for i in range(begin, end):
                var tid = (i - begin) % GBDT_MSE_BLOCK
                var p_i = point_p[i]
                var g_i = grades_p[i]
                var acc_der = Float32(0.0)
                var acc_der2 = Float32(0.0)
                var fv_local = s_fv[tid]
                for j in range(begin, end):
                    var g_j = grades_p[j]
                    if g_j != g_i:
                        var winner_side = g_i > g_j
                        var diff = p_i - point_p[j]
                        if not winner_side:
                            diff = point_p[j] - p_i
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
                der_p[i] = ftz(acc_der)
                der2_p[i] = ftz(acc_der2)
            fv_p[g] = _halving_fold(s_fv)
            g += units

    host_parallelize(group_folds, units)
    _ = len(point)
    _ = len(der)
    _ = len(der2)
    _ = len(fv_partials)
    _ = pairs.n_pairs()


def _group_search_pass(
    pairs: HostPairs,
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
    second_order: Bool = False,
):
    """Fill grouped row-order pair-weight (or Hessian) and gradient planes.
    One value partial and two magnitudes per
    GROUP (each thread's `|plane|` accumulated over its documents in chunk
    order, then the halving tree)."""
    var der = List[Float32](length=n_rows, fill=Float32(0.0))
    var der2 = List[Float32](length=n_rows, fill=Float32(0.0))
    _group_values(pairs, cursor, der, der2, fv_partials)
    # Tried 2026-10-08 (MOJOLEARN_TREES_T29_VERSIONED, the 'versioned' arm of T29, run ge123e6f9): the generated V1 PairLogit
    # objective (gbdt/targets/kernel/tree_t29_pair.mojo + tree_t29_units.mojo, host _group_values_t29); NV/AMD gbdt-rank-pairlogit
    # istella 3.43x/3.76x SLOWER -> deleted (T29 'on' stays). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
    # THE SPLIT (lane gbdt-host-par): one GROUP per unit, as above: its
    # rows' two planes and its two magnitude lanes' halving trees.
    var n_groups = len(pairs.group_w)
    var units = _gbdt_units(n_groups, 8 * n_rows)
    var goff_p = rebind[GbdtIntPtr](pairs.group_offsets.unsafe_ptr())
    # the group row weights are read only without `second_order`
    var grw_p = rebind[HostF32Ptr](pairs.group_row_weights.unsafe_ptr())
    var der_p = rebind[HostF32Ptr](der.unsafe_ptr())
    var der2_p = rebind[HostF32Ptr](der2.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())

    def group_planes(unit: Int) {imm units, imm n_groups, imm n_rows, imm second_order, imm goff_p, imm grw_p, imm der_p, imm der2_p, imm stats_p, imm mag_p}:
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var g = unit
        while g < n_groups:
            var begin = goff_p[g]
            var end = goff_p[g + 1]
            # every lane starts a group at 0.0 (the slabs are reused)
            for t in range(GBDT_MSE_BLOCK):
                s_w[t] = Float32(0.0)
                s_g[t] = Float32(0.0)
            for i in range(begin, end):
                var tid = (i - begin) % GBDT_MSE_BLOCK
                var weight = der2_p[i] if second_order else grw_p[i]
                stats_p[i] = weight
                stats_p[n_rows + i] = der_p[i]
                s_w[tid] = s_w[tid] + abs(weight)
                s_g[tid] = s_g[tid] + abs(der_p[i])
            mag_p[2 * g] = _halving_fold(s_w)
            mag_p[2 * g + 1] = _halving_fold(s_g)
            g += units

    host_parallelize(group_planes, units)
    _ = len(der)
    _ = len(der2)
    _ = len(stats)
    _ = len(mag_partials)
    _ = pairs.n_pairs()


def _pair_values(
    pairs: HostPairs,
    point: List[Float32],
    mut pair_dir: List[Float32],
    mut pair_scale: List[Float32],
    mut fv_partials: List[Float32],
):
    """`pair_logit_pair_kernel` over every pair, in block order."""
    # THE SPLIT (lane gbdt-host-par): the unit is one 256-pair device
    # block: its pairs' two outputs and its value partial's halving tree.
    # Blocks go round robin to `_gbdt_units` tasks.
    var n_pairs = pairs.n_pairs()
    var blocks = pairs.blocks()
    var units = _gbdt_units(blocks, 40 * n_pairs)
    var weights_p = rebind[HostF32Ptr](pairs.weights.unsafe_ptr())
    var win_p = rebind[GbdtIntPtr](pairs.winners.unsafe_ptr())
    var lose_p = rebind[GbdtIntPtr](pairs.losers.unsafe_ptr())
    var point_p = rebind[HostF32Ptr](point.unsafe_ptr())
    var dir_p = rebind[HostF32Ptr](pair_dir.unsafe_ptr())
    var scale_p = rebind[HostF32Ptr](pair_scale.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())

    def pair_blocks(unit: Int) {imm units, imm blocks, imm n_pairs, imm weights_p, imm win_p, imm lose_p, imm point_p, imm dir_p, imm scale_p, imm fv_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slab is reused)
                s_score[t] = Float32(0.0)
                if i >= n_pairs:
                    continue
                var w = weights_p[i]
                var diff = point_p[win_p[i]] - point_p[lose_p[i]]
                var exp_diff = identical_exp(diff)
                var p = Float32(1.0)
                if isfinite(Float32(1.0) + exp_diff):
                    p = exp_diff / (Float32(1.0) + exp_diff)
                p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
                var direction = Float32(1.0) - p
                var scale = ftz(p * (Float32(1.0) - p))
                dir_p[i] = ftz(w * direction)
                scale_p[i] = ftz(w * scale)
                var log_exp_val_plus_one = diff
                if isfinite(Float32(1.0) + exp_diff):
                    log_exp_val_plus_one = identical_log(Float32(1.0) + exp_diff)
                s_score[t] = w * (diff - log_exp_val_plus_one)
            fv_p[b] = _halving_fold(s_score)
            b += units

    host_parallelize(pair_blocks, units)
    _ = len(point)
    _ = len(pair_dir)
    _ = len(pair_scale)
    _ = len(fv_partials)
    _ = pairs.blocks()


@always_inline
def _row_sums(
    offsets: HostU32Ptr, codes: HostU32Ptr, row: Int, pair_dir: HostF32Ptr,
    pair_scale: HostF32Ptr,
) -> Tuple[Float32, Float32]:
    """`pair_logit_row_kernel`'s fold for one row: its endpoints ascending
    (`PairPrep.offsets` / `codes`), read through a task's pointers."""
    var acc_der = Float32(0.0)
    var acc_der2 = Float32(0.0)
    for k in range(Int(offsets[row]), Int(offsets[row + 1])):
        var code = codes[k]
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
    second_order: Bool = False,
):
    """Fill row-order PairLogit search planes, values and magnitudes."""
    # The optional Newton plane serves the pointwise ranking draft. Compile
    # both generated-group and explicit-pair arms before merging that draft;
    # the default retains the existing greedy first-order behavior.
    if pairs.group_layout:
        _group_search_pass(
            pairs, cursor, n_rows, stats, fv_partials, mag_partials, second_order
        )
        return
    var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
    _pair_values(pairs, cursor, pair_dir, pair_scale, fv_partials)
    var row_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    # THE SPLIT (lane gbdt-host-par): the unit is one 256-row device block
    # of `pair_logit_row_kernel`: each row's fold over its own endpoints
    # ascending, its two planes, and the block's two magnitude halving
    # trees. Blocks go round robin to `_gbdt_units` tasks over the endpoints.
    var units = _gbdt_units(row_blocks, 8 * (n_rows + 2 * pairs.n_pairs()))
    var poff_p = rebind[HostU32Ptr](pairs.prep.offsets.unsafe_ptr())
    var codes_p = rebind[HostU32Ptr](pairs.prep.codes.unsafe_ptr())
    var roww_p = rebind[HostF32Ptr](pairs.prep.row_weights.unsafe_ptr())
    var dir_p = rebind[HostF32Ptr](pair_dir.unsafe_ptr())
    var scale_p = rebind[HostF32Ptr](pair_scale.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())

    def row_blocks_fold(unit: Int) {imm units, imm row_blocks, imm n_rows, imm second_order, imm poff_p, imm codes_p, imm roww_p, imm dir_p, imm scale_p, imm stats_p, imm mag_p}:
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var b = unit
        while b < row_blocks:
            for t in range(GBDT_MSE_BLOCK):
                var r = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slabs are reused)
                s_w[t] = Float32(0.0)
                s_g[t] = Float32(0.0)
                if r >= n_rows:
                    continue
                var sums = _row_sums(poff_p, codes_p, r, dir_p, scale_p)
                var weight = sums[1] if second_order else roww_p[r]
                stats_p[r] = weight
                stats_p[n_rows + r] = sums[0]
                s_w[t] = abs(weight)
                s_g[t] = abs(sums[0])
            mag_p[2 * b] = _halving_fold(s_w)
            mag_p[2 * b + 1] = _halving_fold(s_g)
            b += units

    host_parallelize(row_blocks_fold, units)
    _ = len(pair_dir)
    _ = len(pair_scale)
    _ = len(stats)
    _ = len(mag_partials)
    _ = pairs.n_pairs()


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
    # THE SPLITS (lane gbdt-host-par): moves, one statement per row.
    # `row_index` is a permutation, so the inverse's writes land on
    # distinct rows from any position range; the point is a gather by it.
    var inverse = List[Int](length=n_rows, fill=0)
    var units = _gbdt_units(n_rows, 2 * n_rows)
    var chunk = host_predict_chunk(n_rows, units)
    var row_p = rebind[GbdtIntPtr](row_index.unsafe_ptr())
    var inv_p = rebind[GbdtIntPtr](inverse.unsafe_ptr())

    def invert(unit: Int) {imm chunk, imm n_rows, imm row_p, imm inv_p}:
        for pos in range(unit * chunk, min(n_rows, (unit + 1) * chunk)):
            inv_p[row_p[pos]] = pos

    host_parallelize(invert, units)
    _ = len(row_index)
    var point = List[Float32](length=n_rows, fill=Float32(0.0))
    _gather_rows(g_cursor, inverse, n_rows, point)
    var fv = List[Float32](length=pairs.blocks(), fill=Float32(0.0))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    if pairs.group_layout:
        # `launch_pair_logit_group[True, False, ...]` (or the
        # `PAIRLOGIT_EST_REUSE` scatter of the same values): `[der, der2]`
        # at each row's bin position, one value partial per group
        var g_der = List[Float32](length=n_rows, fill=Float32(0.0))
        var g_der2 = List[Float32](length=n_rows, fill=Float32(0.0))
        _group_values(pairs, point, g_der, g_der2, fv)
        var gd_p = rebind[HostF32Ptr](g_der.unsafe_ptr())
        var gd2_p = rebind[HostF32Ptr](g_der2.unsafe_ptr())
        var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())

        def scatter(unit: Int) {imm chunk, imm n_rows, imm inv_p, imm gd_p, imm gd2_p, imm stats_p}:
            for r in range(unit * chunk, min(n_rows, (unit + 1) * chunk)):
                var g_dst = inv_p[r]
                stats_p[g_dst] = gd_p[r]
                stats_p[n_rows + g_dst] = gd2_p[r]

        host_parallelize(scatter, units)
        _ = len(g_der)
        _ = len(g_der2)
    else:
        var pair_dir = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
        var pair_scale = List[Float32](length=pairs.n_pairs(), fill=Float32(0.0))
        _pair_values(pairs, point, pair_dir, pair_scale, fv)
        # one row's endpoint fold per statement, the rows over the endpoints
        var r_units = _gbdt_units(n_rows, 8 * (n_rows + 2 * pairs.n_pairs()))
        var r_chunk = host_predict_chunk(n_rows, r_units)
        var poff_p = rebind[HostU32Ptr](pairs.prep.offsets.unsafe_ptr())
        var codes_p = rebind[HostU32Ptr](pairs.prep.codes.unsafe_ptr())
        var dir_p = rebind[HostF32Ptr](pair_dir.unsafe_ptr())
        var scale_p = rebind[HostF32Ptr](pair_scale.unsafe_ptr())
        var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())

        def row_folds(unit: Int) {imm r_chunk, imm n_rows, imm inv_p, imm poff_p, imm codes_p, imm dir_p, imm scale_p, imm stats_p}:
            for r in range(unit * r_chunk, min(n_rows, (unit + 1) * r_chunk)):
                var sums = _row_sums(poff_p, codes_p, r, dir_p, scale_p)
                var dst = inv_p[r]
                stats_p[dst] = sums[0]
                stats_p[n_rows + dst] = sums[1]

        host_parallelize(row_folds, r_units)
        _ = len(pair_dir)
        _ = len(pair_scale)
    _ = len(inverse)
    gradient.clear()
    cached_der2.clear()
    # the leaves' two sums (`compute_partition_stats`): the (leaf, plane)
    # launches across the cores (lane gbdt-host-par)
    var leaf_sums = _partition_stats_level(
        stats, n_rows, 2, GBDT_PARTITION_CHUNKS, offsets_leaf, sizes_leaf, len(sizes_leaf)
    )
    for leaf in range(len(sizes_leaf)):
        gradient.append(Float64(leaf_sums[2 * leaf]))
        cached_der2.append(Float64(leaf_sums[2 * leaf + 1]) + lambda_reg)
    # lane cpu4-gbdt: the oracle's value fold is the device's
    # `deterministic_sum_lanes_kernel[1]` order (was an ascending chain)
    value = Float64(_deterministic_sum_lanes(fv, 1, pairs.blocks())[0])
