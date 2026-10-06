# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST Apple (lane apple-fast-pairlogit, the plan's `trees-pairlogit`):
PairLogit's pairs ENUMERATED ON THE DEVICE, one block per query group, the
per-pair logistic gradient and the per-document sums in the same launch.

Compiled under `PAIRLOGIT_GROUP_FUSED` (the FAST + Apple default since the
M3 A/B 2026-10-03; `-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF` opts out);
and under IDENTICAL on every vendor since lane/fam2-gbdt F2
(`IDN_PAIRLOGIT_GROUP`, `gbdt/data/pairs.mojo`; the host column restates it).
`PAIRLOGIT_EST_REUSE` (also the FAST + Apple default, needs the first;
`-D MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF` opts out) lets the leaf estimation's first evaluation reuse the search's
sums, the YetiRank `YETI_EST_REUSE_SEARCH` model.

WHAT MAIN DOES PER ITERATION. The reference's pair list is generated ONCE on
the host (`gbdt/data/pairs.mojo::generate_pairs`: every `first < second` of
a query with different grades, weighted by the group weight), uploaded, and
every target call streams it twice: `pair_logit_pair_kernel` stores
`w * direction` and `w * scale` per pair, `pair_logit_row_kernel` reads each
row's endpoint list and gathers the two stores per endpoint. On Istella-S
that is ~1e8 pairs: the pair buffers alone are gigabytes, and each target
call (one search call plus every estimation evaluation, per tree) moves them
through memory at random.

WHAT THIS DOES. The generated pairs are a function of the grades alone, so
the kernel REBUILDS them from the grades every call: one block per query
group, a thread per document `i`, a loop over the group's documents `j`
read through shared memory tiles. For every `j` with a different grade the
thread evaluates the pair's logistic term (`pair_logit.cu:25-40`, the same
arithmetic as `pair_logit_pair_kernel`, routed exp and log, the same
clamps and `ftz` sites) and folds it into `i`'s `der` and `der2` in
increasing `j`: the winner side adds `w * direction` and the pair's value
term, the loser side subtracts `w * direction`; both add `w * scale`. No
pair list, no per-pair scratch, no endpoint lists: the per-call traffic is
the grades and the point, read once per group tile. The function value is
one partial per GROUP (folded by `deterministic_sum_lanes_kernel` as main
folds the per-256-pair partials) and the plane magnitudes two per group.

BITS. A row's sum takes the `j` order, where main's takes increasing pair
index, and the per-row pair weight is `w * count` where main folds `count`
copies of `w`: bits move against the pair-list path, under FAST and (F2)
under IDENTICAL, where the host oracle moves with it. Every row's and every
group's fold is a fixed sequential order on every vendor.

THE SETUP, once per fit (`launch_pair_logit_group_setup`): the same block
per group copies the grades into row order, takes the group weight from the
first row's weight (`data_providers.cpp:841-844`: the learn weight of the
query's first row, 1 without weights), writes the per-row pair weights
(`InitPairLogit`'s replaced target weights) and leaves the group's pair
count and weighted pair count for the ONE host readback that checks
`MAX_PAIR_COUNT_ON_GPU` per group, the constant-target refusal, and sums
`PairsTotalWeight` in Float64 over the groups.

USER-SUPPLIED PAIRS keep main's path even under the define: a caller's pair
list is not a function of the grades.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from gbdt.apple_fast_tree_experiments import AFT_N07, AFT_N08
from std.math import isfinite
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    ftz,
    identical_mul,
)
from gbdt.data.pairs import IDN_PAIRLOGIT_GROUP
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    pinned_block_sum,
    routed_exp,
    routed_log,
)


def pairlogit_group_fused_for[column: Int]() -> Bool:
    """FAST + Apple default; everything else compiles main's pair-list path.
    Default since the M3 A/B 2026-10-03 (gbdt-rank-pairlogit istellarank,
    n=2, same hash, ndcg10 .71995, map .85455): 4,802 -> 3,760 ms.
    `-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF` restores the pair-list path
    (and turns `PAIRLOGIT_EST_REUSE` off with it); the old
    `-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED` is accepted and changes nothing.
    Under IDENTICAL `MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF` does NOT turn the
    group kernel off (the host column restates it under
    `IDN_PAIRLOGIT_GROUP`): the IDENTICAL before arm is
    `-D MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF` (or MOJOLEARN_IDN_ALL_OFF),
    which moves the device and the host column together (lane/review-fixes)."""
    comptime if not is_defined["MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF"]():
        comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
            return True
    # lane/fam2-gbdt F2: IDENTICAL on every vendor (`IDN_PAIRLOGIT_GROUP`,
    # `gbdt/data/pairs.mojo`); the host column restates this kernel under
    # the same constant. Every product next to an add below is
    # `identical_mul` (the pinned product under IDENTICAL, the plain product
    # under FAST), so no vendor fuses it.
    comptime if IDN_PAIRLOGIT_GROUP:
        return True
    return False


def pairlogit_est_reuse_for[column: Int]() -> Bool:
    """FAST + Apple default: the leaf estimation's evaluation
    at the tree's starting point reuses the search call's per-row sums and
    per-group value partials (scattered to bin order) instead of
    re-enumerating the pairs. Same point, same sample (PairLogit draws
    nothing), so only FAST bits move with the fold. Needs the group kernel
    (the accumulators are its stores), so `..._GROUP_FUSED_OFF` turns it
    off too. Default since the M3 A/B 2026-10-03 (same data, same hash):
    3,765 -> 3,480 ms on top of the group kernel.
    `-D MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF` turns it off alone; the old
    `-D MOJOLEARN_PAIRLOGIT_EST_REUSE` is accepted and changes nothing."""
    comptime if not is_defined["MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF"]():
        return pairlogit_group_fused_for[column]()
    return False


comptime PAIRLOGIT_GROUP_FUSED = pairlogit_group_fused_for[TARGET_COLUMN]()
comptime PAIRLOGIT_EST_REUSE = pairlogit_est_reuse_for[TARGET_COLUMN]()


def pl_pairs_once_for[column: Int]() -> Bool:
    """Lane af-sym-multi (2026-10-03), `-D MOJOLEARN_PL_PAIRS_ONCE` (not in
    `SYM_MULTI_ALL`: a recorded DROP), FAST + Apple, on top of the group kernel:
    a group that fits one block evaluates every unordered pair ONCE. The
    group kernel's thread `i` loops over every `j` of the group, so each
    pair's exp, divide, clamp (and the winner's log) run twice, once from
    each endpoint. The once path runs `floor((size - 1) / 2)` rounds; in
    round `r` thread `i` pairs with `j = (i + r) mod size`, a permutation of
    the threads, so every unordered pair appears in exactly one (round,
    thread) (plus the `size / 2` half round when `size` is even, taken by
    the threads below `size / 2`). The computing thread folds its own side
    into its registers and hands the other side's `der` / `der2` term to
    `j` through a shared slot that only it writes in that round; a barrier
    later `j` folds it in. Half the transcendental work of the dominant
    kernel; the per-document fold order changes (FAST bits move, fixed and
    repeatable), the terms are the same. Groups wider than the block keep
    the chunked loop."""
    comptime if (
        is_defined["MOJOLEARN_PL_PAIRS_ONCE"]()
    ):
        return pairlogit_group_fused_for[column]()
    return False


def pl_group_narrow_for[column: Int]() -> Bool:
    """Lane af-sym-multi, FAST + Apple default since 2026-10-04 (rollback
    `-D MOJOLEARN_PL_GROUP_NARROW_OFF`; the old `-D MOJOLEARN_PL_GROUP_NARROW`
    and `-D MOJOLEARN_SYM_MULTI_ALL` change nothing): the group kernel runs on
    128-thread blocks instead of 256. Istella's queries hold ~103 documents,
    so a 256-thread block leaves four of its eight SIMD groups idle through
    every tile loop and barrier; at 128 the idle half is gone and twice the
    blocks fit a core. A group wider than 128 takes two chunks (the chunked
    loop is width-agnostic). Same terms, same per-document fold order as
    the 256-thread kernel (the order is the `j` order, not the thread
    count), so the bits do not move; the setup kernel and the reuse scatter
    keep their width."""
    comptime if not is_defined["MOJOLEARN_PL_GROUP_NARROW_OFF"]():
        return pairlogit_group_fused_for[column]()
    return False


#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-multi@d2c832da0. `-D MOJOLEARN_PL_PAIRS_ONCE`.
#: apple-fast LEDGER 2026-10-03: DROP symmulti pl-once and
#: pl-both (within +-2.4%, identical quality), old base; recorded loser,
#: OUT of SYM_MULTI_ALL, not in the A/B table.
comptime PL_PAIRS_ONCE = pl_pairs_once_for[TARGET_COLUMN]()
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-multi@d2c832da0; the laptop built the .so (Metal side
#: unchecked). `-D MOJOLEARN_PL_GROUP_NARROW` (or SYM_MULTI_ALL) until
#: 2026-10-04.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04):
#: gbdt-rank-pairlogit istella 3095.83 -> 3040.78 ms (-1.8%, tag
#: rab4-symmulti) and 3095.91 -> 3038.19 ms (-1.9%, tag rab7-plgroupnarrow);
#: map 0.854545, ndcg10 0.719953 and output digest identical. KEEP: the FAST
#: + Apple default; rollback -D MOJOLEARN_PL_GROUP_NARROW_OFF. SYM_MULTI_ALL
#: (yetirank +0.1%) stays a NEUTRAL record and no longer selects anything.
comptime PL_GROUP_NARROW = pl_group_narrow_for[TARGET_COLUMN]()
#: the group kernel's block: 128 under `PL_GROUP_NARROW`, else `PLG_THREADS`
#: (`MSE_BLOCK_SIZE`, 256)
# N07: two simdgroups per query block trade more serial query tiles for
# smaller register/shared reservations. Every document and unlike-grade pair
# remains present; queries wider than the block keep the complete tile loop.
# New64-lane arm, not the historical256-vs128 experiment. No evidence.
comptime PLG_LAUNCH_THREADS = 64 if AFT_N07 else (128 if PL_GROUP_NARROW else MSE_BLOCK_SIZE)

#: threads per group block, one document per thread per chunk; the value
#: and magnitude partials use `pinned_block_sum` at this width
comptime PLG_THREADS = MSE_BLOCK_SIZE


def pair_logit_group_setup_kernel(
    targets: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    group_offsets: MutPointer[UInt32, MutAnyOrigin],
    grades: MutPointer[Float32, MutAnyOrigin],
    row_weights: MutPointer[Float32, MutAnyOrigin],
    group_w: MutPointer[Float32, MutAnyOrigin],
    group_pairs: MutPointer[Float32, MutAnyOrigin],
    group_wsum: MutPointer[Float32, MutAnyOrigin],
):
    """One block per group: `grades[i] = targets[i]`, `group_w[g]` = the
    first row's weight, `row_weights[i]` and `weights[i]` = `group_w *`
    (the group's documents whose grade differs from `i`'s), `group_pairs[g]`
    the group's pair count and `group_wsum[g]` its weighted pair count
    (both exact in float32: the per-group cap is 522,753 pairs)."""
    var sh_grade = stack_allocation[
        PLG_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var begin = Int(group_offsets.unsafe_load(g))
    var end = Int(group_offsets.unsafe_load(g + 1))
    var size = end - begin
    # every thread reads the group weight before any weight is rewritten
    var w = weights.unsafe_load(begin)
    barrier()
    var n_chunks = (size + PLG_THREADS - 1) // PLG_THREADS
    var endpoints = Float32(0.0)
    for c in range(n_chunks):
        var i = begin + c * PLG_THREADS + tid
        var in_range = i < end
        var g_i = Float32(0.0)
        if in_range:
            g_i = targets.unsafe_load(i)
        var count = 0
        for t in range(n_chunks):
            var j0 = begin + t * PLG_THREADS
            var jl = j0 + tid
            barrier()
            if jl < end:
                sh_grade.unsafe_store(tid, targets.unsafe_load(jl))
            barrier()
            var tile_n = end - j0
            if tile_n > PLG_THREADS:
                tile_n = PLG_THREADS
            if in_range:
                for k in range(tile_n):
                    if sh_grade.unsafe_load(k) != g_i:
                        count += 1
        if in_range:
            var rw = ftz(identical_mul(w, Float32(count)))
            grades.unsafe_store(i, g_i)
            row_weights.unsafe_store(i, rw)
            endpoints += Float32(count)
    barrier()
    # the weight rewrite, after every chunk's reads of the first row's weight
    for c in range(n_chunks):
        var i = begin + c * PLG_THREADS + tid
        if i < end:
            weights.unsafe_store(i, row_weights.unsafe_load(i))
    var total = pinned_block_sum[block_size=PLG_THREADS](endpoints)
    if tid == 0:
        var pairs = total * Float32(0.5)
        group_w.unsafe_store(g, w)
        group_pairs.unsafe_store(g, pairs)
        group_wsum.unsafe_store(g, identical_mul(w, pairs))


def pair_logit_group_kernel[
    estimation: Bool,
    second_order: Bool,
    store_acc: Bool,
    threads: Int = PLG_THREADS,
    pairs_once: Bool = False,
](
    point: MutPointer[Float32, MutAnyOrigin],
    grades: MutPointer[Float32, MutAnyOrigin],
    group_offsets: MutPointer[UInt32, MutAnyOrigin],
    acc: MutPointer[Float32, MutAnyOrigin],
    group_w_at: Int32,
    row_weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
    der_acc_at: Int32,
    der2_acc_at: Int32,
    fv_acc_at: Int32,
):
    """One block per group, a thread per document per 256-document chunk,
    the group's point and grades tiled through shared memory. The planes
    are `pair_logit_row_kernel`'s: SEARCH `[weight-or-der2, der]` at the
    row, ESTIMATION `[der, der2]` at `write_map[row]`. `function_value`
    takes one partial per group, `plane_magnitudes` two per group. With
    `store_acc` the row sums and the group's value partial are also kept
    in the accumulators for `PAIRLOGIT_EST_REUSE`. Every loop bound and
    every barrier is uniform over the block. The group weights and the
    three accumulators are regions of the ONE `acc` buffer, passed once
    with offsets: `enqueue_function` refuses two mutable arguments derived
    from one allocation as aliasing. `threads` is the block
    (`PLG_LAUNCH_THREADS`, lane af-sym-multi `PL_GROUP_NARROW`) and
    `pairs_once` the each-pair-once path for groups that fit one block
    (`PL_PAIRS_ONCE`); at their defaults this is the kernel as merged."""
    var group_w = acc + Int(group_w_at)
    var der_acc = acc + Int(der_acc_at)
    var der2_acc = acc + Int(der2_acc_at)
    var fv_acc = acc + Int(fv_acc_at)
    comptime assert not (estimation and second_order), (
        "second_order is a SEARCH-mode flag"
    )
    var sh_point = stack_allocation[
        threads, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_grade = stack_allocation[
        threads, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var n_rows = Int(n_rows_in)
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var begin = Int(group_offsets.unsafe_load(g))
    var end = Int(group_offsets.unsafe_load(g + 1))
    var size = end - begin
    var w = group_w.unsafe_load(g)
    var n_chunks = (size + threads - 1) // threads
    var fv_local = Float32(0.0)
    var w_abs = Float32(0.0)
    var g_abs = Float32(0.0)
    # `PL_PAIRS_ONCE`: a group that fits the block, every pair once
    var once_done = False
    comptime if pairs_once:
        if size <= threads:
            var sh_o_der = stack_allocation[
                threads, Scalar[DType.float32], address_space = AddressSpace.SHARED
            ]()
            var sh_o_der2 = stack_allocation[
                threads, Scalar[DType.float32], address_space = AddressSpace.SHARED
            ]()
            var in_range = tid < size
            var i = begin + tid
            var p_i = Float32(0.0)
            var g_i = Float32(0.0)
            if in_range:
                p_i = point.unsafe_load(i)
                g_i = grades.unsafe_load(i)
                sh_point.unsafe_store(tid, p_i)
                sh_grade.unsafe_store(tid, g_i)
            barrier()
            var acc_der = Float32(0.0)
            var acc_der2 = Float32(0.0)
            var half = (size - 1) // 2
            # rounds 1..half, then the half round of an even size: the
            # loop bound is uniform over the block
            var n_rounds = half
            if size % 2 == 0 and size >= 2:
                n_rounds = half + 1
            for r in range(1, n_rounds + 1):
                var full_round = r <= half
                # the half round pairs only the threads below size / 2
                var active = in_range and (full_round or tid < r)
                var j = tid + r
                if j >= size:
                    j -= size
                var o_der = Float32(0.0)
                var o_der2 = Float32(0.0)
                if active:
                    var g_j = sh_grade.unsafe_load(j)
                    if g_j != g_i:
                        # `pair_logit.cu:25-40` for the pair (winner, loser)
                        var winner_side = g_i > g_j
                        var p_j = sh_point.unsafe_load(j)
                        var diff = p_i - p_j
                        if not winner_side:
                            diff = p_j - p_i
                        var exp_diff = routed_exp(diff)
                        var p = Float32(1.0)
                        if isfinite(Float32(1.0) + exp_diff):
                            p = exp_diff / (Float32(1.0) + exp_diff)
                        p = max(
                            min(p, Float32(1.0) - Float32(1e-40)),
                            Float32(1e-40),
                        )
                        var direction = Float32(1.0) - p
                        var scale = ftz(p * (Float32(1.0) - p))
                        var wd = ftz(w * direction)
                        var ws = ftz(w * scale)
                        if winner_side:
                            acc_der = acc_der + wd
                            o_der = -wd
                        else:
                            acc_der = acc_der + (-wd)
                            o_der = wd
                        acc_der2 = acc_der2 + ws
                        o_der2 = ws
                        if compute_fv != Int32(0):
                            var log_exp_val_plus_one = diff
                            if isfinite(Float32(1.0) + exp_diff):
                                log_exp_val_plus_one = routed_log(
                                    Float32(1.0) + exp_diff
                                )
                            fv_local += w * (diff - log_exp_val_plus_one)
                # the previous round's reads of the slots are done
                barrier()
                if active:
                    sh_o_der.unsafe_store(j, o_der)
                    sh_o_der2.unsafe_store(j, o_der2)
                barrier()
                # the other side: in a full round every thread was a target
                # once; in the half round only the threads from size / 2 up
                if in_range and (full_round or tid >= r):
                    acc_der = acc_der + sh_o_der.unsafe_load(tid)
                    acc_der2 = acc_der2 + sh_o_der2.unsafe_load(tid)
            var der = ftz(acc_der)
            var der2 = ftz(acc_der2)
            var weight = Float32(0.0)
            if in_range:
                weight = row_weights.unsafe_load(i)
            var plane0 = weight
            comptime if second_order:
                plane0 = der2
            if in_range:
                comptime if estimation:
                    var dst = i
                    if has_write_map != Int32(0):
                        dst = Int(write_map.unsafe_load(i))
                    stats.unsafe_store(dst, der)
                    stats.unsafe_store(n_rows + dst, der2)
                else:
                    stats.unsafe_store(i, plane0)
                    stats.unsafe_store(n_rows + i, der)
                comptime if store_acc:
                    der_acc.unsafe_store(i, der)
                    der2_acc.unsafe_store(i, der2)
                w_abs += abs(plane0)
                g_abs += abs(der)
            once_done = True
    for c in range(n_chunks):
        if once_done:
            break
        var i = begin + c * threads + tid
        var in_range = i < end
        var p_i = Float32(0.0)
        var g_i = Float32(0.0)
        if in_range:
            p_i = point.unsafe_load(i)
            g_i = grades.unsafe_load(i)
        var acc_der = Float32(0.0)
        var acc_der2 = Float32(0.0)
        for t in range(n_chunks):
            var j0 = begin + t * threads
            var jl = j0 + tid
            barrier()
            if jl < end:
                sh_point.unsafe_store(tid, point.unsafe_load(jl))
                sh_grade.unsafe_store(tid, grades.unsafe_load(jl))
            barrier()
            var tile_n = end - j0
            if tile_n > threads:
                tile_n = threads
            if in_range:
                # N08: unroll two ascending endpoint visits to expose
                # independent shared loads. Keep addition and pair order;
                # this is not the rejected each-pair-once route. No evidence.
                for pair_base in range(0, tile_n, 2 if AFT_N08 else 1):
                    comptime for pair_lane in range(2 if AFT_N08 else 1):
                        var k = pair_base + pair_lane
                        if k < tile_n:
                            var g_j = sh_grade.unsafe_load(k)
                            if g_j != g_i:
                                # `pair_logit.cu:25-40` for the pair (winner, loser)
                                var winner_side = g_i > g_j
                                var diff = p_i - sh_point.unsafe_load(k)
                                if not winner_side:
                                    diff = sh_point.unsafe_load(k) - p_i
                                var exp_diff = routed_exp(diff)
                                var p = Float32(1.0)
                                if isfinite(Float32(1.0) + exp_diff):
                                    p = exp_diff / (Float32(1.0) + exp_diff)
                                p = max(
                                    min(p, Float32(1.0) - Float32(1e-40)),
                                    Float32(1e-40),
                                )
                                var direction = Float32(1.0) - p
                                var scale = ftz(identical_mul(p, Float32(1.0) - p))
                                var wd = ftz(identical_mul(w, direction))
                                if winner_side:
                                    acc_der = acc_der + wd
                                    if compute_fv != Int32(0):
                                        var log_exp_val_plus_one = diff
                                        if isfinite(Float32(1.0) + exp_diff):
                                            log_exp_val_plus_one = routed_log(
                                                Float32(1.0) + exp_diff
                                            )
                                        fv_local = fv_local + identical_mul(
                                            w, diff - log_exp_val_plus_one
                                        )
                                else:
                                    acc_der = acc_der + (-wd)
                                acc_der2 = acc_der2 + ftz(identical_mul(w, scale))
        var der = ftz(acc_der)
        var der2 = ftz(acc_der2)
        var weight = Float32(0.0)
        if in_range:
            weight = row_weights.unsafe_load(i)
        var plane0 = weight
        comptime if second_order:
            plane0 = der2
        if in_range:
            comptime if estimation:
                var dst = i
                if has_write_map != Int32(0):
                    dst = Int(write_map.unsafe_load(i))
                stats.unsafe_store(dst, der)
                stats.unsafe_store(n_rows + dst, der2)
            else:
                stats.unsafe_store(i, plane0)
                stats.unsafe_store(n_rows + i, der)
            comptime if store_acc:
                der_acc.unsafe_store(i, der)
                der2_acc.unsafe_store(i, der2)
            w_abs += abs(plane0)
            g_abs += abs(der)
    if compute_fv != Int32(0):
        var total = pinned_block_sum[block_size=threads](fv_local)
        if tid == 0:
            function_value.unsafe_store(g, total)
            comptime if store_acc:
                fv_acc.unsafe_store(g, total)
    if compute_magnitudes != Int32(0):
        var w_total = pinned_block_sum[block_size=threads](w_abs)
        var g_total = pinned_block_sum[block_size=threads](g_abs)
        if tid == 0:
            plane_magnitudes.unsafe_store(2 * g, w_total)
            plane_magnitudes.unsafe_store(2 * g + 1, g_total)


def pair_logit_group_reuse_kernel(
    acc: MutPointer[Float32, MutAnyOrigin],
    der_acc_at: Int32,
    der2_acc_at: Int32,
    fv_acc_at: Int32,
    n_rows_in: Int32,
    n_groups_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
):
    """`PAIRLOGIT_EST_REUSE`: the estimation planes `[der, der2]` at each
    row's bin position from the search call's accumulators, and the
    per-group value partials copied; one grid over `max(rows, groups)`.
    The accumulators are regions of `acc` at the given offsets (one
    pointer, so no two mutable launch arguments alias)."""
    var der_acc = acc + Int(der_acc_at)
    var der2_acc = acc + Int(der2_acc_at)
    var fv_acc = acc + Int(fv_acc_at)
    var n_rows = Int(n_rows_in)
    var n_groups = Int(n_groups_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n_rows:
        var dst = Int(write_map.unsafe_load(i))
        stats.unsafe_store(dst, der_acc.unsafe_load(i))
        stats.unsafe_store(n_rows + dst, der2_acc.unsafe_load(i))
    if compute_fv != Int32(0) and i < n_groups:
        function_value.unsafe_store(i, fv_acc.unsafe_load(i))


def launch_pair_logit_group_setup(
    ctx: DeviceContext,
    n_groups: Int,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut group_offsets: DeviceBuffer[DType.uint32],
    mut grades: DeviceBuffer[DType.float32],
    mut row_weights: DeviceBuffer[DType.float32],
    mut group_w: DeviceBuffer[DType.float32],
    mut group_pairs: DeviceBuffer[DType.float32],
    mut group_wsum: DeviceBuffer[DType.float32],
) raises:
    """The once-per-fit setup pass (module docstring), one block per group."""
    ctx.enqueue_function[pair_logit_group_setup_kernel](
        targets.unsafe_ptr(), weights.unsafe_ptr(),
        group_offsets.unsafe_ptr(), grades.unsafe_ptr(),
        row_weights.unsafe_ptr(), group_w.unsafe_ptr(),
        group_pairs.unsafe_ptr(), group_wsum.unsafe_ptr(),
        grid_dim=(n_groups, 1, 1),
        block_dim=(PLG_THREADS, 1, 1),
    )


def launch_pair_logit_group[
    estimation: Bool, second_order: Bool, store_acc: Bool
](
    ctx: DeviceContext,
    n_rows: Int,
    n_groups: Int,
    mut point: DeviceBuffer[DType.float32],
    mut grades: DeviceBuffer[DType.float32],
    mut group_offsets: DeviceBuffer[DType.uint32],
    mut row_weights: DeviceBuffer[DType.float32],
    mut write_map: DeviceBuffer[DType.uint32],
    has_write_map: Bool,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
    mut acc: DeviceBuffer[DType.float32],
    group_w_at: Int,
    der_acc_at: Int,
    der2_acc_at: Int,
    fv_acc_at: Int,
) raises:
    """The target call: `point` in row order, one block per group. The
    group weights and the three accumulators are regions of `acc` at the
    given offsets, passed as ONE pointer plus offsets (two pointers
    derived from `acc` are refused as aliasing; the accumulators are
    written only under `store_acc`)."""
    ctx.enqueue_function[
        pair_logit_group_kernel[
            estimation, second_order, store_acc, PLG_LAUNCH_THREADS, PL_PAIRS_ONCE
        ]
    ](
        point.unsafe_ptr(), grades.unsafe_ptr(), group_offsets.unsafe_ptr(),
        acc.unsafe_ptr(), Int32(group_w_at), row_weights.unsafe_ptr(), Int32(n_rows),
        write_map.unsafe_ptr(), Int32(1) if has_write_map else Int32(0),
        stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        Int32(der_acc_at),
        Int32(der2_acc_at),
        Int32(fv_acc_at),
        grid_dim=(n_groups, 1, 1),
        block_dim=(PLG_LAUNCH_THREADS, 1, 1),
    )


def launch_pair_logit_group_reuse(
    ctx: DeviceContext,
    n_rows: Int,
    n_groups: Int,
    mut acc: DeviceBuffer[DType.float32],
    der_acc_at: Int,
    der2_acc_at: Int,
    fv_acc_at: Int,
    mut write_map: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
) raises:
    """`PAIRLOGIT_EST_REUSE`'s scatter, one grid over rows and groups."""
    var n = n_rows
    if n_groups > n:
        n = n_groups
    var blocks = (n + PLG_THREADS - 1) // PLG_THREADS
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function[pair_logit_group_reuse_kernel](
        acc.unsafe_ptr(), Int32(der_acc_at), Int32(der2_acc_at), Int32(fv_acc_at),
        Int32(n_rows), Int32(n_groups),
        write_map.unsafe_ptr(), stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(PLG_THREADS, 1, 1),
    )
