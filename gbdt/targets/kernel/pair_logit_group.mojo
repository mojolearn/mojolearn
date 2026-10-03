# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST Apple (lane apple-fast-pairlogit, the plan's `trees-pairlogit`):
PairLogit's pairs ENUMERATED ON THE DEVICE, one block per query group, the
per-pair logistic gradient and the per-document sums in the same launch.

Compiled only under `PAIRLOGIT_GROUP_FUSED` (`-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED`
on FAST + Apple); IDENTICAL compiles `gbdt/targets/kernel/pair_logit.mojo`'s
path unchanged. `-D MOJOLEARN_PAIRLOGIT_EST_REUSE` (also FAST + Apple, needs
the first) lets the leaf estimation's first evaluation reuse the search's
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
copies of `w`: FAST bits move, IDENTICAL bits do not (nothing here compiles
under IDENTICAL). Every row's and every group's fold is a fixed sequential
order, so FAST runs are repeatable.

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
from std.math import isfinite
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    pinned_block_sum,
    routed_exp,
    routed_log,
)


def pairlogit_group_fused_for[column: Int]() -> Bool:
    """`-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED`, FAST + Apple only (the A/B's B
    arm); everything else compiles main's pair-list path."""
    comptime if is_defined["MOJOLEARN_PAIRLOGIT_GROUP_FUSED"]():
        comptime if column == COLUMN_APPLE and GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
            return True
    return False


def pairlogit_est_reuse_for[column: Int]() -> Bool:
    """`-D MOJOLEARN_PAIRLOGIT_EST_REUSE`: the leaf estimation's evaluation
    at the tree's starting point reuses the search call's per-row sums and
    per-group value partials (scattered to bin order) instead of
    re-enumerating the pairs. Same point, same sample (PairLogit draws
    nothing), so only FAST bits move with the fold. Needs the group kernel
    (the accumulators are its stores)."""
    comptime if is_defined["MOJOLEARN_PAIRLOGIT_EST_REUSE"]():
        return pairlogit_group_fused_for[column]()
    return False


comptime PAIRLOGIT_GROUP_FUSED = pairlogit_group_fused_for[TARGET_COLUMN]()
comptime PAIRLOGIT_EST_REUSE = pairlogit_est_reuse_for[TARGET_COLUMN]()

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
            var rw = ftz(w * Float32(count))
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
        group_wsum.unsafe_store(g, w * pairs)


def pair_logit_group_kernel[
    estimation: Bool, second_order: Bool, store_acc: Bool
](
    point: MutPointer[Float32, MutAnyOrigin],
    grades: MutPointer[Float32, MutAnyOrigin],
    group_offsets: MutPointer[UInt32, MutAnyOrigin],
    acc: MutPointer[Float32, MutAnyOrigin],
    group_w_at: Int,
    row_weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
    der_acc_at: Int,
    der2_acc_at: Int,
    fv_acc_at: Int,
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
    from one allocation as aliasing."""
    var group_w = acc + group_w_at
    var der_acc = acc + der_acc_at
    var der2_acc = acc + der2_acc_at
    var fv_acc = acc + fv_acc_at
    comptime assert not (estimation and second_order), (
        "second_order is a SEARCH-mode flag"
    )
    var sh_point = stack_allocation[
        PLG_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_grade = stack_allocation[
        PLG_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var n_rows = Int(n_rows_in)
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var begin = Int(group_offsets.unsafe_load(g))
    var end = Int(group_offsets.unsafe_load(g + 1))
    var size = end - begin
    var w = group_w.unsafe_load(g)
    var n_chunks = (size + PLG_THREADS - 1) // PLG_THREADS
    var fv_local = Float32(0.0)
    var w_abs = Float32(0.0)
    var g_abs = Float32(0.0)
    for c in range(n_chunks):
        var i = begin + c * PLG_THREADS + tid
        var in_range = i < end
        var p_i = Float32(0.0)
        var g_i = Float32(0.0)
        if in_range:
            p_i = point.unsafe_load(i)
            g_i = grades.unsafe_load(i)
        var acc_der = Float32(0.0)
        var acc_der2 = Float32(0.0)
        for t in range(n_chunks):
            var j0 = begin + t * PLG_THREADS
            var jl = j0 + tid
            barrier()
            if jl < end:
                sh_point.unsafe_store(tid, point.unsafe_load(jl))
                sh_grade.unsafe_store(tid, grades.unsafe_load(jl))
            barrier()
            var tile_n = end - j0
            if tile_n > PLG_THREADS:
                tile_n = PLG_THREADS
            if in_range:
                for k in range(tile_n):
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
                        var scale = ftz(p * (Float32(1.0) - p))
                        var wd = ftz(w * direction)
                        if winner_side:
                            acc_der = acc_der + wd
                            if compute_fv != Int32(0):
                                var log_exp_val_plus_one = diff
                                if isfinite(Float32(1.0) + exp_diff):
                                    log_exp_val_plus_one = routed_log(
                                        Float32(1.0) + exp_diff
                                    )
                                fv_local += w * (diff - log_exp_val_plus_one)
                        else:
                            acc_der = acc_der + (-wd)
                        acc_der2 = acc_der2 + ftz(w * scale)
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
        var total = pinned_block_sum[block_size=PLG_THREADS](fv_local)
        if tid == 0:
            function_value.unsafe_store(g, total)
            comptime if store_acc:
                fv_acc.unsafe_store(g, total)
    if compute_magnitudes != Int32(0):
        var w_total = pinned_block_sum[block_size=PLG_THREADS](w_abs)
        var g_total = pinned_block_sum[block_size=PLG_THREADS](g_abs)
        if tid == 0:
            plane_magnitudes.unsafe_store(2 * g, w_total)
            plane_magnitudes.unsafe_store(2 * g + 1, g_total)


def pair_logit_group_reuse_kernel(
    acc: MutPointer[Float32, MutAnyOrigin],
    der_acc_at: Int,
    der2_acc_at: Int,
    fv_acc_at: Int,
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
    var der_acc = acc + der_acc_at
    var der2_acc = acc + der2_acc_at
    var fv_acc = acc + fv_acc_at
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
        pair_logit_group_kernel[estimation, second_order, store_acc]
    ](
        point.unsafe_ptr(), grades.unsafe_ptr(), group_offsets.unsafe_ptr(),
        acc.unsafe_ptr(), group_w_at, row_weights.unsafe_ptr(), Int32(n_rows),
        write_map.unsafe_ptr(), Int32(1) if has_write_map else Int32(0),
        stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        der_acc_at,
        der2_acc_at,
        fv_acc_at,
        grid_dim=(n_groups, 1, 1),
        block_dim=(PLG_THREADS, 1, 1),
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
        acc.unsafe_ptr(), der_acc_at, der2_acc_at, fv_acc_at,
        Int32(n_rows), Int32(n_groups),
        write_map.unsafe_ptr(), stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(PLG_THREADS, 1, 1),
    )
