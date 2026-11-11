# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The bound-and-compact composite-key selector for the IDENTICAL tiled k-NN
arm (DEVIATION 3060, lane/knn-selector-speed, 2026-09-17).

Kernel-matrix row `knn_selector_bound_compact_for`.

WHAT THE SMALL-K SELECTOR PAYS FOR AT LARGE k
---------------------------------------------
`select_smallk_identical_candidate.mojo::smallk_bucket_kernel` gives one
block of 256 threads to a row of the distance tile. Every thread walks its
256 columns of a 65,536-column tile and keeps its OWN k smallest composite
keys, because in the worst case the row's whole top-k sits in one thread's
columns. At k = 64 that is a 64-slot UInt64 list per thread (a quarter of
everything the thread reads), far past what a thread holds in registers, a
64-step carry chain on nearly every element step, and then 64 rank rounds
with a barrier each and a 63-slot shift on the popping lane. Measured on the
RTX 4090, 400,000 x 4,000: 68 ms of selection at k 64 against 7 ms at k 10,
on both datasets, more than the distances.

WHAT THIS FILE DOES INSTEAD
---------------------------
Same block shape (one block of SBC_BLOCK threads per row, thread `t` owns
columns `t, t + 256, ...`), same key (`composite_key(distance, tile-local
column, select_min)`), four phases and four barriers, none of them per rank:

  1. SCAN. Every thread keeps only its C smallest keys (C = 8 by default), a
     branch-free min/max carry chain in registers.
  2. BOUND. The 256 thread minima go to shared memory; every thread ranks
     its own minimum among them by counting, and the thread of rank k - 1
     publishes its minimum as the bound B.
  3. COMPACT. Every thread counts its keys at or below B and says whether
     its list might be hiding more (its C-th key is below B). With no such
     thread and at most SBC_CAND candidates in the block, the threads write
     their candidates to shared memory at the offsets an integer prefix sum
     gives them.
  4. RANK. Thread j ranks candidate j among the candidates by counting and,
     when the rank is below k, writes the index half of the key and the
     tile cell it names to that output slot.

A row that cannot take phases 3 and 4 (a hiding thread, or more than SBC_CAND
candidates) raises its flag in `flags[row]`, and a second launch of the
UNCHANGED small-k selector (`smallk_bucket_kernel[..., FLAGGED=True]`)
serves exactly the flagged rows; every other block of that launch returns
at its first statement.

WHY NO BIT MOVES
----------------
Keys are unique (each carries its column). B is a key of the row, and at
least k distinct keys are at or below it: the k thread minima of rank
0 .. k - 1, from k different threads. So any key above B has k keys below
it and is not among the row's k smallest: the answer is a subset of
{key <= B}. When fewer than k threads hold a key (a tile narrower than 256
columns) the rank k - 1 minimum is the sentinel and every real key is a
candidate. A thread's keys at or below B are all in its C-list unless the
list is full below B, which is exactly the hiding test; so on the fast path
the candidate buffer holds EVERY key at or below B, the k smallest of it are
the row's k smallest, and a candidate's rank among the candidates (a count
of strictly smaller unique keys) is its rank in the row. The output slot
`rank` gets the key's index half and `values[row, index]`, the very cell
the small-k selector gathers, so distance bits are copied, never computed.
Every reduction here is an integer count, an integer sum or a UInt64
compare; none depends on the schedule. The flagged rows are served by the
small-k selector's own kernel, statement for statement.

This is selection only. No distance chain is read, split or reordered here.

SABOTAGE (reach proof, never shipped):
`-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1` flips bit 0 of the index the
fast path writes at rank 0, so every row served by phase 4 returns a moved
index and a moved distance.
`-D MOJOLEARN_KNN_SELECTOR_BOUND_FALLBACK_SABOTAGE=1` does the same in the
flagged launch (through the small-k kernel's own SABOTAGE instantiation),
so a row that took the fallback is seen to have taken it.
"""
from max.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key
from neighbors.checks.select_smallk_identical_candidate import (
    SMALLK_MAX_K,
    smallk_flagged_launch,
)
from neighbors.checks.smem_distance_tile import (
    partial_lists_flagged_launch,
    smem_tile_col_blocks,
)
from neighbors.impl.matrix.detail.select_warpsort import twiddle_out

comptime SBC_BLOCK = 256
#: Candidate capacity of the block's shared buffer. The bound sits near the
#: row's (k + 10)-th smallest key on hashed data at k = 64, so the expected
#: candidate count is under 80; 256 keeps one thread per candidate.
comptime SBC_CAND = 256
comptime SBC_SCAN_UNROLL = 8
comptime SBC_SCAN_SPAN = SBC_SCAN_UNROLL * SBC_BLOCK
comptime SBC_SENTINEL = UInt64(18446744073709551615)
#: What a hiding thread publishes as its count: one such thread pushes the
#: block total past SBC_CAND, and 256 of them still fit an Int32.
comptime SBC_HIDING = 100000
comptime SBC_SABOTAGE = is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE"]()
comptime SBC_FALLBACK_SABOTAGE = is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_FALLBACK_SABOTAGE"]()
#: The per-thread list depth; the two defines are A/B arms. MEASURED on the
#: RTX 4090, 400,000 x 4,000, selection class of a timer build, k 64: depth
#: 8 reads 8.0 ms with 0 of 28,000 row tiles flagged on either dataset; depth
#: 4 reads 13.2 (Istella-S) and 13.6 ms (taxi) with 1,147 and 1,276 flagged,
#: 7.8 ms of it the scan and 5.4 the flagged launch; depth 2 flags nearly
#: every row and reads 88 to 93 ms (bench/results/knn_selector_2026-09-17/).
comptime SBC_DEPTH = 4 if is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_C4"]() else (
    2 if is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_C2"]() else 8
)
comptime SBC_PHASE_TIMERS = is_defined["MOJOLEARN_KNN_PHASE_TIMERS"]()
#: TIMING ONLY, OUTPUT INVALID on flagged rows: no flagged launch, so a timer
#: build prices that launch by difference. Never shipped.
comptime SBC_TIMING_NOFLAG = is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_TIMING_NOFLAG"]()


@always_inline
def _sbc_insert[C: Int](mut keys: SIMD[DType.uint64, C], pending_in: UInt64):
    """The C smallest of the old list and the new key, ascending: at every
    step the smaller of (carry, slot) stays and the larger moves on. Keys are
    unique, so no compare ties on real keys, and two sentinels yield the
    sentinel either way."""
    var pending = pending_in
    comptime for slot in range(C):
        var current = keys[slot]
        keys[slot] = min(pending, current)
        pending = max(pending, current)


def bound_compact_select_kernel[C: Int](
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    length_in: Int32, k_in: Int32, select_min_in: Int32,
):
    """One block of SBC_BLOCK threads per row; see the module docstring.
    `flags[row]` is written by thread 0 on every row: 0 when phase 4 wrote
    the row's k outputs, 1 when the row is left to the flagged launch.
"""
    comptime assert C >= 1 and C <= 8, "the per-thread list is 1 .. 8 keys"
    var length = Int(length_in)
    var k = Int(k_in)
    var select_min = select_min_in != 0
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var base = row * length
    var keys = SIMD[DType.uint64, C](SBC_SENTINEL)

    # ---- 1. SCAN: the small-k selector's unrolled walk, block-uniform trip
    # count (its DEVIATION 2497), every column of this thread once.
    var batch_base = 0
    while batch_base + SBC_SCAN_SPAN <= length:
        var batch = SIMD[DType.float32, SBC_SCAN_UNROLL](0.0)
        comptime for u in range(SBC_SCAN_UNROLL):
            batch[u] = values.unsafe_load(base + batch_base + tid + u * SBC_BLOCK)
        comptime for u in range(SBC_SCAN_UNROLL):
            _sbc_insert[C](
                keys,
                composite_key(batch[u], UInt32(batch_base + tid + u * SBC_BLOCK), select_min),
            )
        batch_base += SBC_SCAN_SPAN
    var col = batch_base + tid
    while col < length:
        _sbc_insert[C](keys, composite_key(values.unsafe_load(base + col), UInt32(col), select_min))
        col += SBC_BLOCK

    # ---- 2. BOUND: the rank k - 1 thread minimum. Ranks by (key, thread)
    # are a permutation of 0 .. 255 (only sentinels tie), so exactly one
    # thread publishes.
    var heads = stack_allocation[
        SBC_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var bound_slot = stack_allocation[
        1, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var counts = stack_allocation[
        SBC_BLOCK, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    var cand = stack_allocation[
        SBC_CAND, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var mine = keys[0]
    heads[tid] = mine
    barrier()
    var my_rank = 0
    for j in range(SBC_BLOCK):
        var other = heads[j]
        if other < mine or (other == mine and j < tid):
            my_rank += 1
    if my_rank == k - 1:
        bound_slot[0] = mine
    barrier()
    var bound = bound_slot[0]

    # ---- 3. COMPACT: counts, the hiding test, integer prefix sums.
    var count = 0
    comptime for slot in range(C):
        if keys[slot] != SBC_SENTINEL and keys[slot] <= bound:
            count += 1
    # A full list whose last key is below the bound may hide more keys at or
    # below it. (A sentinel last key is never below anything.)
    if keys[C - 1] < bound:
        count = SBC_HIDING
    counts[tid] = Int32(count)
    barrier()
    var offset = 0
    var total = 0
    for j in range(SBC_BLOCK):
        var c = Int(counts[j])
        total += c
        if j < tid:
            offset += c
    var fast = total <= SBC_CAND and total >= k
    if tid == 0:
        flags.unsafe_store(row, UInt32(0) if fast else UInt32(1))
    if fast:
        # ---- 4. RANK: every candidate's rank among the candidates. `fast`
        # is a function of the shared counts alone, so the branch and its
        # barrier are block-uniform.
        comptime for slot in range(C):
            if slot < count:
                cand[offset + slot] = keys[slot]
        barrier()
        if tid < total:
            var key = cand[tid]
            var rank = 0
            for j in range(total):
                if cand[j] < key:
                    rank += 1
            if rank < k:
                var selected = UInt32(key & UInt64(4294967295))
                comptime if SBC_SABOTAGE:
                    if rank == 0:
                        selected = selected ^ UInt32(1)
                        if Int(selected) >= length:
                            selected = UInt32(length - 2) if length >= 2 else UInt32(0)
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, values.unsafe_load(base + Int(selected)))



def bound_compact_lists_kernel(
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    n_lists_in: Int32, k_in: Int32,
):
    """The bound-and-compact phases over the block top-k's per-block lists
    (DEVIATION 3062). One block of SBC_BLOCK threads per row; the row owns
    `n_lists` lists of up to k ASCENDING composite keys in `part`, a list
    ending at its first sentinel (the bounded rank loop's terminator) or at
    k keys; thread `t` owns lists `t, t + 256, ...`.

    Because a list ascends, a thread needs no list of its own: its minimum
    is the smallest first key of its lists; with the bound B known (the
    rank k - 1 of the 256 thread minima, as in `bound_compact_select_kernel`)
    its candidates are the prefixes of its lists at or below B, counted by
    one walk and written to the block's candidate buffer by a second. No
    thread can hide a key, so the only flagged rows are those with more than
    SBC_CAND candidates. A bounded column tile may offer fewer than k keys
    in all: the output slots past the last candidate get the ABSENT pair
    (index 0xFFFFFFFF, which no tile-local column is, and the sentinel's
    distance half; what the partial-key selector writes when a sentinel wins
    a rank), which `partial_topk_merge_kernel` skips.

    WHY NO BIT MOVES: the argument of the module docstring with "every key
    at or below B" now literally enumerated; the distance written is
    `twiddle_out` of the key's high half, the exact inverse of the
    `twiddle_in` that built it (what `partial_keys_select_kernel` writes).
    """
    var n_lists = Int(n_lists_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var base = row * n_lists
    var heads = stack_allocation[
        SBC_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var bound_slot = stack_allocation[
        1, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var counts = stack_allocation[
        SBC_BLOCK, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    var cand = stack_allocation[
        SBC_CAND, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    # ---- 1. the thread minimum: the smallest first key of its lists.
    var mine = SBC_SENTINEL
    var lst = tid
    while lst < n_lists:
        mine = min(mine, part.unsafe_load((base + lst) * k))
        lst += SBC_BLOCK
    heads[tid] = mine
    barrier()
    # ---- 2. BOUND.
    var my_rank = 0
    for j in range(SBC_BLOCK):
        var other = heads[j]
        if other < mine or (other == mine and j < tid):
            my_rank += 1
    if my_rank == k - 1:
        bound_slot[0] = mine
    barrier()
    var bound = bound_slot[0]
    # ---- 3. COMPACT: the prefixes at or below the bound.
    var count = 0
    lst = tid
    while lst < n_lists:
        for s in range(k):
            var key = part.unsafe_load((base + lst) * k + s)
            if key == SBC_SENTINEL or key > bound:
                break
            count += 1
        lst += SBC_BLOCK
    counts[tid] = Int32(count)
    barrier()
    var offset = 0
    var total = 0
    for j in range(SBC_BLOCK):
        var c = Int(counts[j])
        total += c
        if j < tid:
            offset += c
    var fast = total <= SBC_CAND
    if tid == 0:
        flags.unsafe_store(row, UInt32(0) if fast else UInt32(1))
    if fast:
        var at = offset
        lst = tid
        while lst < n_lists:
            for s in range(k):
                var key = part.unsafe_load((base + lst) * k + s)
                if key == SBC_SENTINEL or key > bound:
                    break
                cand[at] = key
                at += 1
            lst += SBC_BLOCK
        barrier()
        # ---- 4. RANK, and the ABSENT pairs of a short row.
        if tid >= total and tid < k:
            out_indices.unsafe_store(row * k + tid, UInt32(4294967295))
            out_values.unsafe_store(row * k + tid, twiddle_out(UInt32(4294967295)))
        if tid < total:
            var key = cand[tid]
            var rank = 0
            for j in range(total):
                if cand[j] < key:
                    rank += 1
            if rank < k:
                var selected = UInt32(key & UInt64(4294967295))
                comptime if SBC_SABOTAGE:
                    if rank == 0:
                        selected = selected ^ UInt32(1)
                out_indices.unsafe_store(row * k + rank, selected)
                out_values.unsafe_store(row * k + rank, twiddle_out(UInt32(key >> UInt64(32))))


def bound_compact_select_launch(
    ctx: DeviceContext,
    values: MutPointer[Float32, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    mut flags: DeviceBuffer[DType.uint32],
    rows: Int, length: Int, k: Int, select_min: Bool = True,
) raises:
    """One query tile's top-k for 1 <= k <= SMALLK_MAX_K: the bound-and-compact
    launch, then the flagged launch of the small-k selector for the rows it
    left. `flags` holds at least `rows` UInt32 and is the caller's to keep
    alive through synchronization (one buffer serves a whole request). `length >= k` is the caller's to guarantee,
    as it is for the small-k selector."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("bound-and-compact selector requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("bound-and-compact selector requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_MAX_K or k > length:
        raise Error("bound-and-compact selector supports only 1 <= k <= min(64, length)")
    if len(flags) < rows:
        raise Error("bound-and-compact selector: the flag buffer is shorter than the query tile")
    var flag_ptr = flags.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[bound_compact_select_kernel[SBC_DEPTH]](
        values, out_values, out_indices, flag_ptr,
        Int32(length), Int32(k), Int32(select_min),
        grid_dim=(rows, 1, 1), block_dim=(SBC_BLOCK, 1, 1),
    )
    comptime if not SBC_TIMING_NOFLAG:
        smallk_flagged_launch[SBC_FALLBACK_SABOTAGE](
            ctx, values, out_values, out_indices, flag_ptr, rows, length, k, select_min
        )
    comptime if SBC_PHASE_TIMERS:
        # Timer builds only: how many rows the flagged launch served.
        var host = ctx.enqueue_create_host_buffer[DType.uint32](len(flags))
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flags)
        ctx.synchronize()
        var flagged = 0
        for r in range(rows):
            flagged += Int(host.unsafe_ptr().unsafe_load(r))
        print("KNN_SELECT_FALLBACK", "rows", rows, "flagged", flagged, "length", length, "k", k, "depth", SBC_DEPTH)
        _ = host^


def bound_compact_lists_launch(
    ctx: DeviceContext,
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    mut flags: DeviceBuffer[DType.uint32],
    rows: Int, cols: Int, k: Int,
) raises:
    """The row's k smallest from the block top-k's sentinel-terminated lists
    (DEVIATION 3062), into the selection destination: the bound-and-compact
    launch over the lists, then the flagged launch of the partial-key
    selector for the rows it left. Replaces `partial_keys_select_launch`
    where the rank loop is bounded."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("bound-and-compact list selector requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or cols <= 0 or cols > 2147483647:
        raise Error("bound-and-compact list selector requires positive Int32 dimensions")
    if k < 1 or k > SMALLK_MAX_K or k > cols:
        raise Error("bound-and-compact list selector supports only 1 <= k <= min(64, cols)")
    if len(flags) < rows:
        raise Error("bound-and-compact list selector: the flag buffer is shorter than the query tile")
    var n_cb = smem_tile_col_blocks(cols)
    var flag_ptr = flags.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[bound_compact_lists_kernel](
        part, out_values, out_indices, flag_ptr,
        Int32(n_cb), Int32(k),
        grid_dim=(rows, 1, 1), block_dim=(SBC_BLOCK, 1, 1),
    )
    comptime if not SBC_TIMING_NOFLAG:
        partial_lists_flagged_launch(ctx, part, out_values, out_indices, flag_ptr, rows, cols, k)
    comptime if SBC_PHASE_TIMERS:
        var host = ctx.enqueue_create_host_buffer[DType.uint32](len(flags))
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flags)
        ctx.synchronize()
        var flagged = 0
        for r in range(rows):
            flagged += Int(host.unsafe_ptr().unsafe_load(r))
        print("KNN_SELECT_FALLBACK", "rows", rows, "flagged", flagged, "length", n_cb, "k", k, "depth", SBC_DEPTH, "lists", 1)
        _ = host^
