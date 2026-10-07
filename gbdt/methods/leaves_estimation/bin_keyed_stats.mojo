# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane S3 `MOJOLEARN_TREES_CTR_SORTFREE_SUMS` (IDENTICAL, default off):
per-leaf sums keyed by each row's leaf id, with no partition sort.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

WHAT IT REPLACES. With CTR-bearing categorical columns a non-symmetric fit
keeps `permutation_count` (4) learn cursors, and every tree estimates its
leaves once per permutation: `compute_non_symmetric_bins_for_model`, then
`DeviceLeafPartitioner.partition_enqueue` (a radix sort of n (leaf, row)
pairs), a gather of target / weight / cursor into leaf order, and the
oracle's `compute_partition_stats` over each leaf's contiguous range
(doc_parallel_boosting.mojo, the `perm_batched` walk). The sort and the
gathers move every row several times per permutation per tree; the sums
themselves only need each row's leaf id. Here the oracle keeps the rows in
their ORIGINAL order, `d_bins` is the per-row leaf id the model walk wrote,
and the per-leaf sums come from `enqueue_bin_keyed_stats`.

COST REASONING (no board shape). Phase 1 reads each row's leaf id and stat
once per (stat, leaf group of up to 256 leaves) into shared memory; every
thread owns one leaf and scans the tile with broadcast reads, so the work is
n x ceil(leaves / 256) global reads and n x leaves / 32 warp instructions,
against a stable radix sort plus three gathers and one scatter of n rows.
Phase 2 folds ceil(n / BK_CHUNK) partials per (leaf, stat) on one block.

FLOAT, NOT INTEGER: the sums are float32 der / der2 / weight planes, and
the walker reads them as floats; integer atomics would need a fixed-point
grid that changes the values themselves. So the accumulation is FLOAT in a
FIXED ORDER (bits change against the partition fold; the same on NVIDIA,
AMD and Apple):
  1. chunk c holds the ORIGINAL rows [c * BK_CHUNK, (c + 1) * BK_CHUNK);
     leaf l's partial for chunk c is 0.0 plus the stat of every row of the
     chunk whose leaf id is l, in ASCENDING row order (rows of other leaves
     are skipped, not added as 0.0). The launch's block width (64, 128 or
     256 leaves) never changes this chain;
  2. leaf l's total: thread t of one BK_FOLD_BLOCK-thread block adds the
     partials of chunks t, t + 256, ... ascending from 0.0, then the
     256-lane halving tree (`halving_block_sum`, core/pinned_reduce.mojo,
     the repo's vendor-independent fixed tree).
The per-leaf row COUNTS (`enqueue_bin_keyed_counts`) are integer atomics:
exact and order-free. No warp-width primitive, no float atomic.

There is no host column to follow: the CPU-only binding refuses categorical
columns outside SymmetricTree (`bindings/_mojolearn_gbdt_host.mojo`, the
`cat_features ... outside SymmetricTree with Logloss` refusal), and this
route runs only with more than one permutation, i.e. with CTR columns.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import identical_mul_add
from core.device_zero import enqueue_fill
from core.pinned_reduce import halving_block_sum

#: Rows per phase-1 chunk: a constant, so the fold order is a pure function
#: of the row count on every vendor. 2048 keeps each leaf's float chain
#: short (at most 2048 adds) and phase 2's partial count near n / 2048.
comptime BK_CHUNK = 2048
#: Threads of the phase-2 fold (one block per (leaf, stat)).
comptime BK_FOLD_BLOCK = 256
#: The widest phase-1 block: leaves per block, and rows per shared tile.
comptime BK_MAX_TPB = 256


def bk_chunks(n_rows: Int) -> Int:
    """Phase-1 chunks over `n_rows` original rows."""
    return (n_rows + BK_CHUNK - 1) // BK_CHUNK


def bk_partials_len(n_leaves: Int, n_stats: Int, n_rows: Int) -> Int:
    """Floats of phase-1 scratch for one call."""
    return n_leaves * n_stats * bk_chunks(n_rows)


def _bk_tpb(n_leaves: Int) -> Int:
    """The phase-1 block width: the least of 64, 128, 256 that covers the
    leaves (more leaves take more blocks in y). Scheduling only: a leaf's
    chain is the chunk's matching rows in ascending order at every width,
    and every width divides `BK_CHUNK`."""
    if n_leaves <= 64:
        return 64
    if n_leaves <= 128:
        return 128
    return BK_MAX_TPB


def _bk_partial_kernel[TPB: Int](
    bins: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    line_size_in: Int32,
    n_rows_in: Int32,
    n_leaves_in: Int32,
    n_chunks_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """Grid (chunks, ceil(leaves / TPB), stats), TPB threads: thread t owns
    leaf `block_idx.y * TPB + t`; the block stages TPB rows of the chunk at a
    time (leaf id, stat) in shared memory and every thread adds, in ascending
    row order, the stats of the rows that are its leaf's."""
    var chunk = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var leaf = Int(block_idx.y) * TPB + t
    var stat = Int(block_idx.z)
    var n_stats = Int(grid_dim.z)
    var n_rows = Int(n_rows_in)
    var line_size = Int(line_size_in)
    var s_bin = stack_allocation[
        TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var s_val = stack_allocation[
        TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var first = chunk * BK_CHUNK
    var last = first + BK_CHUNK
    if last > n_rows:
        last = n_rows
    var me = UInt32(leaf)
    var acc = Float32(0.0)
    var base = first
    # `base` and `last` are uniform over the block: every thread reaches
    # every barrier
    while base < last:
        var r = base + t
        if r < last:
            s_bin[t] = bins.unsafe_load(r)
            s_val[t] = stats.unsafe_load(stat * line_size + r)
        else:
            # no leaf id reaches 2^32 - 1 (a leaf id is below n_leaves)
            s_bin[t] = UInt32(0xFFFFFFFF)
            s_val[t] = Float32(0.0)
        barrier()
        for j in range(TPB):
            if s_bin[j] == me:
                acc += s_val[j]
        barrier()
        base += TPB
    if leaf < Int(n_leaves_in):
        partials.unsafe_store(
            (leaf * n_stats + stat) * Int(n_chunks_in) + chunk, acc
        )


def _bk_finish_kernel(
    partials: MutPointer[Float32, MutAnyOrigin],
    n_chunks_in: Int32,
    out_stats: MutPointer[Float32, MutAnyOrigin],
):
    """Grid (leaves, stats), BK_FOLD_BLOCK threads: thread t adds partials
    t, t + 256, ... ascending from 0.0, then the 256-lane halving tree.
    `out_stats[leaf * n_stats + stat]`, `compute_partition_stats`' layout."""
    var leaf = Int(block_idx.x)
    var stat = Int(block_idx.y)
    var n_stats = Int(grid_dim.y)
    var n_chunks = Int(n_chunks_in)
    var t = Int(thread_idx.x)
    var base = (leaf * n_stats + stat) * n_chunks
    var acc = Float32(0.0)
    var c = t
    while c < n_chunks:
        acc += partials.unsafe_load(base + c)
        c += BK_FOLD_BLOCK
    # every thread calls the fold (its contract); thread 0's is the total
    var total = halving_block_sum[BK_FOLD_BLOCK](acc)
    if t == 0:
        out_stats.unsafe_store(leaf * n_stats + stat, total)


def enqueue_bin_keyed_stats(
    ctx: DeviceContext,
    n_leaves: Int,
    n_stats: Int,
    line_size: Int,
    n_rows: Int,
    mut bins: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    mut out_stats: DeviceBuffer[DType.float32],
) raises:
    """`compute_partition_stats(ctx, n_leaves, 0, n_stats, line_size, ...)`'s
    output (`out_stats[leaf * n_stats + stat]`) from per-row leaf ids in
    ORIGINAL row order, in the module docstring's fixed order. `partials`
    holds at least `bk_partials_len(n_leaves, n_stats, n_rows)` floats."""
    if n_rows < 1 or n_leaves < 1 or n_stats < 1:
        raise Error("enqueue_bin_keyed_stats: empty launch")
    if len(partials) < bk_partials_len(n_leaves, n_stats, n_rows):
        raise Error("enqueue_bin_keyed_stats: partials scratch too small")
    var chunks = bk_chunks(n_rows)
    var tpb = _bk_tpb(n_leaves)
    var gy = (n_leaves + tpb - 1) // tpb
    if tpb == 64:
        ctx.enqueue_function[_bk_partial_kernel[64]](
            bins.unsafe_ptr(), stats.unsafe_ptr(), Int32(line_size),
            Int32(n_rows), Int32(n_leaves), Int32(chunks), partials.unsafe_ptr(),
            grid_dim=(chunks, gy, n_stats), block_dim=(64, 1, 1),
        )
    elif tpb == 128:
        ctx.enqueue_function[_bk_partial_kernel[128]](
            bins.unsafe_ptr(), stats.unsafe_ptr(), Int32(line_size),
            Int32(n_rows), Int32(n_leaves), Int32(chunks), partials.unsafe_ptr(),
            grid_dim=(chunks, gy, n_stats), block_dim=(128, 1, 1),
        )
    else:
        ctx.enqueue_function[_bk_partial_kernel[BK_MAX_TPB]](
            bins.unsafe_ptr(), stats.unsafe_ptr(), Int32(line_size),
            Int32(n_rows), Int32(n_leaves), Int32(chunks), partials.unsafe_ptr(),
            grid_dim=(chunks, gy, n_stats), block_dim=(BK_MAX_TPB, 1, 1),
        )
    ctx.enqueue_function[_bk_finish_kernel](
        partials.unsafe_ptr(), Int32(chunks), out_stats.unsafe_ptr(),
        grid_dim=(n_leaves, n_stats, 1), block_dim=(BK_FOLD_BLOCK, 1, 1),
    )


def _bk_count_kernel[TPB: Int](
    bins: MutPointer[UInt32, MutAnyOrigin],
    n_rows_in: Int32,
    n_leaves_in: Int32,
    counts: MutPointer[Int32, MutAnyOrigin],
):
    """Grid (chunks, ceil(leaves / TPB)): the chunk's rows of each leaf,
    counted as `_bk_partial_kernel` scans them, added to `counts[leaf]` with
    one integer atomic per (chunk, leaf) that has rows. Exact, order-free."""
    var chunk = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var leaf = Int(block_idx.y) * TPB + t
    var n_rows = Int(n_rows_in)
    var s_bin = stack_allocation[
        TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var first = chunk * BK_CHUNK
    var last = first + BK_CHUNK
    if last > n_rows:
        last = n_rows
    var me = UInt32(leaf)
    var cnt = Int32(0)
    var base = first
    while base < last:
        var r = base + t
        if r < last:
            s_bin[t] = bins.unsafe_load(r)
        else:
            s_bin[t] = UInt32(0xFFFFFFFF)
        barrier()
        for j in range(TPB):
            if s_bin[j] == me:
                cnt += Int32(1)
        barrier()
        base += TPB
    if leaf < Int(n_leaves_in) and cnt > Int32(0):
        _ = Atomic.fetch_add(counts.unsafe_offset(leaf), cnt)


def enqueue_bin_keyed_counts(
    ctx: DeviceContext,
    n_leaves: Int,
    n_rows: Int,
    mut bins: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],
) raises:
    """Rows per leaf from per-row leaf ids (`counts` zeroed here first)."""
    if n_rows < 1 or n_leaves < 1:
        raise Error("enqueue_bin_keyed_counts: empty launch")
    if len(counts) < n_leaves:
        raise Error("enqueue_bin_keyed_counts: counts too small")
    enqueue_fill(ctx, counts, Int32(0))
    var chunks = bk_chunks(n_rows)
    var tpb = _bk_tpb(n_leaves)
    var gy = (n_leaves + tpb - 1) // tpb
    if tpb == 64:
        ctx.enqueue_function[_bk_count_kernel[64]](
            bins.unsafe_ptr(), Int32(n_rows), Int32(n_leaves), counts.unsafe_ptr(),
            grid_dim=(chunks, gy, 1), block_dim=(64, 1, 1),
        )
    elif tpb == 128:
        ctx.enqueue_function[_bk_count_kernel[128]](
            bins.unsafe_ptr(), Int32(n_rows), Int32(n_leaves), counts.unsafe_ptr(),
            grid_dim=(chunks, gy, 1), block_dim=(128, 1, 1),
        )
    else:
        ctx.enqueue_function[_bk_count_kernel[BK_MAX_TPB]](
            bins.unsafe_ptr(), Int32(n_rows), Int32(n_leaves), counts.unsafe_ptr(),
            grid_dim=(chunks, gy, 1), block_dim=(BK_MAX_TPB, 1, 1),
        )


def _bk_copy_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(i))


def _bk_copy_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(i))


def enqueue_bk_copy_f32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`dst[0:n] = src[0:n]` (the oracle's working cursor, original order)."""
    if n < 1:
        return
    ctx.enqueue_function[_bk_copy_f32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


def enqueue_bk_copy_u32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    mut src: DeviceBuffer[DType.uint32],
    n: Int,
) raises:
    """`dst[0:n] = src[0:n]` (the oracle's `d_bins`, original order)."""
    if n < 1:
        return
    ctx.enqueue_function[_bk_copy_u32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


def _bk_apply_kernel(
    bins: MutPointer[UInt32, MutAnyOrigin],
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    learning_rate: Float32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`add_model_value_kernel`'s per-row statement, `cursor[r] =
    identical_mul_add(est[leaf], rate, cursor[r])`, with row r's leaf read
    from `bins[r]` instead of a partition range: the same operands and the
    same one rounding per row, so the cursor bits are the partition path's
    for the same leaf values."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < Int(n_in):
        var raw = leaf_values.unsafe_load(Int(bins.unsafe_load(r)))
        cursor.unsafe_store(
            r, identical_mul_add(raw, learning_rate, cursor.unsafe_load(r))
        )


def enqueue_bin_keyed_apply(
    ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.uint32],
    mut leaf_values: DeviceBuffer[DType.float32],
    learning_rate: Float32,
    mut cursor: DeviceBuffer[DType.float32],
    n_rows: Int,
) raises:
    """`AppendModels` onto the ORIGINAL-order cursor from per-row leaf ids."""
    if n_rows < 1:
        return
    ctx.enqueue_function[_bk_apply_kernel](
        bins.unsafe_ptr(), leaf_values.unsafe_ptr(), learning_rate,
        cursor.unsafe_ptr(), Int32(n_rows),
        grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
