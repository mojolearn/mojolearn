# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device exclusive scan `rbc_eps_pass` needs, and a max-reduce.

STANDS IN FOR `thrust::exclusive_scan` at
`cuvs/src/neighbors/ball_cover/registers.cuh:1376,1470` and for
`thrust::reduce(..., thrust::maximum<value_idx>())` at `:1453`.

WHY THIS IS HAND-WRITTEN
------------------------
Thrust is open, so `exclusive_scan` is an candidate to implement rather than a
substitution candidate. Its structure here is trivial and is copied: theirs
is a single device-wide scan launched from the host between two kernel
launches (`:1354` then `:1376` then `:1385`), reading its input from global
memory and writing its output to global memory. There is nothing in it to
fuse and nothing in it that is not a prefix sum.

There would in any case be nothing to substitute: `nn.cumsum.cumsum` and
`max.algorithm.reduction.cumsum` both take neither a `DeviceContext` nor a
`target`, which `archive/reference/VENDOR_LIBRARIES.md` records as the signature of a HOST-ONLY
entry point, and it lists no device scan anywhere in the shipped kernel
libraries.

The shape is `dbscan/gbdt/dbscan/adjgraph/algo.mojo::exclusive_scan_kernel`
copied deliberately rather than imported: another lane owns that file this
round and this one must not depend on its signature. The dynamic chunk is
theirs and is load-bearing — a FIXED rows-per-thread silently stopped
scanning past `SCAN_TPB * chunk` elements and produced a truncated CSR, which
that file records as a bug found by audit rather than by a test.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.block import max as block_max
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime RBC_SCAN_TPB = 256


def rbc_exclusive_scan_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`thrust::exclusive_scan(counts, counts + n + 1, ex_scan, 0)`. One block.

    `ex_scan[n]` ends as the total, which is what both callers read as the
    number of edges. Note that their scan is over `n + 1` INPUTS and an
    exclusive scan never reads the last one, so only `counts[0 .. n)` is
    touched here — the same elements theirs touches.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

    var chunk = (n + RBC_SCAN_TPB - 1) // RBC_SCAN_TPB
    var begin = tid * chunk
    var end = min(begin + chunk, n)
    if begin > n:
        begin = n
    if end < begin:
        end = begin

    var total = Int32(0)
    var i = begin
    while i < end:
        total += counts.unsafe_load(i)
        i += 1

    var offset = block_prefix_sum[block_size=RBC_SCAN_TPB, exclusive=True](
        total
    )

    var running = offset
    i = begin
    while i < end:
        ex_scan.unsafe_store(i, running)
        running += counts.unsafe_load(i)
        i += 1

    if tid == RBC_SCAN_TPB - 1:
        ex_scan.unsafe_store(n, offset + total)


def rbc_exact_edge_total(mut ia: HostBuffer[DType.int32], n: Int) -> Int:
    """The EXACT edge count of an int32 exclusive scan, however far it wrapped.

    `rbc_exclusive_scan_kernel` accumulates in Int32, so `ia[n]` is the edge
    count MODULO 2^32 read as a signed number. Past 2^31 it goes negative;
    past 2^32 it comes back POSITIVE and small, and a bare `ia[n] > 2^31 - 1`
    test passes a garbage CSR (found on an L40S, taxi 4.1M x 16, eps 3: the
    true count was about 2.5e9 and `ia[n]` read -1799116104).

    Each row's degree is at most the number of index rows, below 2^31, so
    the WRAPPING Int32 difference `ia[i + 1] - ia[i]` is that degree
    exactly, whatever the offsets wrapped to. Summing the degrees in Int
    (64-bit) gives the true count. Integer addition, so the order of the
    sum cannot move a bit; it runs on the host over the `ia` copy the count
    pass already reads back, so no kernel and no vendor is involved.
    """
    var p = ia.unsafe_ptr()
    var total = 0
    for i in range(n):
        total += Int(p.unsafe_load(i + 1) - p.unsafe_load(i))
    return total


def rbc_max_reduce_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`thrust::reduce(vd, vd + n, 0, thrust::maximum<value_idx>())`, `:1453`.

    One block. `dst[0]` is the longest row the eps query produced, which is
    the `actual_max` their max_k path writes back through the host scalar.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var best = Int32(0)
    var i = tid
    while i < n:
        var v = src.unsafe_load(i)
        if v > best:
            best = v
        i += RBC_SCAN_TPB
    var m = block_max[block_size=RBC_SCAN_TPB](best)
    if tid == 0:
        dst.unsafe_store(0, m)


def rbc_clamp_kernel(
    vd: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    max_k_in: Int32,
):
    """`thrust::transform(vd, vd + n, vd, min(vd_count, max_k))`, `:1461-1467`.

    Only runs when `actual_max > max_k`, exactly as theirs does. It makes the
    CSR describe the TRUNCATED rows that `tmp` actually holds while `vd`, had
    it not been clamped, would still describe the true degrees.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * RBC_SCAN_TPB + Int(thread_idx.x)
    if i >= n:
        return
    var v = vd.unsafe_load(i)
    if v > max_k_in:
        vd.unsafe_store(i, max_k_in)


# ---------------------------------------------------------------------------
# MOJOLEARN_DBSCAN_FAST_SCAN=1 (lane/apple-fast-core, 2026-10-02): THE CSR
# OFFSET SCAN ON THE WHOLE DEVICE
# ---------------------------------------------------------------------------
# Cause: `rbc_exclusive_scan_kernel` is ONE BLOCK of `RBC_SCAN_TPB` threads
# over every query row of the batch (`registers.mojo::rbc_eps_pass_count`,
# `rbc_eps_pass_fill`'s caller and `rbc_eps_pass_max_k`): on the board's
# DBSCAN block (1,000,000 rows, one ball-cover batch) each thread folds
# about 3,900 degrees serially, twice per batch, while the rest of the GPU
# idles. Here the scan is three launches: a block-local scan of 2,048-row
# chunks with the chunk totals out, one small block scanning the chunk
# totals, and an add of each chunk's offset. Integer adds in Int32 with the
# same wrap as the one-block kernel, so `ex_scan[n]` is the same value
# (`rbc_exact_edge_total` reads it the same way). FAST + Apple only,
# default off.
comptime RBC_PSCAN_PER_THREAD = 8
comptime RBC_PSCAN_CHUNK = RBC_SCAN_TPB * RBC_PSCAN_PER_THREAD


def rbc_fast_scan_on() -> Bool:
    """Always on (lane cgr4-download-loop, 2026-10-03): the device-wide scan
    is the scan on every vendor and tier. Integer adds with the same Int32
    wrap, so every offset and `ex_scan[n]` are the one-block kernel's values;
    the one-block kernel only serves a scan of at most one chunk."""
    return True


comptime RBC_MAX_BLOCKS = 512
"""`rbc_max_reduce_launch`'s grid cap: above this many blocks' worth of rows
every thread grid-strides, and the second pass is one block over at most
this many partials."""


def rbc_max_partial_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Block `b`: the maximum (floor 0) of its grid-stride slice of
    `src[0 .. n)` into `part[b]`. An integer maximum, exact in any order."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var best = Int32(0)
    var i = Int(block_idx.x) * RBC_SCAN_TPB + tid
    var stride = RBC_MAX_BLOCKS * RBC_SCAN_TPB
    while i < n:
        var v = src.unsafe_load(i)
        if v > best:
            best = v
        i += stride
    var m = block_max[block_size=RBC_SCAN_TPB](best)
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), m)


def rbc_max_reduce_launch(
    ctx: DeviceContext,
    mut dst_buf: DeviceBuffer[DType.int32],
    mut src_buf: DeviceBuffer[DType.int32],
    n: Int,
) raises:
    """`rbc_max_reduce_kernel`'s `dst[0]` over the whole device (lane
    cgr4-download-loop): per-block maxima of a grid-stride slice (at most
    `RBC_MAX_BLOCKS` blocks, each slice strided by the full grid), then the
    one-block kernel over those partials. The same integer maximum."""
    var dst = dst_buf.unsafe_ptr()
    var src = src_buf.unsafe_ptr()
    if n <= RBC_SCAN_TPB * 8:
        ctx.enqueue_function[rbc_max_reduce_kernel](  # small-launch(n: at most 8 * RBC_SCAN_TPB rows on this branch): larger n takes the multi-block partial maxima
            dst, src, Int32(n), grid_dim=(1, 1, 1), block_dim=(RBC_SCAN_TPB, 1, 1),
        )
        return
    var part = ctx.enqueue_create_buffer[DType.int32](RBC_MAX_BLOCKS)
    ctx.enqueue_function[rbc_max_partial_kernel](
        part.unsafe_ptr(), src, Int32(n),
        grid_dim=(RBC_MAX_BLOCKS, 1, 1), block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[rbc_max_reduce_kernel](
        dst, part.unsafe_ptr(), Int32(RBC_MAX_BLOCKS),
        grid_dim=(1, 1, 1), block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    _ = part^


def rbc_pscan_local_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    chunk_tot: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Block `b`: the exclusive scan of `counts[b * CHUNK, +CHUNK)` relative
    to the chunk start into `ex_scan`, and the chunk's total into
    `chunk_tot[b]`. Every thread of the block calls the block scan."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var begin = Int(block_idx.x) * RBC_PSCAN_CHUNK + tid * RBC_PSCAN_PER_THREAD
    var end = begin + RBC_PSCAN_PER_THREAD
    if begin > n:
        begin = n
    if end > n:
        end = n
    var total = Int32(0)
    var i = begin
    while i < end:
        total += counts.unsafe_load(i)
        i += 1
    var offset = block_prefix_sum[block_size=RBC_SCAN_TPB, exclusive=True](
        total
    )
    var running = offset
    i = begin
    while i < end:
        ex_scan.unsafe_store(i, running)
        running += counts.unsafe_load(i)
        i += 1
    if tid == RBC_SCAN_TPB - 1:
        chunk_tot.unsafe_store(Int(block_idx.x), offset + total)


def rbc_pscan_chunks_kernel(
    chunk_tot: MutPointer[Int32, MutAnyOrigin],
    n_chunks_in: Int32,
):
    """One block: `chunk_tot[0 .. n_chunks)` exclusive-scanned in place,
    `chunk_tot[n_chunks]` the grand total (`rbc_exclusive_scan_kernel`'s
    shape over the chunk totals, at most a few thousand of them)."""
    var n = Int(n_chunks_in)
    var tid = Int(thread_idx.x)
    var chunk = (n + RBC_SCAN_TPB - 1) // RBC_SCAN_TPB
    var begin = tid * chunk
    var end = min(begin + chunk, n)
    if begin > n:
        begin = n
    if end < begin:
        end = begin
    var total = Int32(0)
    var i = begin
    while i < end:
        total += chunk_tot.unsafe_load(i)
        i += 1
    var offset = block_prefix_sum[block_size=RBC_SCAN_TPB, exclusive=True](
        total
    )
    var running = offset
    i = begin
    while i < end:
        var v = chunk_tot.unsafe_load(i)
        chunk_tot.unsafe_store(i, running)
        running += v
        i += 1
    if tid == RBC_SCAN_TPB - 1:
        chunk_tot.unsafe_store(n, offset + total)


def rbc_pscan_add_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    chunk_tot: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    n_chunks_in: Int32,
):
    """Thread `i < n`: `ex_scan[i] += chunk_tot[i // CHUNK]`; thread `n`
    writes the grand total, as the one-block kernel's last thread does."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        ex_scan.unsafe_store(
            i, ex_scan.unsafe_load(i) + chunk_tot.unsafe_load(i // RBC_PSCAN_CHUNK)
        )
    elif i == n:
        ex_scan.unsafe_store(n, chunk_tot.unsafe_load(Int(n_chunks_in)))


def rbc_exclusive_scan_launch(
    ctx: DeviceContext,
    mut ex_scan: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    n: Int,
) raises:
    """`rbc_exclusive_scan_kernel`'s result over `counts[0 .. n)` into
    `ex_scan[0 .. n]`: the three-launch device-wide scan, or the one-block
    kernel when n fits one chunk (lane cgr4-download-loop: the switch that
    kept the one-block kernel over every n is gone)."""
    if n > RBC_PSCAN_CHUNK:
        var n_chunks = (n + RBC_PSCAN_CHUNK - 1) // RBC_PSCAN_CHUNK
        var chunk_tot = ctx.enqueue_create_buffer[DType.int32](n_chunks + 1)
        ctx.enqueue_function[rbc_pscan_local_kernel](
            ex_scan.unsafe_ptr(),
            chunk_tot.unsafe_ptr(),
            counts.unsafe_ptr(),
            Int32(n),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(RBC_SCAN_TPB, 1, 1),
        )
        ctx.enqueue_function[rbc_pscan_chunks_kernel](
            chunk_tot.unsafe_ptr(),
            Int32(n_chunks),
            grid_dim=(1, 1, 1),
            block_dim=(RBC_SCAN_TPB, 1, 1),
        )
        ctx.enqueue_function[rbc_pscan_add_kernel](
            ex_scan.unsafe_ptr(),
            chunk_tot.unsafe_ptr(),
            Int32(n),
            Int32(n_chunks),
            grid_dim=((n + 1 + RBC_SCAN_TPB - 1) // RBC_SCAN_TPB, 1, 1),
            block_dim=(RBC_SCAN_TPB, 1, 1),
        )
        ctx.synchronize()
        _ = chunk_tot^
        return
    ctx.enqueue_function[rbc_exclusive_scan_kernel](  # small-launch(n: at most one chunk of RBC_PSCAN_CHUNK rows on this branch): larger n takes the device-wide scan
        ex_scan.unsafe_ptr(),
        counts.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(RBC_SCAN_TPB, 1, 1),
    )
