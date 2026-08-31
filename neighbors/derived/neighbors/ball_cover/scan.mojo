# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""The device exclusive scan `rbc_eps_pass` needs, and a max-reduce.

STANDS IN FOR `thrust::exclusive_scan` at
`cuvs/src/neighbors/ball_cover/registers.cuh:1376,1470` and for
`thrust::reduce(..., thrust::maximum<value_idx>())` at `:1453`.

WHY THIS IS HAND-WRITTEN
------------------------
Thrust is open, so `exclusive_scan` is a port candidate rather than a
substitution candidate. Its structure here is trivial and is copied: theirs
is a single device-wide scan launched from the host between two kernel
launches (`:1354` then `:1376` then `:1385`), reading its input from global
memory and writing its output to global memory. There is nothing in it to
fuse and nothing in it that is not a prefix sum.

There would in any case be nothing to substitute: `nn.cumsum.cumsum` and
`max.algorithm.reduction.cumsum` both take neither a `DeviceContext` nor a
`target`, which `VENDOR_LIBRARIES.md` records as the signature of a HOST-ONLY
entry point, and it lists no device scan anywhere in the shipped kernel
libraries.

The shape is `dbscan/gbdt/dbscan/adjgraph/algo.mojo::exclusive_scan_kernel`
copied deliberately rather than imported: another lane owns that file this
round and this one must not depend on its signature. The dynamic chunk is
theirs and is load-bearing — a FIXED rows-per-thread silently stopped
scanning past `SCAN_TPB * chunk` elements and produced a truncated CSR, which
that file records as a bug found by audit rather than by a test.
"""

from std.gpu import block_idx, thread_idx
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.block import max as block_max


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
