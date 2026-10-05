# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A DENSE n x n affinity straight to the Laplacian's row-sorted COO, on the
device (lane cpu2-l9-neighbors, 2026-10-04).

The dense routes of `SpectralClustering` / `SpectralEmbedding`
(`affinity='rbf'`, a dense `'precomputed'` matrix and
`'precomputed_nearest_neighbors'`) used to scan the n^2 matrix on the host
(`_DenseCOO` / `_coo_triples` through `nonzero_f32_count` / `_fill` and
`nonzero_f64_*`), check the COO values in Python, and hand the binding
three host lists, which `compute_graph_laplacian` then uploaded, marked,
sorted and offset. Here the matrix is uploaded once and:

  1. `dense_graph_count_kernel`, one block per row: the entries the
     Laplacian keeps in that row (the nonzeros, plus the diagonal always:
     `_mark_and_insert_diagonal` inserts `(i, i, 0)` where a row lacks one),
     and three flags (a non-finite value, a negative value, any nonzero).
  2. `device_exclusive_scan_total`: the row offsets, the total at `[n]`.
  3. one small readback of the flags and the total (refusal and sizing).
  4. `dense_graph_fill_kernel`, one block per row: an ordered compaction
     (a block scan per tile of columns) writing `(i, j, A[i, j])` in column
     order, the diagonal as `+0.0` when it is zero (or when `drop_diag`, the
     embedding route's `coo_remove_diagonal` followed by the insertion).
  5. `laplacian_from_sorted_device`, unchanged.

The nonzero test is on the BITS (`bits & 0x7FFFFFFF != 0`), the host scan's
`v != 0.0` exactly (-0.0 is a zero, a subnormal and NaN are not), and
independent of any vendor's subnormal flushing. Integer counts and copies
only, in the order `coo_sort` leaves (row, then column; a dense matrix has
no repeated key): the same COO, so the same Laplacian bits as the host scan
route on every vendor. The host column (the metrics host binding's oracle)
keeps the COO route."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.device_fold import device_exclusive_scan_total
from spectral.checks.device_io import download_i32
from spectral.impl.sparse.linalg.detail.laplacian import (
    DeviceCoo,
    LAPLACIAN_TPB,
    laplacian_from_sorted_device,
)

#: Threads per row block. Both kernels hold one Int32 shared page of this
#: size (1 KiB: inside every vendor's shared-memory limit, Metal's 32 KiB
#: included).
comptime DG_TPB = 256

#: `dense_graph_scan`'s readback, in this order.
comptime DG_NONFINITE = 0
comptime DG_NEGATIVE = 1
comptime DG_ANY_NONZERO = 2
comptime DG_TOTAL = 3


def dense_graph_count_kernel(
    dense: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    keep_cnt: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """One block per row: `keep_cnt[i]` = the row's nonzeros plus its
    diagonal slot; `flags[0]` a non-finite value, `flags[1]` a negative
    (finite, nonzero, sign set) value, `flags[2]` any nonzero. Integer sums
    in any order are exact; every racing flag store writes the same word."""
    comptime assert DG_TPB * 4 <= 32768, "dense_graph_count_kernel: shared page exceeds 32 KiB"
    var part = stack_allocation[DG_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x)
    var n = Int(n_in)
    var c = 0
    var nonfinite = False
    var negative = False
    var any_nz = False
    if i < n:
        var base = i * n
        var j = t
        while j < n:
            var b = bitcast[DType.uint32](dense.unsafe_load(base + j))
            var nz = (b & UInt32(0x7FFFFFFF)) != UInt32(0)
            if nz:
                any_nz = True
            if nz or j == i:
                c += 1
            if (b & UInt32(0x7F800000)) == UInt32(0x7F800000):
                nonfinite = True
            elif nz and (b & UInt32(0x80000000)) != UInt32(0):
                negative = True
            j += DG_TPB
    part[t] = Int32(c)
    barrier()
    var w = DG_TPB // 2
    while w >= 1:
        if t < w:
            part[t] = part[t] + part[t + w]
        barrier()
        w = w // 2
    if i < n:
        if t == 0:
            keep_cnt.unsafe_store(i, part[0])
        if nonfinite:
            flags.unsafe_store(DG_NONFINITE, Int32(1))
        if negative:
            flags.unsafe_store(DG_NEGATIVE, Int32(1))
        if any_nz:
            flags.unsafe_store(DG_ANY_NONZERO, Int32(1))


def dense_graph_total_kernel(
    scan: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`flags[3]` = the scan's total (the kept entries), so the flags and the
    size come back in one readback."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        flags.unsafe_store(DG_TOTAL, scan.unsafe_load(Int(n_in)))


def dense_graph_fill_kernel(
    dense: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    drop_diag: Int32,
    indptr: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
):
    """One block per row, tiles of DG_TPB columns in ascending order: each
    kept entry's slot is the row offset plus the kept entries before it
    (an inclusive Hillis-Steele scan of the tile's keep flags in shared
    memory, then the tile's total carried). The trip count is the same for
    every thread of the block, so every barrier is reached by all."""
    comptime assert DG_TPB * 4 <= 32768, "dense_graph_fill_kernel: shared page exceeds 32 KiB"
    var pre = stack_allocation[DG_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x)
    var n = Int(n_in)
    if i >= n:
        return
    var base = Int(indptr.unsafe_load(i))
    var j0 = 0
    while j0 < n:
        var j = j0 + t
        var keep = 0
        var v = Float32(0.0)
        if j < n:
            v = dense.unsafe_load(i * n + j)
            var nz = (bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)) != UInt32(0)
            if j == i:
                keep = 1
                if drop_diag != Int32(0) or not nz:
                    v = Float32(0.0)
            elif nz:
                keep = 1
        pre[t] = Int32(keep)
        barrier()
        var off = 1
        while off < DG_TPB:
            var add = pre[t - off] if t >= off else Int32(0)
            barrier()
            pre[t] = pre[t] + add
            barrier()
            off *= 2
        var incl = Int(pre[t])
        var total = Int(pre[DG_TPB - 1])
        if keep != 0:
            var p = base + incl - 1
            rows.unsafe_store(p, Int32(i))
            cols.unsafe_store(p, Int32(j))
            vals.unsafe_store(p, v)
        base += total
        barrier()
        j0 += DG_TPB


def dense_graph_check_size(n: Int) raises:
    """The COO is indexed by Int32: n^2 kept entries must fit."""
    if n <= 0:
        raise Error("spectral: a dense affinity needs n > 0")
    if n > 46340:
        raise Error(
            "spectral: a dense affinity of n=" + String(n)
            + " exceeds the Int32 COO index bound (n <= 46340)"
        )


def dense_graph_scan(
    ctx: DeviceContext,
    dense: DeviceBuffer[DType.float32],
    n: Int,
    mut scan: DeviceBuffer[DType.int32],
) raises -> List[Int32]:
    """Steps 1-3: `scan` (n + 1 slots) becomes the row offsets, and the four
    words `[non-finite, negative, any nonzero, total kept]` come back."""
    dense_graph_check_size(n)
    var flags = ctx.enqueue_create_buffer[DType.int32](4)
    ctx.enqueue_memset(flags, Int32(0))
    ctx.enqueue_function[dense_graph_count_kernel](
        dense.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), Int32(n), scan.unsafe_ptr(), flags.unsafe_ptr(),
        grid_dim=(n, 1, 1), block_dim=(DG_TPB, 1, 1),
    )
    device_exclusive_scan_total(ctx, scan, n)
    ctx.enqueue_function[dense_graph_total_kernel](  # small-launch(n: the index of the one slot read): one thread copies one word, no walk
        scan.unsafe_ptr(), flags.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var out = download_i32(ctx, flags, 4)
    _ = flags^
    return out^


def dense_graph_laplacian(
    ctx: DeviceContext,
    dense: DeviceBuffer[DType.float32],
    n: Int,
    m: Int,
    drop_diag: Bool,
    var indptr: DeviceBuffer[DType.int32],
    tpb: Int = LAPLACIAN_TPB,
) raises -> DeviceCoo:
    """Steps 4-5: `D - A` on the device from the dense matrix and the row
    offsets `dense_graph_scan` left in `indptr` (`m` = its total)."""
    if m < n:
        raise Error("spectral: dense_graph_laplacian: fewer kept entries than rows")
    var rows = ctx.enqueue_create_buffer[DType.int32](m)
    var cols = ctx.enqueue_create_buffer[DType.int32](m)
    var vals = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_function[dense_graph_fill_kernel](
        dense.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(1 if drop_diag else 0), indptr.unsafe_ptr(),
        rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(),
        grid_dim=(n, 1, 1), block_dim=(DG_TPB, 1, 1),
    )
    return laplacian_from_sorted_device(ctx, n, m, rows^, cols^, vals^, indptr^, tpb)
