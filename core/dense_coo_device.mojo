# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core/dense_coo.mojo::knn_affinity_f32` on the device (cpu-gpu-cleanup
c-core, 2026-10-02): the base binding's GPU route ran the host body over
host threads. The same answer, which depends only on the set of each row's
k smallest (value, column) candidates, never on the order they are visited:

  1. sparse only: per-row candidate counts (an integer atomic add per COO
     entry; an entry outside [0, n) raises a flag), their exclusive scan
     (`frs_exclusive_scan`) and a scatter of the entry ids into row groups.
     The order inside a group is arrival order; the selection below keys on
     (value bits, column), so any order gives the same set.
  2. one thread per row: a NaN or negative candidate marks the row kind 1,
     fewer than k candidates kind 2; otherwise the k smallest keys by
     insertion into a k-slot buffer, and C[i, j] = 1 for each.
  3. the first failing row in row order (an atomic min of the row id) and
     its kind go to `status`, as the host body's row-order check does.
  4. one thread per cell: A[i, j] = 0.5 (C + C^T) in {0, 0.5, 1}, written
     only when no row failed (the host body returns before writing).
Compares, integers and the exact values 0, 0.5 and 1: the same bytes as the
host body (`bindings/_mojolearn_core_host.mojo` keeps it)."""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.fast_radix_sort import frs_exclusive_scan, frs_scan_blocks
from core.device_zero import enqueue_fill

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]
comptime _K = MutPointer[UInt64, MutAnyOrigin]
comptime _B = MutPointer[UInt8, MutAnyOrigin]
comptime _TPB = 128


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


@always_inline
def _key(v: Float32, j: Int) -> UInt64:
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    return (UInt64(Int(b)) << 32) | UInt64(j)


def _coo_count_kernel(rows: _I, cols: _I, nnz: Int32, n: Int32, cnt: _I, flag: _I):
    var e = _tid()
    if e < Int(nnz):
        var r = Int(rows.unsafe_load(e))
        var c = Int(cols.unsafe_load(e))
        if r < 0 or r >= Int(n) or c < 0 or c >= Int(n):
            flag.unsafe_store(0, Int32(1))
            return
        _ = Atomic[DType.int32].fetch_add(cnt + r, Int32(1))


def _coo_scatter_kernel(rows: _I, cols: _I, nnz: Int32, n: Int32, start: _I, fill: _I, order: _I):
    var e = _tid()
    if e < Int(nnz):
        var r = Int(rows.unsafe_load(e))
        var c = Int(cols.unsafe_load(e))
        if r < 0 or r >= Int(n) or c < 0 or c >= Int(n):
            return
        var slot = Atomic[DType.int32].fetch_add(fill + r, Int32(1))
        order.unsafe_store(Int(start.unsafe_load(r)) + Int(slot), Int32(e))


def _knn_rows_kernel(
    dense: _F, cols: _I, vals: _F, start: _I, cnt: _I, order: _I, sparse: Int32, n_: Int32, k_: Int32,
    sel: _K, cmat: _B, bad: _I, first: _I,
):
    var i = _tid()
    var n = Int(n_)
    if i >= n:
        return
    var k = Int(k_)
    var m = Int(cnt.unsafe_load(i)) if sparse != 0 else n
    var s0 = Int(start.unsafe_load(i)) if sparse != 0 else 0
    var buf = sel + i * k
    var have = 0
    for q in range(m):
        var v: Float32
        var j: Int
        if sparse != 0:
            var e = Int(order.unsafe_load(s0 + q))
            v = vals.unsafe_load(e)
            j = Int(cols.unsafe_load(e))
        else:
            v = dense.unsafe_load(i * n + q)
            j = q
        if v != v or v < Float32(0):
            bad.unsafe_store(i, Int32(1))
            _ = Atomic[DType.int32].min(first, Int32(i))
            return
        if k == 0:
            continue
        var key = _key(v, j)
        if have == k and key >= buf.unsafe_load(k - 1):
            continue
        var p = have if have < k else k - 1
        while p > 0 and buf.unsafe_load(p - 1) > key:
            buf.unsafe_store(p, buf.unsafe_load(p - 1))
            p -= 1
        buf.unsafe_store(p, key)
        if have < k:
            have += 1
    if m < k:
        bad.unsafe_store(i, Int32(2))
        _ = Atomic[DType.int32].min(first, Int32(i))
        return
    for q in range(k):
        var j = Int(buf.unsafe_load(q) & UInt64(0xFFFFFFFF))
        cmat.unsafe_store(i * n + j, UInt8(1))


def _status_kernel(bad: _I, first: _I, n_: Int32, status: _I):
    if _tid() == 0:
        var f = Int(first.unsafe_load(0))
        if f < Int(n_):
            status.unsafe_store(0, bad.unsafe_load(f))
            status.unsafe_store(1, Int32(f))
        else:
            status.unsafe_store(0, Int32(0))
            status.unsafe_store(1, Int32(0))


def _aff_kernel(cmat: _B, n_: Int32, status: _I, aff: _F):
    var t = _tid()
    var n = Int(n_)
    if t >= n * n or status.unsafe_load(0) != Int32(0):
        return
    var i = t // n
    var j = t - i * n
    var c = Int(cmat.unsafe_load(i * n + j)) + Int(cmat.unsafe_load(j * n + i))
    aff.unsafe_store(t, Float32(0.5) if c == 1 else (Float32(1) if c == 2 else Float32(0)))


def knn_affinity_f32_device(
    ctx: DeviceContext, dense: Int, rows: Int, cols: Int, vals: Int, nnz: Int, sparse: Bool,
    n: Int, k: Int, aff: Int, status: Int,
) raises:
    """`knn_affinity_f32`'s contract (the caller's zeroed host `aff`, n x n,
    and `status`, 2 int32) with the work on the device."""
    var nn = n * n
    var d_dense = ctx.enqueue_create_buffer[DType.float32](max(0 if sparse else nn, 1))
    var d_rows = ctx.enqueue_create_buffer[DType.int32](max(nnz if sparse else 0, 1))
    var d_cols = ctx.enqueue_create_buffer[DType.int32](max(nnz if sparse else 0, 1))
    var d_vals = ctx.enqueue_create_buffer[DType.float32](max(nnz if sparse else 0, 1))
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_start = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_fill = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_order = ctx.enqueue_create_buffer[DType.int32](max(nnz if sparse else 0, 1))
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    var d_bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(max(n, 1)))
    var d_sel = ctx.enqueue_create_buffer[DType.uint64](max(n * k, 1))
    var d_c = ctx.enqueue_create_buffer[DType.uint8](max(nn, 1))
    var d_bad = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_first = ctx.enqueue_create_buffer[DType.int32](1)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    var d_aff = ctx.enqueue_create_buffer[DType.float32](max(nn, 1))
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_fill, Int32(0))
    enqueue_fill(ctx, d_flag, Int32(0))
    enqueue_fill(ctx, d_c, UInt8(0))
    enqueue_fill(ctx, d_bad, Int32(0))
    enqueue_fill(ctx, d_first, Int32(n))
    enqueue_fill(ctx, d_aff, Float32(0))
    if sparse:
        if nnz > 0:
            ctx.enqueue_copy(dst_buf=d_rows, src_ptr=_I(unsafe_from_address=rows))
            ctx.enqueue_copy(dst_buf=d_cols, src_ptr=_I(unsafe_from_address=cols))
            ctx.enqueue_copy(dst_buf=d_vals, src_ptr=_F(unsafe_from_address=vals))
            ctx.enqueue_function[_coo_count_kernel](
                d_rows.unsafe_ptr(), d_cols.unsafe_ptr(), Int32(nnz), Int32(n), d_cnt.unsafe_ptr(),
                d_flag.unsafe_ptr(), grid_dim=_blocks(nnz), block_dim=_TPB,
            )
        ctx.enqueue_copy(dst_buf=d_start, src_buf=d_cnt)
        frs_exclusive_scan(ctx, d_start, n, d_bsum)
        if nnz > 0:
            ctx.enqueue_function[_coo_scatter_kernel](
                d_rows.unsafe_ptr(), d_cols.unsafe_ptr(), Int32(nnz), Int32(n), d_start.unsafe_ptr(),
                d_fill.unsafe_ptr(), d_order.unsafe_ptr(), grid_dim=_blocks(nnz), block_dim=_TPB,
            )
        var hf = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=hf.unsafe_ptr(), src_buf=d_flag)
        ctx.synchronize()
        if hf.unsafe_ptr()[0] != 0:
            raise Error("knn_affinity_f32: a COO entry lies outside the square graph")
        _ = hf^
    else:
        ctx.enqueue_copy(dst_buf=d_dense, src_ptr=_F(unsafe_from_address=dense))
    ctx.enqueue_function[_knn_rows_kernel](
        d_dense.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), d_start.unsafe_ptr(), d_cnt.unsafe_ptr(),
        d_order.unsafe_ptr(), Int32(1 if sparse else 0), Int32(n), Int32(k), d_sel.unsafe_ptr(), d_c.unsafe_ptr(),
        d_bad.unsafe_ptr(), d_first.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    ctx.enqueue_function[_status_kernel](
        d_bad.unsafe_ptr(), d_first.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(), grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[_aff_kernel](
        d_c.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(), d_aff.unsafe_ptr(), grid_dim=_blocks(nn), block_dim=_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I(unsafe_from_address=status), src_buf=d_status)
    ctx.enqueue_copy(dst_ptr=_F(unsafe_from_address=aff), src_buf=d_aff)
    ctx.synchronize()
    _ = d_dense^
    _ = d_rows^
    _ = d_cols^
    _ = d_vals^
    _ = d_cnt^
    _ = d_start^
    _ = d_fill^
    _ = d_order^
    _ = d_flag^
    _ = d_bsum^
    _ = d_sel^
    _ = d_c^
    _ = d_bad^
    _ = d_first^
    _ = d_status^
    _ = d_aff^
