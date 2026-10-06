# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Device sparse normalization; no host densification, sorting or folds.

Public native entry: sparse_to_csr_device. Input buffers are uploaded as
bytes without host dtype conversion. Stable device radix sorts and an integer
scan form CSR, with the shared units also used by the CPU-only entry.
Capacity remains shape.entries; only the initialized nnz prefix is returned.
No runtime or compiler evidence exists for this new source yet.
"""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len, frs_exclusive_scan, frs_scan_blocks
from core.device_fold import device_count_nonzero_i32
from x_prep.sparse_input import (
    SparseShape, SparseView, SP_CSR, SP_DENSE, BP, UP, IP, FP, WP, validate_shape, validate_parts,
    sparse_pointer_unit, sparse_expand_unit, sparse_count_unit,
    sparse_write_unit, sparse_coalesce, sparse_direct_count_unit, sparse_direct_write_unit, validate_view, SP_MAX,
)

comptime SP_TPB = 128


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _blocks(n: Int) -> Int:
    return max(1, (n + SP_TPB - 1) // SP_TPB)


def _pointer_kernel(s: SparseShape, a: SparseView, bad: IP):
    var t = _tid()
    if t <= s.segments:
        sparse_pointer_unit(t, s, a, bad)


def _expand_kernel(s: SparseShape, a: SparseView, b: SparseView, v: SparseView,
                   rows: UP, cols: UP, vals: WP, bad: IP, keys: UP, order: UP):
    var t = _tid()
    if t < s.entries:
        sparse_expand_unit(t, s, a, b, v, rows, cols, vals, bad)
        keys.unsafe_store(t, cols.unsafe_load(t) if sparse_coalesce(s) else rows.unsafe_load(t))
        order.unsafe_store(t, UInt32(t))


def _row_keys_kernel(rows: UP, order: UP, keys: UP, n: Int):
    var t = _tid()
    if t < n:
        keys.unsafe_store(t, rows.unsafe_load(Int(order.unsafe_load(t))))


def _count_kernel(s: SparseShape, keys: UP, order: UP, cols: UP, counts: IP):
    var t = _tid()
    if t <= s.rows:
        sparse_count_unit(t, s, keys, order, cols, counts)


def _write_kernel(s: SparseShape, v: SparseView, keys: UP, order: UP, cols: UP, vals: WP,
                  indptr: IP, indices: IP, data: FP):
    var t = _tid()
    if t < s.rows:
        sparse_write_unit(t, s, v, keys, order, cols, vals, indptr, indices, data)


def _upload(ctx: DeviceContext, v: SparseView, span: Int) raises -> DeviceBuffer[DType.uint8]:
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(span, 1))
    if span > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=v.data)
    return buf^


def _direct_count_kernel(s: SparseShape, a: SparseView, v: SparseView, counts: IP):
    var r = _tid()
    if r <= s.rows:
        sparse_direct_count_unit(r, s, a, v, counts)


def _direct_write_kernel(s: SparseShape, b: SparseView, v: SparseView, indptr: IP, indices: IP,
                         data: FP, bad: IP):
    var r = _tid()
    if r < s.rows:
        sparse_direct_write_unit(r, s, b, v, indptr, indices, data, bad)


def _direct_offsets(ctx: DeviceContext, s: SparseShape, a: SparseView, v: SparseView) raises -> DeviceBuffer[DType.int32]:
    var dp = ctx.enqueue_create_buffer[DType.int32](s.rows + 1)
    ctx.enqueue_function[_direct_count_kernel](s, a, v, dp.unsafe_ptr(), grid_dim=_blocks(s.rows + 1), block_dim=SP_TPB)
    if s.kind == SP_DENSE:
        var scan = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(s.rows + 1))
        frs_exclusive_scan(ctx, dp, s.rows + 1, scan)
    return dp^


def _nnz(ctx: DeviceContext, mut dp: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    var word = Int32(0)
    ctx.enqueue_copy(dst_ptr=MutPointer(to=word), src_buf=dp.create_sub_buffer[DType.int32](n, 1))
    ctx.synchronize()
    return Int(word)


def sparse_dense_nnz_device(ctx: DeviceContext, s: SparseShape, v: SparseView, span: Int) raises -> Int:
    validate_shape(s)
    validate_view(v)
    if s.kind != SP_DENSE or v.items != s.entries:
        raise Error("sparse input: dense capacity expects a matching dense buffer")
    var dv = _upload(ctx, v, span)
    var vv = v
    vv.data = dv.unsafe_ptr()
    var dp = _direct_offsets(ctx, s, vv, vv)
    return _nnz(ctx, dp, s.rows)


def sparse_to_csr_device(ctx: DeviceContext, s: SparseShape, a: SparseView, a_span: Int,
                         b: SparseView, b_span: Int, v: SparseView, v_span: Int,
                         indptr: IP, indices: IP, data: FP, capacity: Int) raises -> Int:
    """Input views address host byte spans; outputs are caller-owned host buffers.

    All data computation, validation, compaction and dtype conversion execute
    on the device. Only scalar status/nnz and completed CSR arrays come back.
    """
    validate_parts(s, a, b, v)
    var da = _upload(ctx, a, a_span)
    var db = _upload(ctx, b, b_span)
    var dv = _upload(ctx, v, v_span)
    var av = a
    var bv = b
    var vv = v
    av.data = da.unsafe_ptr()
    bv.data = db.unsafe_ptr()
    vv.data = dv.unsafe_ptr()
    var pbad = ctx.enqueue_create_buffer[DType.int32](s.segments + 1)
    ctx.enqueue_function[_pointer_kernel](s, av, pbad.unsafe_ptr(), grid_dim=_blocks(s.segments + 1), block_dim=SP_TPB)
    if device_count_nonzero_i32(ctx, pbad, s.segments + 1) != 0:
        raise Error("sparse input: invalid compressed row/column pointers")
    if s.kind == SP_CSR or s.kind == SP_DENSE:
        var dp = _direct_offsets(ctx, s, av, vv)
        var nnz = _nnz(ctx, dp, s.rows)
        if nnz > capacity:
            raise Error("sparse input: CSR output capacity is too small")
        var di = ctx.enqueue_create_buffer[DType.int32](max(nnz, 1))
        var dd = ctx.enqueue_create_buffer[DType.float32](max(nnz, 1))
        var bad = ctx.enqueue_create_buffer[DType.int32](s.rows)
        ctx.enqueue_function[_direct_write_kernel](s, bv, vv, dp.unsafe_ptr(), di.unsafe_ptr(), dd.unsafe_ptr(), bad.unsafe_ptr(),
            grid_dim=_blocks(s.rows), block_dim=SP_TPB)
        if device_count_nonzero_i32(ctx, bad, s.rows) != 0:
            raise Error("sparse input: invalid index or value outside signed-int64 range")
        ctx.enqueue_copy(dst_ptr=indptr, src_buf=dp)
        if nnz > 0:
            ctx.enqueue_copy(dst_ptr=indices, src_buf=di.create_sub_buffer[DType.int32](0, nnz))
            ctx.enqueue_copy(dst_ptr=data, src_buf=dd.create_sub_buffer[DType.float32](0, nnz))
        ctx.synchronize()
        return nnz
    if capacity < s.entries:
        raise Error("sparse input: CSR output capacity is too small")
    if frs_counts_len(s.entries) > SP_MAX:
        raise Error("sparse input: radix workspace exceeds Int32 indexing")
    var cap = max(s.entries, 1)
    var rows = ctx.enqueue_create_buffer[DType.uint32](cap)
    var cols = ctx.enqueue_create_buffer[DType.uint32](cap)
    var vals = ctx.enqueue_create_buffer[DType.uint64](cap)
    var bad = ctx.enqueue_create_buffer[DType.int32](cap)
    var keys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var order = ctx.enqueue_create_buffer[DType.uint32](cap)
    var temp_keys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var temp_order = ctx.enqueue_create_buffer[DType.uint32](cap)
    var counts = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(s.entries), 1))
    ctx.enqueue_function[_expand_kernel](s, av, bv, vv, rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(),
        bad.unsafe_ptr(), keys.unsafe_ptr(), order.unsafe_ptr(), grid_dim=_blocks(s.entries), block_dim=SP_TPB)
    if device_count_nonzero_i32(ctx, bad, s.entries) != 0:
        raise Error("sparse input: invalid index or value outside signed-int64 range")
    fast_radix_sort_pairs_u32(ctx, s.entries, keys, order, temp_keys, temp_order, counts)
    if sparse_coalesce(s):
        ctx.enqueue_function[_row_keys_kernel](rows.unsafe_ptr(), order.unsafe_ptr(), keys.unsafe_ptr(), s.entries,
            grid_dim=_blocks(s.entries), block_dim=SP_TPB)
        fast_radix_sort_pairs_u32(ctx, s.entries, keys, order, temp_keys, temp_order, counts)
    var dp = ctx.enqueue_create_buffer[DType.int32](s.rows + 1)
    var di = ctx.enqueue_create_buffer[DType.int32](cap)
    var dd = ctx.enqueue_create_buffer[DType.float32](cap)
    var scan = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(s.rows + 1))
    ctx.enqueue_function[_count_kernel](s, keys.unsafe_ptr(), order.unsafe_ptr(), cols.unsafe_ptr(), dp.unsafe_ptr(),
        grid_dim=_blocks(s.rows + 1), block_dim=SP_TPB)
    frs_exclusive_scan(ctx, dp, s.rows + 1, scan)
    var nnz_word = Int32(0)
    ctx.enqueue_copy(dst_ptr=MutPointer(to=nnz_word), src_buf=dp.create_sub_buffer[DType.int32](s.rows, 1))
    ctx.synchronize()
    var nnz = Int(nnz_word)
    ctx.enqueue_function[_write_kernel](s, vv, keys.unsafe_ptr(), order.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(),
        dp.unsafe_ptr(), di.unsafe_ptr(), dd.unsafe_ptr(), grid_dim=_blocks(s.rows), block_dim=SP_TPB)
    ctx.enqueue_copy(dst_ptr=indptr, src_buf=dp)
    if nnz > 0:
        ctx.enqueue_copy(dst_ptr=indices, src_buf=di.create_sub_buffer[DType.int32](0, nnz))
        ctx.enqueue_copy(dst_ptr=data, src_buf=dd.create_sub_buffer[DType.float32](0, nnz))
    ctx.synchronize()
    return nnz
