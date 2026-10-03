# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core/label_encode.mojo::host_unique_inverse` on the device
(cpu-gpu-cleanup w2-pyglue, 2026-10-02): sorted unique classes plus the
inverse code of every row, for one numeric label buffer.

The steps are the host twin's, one launch (or one sort) each:

  1. KEY: one thread per row writes the key's low and high 32-bit halves
     and the row id; a NaN sets the status word (every writer stores 1).
  2. SORT: `fast_radix_sort_pairs_u32` by the low half, the high half
     gathered through that order, then a second stable sort by the high
     half. Two stable sorts, low half first, are the stable sort by the
     full 64-bit key, so the order is the host twin's eight digit passes'.
  3. FLAG: one thread per sorted position, 1 where the full key differs
     from the previous position's.
  4. SCAN: `frs_exclusive_scan` (multi-block) of the flags.
  5. EMIT: one thread per sorted position scatters the class id to its row
     and, where flagged, the row's own 64 bits to the class slot; the last
     position writes the class count.
Integers and bit moves only: the same bytes as the host twin."""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.fast_radix_sort import (
    fast_radix_sort_pairs_u32,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)
from core.device_zero import enqueue_fill
from core.label_encode import label_is_nan, label_sort_key

comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]
comptime _TPB = 256


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _key_kernel(src: _U64, n_: Int32, kind: Int32, lo: _U32, hi: _U32, row: _U32, status: _I):
    var i = _tid()
    if i >= Int(n_):
        return
    var b = src.unsafe_load(i)
    if label_is_nan(b, Int(kind)):
        status.unsafe_store(0, Int32(1))
    var k = label_sort_key(b, Int(kind))
    lo.unsafe_store(i, UInt32(k & UInt64(0xFFFFFFFF)))
    hi.unsafe_store(i, UInt32(k >> UInt64(32)))
    row.unsafe_store(i, UInt32(i))


def _gather_u32_kernel(src: _U32, order: _U32, n_: Int32, dst: _U32):
    var j = _tid()
    if j < Int(n_):
        dst.unsafe_store(j, src.unsafe_load(Int(order.unsafe_load(j))))


def _flag_kernel(hi_s: _U32, lo: _U32, row: _U32, n_: Int32, flag: _I, scan: _I):
    var j = _tid()
    if j >= Int(n_):
        return
    var f = Int32(1)
    if j > 0:
        var same_hi = hi_s.unsafe_load(j) == hi_s.unsafe_load(j - 1)
        var same_lo = lo.unsafe_load(Int(row.unsafe_load(j))) == lo.unsafe_load(Int(row.unsafe_load(j - 1)))
        if same_hi and same_lo:
            f = Int32(0)
    flag.unsafe_store(j, f)
    scan.unsafe_store(j, f)


def _emit_kernel(
    src: _U64, row: _U32, flag: _I, scan: _I, n_: Int32, codes: _I, classes: _U64, count: _I,
):
    var j = _tid()
    var n = Int(n_)
    if j >= n:
        return
    var f = flag.unsafe_load(j)
    var cls = scan.unsafe_load(j) + f - Int32(1)
    var r = Int(row.unsafe_load(j))
    codes.unsafe_store(r, cls)
    if f == Int32(1):
        classes.unsafe_store(Int(cls), src.unsafe_load(r))
    if j == n - 1:
        count.unsafe_store(0, cls + Int32(1))


def device_unique_inverse(
    ctx: DeviceContext, src_addr: Int, n: Int, kind: Int, classes_addr: Int, codes_addr: Int,
) raises -> Int:
    """`host_unique_inverse`'s contract (host `src` of n 64-bit labels,
    host `classes` of n slots, host `codes` of n int32) with every step on
    the device. Returns the class count, or -2 for a NaN label."""
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_lo = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_hi = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_his = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_row = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tk = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tv = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_lo2 = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(n))
    var d_flag = ctx.enqueue_create_buffer[DType.int32](n)
    var d_scan = ctx.enqueue_create_buffer[DType.int32](n)
    var d_bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(n))
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_classes = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U64(unsafe_from_address=src_addr))
    ctx.enqueue_function[_key_kernel](
        d_src.unsafe_ptr(), Int32(n), Int32(kind), d_lo.unsafe_ptr(), d_hi.unsafe_ptr(),
        d_row.unsafe_ptr(), d_status.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    # the low half sorts a copy, so `d_lo` stays in row order for the flag step
    ctx.enqueue_copy(dst_buf=d_lo2, src_buf=d_lo)
    fast_radix_sort_pairs_u32(ctx, n, d_lo2, d_row, d_tk, d_tv, d_cnt)
    ctx.enqueue_function[_gather_u32_kernel](
        d_hi.unsafe_ptr(), d_row.unsafe_ptr(), Int32(n), d_his.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, n, d_his, d_row, d_tk, d_tv, d_cnt)
    ctx.enqueue_function[_flag_kernel](
        d_his.unsafe_ptr(), d_lo.unsafe_ptr(), d_row.unsafe_ptr(), Int32(n), d_flag.unsafe_ptr(),
        d_scan.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    frs_exclusive_scan(ctx, d_scan, n, d_bsum)
    ctx.enqueue_function[_emit_kernel](
        d_src.unsafe_ptr(), d_row.unsafe_ptr(), d_flag.unsafe_ptr(), d_scan.unsafe_ptr(), Int32(n),
        d_codes.unsafe_ptr(), d_classes.unsafe_ptr(), d_status.unsafe_ptr() + 1,
        grid_dim=_blocks(n), block_dim=_TPB,
    )
    var h_status = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h_status.unsafe_ptr(), src_buf=d_status)
    ctx.enqueue_copy(dst_ptr=_I(unsafe_from_address=codes_addr), src_buf=d_codes)
    ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=classes_addr), src_buf=d_classes)
    ctx.synchronize()
    var nan = h_status.unsafe_ptr()[0] != Int32(0)
    var k = Int(h_status.unsafe_ptr()[1])
    _ = h_status^
    _ = d_src^
    _ = d_lo^
    _ = d_hi^
    _ = d_his^
    _ = d_row^
    _ = d_tk^
    _ = d_tv^
    _ = d_lo2^
    _ = d_cnt^
    _ = d_flag^
    _ = d_scan^
    _ = d_bsum^
    _ = d_codes^
    _ = d_classes^
    _ = d_status^
    return -2 if nan else k
