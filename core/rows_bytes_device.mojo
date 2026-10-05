# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Row byte gathers and scatters on the device (lane cpu4-python).

The GPU base binding's `gather_rows_bytes` / `scatter_rows_bytes` were host
memcpy walks over the rows. Here the rows move on the device: the indices
are checked by a kernel (one flag word back; nothing is written on a bad
index, as before), then one thread per byte moves it. A scatter keeps the
host loop's rule for a repeated row: the LAST source row naming it wins
(an `Atomic.max` of the source position per target row, then only that
position writes). Byte moves only: the same bytes on every vendor. The host
walks stay the host bindings' columns (`bindings/host_helpers.mojo`,
`bindings/hotpath_helpers.mojo`).
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

comptime _TPB = 256
comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _HU8 = MutPointer[UInt8, MutUntrackedOrigin]
comptime _HI64 = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _check_rows_kernel(idx: _I64, m_: Int32, bound: Int64, flag: _I32):
    var i = _tid()
    if i < Int(m_):
        var r = idx.unsafe_load(i)
        if r < 0 or r >= bound:
            flag.unsafe_store(0, Int32(1))


def _gather_bytes_kernel(src: _U8, idx: _I64, m_: Int32, width_: Int32, dst: _U8):
    """dst[r, b] = src[idx[r], b] over m rows of width bytes."""
    var t = _tid()
    var width = Int(width_)
    if t >= Int(m_) * width:
        return
    var r = t // width
    var b = t - r * width
    dst.unsafe_store(t, src.unsafe_load(Int(idx.unsafe_load(r)) * width + b))


def _last_writer_kernel(idx: _I64, m_: Int32, owner: _I32):
    """owner[idx[i]] = the largest i naming that row (the host loop's last)."""
    var i = _tid()
    if i < Int(m_):
        _ = Atomic[DType.int32].max(owner + Int(idx.unsafe_load(i)), Int32(i))


def _scatter_bytes_kernel(src: _U8, idx: _I64, m_: Int32, width_: Int32, owner: _I32, dst: _U8):
    """dst[idx[i], b] = src[i, b] when i is the last source naming the row."""
    var t = _tid()
    var width = Int(width_)
    if t >= Int(m_) * width:
        return
    var i = t // width
    var b = t - i * width
    var r = Int(idx.unsafe_load(i))
    if Int(owner.unsafe_load(r)) == i:
        dst.unsafe_store(r * width + b, src.unsafe_load(t))


def _rows_ok(ctx: DeviceContext, mut idx: DeviceBuffer[DType.int64], m: Int, bound: Int) raises -> Bool:
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[_check_rows_kernel](
        idx.unsafe_ptr(), Int32(m), Int64(bound), flag.unsafe_ptr(), grid_dim=_blocks(m), block_dim=_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var ok = h.unsafe_ptr().unsafe_load(0) == Int32(0)
    _ = h^
    _ = flag^
    return ok


def device_gather_rows_bytes(
    ctx: DeviceContext, src_addr: Int, dst_addr: Int, idx_addr: Int, ns: Int, no: Int, width: Int,
) raises -> Bool:
    """`gather_rows_bytes` (no, width >= 1; ns * width and no * width within
    Int32): False, nothing written, when an index is outside [0, ns)."""
    var idx = ctx.enqueue_create_buffer[DType.int64](no)
    ctx.enqueue_copy(dst_buf=idx, src_ptr=_HI64(unsafe_from_address=idx_addr))
    if not _rows_ok(ctx, idx, no, ns):
        _ = idx^
        return False
    var src = ctx.enqueue_create_buffer[DType.uint8](max(ns * width, 1))
    if ns > 0:
        ctx.enqueue_copy(dst_buf=src, src_ptr=_HU8(unsafe_from_address=src_addr))
    var dst = ctx.enqueue_create_buffer[DType.uint8](no * width)
    ctx.enqueue_function[_gather_bytes_kernel](
        src.unsafe_ptr(), idx.unsafe_ptr(), Int32(no), Int32(width), dst.unsafe_ptr(),
        grid_dim=_blocks(no * width), block_dim=_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_HU8(unsafe_from_address=dst_addr), src_buf=dst)
    ctx.synchronize()
    _ = dst^
    _ = src^
    _ = idx^
    return True


def device_scatter_rows_bytes(
    ctx: DeviceContext, src_addr: Int, rows_addr: Int, m: Int, width: Int, dst_addr: Int, nd: Int,
) raises -> Bool:
    """`scatter_rows_bytes` (m, width, nd >= 1; nd * width and m * width
    within Int32): False, nothing written, when a row is outside [0, nd).
    The rows not named keep their bytes (dst goes up and comes back)."""
    var idx = ctx.enqueue_create_buffer[DType.int64](m)
    ctx.enqueue_copy(dst_buf=idx, src_ptr=_HI64(unsafe_from_address=rows_addr))
    if not _rows_ok(ctx, idx, m, nd):
        _ = idx^
        return False
    var owner = ctx.enqueue_create_buffer[DType.int32](nd)
    ctx.enqueue_memset(owner, Int32(-1))
    ctx.enqueue_function[_last_writer_kernel](
        idx.unsafe_ptr(), Int32(m), owner.unsafe_ptr(), grid_dim=_blocks(m), block_dim=_TPB,
    )
    var src = ctx.enqueue_create_buffer[DType.uint8](m * width)
    ctx.enqueue_copy(dst_buf=src, src_ptr=_HU8(unsafe_from_address=src_addr))
    var dst = ctx.enqueue_create_buffer[DType.uint8](nd * width)
    ctx.enqueue_copy(dst_buf=dst, src_ptr=_HU8(unsafe_from_address=dst_addr))
    ctx.enqueue_function[_scatter_bytes_kernel](
        src.unsafe_ptr(), idx.unsafe_ptr(), Int32(m), Int32(width), owner.unsafe_ptr(), dst.unsafe_ptr(),
        grid_dim=_blocks(m * width), block_dim=_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_HU8(unsafe_from_address=dst_addr), src_buf=dst)
    ctx.synchronize()
    _ = dst^
    _ = src^
    _ = owner^
    _ = idx^
    return True
