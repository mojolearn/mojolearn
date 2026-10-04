# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The row-wise first-max-wins argmax of `_labels.argmax_rows` on the device
(lane cpu2-l2-labels, 2026-10-04; re-audit L2: the GPU base binding's
`argmax_rows_f32` / `argmax_rows_f64` were host loops over the read-back
score block of every classifier predict).

The rule is the host column's (`bindings/host_helpers.mojo`): scan the row
from column 0, a value replaces the best only when strictly greater, so ties
keep the lowest column, and a NaN never replaces (every comparison with it is
false), so a row whose column 0 is NaN answers 0. Here the comparison is done
on integer order keys of the IEEE words (`-0.0` and `0.0` one key, NaN
tested apart), so there is no float instruction at all: the same bytes on
NVIDIA, AMD, Apple (which has no float64) and the host column, whatever the
vendor's flush-to-zero or float64 support. One thread per row; no shared
memory (no page to gate).
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime LRD_TPB = 256
#: Rows and columns the Int32 launch arguments hold.
comptime LRD_MAX_N = 2147483000

comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + LRD_TPB - 1) // LRD_TPB if count > 0 else 1


@always_inline
def _nan32(u: UInt32) -> Bool:
    return (u & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)


@always_inline
def _key32(u: UInt32) -> UInt32:
    """Order key of a non-NaN float32 word: key(a) > key(b) iff a > b."""
    if (u & UInt32(0x7FFFFFFF)) == UInt32(0):
        return UInt32(0x80000000)
    if (u >> UInt32(31)) != UInt32(0):
        return ~u
    return u | UInt32(0x80000000)


@always_inline
def _nan64(u: UInt64) -> Bool:
    return (u & UInt64(0x7FFFFFFFFFFFFFFF)) > UInt64(0x7FF0000000000000)


@always_inline
def _key64(u: UInt64) -> UInt64:
    """Order key of a non-NaN float64 word: key(a) > key(b) iff a > b."""
    if (u & UInt64(0x7FFFFFFFFFFFFFFF)) == UInt64(0):
        return UInt64(0x8000000000000000)
    if (u >> UInt64(63)) != UInt64(0):
        return ~u
    return u | UInt64(0x8000000000000000)


def _argmax_rows_u32_kernel(src: _U32, rows_: Int32, cols_: Int32, dst: _I64):
    var r = _tid()
    if r >= Int(rows_):
        return
    var cols = Int(cols_)
    var base = r * cols
    var best = 0
    var b0 = src.unsafe_load(base)
    if not _nan32(b0):
        var best_key = _key32(b0)
        for c in range(1, cols):
            var v = src.unsafe_load(base + c)
            if _nan32(v):
                continue
            var kv = _key32(v)
            if kv > best_key:
                best = c
                best_key = kv
    dst.unsafe_store(r, Int64(best))


def _argmax_rows_u64_kernel(src: _U64, rows_: Int32, cols_: Int32, dst: _I64):
    var r = _tid()
    if r >= Int(rows_):
        return
    var cols = Int(cols_)
    var base = r * cols
    var best = 0
    var b0 = src.unsafe_load(base)
    if not _nan64(b0):
        var best_key = _key64(b0)
        for c in range(1, cols):
            var v = src.unsafe_load(base + c)
            if _nan64(v):
                continue
            var kv = _key64(v)
            if kv > best_key:
                best = c
                best_key = kv
    dst.unsafe_store(r, Int64(best))


def device_argmax_rows(
    ctx: DeviceContext, scores_addr: Int, wide: Bool, rows: Int, cols: Int, dst_addr: Int,
) raises:
    """`argmax_rows_f32` (`wide` False) / `argmax_rows_f64` (`wide` True)
    over the C-order `[rows, cols]` host block at `scores_addr`, one int64
    column index per row into `dst_addr`. `1 <= rows, cols <= LRD_MAX_N`
    (the binding checks)."""
    var total = rows * cols
    var d_dst = ctx.enqueue_create_buffer[DType.int64](rows)
    if wide:
        var d_src = ctx.enqueue_create_buffer[DType.uint64](total)
        ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U64(unsafe_from_address=scores_addr))
        ctx.enqueue_function[_argmax_rows_u64_kernel](
            d_src.unsafe_ptr(), Int32(rows), Int32(cols), d_dst.unsafe_ptr(),
            grid_dim=_blocks(rows), block_dim=LRD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_src^
    else:
        var d_src = ctx.enqueue_create_buffer[DType.uint32](total)
        ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U32(unsafe_from_address=scores_addr))
        ctx.enqueue_function[_argmax_rows_u32_kernel](
            d_src.unsafe_ptr(), Int32(rows), Int32(cols), d_dst.unsafe_ptr(),
            grid_dim=_blocks(rows), block_dim=LRD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_src^
    _ = d_dst^
