# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA's per-series criterion and running first-minimum order choice
on the device (lane cpu4-python).

The GPU base binding's `ic_running_min_f64` / `ic_running_min_f32` walked
the series on the host. One thread per series here, the host column's
statements (`bindings/hotpath_helpers.mojo`): ic = -2 llf + penalty, then
`np.argmin`'s running rule (the first minimum; a NaN taken as the minimum,
the first NaN wins). The float64 form is soft binary64
(`checks/soft_f64.mojo`: correctly rounded, integer instructions, so Apple
too); -2 llf is exact, so the one rounding is the add, the host's. A NaN
criterion is the canonical quiet NaN on both columns (the host column
canonicalizes it the same way since this lane).
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext

from checks.numerics import ftz
from checks.soft_f64 import SF64_NAN, sf64_add, sf64_from_f32, sf64_is_nan, sf64_lt, sf64_mul

comptime _TPB = 256
comptime _F32 = MutPointer[Float32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _HF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _HU64 = MutPointer[UInt64, MutUntrackedOrigin]
comptime _HI64 = MutPointer[Int64, MutUntrackedOrigin]
#: -2.0 as an IEEE binary64 word
comptime _NEG_TWO = UInt64(0xC000000000000000)


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _ic_min_f64_kernel(llf: _F32, n_: Int32, pen_lo: UInt32, pen_hi: UInt32, k_: Int32,
                       ic: _U64, best: _U64, idx: _I64):
    var b = _tid()
    if b >= Int(n_):
        return
    var pen = (UInt64(pen_hi) << 32) | UInt64(pen_lo)
    var v = sf64_add(sf64_mul(_NEG_TWO, sf64_from_f32(llf.unsafe_load(b))), pen)
    if sf64_is_nan(v):
        v = SF64_NAN
    ic.unsafe_store(b, v)
    if Int(k_) == 0:
        best.unsafe_store(b, v)
        idx.unsafe_store(b, Int64(0))
        return
    var cur = best.unsafe_load(b)
    var take = False
    if not sf64_is_nan(cur):
        take = sf64_is_nan(v) or sf64_lt(v, cur)
    if take:
        best.unsafe_store(b, v)
        idx.unsafe_store(b, Int64(Int(k_)))


def _ic_min_f32_kernel(llf: _F32, n_: Int32, pen: Float32, k_: Int32, ic: _F32, best: _F32, idx: _I64):
    var b = _tid()
    if b >= Int(n_):
        return
    var v = ftz(Float32(-2.0) * ftz(llf.unsafe_load(b)) + pen)
    if v != v:
        v = bitcast[DType.float32](UInt32(0x7FC00000))
    ic.unsafe_store(b, v)
    if Int(k_) == 0:
        best.unsafe_store(b, v)
        idx.unsafe_store(b, Int64(0))
        return
    var cur = best.unsafe_load(b)
    var take = False
    if cur == cur:
        take = (v != v) or v < cur
    if take:
        best.unsafe_store(b, v)
        idx.unsafe_store(b, Int64(Int(k_)))


def device_ic_running_min(
    ctx: DeviceContext, wide: Bool, llf_addr: Int, n: Int, penalty: Float64, k: Int,
    ic_addr: Int, best_addr: Int, idx_addr: Int,
) raises:
    """`ic_running_min_f64` (`wide`) / `_f32` over n >= 1 series: the
    running best and its order come from (and return to) the caller's
    arrays (order 0 initialises them)."""
    var llf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=llf, src_ptr=_HF32(unsafe_from_address=llf_addr))
    var idx = ctx.enqueue_create_buffer[DType.int64](n)
    if k > 0:
        ctx.enqueue_copy(dst_buf=idx, src_ptr=_HI64(unsafe_from_address=idx_addr))
    if wide:
        var ic = ctx.enqueue_create_buffer[DType.uint64](n)
        var best = ctx.enqueue_create_buffer[DType.uint64](n)
        if k > 0:
            ctx.enqueue_copy(dst_buf=best, src_ptr=_HU64(unsafe_from_address=best_addr))
        var pb = bitcast[DType.uint64](penalty)
        ctx.enqueue_function[_ic_min_f64_kernel](
            llf.unsafe_ptr(), Int32(n), UInt32(pb & UInt64(0xFFFFFFFF)), UInt32(pb >> 32), Int32(k),
            ic.unsafe_ptr(), best.unsafe_ptr(), idx.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_HU64(unsafe_from_address=ic_addr), src_buf=ic)
        ctx.enqueue_copy(dst_ptr=_HU64(unsafe_from_address=best_addr), src_buf=best)
        ctx.enqueue_copy(dst_ptr=_HI64(unsafe_from_address=idx_addr), src_buf=idx)
        ctx.synchronize()
        _ = ic^
        _ = best^
    else:
        var ic = ctx.enqueue_create_buffer[DType.float32](n)
        var best = ctx.enqueue_create_buffer[DType.float32](n)
        if k > 0:
            ctx.enqueue_copy(dst_buf=best, src_ptr=_HF32(unsafe_from_address=best_addr))
        ctx.enqueue_function[_ic_min_f32_kernel](
            llf.unsafe_ptr(), Int32(n), Float32(penalty), Int32(k),
            ic.unsafe_ptr(), best.unsafe_ptr(), idx.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_HF32(unsafe_from_address=ic_addr), src_buf=ic)
        ctx.enqueue_copy(dst_ptr=_HF32(unsafe_from_address=best_addr), src_buf=best)
        ctx.enqueue_copy(dst_ptr=_HI64(unsafe_from_address=idx_addr), src_buf=idx)
        ctx.synchronize()
        _ = ic^
        _ = best^
    _ = idx^
    _ = llf^
