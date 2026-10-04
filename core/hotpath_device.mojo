# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device twins of the shared host helpers of `bindings/hotpath_helpers.mojo`
(lane fam2-shared, 2026-10-04; cpu-gpu audit case 10).

Each function here takes the SAME host addresses its host helper takes,
uploads them, does the per-row work in kernels (one thread per row, or a
radix sort plus a scan where the host loop was order-dependent), and downloads
the result. Integers, comparisons and bit moves only (float64 through
`checks/soft_f64.mojo` words, because the Apple GPU has no float64), so the
bytes are the host helper's on NVIDIA, AMD and Apple; the host helper stays
the host column and the refusal path (a device twin that sees an input its
host helper would refuse reports it, and the binding reruns the host helper,
which raises its own words).

No shared memory is used (no page to gate); counts use `Int32` atomics (the
Apple GPU has no 64-bit atomic), so every row count here is below 2^31
(`HPD_MAX_N`, tested by the bindings).

The bindings that choose between these and the host helpers are
`bindings/hotpath_device.mojo`.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from checks.soft_f64 import sf64_from_f32, sf64_gt, sf64_is_nan, sf64_lt, sf64_to_f32
from core.device_zero import enqueue_fill
from core.fast_radix_sort import (
    fast_radix_sort_pairs_u32,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)
from core.label_encode_device import device_unique_inverse_resident

#: The dtype codes of `bindings/hotpath_helpers.mojo` (HP_F32 ...).
comptime HPD_F32 = 0
comptime HPD_F64 = 1
comptime HPD_I32 = 2
comptime HPD_I64 = 3
comptime HPD_U32 = 4
comptime HPD_U8 = 5

comptime HPD_TPB = 256
#: Blocks of a grid-stride reduction launch.
comptime HPD_REDUCE_BLOCKS = 1024
#: Row counts the Int32 kernels and atomics hold.
comptime HPD_MAX_N = 2147483000

comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _F32 = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + HPD_TPB - 1) // HPD_TPB if count > 0 else 1


def _read_status(
    ctx: DeviceContext, mut d_status: DeviceBuffer[DType.int32], slot: Int,
) raises -> Int:
    """Synchronizes and returns word `slot` of the 2-word status buffer."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr()[slot])
    _ = h^
    return v


# ===========================================================================
# equal_elements
# ===========================================================================


def _eq_u8_kernel(a: _U8, b: _U8, n_: Int32, dst: _U8):
    var i = _tid()
    if i < Int(n_):
        var one = UInt8(1) if a.unsafe_load(i) == b.unsafe_load(i) else UInt8(0)
        dst.unsafe_store(i, one)


def _eq_u32_kernel(a: _U32, b: _U32, n_: Int32, is_float: Int32, dst: _U8):
    """Bit equality of two 32-bit words; `is_float`: IEEE equality from the
    bits (a NaN equals nothing, +0 equals -0)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var x = a.unsafe_load(i)
    var y = b.unsafe_load(i)
    var eq = x == y
    if is_float != 0:
        var ax = x & UInt32(0x7FFFFFFF)
        var ay = y & UInt32(0x7FFFFFFF)
        if ax > UInt32(0x7F800000) or ay > UInt32(0x7F800000):
            eq = False
        elif (ax | ay) == UInt32(0):
            eq = True
    dst.unsafe_store(i, UInt8(1) if eq else UInt8(0))


def _eq_u64_kernel(a: _U64, b: _U64, n_: Int32, is_float: Int32, dst: _U8):
    """`_eq_u32_kernel` for 64-bit words."""
    var i = _tid()
    if i >= Int(n_):
        return
    var x = a.unsafe_load(i)
    var y = b.unsafe_load(i)
    var eq = x == y
    if is_float != 0:
        var ax = x & UInt64(0x7FFFFFFFFFFFFFFF)
        var ay = y & UInt64(0x7FFFFFFFFFFFFFFF)
        if ax > UInt64(0x7FF0000000000000) or ay > UInt64(0x7FF0000000000000):
            eq = False
        elif (ax | ay) == UInt64(0):
            eq = True
    dst.unsafe_store(i, UInt8(1) if eq else UInt8(0))


def device_equal_elements(
    ctx: DeviceContext, a_addr: Int, b_addr: Int, code: Int, n: Int, dst_addr: Int,
) raises:
    """`equal_elements`: dst[i] (uint8) = 1 where a[i] == b[i], one dtype
    `code`, `1 <= n <= HPD_MAX_N`."""
    var d_dst = ctx.enqueue_create_buffer[DType.uint8](n)
    if code == HPD_U8:
        var d_a = ctx.enqueue_create_buffer[DType.uint8](n)
        var d_b = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=d_a, src_ptr=_U8(unsafe_from_address=a_addr))
        ctx.enqueue_copy(dst_buf=d_b, src_ptr=_U8(unsafe_from_address=b_addr))
        ctx.enqueue_function[_eq_u8_kernel](
            d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int32(n), d_dst.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_a^
        _ = d_b^
    elif code == HPD_F32 or code == HPD_I32 or code == HPD_U32:
        var d_a = ctx.enqueue_create_buffer[DType.uint32](n)
        var d_b = ctx.enqueue_create_buffer[DType.uint32](n)
        ctx.enqueue_copy(dst_buf=d_a, src_ptr=_U32(unsafe_from_address=a_addr))
        ctx.enqueue_copy(dst_buf=d_b, src_ptr=_U32(unsafe_from_address=b_addr))
        ctx.enqueue_function[_eq_u32_kernel](
            d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int32(n),
            Int32(1) if code == HPD_F32 else Int32(0), d_dst.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_a^
        _ = d_b^
    else:
        var d_a = ctx.enqueue_create_buffer[DType.uint64](n)
        var d_b = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(dst_buf=d_a, src_ptr=_U64(unsafe_from_address=a_addr))
        ctx.enqueue_copy(dst_buf=d_b, src_ptr=_U64(unsafe_from_address=b_addr))
        ctx.enqueue_function[_eq_u64_kernel](
            d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int32(n),
            Int32(1) if code == HPD_F64 else Int32(0), d_dst.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_a^
        _ = d_b^
    _ = d_dst^


# ===========================================================================
# gather_i32
# ===========================================================================


def _gather_i32_kernel(table: _I32, nt: Int32, codes: _I32, n_: Int32, dst: _I32, status: _I32):
    """dst[i] = table[codes[i]]; a code outside [0, nt) sets status[0]
    (every writer stores 1) and writes nothing."""
    var i = _tid()
    if i >= Int(n_):
        return
    var c = Int(codes.unsafe_load(i))
    if c < 0 or c >= Int(nt):
        status.unsafe_store(0, Int32(1))
        return
    dst.unsafe_store(i, table.unsafe_load(c))


def device_gather_i32(
    ctx: DeviceContext, table_addr: Int, nt: Int, codes_addr: Int, n: Int, dst_addr: Int,
) raises -> Bool:
    """`gather_i32`. False (and no byte of `dst` written) when a code is out
    of range."""
    var d_t = ctx.enqueue_create_buffer[DType.int32](nt)
    var d_c = ctx.enqueue_create_buffer[DType.int32](n)
    var d_d = ctx.enqueue_create_buffer[DType.int32](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=_I32(unsafe_from_address=table_addr))
    ctx.enqueue_copy(dst_buf=d_c, src_ptr=_I32(unsafe_from_address=codes_addr))
    ctx.enqueue_function[_gather_i32_kernel](
        d_t.unsafe_ptr(), Int32(nt), d_c.unsafe_ptr(), Int32(n), d_d.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    if ok:
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=dst_addr), src_buf=d_d)
        ctx.synchronize()
    _ = d_t^
    _ = d_c^
    _ = d_d^
    _ = d_status^
    return ok


def _gather_u64_kernel(table: _U64, nt: Int32, codes: _I64, n_: Int32, dst: _U64, status: _I32):
    """`_gather_i32_kernel` over 64-bit table words and int64 codes."""
    var i = _tid()
    if i >= Int(n_):
        return
    var c = Int(codes.unsafe_load(i))
    if c < 0 or c >= Int(nt):
        status.unsafe_store(0, Int32(1))
        return
    dst.unsafe_store(i, table.unsafe_load(c))


def device_gather_u64(
    ctx: DeviceContext, table_addr: Int, nt: Int, codes_addr: Int, n: Int, dst_addr: Int,
) raises -> Bool:
    """`gather_i64` / `gather_f64` (a move of 64-bit words, so one kernel
    serves both): dst[i] = table[codes[i]], int64 codes. False (and no byte
    of `dst` written) when a code is out of range."""
    var d_t = ctx.enqueue_create_buffer[DType.uint64](nt)
    var d_c = ctx.enqueue_create_buffer[DType.int64](n)
    var d_d = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=_U64(unsafe_from_address=table_addr))
    ctx.enqueue_copy(dst_buf=d_c, src_ptr=_I64(unsafe_from_address=codes_addr))
    ctx.enqueue_function[_gather_u64_kernel](
        d_t.unsafe_ptr(), Int32(nt), d_c.unsafe_ptr(), Int32(n), d_d.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    if ok:
        ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=dst_addr), src_buf=d_d)
        ctx.synchronize()
    _ = d_t^
    _ = d_c^
    _ = d_d^
    _ = d_status^
    return ok


# ===========================================================================
# threshold_labels_i64
# ===========================================================================


@always_inline
def _thr_up(v: UInt64, thr: UInt64, strict: Int32) -> Bool:
    """`v > thr` (strict) or `v >= thr` over binary64 words; a NaN on either
    side compares false."""
    if sf64_is_nan(v) or sf64_is_nan(thr):
        return False
    if strict != 0:
        return sf64_gt(v, thr)
    return not sf64_lt(v, thr)


def _thr_f32_kernel(src: _F32, n_: Int32, thr: UInt64, strict: Int32, lo: Int64, hi: Int64, dst: _I64):
    var i = _tid()
    if i < Int(n_):
        var v = sf64_from_f32(src.unsafe_load(i))
        dst.unsafe_store(i, hi if _thr_up(v, thr, strict) else lo)


def _thr_f64_kernel(src: _U64, n_: Int32, thr: UInt64, strict: Int32, lo: Int64, hi: Int64, dst: _I64):
    var i = _tid()
    if i < Int(n_):
        dst.unsafe_store(i, hi if _thr_up(src.unsafe_load(i), thr, strict) else lo)


def device_threshold_labels_i64(
    ctx: DeviceContext, src_addr: Int, code: Int, n: Int, thr: Float64, strict: Bool,
    below: Int64, above: Int64, dst_addr: Int,
) raises:
    """`threshold_labels_i64`: float32 (`HPD_F32`) or float64 scores against
    the float64 threshold, compared as binary64 words."""
    var thr_bits = bitcast[DType.uint64](thr)
    var st = Int32(1) if strict else Int32(0)
    var d_dst = ctx.enqueue_create_buffer[DType.int64](n)
    if code == HPD_F32:
        var d_s = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_F32(unsafe_from_address=src_addr))
        ctx.enqueue_function[_thr_f32_kernel](
            d_s.unsafe_ptr(), Int32(n), thr_bits, st, below, above, d_dst.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_s^
    else:
        var d_s = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_U64(unsafe_from_address=src_addr))
        ctx.enqueue_function[_thr_f64_kernel](
            d_s.unsafe_ptr(), Int32(n), thr_bits, st, below, above, d_dst.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
        _ = d_s^
    _ = d_dst^


# ===========================================================================
# bincount_i64, count_mask_u8
# ===========================================================================


def _bincount_i32_kernel(src: _I32, n_: Int32, k: Int32, cnt: _I32, status: _I32):
    """cnt[src[i]] += 1 (Int32 atomics: exact in any order); a value outside
    [0, k) sets status[0]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var v = Int(src.unsafe_load(i))
    if v < 0 or v >= Int(k):
        status.unsafe_store(0, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + v, Int32(1))


def _bincount_i64_kernel(src: _I64, n_: Int32, k: Int32, cnt: _I32, status: _I32):
    var i = _tid()
    if i >= Int(n_):
        return
    var v = Int(src.unsafe_load(i))
    if v < 0 or v >= Int(k):
        status.unsafe_store(0, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + v, Int32(1))


def _widen_counts_kernel(cnt: _I32, k: Int32, base: _I64, accumulate: Int32):
    """base[j] = (base[j] if accumulate else 0) + cnt[j]."""
    var j = _tid()
    if j < Int(k):
        var b = Int64(0)
        if accumulate != 0:
            b = base.unsafe_load(j)
        base.unsafe_store(j, b + Int64(cnt.unsafe_load(j)))


def device_bincount_i64(
    ctx: DeviceContext, src_addr: Int, code: Int, n: Int, k: Int, counts_addr: Int, accumulate: Bool,
) raises -> Bool:
    """`bincount_i64` (`n >= 1`): False when a value is outside [0, k); the
    host counts are then zeroed unless `accumulate`, as the host helper
    leaves them."""
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_base = ctx.enqueue_create_buffer[DType.int64](k)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_status, Int32(0))
    if accumulate:
        ctx.enqueue_copy(dst_buf=d_base, src_ptr=_I64(unsafe_from_address=counts_addr))
    else:
        enqueue_fill(ctx, d_base, Int64(0))
    var ok: Bool
    if code == HPD_I32:
        var d_s = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_I32(unsafe_from_address=src_addr))
        ctx.enqueue_function[_bincount_i32_kernel](
            d_s.unsafe_ptr(), Int32(n), Int32(k), d_cnt.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ok = _read_status(ctx, d_status, 0) == 0
        _ = d_s^
    else:
        var d_s = ctx.enqueue_create_buffer[DType.int64](n)
        ctx.enqueue_copy(dst_buf=d_s, src_ptr=_I64(unsafe_from_address=src_addr))
        ctx.enqueue_function[_bincount_i64_kernel](
            d_s.unsafe_ptr(), Int32(n), Int32(k), d_cnt.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ok = _read_status(ctx, d_status, 0) == 0
        _ = d_s^
    if ok:
        ctx.enqueue_function[_widen_counts_kernel](
            d_cnt.unsafe_ptr(), Int32(k), d_base.unsafe_ptr(), Int32(1),
            grid_dim=_blocks(k), block_dim=HPD_TPB,
        )
    if ok or not accumulate:
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=counts_addr), src_buf=d_base)
    ctx.synchronize()
    _ = d_cnt^
    _ = d_base^
    _ = d_status^
    return ok


def _count_u8_kernel(mask: _U8, n_: Int32, total: _I32):
    """total[0] += the nonzero bytes this thread's grid-stride walk meets."""
    var i = _tid()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var n = Int(n_)
    var c = Int32(0)
    while i < n:
        if mask.unsafe_load(i) != UInt8(0):
            c += 1
        i += stride
    if c != Int32(0):
        _ = Atomic[DType.int32].fetch_add(total, c)


def _count_grid(n: Int) -> Int:
    return max(1, min(HPD_REDUCE_BLOCKS, (n + HPD_TPB - 1) // HPD_TPB))


def device_count_mask_u8(ctx: DeviceContext, mask_addr: Int, n: Int) raises -> Int:
    """`count_mask_u8` (`n >= 1`)."""
    var d_m = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_m, src_ptr=_U8(unsafe_from_address=mask_addr))
    ctx.enqueue_function[_count_u8_kernel](
        d_m.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(),
        grid_dim=_count_grid(n), block_dim=HPD_TPB,
    )
    var c = _read_status(ctx, d_status, 0)
    _ = d_m^
    _ = d_status^
    return c


# ===========================================================================
# fold_pair_f32
# ===========================================================================


def _fold_pair_kernel(dst: _F32, src: _F32, n_: Int32):
    """dst[i] = ftz(ftz(dst[i]) + ftz(src[i])): one float32 rounding."""
    var i = _tid()
    if i < Int(n_):
        var a = ftz(dst.unsafe_load(i))
        var b = ftz(src.unsafe_load(i))
        var s = ftz(a + b)
        dst.unsafe_store(i, s)


def device_fold_pair_f32(ctx: DeviceContext, dst_addr: Int, src_addr: Int, n: Int) raises:
    """`fold_pair_f32` (`n >= 1`)."""
    var d_d = ctx.enqueue_create_buffer[DType.float32](n)
    var d_s = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_d, src_ptr=_F32(unsafe_from_address=dst_addr))
    ctx.enqueue_copy(dst_buf=d_s, src_ptr=_F32(unsafe_from_address=src_addr))
    ctx.enqueue_function[_fold_pair_kernel](
        d_d.unsafe_ptr(), d_s.unsafe_ptr(), Int32(n), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_F32(unsafe_from_address=dst_addr), src_buf=d_d)
    ctx.synchronize()
    _ = d_d^
    _ = d_s^


# ===========================================================================
# encode_labels_<dtype>: the ORDER RULE's encoder through the device
# unique-inverse (`core/label_encode_device.mojo`)
# ===========================================================================


def _widen_label_kernel(s8: _U8, s32: _U32, code: Int32, n_: Int32, dst: _U64):
    """The 64-bit label word `device_unique_inverse_resident` sorts: float32
    widened exactly to binary64, int32 sign-extended, uint32 and uint8
    zero-extended (all three then int64 labels)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var w = UInt64(0)
    if code == HPD_U8:
        w = UInt64(s8.unsafe_load(i))
    else:
        var u = s32.unsafe_load(i)
        if code == HPD_F32:
            w = sf64_from_f32(bitcast[DType.float32](u))
        elif code == HPD_I32:
            w = bitcast[DType.uint64](Int64(bitcast[DType.int32](u)))
        else:
            w = UInt64(u)
    dst.unsafe_store(i, w)


def _narrow_class_kernel(cls: _U64, k: Int32, code: Int32, d8: _U8, d32: _U32):
    """Class word j back in the label dtype (exact: it was widened from it)."""
    var j = _tid()
    if j >= Int(k):
        return
    var w = cls.unsafe_load(j)
    if code == HPD_U8:
        d8.unsafe_store(j, UInt8(w & UInt64(0xFF)))
    elif code == HPD_F32:
        d32.unsafe_store(j, bitcast[DType.uint32](sf64_to_f32(w)))
    else:
        d32.unsafe_store(j, UInt32(w & UInt64(0xFFFFFFFF)))


def _head_u64_kernel(src: _U64, k: Int32, dst: _U64):
    var j = _tid()
    if j < Int(k):
        dst.unsafe_store(j, src.unsafe_load(j))


def device_encode_labels(
    ctx: DeviceContext, src_addr: Int, code: Int, n: Int, classes_addr: Int, max_classes: Int,
    codes_addr: Int,
) raises -> Int:
    """`encode_labels_<dtype>` (`1 <= n <= HPD_MAX_N`): the sorted distinct
    labels into `classes` (the label dtype, the first k of `max_classes`
    slots) and one int32 code per row. Returns k; -1 when there are more
    than `max_classes` classes or a label is NaN (nothing is written: the
    binding reruns the host helper, whose refusal is the definition)."""
    var is_float = code == HPD_F32 or code == HPD_F64
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_classes = ctx.enqueue_create_buffer[DType.uint64](n)
    if code == HPD_F64 or code == HPD_I64:
        ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U64(unsafe_from_address=src_addr))
    else:
        var narrow8 = code == HPD_U8
        var d_s8 = ctx.enqueue_create_buffer[DType.uint8](n if narrow8 else 1)
        var d_s32 = ctx.enqueue_create_buffer[DType.uint32](1 if narrow8 else n)
        if narrow8:
            ctx.enqueue_copy(dst_buf=d_s8, src_ptr=_U8(unsafe_from_address=src_addr))
        else:
            ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_U32(unsafe_from_address=src_addr))
        ctx.enqueue_function[_widen_label_kernel](
            d_s8.unsafe_ptr(), d_s32.unsafe_ptr(), Int32(code), Int32(n), d_src.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=HPD_TPB,
        )
        ctx.synchronize()
        _ = d_s8^
        _ = d_s32^
    var k = device_unique_inverse_resident(
        ctx, d_src, n, 0 if is_float else 1, d_codes, d_classes
    )
    if k < 1 or k > max_classes:
        _ = d_src^
        _ = d_codes^
        _ = d_classes^
        return -1
    if code == HPD_F64 or code == HPD_I64:
        var d_out = ctx.enqueue_create_buffer[DType.uint64](k)
        ctx.enqueue_function[_head_u64_kernel](
            d_classes.unsafe_ptr(), Int32(k), d_out.unsafe_ptr(),
            grid_dim=_blocks(k), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=classes_addr), src_buf=d_out)
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=codes_addr), src_buf=d_codes)
        ctx.synchronize()
        _ = d_out^
    else:
        var out8 = code == HPD_U8
        var d_o8 = ctx.enqueue_create_buffer[DType.uint8](k if out8 else 1)
        var d_o32 = ctx.enqueue_create_buffer[DType.uint32](1 if out8 else k)
        ctx.enqueue_function[_narrow_class_kernel](
            d_classes.unsafe_ptr(), Int32(k), Int32(code), d_o8.unsafe_ptr(), d_o32.unsafe_ptr(),
            grid_dim=_blocks(k), block_dim=HPD_TPB,
        )
        if out8:
            ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=classes_addr), src_buf=d_o8)
        else:
            ctx.enqueue_copy(dst_ptr=_U32(unsafe_from_address=classes_addr), src_buf=d_o32)
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=codes_addr), src_buf=d_codes)
        ctx.synchronize()
        _ = d_o8^
        _ = d_o32^
    _ = d_src^
    _ = d_codes^
    _ = d_classes^
    return k


# ===========================================================================
# Fold bookkeeping: partitions, masks, ranges, index checks
# ===========================================================================


def _fold_flag_kernel(fold: _I32, n_: Int32, want: Int32, flag: _I32, scan: _I32):
    var i = _tid()
    if i < Int(n_):
        var f = Int32(1) if fold.unsafe_load(i) == want else Int32(0)
        flag.unsafe_store(i, f)
        scan.unsafe_store(i, f)


def _mask_flag_kernel(mask: _U8, n_: Int32, flag: _I32, scan: _I32):
    var i = _tid()
    if i < Int(n_):
        var f = Int32(1) if mask.unsafe_load(i) != UInt8(0) else Int32(0)
        flag.unsafe_store(i, f)
        scan.unsafe_store(i, f)


def _scan_total_kernel(flag: _I32, scan: _I32, n_: Int32, status: _I32):
    """status[0] = the number of flagged rows (the exclusive scan's last
    slot plus the last flag). One thread."""
    var last = Int(n_) - 1
    status.unsafe_store(0, scan.unsafe_load(last) + flag.unsafe_load(last))


def _partition_emit_kernel(flag: _I32, scan: _I32, n_: Int32, test: _I64, train: _I64):
    """Row i goes to test[scan[i]] when flagged, else to train[i - scan[i]]:
    both ascending, the host loop's order."""
    var i = _tid()
    if i >= Int(n_):
        return
    var s = Int(scan.unsafe_load(i))
    if flag.unsafe_load(i) != Int32(0):
        test.unsafe_store(s, Int64(i))
    else:
        train.unsafe_store(i - s, Int64(i))


def _device_partition(
    ctx: DeviceContext,
    mut d_flag: DeviceBuffer[DType.int32],
    mut d_scan: DeviceBuffer[DType.int32],
    n: Int,
    test_addr: Int,
    train_addr: Int,
) raises -> Int:
    """Scans the flags, emits the flagged rows (int64, ascending) to the host
    `test` and the others to the host `train`; returns the flagged count. An
    empty side is not written (its address may be null)."""
    var d_bsum = ctx.enqueue_create_buffer[DType.int32](frs_scan_blocks(n))
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    frs_exclusive_scan(ctx, d_scan, n, d_bsum)
    ctx.enqueue_function[_scan_total_kernel](  # small-launch(n: the index of the last slot): one thread reads two words, no walk
        d_flag.unsafe_ptr(), d_scan.unsafe_ptr(), Int32(n), d_status.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var n_test = _read_status(ctx, d_status, 0)
    var n_train = n - n_test
    var d_test = ctx.enqueue_create_buffer[DType.int64](max(n_test, 1))
    var d_train = ctx.enqueue_create_buffer[DType.int64](max(n_train, 1))
    ctx.enqueue_function[_partition_emit_kernel](
        d_flag.unsafe_ptr(), d_scan.unsafe_ptr(), Int32(n), d_test.unsafe_ptr(), d_train.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    if n_test > 0 and test_addr != 0:
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=test_addr), src_buf=d_test)
    if n_train > 0 and train_addr != 0:
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=train_addr), src_buf=d_train)
    ctx.synchronize()
    _ = d_bsum^
    _ = d_status^
    _ = d_test^
    _ = d_train^
    return n_test


def device_select_fold_i64(
    ctx: DeviceContext, fold_addr: Int, n: Int, fold: Int, test_addr: Int, train_addr: Int,
) raises -> Int:
    """`select_fold_i64` (`1 <= n <= HPD_MAX_N`)."""
    var d_fold = ctx.enqueue_create_buffer[DType.int32](n)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](n)
    var d_scan = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_fold, src_ptr=_I32(unsafe_from_address=fold_addr))
    ctx.enqueue_function[_fold_flag_kernel](
        d_fold.unsafe_ptr(), Int32(n), Int32(fold), d_flag.unsafe_ptr(), d_scan.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var n_test = _device_partition(ctx, d_flag, d_scan, n, test_addr, train_addr)
    _ = d_fold^
    _ = d_flag^
    _ = d_scan^
    return n_test


def device_select_mask_u8_i64(
    ctx: DeviceContext, mask_addr: Int, n: Int, test_addr: Int, train_addr: Int,
) raises -> Int:
    """`select_mask_u8_i64` (`1 <= n <= HPD_MAX_N`)."""
    var d_mask = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](n)
    var d_scan = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_mask, src_ptr=_U8(unsafe_from_address=mask_addr))
    ctx.enqueue_function[_mask_flag_kernel](
        d_mask.unsafe_ptr(), Int32(n), d_flag.unsafe_ptr(), d_scan.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var n_test = _device_partition(ctx, d_flag, d_scan, n, test_addr, train_addr)
    _ = d_mask^
    _ = d_flag^
    _ = d_scan^
    return n_test


def _mask_mark_kernel(idx: _I64, k: Int32, n_: Int32, mask: _U8, status: _I32):
    """mask[idx[i]] = 1 (every writer stores 1); an index outside [0, n)
    sets status[0]."""
    var i = _tid()
    if i >= Int(k):
        return
    var r = Int(idx.unsafe_load(i))
    if r < 0 or r >= Int(n_):
        status.unsafe_store(0, Int32(1))
        return
    mask.unsafe_store(r, UInt8(1))


def device_mask_from_indices_u8(
    ctx: DeviceContext, idx_addr: Int, k: Int, n: Int, mask_addr: Int,
) raises -> Int:
    """`mask_from_indices_u8` (`n >= 1`, `k >= 0`): the distinct rows set,
    or -1 (mask not written) when an index is out of range."""
    var d_mask = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_mask, UInt8(0))
    enqueue_fill(ctx, d_status, Int32(0))
    var d_idx = ctx.enqueue_create_buffer[DType.int64](max(k, 1))
    if k > 0:
        ctx.enqueue_copy(dst_buf=d_idx, src_ptr=_I64(unsafe_from_address=idx_addr))
        ctx.enqueue_function[_mask_mark_kernel](
            d_idx.unsafe_ptr(), Int32(k), Int32(n), d_mask.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_blocks(k), block_dim=HPD_TPB,
        )
        ctx.enqueue_function[_count_u8_kernel](
            d_mask.unsafe_ptr(), Int32(n), d_status.unsafe_ptr() + 1,
            grid_dim=_count_grid(n), block_dim=HPD_TPB,
        )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var bad = h.unsafe_ptr()[0] != Int32(0)
    var set = Int(h.unsafe_ptr()[1])
    _ = h^
    if not bad:
        ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=mask_addr), src_buf=d_mask)
        ctx.synchronize()
    _ = d_idx^
    _ = d_mask^
    _ = d_status^
    return -1 if bad else set


def _arange_skip_kernel(dst: _I64, n_: Int32, start: Int64, skip_lo: Int64, skip_len: Int64):
    """dst[i] = start + i, stepping over [skip_lo, skip_lo + skip_len)."""
    var i = _tid()
    if i < Int(n_):
        var v = start + Int64(i)
        if v >= skip_lo:
            v += skip_len
        dst.unsafe_store(i, v)


def device_arange_skip_i64(
    ctx: DeviceContext, dst_addr: Int, start: Int, count: Int, skip_lo: Int, skip_len: Int,
) raises:
    """`count >= 1` ascending int64 values from `start`, omitting `skip_len`
    values at `skip_lo` (`skip_len == 0`: a plain range)."""
    var d = ctx.enqueue_create_buffer[DType.int64](count)
    ctx.enqueue_function[_arange_skip_kernel](
        d.unsafe_ptr(), Int32(count), Int64(start), Int64(skip_lo), Int64(skip_len),
        grid_dim=_blocks(count), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=dst_addr), src_buf=d)
    ctx.synchronize()
    _ = d^


def _check_indices_kernel(idx: _I64, n_: Int32, bound: Int32, seen: _I32, status: _I32):
    """An index outside [0, bound) sets status[0]; an index some other
    thread already claimed sets status[1] (Int32 atomic claim counts)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var v = Int(idx.unsafe_load(i))
    if v < 0 or v >= Int(bound):
        status.unsafe_store(0, Int32(1))
        return
    var prev = Atomic[DType.int32].fetch_add(seen + v, Int32(1))
    if prev != Int32(0):
        status.unsafe_store(1, Int32(1))


def device_check_indices_i64(ctx: DeviceContext, addr: Int, n: Int, bound: Int) raises -> Int:
    """`check_indices_i64` (`n >= 1`, `0 <= bound <= HPD_MAX_N`): 0, 1 (out
    of range) or 2 (duplicate)."""
    var d_idx = ctx.enqueue_create_buffer[DType.int64](n)
    var d_seen = ctx.enqueue_create_buffer[DType.int32](max(bound, 1))
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_seen, Int32(0))
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=_I64(unsafe_from_address=addr))
    ctx.enqueue_function[_check_indices_kernel](
        d_idx.unsafe_ptr(), Int32(n), Int32(bound), d_seen.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var out = 0
    if h.unsafe_ptr()[0] != Int32(0):
        out = 1
    elif h.unsafe_ptr()[1] != Int32(0):
        out = 2
    _ = h^
    _ = d_idx^
    _ = d_seen^
    _ = d_status^
    return out


def _overlap_test_kernel(idx: _I64, n_: Int32, bound: Int32, mask: _U8, status: _I32):
    """status[1] = 1 when an index is marked in `mask`; out of range sets
    status[0]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var v = Int(idx.unsafe_load(i))
    if v < 0 or v >= Int(bound):
        status.unsafe_store(0, Int32(1))
        return
    if mask.unsafe_load(v) != UInt8(0):
        status.unsafe_store(1, Int32(1))


def device_indices_overlap_i64(
    ctx: DeviceContext, a_addr: Int, na: Int, b_addr: Int, nb: Int, bound: Int,
) raises -> Int:
    """`indices_overlap_i64`: 1 when the two index sets share a member, 0
    when not, -1 when an index is out of range (the binding reruns the host
    helper)."""
    var d_a = ctx.enqueue_create_buffer[DType.int64](na)
    var d_b = ctx.enqueue_create_buffer[DType.int64](nb)
    var d_mask = ctx.enqueue_create_buffer[DType.uint8](bound)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_mask, UInt8(0))
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_a, src_ptr=_I64(unsafe_from_address=a_addr))
    ctx.enqueue_copy(dst_buf=d_b, src_ptr=_I64(unsafe_from_address=b_addr))
    ctx.enqueue_function[_mask_mark_kernel](
        d_a.unsafe_ptr(), Int32(na), Int32(bound), d_mask.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(na), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_overlap_test_kernel](
        d_b.unsafe_ptr(), Int32(nb), Int32(bound), d_mask.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(nb), block_dim=HPD_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var out = 0
    if h.unsafe_ptr()[0] != Int32(0):
        out = -1
    elif h.unsafe_ptr()[1] != Int32(0):
        out = 1
    _ = h^
    _ = d_a^
    _ = d_b^
    _ = d_mask^
    _ = d_status^
    return out


# ===========================================================================
# fold_ids: KFold (closed form per row) and the stratified dealing
# ===========================================================================


def _kfold_kernel(fold: _I32, n_: Int32, splits: Int32):
    """Contiguous folds, the first `n % splits` one row longer."""
    var i = _tid()
    var n = Int(n_)
    if i >= n:
        return
    var s = Int(splits)
    var q = n // s
    var rem = n - q * s
    var cut = rem * (q + 1)
    var qq = q if q > 0 else 1  # q == 0 only when every row is below `cut`
    var f = 0
    if i < cut:
        f = i // (q + 1)
    else:
        f = rem + (i - cut) // qq
    fold.unsafe_store(i, Int32(f))


def device_kfold_ids(ctx: DeviceContext, n: Int, splits: Int, fold_addr: Int) raises:
    """`fold_ids` with no class codes: one int32 fold id per row."""
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[_kfold_kernel](
        d.unsafe_ptr(), Int32(n), Int32(splits), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=fold_addr), src_buf=d)
    ctx.synchronize()
    _ = d^


def _class_first_kernel(codes: _I32, n_: Int32, k: Int32, cnt: _I32, first: _I32, status: _I32):
    """cnt[c] += 1 and first[c] = min(first[c], row) for each row's class c;
    a code outside [0, k) sets status[0]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var c = Int(codes.unsafe_load(i))
    if c < 0 or c >= Int(k):
        status.unsafe_store(0, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + c, Int32(1))
    _ = Atomic[DType.int32].min(first + c, Int32(i))


def _pair_iota_kernel(src: _I32, n_: Int32, keys: _U32, vals: _U32):
    """keys[i] = src[i] (non-negative, so its bits order as the value),
    vals[i] = i: the pairs a stable radix sort then orders by key."""
    var i = _tid()
    if i < Int(n_):
        keys.unsafe_store(i, bitcast[DType.uint32](src.unsafe_load(i)))
        vals.unsafe_store(i, UInt32(i))


def _rank_fold_kernel(
    keys: _U32, vals: _U32, n_: Int32, start: _I32, cum: _I32, splits: Int32,
    perm: _I64, has_perm: Int32, fold: _I32, per_fold: _I32,
):
    """Sorted position p holds class c = keys[p] and row r = vals[p]; the
    row's rank in its class (row order: the sort is stable) is p - start[c],
    or perm[start[c] + rank] when `has_perm`; its fold is the first f with
    cum[c * splits + f] > rank (the last fold when none). per_fold[f] += 1."""
    var p = _tid()
    if p >= Int(n_):
        return
    var c = Int(keys.unsafe_load(p))
    var r = Int(vals.unsafe_load(p))
    var s0 = Int(start.unsafe_load(c))
    var rank = p - s0
    if has_perm != 0:
        rank = Int(perm.unsafe_load(s0 + rank))
    var K = Int(splits)
    var f = 0
    while f < K - 1 and Int(cum.unsafe_load(c * K + f)) <= rank:
        f += 1
    fold.unsafe_store(r, Int32(f))
    _ = Atomic[DType.int32].fetch_add(per_fold + f, Int32(1))


def device_class_counts(
    ctx: DeviceContext,
    mut d_codes: DeviceBuffer[DType.int32],
    n: Int,
    k: Int,
    mut d_cnt: DeviceBuffer[DType.int32],
    mut d_order: DeviceBuffer[DType.uint32],
) raises -> Bool:
    """Per class code: its row count into `d_cnt` (k) and the class codes in
    first-seen order into `d_order` (k; classes with no row last). False
    when a code is outside [0, k). Synchronizes."""
    var d_first = ctx.enqueue_create_buffer[DType.int32](k)
    var d_keys = ctx.enqueue_create_buffer[DType.uint32](k)
    var d_tk = ctx.enqueue_create_buffer[DType.uint32](k)
    var d_tv = ctx.enqueue_create_buffer[DType.uint32](k)
    var d_sc = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(k))
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_first, Int32(0x7FFFFFFF))
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_function[_class_first_kernel](
        d_codes.unsafe_ptr(), Int32(n), Int32(k), d_cnt.unsafe_ptr(), d_first.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_pair_iota_kernel](
        d_first.unsafe_ptr(), Int32(k), d_keys.unsafe_ptr(), d_order.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, k, d_keys, d_order, d_tk, d_tv, d_sc)
    var ok = _read_status(ctx, d_status, 0) == 0
    _ = d_first^
    _ = d_keys^
    _ = d_tk^
    _ = d_tv^
    _ = d_sc^
    _ = d_status^
    return ok


def device_rank_folds(
    ctx: DeviceContext,
    mut d_codes: DeviceBuffer[DType.int32],
    n: Int,
    k: Int,
    splits: Int,
    mut d_start: DeviceBuffer[DType.int32],
    mut d_cum: DeviceBuffer[DType.int32],
    perm_addr: Int,
    fold_addr: Int,
    fold_counts_addr: Int,
) raises:
    """Each row's fold from its rank within its class (`_rank_fold_kernel`):
    int32 fold ids to the host `fold_addr`, and, when `fold_counts_addr` is
    not null, the int64 row count of each fold. `d_start` (k) is the class
    offsets in the class-sorted rows, `d_cum` (k * splits) the cumulative
    per-class fold quotas; `perm_addr` (int64, n, or null) permutes ranks."""
    var d_keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_vals = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tk = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tv = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_sc = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(n))
    var d_fold = ctx.enqueue_create_buffer[DType.int32](n)
    var d_pf = ctx.enqueue_create_buffer[DType.int32](splits)
    var d_pf64 = ctx.enqueue_create_buffer[DType.int64](splits)
    var has_perm = perm_addr != 0
    var d_perm = ctx.enqueue_create_buffer[DType.int64](n if has_perm else 1)
    if has_perm:
        ctx.enqueue_copy(dst_buf=d_perm, src_ptr=_I64(unsafe_from_address=perm_addr))
    enqueue_fill(ctx, d_pf, Int32(0))
    ctx.enqueue_function[_pair_iota_kernel](
        d_codes.unsafe_ptr(), Int32(n), d_keys.unsafe_ptr(), d_vals.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, n, d_keys, d_vals, d_tk, d_tv, d_sc)
    ctx.enqueue_function[_rank_fold_kernel](
        d_keys.unsafe_ptr(), d_vals.unsafe_ptr(), Int32(n), d_start.unsafe_ptr(), d_cum.unsafe_ptr(),
        Int32(splits), d_perm.unsafe_ptr(), Int32(1) if has_perm else Int32(0),
        d_fold.unsafe_ptr(), d_pf.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=fold_addr), src_buf=d_fold)
    if fold_counts_addr != 0:
        ctx.enqueue_function[_widen_counts_kernel](
            d_pf.unsafe_ptr(), Int32(splits), d_pf64.unsafe_ptr(), Int32(0),
            grid_dim=_blocks(splits), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=fold_counts_addr), src_buf=d_pf64)
    ctx.synchronize()
    _ = d_keys^
    _ = d_vals^
    _ = d_tk^
    _ = d_tv^
    _ = d_sc^
    _ = d_fold^
    _ = d_pf^
    _ = d_pf64^
    _ = d_perm^


def device_stratified_fold_ids(
    ctx: DeviceContext, codes_addr: Int, n: Int, k: Int, splits: Int,
    counts_addr: Int, fold_addr: Int, fold_counts_addr: Int,
) raises -> Bool:
    """`fold_ids` with class codes (`1 <= n <= HPD_MAX_N`, `k >= 1`,
    `splits >= 2`). The rows are counted, ranked and dealt on the device;
    the host builds only the k x splits quota table from the k class counts
    (control data, the helper's own integer formula). False when a code is
    out of range (nothing written)."""
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_order = ctx.enqueue_create_buffer[DType.uint32](k)
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=_I32(unsafe_from_address=codes_addr))
    var ok = device_class_counts(ctx, d_codes, n, k, d_cnt, d_order)
    if not ok:
        _ = d_codes^
        _ = d_cnt^
        _ = d_order^
        return False
    var h_cnt = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_order = ctx.enqueue_create_host_buffer[DType.uint32](k)
    ctx.enqueue_copy(dst_ptr=h_cnt.unsafe_ptr(), src_buf=d_cnt)
    ctx.enqueue_copy(dst_ptr=h_order.unsafe_ptr(), src_buf=d_order)
    ctx.synchronize()
    var h_start = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_cum = ctx.enqueue_create_host_buffer[DType.int32](k * splits)
    var cp = h_cnt.unsafe_ptr()
    var op = h_order.unsafe_ptr()
    var sp = h_start.unsafe_ptr()
    var qp = h_cum.unsafe_ptr()
    var at = 0
    for c in range(k):
        sp.unsafe_store(c, Int32(at))
        at += Int(cp.unsafe_load(c))
    var offset = 0
    for j in range(k):
        var c = Int(op.unsafe_load(j))
        var count = Int(cp.unsafe_load(c))
        var run = 0
        for fold in range(splits):
            var first = (fold - offset) % splits
            if first < 0:
                first += splits
            var take = 0
            if first < count:
                take = 1 + (count - 1 - first) // splits
            run += take
            qp.unsafe_store(c * splits + fold, Int32(run))
        offset += count
    var d_start = ctx.enqueue_create_buffer[DType.int32](k)
    var d_cum = ctx.enqueue_create_buffer[DType.int32](k * splits)
    var d_cnt64 = ctx.enqueue_create_buffer[DType.int64](k)
    ctx.enqueue_copy(dst_buf=d_start, src_buf=h_start)
    ctx.enqueue_copy(dst_buf=d_cum, src_buf=h_cum)
    ctx.enqueue_function[_widen_counts_kernel](
        d_cnt.unsafe_ptr(), Int32(k), d_cnt64.unsafe_ptr(), Int32(0),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=counts_addr), src_buf=d_cnt64)
    device_rank_folds(ctx, d_codes, n, k, splits, d_start, d_cum, 0, fold_addr, fold_counts_addr)
    _ = h_cnt^
    _ = h_order^
    _ = h_start^
    _ = h_cum^
    _ = d_codes^
    _ = d_cnt^
    _ = d_order^
    _ = d_start^
    _ = d_cum^
    _ = d_cnt64^
    return True


# ===========================================================================
# first_seen_i32
# ===========================================================================


def _first_rank_kernel(order: _U32, k: Int32, cnt: _I32, rank_of: _I32, counts: _I64, status: _I32):
    """Position q of the first-seen order holds class c: rank_of[c] = q,
    counts[q] = cnt[c] (0 for a class with no row), and status[1] counts the
    classes present."""
    var q = _tid()
    if q >= Int(k):
        return
    var c = Int(order.unsafe_load(q))
    var m = cnt.unsafe_load(c)
    rank_of.unsafe_store(c, Int32(q))
    counts.unsafe_store(q, Int64(m))
    if m != Int32(0):
        _ = Atomic[DType.int32].fetch_add(status + 1, Int32(1))


def device_first_seen_i32(
    ctx: DeviceContext, codes_addr: Int, n: Int, k: Int, enc_addr: Int, counts_addr: Int,
) raises -> Int:
    """`first_seen_i32` (`1 <= n <= HPD_MAX_N`, `k >= 1`): the classes
    present, or -1 when a code is out of range (nothing written)."""
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_order = ctx.enqueue_create_buffer[DType.uint32](k)
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=_I32(unsafe_from_address=codes_addr))
    var ok = device_class_counts(ctx, d_codes, n, k, d_cnt, d_order)
    if not ok:
        _ = d_codes^
        _ = d_cnt^
        _ = d_order^
        return -1
    var d_rank = ctx.enqueue_create_buffer[DType.int32](k)
    var d_counts = ctx.enqueue_create_buffer[DType.int64](k)
    var d_enc = ctx.enqueue_create_buffer[DType.int32](n)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_function[_first_rank_kernel](
        d_order.unsafe_ptr(), Int32(k), d_cnt.unsafe_ptr(), d_rank.unsafe_ptr(), d_counts.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_gather_i32_kernel](
        d_rank.unsafe_ptr(), Int32(k), d_codes.unsafe_ptr(), Int32(n), d_enc.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=enc_addr), src_buf=d_enc)
    ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=counts_addr), src_buf=d_counts)
    var m = _read_status(ctx, d_status, 1)
    _ = d_codes^
    _ = d_cnt^
    _ = d_order^
    _ = d_rank^
    _ = d_counts^
    _ = d_enc^
    _ = d_status^
    return m


# ===========================================================================
# strat_fold_assign_i32
# ===========================================================================


def device_strat_fold_assign_i32(
    ctx: DeviceContext, enc_addr: Int, n: Int, k: Int, n_folds: Int, alloc_addr: Int,
    perms_addr: Int, counts_addr: Int, dst_addr: Int,
) raises -> Bool:
    """`strat_fold_assign_i32`. The k x n_folds tables (class offsets,
    cumulative allocations) are built on the host from the caller's k-sized
    `counts` and `alloc` (control data); every row is ranked and assigned on
    the device. False (nothing written) when the tables are inconsistent or
    a class code is out of range: the binding reruns the host helper."""
    var cp = _I64(unsafe_from_address=counts_addr)
    var ap = _I64(unsafe_from_address=alloc_addr)
    var h_start = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_cum = ctx.enqueue_create_host_buffer[DType.int32](k * n_folds)
    var sp = h_start.unsafe_ptr()
    var qp = h_cum.unsafe_ptr()
    var at = 0
    var consistent = True
    for c in range(k):
        var count = Int(cp.unsafe_load(c))
        if count < 0:
            consistent = False
            count = 0
        sp.unsafe_store(c, Int32(at))
        at += count
        var run = 0
        for f in range(n_folds):
            var a = Int(ap.unsafe_load(f * k + c))
            if a < 0:
                consistent = False
                a = 0
            run += a
            qp.unsafe_store(c * n_folds + f, Int32(run))
        if run != count:
            consistent = False
    if at != n:
        consistent = False
    if not consistent:
        _ = h_start^
        _ = h_cum^
        return False
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_cnt64 = ctx.enqueue_create_buffer[DType.int64](k)
    var d_ref64 = ctx.enqueue_create_buffer[DType.int64](k)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    var d_start = ctx.enqueue_create_buffer[DType.int32](k)
    var d_cum = ctx.enqueue_create_buffer[DType.int32](k * n_folds)
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=_I32(unsafe_from_address=enc_addr))
    ctx.enqueue_copy(dst_buf=d_ref64, src_ptr=cp)
    ctx.enqueue_copy(dst_buf=d_start, src_buf=h_start)
    ctx.enqueue_copy(dst_buf=d_cum, src_buf=h_cum)
    # the rows' own class counts must be the caller's `counts`: a row whose
    # rank passes its class's count would read another class's permutation
    ctx.enqueue_function[_bincount_i32_kernel](
        d_codes.unsafe_ptr(), Int32(n), Int32(k), d_cnt.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_widen_counts_kernel](
        d_cnt.unsafe_ptr(), Int32(k), d_cnt64.unsafe_ptr(), Int32(0),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_eq_u64_status_kernel](
        d_cnt64.unsafe_ptr(), d_ref64.unsafe_ptr(), Int32(k), d_status.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    if ok:
        device_rank_folds(ctx, d_codes, n, k, n_folds, d_start, d_cum, perms_addr, dst_addr, 0)
    _ = h_start^
    _ = h_cum^
    _ = d_codes^
    _ = d_cnt^
    _ = d_cnt64^
    _ = d_ref64^
    _ = d_status^
    _ = d_start^
    _ = d_cum^
    return ok


def _eq_u64_status_kernel(a: _I64, b: _I64, k: Int32, status: _I32):
    """status[0] = 1 when a[j] != b[j] for some j < k."""
    var j = _tid()
    if j < Int(k) and a.unsafe_load(j) != b.unsafe_load(j):
        status.unsafe_store(0, Int32(1))
