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

from checks.numerics import ftz, portable_cosf
from metrics.checks.pinned_sum import canonicalize_nan
from checks.soft_f64 import (
    sf64_add,
    sf64_div,
    sf64_floor,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_log,
    sf64_lt,
    sf64_mul,
    sf64_sub,
    sf64_to_f32,
)
from core.device_fold import device_exclusive_scan_total_from
from core.device_zero import enqueue_fill
from core.fast_radix_sort import (
    fast_radix_sort_pairs_u32,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)
from core.label_encode import label_sort_key
from core.label_encode_device import device_unique_inverse_resident

#: The dtype codes of `bindings/hotpath_helpers.mojo` (HP_F32 ...).
comptime HPD_F32 = 0
comptime HPD_F64 = 1
comptime HPD_I32 = 2
comptime HPD_I64 = 3
comptime HPD_U32 = 4
comptime HPD_U8 = 5

comptime HPD_TPB = 256
#: Elements one thread of a tile reduction folds.
comptime HPD_TILE = 256
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
# class_ratio_f64 (lane py-runtime-b)
# ===========================================================================


def _class_ratio_kernel(counts: _I64, k_: Int32, n_: Int64, mode: Int32, dst: _U64, status: _I32):
    """One class per thread: mode 0, scikit-learn's 'balanced' weight
    n / (k * count_c); mode 1, the prior count_c / n. Soft binary64 from
    exact integers (< 2^53) and one correctly rounded division: the words of
    Python's `int / int` and of the host column. A zero count (mode 0)
    writes +0 and sets status[0]."""
    var i = _tid()
    if i < Int(k_):
        var c = Int(counts.unsafe_load(i))
        if mode == 0:
            if c == 0:
                status.unsafe_store(0, Int32(1))
                dst.unsafe_store(i, UInt64(0))
            else:
                dst.unsafe_store(i, sf64_div(sf64_from_int(Int(n_)), sf64_from_int(Int(k_) * c)))
        else:
            dst.unsafe_store(i, sf64_div(sf64_from_int(c), sf64_from_int(Int(n_))))


def device_class_ratio_f64(
    ctx: DeviceContext, counts_addr: Int, k: Int, n: Int, mode: Int, dst_addr: Int,
) raises -> Int:
    """`class_ratio_f64` on the device: the k ratios into the float64 words
    at `dst_addr`. Returns the number of zero-count flags raised (0 or 1)."""
    var d_c = ctx.enqueue_create_buffer[DType.int64](k)
    var d_d = ctx.enqueue_create_buffer[DType.uint64](k)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_c, src_ptr=_I64(unsafe_from_address=counts_addr))
    ctx.enqueue_function[_class_ratio_kernel](
        d_c.unsafe_ptr(), Int32(k), Int64(n), Int32(mode), d_d.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    var zeros = _read_status(ctx, d_status, 0)
    ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=dst_addr), src_buf=d_d)
    ctx.synchronize()
    _ = d_c^
    _ = d_d^
    _ = d_status^
    return zeros


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


def _bincount2_i32_kernel(a: _I32, b: _I32, n_: Int32, ka: Int32, kb: Int32, cnt: _I32, status: _I32):
    """cnt[b[i] * ka + a[i]] += 1 (Int32 atomics: exact in any order); a
    code outside [0, ka) x [0, kb) sets status[0]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var u = Int(a.unsafe_load(i))
    var v = Int(b.unsafe_load(i))
    if u < 0 or u >= Int(ka) or v < 0 or v >= Int(kb):
        status.unsafe_store(0, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + (v * Int(ka) + u), Int32(1))


def device_bincount2_i32(
    ctx: DeviceContext, a_addr: Int, b_addr: Int, n: Int, ka: Int, kb: Int, counts_addr: Int,
) raises -> Bool:
    """The 2-D bincount of two int32 code columns (lane cpu4-python:
    StratifiedGroupKFold's group x class table, a host row loop before):
    int64 counts[v * ka + u] for each row's (u = a[i], v = b[i]), `n >= 1`,
    `ka * kb >= 1`. False (counts untouched) when a code is out of range."""
    var k = ka * kb
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_base = ctx.enqueue_create_buffer[DType.int64](k)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_base, Int64(0))
    enqueue_fill(ctx, d_status, Int32(0))
    var d_a = ctx.enqueue_create_buffer[DType.int32](n)
    var d_b = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_a, src_ptr=_I32(unsafe_from_address=a_addr))
    ctx.enqueue_copy(dst_buf=d_b, src_ptr=_I32(unsafe_from_address=b_addr))
    ctx.enqueue_function[_bincount2_i32_kernel](
        d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int32(n), Int32(ka), Int32(kb),
        d_cnt.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    if ok:
        ctx.enqueue_function[_widen_counts_kernel](
            d_cnt.unsafe_ptr(), Int32(k), d_base.unsafe_ptr(), Int32(0),
            grid_dim=_blocks(k), block_dim=HPD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=counts_addr), src_buf=d_base)
    ctx.synchronize()
    _ = d_a^
    _ = d_b^
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
    """dst[i] = ftz(ftz(dst[i]) + ftz(src[i])): one float32 rounding. A NaN
    sum is stored as the one canonical word (`canonicalize_nan`), as the
    host column does: NVIDIA's add returns 0x7FFFFFFF for every NaN while
    the host and AMD keep the input payload (lane/review-fixes)."""
    var i = _tid()
    if i < Int(n_):
        var a = ftz(dst.unsafe_load(i))
        var b = ftz(src.unsafe_load(i))
        var s = canonicalize_nan(ftz(a + b))
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
    `splits >= 2`). The rows are counted, ranked and dealt on the device,
    and the k x splits quota table is built there too (the helper's own
    integer formula, `_fold_quota_rows_kernel`). False when a code is out of
    range (nothing written)."""
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
    # lane/review-fixes: the k x splits quota table is built on the device
    # (was a host loop over k x splits entries between device phases): the
    # class starts are the exclusive scan of the counts in class order, each
    # class's offset the exclusive scan of the counts in `d_order`, and one
    # thread per class runs the helper's own integer formula over the folds.
    # Integers only: the same table, no bit moves.
    var d_start = ctx.enqueue_create_buffer[DType.int32](k + 1)
    var d_cum = ctx.enqueue_create_buffer[DType.int32](k * splits)
    var d_cnt64 = ctx.enqueue_create_buffer[DType.int64](k)
    var d_ocnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_ooff = ctx.enqueue_create_buffer[DType.int32](k + 1)
    device_exclusive_scan_total_from(ctx, d_cnt, d_start, k)
    ctx.enqueue_function[_gather_by_order_kernel](
        d_cnt.unsafe_ptr(), d_order.unsafe_ptr(), Int32(k), d_ocnt.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    device_exclusive_scan_total_from(ctx, d_ocnt, d_ooff, k)
    ctx.enqueue_function[_fold_quota_rows_kernel](
        d_cnt.unsafe_ptr(), d_order.unsafe_ptr(), d_ooff.unsafe_ptr(), Int32(k), Int32(splits),
        d_cum.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_widen_counts_kernel](
        d_cnt.unsafe_ptr(), Int32(k), d_cnt64.unsafe_ptr(), Int32(0),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_I64(unsafe_from_address=counts_addr), src_buf=d_cnt64)
    device_rank_folds(ctx, d_codes, n, k, splits, d_start, d_cum, 0, fold_addr, fold_counts_addr)
    _ = d_ocnt^
    _ = d_ooff^
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
    cumulative allocations) are built on the device from the caller's
    uploaded `counts` and `alloc` (`_strat_alloc_rows_kernel` and a scan);
    every row is ranked and assigned on the device. False (nothing written) when the tables are inconsistent or
    a class code is out of range: the binding reruns the host helper."""
    # lane/review-fixes: the k x n_folds tables are built on the device from
    # the uploaded `counts` and `alloc` (was a host loop between device
    # phases): one thread per class checks its allocations (non-negative,
    # summing to its count) into status[0] and writes its cumulative row;
    # the class starts are the exclusive scan of the counts. The caller's
    # counts must equal the rows' own counts (checked below), so they sum to
    # n. Integers only: the same tables, no bit moves.
    var cp = _I64(unsafe_from_address=counts_addr)
    var ap = _I64(unsafe_from_address=alloc_addr)
    var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](k)
    var d_cnt64 = ctx.enqueue_create_buffer[DType.int64](k)
    var d_ref64 = ctx.enqueue_create_buffer[DType.int64](k)
    var d_alloc = ctx.enqueue_create_buffer[DType.int64](k * n_folds)
    var d_ref32 = ctx.enqueue_create_buffer[DType.int32](k)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    var d_start = ctx.enqueue_create_buffer[DType.int32](k + 1)
    var d_cum = ctx.enqueue_create_buffer[DType.int32](k * n_folds)
    enqueue_fill(ctx, d_cnt, Int32(0))
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=_I32(unsafe_from_address=enc_addr))
    ctx.enqueue_copy(dst_buf=d_ref64, src_ptr=cp)
    ctx.enqueue_copy(dst_buf=d_alloc, src_ptr=ap)
    ctx.enqueue_function[_strat_alloc_rows_kernel](
        d_ref64.unsafe_ptr(), d_alloc.unsafe_ptr(), Int32(k), Int32(n_folds),
        d_ref32.unsafe_ptr(), d_cum.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=HPD_TPB,
    )
    device_exclusive_scan_total_from(ctx, d_ref32, d_start, k)
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
    _ = d_alloc^
    _ = d_ref32^
    _ = d_codes^
    _ = d_cnt^
    _ = d_cnt64^
    _ = d_ref64^
    _ = d_status^
    _ = d_start^
    _ = d_cum^
    return ok


def _gather_by_order_kernel(cnt: _I32, order: _U32, k_: Int32, dst: _I32):
    """dst[j] = cnt[order[j]] (lane/review-fixes)."""
    var j = _tid()
    if j < Int(k_):
        dst.unsafe_store(j, cnt.unsafe_load(Int(order.unsafe_load(j))))


def _fold_quota_rows_kernel(cnt: _I32, order: _U32, pos_off: _I32, k_: Int32, splits_: Int32, cum: _I32):
    """One thread per position j of `order` (lane/review-fixes): class
    c = order[j] dealt from offset pos_off[j] (the counts of the classes
    before it in `order`); cum[c * splits + fold] is the running number of
    its rows in folds 0 .. fold, the host helper's formula line for line."""
    var j = _tid()
    if j >= Int(k_):
        return
    var splits = Int(splits_)
    var c = Int(order.unsafe_load(j))
    var count = Int(cnt.unsafe_load(c))
    var offset = Int(pos_off.unsafe_load(j))
    var run = 0
    for fold in range(splits):
        var first = (fold - offset) % splits
        if first < 0:
            first += splits
        var take = 0
        if first < count:
            take = 1 + (count - 1 - first) // splits
        run += take
        cum.unsafe_store(c * splits + fold, Int32(run))


def _strat_alloc_rows_kernel(
    cnt64: _I64, alloc: _I64, k_: Int32, n_folds_: Int32, cnt32: _I32, cum: _I32, status: _I32
):
    """One thread per class c (lane/review-fixes): cnt32[c] = counts[c],
    cum[c * n_folds + f] = alloc[0 .. f][c] summed; status[0] = 1 when a
    count or an allocation is negative or the allocations miss the count."""
    var c = _tid()
    var k = Int(k_)
    if c >= k:
        return
    var n_folds = Int(n_folds_)
    var count = Int(cnt64.unsafe_load(c))
    var bad = False
    if count < 0:
        bad = True
        count = 0
    cnt32.unsafe_store(c, Int32(count))
    var run = 0
    for f in range(n_folds):
        var a = Int(alloc.unsafe_load(f * k + c))
        if a < 0:
            bad = True
            a = 0
        run += a
        cum.unsafe_store(c * n_folds + f, Int32(run))
    if run != count:
        bad = True
    if bad:
        status.unsafe_store(0, Int32(1))


def _eq_u64_status_kernel(a: _I64, b: _I64, k: Int32, status: _I32):
    """status[0] = 1 when a[j] != b[j] for some j < k."""
    var j = _tid()
    if j < Int(k) and a.unsafe_load(j) != b.unsafe_load(j):
        status.unsafe_store(0, Int32(1))


# ===========================================================================
# reduce_stat: min, max, argmax (first extreme wins) and the integral test
# ===========================================================================
#
# Python's sequential `min` / `max` / argmax keep the FIRST element unless a
# later one is strictly smaller (larger); a NaN after the first element never
# wins a comparison. With the first element not NaN (the binding tests it),
# the answer is the extreme of the non-NaN elements, the lowest index among
# equals (-0.0 equals 0.0). That is an order-free reduction over the pairs
# (ordered key, index): tiles of HPD_TILE elements, then tiles of tile
# results, until one pair is left. Exact: integer compares only.


@always_inline
def _key_f32(b: UInt32) -> UInt64:
    """An unsigned key in a non-NaN float32's order, -0.0 folded onto 0.0."""
    var x = b
    if x == UInt32(0x80000000):
        x = UInt32(0)
    if (x & UInt32(0x80000000)) != UInt32(0):
        return UInt64(~x)
    return UInt64(x ^ UInt32(0x80000000))


def _reduce_first_kernel(
    s8: _U8, s32: _U32, s64: _U64, code: Int32, n_: Int32, want_max: Int32,
    key_out: _U64, idx_out: _U32,
):
    """Thread j folds elements [j * HPD_TILE, (j + 1) * HPD_TILE) ascending:
    the extreme key and its first index. NaN elements are skipped; a tile of
    NaNs stores the key that loses to every real key."""
    var j = _tid()
    var n = Int(n_)
    var lo = j * HPD_TILE
    if lo >= n:
        return
    var hi = min(n, lo + HPD_TILE)
    var best_k = UInt64(0)
    var best_i = lo
    var have = False
    var i = lo
    while i < hi:
        var valid = True
        var k = UInt64(0)
        if code == HPD_U8:
            k = UInt64(s8.unsafe_load(i))
        elif code == HPD_F64:
            var w = s64.unsafe_load(i)
            if sf64_is_nan(w):
                valid = False
            k = label_sort_key(w, 0)
        elif code == HPD_I64:
            k = label_sort_key(s64.unsafe_load(i), 1)
        else:
            var u = s32.unsafe_load(i)
            if code == HPD_F32:
                if (u & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
                    valid = False
                k = _key_f32(u)
            elif code == HPD_I32:
                k = UInt64(u ^ UInt32(0x80000000))
            else:
                k = UInt64(u)
        if valid:
            var better = not have
            if have:
                if want_max != 0:
                    better = k > best_k
                else:
                    better = k < best_k
            if better:
                best_k = k
                best_i = i
                have = True
        i += 1
    if not have:
        best_k = UInt64(0) if want_max != 0 else UInt64(0xFFFFFFFFFFFFFFFF)
    key_out.unsafe_store(j, best_k)
    idx_out.unsafe_store(j, UInt32(best_i))


def _reduce_pass_kernel(
    key_in: _U64, idx_in: _U32, m_: Int32, want_max: Int32, key_out: _U64, idx_out: _U32,
):
    """Thread j folds tile results [j * HPD_TILE, (j + 1) * HPD_TILE)
    ascending; equal keys keep the earlier (lower-index) entry."""
    var j = _tid()
    var m = Int(m_)
    var lo = j * HPD_TILE
    if lo >= m:
        return
    var hi = min(m, lo + HPD_TILE)
    var best_k = key_in.unsafe_load(lo)
    var best_i = idx_in.unsafe_load(lo)
    var i = lo + 1
    while i < hi:
        var k = key_in.unsafe_load(i)
        var better = k < best_k
        if want_max != 0:
            better = k > best_k
        if better:
            best_k = k
            best_i = idx_in.unsafe_load(i)
        i += 1
    key_out.unsafe_store(j, best_k)
    idx_out.unsafe_store(j, best_i)


@always_inline
def _tiles(m: Int) -> Int:
    return (m + HPD_TILE - 1) // HPD_TILE


def device_reduce_arg(
    ctx: DeviceContext, addr: Int, code: Int, n: Int, want_max: Bool,
) raises -> Int:
    """The index of the first minimum (or maximum) of the n elements of
    dtype `code` at `addr`, NaNs skipped (`1 <= n <= HPD_MAX_N`; the caller
    has checked that element 0 is not NaN)."""
    var wide = code == HPD_F64 or code == HPD_I64
    var narrow8 = code == HPD_U8
    var d_s8 = ctx.enqueue_create_buffer[DType.uint8](n if narrow8 else 1)
    var d_s32 = ctx.enqueue_create_buffer[DType.uint32](n if (not wide and not narrow8) else 1)
    var d_s64 = ctx.enqueue_create_buffer[DType.uint64](n if wide else 1)
    if narrow8:
        ctx.enqueue_copy(dst_buf=d_s8, src_ptr=_U8(unsafe_from_address=addr))
    elif wide:
        ctx.enqueue_copy(dst_buf=d_s64, src_ptr=_U64(unsafe_from_address=addr))
    else:
        ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_U32(unsafe_from_address=addr))
    var m1 = _tiles(n)
    var m2 = _tiles(m1)
    var m3 = _tiles(m2)
    var d_k1 = ctx.enqueue_create_buffer[DType.uint64](m1)
    var d_i1 = ctx.enqueue_create_buffer[DType.uint32](m1)
    var d_k2 = ctx.enqueue_create_buffer[DType.uint64](m2)
    var d_i2 = ctx.enqueue_create_buffer[DType.uint32](m2)
    var d_k3 = ctx.enqueue_create_buffer[DType.uint64](m3)
    var d_i3 = ctx.enqueue_create_buffer[DType.uint32](m3)
    var d_k4 = ctx.enqueue_create_buffer[DType.uint64](1)
    var d_i4 = ctx.enqueue_create_buffer[DType.uint32](1)
    var wm = Int32(1) if want_max else Int32(0)
    ctx.enqueue_function[_reduce_first_kernel](
        d_s8.unsafe_ptr(), d_s32.unsafe_ptr(), d_s64.unsafe_ptr(), Int32(code), Int32(n), wm,
        d_k1.unsafe_ptr(), d_i1.unsafe_ptr(), grid_dim=_blocks(m1), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_reduce_pass_kernel](
        d_k1.unsafe_ptr(), d_i1.unsafe_ptr(), Int32(m1), wm, d_k2.unsafe_ptr(), d_i2.unsafe_ptr(),
        grid_dim=_blocks(m2), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_reduce_pass_kernel](
        d_k2.unsafe_ptr(), d_i2.unsafe_ptr(), Int32(m2), wm, d_k3.unsafe_ptr(), d_i3.unsafe_ptr(),
        grid_dim=_blocks(m3), block_dim=HPD_TPB,
    )
    # m3 <= HPD_TILE for every n <= HPD_MAX_N (n / 256^3 < 128): one thread
    ctx.enqueue_function[_reduce_pass_kernel](
        d_k3.unsafe_ptr(), d_i3.unsafe_ptr(), Int32(m3), wm, d_k4.unsafe_ptr(), d_i4.unsafe_ptr(),
        grid_dim=_blocks(1), block_dim=HPD_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_i4)
    ctx.synchronize()
    var at = Int(h.unsafe_ptr()[0])
    _ = h^
    _ = d_s8^
    _ = d_s32^
    _ = d_s64^
    _ = d_k1^
    _ = d_i1^
    _ = d_k2^
    _ = d_i2^
    _ = d_k3^
    _ = d_i3^
    _ = d_k4^
    _ = d_i4^
    return at


def _integral_kernel(s32: _F32, s64: _U64, code: Int32, n_: Int32, status: _I32):
    """status[0] = 1 when an element is not finite or not integer valued
    (binary64 words: finite, and equal to its own floor)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var w = UInt64(0)
    if code == HPD_F64:
        w = s64.unsafe_load(i)
    else:
        w = sf64_from_f32(s32.unsafe_load(i))
    var e = (w >> 52) & UInt64(0x7FF)
    if e == UInt64(0x7FF) or sf64_floor(w) != w:
        status.unsafe_store(0, Int32(1))


def device_all_integral(ctx: DeviceContext, addr: Int, code: Int, n: Int) raises -> Bool:
    """`reduce_stat`'s integral test over float32 (`HPD_F32`) or float64."""
    var wide = code == HPD_F64
    var d_s32 = ctx.enqueue_create_buffer[DType.float32](1 if wide else n)
    var d_s64 = ctx.enqueue_create_buffer[DType.uint64](n if wide else 1)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    if wide:
        ctx.enqueue_copy(dst_buf=d_s64, src_ptr=_U64(unsafe_from_address=addr))
    else:
        ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_F32(unsafe_from_address=addr))
    ctx.enqueue_function[_integral_kernel](
        d_s32.unsafe_ptr(), d_s64.unsafe_ptr(), Int32(code), Int32(n), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    _ = d_s32^
    _ = d_s64^
    _ = d_status^
    return ok


# ===========================================================================
# uniform_init_f32
# ===========================================================================


def _uniform_init_kernel(dst: _F32, n_: Int32, seed: UInt64, off: UInt64, lo: UInt64, span: UInt64):
    """dst[i] = float32(lo + span * u_i) in binary64 words, u_i the 53-bit
    uniform of splitmix64 at counter off + i of the seed: the host helper's
    statement, value for value (`sequence/schedule.mojo::splitmix64`; times
    2^-53 is exact; one fused multiply-add, one narrowing). Lane
    fix-s1-shared: the host helper's `lo + span * u` is an explicit `fma`
    now (Mojo's default fp-mode contracts it on the host anyway, so a
    separate multiply and add here could differ in the last bit)."""
    var i = _tid()
    if i >= Int(n_):
        return
    var s = seed + (off + UInt64(i)) * UInt64(0x9E3779B97F4A7C15)
    s += UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> 31)
    var u = sf64_mul(sf64_from_int(Int(z >> 11)), UInt64(0x3CA0000000000000))
    dst.unsafe_store(i, sf64_to_f32(sf64_fma(span, u, lo)))


def device_uniform_init_f32(
    ctx: DeviceContext, dst_addr: Int, n: Int, low: Float64, high: Float64, seed: UInt64, offset: UInt64,
) raises:
    """`uniform_init_f32` (`1 <= n <= HPD_MAX_N`, finite `low` and `high`)."""
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[_uniform_init_kernel](
        d.unsafe_ptr(), Int32(n), seed, offset,
        bitcast[DType.uint64](low), bitcast[DType.uint64](high - low),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_F32(unsafe_from_address=dst_addr), src_buf=d)
    ctx.synchronize()
    _ = d^


# ===========================================================================
# cast_f64_to_f32 (the default since lane cpu2-l1-input; it was the candidate
# arm IDN_HPDEV_CAST_F64: bindings/hotpath_device.mojo)
# ===========================================================================


def _narrow_f64_kernel(src: _U64, n_: Int32, dst: _U32):
    """dst[i] = the float32 a hardware double-to-float conversion gives:
    round to nearest even (`sf64_to_f32`); a NaN keeps its sign and the top
    22 payload bits and is quieted, as the host cast leaves it."""
    var i = _tid()
    if i >= Int(n_):
        return
    var w = src.unsafe_load(i)
    if sf64_is_nan(w):
        var sign = UInt32((w >> 63) << 31)
        var payload = UInt32((w >> 29) & UInt64(0x003FFFFF))
        dst.unsafe_store(i, sign | UInt32(0x7FC00000) | payload)
    else:
        dst.unsafe_store(i, bitcast[DType.uint32](sf64_to_f32(w)))


def device_cast_f64_to_f32(ctx: DeviceContext, src_addr: Int, dst_addr: Int, n: Int) raises:
    """`cast_f64_to_f32` (`1 <= n <= HPD_MAX_N`): the float64 words go up,
    the float32 words come down."""
    var d_s = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_d = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_s, src_ptr=_U64(unsafe_from_address=src_addr))
    ctx.enqueue_function[_narrow_f64_kernel](
        d_s.unsafe_ptr(), Int32(n), d_d.unsafe_ptr(), grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_U32(unsafe_from_address=dst_addr), src_buf=d_d)
    ctx.synchronize()
    _ = d_s^
    _ = d_d^


# ===========================================================================
# reduce_stat's exact integer sum (lane fix-s1-shared, IDN_HPDEV_ISUM)
# ===========================================================================
#
# Python's `sum` of integers is exact, so any order gives the same total: a
# 128-bit two's complement accumulator (hi, lo) per tile of HPD_TILE
# elements, then tiles of tile sums, as `device_reduce_arg` folds. Integer
# adds with carry only: the host helper's (hi, lo) on every vendor.


@always_inline
def _isum_add(mut acc_hi: UInt64, mut acc_lo: UInt64, add_hi: UInt64, add_lo: UInt64):
    var nlo = acc_lo + add_lo
    var carry = UInt64(1) if nlo < acc_lo else UInt64(0)
    acc_hi = acc_hi + add_hi + carry
    acc_lo = nlo


def _isum_first_kernel(
    s8: _U8, s32: _U32, s64: _U64, code: Int32, n_: Int32, hi_out: _U64, lo_out: _U64,
):
    """Thread j sums elements [j * HPD_TILE, (j + 1) * HPD_TILE), each
    sign-extended to 128 bits."""
    var j = _tid()
    var n = Int(n_)
    var lo = j * HPD_TILE
    if lo >= n:
        return
    var hi = min(n, lo + HPD_TILE)
    var acc_hi = UInt64(0)
    var acc_lo = UInt64(0)
    var i = lo
    while i < hi:
        var v: Int64
        if code == HPD_U8:
            v = s8.unsafe_load(i).cast[DType.int64]()
        elif code == HPD_I64:
            v = bitcast[DType.int64](s64.unsafe_load(i))
        elif code == HPD_I32:
            v = bitcast[DType.int32](s32.unsafe_load(i)).cast[DType.int64]()
        else:
            v = s32.unsafe_load(i).cast[DType.int64]()
        var ext = UInt64(0xFFFFFFFFFFFFFFFF) if v < 0 else UInt64(0)
        _isum_add(acc_hi, acc_lo, ext, bitcast[DType.uint64](v))
        i += 1
    hi_out.unsafe_store(j, acc_hi)
    lo_out.unsafe_store(j, acc_lo)


def _isum_pass_kernel(hi_in: _U64, lo_in: _U64, m_: Int32, hi_out: _U64, lo_out: _U64):
    """Thread j sums the 128-bit tile sums [j * HPD_TILE, (j + 1) * HPD_TILE)."""
    var j = _tid()
    var m = Int(m_)
    var lo = j * HPD_TILE
    if lo >= m:
        return
    var hi = min(m, lo + HPD_TILE)
    var acc_hi = UInt64(0)
    var acc_lo = UInt64(0)
    var i = lo
    while i < hi:
        _isum_add(acc_hi, acc_lo, hi_in.unsafe_load(i), lo_in.unsafe_load(i))
        i += 1
    hi_out.unsafe_store(j, acc_hi)
    lo_out.unsafe_store(j, acc_lo)


def device_isum(ctx: DeviceContext, addr: Int, code: Int, n: Int) raises -> Tuple[UInt64, UInt64]:
    """The exact sum of the n integers (`HPD_I32`, `HPD_I64`, `HPD_U32` or
    `HPD_U8`) at `addr` as the 128-bit two's complement (hi, lo)
    (`1 <= n <= HPD_MAX_N`)."""
    var wide = code == HPD_I64
    var narrow8 = code == HPD_U8
    var d_s8 = ctx.enqueue_create_buffer[DType.uint8](n if narrow8 else 1)
    var d_s32 = ctx.enqueue_create_buffer[DType.uint32](n if (not wide and not narrow8) else 1)
    var d_s64 = ctx.enqueue_create_buffer[DType.uint64](n if wide else 1)
    if narrow8:
        ctx.enqueue_copy(dst_buf=d_s8, src_ptr=_U8(unsafe_from_address=addr))
    elif wide:
        ctx.enqueue_copy(dst_buf=d_s64, src_ptr=_U64(unsafe_from_address=addr))
    else:
        ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_U32(unsafe_from_address=addr))
    var m1 = _tiles(n)
    var m2 = _tiles(m1)
    var m3 = _tiles(m2)
    var d_h1 = ctx.enqueue_create_buffer[DType.uint64](m1)
    var d_l1 = ctx.enqueue_create_buffer[DType.uint64](m1)
    var d_h2 = ctx.enqueue_create_buffer[DType.uint64](m2)
    var d_l2 = ctx.enqueue_create_buffer[DType.uint64](m2)
    var d_h3 = ctx.enqueue_create_buffer[DType.uint64](m3)
    var d_l3 = ctx.enqueue_create_buffer[DType.uint64](m3)
    var d_h4 = ctx.enqueue_create_buffer[DType.uint64](1)
    var d_l4 = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[_isum_first_kernel](
        d_s8.unsafe_ptr(), d_s32.unsafe_ptr(), d_s64.unsafe_ptr(), Int32(code), Int32(n),
        d_h1.unsafe_ptr(), d_l1.unsafe_ptr(), grid_dim=_blocks(m1), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_isum_pass_kernel](
        d_h1.unsafe_ptr(), d_l1.unsafe_ptr(), Int32(m1), d_h2.unsafe_ptr(), d_l2.unsafe_ptr(),
        grid_dim=_blocks(m2), block_dim=HPD_TPB,
    )
    ctx.enqueue_function[_isum_pass_kernel](
        d_h2.unsafe_ptr(), d_l2.unsafe_ptr(), Int32(m2), d_h3.unsafe_ptr(), d_l3.unsafe_ptr(),
        grid_dim=_blocks(m3), block_dim=HPD_TPB,
    )
    # m3 <= HPD_TILE for every n <= HPD_MAX_N: one thread
    ctx.enqueue_function[_isum_pass_kernel](
        d_h3.unsafe_ptr(), d_l3.unsafe_ptr(), Int32(m3), d_h4.unsafe_ptr(), d_l4.unsafe_ptr(),
        grid_dim=_blocks(1), block_dim=HPD_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.uint64](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_h4)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr() + 1, src_buf=d_l4)
    ctx.synchronize()
    var hi_w = h.unsafe_ptr()[0]
    var lo_w = h.unsafe_ptr()[1]
    _ = h^
    _ = d_s8^
    _ = d_s32^
    _ = d_s64^
    _ = d_h1^
    _ = d_l1^
    _ = d_h2^
    _ = d_l2^
    _ = d_h3^
    _ = d_l3^
    _ = d_h4^
    _ = d_l4^
    return (hi_w, lo_w)


# ===========================================================================
# normal_init_f32 (lane fix-s1-shared, IDN_HPDEV_NORMAL)
# ===========================================================================


@always_inline
def _hpd_lead(u: UInt64) -> Int:
    """The index of the highest set bit of `u` (u != 0)."""
    var x = u
    var n = 0
    if (x >> 32) != UInt64(0):
        x = x >> 32
        n += 32
    if (x >> 16) != UInt64(0):
        x = x >> 16
        n += 16
    if (x >> 8) != UInt64(0):
        x = x >> 8
        n += 8
    if (x >> 4) != UInt64(0):
        x = x >> 4
        n += 4
    if (x >> 2) != UInt64(0):
        x = x >> 2
        n += 2
    if (x >> 1) != UInt64(0):
        n += 1
    return n


def _hpd_sf64_sqrt(a: UInt64) -> UInt64:
    """The correctly rounded binary64 square root of the word `a` (IEEE
    `sqrt`, round to nearest even), integers only: the host `sqrt` on every
    vendor. A NaN or a negative gives the quiet NaN; +-0 and +inf return
    themselves.

    With a = sig * 2^e2, e2 made even, X = sig << 54 lies in [2^106, 2^108),
    so q = floor(sqrt(X)) has exactly 54 bits: 53 result bits and the round
    bit, the remainder X - q^2 the sticky bit. q is found two bits of X at a
    time (the restoring digit-by-digit root) in 128-bit (hi, lo) words."""
    if sf64_is_nan(a):
        return UInt64(0x7FF8000000000000)
    if (a & UInt64(0x7FFFFFFFFFFFFFFF)) == UInt64(0):
        return a
    if (a >> 63) != UInt64(0):
        return UInt64(0x7FF8000000000000)
    var e = Int((a >> 52) & UInt64(0x7FF))
    if e == 0x7FF:
        return a
    var sig = a & UInt64(0x000FFFFFFFFFFFFF)
    if e == 0:
        var lead = _hpd_lead(sig)
        sig = sig << UInt64(52 - lead)
        e = 1 - (52 - lead)
    else:
        sig = sig | UInt64(0x0010000000000000)
    var e2 = e - 1075
    if (e2 & 1) != 0:
        sig = sig << 1
        e2 -= 1
    # X = sig << 54
    var xh = sig >> 10
    var xl = sig << 54
    var rh = UInt64(0)
    var rl = UInt64(0)
    # bit = 4^53 = 2^106
    var bh = UInt64(1) << 42
    var bl = UInt64(0)
    for _ in range(54):
        var tl = rl + bl
        var th = rh + bh + (UInt64(1) if tl < rl else UInt64(0))
        if xh > th or (xh == th and xl >= tl):
            var borrow = UInt64(1) if xl < tl else UInt64(0)
            xl = xl - tl
            xh = xh - th - borrow
            rl = (rl >> 1) | (rh << 63)
            rh = rh >> 1
            var sl = rl + bl
            rh = rh + bh + (UInt64(1) if sl < rl else UInt64(0))
            rl = sl
        else:
            rl = (rl >> 1) | (rh << 63)
            rh = rh >> 1
        bl = (bl >> 2) | (bh << 62)
        bh = bh >> 2
    # rh == 0 here: q < 2^54
    var q = rl
    var sticky = (xh | xl) != UInt64(0)
    var m = q >> 1
    var ex = ((e2 - 54) >> 1) + 1076
    if (q & UInt64(1)) != UInt64(0) and (sticky or (m & UInt64(1)) != UInt64(0)):
        m += UInt64(1)
        if m == (UInt64(1) << 53):
            m = m >> 1
            ex += 1
    return (UInt64(ex) << 52) | (m & UInt64(0x000FFFFFFFFFFFFF))


@always_inline
def _hpd_splitmix_at(seed: UInt64, c: UInt64) -> UInt64:
    """`sequence/schedule.mojo::splitmix64` of the state seed + c * gamma."""
    var s = seed + c * UInt64(0x9E3779B97F4A7C15)
    s += UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _normal_init_kernel(dst: _F32, n_: Int32, seed: UInt64, off: UInt64, mu: UInt64, sd: UInt64):
    """dst[i] = float32(fma(sd, r * cos, mu)), the host helper's Box-Muller
    statement for statement in binary64 words: u1 = 1 - U(2c), u2 = U(2c + 1)
    (U the 53-bit splitmix64 uniform, times 2^-53 exact), r = sqrt(-2 log u1)
    (`sf64_log` is `portable_log64` statement for statement, the root is
    correctly rounded as the host `sqrt`), the angle narrowed to float32 and
    `portable_cosf` of it, then one widening, one multiply and one fused
    multiply-add."""
    var i = _tid()
    if i >= Int(n_):
        return
    var c = (off + UInt64(i)) * UInt64(2)
    var a1 = sf64_mul(sf64_from_int(Int(_hpd_splitmix_at(seed, c) >> 11)), UInt64(0x3CA0000000000000))
    var u1 = sf64_sub(UInt64(0x3FF0000000000000), a1)
    var u2 = sf64_mul(
        sf64_from_int(Int(_hpd_splitmix_at(seed, c + UInt64(1)) >> 11)), UInt64(0x3CA0000000000000)
    )
    # -2.0 * log(u1), then the root
    var r = _hpd_sf64_sqrt(sf64_mul(UInt64(0xC000000000000000), sf64_log(u1)))
    # 6.283185307179586 * u2, narrowed
    var cz = portable_cosf(sf64_to_f32(sf64_mul(UInt64(0x401921FB54442D18), u2)))
    var z = sf64_mul(r, sf64_from_f32(cz))
    dst.unsafe_store(i, sf64_to_f32(sf64_fma(sd, z, mu)))


def device_normal_init_f32(
    ctx: DeviceContext, dst_addr: Int, n: Int, mean: Float64, std: Float64, seed: UInt64, offset: UInt64,
) raises:
    """`normal_init_f32` (`1 <= n <= HPD_MAX_N`, finite `mean` and `std`)."""
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[_normal_init_kernel](
        d.unsafe_ptr(), Int32(n), seed, offset,
        bitcast[DType.uint64](mean), bitcast[DType.uint64](std),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_F32(unsafe_from_address=dst_addr), src_buf=d)
    ctx.synchronize()
    _ = d^


# ===========================================================================
# cast_elements (lane fix-s1-shared, IDN_HPDEV_CAST)
# ===========================================================================
#
# The host helper's one conversion per element (`bindings/hotpath_helpers.mojo`
# `_convert` / `_refused`, DEVIATION 3100) on bits: an integer source is held
# as an Int64, a float source as its binary64 word (a float32 widens
# exactly). Integer -> float rounds once to binary64 (round to nearest even,
# as the C conversion) and, for a float32 target, once more to float32 (the
# item setter's double rounding). Float -> integer refuses NaN, infinities
# and values outside the host's bounds (the same strict/non-strict compares
# on binary64 words) and truncates toward zero. NaN payloads move as the
# hardware conversions move them: float32 -> float32 sets the quiet bit,
# float32 -> float64 widens the payload with the quiet bit set,
# float64 -> float32 keeps the top 22 payload bits, quiet. Any refused
# element sets the status word and the binding reruns the host helper.

#: Bounds of the host helper's `_refused`, as binary64 words.
comptime _W_I64_LO = UInt64(0xC3E0000000000000)  # -2^63 (>=)
comptime _W_I64_HI = UInt64(0x43E0000000000000)  # 2^63 (<)
comptime _W_I32_LO = UInt64(0xC1E0000000200000)  # -2147483649.0 (>)
comptime _W_I32_HI = UInt64(0x41E0000000000000)  # 2^31 (<)
comptime _W_NEG1 = UInt64(0xBFF0000000000000)  # -1.0 (>)
comptime _W_U32_HI = UInt64(0x41F0000000000000)  # 2^32 (<)
comptime _W_U8_HI = UInt64(0x4070000000000000)  # 256.0 (<)


@always_inline
def _hpd_i64_to_f64(v: Int64) -> UInt64:
    """The binary64 word of `v`, rounded to nearest even (the hardware
    int64 -> double conversion; exact below 2^53 in magnitude)."""
    if v == 0:
        return UInt64(0)
    var bits = bitcast[DType.uint64](v)
    var s = bits >> 63
    var u = bits
    if s != UInt64(0):
        u = ~bits + UInt64(1)
    var lead = _hpd_lead(u)
    if lead <= 52:
        return (s << 63) | (UInt64(lead + 1023) << 52) | (
            (u << UInt64(52 - lead)) & UInt64(0x000FFFFFFFFFFFFF)
        )
    var sh = lead - 52
    var m = u >> UInt64(sh)
    var rem = u & ((UInt64(1) << UInt64(sh)) - UInt64(1))
    var half = UInt64(1) << UInt64(sh - 1)
    if rem > half or (rem == half and (m & UInt64(1)) != UInt64(0)):
        m += UInt64(1)
        if m == (UInt64(1) << 53):
            m = m >> 1
            lead += 1
    return (s << 63) | (UInt64(lead + 1023) << 52) | (m & UInt64(0x000FFFFFFFFFFFFF))


@always_inline
def _hpd_trunc_bits(w: UInt64) -> UInt64:
    """The two's complement bits of the finite binary64 word `w` truncated
    toward zero (|w| < 2^63, or w == -2^63)."""
    var e = Int((w >> 52) & UInt64(0x7FF)) - 1023
    if e < 0:
        return UInt64(0)
    var m = (w & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x0010000000000000)
    var v: UInt64
    if e >= 52:
        v = m << UInt64(e - 52)
    else:
        v = m >> UInt64(52 - e)
    if (w >> 63) != UInt64(0):
        return ~v + UInt64(1)
    return v


def _cast_kernel(
    s8: _U8, s32: _U32, s64: _U64, sc: Int32, dc: Int32, n_: Int32,
    d8: _U8, d32: _U32, d64: _U64, status: _I32,
):
    var i = _tid()
    if i >= Int(n_):
        return
    var src_float = sc == HPD_F32 or sc == HPD_F64
    var raw32 = UInt32(0)
    var raw64 = UInt64(0)
    var iv = Int64(0)
    var w = UInt64(0)
    var nan = False
    if sc == HPD_U8:
        iv = s8.unsafe_load(i).cast[DType.int64]()
    elif sc == HPD_F64 or sc == HPD_I64:
        raw64 = s64.unsafe_load(i)
        if sc == HPD_I64:
            iv = bitcast[DType.int64](raw64)
        else:
            nan = sf64_is_nan(raw64)
            w = raw64
    else:
        raw32 = s32.unsafe_load(i)
        if sc == HPD_I32:
            iv = bitcast[DType.int32](raw32).cast[DType.int64]()
        elif sc == HPD_U32:
            iv = raw32.cast[DType.int64]()
        else:
            nan = (raw32 & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)
            if not nan:
                w = sf64_from_f32(bitcast[DType.float32](raw32))
    # ---- float destinations: never refused
    if dc == HPD_F32 or dc == HPD_F64:
        if not src_float:
            var fw = _hpd_i64_to_f64(iv)
            if dc == HPD_F32:
                d32.unsafe_store(i, bitcast[DType.uint32](sf64_to_f32(fw)))
            else:
                d64.unsafe_store(i, fw)
        elif dc == HPD_F32:
            if sc == HPD_F32:
                var q = raw32
                if nan:
                    q = raw32 | UInt32(0x00400000)
                d32.unsafe_store(i, q)
            elif nan:
                var sign = UInt32((raw64 >> 63) << 31)
                var payload = UInt32((raw64 >> 29) & UInt64(0x003FFFFF))
                d32.unsafe_store(i, sign | UInt32(0x7FC00000) | payload)
            else:
                d32.unsafe_store(i, bitcast[DType.uint32](sf64_to_f32(raw64)))
        else:
            if sc == HPD_F64:
                d64.unsafe_store(i, raw64)
            elif nan:
                var sign64 = raw32.cast[DType.uint64]() >> 31
                var frac = (raw32 & UInt32(0x007FFFFF)).cast[DType.uint64]()
                d64.unsafe_store(i, (sign64 << 63) | UInt64(0x7FF8000000000000) | (frac << 29))
            else:
                d64.unsafe_store(i, w)
        return
    # ---- integer destinations
    var ok = False
    var bits = UInt64(0)
    if src_float:
        if not nan:
            if dc == HPD_I64:
                ok = (not sf64_lt(w, _W_I64_LO)) and sf64_lt(w, _W_I64_HI)
            elif dc == HPD_I32:
                ok = sf64_gt(w, _W_I32_LO) and sf64_lt(w, _W_I32_HI)
            elif dc == HPD_U32:
                ok = sf64_gt(w, _W_NEG1) and sf64_lt(w, _W_U32_HI)
            else:
                ok = sf64_gt(w, _W_NEG1) and sf64_lt(w, _W_U8_HI)
        if ok:
            bits = _hpd_trunc_bits(w)
    else:
        if dc == HPD_I64:
            ok = True
        elif dc == HPD_I32:
            ok = iv >= -2147483648 and iv <= 2147483647
        elif dc == HPD_U32:
            ok = iv >= 0 and iv <= 4294967295
        else:
            ok = iv >= 0 and iv <= 255
        bits = bitcast[DType.uint64](iv)
    if not ok:
        status.unsafe_store(0, Int32(1))
        return
    if dc == HPD_U8:
        d8.unsafe_store(i, (bits & UInt64(0xFF)).cast[DType.uint8]())
    elif dc == HPD_I64:
        d64.unsafe_store(i, bits)
    else:
        d32.unsafe_store(i, (bits & UInt64(0xFFFFFFFF)).cast[DType.uint32]())


@always_inline
def _hpd_width(code: Int) -> Int:
    """Bytes per element of a dtype code."""
    if code == HPD_U8:
        return 1
    if code == HPD_F64 or code == HPD_I64:
        return 8
    return 4


def device_cast_elements(
    ctx: DeviceContext, src_addr: Int, src_code: Int, dst_addr: Int, dst_code: Int, n: Int,
) raises -> Bool:
    """`cast_elements` (`1 <= n <= HPD_MAX_N`, valid codes): True when the
    device wrote `dst`; False when some element is one the host helper
    refuses (`dst` untouched; the caller reruns the host helper). `src` may
    equal `dst`: the source is uploaded before anything comes down."""
    var sw = _hpd_width(src_code)
    var dw = _hpd_width(dst_code)
    var d_s8 = ctx.enqueue_create_buffer[DType.uint8](n if sw == 1 else 1)
    var d_s32 = ctx.enqueue_create_buffer[DType.uint32](n if sw == 4 else 1)
    var d_s64 = ctx.enqueue_create_buffer[DType.uint64](n if sw == 8 else 1)
    var d_d8 = ctx.enqueue_create_buffer[DType.uint8](n if dw == 1 else 1)
    var d_d32 = ctx.enqueue_create_buffer[DType.uint32](n if dw == 4 else 1)
    var d_d64 = ctx.enqueue_create_buffer[DType.uint64](n if dw == 8 else 1)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    if sw == 1:
        ctx.enqueue_copy(dst_buf=d_s8, src_ptr=_U8(unsafe_from_address=src_addr))
    elif sw == 8:
        ctx.enqueue_copy(dst_buf=d_s64, src_ptr=_U64(unsafe_from_address=src_addr))
    else:
        ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_U32(unsafe_from_address=src_addr))
    ctx.enqueue_function[_cast_kernel](
        d_s8.unsafe_ptr(), d_s32.unsafe_ptr(), d_s64.unsafe_ptr(), Int32(src_code), Int32(dst_code),
        Int32(n), d_d8.unsafe_ptr(), d_d32.unsafe_ptr(), d_d64.unsafe_ptr(), d_status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=HPD_TPB,
    )
    var ok = _read_status(ctx, d_status, 0) == 0
    if ok:
        if dw == 1:
            ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_d8)
        elif dw == 8:
            ctx.enqueue_copy(dst_ptr=_U64(unsafe_from_address=dst_addr), src_buf=d_d64)
        else:
            ctx.enqueue_copy(dst_ptr=_U32(unsafe_from_address=dst_addr), src_buf=d_d32)
        ctx.synchronize()
    _ = d_s8^
    _ = d_s32^
    _ = d_s64^
    _ = d_d8^
    _ = d_d32^
    _ = d_d64^
    _ = d_status^
    return ok
