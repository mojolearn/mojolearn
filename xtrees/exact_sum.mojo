# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The exactly rounded sum of a float32 vector (`_portable_math.fsum` of its
values, bit for bit) and AdaBoost's normalized sample weights, on the device
for every GPU build and serially on the host column (lane
apple-fast-py2mojo-trees, 2026-10-03).

THE LAW. Every finite float32 is an integer multiple of 2^-149. Its integer
is added, exactly, into 32-bit places held in Int64 limbs (`ES_LIMBS`; limb
i weighs 2^(32 i)); no rounding happens, so the order of the adds and of the
block reduction cannot change the integer. `es_round` then rounds the
integer times 2^-149 ONCE to binary64, nearest/even: `_portable_math.
_scaled_integer(total, -149)`, the last step of the Python fsum. So the
device, every vendor and the host give the Python word. A limb takes at most
2^30 terms below 2^32 each (`ES_MAX_ROWS`), so no limb can overflow.

The normalized weights are w[i] / total, one binary64 division each: the
host column divides in hardware, the device with `sf64_div` (the Apple GPU
has no float64), both correctly rounded, so the words agree.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.soft_f64 import sf64_div
from xtrees.glue_device import widen_f32_word
from xtrees.ops_device import _ctx

comptime ES_LIMBS = 16
comptime ES_MAX_ROWS = 1 << 30
comptime ES_TPB = 64
comptime ES_MAX_BLOCKS = 256
comptime ES_SHARED_BYTES = ES_TPB * ES_LIMBS * 8
comptime Limbs = SIMD[DType.int64, ES_LIMBS]

#: flag slots: a NaN, +inf, -inf, a negative finite entry, a positive entry
comptime ES_NAN = 0
comptime ES_PINF = 1
comptime ES_NINF = 2
comptime ES_NEG = 3
comptime ES_POS = 4
comptime ES_FLAGS = 8


@always_inline
def es_add(mut acc: Limbs, bits: UInt32):
    """acc += the float32 `bits` as an integer count of 2^-149 (finite only:
    the caller screens exponent 0xFF)."""
    var e = Int((bits >> 23) & 0xFF)
    var m = UInt64(bits & 0x7FFFFF)
    var shift = 0
    if e != 0:
        m |= UInt64(1) << 23
        shift = e - 1
    if m == 0:
        return
    var v = m << UInt64(shift % 32)
    var k = shift // 32
    var lo = Int64(v & 0xFFFFFFFF)
    var hi = Int64(v >> 32)
    if (bits >> 31) != 0:
        lo = -lo
        hi = -hi
    acc[k] = acc[k] + lo
    acc[k + 1] = acc[k + 1] + hi


@always_inline
def _clz32(x: UInt64) -> Int:
    var n = 0
    var t = x
    while (t & UInt64(0x80000000)) == 0 and n < 32:
        t <<= 1
        n += 1
    return n


def es_round(acc: Limbs) -> UInt64:
    """The integer sum(acc[i] << 32 i) times 2^-149, rounded once to binary64
    nearest/even, as a word; a zero sum is +0.0 (`_scaled_integer`)."""
    var l = acc
    for i in range(ES_LIMBS - 1):
        var v = l[i]
        var lo = v & Int64(0xFFFFFFFF)
        l[i] = lo
        l[i + 1] = l[i + 1] + ((v - lo) >> 32)
    var sign = UInt64(0)
    if l[ES_LIMBS - 1] < 0:
        sign = UInt64(1) << 63
        var carry = Int64(1)
        for i in range(ES_LIMBS):
            var x = ((~l[i]) & Int64(0xFFFFFFFF)) + carry
            l[i] = x & Int64(0xFFFFFFFF)
            carry = x >> 32
    var t = -1
    for i in range(ES_LIMBS):
        if l[i] != 0:
            t = i
    if t < 0:
        return UInt64(0)
    var bitlen = 32 * t + (32 - _clz32(UInt64(l[t])))
    var mant = UInt64(0)
    var top = 0
    if bitlen <= 53:
        mant = UInt64(l[0]) | (UInt64(l[1]) << 32)
        top = bitlen - 1 - 149
        mant = mant << UInt64(53 - bitlen)
    else:
        var shift = bitlen - 53
        for b in range(bitlen - 1, shift - 1, -1):
            mant = (mant << 1) | ((UInt64(l[b // 32]) >> UInt64(b % 32)) & 1)
        var half = (UInt64(l[(shift - 1) // 32]) >> UInt64((shift - 1) % 32)) & 1
        var sticky = False
        var lowbits = shift - 1
        for i in range(lowbits // 32):
            if l[i] != 0:
                sticky = True
        var rem = lowbits % 32
        if rem > 0 and (UInt64(l[lowbits // 32]) & ((UInt64(1) << UInt64(rem)) - 1)) != 0:
            sticky = True
        if half == 1 and (sticky or (mant & 1) == 1):
            mant += 1
        top = bitlen - 1 - 149
        if mant == (UInt64(1) << 53):
            mant >>= 1
            top += 1
    return sign | (UInt64(top + 1023) << 52) | (mant & UInt64(0x000FFFFFFFFFFFFF))


@always_inline
def es_screen(bits: UInt32, flags: MutPointer[Int32, MutAnyOrigin]) -> Bool:
    """Marks the flags `bits` raises; True when it is finite (to be added)."""
    var e = (bits >> 23) & 0xFF
    var neg = (bits >> 31) != 0
    if e == 0xFF:
        if (bits & 0x7FFFFF) != 0:
            _ = Atomic.max(flags.unsafe_offset(ES_NAN), Int32(1))
        elif neg:
            _ = Atomic.max(flags.unsafe_offset(ES_NINF), Int32(1))
        else:
            _ = Atomic.max(flags.unsafe_offset(ES_PINF), Int32(1))
        return False
    if (bits & 0x7FFFFFFF) != 0:
        if neg:
            _ = Atomic.max(flags.unsafe_offset(ES_NEG), Int32(1))
        else:
            _ = Atomic.max(flags.unsafe_offset(ES_POS), Int32(1))
    return True


def es_partial_kernel(
    x: MutPointer[UInt32, MutAnyOrigin], n: Int64,
    partials: MutPointer[Int64, MutAnyOrigin], flags: MutPointer[Int32, MutAnyOrigin],
):
    """Block b's exact limb sum of its grid-stride rows into
    partials[b * ES_LIMBS:], the flags by atomic max (idempotent)."""
    var tid = Int(thread_idx.x)
    var acc = Limbs(0)
    var r = Int(block_idx.x) * ES_TPB + tid
    var stride = Int(grid_dim.x) * ES_TPB
    while r < Int(n):
        var bits = x.unsafe_load(r)
        if es_screen(bits, flags):
            es_add(acc, bits)
        r += stride
    var sh = stack_allocation[ES_TPB * ES_LIMBS, Int64, address_space=AddressSpace.SHARED]()
    for l in range(ES_LIMBS):
        sh[tid * ES_LIMBS + l] = acc[l]
    barrier()
    var step = ES_TPB // 2
    while step > 0:
        if tid < step:
            for l in range(ES_LIMBS):
                sh[tid * ES_LIMBS + l] = sh[tid * ES_LIMBS + l] + sh[(tid + step) * ES_LIMBS + l]
        barrier()
        step //= 2
    if tid < ES_LIMBS:
        partials.unsafe_store(Int(block_idx.x) * ES_LIMBS + tid, sh[tid])


def es_final_kernel(
    partials: MutPointer[Int64, MutAnyOrigin], n_blocks: Int64, total: MutPointer[UInt64, MutAnyOrigin],
):
    """One block: the block partials added exactly, rounded once (thread 0)."""
    var tid = Int(thread_idx.x)
    var acc = Limbs(0)
    var b = tid
    while b < Int(n_blocks):
        for l in range(ES_LIMBS):
            acc[l] = acc[l] + partials.unsafe_load(b * ES_LIMBS + l)
        b += ES_TPB
    var sh = stack_allocation[ES_TPB * ES_LIMBS, Int64, address_space=AddressSpace.SHARED]()
    for l in range(ES_LIMBS):
        sh[tid * ES_LIMBS + l] = acc[l]
    barrier()
    var step = ES_TPB // 2
    while step > 0:
        if tid < step:
            for l in range(ES_LIMBS):
                sh[tid * ES_LIMBS + l] = sh[tid * ES_LIMBS + l] + sh[(tid + step) * ES_LIMBS + l]
        barrier()
        step //= 2
    if tid == 0:
        var tot = Limbs(0)
        for l in range(ES_LIMBS):
            tot[l] = sh[l]
        total.unsafe_store(0, es_round(tot))


def es_divide_kernel(
    x: MutPointer[UInt32, MutAnyOrigin], n: Int64, total: MutPointer[UInt64, MutAnyOrigin],
    res: MutPointer[UInt64, MutAnyOrigin],
):
    """res[i] = float64(x[i]) / total, correctly rounded (`sf64_div`)."""
    var t = total.unsafe_load(0)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(n):
        res.unsafe_store(r, sf64_div(widen_f32_word(x.unsafe_load(r)), t))
        r += stride


def _es_blocks(n: Int) -> Int:
    return max(1, min((n + ES_TPB - 1) // ES_TPB, ES_MAX_BLOCKS))


def exact_sum_device(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
    flags_out: MutPointer[Int32, MutUntrackedOrigin], want_div: Bool,
) raises:
    """The rounded exact sum word into res[0] and the flags into
    flags_out[0:ES_FLAGS - 1]; `want_div`: `out` holds n + 1 words and
    res[1:] = x / sum, written only when no entry is NaN, infinite or
    negative and one is positive (the normalized sample weights)."""
    comptime assert ES_SHARED_BYTES <= 16384, "exact-sum shared page over the 16 KB budget"
    var ctx = _ctx()
    var nb = _es_blocks(n)
    var d_x = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d_x, src_ptr=x.bitcast[UInt32]())
    var d_part = ctx.enqueue_create_buffer[DType.int64](nb * ES_LIMBS)
    var d_flags = ctx.enqueue_create_buffer[DType.int32](ES_FLAGS)
    d_flags.enqueue_fill(Int32(0))
    var d_tot = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[es_partial_kernel](
        d_x.unsafe_ptr(), Int64(n), d_part.unsafe_ptr(), d_flags.unsafe_ptr(),
        grid_dim=nb, block_dim=ES_TPB,
    )
    ctx.enqueue_function[es_final_kernel](
        d_part.unsafe_ptr(), Int64(nb), d_tot.unsafe_ptr(),
        grid_dim=1, block_dim=ES_TPB,
    )
    var h_flags = ctx.enqueue_create_host_buffer[DType.int32](ES_FLAGS)
    ctx.enqueue_copy(dst_buf=h_flags, src_buf=d_flags)
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_tot)
    ctx.synchronize()
    for i in range(ES_FLAGS - 1):
        flags_out[unsafe_offset=i] = h_flags.unsafe_ptr().unsafe_load(i)
    var ok = (Int(flags_out[unsafe_offset=ES_NAN]) | Int(flags_out[unsafe_offset=ES_PINF])
              | Int(flags_out[unsafe_offset=ES_NINF]) | Int(flags_out[unsafe_offset=ES_NEG])) == 0
    if want_div and ok and Int(flags_out[unsafe_offset=ES_POS]) != 0 and n > 0:
        var d_out = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_function[es_divide_kernel](
            d_x.unsafe_ptr(), Int64(n), d_tot.unsafe_ptr(), d_out.unsafe_ptr(),
            grid_dim=max(1, min((n + 255) // 256, 65535)), block_dim=256,
        )
        ctx.enqueue_copy(dst_ptr=(res + 1).bitcast[UInt64](), src_buf=d_out)
        ctx.synchronize()
        _ = d_out^
    _ = d_x^
    _ = d_part^
    _ = d_flags^
    _ = d_tot^
    _ = h_flags^


def exact_sum_host(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
    flags_out: MutPointer[Int32, MutUntrackedOrigin], want_div: Bool,
):
    """`exact_sum_device` on the host column: the same integer, the same
    rounding, hardware binary64 division (correctly rounded, as `sf64_div`)."""
    var fl = List[Int32](length=ES_FLAGS, fill=0)
    var acc = Limbs(0)
    for i in range(n):
        var bits = bitcast[DType.uint32](x[unsafe_offset=i])
        var e = (bits >> 23) & 0xFF
        var neg = (bits >> 31) != 0
        if e == 0xFF:
            if (bits & 0x7FFFFF) != 0:
                fl[ES_NAN] = 1
            elif neg:
                fl[ES_NINF] = 1
            else:
                fl[ES_PINF] = 1
            continue
        if (bits & 0x7FFFFFFF) != 0:
            if neg:
                fl[ES_NEG] = 1
            else:
                fl[ES_POS] = 1
        es_add(acc, bits)
    for i in range(ES_FLAGS - 1):
        flags_out[unsafe_offset=i] = fl[i]
    var w = es_round(acc)
    res.bitcast[UInt64]()[unsafe_offset=0] = w
    var ok = (Int(fl[ES_NAN]) | Int(fl[ES_PINF]) | Int(fl[ES_NINF]) | Int(fl[ES_NEG])) == 0
    if want_div and ok and Int(fl[ES_POS]) != 0:
        var t = bitcast[DType.float64](w)
        for i in range(n):
            res[unsafe_offset=i + 1] = Float64(x[unsafe_offset=i]) / t
