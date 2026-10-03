# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bagging's out-of-bag bookkeeping in the binding (lane
apple-fast-py2mojo-trees, 2026-10-03): the per-row member counts, the
classifier's hit count and the regressor's oob prediction and R^2 sums, on
the device for every GPU build and serially on the host column.

THE EXACT BINARY64 SUM. `_portable_math.fsum` rounds the EXACT sum of its
terms once. Every finite binary64 is an integer multiple of 2^-1074; its
integer is added into 32-bit places held in Int64 limbs (`E64_LIMBS`, limb i
weighs 2^(32 i)), so no add rounds and neither the order nor the reduction
shape can move the integer. On the device each thread owns one limb row of
a partial buffer (no shared page, no atomics), one thread per limb then adds
the rows, and one thread rounds (`e64_round`: `_scaled_integer(total,
-1074)`, nearest/even). A limb takes at most 2^30 terms below 2^32
(`E64_MAX_ROWS`). The per-row terms are formed with the soft binary64 ops
(`sf64_sub`, `sf64_mul`, `sf64_div`: correctly rounded, the Apple GPU has no
float64) on the device and the hardware ops on the host, the same IEEE
results as the Python `(v - mean) * (v - mean)` and `v / max(c, 1)`.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from checks.soft_f64 import sf64_div, sf64_from_int, sf64_mul, sf64_sub
from xtrees.glue_device import widen_f32_word
from xtrees.ops_device import OPS_TPB, _blocks, _ctx

comptime E64_LIMBS = 72
comptime E64_MAX_ROWS = 1 << 30
comptime E64_THREADS = 64 * 64
#: flags: [0] a non-finite term, [1] a rounded sum overflowed binary64
comptime E64_FLAGS = 4


@always_inline
def e64_add[o: MutOrigin, fo: MutOrigin, //](limbs: MutPointer[Int64, o], w: UInt64, flags: MutPointer[Int32, fo]):
    """limbs += the binary64 word w as an integer count of 2^-1074 (a NaN or
    an infinity sets flags[0] and adds nothing)."""
    var e = Int((w >> 52) & UInt64(0x7FF))
    if e == 0x7FF:
        flags[unsafe_offset=0] = 1
        return
    var m = w & UInt64(0x000FFFFFFFFFFFFF)
    var shift = 0
    if e != 0:
        m |= UInt64(1) << 52
        shift = e - 1
    if m == 0:
        return
    var s = shift % 32
    var k = shift // 32
    var lo = m << UInt64(s)
    var hi = (m >> UInt64(64 - s)) if s > 0 else UInt64(0)
    var p0 = Int64(lo & UInt64(0xFFFFFFFF))
    var p1 = Int64(lo >> 32)
    var p2 = Int64(hi)
    if (w >> 63) != 0:
        p0 = -p0
        p1 = -p1
        p2 = -p2
    limbs[unsafe_offset=k] = limbs[unsafe_offset=k] + p0
    limbs[unsafe_offset=k + 1] = limbs[unsafe_offset=k + 1] + p1
    limbs[unsafe_offset=k + 2] = limbs[unsafe_offset=k + 2] + p2


def e64_round[o: MutOrigin, fo: MutOrigin, //](l: MutPointer[Int64, o], flags: MutPointer[Int32, fo]) -> UInt64:
    """The integer sum(l[i] << 32 i) times 2^-1074 rounded once to binary64,
    nearest/even (`_scaled_integer(total, -1074)`); normalizes `l` in place.
    Overflow sets flags[1] (the Python OverflowError) and returns +inf."""
    for i in range(E64_LIMBS - 1):
        var v = l[unsafe_offset=i]
        var lo = v & Int64(0xFFFFFFFF)
        l[unsafe_offset=i] = lo
        l[unsafe_offset=i + 1] = l[unsafe_offset=i + 1] + ((v - lo) >> 32)
    var sign = UInt64(0)
    if l[unsafe_offset=E64_LIMBS - 1] < 0:
        sign = UInt64(1) << 63
        var carry = Int64(1)
        for i in range(E64_LIMBS):
            var x = ((~l[unsafe_offset=i]) & Int64(0xFFFFFFFF)) + carry
            l[unsafe_offset=i] = x & Int64(0xFFFFFFFF)
            carry = x >> 32
    var t = -1
    for i in range(E64_LIMBS):
        if l[unsafe_offset=i] != 0:
            t = i
    if t < 0:
        return UInt64(0)
    var top_limb = UInt64(l[unsafe_offset=t])
    var width = 0
    while top_limb != 0:
        top_limb >>= 1
        width += 1
    var bitlen = 32 * t + width
    if bitlen <= 53:
        # below 2^53 units of 2^-1074 the integer IS the word (subnormal or
        # the smallest normal binade)
        return sign | (UInt64(l[unsafe_offset=0]) | (UInt64(l[unsafe_offset=1]) << 32))
    var shift = bitlen - 53
    var mant = UInt64(0)
    for b in range(bitlen - 1, shift - 1, -1):
        mant = (mant << 1) | ((UInt64(l[unsafe_offset=b // 32]) >> UInt64(b % 32)) & 1)
    var half = (UInt64(l[unsafe_offset=(shift - 1) // 32]) >> UInt64((shift - 1) % 32)) & 1
    var sticky = False
    var lowbits = shift - 1
    for i in range(lowbits // 32):
        if l[unsafe_offset=i] != 0:
            sticky = True
    var rem = lowbits % 32
    if rem > 0 and (UInt64(l[unsafe_offset=lowbits // 32]) & ((UInt64(1) << UInt64(rem)) - 1)) != 0:
        sticky = True
    if half == 1 and (sticky or (mant & 1) == 1):
        mant += 1
    var top = bitlen - 1 - 1074
    if mant == (UInt64(1) << 53):
        mant >>= 1
        top += 1
    if top > 1023:
        flags[unsafe_offset=1] = 1
        return sign | UInt64(0x7FF0000000000000)
    return sign | (UInt64(top + 1023) << 52) | (mant & UInt64(0x000FFFFFFFFFFFFF))


# ------------------------------------------------------------ counts --


def count_rows_kernel(counts: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], m: Int64,
                      n: Int64, bad: MutPointer[Int32, MutAnyOrigin]):
    """counts[rows[r]] += 1 (integer atomics: order-free); a row outside
    [0, n) sets bad[0]."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r))
        if i < 0 or i >= Int(n):
            bad.unsafe_store(0, Int32(1))
        else:
            _ = Atomic.fetch_add(counts.unsafe_offset(i), Int32(1))
        r += stride


def count_rows_device(counts: MutPointer[Int32, MutUntrackedOrigin], n: Int,
                      rows: MutPointer[Int32, MutUntrackedOrigin], m: Int) raises:
    """counts (int32, n, in and out) += one per listed row."""
    if m <= 0 or n <= 0:
        return
    var ctx = _ctx()
    var d_c = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_c, src_ptr=counts)
    var d_r = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_copy(dst_buf=d_r, src_ptr=rows)
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    d_bad.enqueue_fill(Int32(0))
    ctx.enqueue_function[count_rows_kernel](
        d_c.unsafe_ptr(), d_r.unsafe_ptr(), Int64(m), Int64(n), d_bad.unsafe_ptr(),
        grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.synchronize()
    if h_bad.unsafe_ptr().unsafe_load(0) != Int32(0):
        raise Error("x_trees count_rows: row out of range")
    ctx.enqueue_copy(dst_ptr=counts, src_buf=d_c)
    ctx.synchronize()
    _ = d_r^
    _ = d_bad^
    _ = h_bad^
    _ = d_c^


def count_rows_host(counts: MutPointer[Int32, MutUntrackedOrigin], n: Int,
                    rows: MutPointer[Int32, MutUntrackedOrigin], m: Int) raises:
    for r in range(m):
        if Int(rows[unsafe_offset=r]) < 0 or Int(rows[unsafe_offset=r]) >= n:
            raise Error("x_trees count_rows: row out of range")
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        counts[unsafe_offset=i] = counts[unsafe_offset=i] + 1


def count_equal_kernel(a: MutPointer[Int32, MutAnyOrigin], b: MutPointer[Int32, MutAnyOrigin], n: Int64,
                       total: MutPointer[Int32, MutAnyOrigin]):
    """total += #{r : a[r] == b[r]}: a per-thread count, one integer atomic
    per thread (order-free)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var c = Int32(0)
    while r < Int(n):
        if a.unsafe_load(r) == b.unsafe_load(r):
            c += 1
        r += stride
    if c != 0:
        _ = Atomic.fetch_add(total, c)


def count_equal_device(a: MutPointer[Int32, MutUntrackedOrigin], b: MutPointer[Int32, MutUntrackedOrigin],
                       n: Int) raises -> Int:
    if n <= 0:
        return 0
    var ctx = _ctx()
    var d_a = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_a, src_ptr=a)
    var d_b = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_b, src_ptr=b)
    var d_t = ctx.enqueue_create_buffer[DType.int32](1)
    d_t.enqueue_fill(Int32(0))
    ctx.enqueue_function[count_equal_kernel](
        d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int64(n), d_t.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var h_t = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_t, src_buf=d_t)
    ctx.synchronize()
    var out_count = Int(h_t.unsafe_ptr().unsafe_load(0))
    _ = d_a^
    _ = d_b^
    _ = d_t^
    _ = h_t^
    return out_count


def count_equal_host(a: MutPointer[Int32, MutUntrackedOrigin], b: MutPointer[Int32, MutUntrackedOrigin],
                     n: Int) -> Int:
    var c = 0
    for r in range(n):
        if a[unsafe_offset=r] == b[unsafe_offset=r]:
            c += 1
    return c


# ------------------------------------------------------- regressor R^2 --


@always_inline
def _thread() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def oob_pred_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], counts: MutPointer[Int32, MutAnyOrigin], y: MutPointer[UInt32, MutAnyOrigin],
    n: Int64, pred: MutPointer[UInt64, MutAnyOrigin], part: MutPointer[Int64, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """pred[i] = acc[i] / max(counts[i], 1); thread t's limb row of part
    gets the exact sum of its y rows (E64_THREADS threads, grid-stride)."""
    var t = _thread()
    var row = part + t * E64_LIMBS
    for k in range(E64_LIMBS):
        row[unsafe_offset=k] = 0
    var i = t
    while i < Int(n):
        var c = max(Int(counts.unsafe_load(i)), 1)
        pred.unsafe_store(i, sf64_div(acc.unsafe_load(i), sf64_from_int(c)))
        e64_add(row, widen_f32_word(y.unsafe_load(i)), flags)
        i += E64_THREADS


def oob_sq_kernel(
    y: MutPointer[UInt32, MutAnyOrigin], pred: MutPointer[UInt64, MutAnyOrigin], n: Int64,
    mean: MutPointer[UInt64, MutAnyOrigin], part_tot: MutPointer[Int64, MutAnyOrigin],
    part_res: MutPointer[Int64, MutAnyOrigin], flags: MutPointer[Int32, MutAnyOrigin],
):
    """Thread t's limb rows: the exact sums of (y - mean)^2 and (y - pred)^2
    over its rows, each term two correctly rounded binary64 operations."""
    var t = _thread()
    var rt = part_tot + t * E64_LIMBS
    var rr = part_res + t * E64_LIMBS
    for k in range(E64_LIMBS):
        rt[unsafe_offset=k] = 0
        rr[unsafe_offset=k] = 0
    var mu = mean.unsafe_load(0)
    var i = t
    while i < Int(n):
        var v = widen_f32_word(y.unsafe_load(i))
        var d = sf64_sub(v, mu)
        e64_add(rt, sf64_mul(d, d), flags)
        var r = sf64_sub(v, pred.unsafe_load(i))
        e64_add(rr, sf64_mul(r, r), flags)
        i += E64_THREADS


def limb_reduce_kernel(part: MutPointer[Int64, MutAnyOrigin], total: MutPointer[Int64, MutAnyOrigin]):
    """total[k] = sum over the E64_THREADS rows of limb k (one thread per limb)."""
    var k = _thread()
    if k < E64_LIMBS:
        var s = Int64(0)
        for t in range(E64_THREADS):
            s += part.unsafe_load(t * E64_LIMBS + k)
        total.unsafe_store(k, s)


def mean_kernel(total: MutPointer[Int64, MutAnyOrigin], count_word: UInt64, words: MutPointer[UInt64, MutAnyOrigin],
                flags: MutPointer[Int32, MutAnyOrigin]):
    """One thread, O(1) work (the limbs are E64_LIMBS words): words[0] =
    fsum(y), words[3] = fsum(y) / count (count_word: the row count as binary64)."""
    if _thread() == 0:
        var s = e64_round(total, flags)
        words.unsafe_store(0, s)
        words.unsafe_store(3, sf64_div(s, count_word))


def round_kernel(total: MutPointer[Int64, MutAnyOrigin], words: MutPointer[UInt64, MutAnyOrigin], slot: Int64,
                 flags: MutPointer[Int32, MutAnyOrigin]):
    if _thread() == 0:
        words.unsafe_store(Int(slot), e64_round(total, flags))


def oob_r2_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], counts: MutPointer[Int32, MutUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin], n: Int, pred: MutPointer[Float64, MutUntrackedOrigin],
    words: MutPointer[Float64, MutUntrackedOrigin], flags_out: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """pred = acc / max(counts, 1); words = [fsum(y), fsum((y - mean)^2),
    fsum((y - pred)^2), mean]; flags_out[0:2] (see E64_FLAGS)."""
    var ctx = _ctx()
    var d_acc = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_copy(dst_buf=d_acc, src_ptr=acc.bitcast[UInt64]())
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d_cnt, src_ptr=counts)
    var d_y = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=y.bitcast[UInt32]())
    var d_pred = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_p1 = ctx.enqueue_create_buffer[DType.int64](E64_THREADS * E64_LIMBS)
    var d_p2 = ctx.enqueue_create_buffer[DType.int64](E64_THREADS * E64_LIMBS)
    var d_t1 = ctx.enqueue_create_buffer[DType.int64](E64_LIMBS)
    var d_t2 = ctx.enqueue_create_buffer[DType.int64](E64_LIMBS)
    var d_words = ctx.enqueue_create_buffer[DType.uint64](4)
    var d_flags = ctx.enqueue_create_buffer[DType.int32](E64_FLAGS)
    d_flags.enqueue_fill(Int32(0))
    var grid = E64_THREADS // OPS_TPB
    ctx.enqueue_function[oob_pred_kernel](
        d_acc.unsafe_ptr(), d_cnt.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), d_pred.unsafe_ptr(),
        d_p1.unsafe_ptr(), d_flags.unsafe_ptr(), grid_dim=grid, block_dim=OPS_TPB,
    )
    ctx.enqueue_function[limb_reduce_kernel](d_p1.unsafe_ptr(), d_t1.unsafe_ptr(), grid_dim=1, block_dim=E64_LIMBS)
    ctx.enqueue_function[mean_kernel](d_t1.unsafe_ptr(), sf64_from_int(n), d_words.unsafe_ptr(), d_flags.unsafe_ptr(),
                                      grid_dim=1, block_dim=1)
    var d_mean = d_words.create_sub_buffer[DType.uint64](3, 1)
    ctx.enqueue_function[oob_sq_kernel](
        d_y.unsafe_ptr(), d_pred.unsafe_ptr(), Int64(n), d_mean.unsafe_ptr(), d_p1.unsafe_ptr(), d_p2.unsafe_ptr(),
        d_flags.unsafe_ptr(), grid_dim=grid, block_dim=OPS_TPB,
    )
    ctx.enqueue_function[limb_reduce_kernel](d_p1.unsafe_ptr(), d_t1.unsafe_ptr(), grid_dim=1, block_dim=E64_LIMBS)
    ctx.enqueue_function[limb_reduce_kernel](d_p2.unsafe_ptr(), d_t2.unsafe_ptr(), grid_dim=1, block_dim=E64_LIMBS)
    ctx.enqueue_function[round_kernel](d_t1.unsafe_ptr(), d_words.unsafe_ptr(), Int64(1), d_flags.unsafe_ptr(),
                                       grid_dim=1, block_dim=1)
    ctx.enqueue_function[round_kernel](d_t2.unsafe_ptr(), d_words.unsafe_ptr(), Int64(2), d_flags.unsafe_ptr(),
                                       grid_dim=1, block_dim=1)
    ctx.enqueue_copy(dst_ptr=pred.bitcast[UInt64](), src_buf=d_pred)
    ctx.enqueue_copy(dst_ptr=words.bitcast[UInt64](), src_buf=d_words)
    ctx.enqueue_copy(dst_ptr=flags_out, src_buf=d_flags)
    ctx.synchronize()
    _ = d_acc^
    _ = d_cnt^
    _ = d_y^
    _ = d_pred^
    _ = d_p1^
    _ = d_p2^
    _ = d_t1^
    _ = d_t2^
    _ = d_mean^
    _ = d_words^
    _ = d_flags^


def oob_r2_host(
    acc: MutPointer[Float64, MutUntrackedOrigin], counts: MutPointer[Int32, MutUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin], n: Int, pred: MutPointer[Float64, MutUntrackedOrigin],
    words: MutPointer[Float64, MutUntrackedOrigin], flags_out: MutPointer[Int32, MutUntrackedOrigin],
):
    """`oob_r2_device` on the host column: the same integers, the hardware
    binary64 operations (correctly rounded, as the soft ones)."""
    var l1 = List[Int64](length=E64_LIMBS, fill=0)
    var l2 = List[Int64](length=E64_LIMBS, fill=0)
    var fl = List[Int32](length=E64_FLAGS, fill=0)
    var p1 = l1.unsafe_ptr()
    var pf = fl.unsafe_ptr()
    for i in range(n):
        var c = max(Int(counts[unsafe_offset=i]), 1)
        pred[unsafe_offset=i] = acc[unsafe_offset=i] / Float64(c)
        e64_add(p1, bitcast[DType.uint64](Float64(y[unsafe_offset=i])), pf)
    var s = e64_round(p1, pf)
    var mean = bitcast[DType.float64](s) / Float64(n)
    for k in range(E64_LIMBS):
        l1[k] = 0
    p1 = l1.unsafe_ptr()
    var p2 = l2.unsafe_ptr()
    for i in range(n):
        var v = Float64(y[unsafe_offset=i])
        var d = v - mean
        e64_add(p1, bitcast[DType.uint64](d * d), pf)
        var r = v - pred[unsafe_offset=i]
        e64_add(p2, bitcast[DType.uint64](r * r), pf)
    words.bitcast[UInt64]()[unsafe_offset=0] = s
    words.bitcast[UInt64]()[unsafe_offset=1] = e64_round(p1, pf)
    words.bitcast[UInt64]()[unsafe_offset=2] = e64_round(p2, pf)
    words[unsafe_offset=3] = mean
    for k in range(E64_FLAGS):
        flags_out[unsafe_offset=k] = fl[k]
    _ = l1^
    _ = l2^
    _ = fl^
