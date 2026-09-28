# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device form of the `mi_cd` stage (Ross's estimator, mutual_info_classif
of a continuous feature), lane prep-apple 2026-09-28.

`mi_cd_unit` (x_prep/mutual_info.mojo) scans every row for every row: n^2
distance pairs per feature (4.8 s of 4.9 s in mutual_info_classif at 100k x
16 on the M3 Ultra). x_prep/host/mutual_info.mojo (lane prep-cpu) showed the
unit writes digamma of COUNTS that depend only on the order statistics of
the distance pairs, and finds the same k-th pair and the same counts from
value-sorted columns; its argument is in that file. This is the same search
on the device:

  1. every column's points sorted by (key(x), index), the host's `_order`
     (a bitonic sort of 64-bit words, x_prep/dsort.mojo's scheme);
  2. one thread per column splits the sorted order by class, stably
     (`byc`, `pos`, `start`), and lists the points of classes with more than
     one member (`all`), as `_cd_column` does; a column with a non-finite
     value or secondary word, or a negative label, is flagged;
  3. one thread per (point, column) walks outward in its class and counts
     within the radius by binary search, `_cd_column`'s `point`, with the
     units' own distance functions; a flagged column runs `mi_cd_unit`.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st, key, run_block
from x_prep.prims import sub
from x_prep.mutual_info import MAX_K, digammaf, _sec, _dsec, _less, _within, mi_cd_unit

comptime WP = MutPointer[UInt64, MutAnyOrigin]
comptime UP = MutPointer[UInt32, MutAnyOrigin]
comptime MTILE = 1024
comptime MTG = MTILE // 2
comptime PAD64 = UInt64(0xFFFFFFFFFFFFFFFF)
comptime BIG = Float32(3.4028235e38)
#: scratch words per column (UInt32 each; the sorted keys are apart):
#: byc_x, byc_s, pos, all_x, all_s (n each), start (n + 1), meta (4)
comptime _M_FLAG = 0
comptime _M_NA = 1
comptime _M_NLAB = 2


@always_inline
def _fin(x: Float32) -> Bool:
    return x == x and abs(x) <= BIG


def mi_big_n(n: Int) -> Int:
    var big_n = MTILE
    while big_n < n:
        big_n *= 2
    return big_n


def mi_scratch_words(n: Int, d: Int) -> Int:
    """UInt32 words of per-column scratch for X[n, d] (the keys are apart)."""
    return d * (6 * n + 1 + 4)


def _load_kernel(f: FP, w: WP, Z: Int32, n: Int32, d: Int32, big_n: Int32, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var bn = Int(big_n)
    var c = t // bn
    var j = t - c * bn
    if j < Int(n):
        var x = ld(f, Int(Z) + j * Int(d) + c)
        w[t] = (UInt64(key(x)) << UInt64(32)) | UInt64(j)
    else:
        w[t] = PAD64


def _global_kernel(w: WP, big_n: Int32, k: Int32, j: Int32, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var half = Int(big_n) // 2
    var c = t // half
    var r = t - c * half
    var jj = Int(j)
    var i = 2 * jj * (r // jj) + (r % jj)
    var base = c * Int(big_n)
    var a = w[base + i]
    var b = w[base + i + jj]
    if (a <= b) != ((i & Int(k)) == 0):
        w[base + i] = b
        w[base + i + jj] = a


def _tile_kernel(w: WP, big_n: Int32, k_lo: Int32, k_hi: Int32):
    var s = stack_allocation[MTILE, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var g0 = Int(block_idx.x) * MTILE
    var row0 = g0 % Int(big_n)
    s[tid] = w[g0 + tid]
    s[tid + MTG] = w[g0 + tid + MTG]
    barrier()
    var k = Int(k_lo)
    while k <= Int(k_hi):
        var jj = min(k // 2, MTG)
        while jj >= 1:
            var i = 2 * jj * (tid // jj) + (tid % jj)
            var a = s[i]
            var b = s[i + jj]
            if (a <= b) != (((row0 + i) & k) == 0):
                s[i] = b
                s[i + jj] = a
            barrier()
            jj //= 2
        k *= 2
    w[g0 + tid] = s[tid]
    w[g0 + tid + MTG] = s[tid + MTG]


#: labels whose fill counters `_prep_kernel` keeps in registers (more: in the scratch)
comptime MLAB = 64
#: rows `_prep_kernel` loads before it uses any (lane prep-apple2)
comptime MRUN = 16


def _prep_kernel(f: FP, w: WP, u: UP, q: IP, big_n: Int32):
    """One thread per column c: `_cd_column`'s split of the sorted order.
    Lane prep-apple2: MRUN rows' words are loaded before any is used (the
    gathers through the sorted order were one memory latency each), the fill
    counters of up to MLAB labels live in registers, and the two passes that
    list the points of classes with more than one member are one. The same
    words land in the same slots."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    if c >= d:
        return
    var zb = p(q, 0) + c
    var sb = p(q, 7) + c if p(q, 7) > 0 else 0
    var lb = p(q, 3)
    var cb = p(q, 4)
    var U = c * (6 * n + 5)
    var bx = U
    var bs = U + n
    var pos = U + 2 * n
    var ax = U + 3 * n
    var as_ = U + 4 * n
    var start = U + 5 * n
    var meta = U + 6 * n + 1
    var fu = u
    var flag = 0
    var nlab = 0
    var full = n - n % MRUN
    for j0 in range(0, full, MRUN):
        var bxv = run_block[MRUN](f, zb + j0 * d, d)
        var blv = run_block[MRUN](f, lb + j0, 1)
        var bsv = run_block[MRUN](f, sb - 1 + j0 * d, d) if sb > 0 else SIMD[DType.float32, MRUN](0)
        comptime for v in range(MRUN):
            var l = Int(ftz(blv[v]))
            if not _fin(ftz(bxv[v])) or not _fin(ftz(bsv[v])) or l < 0:
                flag = 1
            elif flag == 0 and l + 1 > nlab:
                nlab = l + 1
        if flag != 0:
            break
    if flag == 0:
        for j in range(full, n):
            var x = ld(f, zb + j * d)
            var s = _sec(f, sb, j * d)
            var l = Int(ld(f, lb + j))
            if not _fin(x) or not _fin(s) or l < 0:
                flag = 1
                break
            if l + 1 > nlab:
                nlab = l + 1
    fu[meta + _M_FLAG] = UInt32(flag)
    fu[meta + _M_NLAB] = UInt32(nlab)
    if flag != 0:
        return
    var wb = c * Int(big_n)
    var fillb = as_
    if nlab <= MLAB:
        var cntr = InlineArray[UInt32, MLAB + 1](fill=UInt32(0))
        for j0 in range(0, full, MRUN):
            var blv = run_block[MRUN](f, lb + j0, 1)
            comptime for v in range(MRUN):
                cntr[Int(ftz(blv[v])) + 1] += 1
        for j in range(full, n):
            var l = Int(ld(f, lb + j))
            cntr[l + 1] += 1
        for l in range(nlab):
            cntr[l + 1] += cntr[l]
        for l in range(nlab + 1):
            fu[start + l] = cntr[l]
        # cntr[l] is now label l's fill counter
        var multi = InlineArray[Bool, MLAB](fill=False)
        for l in range(nlab):
            multi[l] = Int(ld(f, cb + l)) > 1
        var na = 0
        var r0 = 0
        while r0 < n:
            var m = min(MRUN, n - r0)
            var jj = InlineArray[Int, MRUN](fill=0)
            for v in range(m):
                jj[v] = Int(w[wb + r0 + v] & UInt64(0xFFFFFFFF))
            var ll = InlineArray[Int, MRUN](fill=0)
            var xx = InlineArray[UInt32, MRUN](fill=UInt32(0))
            var ss = InlineArray[UInt32, MRUN](fill=UInt32(0))
            for v in range(m):
                ll[v] = Int(ld(f, lb + jj[v]))
                xx[v] = bitcast[DType.uint32](ld(f, zb + jj[v] * d))
                ss[v] = bitcast[DType.uint32](_sec(f, sb, jj[v] * d))
            for v in range(m):
                var l = ll[v]
                var at = Int(cntr[l])
                cntr[l] = UInt32(at + 1)
                fu[bx + at] = xx[v]
                fu[bs + at] = ss[v]
                fu[pos + jj[v]] = UInt32(at)
                if multi[l]:
                    fu[ax + na] = xx[v]
                    fu[as_ + na] = ss[v]
                    na += 1
            r0 += m
        fu[meta + _M_NA] = UInt32(na)
        return
    for l in range(nlab + 1):
        fu[start + l] = 0
    for j in range(n):
        var l = Int(ld(f, lb + j))
        fu[start + l + 1] = fu[start + l + 1] + 1
    for l in range(nlab):
        fu[start + l + 1] = fu[start + l + 1] + fu[start + l]
    var na = 0
    # the fill counters (one per label) live in the all_s slots until the
    # class split is done; all_s is written last
    for l in range(nlab):
        fu[fillb + l] = fu[start + l]
    for r in range(n):
        var j = Int(w[wb + r] & UInt64(0xFFFFFFFF))
        var l = Int(ld(f, lb + j))
        var at = Int(fu[fillb + l])
        fu[fillb + l] = UInt32(at + 1)
        fu[bx + at] = bitcast[DType.uint32](ld(f, zb + j * d))
        fu[bs + at] = bitcast[DType.uint32](_sec(f, sb, j * d))
        fu[pos + j] = UInt32(at)
    for r in range(n):
        var j = Int(w[wb + r] & UInt64(0xFFFFFFFF))
        var l = Int(ld(f, lb + j))
        if Int(ld(f, cb + l)) > 1:
            fu[ax + na] = bitcast[DType.uint32](ld(f, zb + j * d))
            na += 1
    na = 0
    for r in range(n):
        var j = Int(w[wb + r] & UInt64(0xFFFFFFFF))
        var l = Int(ld(f, lb + j))
        if Int(ld(f, cb + l)) > 1:
            fu[as_ + na] = bitcast[DType.uint32](_sec(f, sb, j * d))
            na += 1
    fu[meta + _M_NA] = UInt32(na)


@always_inline
def _fx(u: UP, i: Int) -> Float32:
    return bitcast[DType.float32](u[i])


@always_inline
def _dist(xj: Float32, xi: Float32) -> Float32:
    return abs(sub(xj, xi))


def _count_within(u: UP, axb: Int, asb: Int, hi: Int, xi: Float32, si: Float32, rp: Float32,
                  rs: Float32) -> Int:
    """x_prep/host/mutual_info.mojo `_count_within` over all[0:hi]."""
    var l = 0
    var h = hi
    while l < h:
        var m = (l + h) // 2
        if _fx(u, axb + m) < xi:
            l = m + 1
        else:
            h = m
    var mid = l
    var cnt = 0
    l = mid
    h = hi
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, axb + m), xi) < rp:
            l = m + 1
        else:
            h = m
    cnt += l - mid
    var r = l
    while r < hi and _dist(_fx(u, axb + r), xi) == rp:
        if _within(rp, _dsec(xi, _fx(u, axb + r), si, _fx(u, asb + r)), rp, rs):
            cnt += 1
        r += 1
    l = 0
    h = mid
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, axb + m), xi) < rp:
            h = m
        else:
            l = m + 1
    cnt += mid - l
    var qq = l - 1
    while qq >= 0 and _dist(_fx(u, axb + qq), xi) == rp:
        if _within(rp, _dsec(xi, _fx(u, axb + qq), si, _fx(u, asb + qq)), rp, rs):
            cnt += 1
        qq -= 1
    return cnt


def _point_kernel(f: FP, u: UP, q: IP, total: Int32):
    """t = i*d + c: `_cd_column`'s `point(i)` for column c, or the unit when
    the column is flagged."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var U = c * (6 * n + 5)
    var meta = U + 6 * n + 1
    if u[meta + _M_FLAG] != UInt32(0):
        mi_cd_unit(t, f, q)
        return
    var zb = p(q, 0) + c
    var sb = p(q, 7) + c if p(q, 7) > 0 else 0
    var k = p(q, 5)
    var li = Int(ld(f, p(q, 3) + i))
    var cnt = Int(ld(f, p(q, 4) + li))
    var out = p(q, 6) + t
    if cnt <= 1:
        st(f, out, Float32(0))
        return
    var kl = k if k < cnt - 1 else cnt - 1
    var xi = ld(f, zb + i * d)
    var si = _sec(f, sb, i * d)
    var bx = U
    var bsb = U + n
    var start = U + 5 * n
    var b = Int(u[start + li])
    var e = Int(u[start + li + 1])
    var bp = InlineArray[Float32, MAX_K](fill=BIG)
    var bs = InlineArray[Float32, MAX_K](fill=BIG)
    var pi = Int(u[U + 2 * n + i])
    var L = pi - 1
    var R = pi + 1
    var taken = 0
    var D = Float32(0)
    while True:
        var hasl = L >= b
        var hasr = R < e
        if not hasl and not hasr:
            break
        var dl = _dist(_fx(u, bx + L), xi) if hasl else BIG
        var dr = _dist(_fx(u, bx + R), xi) if hasr else BIG
        var take_left = hasl and (not hasr or dl <= dr)
        var dp = dl if take_left else dr
        if taken >= kl and dp > D:
            break
        var dsec: Float32
        if take_left:
            dsec = _dsec(xi, _fx(u, bx + L), si, _fx(u, bsb + L))
            L -= 1
        else:
            dsec = _dsec(xi, _fx(u, bx + R), si, _fx(u, bsb + R))
            R += 1
        if _less(dp, dsec, bp[kl - 1], bs[kl - 1]):
            var m = kl - 1
            while m > 0 and _less(dp, dsec, bp[m - 1], bs[m - 1]):
                bp[m] = bp[m - 1]
                bs[m] = bs[m - 1]
                m -= 1
            bp[m] = dp
            bs[m] = dsec
        taken += 1
        D = dp
    var rp = bp[kl - 1]
    var rs = bs[kl - 1]
    var na = Int(u[meta + _M_NA])
    var mall = _count_within(u, U + 3 * n, U + 4 * n, na, xi, si, rp, rs)
    st(f, out, sub(sub(digammaf(Float32(kl)), digammaf(Float32(cnt))), digammaf(Float32(mall))))


def _blocks(total: Int, bs: Int) -> Int:
    return (total + bs - 1) // bs


def mi_cd_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint64],
                 mut du: DeviceBuffer[DType.uint32], mut dq: DeviceBuffer[DType.int32], qoff: Int, total: Int,
                 n: Int, d: Int, Z: Int) raises:
    """Enqueue the whole `mi_cd` stage (its params at dq[qoff:])."""
    if n <= 0 or d <= 0:
        return
    var big_n = mi_big_n(n)
    var tot = d * big_n
    if tot > 2 ** 31 - 1:
        raise Error("x_prep: mi_cd too large for the device search")
    var f = df.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var u = du.unsafe_ptr()
    var qp = dq.unsafe_ptr() + qoff
    comptime BS = 256
    ctx.enqueue_function[_load_kernel](
        f, w, Int32(Z), Int32(n), Int32(d), Int32(big_n), Int32(tot), grid_dim=_blocks(tot, BS), block_dim=BS,
    )
    var tiles = tot // MTILE
    ctx.enqueue_function[_tile_kernel](w, Int32(big_n), Int32(2), Int32(MTILE), grid_dim=tiles, block_dim=MTG)
    var pairs = tot // 2
    var k = 2 * MTILE
    while k <= big_n:
        var j = k // 2
        while j >= MTILE:
            ctx.enqueue_function[_global_kernel](
                w, Int32(big_n), Int32(k), Int32(j), Int32(pairs), grid_dim=_blocks(pairs, BS), block_dim=BS,
            )
            j //= 2
        ctx.enqueue_function[_tile_kernel](w, Int32(big_n), Int32(k), Int32(k), grid_dim=tiles, block_dim=MTG)
        k *= 2
    ctx.enqueue_function[_prep_kernel](f, w, u, qp, Int32(big_n), grid_dim=_blocks(d, 32), block_dim=32)
    ctx.enqueue_function[_point_kernel](f, u, qp, Int32(total), grid_dim=_blocks(total, BS), block_dim=BS)
