# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple candidates for mutual information (lane af-mi, 2026-10-03;
docs/apple-fast/notes/mi.md, docs/apple-fast/ab/mi.md). FAST ON APPLE ONLY:
x_prep/device.mojo launches anything here under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and a
`-D MOJOLEARN_MI_*` define, so the IDENTICAL binding and the other vendors
never compile a launch of these kernels.

1. `mi_cc_device` (MOJOLEARN_MI_REG_SORTCOUNT, _REG_TIES, _REG_RANKMAJOR):
   the `mi_cc` stage (Kraskov, mutual_info_regression) as the sorted search
   the host already argues for in x_prep/host/mutual_info.mojo `_cc_column`,
   instead of `mi_cc_unit`'s two brute-force scans of every row by every
   (point, feature). Every unit writes digamma of COUNTS, and the counts are
   order statistics of the distance pairs: any search that finds the same
   k-th smallest pair and the same counts within it writes the same words.
   The distances are the unit's own functions (`ld`, `sub`, `abs`, `_sec`,
   `_dsec`, `_less`, `_within`).
   - SORTCOUNT: every column sorted by (key(x), j) (x_prep/dmi.mojo's load
     and bitonic kernels), y sorted once by (key(y), j); one thread per
     (point, column) walks outward in x with the host's stop rule (a side
     stops once |dx| exceeds the current k-th primary), then nx and ny by
     binary search from the point's own sorted position.
   - TIES: the columns sorted by (key(x), key(y), key(sx)) (128-bit compare:
     the sort word and its payload word). Inside an x-run the members are in
     y order, so the k nearest in y are an outward walk; inside an (x, y)
     stretch they are in sx order, so the k-th pair among equal (x, y) (a
     Chebyshev k-NN on the noise words) is an outward walk in sx that stops
     once (0, sxd) cannot enter the k best (every later pair has a larger
     sxd, and its secondary is at least sxd). The counts are dmi.mojo's run
     counts: nx over a second copy of the column in (key(x), key(sx)) order,
     ny over y in (key(y), key(sy)) order. A tied column (taxi codes,
     istella zeros, the 5-valued istella target) is otherwise O(run) per
     point in the plain walk.
   - RANKMAJOR: thread t = c * n + r handles the point at sorted rank r of
     column c (its index from the sort payload), so adjacent threads walk
     adjacent sorted positions of one column; SORTCOUNT and TIES map t =
     i * d + c as the unit does (adjacent threads: different columns).
   A column with a non-finite value or noise word, or a non-finite y, runs
   `mi_cc_unit` for its points (the host's rule).
2. `mi_colscale_fast_kernel`, `mi_reduce_fast_kernel` (MOJOLEARN_MI_FAST_FOLDS):
   the two one-thread-per-column folds over n rows as threadgroup folds
   (x_prep/fastred.mojo's form): the sum order changes (FAST), pairwise is
   never less accurate than row order.
3. `mi_cd_device_rank` (MOJOLEARN_MI_CLF_RANKMAJOR): dmi.mojo's `mi_cd`
   stage with the point kernel in the rank-major mapping of (1).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_prep.common import FP, IP, p, ld, st, key
from x_prep.prims import add, sub, div, sqrtf, zero_to_one
from x_prep.mutual_info import MAX_K, digammaf, _sec, _dsec, _less, _within, mi_cc_unit, mi_cd_unit
from x_prep.dmi import (
    WP, UP, MTILE, MTG, PAD64, BIG, mi_big_n, _blocks, _fin, _fx, _dist, _kx, _take, _take_run, _run_hi, _run_lo,
    _count_run, _count_within_ties, _load_kernel, _tile_kernel, _global_kernel, _prep_kernel, _M_FLAG, _M_NA,
)

#: threads of a column fold (mi_colscale, mi_reduce)
comptime TGF = 256
#: threads per block of the one-thread-per-unit launches
comptime CBS = 256


# ------------------------------------------------------------------ 128-bit bitonic sort
@always_inline
def _le2(a: UInt64, pa: UInt64, b: UInt64, pb: UInt64) -> Bool:
    """(a, pa) <= (b, pb): the sort word first, the payload word on a tie."""
    return a < b or (a == b and pa <= pb)


def _cc_load_xy_kernel(f: FP, w: WP, pl: WP, Z: Int32, n: Int32, d: Int32, big_n: Int32, total: Int32,
                       zs1: Int32, Y: Int32):
    """t = c * N + j: w[t] = (key(x), key(y)), pl[t] = (key(sx), j) for X[j, c]
    and its noise word sx (zs1 = offset + 1, 0 for none); PAD64 past n."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var bn = Int(big_n)
    var c = t // bn
    var j = t - c * bn
    if j < Int(n):
        var dd = Int(d)
        var x = ld(f, Int(Z) + j * dd + c)
        var y = ld(f, Int(Y) + j)
        var sb = Int(zs1) + c if Int(zs1) > 0 else 0
        var sx = _sec(f, sb, j * dd)
        w[t] = (UInt64(key(x)) << UInt64(32)) | UInt64(key(y))
        pl[t] = (UInt64(key(sx)) << UInt64(32)) | UInt64(j)
    else:
        w[t] = PAD64
        pl[t] = PAD64


def _cc_global_kernel(w: WP, pl: WP, big_n: Int32, k: Int32, j: Int32, total: Int32):
    """dmi.mojo `_global_kernel` comparing (w, pl) pairs."""
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
    var pa = pl[base + i]
    var pb = pl[base + i + jj]
    if _le2(a, pa, b, pb) != ((i & Int(k)) == 0):
        w[base + i] = b
        w[base + i + jj] = a
        pl[base + i] = pb
        pl[base + i + jj] = pa


def _cc_tile_kernel(w: WP, pl: WP, big_n: Int32, k_lo: Int32, k_hi: Int32):
    """dmi.mojo `_tile_kernel` comparing (w, pl) pairs."""
    var s = stack_allocation[MTILE, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var sp = stack_allocation[MTILE, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var g0 = Int(block_idx.x) * MTILE
    var row0 = g0 % Int(big_n)
    s[tid] = w[g0 + tid]
    s[tid + MTG] = w[g0 + tid + MTG]
    sp[tid] = pl[g0 + tid]
    sp[tid + MTG] = pl[g0 + tid + MTG]
    barrier()
    var k = Int(k_lo)
    while k <= Int(k_hi):
        var jj = min(k // 2, MTG)
        while jj >= 1:
            var i = 2 * jj * (tid // jj) + (tid % jj)
            var a = s[i]
            var b = s[i + jj]
            var pa = sp[i]
            var pb = sp[i + jj]
            if _le2(a, pa, b, pb) != (((row0 + i) & k) == 0):
                s[i] = b
                s[i + jj] = a
                sp[i] = pb
                sp[i + jj] = pa
            barrier()
            jj //= 2
        k *= 2
    w[g0 + tid] = s[tid]
    w[g0 + tid + MTG] = s[tid + MTG]
    pl[g0 + tid] = sp[tid]
    pl[g0 + tid + MTG] = sp[tid + MTG]


def _cc_sort[PAIRS: Bool](ctx: DeviceContext, w: WP, pl: WP, big_n: Int, tot: Int) raises:
    """dmi.mojo `mi_cd_device`'s bitonic sequence over tot = cols * big_n
    words (the payload moved along); PAIRS: compare (w, pl) pairs."""
    var tiles = tot // MTILE
    var pairs = tot // 2
    comptime if PAIRS:
        ctx.enqueue_function[_cc_tile_kernel](w, pl, Int32(big_n), Int32(2), Int32(MTILE), grid_dim=tiles,
                                              block_dim=MTG)
    else:
        ctx.enqueue_function[_tile_kernel](w, pl, Int32(big_n), Int32(2), Int32(MTILE), grid_dim=tiles, block_dim=MTG)
    var k = 2 * MTILE
    while k <= big_n:
        var j = k // 2
        while j >= MTILE:
            comptime if PAIRS:
                ctx.enqueue_function[_cc_global_kernel](w, pl, Int32(big_n), Int32(k), Int32(j), Int32(pairs),
                                                        grid_dim=_blocks(pairs, CBS), block_dim=CBS)
            else:
                ctx.enqueue_function[_global_kernel](w, pl, Int32(big_n), Int32(k), Int32(j), Int32(pairs),
                                                     grid_dim=_blocks(pairs, CBS), block_dim=CBS)
            j //= 2
        comptime if PAIRS:
            ctx.enqueue_function[_cc_tile_kernel](w, pl, Int32(big_n), Int32(k), Int32(k), grid_dim=tiles,
                                                  block_dim=MTG)
        else:
            ctx.enqueue_function[_tile_kernel](w, pl, Int32(big_n), Int32(k), Int32(k), grid_dim=tiles, block_dim=MTG)
        k *= 2


# ------------------------------------------------------------------ mi_cc: flags, gathers
def _cc_flag_kernel(f: FP, u: UP, q: IP, flags: Int32, total: Int32):
    """q = [Z, n, d, Y, k, TERM, ZS, YS]; t = i*d + c. A non-finite x or
    noise word flags column c (u[flags + c] = 1); a non-finite y or its
    noise word flags slot d (every column). The slots arrive zeroed."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var d = p(q, 2)
    var i = t // d
    var c = t - i * d
    var x = ld(f, p(q, 0) + i * d + c)
    var sx = _sec(f, p(q, 6), i * d + c)
    if not _fin(x) or not _fin(sx):
        u[Int(flags) + c] = UInt32(1)
    if c == 0:
        var y = ld(f, p(q, 3) + i)
        var sy = _sec(f, p(q, 7), i)
        if not _fin(y) or not _fin(sy):
            u[Int(flags) + d] = UInt32(1)


def _cc_gather_x_kernel(f: FP, pl: WP, u: UP, q: IP, big_n: Int32, ub: Int32, stride: Int32, with_y: Int32,
                        total: Int32):
    """t = c * n + r: column c's sorted arrays at U = ub + c * stride: AX[r],
    ASX[r] (the x word and its noise word of the point at rank r, j = the low
    word of pl[c * N + r]); with_y: AY[r], ASY[r] (its y and noise word) and
    POS[j] = r."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t // n
    var r = t - c * n
    var j = Int(pl[c * Int(big_n) + r] & UInt64(0xFFFFFFFF))
    var U = Int(ub) + c * Int(stride)
    u[U + r] = bitcast[DType.uint32](ld(f, p(q, 0) + j * d + c))
    u[U + n + r] = bitcast[DType.uint32](_sec(f, p(q, 6), j * d + c))
    if with_y != Int32(0):
        u[U + 2 * n + r] = bitcast[DType.uint32](ld(f, p(q, 3) + j))
        u[U + 3 * n + r] = bitcast[DType.uint32](_sec(f, p(q, 7), j))
        u[U + 4 * n + j] = UInt32(r)


def _cc_gather_y_kernel(f: FP, pl: WP, u: UP, q: IP, yb: Int32, total: Int32):
    """t = r: YSX[r], YSS[r] = the y word and its noise word of the point at
    sorted rank r (j = the low word of pl[r]); YPOS[j] = r."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(total):
        return
    var n = p(q, 1)
    var j = Int(pl[r] & UInt64(0xFFFFFFFF))
    var Y = Int(yb)
    u[Y + r] = bitcast[DType.uint32](ld(f, p(q, 3) + j))
    u[Y + n + r] = bitcast[DType.uint32](_sec(f, p(q, 7), j))
    u[Y + 2 * n + j] = UInt32(r)


# ------------------------------------------------------------------ mi_cc: the searches
@always_inline
def _offer(u: UP, AX: Int, ASX: Int, AY: Int, ASY: Int, r: Int, xi: Float32, sxi: Float32, yi: Float32,
           syi: Float32, k: Int, mut bp: InlineArray[Float32, MAX_K], mut bs: InlineArray[Float32, MAX_K]):
    """`mi_cc_unit`'s pair of candidate r (the joint Chebyshev distance as a
    (primary, secondary) pair) offered to the k best."""
    var xr = _fx(u, AX + r)
    var yr = _fx(u, AY + r)
    var dx = _dist(xr, xi)
    var dy = _dist(yr, yi)
    var sxd = _dsec(xi, xr, sxi, _fx(u, ASX + r))
    var syd = _dsec(yi, yr, syi, _fx(u, ASY + r))
    var dp = dx
    var dsec = sxd
    if _less(dx, sxd, dy, syd):
        dp = dy
        dsec = syd
    _ = _take(dp, dsec, k, bp, bs)


def _count_at(u: UP, xb: Int, sb: Int, n: Int, mid: Int, xi: Float32, si: Float32, rp: Float32, rs: Float32) -> Int:
    """x_prep/host/mutual_info.mojo `_count_within` over [0, n) split at the
    point's own sorted position `mid` instead of the lower bound of its value:
    the primary distance is monotone away from `mid` on either side, so the
    binary searches and the boundary walks find the same set."""
    var cnt = 0
    var l = mid
    var h = n
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, xb + m), xi) < rp:
            l = m + 1
        else:
            h = m
    cnt += l - mid
    var r = l
    while r < n and _dist(_fx(u, xb + r), xi) == rp:
        if _within(rp, _dsec(xi, _fx(u, xb + r), si, _fx(u, sb + r)), rp, rs):
            cnt += 1
        r += 1
    l = 0
    h = mid
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, xb + m), xi) < rp:
            h = m
        else:
            l = m + 1
    cnt += mid - l
    var qq = l - 1
    while qq >= 0 and _dist(_fx(u, xb + qq), xi) == rp:
        if _within(rp, _dsec(xi, _fx(u, xb + qq), si, _fx(u, sb + qq)), rp, rs):
            cnt += 1
        qq -= 1
    return cnt


def _count_ties_at(u: UP, xb: Int, sb: Int, n: Int, mid: Int, xi: Float32, si: Float32, rp: Float32,
                   rs: Float32) -> Int:
    """dmi.mojo `_count_within_ties` (the boundary runs, in noise-word order,
    counted by binary search) split at the own position `mid`; a partial run
    on either side of `mid` is still in noise-word order."""
    var cnt = 0
    var l = mid
    var h = n
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, xb + m), xi) < rp:
            l = m + 1
        else:
            h = m
    cnt += l - mid
    var r = l
    while r < n and _dist(_fx(u, xb + r), xi) == rp:
        var re = _run_hi(u, xb, r, n)
        cnt += _count_run(u, xb, sb, r, re, xi, si, rp, rs)
        r = re
    l = 0
    h = mid
    while l < h:
        var m = (l + h) // 2
        if _dist(_fx(u, xb + m), xi) < rp:
            h = m
        else:
            l = m + 1
    cnt += mid - l
    var qq = l - 1
    while qq >= 0 and _dist(_fx(u, xb + qq), xi) == rp:
        var rs0 = _run_lo(u, xb, 0, qq + 1)
        cnt += _count_run(u, xb, sb, rs0, qq + 1, xi, si, rp, rs)
        qq = rs0 - 1
    return cnt


@always_inline
def _rhi(u: UP, xb: Int, lo: Int, hi: Int) -> Int:
    """dmi.mojo `_run_hi` with a one-compare answer for a singleton run."""
    if lo + 1 >= hi or _kx(u, xb + lo + 1) != _kx(u, xb + lo):
        return lo + 1
    return _run_hi(u, xb, lo, hi)


@always_inline
def _rlo(u: UP, xb: Int, lo: Int, hi: Int) -> Int:
    """dmi.mojo `_run_lo` with a one-compare answer for a singleton run."""
    if hi - 2 < lo or _kx(u, xb + hi - 2) != _kx(u, xb + hi - 1):
        return hi - 1
    return _run_lo(u, xb, lo, hi)


def _cc_run_ties(u: UP, AX: Int, ASX: Int, AY: Int, ASY: Int, r0: Int, r1: Int, xi: Float32, sxi: Float32,
                 yi: Float32, syi: Float32, k: Int, mut bp: InlineArray[Float32, MAX_K],
                 mut bs: InlineArray[Float32, MAX_K]):
    """Offer the members of x-run [r0, r1) (in y order) whose |dy| does not
    exceed the current k-th primary: every member's pair has primary
    max(dx, dy) >= dy, and dy is monotone away from yi's position."""
    var l = r0
    var h = r1
    while l < h:
        var m = (l + h) // 2
        if _fx(u, AY + m) < yi:
            l = m + 1
        else:
            h = m
    var L = l - 1
    var R = l
    while L >= r0 or R < r1:
        var dl = _dist(_fx(u, AY + L), yi) if L >= r0 else BIG
        var dr = _dist(_fx(u, AY + R), yi) if R < r1 else BIG
        var take_left = L >= r0 and (R >= r1 or dl <= dr)
        var dy = dl if take_left else dr
        if dy > bp[k - 1]:
            break
        _offer(u, AX, ASX, AY, ASY, L if take_left else R, xi, sxi, yi, syi, k, bp, bs)
        if take_left:
            L -= 1
        else:
            R += 1


def _cc_knn_ties(u: UP, AX: Int, ASX: Int, AY: Int, ASY: Int, n: Int, pos: Int, xi: Float32, sxi: Float32,
                 yi: Float32, syi: Float32, k: Int, mut bp: InlineArray[Float32, MAX_K],
                 mut bs: InlineArray[Float32, MAX_K]):
    """The k smallest pairs of the point at sorted position `pos` in the
    (key(x), key(y), key(sx)) order. A: the own (x, y) stretch outward in sx
    (pairs (0, dsec >= sxd): stop once (0, sxd) cannot enter). B: the rest of
    the own x-run outward in y (pairs (dy, syd): stop once dy exceeds the
    k-th primary). C: the other runs outward in x (a run stops the walk once
    its dx exceeds the k-th primary; inside, `_cc_run_ties`). Every candidate
    that could be among the k smallest is offered, in some order, and the
    k-th smallest pair is an order statistic: the unit's."""
    var b0 = _rlo(u, AX, 0, pos + 1)
    var b1 = _rhi(u, AX, pos, n)
    var s0 = _rlo(u, AY, b0, pos + 1)
    var s1 = _rhi(u, AY, pos, b1)
    # A: the stretch, outward in sx from pos
    var L = pos - 1
    var R = pos + 1
    while L >= s0 or R < s1:
        var dl = abs(sub(_fx(u, ASX + L), sxi)) if L >= s0 else BIG
        var dr = abs(sub(_fx(u, ASX + R), sxi)) if R < s1 else BIG
        var take_left = L >= s0 and (R >= s1 or dl <= dr)
        var sxd = dl if take_left else dr
        if not _less(Float32(0), sxd, bp[k - 1], bs[k - 1]):
            break
        _offer(u, AX, ASX, AY, ASY, L if take_left else R, xi, sxi, yi, syi, k, bp, bs)
        if take_left:
            L -= 1
        else:
            R += 1
    # B: the own run past the stretch, outward in y
    L = s0 - 1
    R = s1
    while L >= b0 or R < b1:
        var dl = _dist(_fx(u, AY + L), yi) if L >= b0 else BIG
        var dr = _dist(_fx(u, AY + R), yi) if R < b1 else BIG
        var take_left = L >= b0 and (R >= b1 or dl <= dr)
        var dy = dl if take_left else dr
        if dy > bp[k - 1]:
            break
        _offer(u, AX, ASX, AY, ASY, L if take_left else R, xi, sxi, yi, syi, k, bp, bs)
        if take_left:
            L -= 1
        else:
            R += 1
    # C: the other runs, outward in x
    var Lr = b0
    var Rr = b1
    while Lr > 0 or Rr < n:
        var dl = _dist(_fx(u, AX + Lr - 1), xi) if Lr > 0 else BIG
        var dr = _dist(_fx(u, AX + Rr), xi) if Rr < n else BIG
        var take_left = Lr > 0 and (Rr >= n or dl <= dr)
        var dx = dl if take_left else dr
        if dx > bp[k - 1]:
            break
        var r0: Int
        var r1: Int
        if take_left:
            r0 = _rlo(u, AX, 0, Lr)
            r1 = Lr
            Lr = r0
        else:
            r0 = Rr
            r1 = _rhi(u, AX, Rr, n)
            Rr = r1
        _cc_run_ties(u, AX, ASX, AY, ASY, r0, r1, xi, sxi, yi, syi, k, bp, bs)


def _cc_point_kernel[TIES: Bool, RANK: Bool](f: FP, u: UP, pl: WP, q: IP, big_n: Int32, ub: Int32, bb: Int32,
                                             yb: Int32, flags: Int32, total: Int32):
    """`mi_cc_unit`'s word for one (point, column): RANK maps t = c * n + r
    (the point at sorted rank r, its index from the payload), else t =
    i * d + c. A flagged column, or a flagged y, runs the unit."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 4)
    var i: Int
    var c: Int
    var pos: Int
    comptime if RANK:
        c = t // n
        pos = t - c * n
        i = Int(pl[c * Int(big_n) + pos] & UInt64(0xFFFFFFFF))
    else:
        i = t // d
        c = t - i * d
        pos = Int(u[Int(ub) + c * 5 * n + 4 * n + i])
    var tu = i * d + c
    if u[Int(flags) + c] != UInt32(0) or u[Int(flags) + d] != UInt32(0):
        mi_cc_unit(tu, f, q)
        return
    var U = Int(ub) + c * 5 * n
    var AX = U
    var ASX = U + n
    var AY = U + 2 * n
    var ASY = U + 3 * n
    var YSX = Int(yb)
    var YSS = Int(yb) + n
    var ypos = Int(u[Int(yb) + 2 * n + i])
    var xi = _fx(u, AX + pos)
    var sxi = _fx(u, ASX + pos)
    var yi = _fx(u, AY + pos)
    var syi = _fx(u, ASY + pos)
    var bp = InlineArray[Float32, MAX_K](fill=BIG)
    var bs = InlineArray[Float32, MAX_K](fill=BIG)
    comptime if TIES:
        _cc_knn_ties(u, AX, ASX, AY, ASY, n, pos, xi, sxi, yi, syi, k, bp, bs)
    else:
        # the host's `point(i)`: outward in x, a side stops past the k-th primary
        var L = pos - 1
        var R = pos + 1
        while L >= 0 or R < n:
            var dl = _dist(_fx(u, AX + L), xi) if L >= 0 else BIG
            var dr = _dist(_fx(u, AX + R), xi) if R < n else BIG
            var take_left = L >= 0 and (R >= n or dl <= dr)
            var dx = dl if take_left else dr
            if dx > bp[k - 1]:
                break
            _offer(u, AX, ASX, AY, ASY, L if take_left else R, xi, sxi, yi, syi, k, bp, bs)
            if take_left:
                L -= 1
            else:
                R += 1
    var rp = bp[k - 1]
    var rs = bs[k - 1]
    var nx: Int
    var ny: Int
    comptime if TIES:
        var V = Int(bb) + c * 2 * n
        nx = _count_within_ties(u, V, V + n, n, xi, sxi, rp, rs)
        ny = _count_ties_at(u, YSX, YSS, n, ypos, yi, syi, rp, rs)
    else:
        nx = _count_at(u, AX, ASX, n, pos, xi, sxi, rp, rs)
        ny = _count_at(u, YSX, YSS, n, ypos, yi, syi, rp, rs)
    st(f, p(q, 5) + tu, add(digammaf(Float32(nx)), digammaf(Float32(ny))))


def mi_cc_device[TIES: Bool, RANK: Bool](ctx: DeviceContext, mut df: DeviceBuffer[DType.float32],
                                        mut dq: DeviceBuffer[DType.int32], qoff: Int, total: Int, n: Int, d: Int,
                                        Z: Int, zs1: Int, Y: Int, ys1: Int) raises:
    """Enqueue the whole `mi_cc` stage (q = [Z, n, d, Y, k, TERM, ZS, YS] at
    dq[qoff:]) as the sorted search. Its scratch is allocated here and freed
    after one synchronize at the end of the stage (nothing of it outlives the
    FAST branch). UInt64 words: the x sort words and payload (TIES: twice,
    the second copy in (key(x), key(sx)) order), then y's; UInt32 words: per
    column AX, ASX, AY, ASY, POS (n each), TIES: per column BX, BSX, then
    YSX, YSS, YPOS, then d + 1 flags."""
    if n <= 0 or d <= 0:
        return
    var big_n = mi_big_n(n)
    var tot = d * big_n
    comptime NW = 4 if TIES else 2
    var wwords = NW * tot + 2 * big_n
    var ub = 0
    var bb = 5 * n * d
    var yb = bb + (2 * n * d if TIES else 0)
    var flags = yb + 3 * n
    var uwords = flags + d + 1
    if tot > 2 ** 31 - 1 or uwords > 2 ** 31 - 1 or wwords > 2 ** 31 - 1:
        raise Error("x_prep: mi_cc too large for the device search")
    var dw = ctx.enqueue_create_buffer[DType.uint64](wwords)
    var du = ctx.enqueue_create_buffer[DType.uint32](uwords)
    var f = df.unsafe_ptr()
    var qp = dq.unsafe_ptr() + qoff
    var u = du.unsafe_ptr()
    var wb = Int(dw.unsafe_ptr())
    var wa = WP(unsafe_from_address=wb)
    var pa = WP(unsafe_from_address=wb + 8 * tot)
    var wy = WP(unsafe_from_address=wb + 8 * (NW * tot))
    var py = WP(unsafe_from_address=wb + 8 * (NW * tot + big_n))
    ctx.enqueue_memset(du.create_sub_buffer[DType.uint32](flags, d + 1), UInt32(0))
    ctx.enqueue_function[_cc_flag_kernel](f, u, qp, Int32(flags), Int32(total), grid_dim=_blocks(total, CBS),
                                          block_dim=CBS)
    comptime if TIES:
        ctx.enqueue_function[_cc_load_xy_kernel](
            f, wa, pa, Int32(Z), Int32(n), Int32(d), Int32(big_n), Int32(tot), Int32(zs1), Int32(Y),
            grid_dim=_blocks(tot, CBS), block_dim=CBS,
        )
        _cc_sort[True](ctx, wa, pa, big_n, tot)
    else:
        ctx.enqueue_function[_load_kernel](
            f, wa, pa, Int32(Z), Int32(n), Int32(d), Int32(big_n), Int32(tot), Int32(zs1), Int32(0),
            grid_dim=_blocks(tot, CBS), block_dim=CBS,
        )
        _cc_sort[False](ctx, wa, pa, big_n, tot)
    ctx.enqueue_function[_cc_gather_x_kernel](
        f, pa, u, qp, Int32(big_n), Int32(ub), Int32(5 * n), Int32(1), Int32(n * d), grid_dim=_blocks(n * d, CBS),
        block_dim=CBS,
    )
    comptime if TIES:
        var wb2 = WP(unsafe_from_address=wb + 8 * (2 * tot))
        var pb2 = WP(unsafe_from_address=wb + 8 * (3 * tot))
        ctx.enqueue_function[_load_kernel](
            f, wb2, pb2, Int32(Z), Int32(n), Int32(d), Int32(big_n), Int32(tot), Int32(zs1), Int32(1),
            grid_dim=_blocks(tot, CBS), block_dim=CBS,
        )
        _cc_sort[False](ctx, wb2, pb2, big_n, tot)
        ctx.enqueue_function[_cc_gather_x_kernel](
            f, pb2, u, qp, Int32(big_n), Int32(bb), Int32(2 * n), Int32(0), Int32(n * d),
            grid_dim=_blocks(n * d, CBS), block_dim=CBS,
        )
    ctx.enqueue_function[_load_kernel](
        f, wy, py, Int32(Y), Int32(n), Int32(1), Int32(big_n), Int32(big_n), Int32(ys1), Int32(1 if TIES else 0),
        grid_dim=_blocks(big_n, CBS), block_dim=CBS,
    )
    _cc_sort[False](ctx, wy, py, big_n, big_n)
    ctx.enqueue_function[_cc_gather_y_kernel](f, py, u, qp, Int32(yb), Int32(n), grid_dim=_blocks(n, CBS),
                                              block_dim=CBS)
    ctx.enqueue_function[_cc_point_kernel[TIES, RANK]](
        f, u, pa, qp, Int32(big_n), Int32(ub), Int32(bb), Int32(yb), Int32(flags), Int32(total),
        grid_dim=_blocks(total, CBS), block_dim=CBS,
    )
    ctx.synchronize()
    _ = dw^
    _ = du^


# ------------------------------------------------------------------ MI_FAST_FOLDS
def mi_colscale_fast_kernel(f: FP, q: IP):
    """`mi_colscale_unit` for column block_idx.x by a threadgroup fold:
    q = [X, n, d, ST, SCALE, MABS]; each thread sums abs(x / SCALE) over rows
    tid, tid + TGF, ..., then the tree."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var s = zero_to_one(sqrtf(ld(f, p(q, 3) + 2 * d + c)))
    var sh = stack_allocation[TGF, Float32, address_space = AddressSpace.SHARED]()
    var acc = Float32(0)
    for i in range(tid, n, TGF):
        acc = add(acc, abs(div(ld(f, p(q, 0) + i * d + c), s)))
    sh[tid] = acc
    barrier()
    var w = TGF // 2
    while w >= 1:
        if tid < w:
            sh[tid] = add(sh[tid], sh[tid + w])
        barrier()
        w //= 2
    if tid == 0:
        st(f, p(q, 4) + c, s)
        var m = div(sh[0], Float32(n))
        st(f, p(q, 5) + c, m if m > Float32(1) else Float32(1))


def mi_reduce_fast_kernel(f: FP, q: IP):
    """`mi_reduce_unit` for column block_idx.x: TERM summed by the fold, the
    rest (digamma, NUSED, the clip at 0) by thread 0 as the unit writes it.
    q = [TERM, n, d, KIND, k, NUSED, OUT, CNT, KS]."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var sh = stack_allocation[TGF, Float32, address_space = AddressSpace.SHARED]()
    var acc = Float32(0)
    for i in range(tid, n, TGF):
        acc = add(acc, ld(f, p(q, 0) + i * d + c))
    sh[tid] = acc
    barrier()
    var w = TGF // 2
    while w >= 1:
        if tid < w:
            sh[tid] = add(sh[tid], sh[tid + w])
        barrier()
        w //= 2
    if tid != 0:
        return
    var s = sh[0]
    var mi: Float32
    if p(q, 3) == 0:
        mi = sub(add(digammaf(Float32(n)), digammaf(Float32(p(q, 4)))), div(s, Float32(n)))
    else:
        var used = p(q, 5)
        if p(q, 3) == 2:
            used = 0
            for kk in range(p(q, 8)):
                var cnt = Int(ld(f, p(q, 7) + c * p(q, 8) + kk))
                if cnt > 1:
                    used += cnt
        mi = add(digammaf(Float32(used)), div(s, Float32(used))) if used > 0 else Float32(0)
    st(f, p(q, 6) + c, mi if mi > Float32(0) else Float32(0))


# ------------------------------------------------------------------ MI_CLF_RANKMAJOR
def _cd_point_rank_kernel(f: FP, u: UP, pl: WP, q: IP, big_n: Int32, total: Int32):
    """dmi.mojo `_point_ties_kernel` with t = c * n + r: the point at sorted
    rank r of column c (its index from the sort payload pl[c * N + r])."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t // n
    var r = t - c * n
    var i = Int(pl[c * Int(big_n) + r])
    var tu = i * d + c
    var U = c * (6 * n + 5)
    var meta = U + 6 * n + 1
    if u[meta + _M_FLAG] != UInt32(0):
        mi_cd_unit(tu, f, q)
        return
    var zb = p(q, 0) + c
    var sb = p(q, 7) + c if p(q, 7) > 0 else 0
    var k = p(q, 5)
    var li = Int(ld(f, p(q, 3) + i))
    var cnt = Int(ld(f, p(q, 4) + li))
    var out = p(q, 6) + tu
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
    var g0 = _run_lo(u, bx, b, pi + 1)
    var g1 = _run_hi(u, bx, pi, e)
    _take_run(u, bx, bsb, g0, g1, pi, xi, si, kl, bp, bs)
    var considered = g1 - g0 - 1
    var D = _dist(_fx(u, bx + pi), xi)
    var Lr = g0
    var Rr = g1
    while Lr > b or Rr < e:
        var dl = _dist(_fx(u, bx + Lr - 1), xi) if Lr > b else BIG
        var dr = _dist(_fx(u, bx + Rr), xi) if Rr < e else BIG
        var take_left = Lr > b and (Rr >= e or dl <= dr)
        var dn = dl if take_left else dr
        if considered >= kl and dn > D:
            break
        D = dn
        if take_left:
            var ls = _run_lo(u, bx, b, Lr)
            _take_run(u, bx, bsb, ls, Lr, -1, xi, si, kl, bp, bs)
            considered += Lr - ls
            Lr = ls
        else:
            var rend = _run_hi(u, bx, Rr, e)
            _take_run(u, bx, bsb, Rr, rend, -1, xi, si, kl, bp, bs)
            considered += rend - Rr
            Rr = rend
    var rp = bp[kl - 1]
    var rs = bs[kl - 1]
    var na = Int(u[meta + _M_NA])
    var mall = _count_within_ties(u, U + 3 * n, U + 4 * n, na, xi, si, rp, rs)
    st(f, out, sub(sub(digammaf(Float32(kl)), digammaf(Float32(cnt))), digammaf(Float32(mall))))


def mi_cd_device_rank(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint64],
                      mut du: DeviceBuffer[DType.uint32], mut dq: DeviceBuffer[DType.int32], qoff: Int, total: Int,
                      n: Int, d: Int, Z: Int, zs1: Int) raises:
    """dmi.mojo `mi_cd_device` with ties on and the point kernel in the
    rank-major mapping; the same sort, split and scratch."""
    if n <= 0 or d <= 0:
        return
    var big_n = mi_big_n(n)
    var tot = d * big_n
    if tot > 2 ** 31 - 1:
        raise Error("x_prep: mi_cd too large for the device search")
    var f = df.unsafe_ptr()
    var w = WP(unsafe_from_address=Int(dw.unsafe_ptr()))
    var pl = WP(unsafe_from_address=Int(dw.unsafe_ptr()) + 8 * tot)
    var u = du.unsafe_ptr()
    var qp = dq.unsafe_ptr() + qoff
    ctx.enqueue_function[_load_kernel](
        f, w, pl, Int32(Z), Int32(n), Int32(d), Int32(big_n), Int32(tot), Int32(zs1), Int32(1),
        grid_dim=_blocks(tot, CBS), block_dim=CBS,
    )
    _cc_sort[False](ctx, w, pl, big_n, tot)
    ctx.enqueue_function[_prep_kernel](f, pl, u, qp, Int32(big_n), grid_dim=_blocks(d, 32), block_dim=32)
    ctx.enqueue_function[_cd_point_rank_kernel](f, u, pl, qp, Int32(big_n), Int32(total),
                                                grid_dim=_blocks(total, CBS), block_dim=CBS)
