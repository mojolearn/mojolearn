# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IsotonicRegression, FAST on Apple: the whole fit on the grid
(lane/apple-fast-isotonic-knn, 2026-10-02). FAST + Apple DEFAULT in the
x_linear binding (bindings/_mojolearn_x_linear.mojo; `-D
MOJOLEARN_ISOTONIC_FAST_PAR_OFF` reverts to the team fit); IDENTICAL never
compiles this module. The parallel pairwise pool merges (PAIRMERGE, below)
replace step 6's boundary rounds by default (`-D
MOJOLEARN_ISOTONIC_FAST_PAIRMERGE_OFF` reverts to the boundary rounds). M2 A/B
ik-iso-pair-istella-b: PAR alone timed out, PAR + PAIRMERGE 165.5 ms, r2 .188.

Cause: x_linear/isotonic.mojo `isotonic_fit` runs on the lead thread of ONE
block (isotonic.mojo:129 `if not t.lead(): return`): the merge sort of a
million rows (isotonic.mojo:93 `_iso_sort`, one device thread), the pooling
of equal x (`_make_unique`, :157), the pool-adjacent-violators (:190) and
the trim (:236), all one thread. The board's isotonic row is 12-16x behind
scikit-learn on both datasets.

Here, every step is a grid launch over the rows or the pooled points:

  1. the rows with a positive weight, compacted by a scan; a 64-bit sort
     key per row (x's bits, then y's, each made order-preserving), the
     row index the tie-break: `_iso_less`'s total order;
  2. a bottom-up merge sort of the keys by ranking: in each round a thread
     places its element by its index in its own run plus a binary search
     in the partner run (log2 n rounds, contiguous reads);
  3. the sorted copies xs | ys | ws by a gather;
  4. `_make_unique`'s pools as the runs of EQUAL x (flags, scan); the serial
     rule pools two unequal x closer than 1e-6 too, so one such pair sets a
     flag and the caller runs the team fit (the same result, no new rule);
  5. each pool's weighted mean from part sums (a thread per 4,096 rows of
     a pool, then a thread per pool), written in reverse for a decreasing
     fit, as the serial code reverses;
  6. PAVA on blocks of 256 pooled points in parallel (the serial code
     itself on each range), then log2 rounds merging adjacent ranges at
     their boundary pools: the L2 isotonic fit is unique, so any order of
     adjacent-violator merges gives the same pools. The last rounds are a
     few threads each walking one boundary outward. PAIRMERGE instead runs
     2 log2 n + 4 rounds in which EVERY pool start of the round's rank
     parity (rank by a scan of the start flags) merges with its right
     neighbour when it violates (a thread per pool, disjoint pairs), then a
     grid check writes meta[4] when a violation is left: the caller then
     runs the team fit, so an unconverged answer never leaves;
  7. each pool's value to its points (pool starts by a max-scan), the clip,
     the trim's keep flags and their scan, the scatter into res.

Weighted means and pool means sum in another order than the serial chains:
FAST promises quality, not bits. Layout of res / fw as `isotonic_fit`.
"""
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, IP, ld, st, ldi, sti, fa, fs, fm, fd, fmad, fmax, fmin, i2f

comptime IF_TPB = 256
#: Rows per part of a pool's sum (step 5).
comptime IF_PART = 4096
#: Pooled points per PAVA block (step 6).
comptime IF_PAVA_L = 256
comptime KP = MutPointer[UInt64, MutAnyOrigin]


def _ifb(count: Int) -> Int:
    return max((count + IF_TPB - 1) // IF_TPB, 1)


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * IF_TPB + Int(thread_idx.x)


@always_inline
def _ord_bits(v: Float32) -> UInt64:
    """float32 -> an unsigned word ordered as the float (finite inputs;
    -0 as +0, as `_iso_less` compares them equal)."""
    var f = v
    if f == Float32(0):
        f = Float32(0)
    var u = bitcast[DType.uint32](f)
    if (u & UInt32(0x80000000)) != UInt32(0):
        u = ~u
    else:
        u = u | UInt32(0x80000000)
    return UInt64(u)


@always_inline
def _key_less(ka: UInt64, ia: Int, kb: UInt64, ib: Int) -> Bool:
    if ka != kb:
        return ka < kb
    return ia < ib


# ---------------------------------------------------------------- scans
def if_scan_block_kernel[MX: Bool](inp: IP, dst: IP, blk: IP, n: Int32):
    """Inclusive scan (sum, or max when MX) of each block's 256 words into
    `dst`; the block's total into blk[block]."""
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var g = b * IF_TPB + tid
    var sh = stack_allocation[IF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var ident = Int32(-1) if MX else Int32(0)
    var v = ident
    if g < Int(n):
        v = inp.unsafe_load(g)
    sh[tid] = v
    barrier()
    var off = 1
    while off < IF_TPB:
        var t = sh[tid]
        if tid >= off:
            var u = sh[tid - off]
            comptime if MX:
                if u > t:
                    t = u
            else:
                t = t + u
        barrier()
        sh[tid] = t
        barrier()
        off *= 2
    if g < Int(n):
        dst.unsafe_store(g, sh[tid])
    if tid == IF_TPB - 1:
        blk.unsafe_store(b, sh[tid])


def if_scan_totals_kernel[MX: Bool](blk: IP, nb: Int32):
    """ONE block: the inclusive scan of the block totals, in place, 256 at
    a time with a carry."""
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[IF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var ident = Int32(-1) if MX else Int32(0)
    var carry = ident
    var c0 = 0
    while c0 < Int(nb):
        var i = c0 + tid
        var v = ident
        if i < Int(nb):
            v = blk.unsafe_load(i)
        sh[tid] = v
        barrier()
        var off = 1
        while off < IF_TPB:
            var t = sh[tid]
            if tid >= off:
                var u = sh[tid - off]
                comptime if MX:
                    if u > t:
                        t = u
                else:
                    t = t + u
            barrier()
            sh[tid] = t
            barrier()
            off *= 2
        var r = sh[tid]
        var last = sh[IF_TPB - 1]
        comptime if MX:
            if carry > r:
                r = carry
            if carry > last:
                last = carry
        else:
            r = r + carry
            last = last + carry
        if i < Int(nb):
            blk.unsafe_store(i, r)
        carry = last
        barrier()
        c0 += IF_TPB


def if_scan_add_kernel[MX: Bool](dst: IP, blk: IP, n: Int32):
    """Each block's words carry the scanned total of the blocks before it."""
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var g = b * IF_TPB + tid
    if b > 0 and g < Int(n):
        var c = blk.unsafe_load(b - 1)
        var v = dst.unsafe_load(g)
        comptime if MX:
            if c > v:
                v = c
        else:
            v = v + c
        dst.unsafe_store(g, v)


def _if_scan[MX: Bool](
    mut ctx: DeviceContext, mut inp: DeviceBuffer[DType.int32], mut dst: DeviceBuffer[DType.int32],
    mut blk: DeviceBuffer[DType.int32], n: Int,
) raises:
    """dst = the inclusive scan of inp[0, n) (sum, or max when MX)."""
    var nb = _ifb(n)
    ctx.enqueue_function[if_scan_block_kernel[MX]](
        inp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), blk.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), grid_dim=nb, block_dim=IF_TPB)
    ctx.enqueue_function[if_scan_totals_kernel[MX]](blk.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(nb), grid_dim=1, block_dim=IF_TPB)
    ctx.enqueue_function[if_scan_add_kernel[MX]](
        dst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), blk.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), grid_dim=nb, block_dim=IF_TPB)


# ---------------------------------------------------------------- rows
def if_flag_kernel(y: FP, n: Int32, flag: IP):
    """flag[i] = 1 for a row with a positive weight (weights at y[n, 2n))."""
    var i = _gid()
    if i < Int(n):
        sti(flag, i, 1 if ld(y, Int(n) + i) > Float32(0) else 0)


def if_perm_kernel(x: FP, y: FP, n: Int32, has_w: Int32, flag: IP, scan: IP, perm: IP, keys: KP, meta: IP):
    """Row i to its slot of perm (its scan position when weighted, i
    otherwise) with its sort key; meta[0] = nk, the row count."""
    var nn = Int(n)
    var i = _gid()
    if i < nn:
        var pos = i
        var keep = True
        if has_w != 0:
            keep = ldi(flag, i) != 0
            pos = ldi(scan, i) - 1
        if keep:
            sti(perm, pos, i)
            keys.unsafe_store(pos, (_ord_bits(ld(x, i)) << 32) | _ord_bits(ld(y, i)))
        if i == nn - 1:
            sti(meta, 0, ldi(scan, i) if has_w != 0 else nn)


def if_sort_kernel(ks: KP, ps: IP, kd: KP, pd: IP, meta: IP, width: Int32):
    """One merge round: sorted runs of `width` in (ks, ps) to sorted runs
    of 2 * width in (kd, pd). Element j of a run lands at its index in the
    run plus the count of the partner run's elements below it (a total
    order: no two elements compare equal)."""
    var nk = ldi(meta, 0)
    var j = _gid()
    if j >= nk:
        return
    var w = Int(width)
    var base = (j // (2 * w)) * (2 * w)
    var a1 = min(base + w, nk)
    var b1 = min(base + 2 * w, nk)
    var key = ks.unsafe_load(j)
    var id = ldi(ps, j)
    var lo = 0
    var hi = 0
    var pos = 0
    if j < a1:
        lo = a1
        hi = b1
        while lo < hi:
            var mid = (lo + hi) // 2
            if _key_less(ks.unsafe_load(mid), ldi(ps, mid), key, id):
                lo = mid + 1
            else:
                hi = mid
        pos = (j - base) + (lo - a1)
    else:
        lo = base
        hi = a1
        while lo < hi:
            var mid = (lo + hi) // 2
            if _key_less(ks.unsafe_load(mid), ldi(ps, mid), key, id):
                lo = mid + 1
            else:
                hi = mid
        pos = (j - a1) + (lo - base)
    kd.unsafe_store(base + pos, key)
    sti(pd, base + pos, id)


def if_gather_kernel(x: FP, y: FP, n: Int32, has_w: Int32, perm: IP, fw: FP, meta: IP):
    """xs | ys | ws (fw + 3n, 4n, 5n) in sorted order."""
    var nn = Int(n)
    var nk = ldi(meta, 0)
    var j = _gid()
    if j < nk:
        var r = ldi(perm, j)
        st(fw, 3 * nn + j, ld(x, r))
        st(fw, 4 * nn + j, ld(y, r))
        st(fw, 5 * nn + j, ld(y, nn + r) if has_w != 0 else Float32(1))


def if_segflag_kernel(fw: FP, n: Int32, meta: IP, flag: IP):
    """flag[j] = 1 at the first row of each run of equal x. Two unequal
    neighbours under `_make_unique`'s 1e-6 set meta[1]: the serial rule
    would pool them, so the caller runs the team fit."""
    var nn = Int(n)
    var nk = ldi(meta, 0)
    var j = _gid()
    if j < nk:
        var f = 1
        if j > 0:
            var xj = ld(fw, 3 * nn + j)
            var xp = ld(fw, 3 * nn + j - 1)
            if xj == xp:
                f = 0
            elif fs(xj, xp) < Float32(1e-6):
                sti(meta, 1, 1)
        sti(flag, j, f)


def if_segstart_kernel(flag: IP, scan: IP, meta: IP, sstart: IP):
    """sstart[s] = the first row of pool s; meta[2] = m, the pool count."""
    var nk = ldi(meta, 0)
    var j = _gid()
    if j < nk:
        if ldi(flag, j) != 0:
            sti(sstart, ldi(scan, j) - 1, j)
        if j == nk - 1:
            sti(meta, 2, ldi(scan, j))


@always_inline
def _seg_bounds(sstart: IP, nk: Int, m: Int, s: Int) -> Tuple[Int, Int]:
    var s0 = ldi(sstart, s)
    var s1 = ldi(sstart, s + 1) if s + 1 < m else nk
    return (s0, s1)


@always_inline
def _seg_parts(s0: Int, s1: Int) -> Int:
    return (s1 - s0 + IF_PART - 1) // IF_PART


def if_partcount_kernel(sstart: IP, meta: IP, n: Int32, flag: IP):
    """flag[s] = the number of IF_PART-row parts of pool s (0 past m)."""
    var nk = ldi(meta, 0)
    var m = ldi(meta, 2)
    var s = _gid()
    if s < Int(n):
        var c = 0
        if s < m:
            var b = _seg_bounds(sstart, nk, m, s)
            c = _seg_parts(b[0], b[1])
        sti(flag, s, c)


def if_parttag_kernel(sstart: IP, scan: IP, meta: IP, pseg: IP):
    """pseg[p] = the pool of part p; meta[3] = the part count."""
    var nk = ldi(meta, 0)
    var m = ldi(meta, 2)
    var s = _gid()
    if s >= m:
        return
    var b = _seg_bounds(sstart, nk, m, s)
    var ps = _seg_parts(b[0], b[1])
    var off = ldi(scan, s) - ps
    for p in range(ps):
        sti(pseg, off + p, s)
    if s == m - 1:
        sti(meta, 3, ldi(scan, s))


def if_partsum_kernel(fw: FP, n: Int32, sstart: IP, scan: IP, meta: IP, pseg: IP, py: FP, pw: FP):
    """Part p: sum of w * y and of w over its rows."""
    var nn = Int(n)
    var nk = ldi(meta, 0)
    var m = ldi(meta, 2)
    var np = ldi(meta, 3)
    var p = _gid()
    if p >= np:
        return
    var s = ldi(pseg, p)
    var b = _seg_bounds(sstart, nk, m, s)
    var ps = _seg_parts(b[0], b[1])
    var q = p - (ldi(scan, s) - ps)
    var lo = b[0] + q * IF_PART
    var hi = min(lo + IF_PART, b[1])
    var cy = Float32(0)
    var cw = Float32(0)
    for j in range(lo, hi):
        var wj = ld(fw, 5 * nn + j)
        cy = fmad(ld(fw, 4 * nn + j), wj, cy)
        cw = fa(cw, wj)
    st(py, p, cy)
    st(pw, p, cw)


def if_segmean_kernel(fw: FP, n: Int32, inc: Int32, sstart: IP, scan: IP, meta: IP, py: FP, pw: FP, iw: IP):
    """Pool s: ux[s] = its x, and at q (= s, or m - 1 - s for a decreasing
    fit) its weighted mean uy[q], weight uw[q] and the PAVA link iw[q] = q."""
    var nn = Int(n)
    var nk = ldi(meta, 0)
    var m = ldi(meta, 2)
    var s = _gid()
    if s >= m:
        return
    var b = _seg_bounds(sstart, nk, m, s)
    var ps = _seg_parts(b[0], b[1])
    var off = ldi(scan, s) - ps
    var cy = Float32(0)
    var cw = Float32(0)
    for p in range(off, off + ps):
        cy = fa(cy, ld(py, p))
        cw = fa(cw, ld(pw, p))
    var q = s if inc != 0 else m - 1 - s
    st(fw, s, ld(fw, 3 * nn + b[0]))
    st(fw, nn + q, fd(cy, cw))
    st(fw, 2 * nn + q, cw)
    sti(iw, q, q)


# ---------------------------------------------------------------- PAVA
def _pava_range(uy: FP, uw: FP, iw: IP, lo: Int, hi: Int):
    """x_linear/isotonic.mojo's `_inplace_contiguous_isotonic_regression`
    loop on the points [lo, hi): pool start i links to its end (iw[i]) and
    the end back to i."""
    var i = lo
    while i < hi:
        var k = ldi(iw, i) + 1
        if k == hi:
            break
        if ld(uy, i) < ld(uy, k):
            i = k
            continue
        var swy = fm(ld(uw, i), ld(uy, i))
        var sw = ld(uw, i)
        while True:
            var prev_y = ld(uy, k)
            swy = fmad(ld(uw, k), ld(uy, k), swy)
            sw = fa(sw, ld(uw, k))
            k = ldi(iw, k) + 1
            if k == hi or prev_y < ld(uy, k):
                st(uy, i, fd(swy, sw))
                st(uw, i, sw)
                sti(iw, i, k - 1)
                sti(iw, k - 1, i)
                if i > lo:
                    i = ldi(iw, i - 1)
                break


def _pava_merge(uy: FP, uw: FP, iw: IP, lo: Int, mid: Int, hi: Int):
    """[lo, mid) and [mid, hi) each isotonic: pool across the boundary
    while the left pool's value is not below the right's, then outward
    while the merged pool violates with a neighbour, as PAVA does."""
    var ls = ldi(iw, mid - 1)
    var rs = mid
    var re = ldi(iw, rs)
    if ld(uy, ls) < ld(uy, rs):
        return
    while True:
        var wl = ld(uw, ls)
        var wr = ld(uw, rs)
        var sw = fa(wl, wr)
        var swy = fmad(wr, ld(uy, rs), fm(wl, ld(uy, ls)))
        st(uy, ls, fd(swy, sw))
        st(uw, ls, sw)
        sti(iw, ls, re)
        sti(iw, re, ls)
        if ls > lo and ld(uy, ldi(iw, ls - 1)) >= ld(uy, ls):
            rs = ls
            ls = ldi(iw, ls - 1)
            continue
        if re + 1 < hi and ld(uy, ls) >= ld(uy, re + 1):
            rs = re + 1
            re = ldi(iw, rs)
            continue
        break


def if_pava_block_kernel(fw: FP, n: Int32, meta: IP, iw: IP):
    """Thread b: PAVA on the pooled points [b L, (b + 1) L)."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var lo = _gid() * IF_PAVA_L
    if lo < m:
        _pava_range(fw + nn, fw + 2 * nn, iw, lo, min(lo + IF_PAVA_L, m))


def if_pava_merge_kernel(fw: FP, n: Int32, meta: IP, iw: IP, span: Int32):
    """Thread b: the ranges [2 b span, (2 b + 1) span) and the next merged."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var sp = Int(span)
    var lo = _gid() * 2 * sp
    var mid = lo + sp
    var hi = min(lo + 2 * sp, m)
    if mid < hi:
        _pava_merge(fw + nn, fw + 2 * nn, iw, lo, mid, hi)


# ---------------------------------------------------------------- pair merge
def if_pm_flag_kernel(iw: IP, meta: IP, n: Int32, flag: IP):
    """flag[p] = 1 when p starts a pool (its end links back to it), else 0:
    the scan gives every pool its rank."""
    var m = ldi(meta, 2)
    var p = _gid()
    if p < Int(n):
        var v = 0
        if p < m:
            var e = ldi(iw, p)
            if e >= p and ldi(iw, e) == p:
                v = 1
        sti(flag, p, v)


def if_pm_merge_kernel(fw: FP, n: Int32, meta: IP, iw: IP, scan: IP, parity: Int32):
    """Thread p, a pool start whose rank (the inclusive scan of the start
    flags) has the round's parity: when its value is not below the next
    pool's, the two become one pool (mean, weight, links), the statements of
    `_pava_merge`'s one step. A pool's right neighbour has the other parity,
    so no two merges of a round touch the same pool."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var p = _gid()
    if p >= m:
        return
    var e = ldi(iw, p)
    if e < p or ldi(iw, e) != p:
        return
    if (ldi(scan, p) & 1) != Int(parity):
        return
    var rs = e + 1
    if rs >= m:
        return
    var uy = fw + nn
    var uw = fw + 2 * nn
    if ld(uy, p) < ld(uy, rs):
        return
    var re = ldi(iw, rs)
    var wl = ld(uw, p)
    var wr = ld(uw, rs)
    var sw = fa(wl, wr)
    var swy = fmad(wr, ld(uy, rs), fm(wl, ld(uy, p)))
    st(uy, p, fd(swy, sw))
    st(uw, p, sw)
    sti(iw, p, re)
    sti(iw, re, p)


def if_pm_check_kernel(fw: FP, n: Int32, meta: IP, iw: IP):
    """meta[4] = 1 when some pool still violates with its right neighbour."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var p = _gid()
    if p >= m:
        return
    var e = ldi(iw, p)
    if e < p or ldi(iw, e) != p or e + 1 >= m:
        return
    if not (ld(fw + nn, p) < ld(fw + nn, e + 1)):
        sti(meta, 4, 1)


# ---------------------------------------------------------------- output
def if_startflag_kernel(iw: IP, meta: IP, n: Int32, flag: IP):
    """flag[p] = p when p starts a pool (its end links back to it), else -1:
    the max-scan then gives every point its pool's start."""
    var m = ldi(meta, 2)
    var p = _gid()
    if p < Int(n):
        var v = -1
        if p < m:
            var e = ldi(iw, p)
            if e >= p and ldi(iw, e) == p:
                v = p
        sti(flag, p, v)


def if_fit_kernel(fw: FP, n: Int32, inc: Int32, has_lo: Int32, has_hi: Int32, fp: FP, scan: IP, meta: IP):
    """The fitted value of pool s (its PAVA pool's mean, read back through
    the reversal of a decreasing fit, clipped) into fw + 3n."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var s = _gid()
    if s >= m:
        return
    var p = s if inc != 0 else m - 1 - s
    var v = ld(fw, nn + ldi(scan, p))
    if has_lo != 0:
        v = fmax(v, ld(fp, 0))
    if has_hi != 0:
        v = fmin(v, ld(fp, 1))
    st(fw, 3 * nn + s, v)


def if_trimflag_kernel(fw: FP, n: Int32, meta: IP, flag: IP):
    """The trim: keep the ends of every run of equal fitted values."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var s = _gid()
    if s < nn:
        var keep = 0
        if s < m:
            keep = 1
            if s > 0 and s < m - 1:
                var v = ld(fw, 3 * nn + s)
                keep = 1 if (v != ld(fw, 3 * nn + s - 1) or v != ld(fw, 3 * nn + s + 1)) else 0
        sti(flag, s, keep)


def if_out_kernel(fw: FP, n: Int32, flag: IP, scan: IP, meta: IP, res: FP):
    """res: kept | X_min | X_max | xs kept.. | ys kept.. (`isotonic_fit`'s)."""
    var nn = Int(n)
    var m = ldi(meta, 2)
    var s = _gid()
    if s >= m:
        return
    if ldi(flag, s) != 0:
        var pos = ldi(scan, s) - 1
        st(res, 3 + pos, ld(fw, s))
        st(res, 3 + nn + pos, ld(fw, 3 * nn + s))
    if s == m - 1:
        st(res, 0, i2f(ldi(scan, s)))
        st(res, 1, ld(fw, 0))
        st(res, 2, ld(fw, m - 1))


def isotonic_fast(
    mut ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises -> Bool:
    """The fit on the grid; False when the shape is not the fit's or two
    unequal x under 1e-6 need the serial pooling rule (the caller then
    runs the team fit; nothing of res is trusted)."""
    if n < 1 or d != 1 or len(ip) < 4 or len(fp) < 2 or n_out < 3 + 2 * n:
        return False
    var inc = Int(ip[0])
    var has_lo = Int(ip[1])
    var has_hi = Int(ip[2])
    var has_w = Int(ip[3])
    if n_x < n or n_y < (2 * n if has_w != 0 else n):
        return False
    var nb = _ifb(n)
    var npmax = n + n // IF_PART + 2
    var dx = ctx.enqueue_create_buffer[DType.float32](n_x)
    var dy = ctx.enqueue_create_buffer[DType.float32](n_y)
    var dfp = ctx.enqueue_create_buffer[DType.float32](len(fp))
    var dout = ctx.enqueue_create_buffer[DType.float32](n_out)
    var dfw = ctx.enqueue_create_buffer[DType.float32](6 * n)
    var diw = ctx.enqueue_create_buffer[DType.int32](n)
    var dperm = ctx.enqueue_create_buffer[DType.int32](n)
    var dtmp = ctx.enqueue_create_buffer[DType.int32](n)
    var dkeys = ctx.enqueue_create_buffer[DType.uint64](n)
    var dkeys2 = ctx.enqueue_create_buffer[DType.uint64](n)
    var dflag = ctx.enqueue_create_buffer[DType.int32](npmax)
    var dscan = ctx.enqueue_create_buffer[DType.int32](npmax)
    var dblk = ctx.enqueue_create_buffer[DType.int32](_ifb(npmax))
    var dsstart = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var dpseg = ctx.enqueue_create_buffer[DType.int32](npmax)
    var dpy = ctx.enqueue_create_buffer[DType.float32](npmax)
    var dpw = ctx.enqueue_create_buffer[DType.float32](npmax)
    var dmeta = ctx.enqueue_create_buffer[DType.int32](8)
    var hfp = fp.copy()
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    dout.enqueue_fill(Float32(0))
    dfw.enqueue_fill(Float32(0))
    dmeta.enqueue_fill(Int32(0))
    dflag.enqueue_fill(Int32(0))
    dscan.enqueue_fill(Int32(0))
    # 1. the rows with a positive weight, their keys
    if has_w != 0:
        ctx.enqueue_function[if_flag_kernel](
            dy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
        _if_scan[False](ctx, dflag, dscan, dblk, n)
    ctx.enqueue_function[if_perm_kernel](
        dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(has_w), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dperm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dkeys.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    # 2. the merge sort by ranking (ping-pong; nk <= n bounds the rounds)
    var in_a = True
    var width = 1
    while width < n:
        if in_a:
            ctx.enqueue_function[if_sort_kernel](
                dkeys.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dperm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dkeys2.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dtmp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(width), grid_dim=nb, block_dim=IF_TPB)
        else:
            ctx.enqueue_function[if_sort_kernel](
                dkeys2.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dtmp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dkeys.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dperm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(width), grid_dim=nb, block_dim=IF_TPB)
        in_a = not in_a
        width *= 2
    # 3. the sorted copies
    if in_a:
        ctx.enqueue_function[if_gather_kernel](
            dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(has_w), dperm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    else:
        ctx.enqueue_function[if_gather_kernel](
            dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(has_w), dtmp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    # 4. the pools of equal x
    ctx.enqueue_function[if_segflag_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    _if_scan[False](ctx, dflag, dscan, dblk, n)
    ctx.enqueue_function[if_segstart_kernel](
        dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dsstart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=nb, block_dim=IF_TPB)
    # 5. each pool's weighted mean
    ctx.enqueue_function[if_partcount_kernel](
        dsstart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    _if_scan[False](ctx, dflag, dscan, dblk, n)
    ctx.enqueue_function[if_parttag_kernel](
        dsstart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dpseg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=nb, block_dim=IF_TPB)
    ctx.enqueue_function[if_partsum_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dsstart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dpseg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dpy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dpw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=_ifb(npmax), block_dim=IF_TPB)
    ctx.enqueue_function[if_segmean_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(inc), dsstart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dpy.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dpw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    # 6. PAVA: blocks, then the merge rounds
    ctx.enqueue_function[if_pava_block_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=_ifb((n + IF_PAVA_L - 1) // IF_PAVA_L), block_dim=IF_TPB)
    comptime if not is_defined["MOJOLEARN_ISOTONIC_FAST_PAIRMERGE_OFF"]():
        var rounds = 4
        var t2 = 1
        while t2 < n:
            t2 *= 2
            rounds += 2
        for rd in range(rounds):
            ctx.enqueue_function[if_pm_flag_kernel](
                diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
            _if_scan[False](ctx, dflag, dscan, dblk, n)
            ctx.enqueue_function[if_pm_merge_kernel](
                dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(rd & 1),
                grid_dim=nb, block_dim=IF_TPB)
        ctx.enqueue_function[if_pm_check_kernel](
            dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    else:
        var span = IF_PAVA_L
        while span < n:
            ctx.enqueue_function[if_pava_merge_kernel](
                dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(span),
                grid_dim=_ifb((n + 2 * span - 1) // (2 * span)), block_dim=IF_TPB)
            span *= 2
    # 7. the values, the clip, the trim, the output
    ctx.enqueue_function[if_startflag_kernel](
        diw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    _if_scan[True](ctx, dflag, dscan, dblk, n)
    ctx.enqueue_function[if_fit_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(inc), Int32(has_lo), Int32(has_hi), dfp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    ctx.enqueue_function[if_trimflag_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    _if_scan[False](ctx, dflag, dscan, dblk, n)
    ctx.enqueue_function[if_out_kernel](
        dfw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dscan.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dmeta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=nb, block_dim=IF_TPB)
    var hmeta = List[Int32](length=8, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.enqueue_copy(dst_ptr=hmeta.unsafe_ptr(), src_buf=dmeta)
    ctx.synchronize()
    var good = hmeta[1] == 0 and hmeta[4] == 0
    _ = hfp^
    _ = hmeta^
    _ = dx^
    _ = dy^
    _ = dfp^
    _ = dout^
    _ = dfw^
    _ = diw^
    _ = dperm^
    _ = dtmp^
    _ = dkeys^
    _ = dkeys2^
    _ = dflag^
    _ = dscan^
    _ = dblk^
    _ = dsstart^
    _ = dpseg^
    _ = dpy^
    _ = dpw^
    _ = dmeta^
    return good
