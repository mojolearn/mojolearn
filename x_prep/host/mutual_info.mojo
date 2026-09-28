# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host's neighbour searches of mutual_info (ops 68 `mi_cc`, 69 `mi_cd`,
94 `mi_dc` of x_prep/units.mojo), lane prep-cpu 2026-09-28: the device
units' words (x_prep/mutual_info.mojo) from sorted columns instead of a
brute-force scan of every row for every row.

WHAT THE UNITS WRITE. Each unit writes digamma of COUNTS: the k-th
smallest neighbour distance (rp, rs) of a point (a (primary, secondary)
pair, DEVIATION 5407, compared by `_less`), then how many points lie
`_within` it. The counts are functions of the ORDER STATISTICS of the
distance pairs, never of the order the scan met them: the unit's insertion
keeps the k smallest pairs of its candidates, and (rp, rs) enter every
later step only through `<` and `==` (a -0.0 / +0.0 secondary that a
different scan order might keep compares equal to the other). So any
search that finds the same k-th smallest pair and the same counts writes
the same words.

HOW. Distances come from the same functions (`ld`, `sub`, `abs`, `_sec`,
`_dsec`, `_less`, `_within`). On a column sorted by value, the primary
distance |x_j - x_i| is non-decreasing moving away from x_i on either side
(float32 subtraction rounds monotonically, and `ftz` keeps that). So:
- the k nearest (Ross, same class) are the first k met walking outward by
  primary, plus every point tied with the k-th primary; the k-th smallest
  PAIR of all candidates is the k-th smallest of those (the unit's own
  insertion picks it);
- the Kraskov joint (Chebyshev) search walks outward in x and stops a side
  once |dx| exceeds the current k-th primary (the joint primary is at
  least |dx|);
- a count `_within` (rp, rs) is a binary search per side for primary < rp,
  plus the points tied at primary == rp, each tested with `_within`.

A column with a non-finite value or secondary word (where an overflowed
difference could break the monotone walk) runs the device's units
unchanged, split over the pool. Checked word for word against the units by
`check_host_mi` in x_prep/seams/prep_check.mojo (ties, duplicate values,
secondary words, singleton classes, k above a class's size - 1), and by the
lane check CPU == GPU on x-prep-mutual-info and x-prep-mi-discrete.
"""
from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from x_prep.common import FP, IP, p, ld, st, key
from x_prep.prims import add, sub
from x_prep.mutual_info import (
    MAX_K, digammaf, _sec, _dsec, _less, _within, mi_cc_unit, mi_cd_unit, mi_dc_unit,
)

comptime BIG = Float32(3.4028235e38)


@always_inline
def _finite(x: Float32) -> Bool:
    return x == x and abs(x) <= BIG


def _order(vals: List[Float32]) -> List[Int]:
    """Indices of vals ascending by `key` (index on a tie)."""
    var n = len(vals)
    var ks = List[UInt64](capacity=n)
    for j in range(n):
        ks.append((UInt64(key(vals[j])) << 32) | UInt64(j))
    sort(ks)
    var out = List[Int](capacity=n)
    for j in range(n):
        out.append(Int(ks[j] & UInt64(0xFFFFFFFF)))
    return out^


struct _Sorted(Movable):
    """Values (and secondary words) of a set of points, ascending."""
    var x: List[Float32]
    var s: List[Float32]

    def __init__(out self):
        self.x = List[Float32]()
        self.s = List[Float32]()


@always_inline
def _dist(xj: Float32, xi: Float32) -> Float32:
    return abs(sub(xj, xi))


def _lower(a: _Sorted, lo: Int, hi: Int, xi: Float32) -> Int:
    """The first position in [lo, hi) whose value is not below xi."""
    var l = lo
    var h = hi
    while l < h:
        var m = (l + h) // 2
        if a.x[m] < xi:
            l = m + 1
        else:
            h = m
    return l


def _count_within(a: _Sorted, lo: Int, hi: Int, xi: Float32, si: Float32, rp: Float32, rs: Float32) -> Int:
    """How many points of a[lo:hi] are `_within` (rp, rs) of (xi, si)."""
    var mid = _lower(a, lo, hi, xi)
    var cnt = 0
    # right side [mid, hi): the primary distance rises with the position
    var l = mid
    var h = hi
    while l < h:
        var m = (l + h) // 2
        if _dist(a.x[m], xi) < rp:
            l = m + 1
        else:
            h = m
    cnt += l - mid
    var r = l
    while r < hi and _dist(a.x[r], xi) == rp:
        if _within(rp, _dsec(xi, a.x[r], si, a.s[r]), rp, rs):
            cnt += 1
        r += 1
    # left side [lo, mid): the primary distance rises as the position falls
    l = lo
    h = mid
    while l < h:
        var m = (l + h) // 2
        if _dist(a.x[m], xi) < rp:
            h = m
        else:
            l = m + 1
    cnt += mid - l
    var q = l - 1
    while q >= lo and _dist(a.x[q], xi) == rp:
        if _within(rp, _dsec(xi, a.x[q], si, a.s[q]), rp, rs):
            cnt += 1
        q -= 1
    return cnt


@always_inline
def _insert(mut bp: InlineArray[Float32, MAX_K], mut bs: InlineArray[Float32, MAX_K], kk: Int,
            dp: Float32, dsec: Float32):
    """The units' insertion of one candidate into the kk smallest pairs."""
    if _less(dp, dsec, bp[kk - 1], bs[kk - 1]):
        var m = kk - 1
        while m > 0 and _less(dp, dsec, bp[m - 1], bs[m - 1]):
            bp[m] = bp[m - 1]
            bs[m] = bs[m - 1]
            m -= 1
        bp[m] = dp
        bs[m] = dsec


def _run_split[F: def(Int) -> None](ref body: F, items: Int):
    var tasks = host_predict_task_count(items)
    if tasks <= 1:
        for i in range(items):
            body(i)
        return
    var chunk = host_predict_chunk(items, tasks)

    def task(c: Int) {imm body, imm chunk, imm items}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > items:
            hi = items
        for i in range(lo, hi):
            body(i)

    host_parallelize(task, tasks)


# ---------------------------------------------------------------- Ross (mi_cd, mi_dc)
def _cd_column(f: FP, n: Int, zb: Int, zs: Int, sb: Int, lb: Int, ls: Int, cb: Int, k: Int,
               obase: Int, ostride: Int) -> Bool:
    """`_cd_term` of every point i into f[obase + i*ostride]; False (nothing
    written) when the column holds a non-finite value or secondary word."""
    var xs = List[Float32](capacity=n)
    var ss = List[Float32](capacity=n)
    var lab = List[Int](capacity=n)
    var nlab = 0
    for j in range(n):
        var x = ld(f, zb + j * zs)
        var s = _sec(f, sb, j * zs)
        if not _finite(x) or not _finite(s):
            return False
        xs.append(x)
        ss.append(s)
        var l = Int(ld(f, lb + j * ls))
        if l < 0:
            return False
        lab.append(l)
        if l + 1 > nlab:
            nlab = l + 1
    var lcnt = List[Int](capacity=nlab)
    for l in range(nlab):
        lcnt.append(Int(ld(f, cb + l)))
    var order = _order(xs)
    # every point of a class with more than one member, by value
    var all_ = _Sorted()
    # per class, its points by value (a stable counting split of `order`)
    var start = List[Int](length=nlab + 1, fill=0)
    for j in range(n):
        start[lab[j] + 1] += 1
    for l in range(nlab):
        start[l + 1] += start[l]
    var fill = start.copy()
    var byc = _Sorted()
    byc.x = List[Float32](length=n, fill=Float32(0))
    byc.s = List[Float32](length=n, fill=Float32(0))
    var pos = List[Int](length=n, fill=0)
    for r in range(n):
        var j = order[r]
        var l = lab[j]
        if lcnt[l] > 1:
            all_.x.append(xs[j])
            all_.s.append(ss[j])
        var at = fill[l]
        fill[l] += 1
        byc.x[at] = xs[j]
        byc.s[at] = ss[j]
        pos[j] = at
    var na = len(all_.x)

    def point(i: Int) {imm f, imm xs, imm ss, imm lab, imm lcnt, imm start, imm byc, imm pos, imm all_, imm na, imm k, imm obase, imm ostride}:
        var li = lab[i]
        var cnt = lcnt[li]
        if cnt <= 1:
            st(f, obase + i * ostride, Float32(0))
            return
        var kl = k if k < cnt - 1 else cnt - 1
        var xi = xs[i]
        var si = ss[i]
        var b = start[li]
        var e = start[li + 1]
        var bp = InlineArray[Float32, MAX_K](fill=BIG)
        var bs = InlineArray[Float32, MAX_K](fill=BIG)
        var L = pos[i] - 1
        var R = pos[i] + 1
        var taken = 0
        var D = Float32(0)
        # the kl nearest by primary, then every point tied with the kl-th
        while True:
            var hasl = L >= b
            var hasr = R < e
            if not hasl and not hasr:
                break
            var dl = _dist(byc.x[L], xi) if hasl else BIG
            var dr = _dist(byc.x[R], xi) if hasr else BIG
            var take_left = hasl and (not hasr or dl <= dr)
            var dp = dl if take_left else dr
            if taken >= kl and dp > D:
                break
            if take_left:
                _insert(bp, bs, kl, dp, _dsec(xi, byc.x[L], si, byc.s[L]))
                L -= 1
            else:
                _insert(bp, bs, kl, dp, _dsec(xi, byc.x[R], si, byc.s[R]))
                R += 1
            taken += 1
            D = dp
        var rp = bp[kl - 1]
        var rs = bs[kl - 1]
        var mall = _count_within(all_, 0, na, xi, si, rp, rs)
        st(f, obase + i * ostride,
           sub(sub(digammaf(Float32(kl)), digammaf(Float32(cnt))), digammaf(Float32(mall))))

    _run_split(point, n)
    return True


def mi_cd_host_stage(total: Int, f: FP, q: IP):
    """A whole `mi_cd` stage: q = [Z, n, d, Y, LABCNT, k, TERM, ZS]."""
    var n = p(q, 1)
    var d = p(q, 2)
    if n <= 0 or d <= 0:
        return
    for c in range(d):
        var sb = p(q, 7) + c if p(q, 7) > 0 else 0
        if not _cd_column(f, n, p(q, 0) + c, d, sb, p(q, 3), 1, p(q, 4), p(q, 5), p(q, 6) + c, d):
            def unit(i: Int) {imm f, imm q, imm d, imm c}:
                mi_cd_unit(i * d + c, f, q)
            _run_split(unit, n)


def mi_dc_host_stage(total: Int, f: FP, q: IP):
    """A whole `mi_dc` stage: q = [ZY, n, d, XC, CNT, KS, k, TERM, ZYS]."""
    var n = p(q, 1)
    var d = p(q, 2)
    if n <= 0 or d <= 0:
        return
    for c in range(d):
        if not _cd_column(f, n, p(q, 0), 1, p(q, 8), p(q, 3) + c, d, p(q, 4) + c * p(q, 5), p(q, 6),
                          p(q, 7) + c, d):
            def unit(i: Int) {imm f, imm q, imm d, imm c}:
                mi_dc_unit(i * d + c, f, q)
            _run_split(unit, n)


# ---------------------------------------------------------------- Kraskov (mi_cc)
def _cc_column(f: FP, q: IP, c: Int, ys: _Sorted, ysx: List[Float32], yss: List[Float32]) -> Bool:
    """`mi_cc_unit` of every point of column c; False (nothing written) on a
    non-finite value or secondary word in the column."""
    var n = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 4)
    var zs = p(q, 6)
    var xs = List[Float32](capacity=n)
    var sx = List[Float32](capacity=n)
    for j in range(n):
        var x = ld(f, p(q, 0) + j * d + c)
        var s = _sec(f, zs, j * d + c)
        if not _finite(x) or not _finite(s):
            return False
        xs.append(x)
        sx.append(s)
    var order = _order(xs)
    var ax = _Sorted()
    var ay = List[Float32](capacity=n)
    var asy = List[Float32](capacity=n)
    var pos = List[Int](length=n, fill=0)
    for r in range(n):
        var j = order[r]
        ax.x.append(xs[j])
        ax.s.append(sx[j])
        ay.append(ysx[j])
        asy.append(yss[j])
        pos[j] = r
    var term = p(q, 5)

    def point(i: Int) {imm f, imm xs, imm sx, imm ysx, imm yss, imm ax, imm ay, imm asy, imm pos, imm ys, imm n, imm d, imm k, imm c, imm term}:
        var xi = xs[i]
        var sxi = sx[i]
        var yi = ysx[i]
        var syi = yss[i]
        var bp = InlineArray[Float32, MAX_K](fill=BIG)
        var bs = InlineArray[Float32, MAX_K](fill=BIG)
        var L = pos[i] - 1
        var R = pos[i] + 1
        var goL = L >= 0
        var goR = R < n
        while goL or goR:
            var dl = _dist(ax.x[L], xi) if goL else BIG
            var dr = _dist(ax.x[R], xi) if goR else BIG
            var take_left = goL and (not goR or dl <= dr)
            var r = L if take_left else R
            var dx = dl if take_left else dr
            if dx > bp[k - 1]:
                break  # both sides are at least this far in x from here on
            var dy = _dist(ay[r], yi)
            var sxd = _dsec(xi, ax.x[r], sxi, ax.s[r])
            var syd = _dsec(yi, ay[r], syi, asy[r])
            var dp = dx
            var dsec = sxd
            if _less(dx, sxd, dy, syd):
                dp = dy
                dsec = syd
            _insert(bp, bs, k, dp, dsec)
            if take_left:
                L -= 1
                goL = L >= 0
            else:
                R += 1
                goR = R < n
        var rp = bp[k - 1]
        var rs = bs[k - 1]
        var nx = _count_within(ax, 0, n, xi, sxi, rp, rs)
        var ny = _count_within(ys, 0, n, yi, syi, rp, rs)
        st(f, term + i * d + c, add(digammaf(Float32(nx)), digammaf(Float32(ny))))

    _run_split(point, n)
    return True


def mi_cc_host_stage(total: Int, f: FP, q: IP):
    """A whole `mi_cc` stage: q = [Z, n, d, Y, k, TERM, ZS, YS]."""
    var n = p(q, 1)
    var d = p(q, 2)
    if n <= 0 or d <= 0:
        return
    var ysx = List[Float32](capacity=n)
    var yss = List[Float32](capacity=n)
    var yfinite = True
    for j in range(n):
        var y = ld(f, p(q, 3) + j)
        var s = _sec(f, p(q, 7), j)
        if not _finite(y) or not _finite(s):
            yfinite = False
        ysx.append(y)
        yss.append(s)
    var ys = _Sorted()
    if yfinite:
        var order = _order(ysx)
        for r in range(n):
            ys.x.append(ysx[order[r]])
            ys.s.append(yss[order[r]])
    for c in range(d):
        if not yfinite or not _cc_column(f, q, c, ys, ysx, yss):
            def unit(i: Int) {imm f, imm q, imm d, imm c}:
                mi_cc_unit(i * d + c, f, q)
            _run_split(unit, n)
