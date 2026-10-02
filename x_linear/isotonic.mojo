# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IsotonicRegression (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/isotonic.py` (`IsotonicRegression._build_y`:
rows ordered by (X, y) -- the order comes from Python --, `_make_unique`,
`isotonic_regression` with the order reversed for a decreasing fit and the
clip to [y_min, y_max], then the trim that keeps the first and last of each
run of equal fitted values) and `sklearn/_isotonic.pyx`
(`_make_unique`: equal X within float32 resolution 1e-6 pooled by weighted
mean; `_inplace_contiguous_isotonic_regression`: the single-pass
pool-adjacent-violators with backtracking), and scipy's `interp1d(kind=
'linear')` for predict (searchsorted left, clamped to [1, m-1],
slope * (x - x_lo) + y_lo). THIS PASS IS SEQUENTIAL; the brief's parallel
PAVA (a prefix-scan formulation) is pass 2's speed work. Out-of-bounds
'nan' writes the canonical quiet NaN word 0x7FC00000 (a constant, never a
computed NaN, so its bits are the same on every target).
"""
from std.memory import bitcast
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fmax, fmin, ld, st, ldi, sti, i2f, copy
from x_linear.team import Team

comptime OOB_NAN = 0
comptime OOB_CLIP = 1


@always_inline
def _iso_less(x: FP, y: FP, a: Int, b: Int) -> Bool:
    """Row a before row b: by x, then y, then the row index (a total order,
    so every sort of it is the stable sort by (x, y))."""
    var xa = ld(x, a)
    var xb = ld(x, b)
    if xa != xb:
        return xa < xb
    var ya = ld(y, a)
    var yb = ld(y, b)
    if ya != yb:
        return ya < yb
    return a < b


def iso_merge_passes(x: FP, y: FP, perm: IP, tmp: IP, lo: Int, hi: Int, width0: Int):
    """Bottom-up merge sort of perm[lo, hi) by `_iso_less`, starting from
    sorted runs of `width0` (1 = unsorted); the result lands in `perm`."""
    var width = width0
    var src = perm
    var dst = tmp
    var flipped = False
    while width < hi - lo:
        var s = lo
        while s < hi:
            var mid = s + width
            if mid > hi:
                mid = hi
            var e = s + 2 * width
            if e > hi:
                e = hi
            var i = s
            var j = mid
            var k = s
            while i < mid and j < e:
                var ri = ldi(src, i)
                var rj = ldi(src, j)
                if _iso_less(x, y, rj, ri):
                    sti(dst, k, rj)
                    j += 1
                else:
                    sti(dst, k, ri)
                    i += 1
                k += 1
            while i < mid:
                sti(dst, k, ldi(src, i))
                i += 1
                k += 1
            while j < e:
                sti(dst, k, ldi(src, j))
                j += 1
                k += 1
            s = e
        var sw = src
        src = dst
        dst = sw
        flipped = not flipped
        width *= 2
    if flipped:
        for i in range(lo, hi):
            sti(perm, i, ldi(src, i))


def iso_fit_sorted(x: FP, y: FP, n: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, perm: IP, nk: Int):
    """The fit after its sort: perm[0, nk) the rows in (x, y, row) order. The
    host column's entry is `isotonic_fit_host` (x_linear/isotonic_host.mojo,
    the sort on host threads); the device runs these steps as grid kernels
    (x_linear/device.mojo `_iso_fit_grid`: the same gathers, starts, group
    chains and PAVA statements, so the same words)."""
    var has_w = ldi(ip, 3) != 0
    var xs = fw + 3 * n
    var ys = fw + 4 * n
    var ws = fw + 5 * n
    for j in range(nk):
        iso_gather_one(j, x, y, n, has_w, perm, xs, ys, ws)
    var starts = iw + n
    var m = iso_bounds(xs, nk, starts)
    for g in range(m):
        iso_group(g, xs, ys, ws, starts, fw, n)
    iso_after_unique(m, n, ip, fp, res, fw, iw)


@always_inline
def iso_gather_one(j: Int, x: FP, y: FP, n: Int, has_w: Bool, perm: IP, xs: FP, ys: FP, ws: FP):
    var r = ldi(perm, j)
    st(xs, j, ld(x, r))
    st(ys, j, ld(y, r))
    st(ws, j, ld(y, n + r) if has_w else Float32(1))


def iso_bounds_from(xs: FP, nk: Int, cand: IP, nc: Int, starts: IP) -> Int:
    """`iso_bounds` over the candidate rows cand[0:nc] (ascending): the rows
    whose x differs from the row before. A row with the previous row's x
    never starts a group (it is 0 above the group's first x or the same
    distance as the row before), so walking only the candidates takes the
    same steps and writes the same starts (lane/neural-pass117)."""
    var m = 0
    sti(starts, 0, 0)
    var cx = ld(xs, 0)
    for q in range(nc):
        var j = ldi(cand, q)
        var xj = ld(xs, j)
        if fs(xj, cx) >= Float32(1e-6):
            m += 1
            sti(starts, m, j)
            cx = xj
    m += 1
    sti(starts, m, nk)
    return m


def iso_bounds(xs: FP, nk: Int, starts: IP) -> Int:
    """_make_unique's groups: starts[g] the first sorted row of group g (a row
    starts a group when it is at least 1e-6 above the group's first x),
    starts[m] = nk; returns m."""
    var m = 0
    sti(starts, 0, 0)
    var cx = ld(xs, 0)
    for j in range(1, nk):
        var xj = ld(xs, j)
        if fs(xj, cx) >= Float32(1e-6):
            m += 1
            sti(starts, m, j)
            cx = xj
    m += 1
    sti(starts, m, nk)
    return m


@always_inline
def iso_group(g: Int, xs: FP, ys: FP, ws: FP, starts: IP, fw: FP, n: Int):
    """Group g's pooled x, weight and mean: _make_unique's chains (the first
    group summed from zero, every other started by its first row)."""
    var lo = ldi(starts, g)
    var hi = ldi(starts, g + 1)
    var cw = Float32(0)
    var cy = Float32(0)
    var j0 = lo
    if g > 0:
        cw = ld(ws, lo)
        cy = fm(ld(ys, lo), cw)
        j0 = lo + 1
    var j = j0
    # 16 rows' loads before their folds (scheduling only; lane/neural-pass117:
    # one group can hold most of the rows, and its chain waited on a load
    # per row)
    while j + 16 <= hi:
        var bw = SIMD[DType.float32, 16]()
        var by = SIMD[DType.float32, 16]()
        comptime for u in range(16):
            bw[u] = ld(ws, j + u)
            by[u] = ld(ys, j + u)
        comptime for u in range(16):
            cw = fa(cw, bw[u])
            cy = fmad(by[u], bw[u], cy)
        j += 16
    while j < hi:
        var wj = ld(ws, j)
        cw = fa(cw, wj)
        cy = fmad(ld(ys, j), wj, cy)
        j += 1
    st(fw, g, ld(xs, lo))
    st(fw, 2 * n + g, cw)
    st(fw, n + g, fd(cy, cw))


def iso_after_unique(m: Int, n: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """isotonic_fit after _make_unique: PAVA (reversed for a decreasing fit),
    the clip and the trim."""
    var inc = ldi(ip, 0) != 0
    var ux = 0
    var uy = n
    var uw = 2 * n
    # a decreasing fit runs PAVA on the reversed sequence
    if not inc:
        for a in range(m // 2):
            var b = m - 1 - a
            var sv = ld(fw, uy + a)
            st(fw, uy + a, ld(fw, uy + b))
            st(fw, uy + b, sv)
            sv = ld(fw, uw + a)
            st(fw, uw + a, ld(fw, uw + b))
            st(fw, uw + b, sv)
    # _inplace_contiguous_isotonic_regression
    for i in range(m):
        sti(iw, i, i)
    var i = 0
    while i < m:
        var k = ldi(iw, i) + 1
        if k == m:
            break
        if ld(fw, uy + i) < ld(fw, uy + k):
            i = k
            continue
        var swy = fm(ld(fw, uw + i), ld(fw, uy + i))
        var sw = ld(fw, uw + i)
        while True:
            var prev_y = ld(fw, uy + k)
            swy = fmad(ld(fw, uw + k), ld(fw, uy + k), swy)
            sw = fa(sw, ld(fw, uw + k))
            k = ldi(iw, k) + 1
            if k == m or prev_y < ld(fw, uy + k):
                st(fw, uy + i, fd(swy, sw))
                st(fw, uw + i, sw)
                sti(iw, i, k - 1)
                sti(iw, k - 1, i)
                if i > 0:
                    i = ldi(iw, i - 1)
                break
    i = 0
    while i < m:
        var k = ldi(iw, i) + 1
        for j in range(i + 1, k):
            st(fw, uy + j, ld(fw, uy + i))
        i = k
    if not inc:
        for a in range(m // 2):
            var b = m - 1 - a
            var sv = ld(fw, uy + a)
            st(fw, uy + a, ld(fw, uy + b))
            st(fw, uy + b, sv)
    if ldi(ip, 1) != 0 or ldi(ip, 2) != 0:
        for j in range(m):
            var v = ld(fw, uy + j)
            if ldi(ip, 1) != 0:
                v = fmax(v, ld(fp, 0))
            if ldi(ip, 2) != 0:
                v = fmin(v, ld(fp, 1))
            st(fw, uy + j, v)
    # the trim: keep the ends of every run of equal values
    var kept = 0
    for j in range(m):
        var keep = True
        if j > 0 and j < m - 1:
            var v = ld(fw, uy + j)
            keep = v != ld(fw, uy + j - 1) or v != ld(fw, uy + j + 1)
        if keep:
            st(res, 3 + kept, ld(fw, ux + j))
            st(res, 3 + n + kept, ld(fw, uy + j))
            kept += 1
    st(res, 0, i2f(kept))
    st(res, 1, ld(fw, ux))
    st(res, 2, ld(fw, ux + m - 1))


def isotonic_predict(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """x: the n query points; y: xs m | ys m. ip: [m, out_of_bounds (0 nan, 1 clip)];
    fp: [X_min, X_max]. res: n predictions. Queries dealt across the team."""
    var m = ldi(ip, 0)
    var oob = ldi(ip, 1)
    # DEVIATION 5006 (IDENTITY_PATHS row 106): the constant word, never 0/0
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    for q in range(t.tid, n, t.nt):
        iso_predict_one(q, x, y, m, oob, fp, res)


@always_inline
def iso_predict_one(q: Int, x: FP, y: FP, m: Int, oob: Int, fp: FP, res: FP):
    """isotonic_predict for query q (lane/neural-pass107: the device runs one
    thread a query)."""
    # DEVIATION 5006 (IDENTITY_PATHS row 106): the constant word, never 0/0
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var tq = ld(x, q)
    if oob == OOB_CLIP:
        tq = fmin(fmax(tq, ld(fp, 0)), ld(fp, 1))
    if m == 1:
        st(res, q, ld(y, m))
        return
    if tq < ld(y, 0) or tq > ld(y, m - 1):
        st(res, q, nan)
        return
    # searchsorted(xs, t, 'left'), clamped to [1, m - 1]
    var lo = 0
    var hi = m
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(y, mid) < tq:
            lo = mid + 1
        else:
            hi = mid
    var idx = lo
    if idx < 1:
        idx = 1
    if idx > m - 1:
        idx = m - 1
    var x_lo = ld(y, idx - 1)
    var x_hi = ld(y, idx)
    var y_lo = ld(y, m + idx - 1)
    var y_hi = ld(y, m + idx)
    var slope = fd(fs(y_hi, y_lo), fs(x_hi, x_lo))
    st(res, q, fmad(slope, fs(tq, x_lo), y_lo))
