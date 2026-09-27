# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""STL, Seasonal-Trend decomposition by Loess (Cleveland et al. 1990), as
statsmodels `statsmodels/tsa/stl/_stl.pyx` (0.15) states it: `_onestp`,
`_ss`, `_fts`, `_ma`, `_ess`, `_est`, `_rwts`, the same loops in the same
order, in float32 (no float64 on the device). One series per element: the
GPU runs a batch of series one thread each, the host loops over them, the
body is this file.

Deviations from the reference, each value-preserving up to float32:
  * `_est` returns a validity flag where the reference returns NaN (no
    computed NaN reaches an output, IDENTITY_PATHS Clause B); the callers'
    `isnan` fallbacks read the flag;
  * `_rwts`'s `np.partition` is a heapsort of a copy (the two middle order
    statistics are values, so any correct selection gives the same bits);
  * the constants .999 and .001 are float32.
"""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_sqrt


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def stl_est(
    y: FP, n: Int, len_: Int, ideg: Int, xs: Int, nleft: Int, nright: Int,
    w: FP, userw: Bool, rw: FP,
) -> Tuple[Bool, Float32]:
    var rng = Float32(n) - Float32(1.0)
    var h = Float32(max(xs - nleft, nright - xs))
    if len_ > n:
        h = add(h, Float32((len_ - n) // 2))
    var h9 = mul(Float32(0.999), h)
    var h1 = mul(Float32(0.001), h)
    var a = Float32(0.0)
    for j in range(nleft - 1, nright):
        var wj = Float32(0.0)
        var r = abs(Float32(j + 1 - xs))
        if r <= h9:
            if r <= h1:
                wj = Float32(1.0)
            else:
                var q = div(r, h)
                var u = sub(Float32(1.0), mul(mul(q, q), q))
                wj = mul(mul(u, u), u)
            if userw:
                wj = mul(wj, ld(rw, j))
            a = add(a, wj)
        st(w, j, wj)
    if a <= Float32(0.0):
        return (False, Float32(0.0))
    for j in range(nleft - 1, nright):
        st(w, j, div(ld(w, j), a))
    if h > Float32(0.0) and ideg > 0:
        a = Float32(0.0)
        for j in range(nleft - 1, nright):
            a = fma3(ld(w, j), Float32(j + 1), a)
        var b = sub(Float32(xs), a)
        var c = Float32(0.0)
        for j in range(nleft - 1, nright):
            var d = sub(Float32(j + 1), a)
            c = fma3(ld(w, j), mul(d, d), c)
        if ftz(identical_sqrt(c)) > mul(Float32(0.001), rng):
            b = div(b, c)
            for j in range(nleft - 1, nright):
                st(w, j, mul(ld(w, j), fma3(b, sub(Float32(j + 1), a), Float32(1.0))))
    var ys = Float32(0.0)
    for j in range(nleft - 1, nright):
        ys = fma3(ld(w, j), ld(y, j), ys)
    return (True, ys)


def stl_ess(
    y: FP, n: Int, len_: Int, ideg: Int, njump: Int, userw: Bool, rw: FP, ys: FP, res: FP,
):
    if n < 2:
        st(ys, 0, ld(y, 0))
        return
    var newnj = min(njump, n - 1)
    var nleft = 0
    var nright = 0
    if len_ >= n:
        nleft = 1
        nright = n
        var i = 0
        while i < n:
            var r = stl_est(y, n, len_, ideg, i + 1, nleft, nright, res, userw, rw)
            st(ys, i, r[1] if r[0] else ld(y, i))
            i += newnj
    elif newnj == 1:
        var nsh = (len_ + 2) // 2
        nleft = 1
        nright = len_
        for i in range(n):
            if (i + 1) > nsh and nright != n:
                nleft += 1
                nright += 1
            var r = stl_est(y, n, len_, ideg, i + 1, nleft, nright, res, userw, rw)
            st(ys, i, r[1] if r[0] else ld(y, i))
    else:
        var nsh = (len_ + 1) // 2
        var i = 0
        while i < n:
            if (i + 1) < nsh:
                nleft = 1
                nright = len_
            elif (i + 1) >= (n - nsh + 1):
                nleft = n - len_ + 1
                nright = n
            else:
                nleft = i + 1 - nsh + 1
                nright = len_ + i + 1 - nsh
            var r = stl_est(y, n, len_, ideg, i + 1, nleft, nright, res, userw, rw)
            st(ys, i, r[1] if r[0] else ld(y, i))
            i += newnj
    if newnj == 1:
        return
    var i = 0
    while i < n - newnj:
        var delta = div(sub(ld(ys, i + newnj), ld(ys, i)), Float32(newnj))
        for j in range(i, i + newnj):
            st(ys, j, fma3(delta, Float32(j - i), ld(ys, i)))
        i += newnj
    var k = ((n - 1) // newnj) * newnj + 1
    if k != n:
        var r = stl_est(y, n, len_, ideg, n, nleft, nright, res, userw, rw)
        st(ys, n - 1, r[1] if r[0] else ld(y, n - 1))
        if k != n - 1:
            var delta = div(sub(ld(ys, n - 1), ld(ys, k - 1)), Float32(n - k))
            for j in range(k, n):
                st(ys, j, fma3(delta, Float32(j + 1 - k), ld(ys, k - 1)))


def stl_ma(x: FP, n: Int, len_: Int, ave: FP):
    var newn = n - len_ + 1
    var flen = Float32(len_)
    var v = Float32(0.0)
    for i in range(len_):
        v = add(v, ld(x, i))
    st(ave, 0, div(v, flen))
    var k = len_
    var m = 0
    for j in range(1, newn):
        v = add(v, sub(ld(x, k), ld(x, m)))
        st(ave, j, div(v, flen))
        k += 1
        m += 1


def stl_ss(
    y: FP, n: Int, np_: Int, ns: Int, isdeg: Int, nsjump: Int, userw: Bool, rw: FP,
    season: FP, work1: FP, work2: FP, work3: FP, work4: FP,
):
    for j in range(np_):
        var k = (n - (j + 1)) // np_ + 1
        for i in range(k):
            st(work1, i, ld(y, i * np_ + j))
        if userw:
            for i in range(k):
                st(work3, i, ld(rw, i * np_ + j))
        stl_ess(work1, k, ns, isdeg, nsjump, userw, work3, work2 + 1, work4)
        var nright = min(ns, k)
        var r0 = stl_est(work1, k, ns, isdeg, 0, 1, nright, work4, userw, work3)
        st(work2, 0, r0[1] if r0[0] else ld(work2, 1))
        var nleft = max(1, k - ns + 1)
        var r1 = stl_est(work1, k, ns, isdeg, k + 1, nleft, k, work4, userw, work3)
        st(work2, k + 1, r1[1] if r1[0] else ld(work2, k))
        for m in range(k + 2):
            st(season, m * np_ + j, ld(work2, m))


def _sift(a: FP, start: Int, end: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child > end:
            return
        var sw = root
        if a.unsafe_load(sw) < a.unsafe_load(child):
            sw = child
        if child + 1 <= end and a.unsafe_load(sw) < a.unsafe_load(child + 1):
            sw = child + 1
        if sw == root:
            return
        var t = a.unsafe_load(root)
        a.unsafe_store(root, a.unsafe_load(sw))
        a.unsafe_store(sw, t)
        root = sw


def heapsort(a: FP, n: Int):
    var start = (n - 2) // 2
    while start >= 0:
        _sift(a, start, n - 1)
        start -= 1
    var end = n - 1
    while end > 0:
        var t = a.unsafe_load(end)
        a.unsafe_store(end, a.unsafe_load(0))
        a.unsafe_store(0, t)
        end -= 1
        _sift(a, 0, end)


def stl_rwts(y: FP, n: Int, fit: FP, rw: FP, sortbuf: FP):
    for i in range(n):
        var r = abs(sub(ld(y, i), ld(fit, i)))
        st(rw, i, r)
        sortbuf.unsafe_store(i, r)
    heapsort(sortbuf, n)
    var m0 = n // 2
    var m1 = n - m0 - 1
    var cmad = mul(Float32(3.0), add(sortbuf.unsafe_load(m0), sortbuf.unsafe_load(m1)))
    if cmad == Float32(0.0):
        for i in range(n):
            st(rw, i, Float32(1.0))
        return
    var c9 = mul(Float32(0.999), cmad)
    var c1 = mul(Float32(0.001), cmad)
    for i in range(n):
        var r = ld(rw, i)
        if r <= c1:
            st(rw, i, Float32(1.0))
        elif r <= c9:
            var q = div(r, cmad)
            var u = sub(Float32(1.0), mul(q, q))
            st(rw, i, mul(u, u))
        else:
            st(rw, i, Float32(0.0))


def op_stl(t: Int, a: Args):
    """Series t. p0 y [B, n], p1 season, p2 trend, p3 weights, p4 resid
    [B, n] out; p5 work [B, 5 (n + 2 np)], p6 sort scratch [B, n].
    i0 n, i1 np, i2 ns, i3 nt, i4 nl, i5 degrees (s + 2 t + 4 l),
    i6 seasonal_jump, i7 trend_jump, i8 low_pass_jump, i9 inner, i10 outer."""
    var n = a.i0
    var np_ = a.i1
    var ns = a.i2
    var nt = a.i3
    var nl = a.i4
    var isdeg = a.i5 & 1
    var itdeg = (a.i5 >> 1) & 1
    var ildeg = (a.i5 >> 2) & 1
    var n2 = n + 2 * np_
    var y = a.p0 + t * n
    var season = a.p1 + t * n
    var trend = a.p2 + t * n
    var rw = a.p3 + t * n
    var resid = a.p4 + t * n
    var work = a.p5 + t * 5 * n2
    var w0 = work
    var w1 = work + n2
    var w2 = work + 2 * n2
    var w3 = work + 3 * n2
    var w4 = work + 4 * n2
    var sortbuf = a.p6 + t * n
    for i in range(n):
        st(season, i, Float32(0.0))
        st(trend, i, Float32(0.0))
        st(rw, i, Float32(1.0))
    var use_rw = False
    var k = 0
    while True:
        for _ in range(a.i9):
            for i in range(n):
                st(w0, i, sub(ld(y, i), ld(trend, i)))
            stl_ss(w0, n, np_, ns, isdeg, a.i6, use_rw, rw, w1, w2, w3, w4, season)
            stl_ma(w1, n2, np_, w2)
            stl_ma(w2, n2 - np_ + 1, np_, w0)
            stl_ma(w0, n2 - 2 * np_ + 2, 3, w2)
            stl_ess(w2, n, nl, ildeg, a.i8, False, w3, w0, w4)
            for i in range(n):
                st(season, i, sub(ld(w1, np_ + i), ld(w0, i)))
                st(w0, i, sub(ld(y, i), ld(season, i)))
            stl_ess(w0, n, nt, itdeg, a.i7, use_rw, rw, trend, w2)
        k += 1
        if k > a.i10:
            break
        for i in range(n):
            st(w0, i, add(ld(trend, i), ld(season, i)))
        stl_rwts(y, n, w0, rw, sortbuf)
        use_rw = True
    for i in range(n):
        st(resid, i, sub(sub(ld(y, i), ld(season, i)), ld(trend, i)))
