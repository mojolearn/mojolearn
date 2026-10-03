# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""STL's passes as ONE THREAD PER OUTPUT POINT over every series of the batch
(lane/apple-fast-tsa2, `-D MOJOLEARN_TSA2_STL`; FAST + Apple only, reached
from `sequence/pyapi.mojo::stl_py` under that define). `sequence/stl.mojo`
runs a whole series on one thread: at the board's shape (64 series of 1,440
hourly points, period 24, five inner passes) every LOESS point is a
dependent step of one thread, 359 ms on the M3 against statsmodels' 119 ms.

Here each inner pass is seven launches over the batch, queued with no wait
between them (`stl_py` waits once, after the last):
  * `op_stl_seas`: a thread per slot of the extended seasonal series
    `[B, n + 2 np]`: (subseries j, position m) computes `_ss`'s LOESS value
    at that position (the `_ess` point for 1 <= m <= k, the two
    extrapolated ends for m = 0 and m = k + 1), detrending its window's
    points on the fly (`y - trend`, the same subtraction `op_stl` stores).
  * `op_stl_ma` x3: the three moving averages of the low-pass filter, each
    output a direct ascending sum of its window (the reference's running
    sum is a serial recurrence; a direct sum of the same window is FAST's
    fold of it).
  * `op_stl_loess`: `_ess` with jump 1 as a thread per point (the low-pass
    LOESS, then the trend LOESS).
  * `op_stl_deseas`: season = extended seasonal - low pass, and the
    deseasonalised series the trend LOESS reads.
`op_stl_finish` writes the residual and the unit robustness weights once.

THE LOESS POINT IS `stl_est`'S CHAIN. `stl_est` stores the tricube weights
in a scratch row, normalises them, applies the linear correction, then
folds. A thread here keeps no row: it recomputes each weight by the same
operations in each pass (`_wraw`, then the quotient by the same sum), so
every value folded is the stored one's bits, in the stored order. The
window [nleft, nright] of a point is `_ess`'s sliding window written in
closed form for jump 1 (the only jump this file serves; other jumps and
the robust outer passes keep `op_stl`).
"""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_sqrt


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _wraw(
    j: Int, xs: Int, h: Float32, h9: Float32, h1: Float32,
    rw: FP, rwbase: Int, rwstride: Int, userw: Bool,
) -> Float32:
    """`stl_est`'s first-loop weight of window point j (0 past h9)."""
    var r = abs(Float32(j + 1 - xs))
    if r <= h9:
        var wj: Float32
        if r <= h1:
            wj = Float32(1.0)
        else:
            var q = _div(r, h)
            var u = sub(Float32(1.0), mul(mul(q, q), q))
            wj = mul(mul(u, u), u)
        if userw:
            wj = mul(wj, ld(rw, rwbase + j * rwstride))
        return wj
    return Float32(0.0)


@always_inline
def _val(src: FP, sbase: Int, sstride: Int, dsub: FP, use_sub: Bool, j: Int) -> Float32:
    """Window point j of the series this thread smooths: `src[sbase + j
    sstride]`, minus `dsub` at the same slot when `use_sub` (the detrended
    cycle-subseries, `op_stl`'s `sub(ld(y, i), ld(trend, i))`)."""
    var v = ld(src, sbase + j * sstride)
    if use_sub:
        return sub(v, ld(dsub, sbase + j * sstride))
    return v


def stl_est_point(
    src: FP, sbase: Int, sstride: Int, dsub: FP, use_sub: Bool,
    n: Int, len_: Int, ideg: Int, xs: Int, nleft: Int, nright: Int,
    rw: FP, rwbase: Int, rwstride: Int, userw: Bool,
) -> Tuple[Bool, Float32]:
    """`sequence/stl.mojo::stl_est` for one point, no scratch row: the
    weights are recomputed per pass by the same operations."""
    var rng = Float32(n) - Float32(1.0)
    var h = Float32(max(xs - nleft, nright - xs))
    if len_ > n:
        h = add(h, Float32((len_ - n) // 2))
    var h9 = mul(Float32(0.999), h)
    var h1 = mul(Float32(0.001), h)
    var a = Float32(0.0)
    for j in range(nleft - 1, nright):
        var r = abs(Float32(j + 1 - xs))
        if r <= h9:
            a = add(a, _wraw(j, xs, h, h9, h1, rw, rwbase, rwstride, userw))
    if a <= Float32(0.0):
        return (False, Float32(0.0))
    var lin = False
    var a2 = Float32(0.0)
    var b = Float32(0.0)
    if h > Float32(0.0) and ideg > 0:
        for j in range(nleft - 1, nright):
            var w = _div(_wraw(j, xs, h, h9, h1, rw, rwbase, rwstride, userw), a)
            a2 = fma3(w, Float32(j + 1), a2)
        b = sub(Float32(xs), a2)
        var c = Float32(0.0)
        for j in range(nleft - 1, nright):
            var w = _div(_wraw(j, xs, h, h9, h1, rw, rwbase, rwstride, userw), a)
            var d = sub(Float32(j + 1), a2)
            c = fma3(w, mul(d, d), c)
        if ftz(identical_sqrt(c)) > mul(Float32(0.001), rng):
            b = _div(b, c)
            lin = True
    var ys = Float32(0.0)
    for j in range(nleft - 1, nright):
        var w = _div(_wraw(j, xs, h, h9, h1, rw, rwbase, rwstride, userw), a)
        if lin:
            w = mul(w, fma3(b, sub(Float32(j + 1), a2), Float32(1.0)))
        ys = fma3(w, _val(src, sbase, sstride, dsub, use_sub, j), ys)
    return (True, ys)


@always_inline
def _ess_window(i: Int, n: Int, len_: Int) -> Tuple[Int, Int]:
    """`stl_ess`'s (nleft, nright) at point i (0-based) for jump 1: the
    whole series when len_ >= n, else the window of len_ points that
    starts at 1 and slides right once the point passes (len_ + 2) // 2,
    until it ends at n."""
    if len_ >= n:
        return (1, n)
    var nsh = (len_ + 2) // 2
    var shift = i + 1 - nsh
    if shift < 0:
        shift = 0
    if shift > n - len_:
        shift = n - len_
    return (1 + shift, len_ + shift)


@always_inline
def _ess_point(
    src: FP, sbase: Int, sstride: Int, dsub: FP, use_sub: Bool,
    n: Int, len_: Int, ideg: Int, i: Int, rw: FP, rwbase: Int, rwstride: Int, userw: Bool,
) -> Float32:
    """`stl_ess`'s output at point i (jump 1): the LOESS value, or the
    point itself when every weight is zero."""
    if n < 2:
        return _val(src, sbase, sstride, dsub, use_sub, 0)
    var w = _ess_window(i, n, len_)
    var r = stl_est_point(src, sbase, sstride, dsub, use_sub, n, len_, ideg, i + 1, w[0], w[1],
                          rw, rwbase, rwstride, userw)
    if r[0]:
        return r[1]
    return _val(src, sbase, sstride, dsub, use_sub, i)


def op_stl_seas(t: Int, a: Args):
    """Slot t of the extended seasonal series [B, n2], n2 = n + 2 np: series
    b = t // n2, slot q = t % n2 = m np + j (subseries j, position m,
    0 <= m <= k + 1 with k the subseries' length; the slots of a series
    are exactly its n2 positions). p0 y [B, n], p1 trend [B, n], p2 rw
    [B, n], p3 out [B, n2]; i0 n, i1 np, i2 ns, i3 isdeg, i4 n2, i5 userw.
    The value is `stl_ss`'s: the `_ess` point for 1 <= m <= k, the LOESS at
    xs = 0 over the first min(ns, k) points for m = 0 (falling back to the
    point m = 1 as `stl_ss` does), the LOESS at xs = k + 1 over the last
    min(ns, k) points for m = k + 1 (falling back to the point m = k)."""
    var n = a.i0
    var np_ = a.i1
    var ns = a.i2
    var isdeg = a.i3
    var n2 = a.i4
    var userw = a.i5 != 0
    var b = t // n2
    var q = t - b * n2
    var m = q // np_
    var j = q - m * np_
    var k = (n - (j + 1)) // np_ + 1
    var sbase = b * n + j
    var src = a.p0
    var dsub = a.p1
    var rw = a.p2
    var v: Float32
    if m >= 1 and m <= k:
        v = _ess_point(src, sbase, np_, dsub, True, k, ns, isdeg, m - 1, rw, sbase, np_, userw)
    elif m == 0:
        var r0 = stl_est_point(src, sbase, np_, dsub, True, k, ns, isdeg, 0, 1, min(ns, k),
                               rw, sbase, np_, userw)
        if r0[0]:
            v = r0[1]
        else:
            v = _ess_point(src, sbase, np_, dsub, True, k, ns, isdeg, 0, rw, sbase, np_, userw)
    else:
        var r1 = stl_est_point(src, sbase, np_, dsub, True, k, ns, isdeg, k + 1, max(1, k - ns + 1), k,
                               rw, sbase, np_, userw)
        if r1[0]:
            v = r1[1]
        else:
            v = _ess_point(src, sbase, np_, dsub, True, k, ns, isdeg, k - 1, rw, sbase, np_, userw)
    st(a.p3, t, v)


def op_stl_ma(t: Int, a: Args):
    """Point t of the moving average: series b = t // i0, output i = t % i0
    = the ascending sum of p0[b i2 + i, ..., + i1 - 1] over i1 (`stl_ma`'s
    window), divided by i1, into p1[b i3 + i]. i0 out_len, i1 len,
    i2 source stride, i3 destination stride."""
    var out_len = a.i0
    var len_ = a.i1
    var b = t // out_len
    var i = t - b * out_len
    var base = b * a.i2 + i
    var v = Float32(0.0)
    for u in range(len_):
        v = add(v, ld(a.p0, base + u))
    st(a.p1, b * a.i3 + i, _div(v, Float32(len_)))


def op_stl_loess(t: Int, a: Args):
    """Point t of `stl_ess` (jump 1) over series b = t // i0 of p0 (stride
    i3), length i0, window i1, degree i2, into p1 (stride i4); p2 the
    robustness weights (stride i0) when i5."""
    var n = a.i0
    var b = t // n
    var i = t - b * n
    var v = _ess_point(a.p0, b * a.i3, 1, a.p0, False, n, a.i1, a.i2, i, a.p2, b * n, 1, a.i5 != 0)
    st(a.p1, b * a.i4 + i, v)


def op_stl_deseas(t: Int, a: Args):
    """Point t of season [B, n] = ext[b n2 + np + i] - lowpass[b n2 + i], and
    of the deseasonalised series p4[b n2 + i] = y - season (`op_stl`'s two
    stores). p0 y, p1 ext, p2 lowpass, p3 season, p4 deseasonalised;
    i0 n, i1 np, i2 n2."""
    var n = a.i0
    var b = t // n
    var i = t - b * n
    var s = sub(ld(a.p1, b * a.i2 + a.i1 + i), ld(a.p2, b * a.i2 + i))
    st(a.p3, t, s)
    st(a.p4, b * a.i2 + i, sub(ld(a.p0, t), ftz(s)))


def op_stl_finish(t: Int, a: Args):
    """Element t: p4 resid = p0 y - p1 season - p2 trend; p3 weights = 1."""
    st(a.p4, t, sub(sub(ld(a.p0, t), ld(a.p1, t)), ld(a.p2, t)))
    st(a.p3, t, Float32(1.0))
