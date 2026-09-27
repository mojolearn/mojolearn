# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exponential smoothing state-space models with an additive, optionally
damped, trend and an optional additive or multiplicative season:
ETS(A|M, N|A|Ad, N|A|M), as statsforecast states them
(`python/statsforecast/ets.py` `ets_f` / `etsmodel` / `initparam` /
`initstate`; `src/ets.cpp` `Update`, `Forecast`, `Calc`,
`ObjectiveFunction`, `Optimize`): the likelihood criterion
n log(sum e^2) (+ 2 sum log|f| for multiplicative errors) over the smoothing
parameters AND the initial states, minimised by statsforecast's Nelder-Mead
(`sequence/nm.mojo`) inside the usual bounds (alpha, beta, gamma in
[1e-4, 0.9999] with beta <= the initial alpha and gamma <= 1 - the initial
alpha, phi in [0.8, 0.98]). Float32, one series per element.

Initial states (`initstate`): with a season of period m, the seasonal
indices come from a classical decomposition when n >= 3m (statsmodels
`seasonal_decompose`: the centred 2 x m moving average, the per-position
means of the detrended series, centred to sum 0 or mean 1), else from a
least-squares fit of [1, t, cos(2 pi t / m), sin(2 pi t / m)] (Householder
QR here, `np.linalg.lstsq` there). The m - 1 free seasonal states are the
indices m - 1 .. 1 (reversed); the last state is m [M] - their sum. The
level and trend are the least-squares line through the first
min(max(10, 2m), n) seasonally adjusted observations.

Pinned spellings (sequence/README.md, DEVIATIONS 5517-5518):
- 5517 the seasonal update s0 = s_old + gamma (t - s_old), one fma
  (alt: (1 - gamma) s_old + gamma t).
- 5518 the decomposition's moving average: the taps summed in index order
  with the half-weight ends as 0.5 x, then ONE division by m (alt: every
  tap times 1/m).
The seasonal states live in a ring (head index), so a step is O(1): the
reference's shift `s[1:] = old_s[:-1]` is the head moving back one slot.

Not carried here (sequence/NOT_IMPLEMENTED.tsv): multiplicative trend,
automatic model selection (the 'Z' letters), the admissible / both bound
checks (the reference's objective does not apply them either), non-normal
errors, prediction intervals."""
from std.memory import bitcast

from sequence.nm import Objective, nelder_mead
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_cos, identical_div, identical_log, identical_pow, identical_sin, identical_sqrt

comptime ERR_A = 0
comptime ERR_M = 1
comptime SEAS_N = 0
comptime SEAS_A = 1
comptime SEAS_M = 2
comptime NM_CAP = 64          # Nelder-Mead coordinates < 64: m <= 58
comptime HUGE_N = Float32(1e10)
comptime TOL = Float32(1e-10)


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _log(x: Float32) -> Float32:
    return ftz(identical_log(x))


@always_inline
def _pos_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def ets_scratch(n: Int, m: Int) -> Int:
    """Floats of scratch one series needs (op_ets's p3 row stride, i8)."""
    var k = 6 + (m - 1 if m > 1 else 0)
    var kc = k if k > 8 else 8
    return 3 * kc + (k + 1) * k + (k + 1) + 4 * k + 2 * m + 6 * n + 8


@always_inline
def seas_update(sl: Float32, gamma: Float32, tt: Float32) -> Float32:
    """5517: s0 = s_old + gamma (t - s_old), one fma."""
    return fma3(gamma, sub(tt, sl), sl)


@always_inline
def ets_lik(y: FP, n: Int, err: Int, trend: Bool, season: Int, m: Int, ring: FP,
            alpha: Float32, beta: Float32, gamma: Float32, phi: Float32,
            l0: Float32, b0: Float32) -> Tuple[Float32, Float32, Float32, Int]:
    """(lik, final level, final trend, ring head) of Calc. `ring` holds the
    m initial seasonal states in order (head 0) and is updated in place;
    the final state j is ring[(head + j) % m]."""
    var l = l0
    var b = b0 if trend else Float32(0.0)
    var sse = Float32(0.0)
    var slog = Float32(0.0)
    var ba = div(beta, alpha)
    var head = 0
    for i in range(n):
        var ol = l
        var ob = b
        var phib = mul(phi, ob) if trend else Float32(0.0)
        var q = add(ol, phib)
        var f0 = q
        var sl = Float32(0.0)
        var last = 0
        if season != SEAS_N:
            last = head + m - 1
            if last >= m:
                last -= m
            sl = ld(ring, last)
            f0 = add(q, sl) if season == SEAS_A else mul(q, sl)
        var yi = ld(y, i)
        var e: Float32
        if err == ERR_A:
            e = sub(yi, f0)
        else:
            var fd = add(f0, Float32(1e-10)) if abs(f0) < Float32(1e-10) else f0
            e = div(sub(yi, f0), fd)
        var p = yi
        if season == SEAS_A:
            p = sub(yi, sl)
        elif season == SEAS_M:
            p = HUGE_N if abs(sl) < TOL else div(yi, sl)
        l = fma3(alpha, sub(p, q), q)
        if trend:
            b = fma3(ba, sub(sub(l, ol), phib), phib)
        if season != SEAS_N:
            var tt: Float32
            if season == SEAS_A:
                tt = sub(yi, q)
            else:
                tt = HUGE_N if abs(q) < TOL else div(yi, q)
            head = last
            st(ring, head, seas_update(sl, gamma, tt))
        sse = fma3(e, e, sse)
        var v = abs(f0)
        slog = add(slog, _log(v) if v > Float32(0.0) else _log(add(v, Float32(1e-8))))
    var lik: Float32
    if sse > Float32(0.0):
        lik = mul(Float32(n), _log(sse))
    else:
        lik = mul(Float32(n), _log(add(sse, Float32(1e-8))))
    if err == ERR_M:
        lik = fma3(Float32(2.0), slog, lik)
    return (lik, l, b, head)


@always_inline
def fill_ring(ring: FP, xs: FP, m: Int, season: Int) -> Bool:
    """The m initial seasonal states from the m - 1 free ones: the last is
    m [M] - their sum (ascending). False when a multiplicative state is
    negative (the reference's objective is then +inf)."""
    var acc = Float32(0.0)
    var ok = True
    for k in range(m - 1):
        var v = ld(xs, k)
        st(ring, k, v)
        acc = add(acc, v)
        if v < Float32(0.0):
            ok = False
    var lastv = sub(Float32(m) if season == SEAS_M else Float32(0.0), acc)
    st(ring, m - 1, lastv)
    if lastv < Float32(0.0):
        ok = False
    return ok or season != SEAS_M


struct EtsObj(Objective):
    var y: FP
    var n: Int
    var err: Int
    var trend: Bool
    var season: Int
    var m: Int
    var ring: FP
    var oa: Bool
    var ob: Bool
    var og: Bool
    var op: Bool
    var alpha: Float32
    var beta: Float32
    var gamma: Float32
    var phi: Float32

    @always_inline
    def __init__(out self, y: FP, n: Int, err: Int, trend: Bool, season: Int, m: Int, ring: FP,
                 oa: Bool, ob: Bool, og: Bool, op: Bool,
                 alpha: Float32, beta: Float32, gamma: Float32, phi: Float32):
        self.y = y
        self.n = n
        self.err = err
        self.trend = trend
        self.season = season
        self.m = m
        self.ring = ring
        self.oa = oa
        self.ob = ob
        self.og = og
        self.op = op
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi

    @always_inline
    def n_smooth(self) -> Int:
        return Int(self.oa) + Int(self.ob) + Int(self.og) + Int(self.op)

    @always_inline
    def unpack(self, x: FP) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
        """(alpha, beta, gamma, phi, l0, b0)."""
        var j = 0
        var a = self.alpha
        var b = self.beta
        var g = self.gamma
        var p = self.phi
        if self.oa:
            a = ld(x, j)
            j += 1
        if self.ob:
            b = ld(x, j)
            j += 1
        if self.og:
            g = ld(x, j)
            j += 1
        if self.op:
            p = ld(x, j)
            j += 1
        var l0 = ld(x, j)
        var b0 = ld(x, j + 1) if self.trend else Float32(0.0)
        return (a, b, g, p, l0, b0)

    @always_inline
    def seasons(self, x: FP) -> FP:
        return x + self.n_smooth() + 1 + Int(self.trend)

    @always_inline
    def eval(mut self, x: FP) -> Float32:
        var u = self.unpack(x)
        if self.season != SEAS_N:
            if not fill_ring(self.ring, self.seasons(x), self.m, self.season):
                return _pos_inf()
        var r = ets_lik(self.y, self.n, self.err, self.trend, self.season, self.m, self.ring,
                        u[0], u[1], u[2], u[3], u[4], u[5])
        return r[0] if r[0] > Float32(-1e10) else Float32(-1e10)


@always_inline
def _moving_average(y: FP, i: Int, m: Int) -> Float32:
    """5518: statsmodels' centred moving average at i (2 x m for an even m):
    the taps in index order, the half-weight ends as 0.5 x, one division."""
    var half = m // 2
    var acc: Float32
    if m % 2 == 0:
        acc = mul(Float32(0.5), ld(y, i - half))
        for k in range(1, m):
            acc = add(acc, ld(y, i - half + k))
        acc = fma3(Float32(0.5), ld(y, i + half), acc)
    else:
        acc = ld(y, i - half)
        for k in range(1, m):
            acc = add(acc, ld(y, i - half + k))
    return div(acc, Float32(m))


@always_inline
def season_decompose(y: FP, n: Int, m: Int, season: Int, pa: FP):
    """statsmodels seasonal_decompose's period averages (n >= 2m): the
    detrended values at each position mod m averaged ascending over the
    indices where the moving average exists, then centred (sum 0, A) or
    normalised (mean 1, M)."""
    var half = m // 2
    for p in range(m):
        var acc = Float32(0.0)
        var cnt = 0
        var i = p
        while i < half:
            i += m
        while i <= n - 1 - half:
            var tr = _moving_average(y, i, m)
            var d = sub(ld(y, i), tr) if season == SEAS_A else div(ld(y, i), tr)
            acc = add(acc, d)
            cnt += 1
            i += m
        st(pa, p, div(acc, Float32(cnt)))
    var mean = Float32(0.0)
    for p in range(m):
        mean = add(mean, ld(pa, p))
    mean = div(mean, Float32(m))
    for p in range(m):
        st(pa, p, sub(ld(pa, p), mean) if season == SEAS_A else div(ld(pa, p), mean))


@always_inline
def fourier_fit(y: FP, n: Int, m: Int, qa: FP) -> Tuple[Float32, Float32]:
    """(c0, c1) of the least-squares fit of y on [1, t, cos, sin], t = 1..n,
    angle 2 pi (t mod m) / m: Householder QR, columns in order, sums
    ascending, then back substitution. qa holds 4n + n + 8 floats."""
    var rhs = qa + 4 * n
    var dg = rhs + n
    var cf = dg + 4
    var twopi = Float32(6.2831853071795864769)
    for i in range(n):
        var t = i + 1
        var ang = mul(twopi, div(Float32(t % m), Float32(m)))
        st(qa, 4 * i, Float32(1.0))
        st(qa, 4 * i + 1, Float32(t))
        st(qa, 4 * i + 2, ftz(identical_cos(ang)))
        st(qa, 4 * i + 3, ftz(identical_sin(ang)))
        st(rhs, i, ld(y, i))
    for j in range(4):
        var ss = Float32(0.0)
        for i in range(j, n):
            var v = ld(qa, 4 * i + j)
            ss = fma3(v, v, ss)
        var nrm = ftz(identical_sqrt(ss))
        if nrm == Float32(0.0):
            st(dg, j, Float32(0.0))
            continue
        var x0 = ld(qa, 4 * j + j)
        var al = sub(Float32(0.0), nrm) if x0 >= Float32(0.0) else nrm
        st(qa, 4 * j + j, sub(x0, al))
        var vtv = Float32(0.0)
        for i in range(j, n):
            var v = ld(qa, 4 * i + j)
            vtv = fma3(v, v, vtv)
        for c in range(j + 1, 5):
            var dot = Float32(0.0)
            for i in range(j, n):
                var w = ld(qa, 4 * i + c) if c < 4 else ld(rhs, i)
                dot = fma3(ld(qa, 4 * i + j), w, dot)
            var f = div(add(dot, dot), vtv)     # H = I - 2 v v' / v'v
            for i in range(j, n):
                var v = ld(qa, 4 * i + j)
                if c < 4:
                    st(qa, 4 * i + c, fma3(sub(Float32(0.0), f), v, ld(qa, 4 * i + c)))
                else:
                    st(rhs, i, fma3(sub(Float32(0.0), f), v, ld(rhs, i)))
        st(dg, j, al)
    for jj in range(4):
        var j = 3 - jj
        var s = ld(rhs, j)
        for k in range(j + 1, 4):
            s = fma3(sub(Float32(0.0), ld(qa, 4 * j + k)), ld(cf, k), s)
        var d = ld(dg, j)
        st(cf, j, div(s, d) if d != Float32(0.0) else Float32(0.0))
    return (ld(cf, 0), ld(cf, 1))


@always_inline
def ets_init_state(y: FP, n: Int, trend: Bool, season: Int, m: Int, sinit: FP, seas: FP, qa: FP
                   ) -> Tuple[Float32, Float32]:
    """statsforecast initstate: (l0, b0); the m - 1 free seasonal states go
    to sinit. `seas` (n floats) and `qa` (5n + 8) are scratch."""
    var maxn = 10 if n > 10 else n
    var ysa = y
    if season != SEAS_N:
        if n < 3 * m:
            var c = fourier_fit(y, n, m, qa)
            for i in range(n):
                var lin = add(c[0], mul(c[1], Float32(i + 1)))
                st(seas, i, sub(ld(y, i), lin) if season == SEAS_A else div(ld(y, i), lin))
        else:
            season_decompose(y, n, m, season, qa)
            for i in range(n):
                st(seas, i, ld(qa, i % m))
        for k in range(m - 1):
            st(sinit, k, ld(seas, m - 1 - k))
        if season == SEAS_M:
            var sm = Float32(0.0)
            for k in range(m - 1):
                var v = ld(sinit, k)
                v = v if v > Float32(1e-2) else Float32(1e-2)
                st(sinit, k, v)
                sm = add(sm, v)
            if sm > Float32(m):
                var den = Float32(0.0)
                for k in range(m - 1):
                    den = add(den, add(ld(sinit, k), Float32(1e-2)))
                for k in range(m - 1):
                    st(sinit, k, div(ld(sinit, k), den))
        # the seasonally adjusted series, over the first maxn points only
        var mx = 2 * m if 2 * m > 10 else 10
        maxn = mx if mx < n else n
        for i in range(maxn):
            var s = ld(seas, i)
            if season == SEAS_A:
                st(seas, i, sub(ld(y, i), s))
            else:
                st(seas, i, div(ld(y, i), s if s > Float32(1e-2) else Float32(1e-2)))
        ysa = seas
    var l0: Float32
    var b0 = Float32(0.0)
    var sy = Float32(0.0)
    for i in range(maxn):
        sy = add(sy, ld(ysa, i))
    var ybar = div(sy, Float32(maxn))
    if trend:
        var tbar = div(Float32(maxn + 1), Float32(2.0))
        var sxy = Float32(0.0)
        var sxx = Float32(0.0)
        for i in range(maxn):
            var dt = sub(Float32(i + 1), tbar)
            sxy = fma3(dt, sub(ld(ysa, i), ybar), sxy)
            sxx = fma3(dt, dt, sxx)
        b0 = div(sxy, sxx) if sxx > Float32(0.0) else Float32(0.0)
        l0 = sub(ybar, mul(b0, tbar))
        if abs(add(l0, b0)) < Float32(1e-8):
            l0 = mul(l0, Float32(1.001))
            b0 = mul(b0, Float32(0.999))
    else:
        l0 = ybar
    return (l0, b0)


def op_ets(t: Int, a: Args):
    """Series t. p0 y [B, n]; p1 forecast [B, h] out; p2 info [B, 10] out
    (alpha, beta, phi, l0, b0, lik, iterations, parameter count, gamma, 0);
    p3 scratch [B, i8]; p4 initial seasonal states [B, m] out (season only).
    i0 n, i1 h, i2 error (0 A, 1 M), i3 trend (0 N, 1 A), i4 damped,
    i5 fixed mask (1 alpha, 2 beta, 4 phi, 8 gamma), i6 season (0 N, 1 A,
    2 M), i7 m, i8 scratch stride (ets_scratch); f0 alpha, f1 beta, f2 phi,
    f3 gamma (read where fixed)."""
    var n = a.i0
    var h = a.i1
    var y = a.p0 + t * n
    var trend = a.i3 == 1
    var damped = trend and a.i4 != 0
    var season = a.i6
    var m = a.i7 if season != SEAS_N else 1
    var kmax = 6 + (m - 1 if season != SEAS_N else 0)
    var kc = kmax if kmax > 8 else 8
    var sc = a.p3 + t * a.i8
    var x = sc
    var lo = x + kc
    var hi = lo + kc
    var nm_scr = hi + kc
    var ring = nm_scr + (kmax + 1) * kmax + (kmax + 1) + 4 * kmax
    var sinit = ring + m
    var seas = sinit + m
    var qa = seas + n
    # initparam (usual bounds)
    var lo_a = Float32(1e-4)
    var hi_a = Float32(0.9999)
    var fa = (a.i5 & 1) != 0
    var fb = (a.i5 & 2) != 0
    var fp = (a.i5 & 4) != 0
    var fg = (a.i5 & 8) != 0
    var alpha: Float32
    if fa:
        alpha = a.f0
    elif season != SEAS_N:
        alpha = add(lo_a, div(mul(Float32(0.2), sub(hi_a, lo_a)), Float32(m)))
    else:
        alpha = fma3(Float32(0.2), sub(hi_a, lo_a), lo_a)
    var hi_b = hi_a if hi_a < alpha else alpha
    var beta = a.f1 if fb else fma3(Float32(0.1), sub(hi_b, lo_a), lo_a)
    var one_a = sub(Float32(1.0), alpha)
    var hi_g = hi_a if hi_a < one_a else one_a
    var gamma = Float32(0.0)
    if season != SEAS_N:
        gamma = a.f3 if fg else fma3(Float32(0.05), sub(hi_g, lo_a), lo_a)
    var phi = Float32(1.0)
    if damped:
        phi = a.f2 if fp else fma3(Float32(0.99), sub(Float32(0.98), Float32(0.8)), Float32(0.8))
    var st0 = ets_init_state(y, n, trend, season, m, sinit, seas, qa)
    var oa = not fa
    var ob = trend and not fb
    var og = season != SEAS_N and not fg
    var op = damped and not fp
    var k = 0
    if oa:
        st(x, k, alpha)
        st(lo, k, lo_a)
        st(hi, k, hi_a)
        k += 1
    if ob:
        st(x, k, beta)
        st(lo, k, lo_a)
        st(hi, k, hi_b)
        k += 1
    if og:
        st(x, k, gamma)
        st(lo, k, lo_a)
        st(hi, k, hi_g)
        k += 1
    if op:
        st(x, k, phi)
        st(lo, k, Float32(0.8))
        st(hi, k, Float32(0.98))
        k += 1
    st(x, k, st0[0])
    st(lo, k, Float32(-3.0e38))
    st(hi, k, Float32(3.0e38))
    k += 1
    if trend:
        st(x, k, st0[1])
        st(lo, k, Float32(-3.0e38))
        st(hi, k, Float32(3.0e38))
        k += 1
    if season != SEAS_N:
        for j in range(m - 1):
            st(x, k, ld(sinit, j))
            st(lo, k, Float32(-3.0e38))
            st(hi, k, Float32(3.0e38))
            k += 1
    var obj = EtsObj(y, n, a.i2, trend, season, m, ring, oa, ob, og, op, alpha, beta, gamma, phi)
    var it = nelder_mead[EtsObj, NM_CAP](obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4))
    var u = obj.unpack(x)
    if season != SEAS_N:
        _ = fill_ring(ring, obj.seasons(x), m, season)
        for j in range(m):
            st(a.p4, t * m + j, ld(ring, j))
    var r = ets_lik(y, n, a.i2, trend, season, m, ring, u[0], u[1], u[2], u[3], u[4], u[5])
    var l = r[1]
    var b = r[2]
    var head = r[3]
    var phistar = u[3]
    for i in range(h):
        var f = add(l, mul(phistar, b)) if trend else l
        if season != SEAS_N:
            var j = m - 1 - (i % m)
            var ix = head + j
            if ix >= m:
                ix -= m
            f = add(f, ld(ring, ix)) if season == SEAS_A else mul(f, ld(ring, ix))
        st(a.p1, t * h + i, f)
        if i < h - 1:
            phistar = add(phistar, ftz(identical_pow(u[3], Float32(i + 1))))
    var info = a.p2 + t * 10
    st(info, 0, u[0])
    st(info, 1, u[1] if trend else Float32(0.0))
    st(info, 2, u[3])
    st(info, 3, u[4])
    st(info, 4, u[5])
    st(info, 5, r[0])
    st(info, 6, Float32(it))
    st(info, 7, Float32(k))
    st(info, 8, u[2])
    st(info, 9, Float32(0.0))


def op_ets_lik(t: Int, a: Args):
    """The seam probe for 5517 (and the recursion): series t's Calc at given
    parameters. p0 y [B, n]; p1 parameters [B, 6 + m] (alpha, beta, gamma,
    phi, l0, b0, the m seasonal states); p2 out [B, 3 + m] (lik, final
    level, final trend, the final seasonal states in order); p3 ring
    scratch [B, m]. i0 n, i2 error, i3 trend (0 N, 1 A), i6 season, i7 m."""
    var n = a.i0
    var season = a.i6
    var m = a.i7 if season != SEAS_N else 1
    var y = a.p0 + t * n
    var pr = a.p1 + t * (6 + m)
    var ring = a.p3 + t * m
    for j in range(m):
        st(ring, j, ld(pr, 6 + j))
    var r = ets_lik(y, n, a.i2, a.i3 == 1, season, m, ring, ld(pr, 0), ld(pr, 1), ld(pr, 2), ld(pr, 3),
                    ld(pr, 4), ld(pr, 5))
    var o = a.p2 + t * (3 + m)
    st(o, 0, r[0])
    st(o, 1, r[1])
    st(o, 2, r[2])
    for j in range(m):
        var ix = r[3] + j
        if ix >= m:
            ix -= m
        st(o, 3 + j, ld(ring, ix) if season != SEAS_N else Float32(0.0))


def op_ets_init(t: Int, a: Args):
    """The seam probe for 5518 / 5519: series t's initstate. p0 y [B, n];
    p1 out [B, 1 + m] (l0, b0, the m - 1 free seasonal states); p3 scratch
    [B, i8] (6n + 8). i0 n, i3 trend, i6 season, i7 m."""
    var n = a.i0
    var season = a.i6
    var m = a.i7 if season != SEAS_N else 1
    var y = a.p0 + t * n
    var o = a.p1 + t * (1 + m)
    var seas = a.p3 + t * a.i8
    var qa = seas + n
    var r = ets_init_state(y, n, a.i3 == 1, season, m, o + 2, seas, qa)
    st(o, 0, r[0])
    st(o, 1, r[1])
