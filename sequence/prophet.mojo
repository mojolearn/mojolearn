# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A Prophet-style forecaster: the linear-growth model of Prophet
(`prophet/forecaster.py`, `stan/prophet.stan`): the piecewise-linear trend
(k + A delta) t + (m - A (t_change delta)) with the changepoint matrix A of
`get_changepoint_matrix`, Fourier seasonality and holiday indicator features
X with coefficients beta, additive or multiplicative
(`trend (1 + X_sm beta) + X_sa beta`), and the MAP estimate under Prophet's
priors (k, m ~ N(0, 5), delta ~ Laplace(0, tau), sigma_obs ~ N(0, 0.5),
beta ~ N(0, sigmas)) with Prophet's scaling (y / max|y|, t to [0, 1]) and
initialisation (the line through the first and last points).

The fit is ours: Stan optimises by its L-BFGS; here an L-BFGS (memory 5,
two-loop recursion, backtracking Armijo line search, the subgradient
sign(delta)/tau of the Laplace prior) in float32, the log of sigma as the
free coordinate, one series per GPU thread over shared t and X. Parity with
the prophet package is at a tolerance (brief).

Features: sin(2 pi (i+1) frac), cos(...) with frac = (t mod P) / P computed
by the caller in float64 (exact fmod, one rounding) and the trigonometry by
the portable float32 seams, so the features are the same bits everywhere."""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_cos, identical_div, identical_exp, identical_sin, identical_sqrt

comptime TWO_PI: Float32 = 6.283185307179586
comptime MEM = 5
#: points per block of the likelihood's gradient folds (_fg_data)
comptime PB = 16


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def op_prophet_features(t: Int, a: Args):
    """Row t of X [N, K]: for each seasonality s (i0 of them, p1[s] its
    order) sin / cos pairs of 2 pi (i+1) frac[t, s] (p0 [N, ns]), then the
    i1 holiday columns of p2 [N, nh]; p3 X out; i2 K."""
    var ns = a.i0
    var nh = a.i1
    var K = a.i2
    var col = 0
    for s in range(ns):
        var order = Int(a.p1.unsafe_load(s))
        var f = ld(a.p0, t * ns + s)
        for i in range(order):
            var c = mul(mul(TWO_PI, Float32(i + 1)), f)
            st(a.p3, t * K + col, ftz(identical_sin(c)))
            st(a.p3, t * K + col + 1, ftz(identical_cos(c)))
            col += 2
    for h in range(nh):
        st(a.p3, t * K + col + h, ld(a.p2, t * nh + h))


struct ProphetData:
    var t: FP        # [N] scaled time
    var X: FP        # [N, K]
    var cp: FP       # [S] changepoint times (scaled)
    var sig: FP      # [K] prior scales
    var N: Int
    var K: Int
    var S: Int
    var tau: Float32
    var mult: Bool

    def __init__(out self, t: FP, X: FP, cp: FP, sig: FP, N: Int, K: Int, S: Int, tau: Float32, mult: Bool):
        self.t = t
        self.X = X
        self.cp = cp
        self.sig = sig
        self.N = N
        self.K = K
        self.S = S
        self.tau = tau
        self.mult = mult


@always_inline
def _fg_data(d: ProphetData, y: FP, th: FP, g: FP, lo: Int, hi: Int) -> Float32:
    """The likelihood's points lo .. hi - 1 (ascending): their gradient terms
    folded into g (the caller zeroes it) and their sum of squared residuals
    (returned, from zero)."""
    var S = d.S
    var K = d.K
    var k = ld(th, 0)
    var m = ld(th, 1)
    var u = ld(th, 2 + S)
    var sigma = ftz(identical_exp(u))
    var s2 = mul(sigma, sigma)
    var sse = Float32(0.0)
    # POINT BLOCKS (apple2): every gradient coordinate's chain is still over
    # the points in ascending order, one fma or add per point, but the
    # running value stays in a register for PB points instead of a load and
    # a store of g per point (a device-memory round trip on every term; ld / st
    # only flush, and every op flushes already, so the words are the same).
    # The per-point terms (ti, tr, se, w) are the same expressions.
    var i = lo
    while i < hi:
        var nb = min(PB, hi - i)
        var tiv = SIMD[DType.float32, PB]()
        var wtv = SIMD[DType.float32, PB]()
        var wbv = SIMD[DType.float32, PB]()
        for b in range(nb):
            var ti = ld(d.t, i + b)
            var tr = fma3(k, ti, m)
            for j in range(S):
                var c = ld(d.cp, j)
                if ti >= c:
                    tr = fma3(ld(th, 2 + j), sub(ti, c), tr)
            var se = Float32(0.0)
            for q in range(K):
                se = fma3(ld(d.X, (i + b) * K + q), ld(th, 3 + S + q), se)
            var yhat: Float32
            if d.mult:
                yhat = fma3(tr, se, tr)
            else:
                yhat = add(tr, se)
            var r = sub(ld(y, i + b), yhat)
            sse = fma3(r, r, sse)
            var w = div(-r, s2)
            tiv[b] = ti
            wtv[b] = mul(w, add(Float32(1.0), se)) if d.mult else w
            wbv[b] = mul(w, tr) if d.mult else w
        var g0 = ld(g, 0)
        var g1 = ld(g, 1)
        for b in range(nb):
            g0 = fma3(wtv[b], tiv[b], g0)
            g1 = add(g1, wtv[b])
        st(g, 0, g0)
        st(g, 1, g1)
        for j in range(S):
            var c = ld(d.cp, j)
            var acc = ld(g, 2 + j)
            for b in range(nb):
                if tiv[b] >= c:
                    acc = fma3(wtv[b], sub(tiv[b], c), acc)
            st(g, 2 + j, acc)
        for q in range(K):
            var acc = ld(g, 3 + S + q)
            for b in range(nb):
                acc = fma3(wbv[b], ld(d.X, (i + b) * K + q), acc)
            st(g, 3 + S + q, acc)
        i += nb
    return sse


@always_inline
def _fg_prior(d: ProphetData, th: FP, g: FP, sse: Float32) -> Float32:
    """The priors and the sigma terms on top of the likelihood's gradient in
    g and its sse; returns the objective."""
    var S = d.S
    var K = d.K
    var k = ld(th, 0)
    var m = ld(th, 1)
    var u = ld(th, 2 + S)
    var sigma = ftz(identical_exp(u))
    var s2 = mul(sigma, sigma)
    var f = add(div(mul(k, k), Float32(50.0)), div(mul(m, m), Float32(50.0)))
    st(g, 0, add(ld(g, 0), div(k, Float32(25.0))))
    st(g, 1, add(ld(g, 1), div(m, Float32(25.0))))
    for j in range(S):
        var dl = ld(th, 2 + j)
        f = add(f, div(abs(dl), d.tau))
        var sg = Float32(0.0)
        if dl > Float32(0.0):
            sg = Float32(1.0)
        elif dl < Float32(0.0):
            sg = Float32(-1.0)
        st(g, 2 + j, add(ld(g, 2 + j), div(sg, d.tau)))
    for q in range(K):
        var b = ld(th, 3 + S + q)
        var sq = mul(ld(d.sig, q), ld(d.sig, q))
        f = add(f, div(mul(b, b), mul(Float32(2.0), sq)))
        st(g, 3 + S + q, add(ld(g, 3 + S + q), div(b, sq)))
    # sigma: prior N(0, 0.5), likelihood N log sigma + sse / (2 sigma^2)
    f = add(f, div(s2, Float32(0.5)))
    f = fma3(Float32(d.N), u, f)
    f = add(f, div(sse, mul(Float32(2.0), s2)))
    st(g, 2 + S, add(sub(add(div(s2, Float32(0.25)), Float32(d.N)), div(sse, s2)), Float32(0.0)))
    return f


def prophet_fg(d: ProphetData, y: FP, th: FP, g: FP) -> Float32:
    """-log posterior (up to a constant) at th and its gradient into g.
    th = [k, m, delta (S), log sigma, beta (K)]. (apple2: the likelihood
    loop and the priors are two inlined helpers, the same operations in the
    same order, so FAST's parallel likelihood can share them.)"""
    var P = 3 + d.S + d.K
    for j in range(P):
        st(g, j, Float32(0.0))
    var sse = _fg_data(d, y, th, g, 0, d.N)
    return _fg_prior(d, th, g, sse)


# ------------------------------------------------------------------ FAST
def op_prophet_fg_part(t: Int, a: Args):
    """FAST (apple2): chunk t of i4 chunks of the likelihood. p0 scaled y
    [N]; p1 t; p2 X; p3 changepoints; p4 prior scales; p5 th [P]; p6 parts
    [i4, P + 1] out (the chunk's gradient terms, then its sse). i0 N, i1 K,
    i2 S, i3 multiplicative; f0 tau."""
    var N = a.i0
    var P = 3 + a.i2 + a.i1
    var C = a.i4
    var ch = (N + C - 1) // C
    var lo = t * ch
    var hi = min(N, lo + ch)
    var row = a.p6 + t * (P + 1)
    for j in range(P):
        st(row, j, Float32(0.0))
    var d = ProphetData(a.p1, a.p2, a.p3, a.p4, N, a.i1, a.i2, a.f0, a.i3 != 0)
    var sse = _fg_data(d, a.p0, a.p5, row, lo, hi) if lo < hi else Float32(0.0)
    st(row, P, sse)


def op_prophet_fg_sum(t: Int, a: Args):
    """FAST: p7[t] = sum over the i4 chunks of p6[c, t], in order; i5 = P + 1."""
    var s = Float32(0.0)
    for c in range(a.i4):
        s = add(s, ld(a.p6, c * a.i5 + t))
    st(a.p7, t, s)


@always_inline
def _dot(a: FP, b: FP, n: Int) -> Float32:
    var s = Float32(0.0)
    for i in range(n):
        s = fma3(ld(a, i), ld(b, i), s)
    return s


def lbfgs_prophet(d: ProphetData, y: FP, th: FP, w: FP, max_iter: Int) -> Tuple[Float32, Int]:
    """Minimise prophet_fg from th (in place). w: scratch of
    (6 + 2 MEM) P + 2 MEM floats. Returns (f, iterations)."""
    var P = 3 + d.S + d.K
    var g = w
    var dvec = g + P
    var thn = dvec + P
    var gn = thn + P
    var q = gn + P
    var sm = q + P
    var ym = sm + MEM * P
    var rho = ym + MEM * P
    var al = rho + MEM
    var f = prophet_fg(d, y, th, g)
    var npairs = 0
    var head = 0
    var small = 0
    var it = 0
    while it < max_iter:
        # two-loop recursion: q = g; newest to oldest, then oldest to newest
        for i in range(P):
            st(q, i, ld(g, i))
        var idx = head
        for _ in range(npairs):
            idx = (idx - 1 + MEM) % MEM
            var aa = mul(ld(rho, idx), _dot(sm + idx * P, q, P))
            st(al, idx, aa)
            for i in range(P):
                st(q, i, sub(ld(q, i), mul(aa, ld(ym + idx * P, i))))
        var gamma = Float32(1.0)
        if npairs > 0:
            var last = (head - 1 + MEM) % MEM
            var yy = _dot(ym + last * P, ym + last * P, P)
            if yy > Float32(0.0):
                gamma = div(_dot(sm + last * P, ym + last * P, P), yy)
        else:
            var gg = ftz(identical_sqrt(_dot(g, g, P)))
            if gg > Float32(1.0):
                gamma = div(Float32(1.0), gg)
        for i in range(P):
            st(q, i, mul(gamma, ld(q, i)))
        var start = (head - npairs + MEM) % MEM
        idx = start
        for _ in range(npairs):
            var bb = mul(ld(rho, idx), _dot(ym + idx * P, q, P))
            var coef = sub(ld(al, idx), bb)
            for i in range(P):
                st(q, i, fma3(coef, ld(sm + idx * P, i), ld(q, i)))
            idx = (idx + 1) % MEM
        for i in range(P):
            st(dvec, i, -ld(q, i))
        var gd = _dot(g, dvec, P)
        if not (gd < Float32(0.0)):
            for i in range(P):
                st(dvec, i, -ld(g, i))
            gd = -_dot(g, g, P)
            npairs = 0
        # backtracking Armijo
        var step = Float32(1.0)
        var fnew = Float32(0.0)
        var ok = False
        for _ in range(40):
            for i in range(P):
                st(thn, i, fma3(step, ld(dvec, i), ld(th, i)))
            fnew = prophet_fg(d, y, thn, gn)
            if fnew <= fma3(mul(Float32(1e-4), step), gd, f):
                ok = True
                break
            step = mul(step, Float32(0.5))
        it += 1
        if not ok:
            break
        # the new pair, kept only with positive curvature (q and dvec are
        # free here; the slot is written only when the pair is kept)
        for i in range(P):
            st(q, i, sub(ld(thn, i), ld(th, i)))
            st(dvec, i, sub(ld(gn, i), ld(g, i)))
        var sy = _dot(q, dvec, P)
        if sy > Float32(1e-12):
            var slot = head
            for i in range(P):
                st(sm + slot * P, i, ld(q, i))
                st(ym + slot * P, i, ld(dvec, i))
            st(rho, slot, div(Float32(1.0), sy))
            head = (head + 1) % MEM
            if npairs < MEM:
                npairs += 1
        var fscale = abs(f)
        if abs(fnew) > fscale:
            fscale = abs(fnew)
        if fscale < Float32(1.0):
            fscale = Float32(1.0)
        var df = sub(f, fnew)
        for i in range(P):
            st(th, i, ld(thn, i))
            st(g, i, ld(gn, i))
        f = fnew
        var gmax = Float32(0.0)
        for i in range(P):
            var v = abs(ld(g, i))
            if v > gmax:
                gmax = v
        if gmax < Float32(1e-5):
            break
        if df <= mul(Float32(1e-7), fscale):
            small += 1
            if small >= 3:
                break
        else:
            small = 0
    return (f, it)


def op_prophet_fit(t: Int, a: Args):
    """Series t. p0 y [B, N]; p1 t [N]; p2 X [N, K]; p3 changepoints [S];
    p4 prior scales [K]; p5 params out [B, P]; p6 info out [B, 4]
    (y_scale, objective, iterations, 0); p7 scratch [B, stride].
    i0 N, i1 K, i2 S, i3 multiplicative, i4 stride, i5 max_iter; f0 tau."""
    var N = a.i0
    var K = a.i1
    var S = a.i2
    var P = 3 + S + K
    var sc = a.p7 + t * a.i4
    var ys = sc
    var th = ys + N
    var w = th + P
    var y = a.p0 + t * N
    var scale = Float32(0.0)
    for i in range(N):
        var v = abs(ld(y, i))
        if v > scale:
            scale = v
    if scale == Float32(0.0):
        scale = Float32(1.0)
    for i in range(N):
        st(ys, i, div(ld(y, i), scale))
    var d = ProphetData(a.p1, a.p2, a.p3, a.p4, N, K, S, a.f0, a.i3 != 0)
    # linear_growth_init: through the first and last points
    var t0 = ld(a.p1, 0)
    var t1 = ld(a.p1, N - 1)
    var k = div(sub(ld(ys, N - 1), ld(ys, 0)), sub(t1, t0)) if t1 != t0 else Float32(0.0)
    st(th, 0, k)
    st(th, 1, sub(ld(ys, 0), mul(k, t0)))
    for j in range(S + 1 + K):
        st(th, 2 + j, Float32(0.0))
    var r = lbfgs_prophet(d, ys, th, w, a.i5)
    for j in range(P):
        st(a.p5, t * P + j, ld(th, j))
    var info = a.p6 + t * 4
    st(info, 0, scale)
    st(info, 1, r[0])
    st(info, 2, Float32(r[1]))
    st(info, 3, Float32(0.0))


def op_prophet_predict(t: Int, a: Args):
    """Element t = b * M + i: p0 params [B, P]; p1 info [B, 4] (y_scale);
    p2 t [M]; p3 X [M, K]; p4 changepoints [S]; p5 yhat out [B, M];
    p6 trend out [B, M] (unscaled). i0 M, i1 K, i2 S, i3 multiplicative."""
    var M = a.i0
    var K = a.i1
    var S = a.i2
    var P = 3 + S + K
    var b = t // M
    var i = t - b * M
    var th = a.p0 + b * P
    var ti = ld(a.p2, i)
    var tr = fma3(ld(th, 0), ti, ld(th, 1))
    for j in range(S):
        var c = ld(a.p4, j)
        if ti >= c:
            tr = fma3(ld(th, 2 + j), sub(ti, c), tr)
    var se = Float32(0.0)
    for q in range(K):
        se = fma3(ld(a.p3, i * K + q), ld(th, 3 + S + q), se)
    var yh = fma3(tr, se, tr) if a.i3 != 0 else add(tr, se)
    var scale = ld(a.p1, b * 4)
    st(a.p5, t, mul(yh, scale))
    st(a.p6, t, mul(tr, scale))
