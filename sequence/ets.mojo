# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exponential smoothing state-space models with an additive, optionally
damped, trend: ETS(A|M, N|A|Ad, N), as statsforecast states them
(`python/statsforecast/ets.py` `ets_f` / `etsmodel` / `initparam` /
`initstate`; `src/ets.cpp` `Update`, `Forecast`, `Calc`,
`ObjectiveFunction`, `Optimize`): the likelihood criterion
n log(sum e^2) (+ 2 sum log|f| for multiplicative errors) over the smoothing
parameters AND the initial states, minimised by statsforecast's Nelder-Mead
(`sequence/nm.mojo`) inside the usual bounds (alpha, beta in
[1e-4, 0.9999] with beta <= the initial alpha, phi in [0.8, 0.98]); the
initial level and trend from the least-squares line through the first
min(10, n) observations. Float32, one series per element.

Not carried here (sequence/NOT_IMPLEMENTED.tsv): seasonal components,
multiplicative trend, automatic model selection (the 'Z' letters),
the admissible / both bound checks (the reference's objective does not
apply them either), non-normal errors, prediction intervals."""
from sequence.nm import Objective, nelder_mead
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_log, identical_pow

comptime ERR_A = 0
comptime ERR_M = 1


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _log(x: Float32) -> Float32:
    return ftz(identical_log(x))


def ets_lik(y: FP, n: Int, err: Int, trend: Bool, alpha: Float32, beta: Float32, phi: Float32,
            l0: Float32, b0: Float32) -> Tuple[Float32, Float32, Float32]:
    """(lik, final level, final trend) of Calc for the non-seasonal models."""
    var l = l0
    var b = b0 if trend else Float32(0.0)
    var sse = Float32(0.0)
    var slog = Float32(0.0)
    var ba = div(beta, alpha)
    for i in range(n):
        var ol = l
        var ob = b
        var phib = mul(phi, ob) if trend else Float32(0.0)
        var q = add(ol, phib)
        var f0 = q
        var yi = ld(y, i)
        var e: Float32
        if err == ERR_A:
            e = sub(yi, f0)
        else:
            var fd = add(f0, Float32(1e-10)) if abs(f0) < Float32(1e-10) else f0
            e = div(sub(yi, f0), fd)
        l = fma3(alpha, sub(yi, q), q)
        if trend:
            b = fma3(ba, sub(sub(l, ol), phib), phib)
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
    return (lik, l, b)


struct EtsObj(Objective):
    var y: FP
    var n: Int
    var err: Int
    var trend: Bool
    var oa: Bool
    var ob: Bool
    var op: Bool
    var alpha: Float32
    var beta: Float32
    var phi: Float32

    def __init__(out self, y: FP, n: Int, err: Int, trend: Bool, oa: Bool, ob: Bool, op: Bool,
                 alpha: Float32, beta: Float32, phi: Float32):
        self.y = y
        self.n = n
        self.err = err
        self.trend = trend
        self.oa = oa
        self.ob = ob
        self.op = op
        self.alpha = alpha
        self.beta = beta
        self.phi = phi

    def unpack(self, x: FP) -> Tuple[Float32, Float32, Float32, Float32, Float32]:
        var j = 0
        var a = self.alpha
        var b = self.beta
        var p = self.phi
        if self.oa:
            a = ld(x, j)
            j += 1
        if self.ob:
            b = ld(x, j)
            j += 1
        if self.op:
            p = ld(x, j)
            j += 1
        var l0 = ld(x, j)
        var b0 = ld(x, j + 1) if self.trend else Float32(0.0)
        return (a, b, p, l0, b0)

    def eval(mut self, x: FP) -> Float32:
        var u = self.unpack(x)
        var r = ets_lik(self.y, self.n, self.err, self.trend, u[0], u[1], u[2], u[3], u[4])
        return r[0] if r[0] > Float32(-1e10) else Float32(-1e10)


def op_ets(t: Int, a: Args):
    """Series t. p0 y [B, n]; p1 forecast [B, h] out; p2 info [B, 8] out
    (alpha, beta, phi, l0, b0, lik, iterations, parameter count);
    p3 scratch [B, 128]. i0 n, i1 h, i2 error (0 A, 1 M), i3 trend (0 N,
    1 A), i4 damped, i5 fixed mask (1 alpha, 2 beta, 4 phi); f0 alpha,
    f1 beta, f2 phi (read where fixed)."""
    var n = a.i0
    var h = a.i1
    var y = a.p0 + t * n
    var trend = a.i3 == 1
    var damped = trend and a.i4 != 0
    var sc = a.p3 + t * 128
    var x = sc
    var lo = x + 8
    var hi = lo + 8
    var nm_scr = hi + 8
    # initparam (m = 1, usual bounds)
    var lo_a = Float32(1e-4)
    var hi_a = Float32(0.9999)
    var fa = (a.i5 & 1) != 0
    var fb = (a.i5 & 2) != 0
    var fp = (a.i5 & 4) != 0
    var alpha = a.f0 if fa else fma3(Float32(0.2), sub(hi_a, lo_a), lo_a)
    var hi_b = hi_a if hi_a < alpha else alpha
    var beta = a.f1 if fb else fma3(Float32(0.1), sub(hi_b, lo_a), lo_a)
    var phi = Float32(1.0)
    if damped:
        phi = a.f2 if fp else fma3(Float32(0.99), sub(Float32(0.98), Float32(0.8)), Float32(0.8))
    # initstate: least squares line through the first min(10, n) points
    var maxn = 10 if n > 10 else n
    var l0: Float32
    var b0 = Float32(0.0)
    var sy = Float32(0.0)
    for i in range(maxn):
        sy = add(sy, ld(y, i))
    var ybar = div(sy, Float32(maxn))
    if trend:
        var tbar = div(Float32(maxn + 1), Float32(2.0))
        var sxy = Float32(0.0)
        var sxx = Float32(0.0)
        for i in range(maxn):
            var dt = sub(Float32(i + 1), tbar)
            sxy = fma3(dt, sub(ld(y, i), ybar), sxy)
            sxx = fma3(dt, dt, sxx)
        b0 = div(sxy, sxx) if sxx > Float32(0.0) else Float32(0.0)
        l0 = sub(ybar, mul(b0, tbar))
        if abs(add(l0, b0)) < Float32(1e-8):
            l0 = mul(l0, Float32(1.001))
            b0 = mul(b0, Float32(0.999))
    else:
        l0 = ybar
    var oa = not fa
    var ob = trend and not fb
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
    if op:
        st(x, k, phi)
        st(lo, k, Float32(0.8))
        st(hi, k, Float32(0.98))
        k += 1
    st(x, k, l0)
    st(lo, k, Float32(-3.0e38))
    st(hi, k, Float32(3.0e38))
    k += 1
    if trend:
        st(x, k, b0)
        st(lo, k, Float32(-3.0e38))
        st(hi, k, Float32(3.0e38))
        k += 1
    var obj = EtsObj(y, n, a.i2, trend, oa, ob, op, alpha, beta, phi)
    var it = nelder_mead(obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4))
    var u = obj.unpack(x)
    var r = ets_lik(y, n, a.i2, trend, u[0], u[1], u[2], u[3], u[4])
    var l = r[1]
    var b = r[2]
    var phistar = u[2]
    for i in range(h):
        var f = add(l, mul(phistar, b)) if trend else l
        st(a.p1, t * h + i, f)
        if i < h - 1:
            phistar = add(phistar, ftz(identical_pow(u[2], Float32(i + 1))))
    var info = a.p2 + t * 8
    st(info, 0, u[0])
    st(info, 1, u[1] if trend else Float32(0.0))
    st(info, 2, u[2])
    st(info, 3, u[3])
    st(info, 4, u[4])
    st(info, 5, r[0])
    st(info, 6, Float32(it))
    st(info, 7, Float32(k))
