# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Theta forecasters of statsforecast (`python/statsforecast/theta.py`
`auto_theta` / `thetamodel` / `forecast_theta`, `src/theta.cpp` `init_state`,
`update`, `calc`, `optimize`): the standard (STM), optimised (OTM), dynamic
standard (DSTM) and dynamic optimised (DOTM) theta models of Fiorucci et al.
(2016), their MSE objective minimised by statsforecast's Nelder-Mead
(`sequence/nm.mojo`), and auto_theta's seasonal handling: the ACF test at lag
m against the 95% normal quantile, statsmodels' classical `seasonal_decompose`
(centred moving average, phase means normalised), multiplicative unless the
data or the seasonal indices forbid it, the forecast reseasonalised by the
last season. Float32, one series per element.

Differences, all value-level: (1 - alpha)^i is a running product rather than
std::pow; the objective's 1-step forecast is the state update's mu (the
reference computes nmse-step forecasts and uses only the first for the
objective); the ACF, the decomposition and the objective are float32."""
from sequence.nm import Objective, nelder_mead
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_sqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: lane/apple-fast-tier (2026-10-02). `theta_run` writes five state words
#: and one error per step to the thread's device scratch and reads the
#: previous row back, on a kernel of one thread per series (64 series on
#: taxi-hourly): every step is a round trip to device memory with no other
#: threads to hide it behind, inside Nelder-Mead's up to 1000 evaluations
#: (board: theta taxi-hourly FAST 1,747 ms, IDENTICAL 388). Only row n - 1
#: is ever read after the run (by the forecast) and the error sum is a
#: running fma, so THETA_REG keeps the recurrence in registers: the same
#: operations in the same order, four words written. Default on FAST + Apple
#: since the M3 A/B (lane/apple-fast-tier 78d5b99d1, theta taxi-hourly, n=1):
#: alone 1,770 -> 1,326 ms; with MOJOLEARN_SEQ_FAST_FMA 1,769 -> 220 ms,
#: forecast_rmse 49.28 -> 49.02. -D MOJOLEARN_SEQ_THETA_REG_OFF restores the
#: stored-row code; the old -D MOJOLEARN_SEQ_THETA_REG=1 is harmless.
#: IDENTICAL compiles the stored-row code.
comptime THETA_REG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEQ_THETA_REG_OFF"]()
)

#: lane apple-fast-regress (2026-10-03; docs/apple-fast/notes/regress-oct3.md).
#: Since SEQ_FAST_FMA became the FAST + Apple default (95a09d1fd), the
#: board's dynamic-optimized-theta taxi-hourly row went 727 -> 2,087 ms
#: while synthetic went 485 -> 377: the fused fmas move the objective's
#: last bits; the likely cause (the -D MOJOLEARN_SEQ_FAST_FMA_OFF arm
#: confirms it) is that on taxi-hourly Nelder-Mead's simplex no longer
#: settles on a fixed point and a series runs on toward the 1,000-iteration
#: cap, cycling.
#: MOJOLEARN_SEQ_FAST_THETA_SNAP hands the theta fits the cycle watch GARCH
#: already uses (sequence/nm.mojo, `snap`): a state that returns bit for bit
#: to an earlier one runs only the iterations left of its last lap, the
#: same final state, best vertex and iteration count as running them all
#: (no result moves). The snapshot is 16 floats of the 64 the row reserves
#: for Nelder-Mead (k <= 3 coordinates use at most 28). Default on FAST +
#: Apple since the M3 A/B (n=3, digests identical): dynamic-optimized-theta
#: taxi-hourly 2,087.7 -> 561.9 ms, rmse the same. SEQ_FAST_FMA stays on.
#: -D MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF restores the plain run; the old
#: -D MOJOLEARN_SEQ_FAST_THETA_SNAP=1 is harmless.
comptime THETA_SNAP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF"]()
)
#: the snapshot's offset in the Nelder-Mead scratch (after the (k + 1) k +
#: (k + 1) + 4 k <= 28 floats of k <= 3; (k + 1) k + (k + 1) <= 16 floats)
comptime THETA_SNAP_OFF = 32

comptime STM = 0
comptime OTM = 1
comptime DSTM = 2
comptime DOTM = 3


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def theta_run(
    y: FP, n: Int, model: Int, level0: Float32, alpha: Float32, theta: Float32, states: FP, e: FP,
) -> Float32:
    """`calc` with the states [n + h, 5] (level, meany, A, B, mu) written;
    returns sum(e[3:]^2) / max(mean|y|, 1e-10)."""
    var dyn = model == DSTM or model == DOTM
    var A: Float32
    var B: Float32
    var mu: Float32
    var y0 = ld(y, 0)
    var k = sub(Float32(1.0), div(Float32(1.0), theta))
    if dyn:
        A = y0
        B = Float32(0.0)
        mu = y0
    else:
        var s = Float32(0.0)
        var w = Float32(0.0)
        for i in range(n):
            var v = ld(y, i)
            s = add(s, v)
            w = fma3(v, Float32(i + 1), w)
        var ym = div(s, Float32(n))
        var wa = div(w, Float32(n))
        B = div(mul(Float32(6.0), sub(mul(Float32(2.0), wa), mul(Float32(n + 1), ym))),
                Float32(n * n - 1))
        A = sub(ym, div(mul(Float32(n + 1), B), Float32(2.0)))
        mu = fma3(k, add(A, B), level0)
    var oma = sub(Float32(1.0), alpha)
    st(states, 0, fma3(alpha, y0, mul(oma, level0)))
    st(states, 1, y0)
    st(states, 2, A)
    st(states, 3, B)
    st(states, 4, mu)
    st(e, 0, sub(y0, mu))
    var pw = oma          # (1 - alpha)^i at i = 1
    for i in range(1, n):
        var r0 = (i - 1) * 5
        var r = i * 5
        var lev = ld(states, r0)
        var my = ld(states, r0 + 1)
        var An = ld(states, r0 + 2)
        var Bn = ld(states, r0 + 3)
        var pw1 = mul(pw, oma)
        var m = fma3(k, add(mul(An, pw), div(mul(Bn, sub(Float32(1.0), pw1)), alpha)), lev)
        var yi = ld(y, i)
        st(e, i, sub(yi, m))
        st(states, r + 4, m)
        st(states, r, fma3(alpha, yi, mul(oma, lev)))
        var my2 = div(fma3(Float32(i), my, yi), Float32(i + 1))
        st(states, r + 1, my2)
        if dyn:
            var b2 = div(add(mul(Float32(i - 1), Bn), div(mul(Float32(6.0), sub(yi, my)), Float32(i + 1))),
                         Float32(i + 2))
            st(states, r + 3, b2)
            st(states, r + 2, sub(my2, div(mul(b2, Float32(i + 2)), Float32(2.0))))
        else:
            st(states, r + 2, An)
            st(states, r + 3, Bn)
        pw = pw1
    var sa = Float32(0.0)
    for i in range(n):
        sa = add(sa, abs(ld(y, i)))
    var mean_y = div(sa, Float32(n))
    if mean_y < Float32(1e-10):
        mean_y = Float32(1e-10)
    var sse = Float32(0.0)
    for i in range(3, n):
        var v = ld(e, i)
        sse = fma3(v, v, sse)
    return div(sse, mean_y)


@always_inline
def theta_run_reg(
    y: FP, n: Int, model: Int, level0: Float32, alpha: Float32, theta: Float32, last: FP,
) -> Float32:
    """`theta_run` with the state rows in registers (MOJOLEARN_SEQ_THETA_REG):
    writes row n - 1 (level, meany, A, B) to `last[0:4]` and returns
    sum(e[3:]^2) / max(mean|y|, 1e-10), the error sum accumulated as each
    error is formed, in the same order as the stored-row code."""
    var dyn = model == DSTM or model == DOTM
    var A: Float32
    var B: Float32
    var mu: Float32
    var y0 = ld(y, 0)
    var k = sub(Float32(1.0), div(Float32(1.0), theta))
    if dyn:
        A = y0
        B = Float32(0.0)
        mu = y0
    else:
        var s = Float32(0.0)
        var w = Float32(0.0)
        for i in range(n):
            var v = ld(y, i)
            s = add(s, v)
            w = fma3(v, Float32(i + 1), w)
        var ym = div(s, Float32(n))
        var wa = div(w, Float32(n))
        B = div(mul(Float32(6.0), sub(mul(Float32(2.0), wa), mul(Float32(n + 1), ym))),
                Float32(n * n - 1))
        A = sub(ym, div(mul(Float32(n + 1), B), Float32(2.0)))
        mu = fma3(k, add(A, B), level0)
    var oma = sub(Float32(1.0), alpha)
    var lev = fma3(alpha, y0, mul(oma, level0))
    var my = y0
    var An = A
    var Bn = B
    _ = sub(y0, mu)   # e[0]: outside the error sum (i >= 3)
    var sse = Float32(0.0)
    var pw = oma          # (1 - alpha)^i at i = 1
    for i in range(1, n):
        var pw1 = mul(pw, oma)
        var m = fma3(k, add(mul(An, pw), div(mul(Bn, sub(Float32(1.0), pw1)), alpha)), lev)
        var yi = ld(y, i)
        var ei = sub(yi, m)
        if i >= 3:
            sse = fma3(ei, ei, sse)
        var lev2 = fma3(alpha, yi, mul(oma, lev))
        var my2 = div(fma3(Float32(i), my, yi), Float32(i + 1))
        if dyn:
            var b2 = div(add(mul(Float32(i - 1), Bn), div(mul(Float32(6.0), sub(yi, my)), Float32(i + 1))),
                         Float32(i + 2))
            Bn = b2
            An = sub(my2, div(mul(b2, Float32(i + 2)), Float32(2.0)))
        lev = lev2
        my = my2
        pw = pw1
    st(last, 0, lev)
    st(last, 1, my)
    st(last, 2, An)
    st(last, 3, Bn)
    var sa = Float32(0.0)
    for i in range(n):
        sa = add(sa, abs(ld(y, i)))
    var mean_y = div(sa, Float32(n))
    if mean_y < Float32(1e-10):
        mean_y = Float32(1e-10)
    return div(sse, mean_y)


@always_inline
def theta_forecast_reg(n: Int, h: Int, model: Int, alpha: Float32, theta: Float32, last: FP, f: FP):
    """`theta_forecast` from `theta_run_reg`'s last row, the h rows in registers."""
    var dyn = model == DSTM or model == DOTM
    var k = sub(Float32(1.0), div(Float32(1.0), theta))
    var oma = sub(Float32(1.0), alpha)
    var pw = Float32(1.0)
    for _ in range(n):
        pw = mul(pw, oma)
    var lev = ld(last, 0)
    var my = ld(last, 1)
    var An = ld(last, 2)
    var Bn = ld(last, 3)
    for j in range(h):
        var i = n + j
        var pw1 = mul(pw, oma)
        var m = fma3(k, add(mul(An, pw), div(mul(Bn, sub(Float32(1.0), pw1)), alpha)), lev)
        var lev2 = fma3(alpha, m, mul(oma, lev))
        var my2 = div(fma3(Float32(i), my, m), Float32(i + 1))
        if dyn:
            var b2 = div(add(mul(Float32(i - 1), Bn), div(mul(Float32(6.0), sub(m, my)), Float32(i + 1))),
                         Float32(i + 2))
            Bn = b2
            An = sub(my2, div(mul(b2, Float32(i + 2)), Float32(2.0)))
        lev = lev2
        my = my2
        st(f, j, m)
        pw = pw1


@always_inline
def theta_forecast(n: Int, h: Int, model: Int, alpha: Float32, theta: Float32, states: FP, f: FP):
    """`forecast`: h updates past row n - 1 with y = mu."""
    var dyn = model == DSTM or model == DOTM
    var k = sub(Float32(1.0), div(Float32(1.0), theta))
    var oma = sub(Float32(1.0), alpha)
    var pw = Float32(1.0)
    for _ in range(n):
        pw = mul(pw, oma)
    for j in range(h):
        var i = n + j
        var r0 = (i - 1) * 5
        var r = i * 5
        var lev = ld(states, r0)
        var my = ld(states, r0 + 1)
        var An = ld(states, r0 + 2)
        var Bn = ld(states, r0 + 3)
        var pw1 = mul(pw, oma)
        var m = fma3(k, add(mul(An, pw), div(mul(Bn, sub(Float32(1.0), pw1)), alpha)), lev)
        st(states, r + 4, m)
        st(states, r, fma3(alpha, m, mul(oma, lev)))
        var my2 = div(fma3(Float32(i), my, m), Float32(i + 1))
        st(states, r + 1, my2)
        if dyn:
            var b2 = div(add(mul(Float32(i - 1), Bn), div(mul(Float32(6.0), sub(m, my)), Float32(i + 1))),
                         Float32(i + 2))
            st(states, r + 3, b2)
            st(states, r + 2, sub(my2, div(mul(b2, Float32(i + 2)), Float32(2.0))))
        else:
            st(states, r + 2, An)
            st(states, r + 3, Bn)
        st(f, j, m)
        pw = pw1


struct ThetaObj(Objective):
    var y: FP
    var n: Int
    var model: Int
    var opt_level: Bool
    var opt_alpha: Bool
    var opt_theta: Bool
    var level: Float32
    var alpha: Float32
    var theta: Float32
    var states: FP
    var e: FP

    @always_inline
    def __init__(out self, y: FP, n: Int, model: Int, ol: Bool, oa: Bool, ot: Bool,
                 level: Float32, alpha: Float32, theta: Float32, states: FP, e: FP):
        self.y = y
        self.n = n
        self.model = model
        self.opt_level = ol
        self.opt_alpha = oa
        self.opt_theta = ot
        self.level = level
        self.alpha = alpha
        self.theta = theta
        self.states = states
        self.e = e

    @always_inline
    def params(self, x: FP) -> Tuple[Float32, Float32, Float32]:
        var j = 0
        var l = self.level
        var a = self.alpha
        var t = self.theta
        if self.opt_level:
            l = ld(x, j)
            j += 1
        if self.opt_alpha:
            a = ld(x, j)
            j += 1
        if self.opt_theta:
            t = ld(x, j)
        return (l, a, t)

    @always_inline
    def eval(mut self, x: FP) -> Float32:
        var p = self.params(x)
        var mse = Float32(0.0)
        comptime if THETA_REG:
            mse = theta_run_reg(self.y, self.n, self.model, p[0], p[1], p[2], self.states)
        else:
            mse = theta_run(self.y, self.n, self.model, p[0], p[1], p[2], self.states, self.e)
        return mse if mse > Float32(-1e10) else Float32(-1e10)


@always_inline
def _acf_decide(y: FP, n: Int, m: Int) -> Bool:
    """auto_theta's seasonal test: statsmodels acf (demeaned, / n, no FFT)
    at lags 1..m; |r_m| / sqrt((1 + 2 sum_{k<m} r_k^2) / n) > 1.6448536."""
    var s = Float32(0.0)
    for i in range(n):
        s = add(s, ld(y, i))
    var mean = div(s, Float32(n))
    var c0 = Float32(0.0)
    for i in range(n):
        var d = sub(ld(y, i), mean)
        c0 = fma3(d, d, c0)
    if c0 == Float32(0.0):
        return False
    var acc = Float32(0.0)
    var rm = Float32(0.0)
    for k in range(1, m + 1):
        var c = Float32(0.0)
        for i in range(n - k):
            c = fma3(sub(ld(y, i), mean), sub(ld(y, i + k), mean), c)
        var r = div(c, c0)
        if k < m:
            acc = fma3(r, r, acc)
        else:
            rm = r
    var stat = ftz(identical_sqrt(div(fma3(Float32(2.0), acc, Float32(1.0)), Float32(n))))
    return div(abs(rm), stat) > Float32(1.6448536269514722)


@always_inline
def _decompose(y: FP, n: Int, m: Int, mult: Bool, trend: FP, seas: FP):
    """statsmodels seasonal_decompose(model, period=m), two-sided filter
    ([.5, 1, ..., 1, .5] / m for even m, ones / m for odd): seas[0:m] is the
    normalised phase mean of the detrended series (defined points only)."""
    var even = m % 2 == 0
    var L = m + 1 if even else m
    var half = L // 2
    for t in range(n):
        st(trend, t, Float32(0.0))
    for t in range(half, n - half):
        var s = Float32(0.0)
        for j in range(L):
            var w = Float32(1.0)
            if even and (j == 0 or j == L - 1):
                w = Float32(0.5)
            s = fma3(w, ld(y, t - half + j), s)
        st(trend, t, div(s, Float32(m)))
    var tot = Float32(0.0)
    for p in range(m):
        var s = Float32(0.0)
        var c = 0
        var t = p
        while t < n:
            if t >= half and t < n - half:
                var d = div(ld(y, t), ld(trend, t)) if mult else sub(ld(y, t), ld(trend, t))
                s = add(s, d)
                c += 1
            t += m
        var v = div(s, Float32(c)) if c > 0 else Float32(0.0)
        st(seas, p, v)
        tot = add(tot, v)
    var mean = div(tot, Float32(m))
    for p in range(m):
        st(seas, p, div(ld(seas, p), mean) if mult else sub(ld(seas, p), mean))


def op_theta(t: Int, a: Args):
    """Series t. p0 y [B, n]; p1 forecast [B, h] out; p2 info [B, 8] out
    (level, alpha, theta, mse, model, decomposed, multiplicative, iters);
    p3 scratch [B, stride]. i0 n, i1 h, i2 m, i3 model (-1 auto), i4
    decomposition (0 multiplicative, 1 additive), i5 fixed mask
    (1 level, 2 alpha, 4 theta), i6 scratch stride; f0 level, f1 alpha,
    f2 theta (the fixed values)."""
    var n = a.i0
    var h = a.i1
    var m = a.i2
    var sc = a.p3 + t * a.i6
    var y = a.p0 + t * n
    var yd = sc
    var trend = yd + n
    var seas = trend + n
    var states = seas + (m if m > 0 else 1)
    var e = states + 5 * (n + h)
    var nm_scr = e + n
    var x = nm_scr + 64
    var lo = x + 4
    var hi = lo + 4
    var f = hi + 4
    var decompose = False
    var mult = a.i4 == 0
    if m >= 4 and n >= 2 * m:
        decompose = _acf_decide(y, n, m)
    for i in range(n):
        st(yd, i, ld(y, i))
    if decompose:
        var pos = True
        for i in range(n):
            if not (ld(y, i) > Float32(0.0)):
                pos = False
        if mult and not pos:
            mult = False
        _decompose(y, n, m, mult, trend, seas)
        if mult:
            for p in range(m):
                if ld(seas, p) < Float32(0.01):
                    mult = False
            if not mult:
                _decompose(y, n, m, False, trend, seas)
        for i in range(n):
            var s = ld(seas, i % m)
            st(yd, i, div(ld(y, i), s) if mult else sub(ld(y, i), s))
    var best_mse = Float32(3.0e38)
    var best_model = 0
    var bl = Float32(0.0)
    var ba = Float32(0.0)
    var bt = Float32(0.0)
    var iters = 0
    var m_lo = 0 if a.i3 < 0 else a.i3
    var m_hi = 3 if a.i3 < 0 else a.i3
    for model in range(m_lo, m_hi + 1):
        var fixed = a.i5
        var ol = (fixed & 1) == 0
        var oa = (fixed & 2) == 0
        var ot = (fixed & 4) == 0 and (model == OTM or model == DOTM)
        var l0 = div(ld(yd, 0), Float32(2.0)) if ol else a.f0
        var a0 = Float32(0.5) if oa else a.f1
        var t0 = Float32(2.0) if ((fixed & 4) == 0 or model == STM or model == DSTM) else a.f2
        var obj = ThetaObj(yd, n, model, ol, oa, ot, l0, a0, t0, states, e)
        var k = 0
        if ol:
            st(x, k, l0)
            st(lo, k, Float32(-1e10))
            st(hi, k, Float32(1e10))
            k += 1
        if oa:
            st(x, k, a0)
            st(lo, k, Float32(0.1))
            st(hi, k, Float32(0.99))
            k += 1
        if ot:
            st(x, k, t0)
            st(lo, k, Float32(1.0))
            st(hi, k, Float32(1e10))
            k += 1
        var it = 0
        if k > 0:
            comptime if THETA_SNAP:
                it = nelder_mead(obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4),
                                 nm_scr + THETA_SNAP_OFF)
            else:
                it = nelder_mead(obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4))
        var p = obj.params(x)
        var mse = Float32(0.0)
        comptime if THETA_REG:
            mse = theta_run_reg(yd, n, model, p[0], p[1], p[2], states)
        else:
            mse = theta_run(yd, n, model, p[0], p[1], p[2], states, e)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            bl = p[0]
            ba = p[1]
            bt = p[2]
            iters = it
    comptime if THETA_REG:
        _ = theta_run_reg(yd, n, best_model, bl, ba, bt, states)
        theta_forecast_reg(n, h, best_model, ba, bt, states, f)
    else:
        _ = theta_run(yd, n, best_model, bl, ba, bt, states, e)
        theta_forecast(n, h, best_model, ba, bt, states, f)
    for j in range(h):
        var v = ld(f, j)
        if decompose:
            var s = ld(seas, (n - m + (j % m)) % m)
            v = mul(v, s) if mult else add(v, s)
        st(a.p1, t * h + j, v)
    var info = a.p2 + t * 8
    st(info, 0, bl)
    st(info, 1, ba)
    st(info, 2, bt)
    st(info, 3, best_mse)
    st(info, 4, Float32(best_model))
    st(info, 5, Float32(1.0) if decompose else Float32(0.0))
    st(info, 6, Float32(1.0) if (decompose and mult) else Float32(0.0))
    st(info, 7, Float32(iters))
