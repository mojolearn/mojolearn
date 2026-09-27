# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH(p, o, q) with a constant or zero mean and normal errors, as the arch
package states it (`arch/univariate/volatility.py` `GARCH`: `backcast`,
`variance_bounds`, `starting_values`, `bounds`, `_analytic_forecast`;
`arch/univariate/recursions_python.py` `garch_recursion`, `bounds_check`,
`ewma_recursion`; `arch/univariate/distribution.py` `Normal.loglikelihood`):
the variance recursion with backcasting and the EWMA variance bounds, the
starting-value grid, and the Gaussian log-likelihood. Float32, one series per
element.

The optimiser is ours: arch maximises by SciPy's SLSQP under the linear
constraint sum(alpha) + sum(gamma) / 2 + sum(beta) <= 1; here statsforecast's
Nelder-Mead (`sequence/nm.mojo`) minimises -loglik inside arch's box bounds,
with the constraint's violators scored +1e30, restarted once from its optimum.
The maximum is the same point when it is interior; the path is not."""
from sequence.nm import Objective, nelder_mead
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_log, identical_pow, identical_sqrt

comptime LOG_2PI: Float32 = 1.8378770664093453


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def _backcast(r: FP, n: Int) -> Float32:
    """sum_{i < tau} 0.94^i r_i^2 / sum 0.94^i, tau = min(75, n)."""
    var tau = 75 if n > 75 else n
    var w = Float32(1.0)
    var sw = Float32(0.0)
    var s = Float32(0.0)
    for i in range(tau):
        var v = ld(r, i)
        s = fma3(w, mul(v, v), s)
        sw = add(sw, w)
        w = mul(w, Float32(0.94))
    return div(s, sw)


def _var_bounds(r: FP, n: Int, vb: FP):
    """variance_bounds (power 2): EWMA(0.94) of r^2 from the backcast,
    [v / 1e6, v * 1e6] floored at var(r) / 1e8, the upper at least
    1 + max r^2 and at most 1e7 (1 + max r^2)."""
    var s = Float32(0.0)
    var mx = Float32(0.0)
    for i in range(n):
        var v = ld(r, i)
        s = add(s, v)
        var v2 = mul(v, v)
        if v2 > mx:
            mx = v2
    var mean = div(s, Float32(n))
    var ss = Float32(0.0)
    for i in range(n):
        var d = sub(ld(r, i), mean)
        ss = fma3(d, d, ss)
    var var_ = div(ss, Float32(n))
    var lo = div(var_, Float32(1e8))
    var up_min = add(Float32(1.0), mx)
    var up_max = mul(Float32(1e7), up_min)
    var e = _backcast(r, n)
    for t in range(n):
        if t > 0:
            var v = ld(r, t - 1)
            e = fma3(Float32(0.06), mul(v, v), mul(Float32(0.94), e))
        var b0 = div(e, Float32(1e6))
        var b1 = mul(e, Float32(1e6))
        if b0 < lo:
            b0 = lo
        if b1 < up_min:
            b1 = up_min
        if b1 > up_max:
            b1 = up_max
        st(vb, 2 * t, b0)
        st(vb, 2 * t + 1, b1)


def garch_sigma2(par: FP, r: FP, n: Int, p: Int, o: Int, q: Int, backcast: Float32, vb: FP, s2: FP):
    """garch_recursion with bounds_check, power 2."""
    for t in range(n):
        var v = ld(par, 0)
        var loc = 1
        for j in range(p):
            if t - 1 - j < 0:
                v = fma3(ld(par, loc), backcast, v)
            else:
                var x = ld(r, t - 1 - j)
                v = fma3(ld(par, loc), mul(x, x), v)
            loc += 1
        for j in range(o):
            if t - 1 - j < 0:
                v = fma3(ld(par, loc), mul(Float32(0.5), backcast), v)
            else:
                var x = ld(r, t - 1 - j)
                if x < Float32(0.0):
                    v = fma3(ld(par, loc), mul(x, x), v)
            loc += 1
        for j in range(q):
            if t - 1 - j < 0:
                v = fma3(ld(par, loc), backcast, v)
            else:
                v = fma3(ld(par, loc), ld(s2, t - 1 - j), v)
            loc += 1
        var lo = ld(vb, 2 * t)
        var hi = ld(vb, 2 * t + 1)
        if not (v == v):
            # 0 x inf from an overflowed square: the upper bound, never a
            # computed NaN (IDENTITY_PATHS Clause B)
            v = hi
        if v < lo:
            v = lo
        elif v > hi:
            v = add(hi, ftz(identical_log(div(v, hi))))
        st(s2, t, v)


def garch_nll(par: FP, r: FP, n: Int, p: Int, o: Int, q: Int, backcast: Float32, vb: FP, s2: FP) -> Float32:
    garch_sigma2(par, r, n, p, o, q, backcast, vb, s2)
    var ll = Float32(0.0)
    for t in range(n):
        var v = ld(s2, t)
        var x = ld(r, t)
        ll = add(ll, add(add(LOG_2PI, ftz(identical_log(v))), div(mul(x, x), v)))
    ll = mul(Float32(0.5), ll)
    if not (ll <= Float32(3.0e38)):
        # an overflowed likelihood (inf, or inf - inf) scores as the worst
        # finite value, never a NaN
        return Float32(3.0e38)
    return ll


struct GarchObj(Objective):
    var y: FP
    var r: FP
    var n: Int
    var p: Int
    var o: Int
    var q: Int
    var has_mean: Bool
    var backcast: Float32
    var vb: FP
    var s2: FP

    def __init__(out self, y: FP, r: FP, n: Int, p: Int, o: Int, q: Int, has_mean: Bool,
                 backcast: Float32, vb: FP, s2: FP):
        self.y = y
        self.r = r
        self.n = n
        self.p = p
        self.o = o
        self.q = q
        self.has_mean = has_mean
        self.backcast = backcast
        self.vb = vb
        self.s2 = s2

    def eval(mut self, x: FP) -> Float32:
        var vol = x + 1 if self.has_mean else x
        var s = Float32(0.0)
        for j in range(self.p):
            s = add(s, ld(vol, 1 + j))
        for j in range(self.o):
            s = fma3(Float32(0.5), ld(vol, 1 + self.p + j), s)
        for j in range(self.q):
            s = add(s, ld(vol, 1 + self.p + self.o + j))
        if s > Float32(1.0):
            return Float32(1e30)
        var mu = ld(x, 0) if self.has_mean else Float32(0.0)
        for t in range(self.n):
            st(self.r, t, sub(ld(self.y, t), mu))
        return garch_nll(vol, self.r, self.n, self.p, self.o, self.q, self.backcast, self.vb, self.s2)


@always_inline
def _grid(i: Int) -> Float32:
    """arch's alphas = gammas = [0.01, 0.05, 0.1, 0.2]."""
    if i == 0:
        return Float32(0.01)
    if i == 1:
        return Float32(0.05)
    if i == 2:
        return Float32(0.1)
    return Float32(0.2)


@always_inline
def _pers(i: Int) -> Float32:
    """arch's persistence grid [0.5, 0.7, 0.9, 0.98]."""
    if i == 0:
        return Float32(0.5)
    if i == 1:
        return Float32(0.7)
    if i == 2:
        return Float32(0.9)
    return Float32(0.98)


def op_garch(t: Int, a: Args):
    """Series t. p0 y [B, n]; p1 params out [B, 1 + 1 + p + o + q]
    (mu, omega, alpha, gamma, beta; mu 0 for a zero mean); p2 info [B, 4]
    out (loglik, iterations); p3 sigma [B, n] out (conditional
    volatility); p4 variance forecast [B, h] out; p5 scratch [B, stride].
    i0 n, i1 h, i2 p, i3 o, i4 q, i5 constant mean, i6 stride."""
    var n = a.i0
    var h = a.i1
    var p = a.i2
    var o = a.i3
    var q = a.i4
    var has_mean = a.i5 != 0
    var k = 1 + p + o + q
    var np_ = k + (1 if has_mean else 0)
    var y = a.p0 + t * n
    var sc = a.p5 + t * a.i6
    var r = sc
    var s2 = r + n
    var vb = s2 + n
    var x = vb + 2 * n
    var lo = x + 16
    var hi = lo + 16
    var cand = hi + 16
    var nm_scr = cand + 16
    # resids at the mean's starting value (arch: mean(y) or 0)
    var mu0 = Float32(0.0)
    if has_mean:
        var s = Float32(0.0)
        for i in range(n):
            s = add(s, ld(y, i))
        mu0 = div(s, Float32(n))
    for i in range(n):
        st(r, i, sub(ld(y, i), mu0))
    var backcast = _backcast(r, n)
    _var_bounds(r, n, vb)
    # starting_values: the (alpha, gamma, persistence) grid, best loglik
    var target = Float32(0.0)
    for i in range(n):
        var v = ld(r, i)
        target = fma3(v, v, target)
    target = div(target, Float32(n))
    var best = Float32(3.0e38)
    for ia in range(4):
        for ig in range(4):
            for ib in range(4):
                var agb = _pers(ib)
                st(cand, 0, mul(sub(Float32(1.0), agb), target))
                for j in range(k - 1):
                    st(cand, 1 + j, mul(sub(Float32(1.0), agb), target))
                if p > 0:
                    for j in range(p):
                        st(cand, 1 + j, div(_grid(ia), Float32(p)))
                    agb = sub(agb, _grid(ia))
                if o > 0:
                    for j in range(o):
                        st(cand, 1 + p + j, div(_grid(ig), Float32(o)))
                    agb = sub(agb, div(_grid(ig), Float32(2.0)))
                if q > 0:
                    for j in range(q):
                        st(cand, 1 + p + o + j, div(agb, Float32(q)))
                var nll = garch_nll(cand, r, n, p, o, q, backcast, vb, s2)
                if nll < best:
                    best = nll
                    for j in range(k):
                        st(x, j + (1 if has_mean else 0), ld(cand, j))
    # box bounds (arch): mu free, omega [1e-8 v, 10 v], alpha, beta [0, 1],
    # gamma [-1, 2] under a matching alpha else [0, 2]
    var off = 0
    if has_mean:
        var mx = Float32(0.0)
        for i in range(n):
            var v = abs(ld(y, i))
            if v > mx:
                mx = v
        st(x, 0, mu0)
        st(lo, 0, sub(Float32(0.0), fma3(Float32(10.0), mx, Float32(1.0))))
        st(hi, 0, fma3(Float32(10.0), mx, Float32(1.0)))
        off = 1
    st(lo, off, mul(Float32(1e-8), target))
    st(hi, off, mul(Float32(10.0), target))
    for j in range(p):
        st(lo, off + 1 + j, Float32(0.0))
        st(hi, off + 1 + j, Float32(1.0))
    for j in range(o):
        st(lo, off + 1 + p + j, Float32(-1.0) if j < p else Float32(0.0))
        st(hi, off + 1 + p + j, Float32(2.0))
    for j in range(q):
        st(lo, off + 1 + p + o + j, Float32(0.0))
        st(hi, off + 1 + p + o + j, Float32(1.0))
    var obj = GarchObj(y, r, n, p, o, q, has_mean, backcast, vb, s2)
    var it = nelder_mead(obj, x, lo, hi, np_, nm_scr, Float32(0.05), Float32(1e-4), 2000, Float32(1e-6))
    # one restart from the optimum (a fresh simplex around it): the
    # likelihood is flat along the persistence ridge and a single simplex
    # can stall short of the maximum
    it += nelder_mead(obj, x, lo, hi, np_, nm_scr, Float32(0.05), Float32(1e-4), 2000, Float32(1e-6))
    var nll = obj.eval(x)
    var outp = a.p1 + t * (1 + k)
    st(outp, 0, ld(x, 0) if has_mean else Float32(0.0))
    for j in range(k):
        st(outp, 1 + j, ld(x, off + j))
    var info = a.p2 + t * 4
    st(info, 0, -nll)
    st(info, 1, Float32(it))
    for i in range(n):
        st(a.p3, t * n + i, ftz(identical_sqrt(ld(s2, i))))
    # _analytic_forecast from the last observation (start = n - 1)
    var par = x + off
    var m = p if p > o else o
    if q > m:
        m = q
    var rr = nm_scr          # m + h squared resids
    var ar = rr + (m + h)    # m + h squared asymmetric resids
    var sg = ar + (m + h)    # m + h variances
    for j in range(m):
        var i = n - m + j
        if i >= 0:
            var v = ld(r, i)
            st(rr, j, mul(v, v))
            st(ar, j, mul(v, v) if v < Float32(0.0) else Float32(0.0))
            st(sg, j, ld(s2, i))
        else:
            st(rr, j, backcast)
            st(ar, j, mul(Float32(0.5), backcast))
            st(sg, j, backcast)
    # the one-step forecast is the recursion one step past the sample
    for hh in range(h):
        var v = ld(par, 0)
        var sl = hh + m - 1
        for j in range(p):
            v = fma3(ld(par, 1 + j), ld(rr, sl - j), v)
        for j in range(o):
            v = fma3(ld(par, 1 + p + j), ld(ar, sl - j), v)
        for j in range(q):
            v = fma3(ld(par, 1 + p + o + j), ld(sg, sl - j), v)
        if not (v <= Float32(3.0e38)):
            v = Float32(3.0e38)
        st(rr, hh + m, v)
        st(ar, hh + m, mul(Float32(0.5), v))
        st(sg, hh + m, v)
        st(a.p4, t * h + hh, v)
