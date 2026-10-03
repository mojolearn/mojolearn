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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_log, identical_pow, identical_sqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: lane/apple-fast-seq (2026-10-02). `GarchObj.eval` writes the n residuals
#: to the thread's device scratch, `garch_sigma2` writes sigma2_t and reads
#: sigma2_{t-1} and r_{t-1} back, and `garch_nll` reads both rows again, on
#: a kernel of one thread per series (64 on taxi-hourly) inside two
#: Nelder-Mead runs of up to 2000 iterations: every step is a round trip to
#: device memory with nothing to hide the latency behind. For p, o, q <= 1
#: (the board's GARCH(1, 1)) each step reads only the previous residual and
#: variance, and the likelihood is a running sum, so
#: `-D MOJOLEARN_SEQ_GARCH_REG=1` keeps the recursion in registers for the
#: optimiser's evaluations (`garch_nll_reg`): the same operations in the
#: same order, nothing stored; the final evaluation still stores sigma2
#: (the sigma output and the forecast read it). FAST only. Default on FAST +
#: Apple since the M3 A/B (lane/apple-fast-seq 8b3f1d90e, n=1, mean_llf
#: identical): garch taxi-hourly 2,770 -> 968 ms. -D MOJOLEARN_SEQ_GARCH_REG_OFF
#: restores the stored code; the old -D MOJOLEARN_SEQ_GARCH_REG=1 is harmless.
comptime GARCH_REG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEQ_GARCH_REG_OFF"]()
)
#: lane/apple-fast-seq (2026-10-02). `op_garch`'s starting values are the
#: best of arch's 4 x 4 x 4 grid, 64 serial `garch_nll` passes over the
#: series per thread (`sequence/garch.mojo` op_garch, "starting_values")
#: before the optimiser starts: 64 threads x 64 passes on taxi-hourly.
#: `-D MOJOLEARN_SEQ_GARCH_GRID=1` runs the grid as one thread per (series,
#: candidate) (`_garch_grid_cell`, scratch-free, p, o, q <= 1) in a launch
#: of B x 64 elements before the fit, and the fit's thread takes the argmin
#: over the 64 values in candidate order with the serial loop's strict `<`
#: (the first lowest wins), so the chosen candidate and its bits are the
#: serial loop's. FAST only; IDENTICAL compiles the serial grid. Default on
#: FAST + Apple since the M3 A/B (lane/apple-fast-seq 8b3f1d90e, n=1, mean_llf
#: identical): with GARCH_REG, taxi-hourly 2,769 -> 905 ms, synthetic 1,525 ->
#: 528 ms. -D MOJOLEARN_SEQ_GARCH_GRID_OFF restores the serial grid; the old
#: -D MOJOLEARN_SEQ_GARCH_GRID=1 is harmless.
comptime GARCH_GRID = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEQ_GARCH_GRID_OFF"]()
)
comptime GARCH_GRID_N = 64

comptime LOG_2PI: Float32 = 1.8378770664093453
#: floats at the end of each scratch row for Nelder-Mead's cycle snapshot:
#: (k + 1) k + (k + 1) for k <= 8 coordinates
comptime GARCH_SNAP = 81


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
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


@always_inline
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


@always_inline
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


@always_inline
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


@always_inline
def _garch_step_reg(w: Float32, pa: Float32, pg: Float32, pb: Float32, p: Int, o: Int, q: Int,
                    t: Int, rp: Float32, sp: Float32, backcast: Float32, lo: Float32, hi: Float32) -> Float32:
    """One step of `garch_sigma2` for p, o, q <= 1 from registers: the
    previous residual rp and variance sp, the bounds lo, hi of this step."""
    var v = w
    if p > 0:
        if t < 1:
            v = fma3(pa, backcast, v)
        else:
            v = fma3(pa, mul(rp, rp), v)
    if o > 0:
        if t < 1:
            v = fma3(pg, mul(Float32(0.5), backcast), v)
        elif rp < Float32(0.0):
            v = fma3(pg, mul(rp, rp), v)
    if q > 0:
        if t < 1:
            v = fma3(pb, backcast, v)
        else:
            v = fma3(pb, sp, v)
    if not (v == v):
        v = hi
    if v < lo:
        v = lo
    elif v > hi:
        v = add(hi, ftz(identical_log(div(v, hi))))
    return v


@always_inline
def garch_nll_reg(w: Float32, pa: Float32, pg: Float32, pb: Float32, y: FP, mu: Float32, n: Int,
                  p: Int, o: Int, q: Int, backcast: Float32, vb: FP) -> Float32:
    """`GarchObj.eval`'s residuals + `garch_sigma2` + `garch_nll` for
    p, o, q <= 1 with nothing stored (MOJOLEARN_SEQ_GARCH_REG): r_t = y_t -
    mu is formed as it is used, r_{t-1} and sigma2_{t-1} stay in registers
    and the log-likelihood terms are summed in the same pass; the same
    operations in the same order. w, pa, pg, pb: omega, alpha, gamma, beta
    (unused ones 0)."""
    var ll = Float32(0.0)
    var rp = Float32(0.0)
    var sp = Float32(0.0)
    for t in range(n):
        var v = _garch_step_reg(w, pa, pg, pb, p, o, q, t, rp, sp, backcast, ld(vb, 2 * t), ld(vb, 2 * t + 1))
        var x = sub(ld(y, t), mu)
        ll = add(ll, add(add(LOG_2PI, ftz(identical_log(v))), div(mul(x, x), v)))
        rp = x
        sp = v
    ll = mul(Float32(0.5), ll)
    if not (ll <= Float32(3.0e38)):
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

    @always_inline
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

    @always_inline
    def eval(mut self, x: FP) -> Float32:
        comptime if GARCH_REG:
            if self.p <= 1 and self.o <= 1 and self.q <= 1:
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
                var pa = ld(vol, 1) if self.p > 0 else Float32(0.0)
                var pg = ld(vol, 1 + self.p) if self.o > 0 else Float32(0.0)
                var pb = ld(vol, 1 + self.p + self.o) if self.q > 0 else Float32(0.0)
                return garch_nll_reg(ld(vol, 0), pa, pg, pb, self.y, mu, self.n, self.p, self.o, self.q,
                                     self.backcast, self.vb)
        return self.eval_stored(x)

    @always_inline
    def eval_stored(mut self, x: FP) -> Float32:
        """The objective with the residuals and sigma2 left in r and s2."""
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


@always_inline
def _garch_grid_cell(t: Int, a: Args):
    """MOJOLEARN_SEQ_GARCH_GRID: element t = b GARCH_GRID_N + c: the
    negative log-likelihood of series b's starting-value candidate
    c = 16 ia + 4 ig + ib (the serial grid's nesting order) into p6[t].
    Scratch-free: the residuals, the backcast, the variance bounds, the
    candidate and the recursion are formed from y in registers, the same
    operations in the same order as op_garch's prologue, `_backcast`,
    `_var_bounds`, the grid loop and `garch_nll_reg` (p, o, q <= 1)."""
    var n = a.i0
    var p = a.i2
    var o = a.i3
    var q = a.i4
    var has_mean = a.i5 != 0
    var b = t // GARCH_GRID_N
    var c = t - b * GARCH_GRID_N
    var ia = c // 16
    var ig = (c - ia * 16) // 4
    var ib = c - ia * 16 - ig * 4
    var y = a.p0 + b * n
    var mu0 = Float32(0.0)
    if has_mean:
        var s = Float32(0.0)
        for i in range(n):
            s = add(s, ld(y, i))
        mu0 = div(s, Float32(n))
    # _backcast over r_i = y_i - mu0
    var tau = 75 if n > 75 else n
    var w = Float32(1.0)
    var sw = Float32(0.0)
    var sb = Float32(0.0)
    for i in range(tau):
        var v = sub(ld(y, i), mu0)
        sb = fma3(w, mul(v, v), sb)
        sw = add(sw, w)
        w = mul(w, Float32(0.94))
    var backcast = div(sb, sw)
    # _var_bounds' scalars
    var s1 = Float32(0.0)
    var mx = Float32(0.0)
    for i in range(n):
        var v = sub(ld(y, i), mu0)
        s1 = add(s1, v)
        var v2 = mul(v, v)
        if v2 > mx:
            mx = v2
    var mean = div(s1, Float32(n))
    var ss = Float32(0.0)
    for i in range(n):
        var d = sub(sub(ld(y, i), mu0), mean)
        ss = fma3(d, d, ss)
    var var_ = div(ss, Float32(n))
    var lo_ = div(var_, Float32(1e8))
    var up_min = add(Float32(1.0), mx)
    var up_max = mul(Float32(1e7), up_min)
    # target
    var target = Float32(0.0)
    for i in range(n):
        var v = sub(ld(y, i), mu0)
        target = fma3(v, v, target)
    target = div(target, Float32(n))
    # the candidate: the grid loop's cand[0 .. k - 1] (omega, alpha, gamma, beta)
    var agb = _pers(ib)
    var cw = mul(sub(Float32(1.0), agb), target)
    var ca = Float32(0.0)
    var cg = Float32(0.0)
    var cb = Float32(0.0)
    if p > 0:
        ca = div(_grid(ia), Float32(p))
        agb = sub(agb, _grid(ia))
    if o > 0:
        cg = div(_grid(ig), Float32(o))
        agb = sub(agb, div(_grid(ig), Float32(2.0)))
    if q > 0:
        cb = div(agb, Float32(q))
    # garch_nll_reg with _var_bounds' EWMA bounds formed per step
    var ll = Float32(0.0)
    var rp = Float32(0.0)
    var sp = Float32(0.0)
    var e = backcast
    for tt in range(n):
        if tt > 0:
            e = fma3(Float32(0.06), mul(rp, rp), mul(Float32(0.94), e))
        var b0 = div(e, Float32(1e6))
        var b1 = mul(e, Float32(1e6))
        if b0 < lo_:
            b0 = lo_
        if b1 < up_min:
            b1 = up_min
        if b1 > up_max:
            b1 = up_max
        var v = _garch_step_reg(cw, ca, cg, cb, p, o, q, tt, rp, sp, backcast, b0, b1)
        var x = sub(ld(y, tt), mu0)
        ll = add(ll, add(add(LOG_2PI, ftz(identical_log(v))), div(mul(x, x), v)))
        rp = x
        sp = v
    ll = mul(Float32(0.5), ll)
    if not (ll <= Float32(3.0e38)):
        ll = Float32(3.0e38)
    st(a.p6, t, ll)


def op_garch(t: Int, a: Args):
    """Series t. p0 y [B, n]; p1 params out [B, 1 + 1 + p + o + q]
    (mu, omega, alpha, gamma, beta; mu 0 for a zero mean); p2 info [B, 4]
    out (loglik, iterations); p3 sigma [B, n] out (conditional
    volatility); p4 variance forecast [B, h] out; p5 scratch [B, stride].
    i0 n, i1 h, i2 p, i3 o, i4 q, i5 constant mean, i6 stride.
    MOJOLEARN_SEQ_GARCH_GRID: i8 1 runs element t as a grid cell
    (`_garch_grid_cell`, p6 [B, GARCH_GRID_N] out); i8 2 takes the
    starting values from p6's argmin instead of the serial grid."""
    comptime if GARCH_GRID:
        if a.i8 == 1:
            _garch_grid_cell(t, a)
            return
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
    var grid_done = False
    comptime if GARCH_GRID:
        if a.i8 == 2:
            # the grid's values from p6, in candidate order, the serial
            # loop's strict `<`: the first lowest candidate wins
            grid_done = True
            var nl = a.p6 + t * GARCH_GRID_N
            var bc = -1
            for c in range(GARCH_GRID_N):
                var gv = ld(nl, c)
                if gv < best:
                    best = gv
                    bc = c
            if bc >= 0:
                var ia = bc // 16
                var ig = (bc - ia * 16) // 4
                var ib = bc - ia * 16 - ig * 4
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
                for j in range(k):
                    st(x, j + (1 if has_mean else 0), ld(cand, j))
    if not grid_done:
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
    # the cycle watch's snapshot: the last GARCH_SNAP floats of the row
    var snap = a.p5 + t * a.i6 + (a.i6 - GARCH_SNAP)
    # i7 / f0: the FAST stall stop (sequence/nm.mojo; compiled out of IDENTICAL)
    var it = nelder_mead(obj, x, lo, hi, np_, nm_scr, Float32(0.05), Float32(1e-4), 2000, Float32(1e-6), snap,
                         stall_iters=a.i7, stall_rel=a.f0)
    # one restart from the optimum (a fresh simplex around it): the
    # likelihood is flat along the persistence ridge and a single simplex
    # can stall short of the maximum
    it += nelder_mead(obj, x, lo, hi, np_, nm_scr, Float32(0.05), Float32(1e-4), 2000, Float32(1e-6), snap,
                      stall_iters=a.i7, stall_rel=a.f0)
    var nll = Float32(0.0)
    comptime if GARCH_REG:
        nll = obj.eval_stored(x)     # the sigma output and the forecast read r and s2
    else:
        nll = obj.eval(x)
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
