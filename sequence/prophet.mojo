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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_cos, identical_div, identical_exp, identical_sin, identical_sqrt
from std.sys.compile import is_defined
from std.memory import bitcast
from checks.soft_f64 import SF64_NAN, SF64_ONE, SF64_SIGN, SF64_ZERO, sf64_add, sf64_div, sf64_lt, sf64_sub, sf64_to_f32

comptime TWO_PI: Float32 = 6.283185307179586
comptime MEM = 5

#: lane/fam2-timeseries (2026-10-04), IDENTICAL ON EVERY VENDOR AND THE HOST
#: COLUMN: the fit is the block-cooperative one (`sequence/prophet_coop.mojo`,
#: one block of COOP_NT threads per series: every likelihood accumulator a
#: strided fold per thread, a 32-lane butterfly, then the simdgroups'
#: partials ascending; every L-BFGS dot product a lane-strided fold and a
#: butterfly). It was FAST + Apple only because its fold orders differ from
#: the one-thread chain; old bits do not bind a new version, so the ORDER
#: CHANGES EVERYWHERE TOGETHER: NVIDIA, AMD and Apple run the cooperative
#: kernel (PROPHET_COOP, which reads this switch) and the host column's
#: `op_prophet_fit` replays the same folds in the same order one after
#: another (`_coop_fg`, `_dot_coop` below). Same model, priors, start, line
#: search and stopping rules. `-D MOJOLEARN_IDN_PROPHET_COOP_OFF=1` (or
#: MOJOLEARN_IDN_ALL_OFF) restores the one-thread chain on the device
#: (prophet_fit_team) and in the host column.
comptime PROPHET_COOP_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_PROPHET_COOP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: the cooperative block's threads and simdgroup width, restated for the
#: host replay; prophet_coop.mojo's PROPHET_COOP_TPB and PW are these
comptime COOP_NT = 256
comptime COOP_PW = 32
#: accumulators the host replay folds per pass over the points
comptime COOP_JB = 128


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


struct ProphetData(ImplicitlyCopyable, Movable):
    var t: FP        # [N] scaled time
    var X: FP        # [N, K]
    var cp: FP       # [S] changepoint times (scaled)
    var sig: FP      # [K] prior scales
    var N: Int
    var K: Int
    var S: Int
    var tau: Float32
    var mult: Bool

    # Metal: pointer-taking callees on a kernel path must inline (an
    # out-of-line call can mis-read the t/X/cp/sig pointers; lane
    # apple-fast-prophetfix, prophet_fit_team quit L-BFGS at iteration 1)
    @always_inline
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
    for i in range(lo, hi):
        var ti = ld(d.t, i)
        var tr = fma3(k, ti, m)
        for j in range(S):
            var c = ld(d.cp, j)
            if ti >= c:
                tr = fma3(ld(th, 2 + j), sub(ti, c), tr)
        var se = Float32(0.0)
        for q in range(K):
            se = fma3(ld(d.X, i * K + q), ld(th, 3 + S + q), se)
        var yhat: Float32
        if d.mult:
            yhat = fma3(tr, se, tr)
        else:
            yhat = add(tr, se)
        var r = sub(ld(y, i), yhat)
        sse = fma3(r, r, sse)
        var w = div(-r, s2)
        var wt = mul(w, add(Float32(1.0), se)) if d.mult else w
        st(g, 0, fma3(wt, ti, ld(g, 0)))
        st(g, 1, add(ld(g, 1), wt))
        for j in range(S):
            var c = ld(d.cp, j)
            if ti >= c:
                st(g, 2 + j, fma3(wt, sub(ti, c), ld(g, 2 + j)))
        var wb = mul(w, tr) if d.mult else w
        for q in range(K):
            st(g, 3 + S + q, fma3(wb, ld(d.X, i * K + q), ld(g, 3 + S + q)))
    return sse


@always_inline
def _fg_prior_v(sig: FP, tau: Float32, N: Int, S: Int, K: Int, th: FP, g: FP, sse: Float32) -> Float32:
    """The priors and the sigma terms on top of the likelihood's gradient in
    g and its sse; returns the objective. The scalars and the prior-scale
    pointer by value (no struct on the Metal team path, lane
    apple-fast-prophetfix)."""
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
        f = add(f, div(abs(dl), tau))
        var sg = Float32(0.0)
        if dl > Float32(0.0):
            sg = Float32(1.0)
        elif dl < Float32(0.0):
            sg = Float32(-1.0)
        st(g, 2 + j, add(ld(g, 2 + j), div(sg, tau)))
    for q in range(K):
        var b = ld(th, 3 + S + q)
        var sq = mul(ld(sig, q), ld(sig, q))
        f = add(f, div(mul(b, b), mul(Float32(2.0), sq)))
        st(g, 3 + S + q, add(ld(g, 3 + S + q), div(b, sq)))
    # sigma: prior N(0, 0.5), likelihood N log sigma + sse / (2 sigma^2)
    f = add(f, div(s2, Float32(0.5)))
    f = fma3(Float32(N), u, f)
    f = add(f, div(sse, mul(Float32(2.0), s2)))
    st(g, 2 + S, add(sub(add(div(s2, Float32(0.25)), Float32(N)), div(sse, s2)), Float32(0.0)))
    return f


@always_inline
def _fg_prior(d: ProphetData, th: FP, g: FP, sse: Float32) -> Float32:
    """The priors and the sigma terms on top of the likelihood's gradient in
    g and its sse; returns the objective (`_fg_prior_v` on d's fields)."""
    return _fg_prior_v(d.sig, d.tau, d.N, d.S, d.K, th, g, sse)


@always_inline
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


@always_inline
def _bitrev5(c: Int) -> Int:
    """The 5-bit reversal: leaf c of the butterfly's sum tree is lane
    _bitrev5(c) (pairs 16 apart first, then 8, 4, 2, 1)."""
    return ((c & 1) << 4) | ((c & 2) << 2) | (c & 4) | ((c & 8) >> 2) | ((c & 16) >> 4)


@always_inline
def _dot_coop(a: FP, b: FP, n: Int) -> Float32:
    """prophet_coop's `_cdot` replayed by one thread: each lane's strided
    fma chain from 0 (coordinates lane, lane + 32, ...), then `_wsum`'s
    butterfly, whose value in every lane is the pairwise tree over the
    lanes taken in bit-reversed order (float addition commutes)."""
    var stack = InlineArray[Float32, 5](fill=Float32(0.0))
    var out = Float32(0.0)
    for c in range(COOP_PW):
        var s = Float32(0.0)
        var i = _bitrev5(c)
        while i < n:
            s = fma3(ld(a, i), ld(b, i), s)
            i += COOP_PW
        var level = 0
        while level < 5 and ((c >> level) & 1) == 1:
            s = add(stack[level], s)
            level += 1
        if level < 5:
            stack[level] = s
        else:
            out = s
    return out


@always_inline
def _dotx[COOP: Bool](a: FP, b: FP, n: Int) -> Float32:
    comptime if COOP:
        return _dot_coop(a, b, n)
    return _dot(a, b, n)


@always_inline
def _coop_fg(d: ProphetData, y: FP, th: FP, g: FP) -> Float32:
    """`CoopFG.fg` (sequence/prophet_coop.mojo) replayed by one thread, the
    host column of the cooperative fit: accumulator j of thread tid is its
    chain over points tid, tid + COOP_NT, ... from 0; a simdgroup's 32 chains
    are summed by the butterfly's tree; the COOP_NT / 32 partials are added
    ascending from 0; then the priors per coordinate and the objective's
    prior terms folded per lane and summed by the butterfly. The point
    values (residual, weights) are pass A's, recomputed per block of COOP_JB
    accumulators so no scratch beyond the stack is needed."""
    var S = d.S
    var K = d.K
    var N = d.N
    var P = 3 + S + K
    var nw = COOP_NT // COOP_PW
    var k = ld(th, 0)
    var m = ld(th, 1)
    var u = ld(th, 2 + S)
    var sigma = ftz(identical_exp(u))
    var s2 = mul(sigma, sigma)
    for j in range(P):
        st(g, j, Float32(0.0))
    var sse = Float32(0.0)
    var acc = InlineArray[Float32, COOP_JB](fill=Float32(0.0))
    var stack = InlineArray[Float32, 5 * COOP_JB](fill=Float32(0.0))
    var j0 = 0
    while j0 < P + 1:
        var jn = P + 1 - j0
        if jn > COOP_JB:
            jn = COOP_JB
        for wid in range(nw):
            for c in range(COOP_PW):
                var tid = wid * COOP_PW + _bitrev5(c)
                for jj in range(jn):
                    acc[jj] = Float32(0.0)
                var i = tid
                while i < N:
                    var ti = ld(d.t, i)
                    var tr = fma3(k, ti, m)
                    for j in range(S):
                        var cj = ld(d.cp, j)
                        if ti >= cj:
                            tr = fma3(ld(th, 2 + j), sub(ti, cj), tr)
                    var se = Float32(0.0)
                    for q in range(K):
                        se = fma3(ld(d.X, i * K + q), ld(th, 3 + S + q), se)
                    var yhat: Float32
                    if d.mult:
                        yhat = fma3(tr, se, tr)
                    else:
                        yhat = add(tr, se)
                    var r = sub(ld(y, i), yhat)
                    var w = div(-r, s2)
                    var wt = mul(w, add(Float32(1.0), se)) if d.mult else w
                    var wb = mul(w, tr) if d.mult else w
                    for jj in range(jn):
                        var j = j0 + jj
                        if j == 0:
                            acc[jj] = fma3(wt, ti, acc[jj])
                        elif j == 1:
                            acc[jj] = add(acc[jj], wt)
                        elif j < 2 + S:
                            var cj = ld(d.cp, j - 2)
                            if ti >= cj:
                                acc[jj] = fma3(wt, sub(ti, cj), acc[jj])
                        elif j == 2 + S:
                            pass
                        elif j < P:
                            acc[jj] = fma3(wb, ld(d.X, i * K + (j - 3 - S)), acc[jj])
                        else:
                            acc[jj] = fma3(r, r, acc[jj])
                    i += COOP_NT
                # the butterfly's tree, one leaf (lane) at a time
                for jj in range(jn):
                    var v = acc[jj]
                    var level = 0
                    while level < 5 and ((c >> level) & 1) == 1:
                        v = add(stack[level * COOP_JB + jj], v)
                        level += 1
                    if level < 5:
                        stack[level * COOP_JB + jj] = v
                    else:
                        # the simdgroup's partial, folded ascending over wid
                        var j = j0 + jj
                        if j < P:
                            st(g, j, add(ld(g, j), v))
                        else:
                            sse = add(sse, v)
        j0 += jn
    # the priors: each lane folds its own coordinates' terms, then the butterfly
    var fstack = InlineArray[Float32, 5](fill=Float32(0.0))
    var f = Float32(0.0)
    for c in range(COOP_PW):
        var fc = Float32(0.0)
        var i = _bitrev5(c)
        while i < P:
            var gi = ld(g, i)
            var x = ld(th, i)
            if i == 0 or i == 1:
                fc = add(fc, div(mul(x, x), Float32(50.0)))
                gi = add(gi, div(x, Float32(25.0)))
            elif i < 2 + S:
                fc = add(fc, div(abs(x), d.tau))
                var sg = Float32(0.0)
                if x > Float32(0.0):
                    sg = Float32(1.0)
                elif x < Float32(0.0):
                    sg = Float32(-1.0)
                gi = add(gi, div(sg, d.tau))
            elif i == 2 + S:
                gi = sub(add(div(s2, Float32(0.25)), Float32(N)), div(sse, s2))
            else:
                var sq = mul(ld(d.sig, i - 3 - S), ld(d.sig, i - 3 - S))
                fc = add(fc, div(mul(x, x), mul(Float32(2.0), sq)))
                gi = add(gi, div(x, sq))
            st(g, i, gi)
            i += COOP_PW
        var level = 0
        while level < 5 and ((c >> level) & 1) == 1:
            fc = add(fstack[level], fc)
            level += 1
        if level < 5:
            fstack[level] = fc
        else:
            f = fc
    # sigma: prior N(0, 0.5), likelihood N log sigma + sse / (2 sigma^2)
    f = add(f, div(s2, Float32(0.5)))
    f = fma3(Float32(N), u, f)
    f = add(f, div(sse, mul(Float32(2.0), s2)))
    return f


trait ProphetFG:
    """The objective `lbfgs_steps` minimises: -log posterior at th and its
    gradient into g (`prophet_fg`, or the block-cooperative form of
    sequence/fit_team.mojo)."""

    @always_inline
    def fg(mut self, th: FP, g: FP) -> Float32:
        ...


struct PlainFG(ProphetFG):
    """prophet_fg over one series, in one thread."""
    var d: ProphetData
    var y: FP

    @always_inline
    def __init__(out self, d: ProphetData, y: FP):
        self.d = d
        self.y = y

    @always_inline
    def fg(mut self, th: FP, g: FP) -> Float32:
        return prophet_fg(self.d, self.y, th, g)


struct LBState(ImplicitlyCopyable, Movable):
    """`lbfgs_prophet`'s loop state between iterations (lane neural-pass143);
    with th, g and the pairs in w, everything the next iteration reads."""
    var f: Float32
    var npairs: Int
    var head: Int
    var small: Int
    var it: Int
    var done: Bool

    @always_inline
    def __init__(out self, f: Float32):
        self.f = f
        self.npairs = 0
        self.head = 0
        self.small = 0
        self.it = 0
        self.done = False


struct CoopOrderFG(ProphetFG):
    """The cooperative fit's objective replayed by one thread (`_coop_fg`):
    the host column under PROPHET_COOP_IDN."""
    var d: ProphetData
    var y: FP

    @always_inline
    def __init__(out self, d: ProphetData, y: FP):
        self.d = d
        self.y = y

    @always_inline
    def fg(mut self, th: FP, g: FP) -> Float32:
        return _coop_fg(self.d, self.y, th, g)


@always_inline
def lbfgs_start[F: ProphetFG](mut fg: F, th: FP, w: FP) -> LBState:
    """The objective at the start point (gradient into w's g)."""
    return LBState(fg.fg(th, w))


@always_inline
def lbfgs_steps[F: ProphetFG, COOP: Bool = False](mut fg: F, mut s: LBState, P: Int, th: FP, w: FP,
                                                 max_iter: Int, budget: Int = -1) -> Int:
    """`lbfgs_prophet`'s iterations from state `s`: at most `budget` of them
    (all when budget < 0); `s.done` says whether the loop ended. Returns the
    number run here. COOP (lane/fam2-timeseries): every dot product is the
    cooperative block's (`_dot_coop`); everything else of
    `lbfgs_coop_steps` is this loop statement for statement (its elementwise
    updates are per coordinate, its gradient maximum is order-free)."""
    var g = w
    var dvec = g + P
    var thn = dvec + P
    var gn = thn + P
    var q = gn + P
    var sm = q + P
    var ym = sm + MEM * P
    var rho = ym + MEM * P
    var al = rho + MEM
    var f = s.f
    var npairs = s.npairs
    var head = s.head
    var small = s.small
    var it = s.it
    var steps = 0
    var finished = True
    while it < max_iter:
        # the slice boundary (lane neural-pass143): the state is exactly
        # what `s` and w hold here, so a later call resumes with the same
        # iteration
        if budget >= 0 and steps >= budget:
            finished = False
            break
        steps += 1
        # two-loop recursion: q = g; newest to oldest, then oldest to newest
        for i in range(P):
            st(q, i, ld(g, i))
        var idx = head
        for _ in range(npairs):
            idx = (idx - 1 + MEM) % MEM
            var aa = mul(ld(rho, idx), _dotx[COOP](sm + idx * P, q, P))
            st(al, idx, aa)
            for i in range(P):
                st(q, i, sub(ld(q, i), mul(aa, ld(ym + idx * P, i))))
        var gamma = Float32(1.0)
        if npairs > 0:
            var last = (head - 1 + MEM) % MEM
            var yy = _dotx[COOP](ym + last * P, ym + last * P, P)
            if yy > Float32(0.0):
                gamma = div(_dotx[COOP](sm + last * P, ym + last * P, P), yy)
        else:
            var gg = ftz(identical_sqrt(_dotx[COOP](g, g, P)))
            if gg > Float32(1.0):
                gamma = div(Float32(1.0), gg)
        for i in range(P):
            st(q, i, mul(gamma, ld(q, i)))
        var start = (head - npairs + MEM) % MEM
        idx = start
        for _ in range(npairs):
            var bb = mul(ld(rho, idx), _dotx[COOP](ym + idx * P, q, P))
            var coef = sub(ld(al, idx), bb)
            for i in range(P):
                st(q, i, fma3(coef, ld(sm + idx * P, i), ld(q, i)))
            idx = (idx + 1) % MEM
        for i in range(P):
            st(dvec, i, -ld(q, i))
        var gd = _dotx[COOP](g, dvec, P)
        if not (gd < Float32(0.0)):
            for i in range(P):
                st(dvec, i, -ld(g, i))
            gd = -_dotx[COOP](g, g, P)
            npairs = 0
        # backtracking Armijo
        var step = Float32(1.0)
        var fnew = Float32(0.0)
        var ok = False
        for _ in range(40):
            for i in range(P):
                st(thn, i, fma3(step, ld(dvec, i), ld(th, i)))
            fnew = fg.fg(thn, gn)
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
        var sy = _dotx[COOP](q, dvec, P)
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
    s.f = f
    s.npairs = npairs
    s.head = head
    s.small = small
    s.it = it
    s.done = finished
    return steps


@always_inline
def lbfgs_prophet(d: ProphetData, y: FP, th: FP, w: FP, max_iter: Int) -> Tuple[Float32, Int]:
    """Minimise prophet_fg from th (in place). w: scratch of
    (6 + 2 MEM) P + 2 MEM floats. Returns (f, iterations)."""
    var P = 3 + d.S + d.K
    comptime if PROPHET_COOP_IDN:
        # the cooperative fit's fold orders, replayed (the host column)
        var cfg = CoopOrderFG(d, y)
        var cs = lbfgs_start(cfg, th, w)
        _ = lbfgs_steps[CoopOrderFG, True](cfg, cs, P, th, w, max_iter)
        return (cs.f, cs.it)
    var fg = PlainFG(d, y)
    var s = lbfgs_start(fg, th, w)
    _ = lbfgs_steps(fg, s, P, th, w, max_iter)
    return (s.f, s.it)


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


# ---- input preparation on the executor (lane cpu4-python) ------------------
# `prophet_features` computed t = (days - start) / t_scale and the seasonal
# phases fmod(days, P) / P (+ 1 when negative) in float64 on the host, one
# rounding to float32 each. Here every float64 step is soft binary64 over the
# IEEE words (checks/soft_f64.mojo: correctly rounded, integer instructions
# only, so Apple without float64 too) and the fmod is the exact integer long
# division it was: the IEEE results the host float64 unit gave, the same
# bits, on every column.


@always_inline
def _words64(a: FP, i: Int) -> UInt64:
    """Float64 i of a buffer holding float64 words as float32 pairs (low
    word first: the little-endian layout of the caller's float64 array)."""
    var lo = UInt64(bitcast[DType.uint32](a.unsafe_load(2 * i)))
    var hi = UInt64(bitcast[DType.uint32](a.unsafe_load(2 * i + 1)))
    return (hi << 32) | lo


@always_inline
def _arg64(a: Args, at: Int) -> UInt64:
    """A 64-bit word carried as four 16-bit Int slots i[at..at+3], low first."""
    var w0: Int
    var w1: Int
    var w2: Int
    var w3: Int
    if at == 2:
        w0 = a.i2; w1 = a.i3; w2 = a.i4; w3 = a.i5
    else:
        w0 = a.i6; w1 = a.i7; w2 = a.i8; w3 = a.i9
    return (UInt64(w0 & 0xFFFF) | (UInt64(w1 & 0xFFFF) << 16)
            | (UInt64(w2 & 0xFFFF) << 32) | (UInt64(w3 & 0xFFFF) << 48))


def fmod_exact64_bits(ua: UInt64, ub: UInt64) -> UInt64:
    """C fmod(a, b) on IEEE words for a finite b > 0, exact (the sign of a),
    by integer long division of the significands; a NaN or infinite a gives
    the canonical NaN. `sequence/prophet_prep.mojo::fmod_exact64` on words."""
    var ea = Int((ua >> 52) & UInt64(0x7FF))
    var eb = Int((ub >> 52) & UInt64(0x7FF))
    if ea == 0x7FF:
        return SF64_NAN
    if (ua & ~SF64_SIGN) < ub:
        return ua
    var ma = ua & UInt64(0xFFFFFFFFFFFFF)
    var mb = ub & UInt64(0xFFFFFFFFFFFFF)
    if ea == 0:
        ea = 1
    else:
        ma |= UInt64(1) << 52
    if eb == 0:
        eb = 1
    else:
        mb |= UInt64(1) << 52
    var r = ma % mb
    for _ in range(ea - eb):
        r = (r << 1) % mb
    var bits = UInt64(0)
    if r != 0:
        var sh = 0
        while (r << UInt64(sh)) < (UInt64(1) << 52):
            sh += 1
        var m = r << UInt64(sh)
        var E = eb - sh
        if E >= 1:
            bits = (UInt64(E) << 52) | (m & UInt64(0xFFFFFFFFFFFFF))
        else:
            bits = m >> UInt64(1 - E)
    return bits | (ua & SF64_SIGN)


def op_prophet_prep(t: Int, a: Args):
    """Element t of an (N, W) grid, W = 1 + max(ns, 1): column 0 writes
    p2[r] = f32((days[r] - start) / t_scale); column 1 + s writes
    p3[r, s] = f32(fmod(days[r], P_s) / P_s, + 1 when negative), or 0 when
    ns == 0. p0: days (N float64 as float32 pairs), p1: periods (ns float64
    as pairs); i0 = N, i1 = ns; start in i2..i5, t_scale in i6..i9 (16-bit
    words, low first)."""
    var ns = a.i1
    var W = 1 + (ns if ns > 0 else 1)
    var r = t // W
    var c = t - r * W
    var d = _words64(a.p0, r)
    if c == 0:
        var v = sf64_div(sf64_sub(d, _arg64(a, 2)), _arg64(a, 6))
        a.p2.unsafe_store(r, sf64_to_f32(v))
        return
    var s = c - 1
    if ns == 0:
        a.p3.unsafe_store(r, Float32(0.0))
        return
    var P = _words64(a.p1, s)
    var q = sf64_div(fmod_exact64_bits(d, P), P)
    if sf64_lt(q, SF64_ZERO):
        q = sf64_add(q, SF64_ONE)
    a.p3.unsafe_store(r * ns + s, sf64_to_f32(q))
