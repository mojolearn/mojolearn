# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ETS without a season on one 32-lane cell per series with the Nelder-Mead
candidates evaluated side by side: the IDENTICAL twin of theta_spec's
`nm_spec`, -D MOJOLEARN_IDN_NM_SPEC (lane seq-nm-spec 2026-10-11, default
off; idea E1).

Structural reason: `op_ets` (sequence/ets.mojo) runs one thread per series
and statsforecast's Nelder-Mead evaluates its candidates one after another
(reflection, then expansion or a contraction, 2 + k whole-series passes on
a shrink): a latency-bound serial chain. Every candidate of an iteration is
a function of the simplex alone, and with no season the objective
(`ets_lik`) has no side effect (the seasonal ring is never touched), so
lanes 0..3 of the cell evaluate the reflection, expansion, outside and
inside contraction AT ONCE (lanes 0..k the k + 1 initial and shrunk
vertices), the values are broadcast (`shuffle_idx`) and every lane replays
`nm_steps`' decision on them: the same sort, stop test, fixed-point rule
(no stall rule on IDENTICAL, no snapshot for ETS). Each accepted value is
the serial objective on the same words, so the fit is op_ets' fit bit for
bit: NO bit moves on NVIDIA, AMD or the host column (the host keeps the
serial op_ets). Expected: damped-ets AMD -50% (critical path from ~1.5-2
evaluations per iteration to 1).

Only the no-season forms ETS(A|M, N|A|Ad, N) take this path (k <= 5
coordinates: alpha, beta, phi, l0, b0); a seasonal model keeps op_ets
(the ring is shared memory the candidates would race on). The dispatch
(sequence/exec_device.mojo) decides on the model form (a.i6), never on a
size."""
from std.gpu.primitives.warp import shuffle_idx
from std.memory import bitcast
from std.sys.compile import is_defined

from sequence.coop_lane import coop_src
from sequence.ets import SEAS_N, ets_init_state, ets_lik
from sequence.ops import FP, Args, add, fma3, ld, mul st, sub
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_pow, identical_sqrt

comptime ETS_SPEC = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_NM_SPEC"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: coordinate capacity (no season: alpha, beta, phi, l0, b0 <= 5)
comptime EK = 6
#: simplex capacity ((EK + 1) EK)
comptime ES = 42


@fieldwise_init
struct EtsEv(ImplicitlyCopyable, Movable):
    """`EtsObj.unpack` + `EtsObj.eval` for a model without a season, on
    coordinates held in registers."""
    var y: FP
    var n: Int
    var err: Int
    var trend: Bool
    var ring: FP
    var oa: Bool
    var ob: Bool
    var op: Bool
    var alpha: Float32
    var beta: Float32
    var phi: Float32

    @always_inline
    def unpack(self, c: InlineArray[Float32, EK]) -> Tuple[Float32, Float32, Float32, Float32, Float32]:
        """(alpha, beta, phi, l0, b0), EtsObj.unpack's order (gamma is
        never free without a season)."""
        var j = 0
        var a = self.alpha
        var b = self.beta
        var p = self.phi
        if self.oa:
            a = c[j]
            j += 1
        if self.ob:
            b = c[j]
            j += 1
        if self.op:
            p = c[j]
            j += 1
        var l0 = c[j]
        var b0 = c[j + 1] if self.trend else Float32(0.0)
        return (a, b, p, l0, b0)

    @always_inline
    def ev(self, c: InlineArray[Float32, EK]) -> Float32:
        var u = self.unpack(c)
        var r = ets_lik(self.y, self.n, self.err, self.trend, SEAS_N, 1, self.ring,
                        u[0], u[1], Float32(0.0), u[2], u[3], u[4])
        return r[0] if r[0] > Float32(-1e10) else Float32(-1e10)


@always_inline
def _clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var r = v if v > lo else lo
    return r if r < hi else hi


@always_inline
def _put(mut arr: InlineArray[Float32, ES], i: Int, v: Float32) -> Bool:
    var old = bitcast[DType.uint32](arr[i])
    var w = ftz(v)
    arr[i] = w
    return old != bitcast[DType.uint32](w)


@always_inline
def _putf(mut arr: InlineArray[Float32, EK + 1], i: Int, v: Float32) -> Bool:
    var old = bitcast[DType.uint32](arr[i])
    var w = ftz(v)
    arr[i] = w
    return old != bitcast[DType.uint32](w)


@always_inline
def _vertex(S: InlineArray[Float32, ES], r: Int, k: Int) -> InlineArray[Float32, EK]:
    var c = InlineArray[Float32, EK](fill=Float32(0.0))
    for j in range(k):
        c[j] = S[r * k + j]
    return c^


@always_inline
def nm_spec_ets(
    ev: EtsEv, mut x: InlineArray[Float32, EK], lo: InlineArray[Float32, EK], hi: InlineArray[Float32, EK],
    k: Int, lane: Int, init_step: Float32, zero_pert: Float32, max_iter: Int, tol_std: Float32,
) -> Int:
    """`nelder_mead` (sequence/nm.mojo; k <= EK, IDENTICAL: no stall rule,
    no snapshot) with the candidates of each step evaluated across lanes;
    x receives the best vertex; returns the iteration count. Every lane
    calls it and every lane returns the same words."""
    var S = InlineArray[Float32, ES](fill=Float32(0.0))
    var F = InlineArray[Float32, EK + 1](fill=Float32(0.0))
    # nm_start: the clamped start in every row, the diagonal perturbed
    for i in range(k + 1):
        for j in range(k):
            S[i * k + j] = ftz(_clamp(x[j], lo[j], hi[j]))
    for i in range(k):
        var v = S[i * k + i]
        if v == Float32(0.0):
            v = zero_pert
        else:
            v = mul(v, add(Float32(1.0), init_step))
        S[i * k + i] = ftz(_clamp(v, lo[i], hi[i]))
    var r0 = lane if lane <= k else 0
    var f0 = ev.ev(_vertex(S, r0, k))
    for i in range(EK + 1):
        var v = shuffle_idx(f0, coop_src(lane, i))
        if i <= k:
            F[i] = ftz(v)
    var nf = Float32(k)
    var gamma = add(Float32(1.0), ftz(identical_div(Float32(2.0), nf)))
    var rho = sub(Float32(0.75), ftz(identical_div(Float32(1.0), mul(Float32(2.0), nf))))
    var sigma = sub(Float32(1.0), ftz(identical_div(Float32(1.0), nf)))
    var xo = InlineArray[Float32, EK](fill=Float32(0.0))
    var xr = InlineArray[Float32, EK](fill=Float32(0.0))
    var xe = InlineArray[Float32, EK](fill=Float32(0.0))
    var xc = InlineArray[Float32, EK](fill=Float32(0.0))
    var xi = InlineArray[Float32, EK](fill=Float32(0.0))
    var order = InlineArray[Int, EK + 1](fill=0)
    var changed = True
    var it = 0
    var best = 0
    var role = lane if lane < 4 else 0
    while it < max_iter:
        # stable argsort of F (ties: lower index first)
        for i in range(k + 1):
            order[i] = i
        for i in range(1, k + 1):
            var kk = order[i]
            var j = i - 1
            while j >= 0 and F[order[j]] > F[kk]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = kk
        best = order[0]
        var worst = order[k]
        var second = order[k - 1]
        # population standard deviation of F
        var mean = Float32(0.0)
        for i in range(k + 1):
            mean = add(mean, F[i])
        mean = ftz(identical_div(mean, Float32(k + 1)))
        var ss = Float32(0.0)
        for i in range(k + 1):
            var d = sub(F[i], mean)
            ss = fma3(d, d, ss)
        if ftz(identical_sqrt(ftz(identical_div(ss, Float32(k + 1))))) < tol_std:
            break
        if not changed:
            it = max_iter
            break
        changed = False
        # the centroid without the worst vertex (st/ld round trips are ftz)
        for j in range(k):
            var s = Float32(0.0)
            for i in range(k + 1):
                s = add(s, S[i * k + j])
            xo[j] = ftz(ftz(identical_div(sub(s, S[worst * k + j]), nf)))
        for j in range(k):
            var o = xo[j]
            xr[j] = ftz(_clamp(add(o, sub(o, S[worst * k + j])), lo[j], hi[j]))
        for j in range(k):
            var o = xo[j]
            xe[j] = ftz(_clamp(fma3(gamma, sub(xr[j], o), o), lo[j], hi[j]))
            xc[j] = ftz(_clamp(fma3(rho, sub(xr[j], o), o), lo[j], hi[j]))
            xi[j] = ftz(_clamp(sub(o, mul(rho, sub(xr[j], o))), lo[j], hi[j]))
        # nm_steps compares the raw candidate values with the stored (ftz) ones
        var cand = InlineArray[Float32, EK](fill=Float32(0.0))
        for j in range(k):
            if role == 1:
                cand[j] = xe[j]
            elif role == 2:
                cand[j] = xc[j]
            elif role == 3:
                cand[j] = xi[j]
            else:
                cand[j] = xr[j]
        var fl = ev.ev(cand)
        var fr = shuffle_idx(fl, coop_src(lane, 0))
        var fe = shuffle_idx(fl, coop_src(lane, 1))
        var fc = shuffle_idx(fl, coop_src(lane, 2))
        var fi = shuffle_idx(fl, coop_src(lane, 3))
        if F[best] <= fr and fr < F[second]:
            for j in range(k):
                changed |= _put(S, worst * k + j, xr[j])
            changed |= _putf(F, worst, fr)
            it += 1
            continue
        if fr < F[best]:
            if fe < fr:
                for j in range(k):
                    changed |= _put(S, worst * k + j, xe[j])
                changed |= _putf(F, worst, fe)
            else:
                for j in range(k):
                    changed |= _put(S, worst * k + j, xr[j])
                changed |= _putf(F, worst, fr)
            it += 1
            continue
        var accepted = False
        if F[second] <= fr and fr < F[worst]:
            if fc <= fr:
                for j in range(k):
                    changed |= _put(S, worst * k + j, xc[j])
                changed |= _putf(F, worst, fc)
                accepted = True
        else:
            if fi < F[worst]:
                for j in range(k):
                    changed |= _put(S, worst * k + j, xi[j])
                changed |= _putf(F, worst, fi)
                accepted = True
        if not accepted:
            # shrink toward the best vertex; the k shrunk vertices at once
            for i in range(k + 1):
                if i == best:
                    continue
                for j in range(k):
                    var b = S[best * k + j]
                    changed |= _put(S, i * k + j, _clamp(fma3(sigma, sub(S[i * k + j], b), b), lo[j], hi[j]))
            var rs = lane if lane <= k else best
            var fs = ev.ev(_vertex(S, rs, k))
            for i in range(EK + 1):
                var v = shuffle_idx(fs, coop_src(lane, i))
                if i <= k and i != best:
                    changed |= _putf(F, i, v)
        it += 1
    for j in range(k):
        x[j] = S[best * k + j]
    return it + 1


def op_ets_spec(t: Int, lane: Int, a: Args):
    """`op_ets` (sequence/ets.mojo) for series t without a season (a.i6 ==
    SEAS_N, the dispatch's condition) on its cell; `nm_spec_ets` in place
    of `nelder_mead`. Same arguments and outputs as op_ets; lane 0 stores
    them. Every lane computes the same words."""
    var n = a.i0
    var h = a.i1
    var y = a.p0 + t * n
    var trend = a.i3 == 1
    var damped = trend and a.i4 != 0
    # op_ets' scratch layout with m = 1, season none
    var kmax = 6
    var kc = 8
    var sc = a.p3 + t * a.i8
    var nm_scr = sc + 3 * kc
    var ring = nm_scr + (kmax + 1) * kmax + (kmax + 1) + 4 * kmax
    var sinit = ring + 1
    var seas = sinit + 1
    var qa = seas + n
    # initparam (usual bounds), op_ets' chains
    var lo_a = Float32(1e-4)
    var hi_a = Float32(0.9999)
    var fa = (a.i5 & 1) != 0
    var fb = (a.i5 & 2) != 0
    var fp = (a.i5 & 4) != 0
    var alpha: Float32
    if fa:
        alpha = a.f0
    else:
        alpha = fma3(Float32(0.2), sub(hi_a, lo_a), lo_a)
    var hi_b = hi_a if hi_a < alpha else alpha
    var beta = a.f1 if fb else fma3(Float32(0.1), sub(hi_b, lo_a), lo_a)
    var phi = Float32(1.0)
    if damped:
        phi = a.f2 if fp else fma3(Float32(0.99), sub(Float32(0.98), Float32(0.8)), Float32(0.8))
    var st0 = ets_init_state(y, n, trend, SEAS_N, 1, sinit, seas, qa)
    var oa = not fa
    var ob = trend and not fb
    var op = damped and not fp
    # op_ets stores x/lo/hi through st (ftz); the registers hold the same words
    var x = InlineArray[Float32, EK](fill=Float32(0.0))
    var lo = InlineArray[Float32, EK](fill=Float32(0.0))
    var hi = InlineArray[Float32, EK](fill=Float32(0.0))
    var k = 0
    if oa:
        x[k] = ftz(alpha)
        lo[k] = ftz(lo_a)
        hi[k] = ftz(hi_a)
        k += 1
    if ob:
        x[k] = ftz(beta)
        lo[k] = ftz(lo_a)
        hi[k] = ftz(hi_b)
        k += 1
    if op:
        x[k] = ftz(phi)
        lo[k] = ftz(Float32(0.8))
        hi[k] = ftz(Float32(0.98))
        k += 1
    x[k] = ftz(st0[0])
    lo[k] = ftz(Float32(-3.0e38))
    hi[k] = ftz(Float32(3.0e38))
    k += 1
    if trend:
        x[k] = ftz(st0[1])
        lo[k] = ftz(Float32(-3.0e38))
        hi[k] = ftz(Float32(3.0e38))
        k += 1
    var ev = EtsEv(y, n, a.i2, trend, ring, oa, ob, op, alpha, beta, phi)
    var it = nm_spec_ets(ev, x, lo, hi, k, lane, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4))
    var u = ev.unpack(x)
    var r = ets_lik(y, n, a.i2, trend, SEAS_N, 1, ring, u[0], u[1], Float32(0.0), u[2], u[3], u[4])
    if lane != 0:
        return
    var l = r[1]
    var b = r[2]
    var phistar = u[2]
    for i in range(h):
        var f = add(mul(phistar, b), l) if trend else l
        st(a.p1, t * h + i, f)
        if i < h - 1:
            phistar = add(phistar, ftz(identical_pow(u[2], Float32(i + 1))))
    var info = a.p2 + t * 10
    st(info, 0, u[0])
    st(info, 1, u[1] if trend else Float32(0.0))
    st(info, 2, u[2])
    st(info, 3, u[3])
    st(info, 4, u[4])
    st(info, 5, r[0])
    st(info, 6, Float32(it))
    st(info, 7, Float32(k))
    st(info, 8, Float32(0.0))
    st(info, 9, Float32(0.0))
