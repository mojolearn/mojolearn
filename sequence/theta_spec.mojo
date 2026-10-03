# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Theta on one simdgroup per series with the Nelder-Mead candidates
evaluated side by side (lane/apple-fast-gap-tsa,
FAST + Apple, on THETA_REG; default on, -D MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF off;
docs/apple-fast/notes/gap-tsa.md).

`op_theta` runs one thread per series (64 threads on the board's taxi-hourly
cell) and inside it statsforecast's Nelder-Mead evaluates its candidates one
after another: an iteration that reflects and then expands or contracts
runs two whole-series objective passes back to back, a shrink 2 + k. Every
candidate of an iteration is a function of the simplex alone (reflection,
expansion, outside and inside contraction from the centroid and the worst
vertex), and the objective has no side effect in its hoisted form
(`theta_sse_hoisted`), so here lanes 0..3 of the series' simdgroup evaluate
the four candidates AT ONCE (and lanes 0..k the k + 1 initial vertices, and
the shrunk vertices), the values are broadcast (`shuffle_idx`), and every
lane runs `nm_steps`' decision on them. The simplex, its values and the
cycle snapshot live in each lane's registers (every lane holds the same
words), so no lane reads another's memory. Each objective value is the
serial value (the same function on the same words) and the decision is
`nm_steps`' (same sort, stop test, fixed-point and cycle rules; theta's
stall rule is off), so the fit is the serial fit, bit for bit: the
evaluations the serial run skips are simply not used. Lanes 4..31 repeat
lane 0's point. Everything else (the ACF test, the decomposition, the
final runs, the forecast) runs on every lane identically; lane 0 stores
the outputs."""
from std.gpu.primitives.warp import shuffle_idx
from std.memory import bitcast
from std.sys.compile import is_defined

from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from sequence.theta import (
    DOTM, DSTM, OTM, STM, THETA_REG, THETA_SNAP, _acf_decide, _decompose, div, theta_forecast_reg,
    theta_invariants, theta_run_reg, theta_sse_hoisted,
)
from checks.numerics import ftz, identical_div, identical_sqrt

#: the switch (FAST + Apple with THETA_REG). Default since the M3 A/B (n=1,
#: quality identical; theta taxi-hourly 218 -> 20.3 ms);
#: -D MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF turns it off; the old
#: -D MOJOLEARN_SEQ_FAST_THETA_SPEC is harmless.
comptime THETA_SPEC = THETA_REG and not is_defined["MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF"]()


@fieldwise_init
struct ThetaEv(ImplicitlyCopyable, Movable):
    """`ThetaObj.params` + `ThetaObj.eval` (hoisted) on three coordinates."""
    var y: FP
    var n: Int
    var dyn: Bool
    var ol: Bool
    var oa: Bool
    var ot: Bool
    var level: Float32
    var alpha: Float32
    var theta: Float32
    var hA: Float32
    var hB: Float32
    var hmean: Float32

    @always_inline
    def params(self, x0: Float32, x1: Float32, x2: Float32) -> Tuple[Float32, Float32, Float32]:
        var j = 0
        var l = self.level
        var a = self.alpha
        var t = self.theta
        if self.ol:
            l = x0
            j += 1
        if self.oa:
            a = x0 if j == 0 else x1
            j += 1
        if self.ot:
            t = x0 if j == 0 else (x1 if j == 1 else x2)
        return (l, a, t)

    @always_inline
    def ev(self, x0: Float32, x1: Float32, x2: Float32) -> Float32:
        var p = self.params(x0, x1, x2)
        var mse: Float32
        if self.dyn:
            mse = theta_sse_hoisted[True](self.y, self.n, p[0], p[1], p[2], self.hA, self.hB, self.hmean)
        else:
            mse = theta_sse_hoisted[False](self.y, self.n, p[0], p[1], p[2], self.hA, self.hB, self.hmean)
        return mse if mse > Float32(-1e10) else Float32(-1e10)


@always_inline
def _clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var r = v if v > lo else lo
    return r if r < hi else hi


@always_inline
def _put(mut arr: InlineArray[Float32, 12], i: Int, v: Float32) -> Bool:
    var old = bitcast[DType.uint32](arr[i])
    var w = ftz(v)
    arr[i] = w
    return old != bitcast[DType.uint32](w)


@always_inline
def _putf(mut arr: InlineArray[Float32, 4], i: Int, v: Float32) -> Bool:
    var old = bitcast[DType.uint32](arr[i])
    var w = ftz(v)
    arr[i] = w
    return old != bitcast[DType.uint32](w)


@always_inline
def _co(arr: InlineArray[Float32, 4], j: Int, k: Int) -> Float32:
    return arr[j] if j < k else Float32(0.0)


@always_inline
def _row(S: InlineArray[Float32, 12], r: Int, j: Int, k: Int) -> Float32:
    return S[r * k + j] if j < k else Float32(0.0)


@always_inline
def nm_spec(
    ev: ThetaEv, mut x: InlineArray[Float32, 4], lo: InlineArray[Float32, 4], hi: InlineArray[Float32, 4],
    k: Int, lane: Int, init_step: Float32, zero_pert: Float32, max_iter: Int, tol_std: Float32, snap_on: Bool,
) -> Int:
    """`nelder_mead` (sequence/nm.mojo, k <= 3, no stall rule) with the
    candidates of each step evaluated across lanes; x receives the best
    vertex; returns the iteration count. Every lane calls it and every
    lane returns the same words."""
    var S = InlineArray[Float32, 12](fill=Float32(0.0))
    var F = InlineArray[Float32, 4](fill=Float32(0.0))
    var SN = InlineArray[Float32, 12](fill=Float32(0.0))
    var SNF = InlineArray[Float32, 4](fill=Float32(0.0))
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
    var f0 = ev.ev(_row(S, r0, 0, k), _row(S, r0, 1, k), _row(S, r0, 2, k))
    for i in range(4):
        var v = shuffle_idx(f0, UInt32(i))
        if i <= k:
            F[i] = ftz(v)
    var nf = Float32(k)
    var gamma = add(Float32(1.0), ftz(identical_div(Float32(2.0), nf)))
    var rho = sub(Float32(0.75), ftz(identical_div(Float32(1.0), mul(Float32(2.0), nf))))
    var sigma = sub(Float32(1.0), ftz(identical_div(Float32(1.0), nf)))
    var xo = InlineArray[Float32, 4](fill=Float32(0.0))
    var xr = InlineArray[Float32, 4](fill=Float32(0.0))
    var xe = InlineArray[Float32, 4](fill=Float32(0.0))
    var xc = InlineArray[Float32, 4](fill=Float32(0.0))
    var xi = InlineArray[Float32, 4](fill=Float32(0.0))
    var order = InlineArray[Int, 4](fill=0)
    var changed = True
    var have_snap = False
    var snap_it = 0
    var power = 1
    var stop_at = -1
    var it = 0
    var best = 0
    var role = lane if lane < 4 else 0
    while it < max_iter:
        if it == stop_at:
            it = max_iter
            break
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
        if snap_on and stop_at < 0:
            if have_snap:
                var same = True
                for i in range((k + 1) * k):
                    if bitcast[DType.uint32](S[i]) != bitcast[DType.uint32](SN[i]):
                        same = False
                for i in range(k + 1):
                    if bitcast[DType.uint32](F[i]) != bitcast[DType.uint32](SNF[i]):
                        same = False
                if same:
                    var period = it - snap_it
                    var rest = (max_iter - it) % period
                    stop_at = it + (rest if rest > 0 else period)
            if stop_at < 0 and (not have_snap or it - snap_it == power):
                for i in range((k + 1) * k):
                    SN[i] = S[i]
                for i in range(k + 1):
                    SNF[i] = F[i]
                if have_snap:
                    power *= 2
                snap_it = it
                have_snap = True
        # the centroid without the worst vertex, and the four candidates
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
        var c0: Float32
        var c1: Float32
        var c2: Float32
        if role == 1:
            c0 = _co(xe, 0, k)
            c1 = _co(xe, 1, k)
            c2 = _co(xe, 2, k)
        elif role == 2:
            c0 = _co(xc, 0, k)
            c1 = _co(xc, 1, k)
            c2 = _co(xc, 2, k)
        elif role == 3:
            c0 = _co(xi, 0, k)
            c1 = _co(xi, 1, k)
            c2 = _co(xi, 2, k)
        else:
            c0 = _co(xr, 0, k)
            c1 = _co(xr, 1, k)
            c2 = _co(xr, 2, k)
        var fl = ev.ev(c0, c1, c2)
        var fr = shuffle_idx(fl, UInt32(0))
        var fe = shuffle_idx(fl, UInt32(1))
        var fc = shuffle_idx(fl, UInt32(2))
        var fi = shuffle_idx(fl, UInt32(3))
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
            var fs = ev.ev(_row(S, rs, 0, k), _row(S, rs, 1, k), _row(S, rs, 2, k))
            for i in range(4):
                var v = shuffle_idx(fs, UInt32(i))
                if i <= k and i != best:
                    changed |= _putf(F, i, v)
        it += 1
    for j in range(k):
        x[j] = S[best * k + j]
    return it + 1


def op_theta_spec(t: Int, lane: Int, a: Args):
    """`op_theta` (sequence/theta.mojo, THETA_REG + THETA_HOIST semantics)
    for series t on its simdgroup; `nm_spec` in place of `nelder_mead`.
    Lane 0 stores the forecast and the info row."""
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
    var xs = nm_scr + 64
    var f = xs + 12
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
    var inv = theta_invariants(yd, n)
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
        var ev = ThetaEv(yd, n, model == DSTM or model == DOTM, ol, oa, ot, l0, a0, t0, inv[0], inv[1], inv[2])
        var x = InlineArray[Float32, 4](fill=Float32(0.0))
        var lo = InlineArray[Float32, 4](fill=Float32(0.0))
        var hi = InlineArray[Float32, 4](fill=Float32(0.0))
        var k = 0
        if ol:
            x[k] = ftz(l0)
            lo[k] = ftz(Float32(-1e10))
            hi[k] = ftz(Float32(1e10))
            k += 1
        if oa:
            x[k] = ftz(a0)
            lo[k] = ftz(Float32(0.1))
            hi[k] = ftz(Float32(0.99))
            k += 1
        if ot:
            x[k] = ftz(t0)
            lo[k] = ftz(Float32(1.0))
            hi[k] = ftz(Float32(1e10))
            k += 1
        var it = 0
        if k > 0:
            it = nm_spec(ev, x, lo, hi, k, lane, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4), THETA_SNAP)
        var p = ev.params(_co(x, 0, k), _co(x, 1, k), _co(x, 2, k))
        var mse = theta_run_reg(yd, n, model, p[0], p[1], p[2], states)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            bl = p[0]
            ba = p[1]
            bt = p[2]
            iters = it
    _ = theta_run_reg(yd, n, best_model, bl, ba, bt, states)
    theta_forecast_reg(n, h, best_model, ba, bt, states, f)
    if lane != 0:
        return
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
