# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Prophet team fit with the WHOLE block on every step (Apple FAST,
default; off: `-D MOJOLEARN_PROPHET_COOP_OFF`; lane apple-fast-prophetspeed).

`prophet_fit_team` (sequence/fit_team.mojo) keeps op_prophet_fit's bits:
each of the P + 1 likelihood accumulators is one thread's ascending chain
over all N points (1,392 dependent steps on taxi-hourly), and every thread
runs the whole L-BFGS control serially over its own private row (P-long
dot products and updates, rows R words apart, so no load of a simdgroup
coalesces). On the M3 the 64-series fit took 358-429 ms.

Here, one block of PROPHET_COOP_TPB threads per series:

  * pass A deals the points out (as TeamFG); pass B gives every accumulator
    to the whole block: each thread folds its own strided points, a
    simdgroup butterfly (shuffle_xor) sums the 32 lanes, lane 0 stores the
    simdgroup's partial, and after one barrier the partials of the NW
    simdgroups are folded in ascending order;
  * the L-BFGS control runs in every SIMDGROUP over that simdgroup's own
    row of vectors, lane l owning coordinates l, l + 32, ...; elementwise
    updates touch only a lane's own coordinates, and every dot product is
    a butterfly, whose result is the same bits in every lane (each step
    adds a pair in both orders, and float addition commutes). The NW
    simdgroups compute the same values, so control stays uniform and every
    barrier is reached by the whole block. A lane's per-pair scalars (rho,
    alpha) live in its own slots.

Same model, priors, start, line search and stopping rules as lbfgs_steps;
the folds' orders differ, so the bits differ: FAST only, Apple only. The
fit is sliced over launches as prophet_fit_team (macOS long-launch rule)."""
from std.gpu.primitives.warp import shuffle_xor
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_exp, identical_sqrt
from sequence.fit_team import PT_DONE, TEAM_REC, SeqTeam, _ldi, _spend, _sti
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from sequence.prophet import MEM, PROPHET_COOP_IDN

#: the switch: FAST on Apple by default since the M3 A/B (lane
#: apple-fast-prophetspeed 805207038, n=1: prophet synthetic 431 -> 45 ms,
#: taxi-hourly 360 -> 43 ms; forecast_rmse 1.015 -> 1.015, 32.03 -> 32.04).
#: -D MOJOLEARN_PROPHET_COOP_OFF turns it off; the old -D MOJOLEARN_PROPHET_COOP
#: is harmless. IDENTICAL and every other vendor keep prophet_fit_team.
#: lane/fam2-timeseries (2026-10-04): IDENTICAL ON EVERY VENDOR too
#: (PROPHET_COOP_IDN, sequence/prophet.mojo). The fold orders here are a
#: function of the block size and the 32-lane group alone (a thread's
#: strided chain, the shuffle_xor butterfly inside each group of 32, the
#: groups' partials ascending), so NVIDIA, AMD (a 64-lane wavefront holds two
#: groups; offsets of at most 16 stay inside one) and Apple fold the same
#: way, and the host column replays that order (`_coop_fg`, `_dot_coop`).
#: -D MOJOLEARN_IDN_PROPHET_COOP_OFF=1 restores prophet_fit_team's chain.
comptime PROPHET_COOP = (
    (
        GLOBAL_NUMERIC_MODE == NUMERIC_FAST
        and has_apple_gpu_accelerator()
        and not is_defined["MOJOLEARN_PROPHET_COOP_OFF"]()
    )
    or PROPHET_COOP_IDN
)
#: threads of a series' block on the cooperative path (prophet.mojo COOP_NT)
comptime PROPHET_COOP_TPB = 256
#: simdgroup width (prophet.mojo COOP_PW)
comptime PW = 32


def prophet_coop_vrow(P: Int) -> Int:
    """Words of one simdgroup's vector row: th, g, d, thn, gn, q (P each),
    the MEM s / y pairs, each lane's rho and alpha slots."""
    return (6 + 2 * MEM) * P + 2 * MEM * PW


def prophet_coop_priv(P: Int, nt: Int) -> Int:
    """Words of a series' private area: nt thread records, then the NW
    simdgroups' vector rows."""
    return nt * TEAM_REC + (nt // PW) * prophet_coop_vrow(P)


def prophet_coop_shared(N: Int, P: Int, nt: Int) -> Int:
    """Words of a series' shared row: scaled y, residuals, the two weights
    (N each), then NW partial rows of P + 1."""
    return 4 * N + (nt // PW) * (P + 1)


@always_inline
def _wsum(v: Float32) -> Float32:
    """The 32 lanes' sum, the same bits in every lane (butterfly)."""
    var s = v
    comptime for k in range(5):
        s = add(s, shuffle_xor(s, UInt32(16 >> k)))
    return s


@always_inline
def _wmax(v: Float32) -> Float32:
    var s = v
    comptime for k in range(5):
        var o = shuffle_xor(s, UInt32(16 >> k))
        s = o if o > s else s
    return s


@always_inline
def _cdot(a: FP, b: FP, P: Int, lane: Int) -> Float32:
    var s = Float32(0.0)
    for i in range(lane, P, PW):
        s = fma3(ld(a, i), ld(b, i), s)
    return _wsum(s)


struct CoopFG(ImplicitlyCopyable, Movable):
    """-log posterior and gradient over the block (module docstring). th and
    g are the calling simdgroup's vectors; on return g holds the lane's own
    coordinates."""
    var team: SeqTeam
    var lane: Int
    var wid: Int
    var nw: Int
    var t: FP
    var X: FP
    var cp: FP
    var sig: FP
    var N: Int
    var K: Int
    var S: Int
    var tau: Float32
    var mult: Bool
    var y: FP
    var rr: FP
    var wt: FP
    var wb: FP
    var part: FP

    @always_inline
    def __init__(out self, team: SeqTeam, t: FP, X: FP, cp: FP, sig: FP, N: Int, K: Int, S: Int,
                 tau: Float32, mult: Bool, y: FP, rr: FP, wt: FP, wb: FP, part: FP):
        self.team = team
        self.lane = team.tid % PW
        self.wid = team.tid // PW
        self.nw = team.nt // PW
        self.t = t
        self.X = X
        self.cp = cp
        self.sig = sig
        self.N = N
        self.K = K
        self.S = S
        self.tau = tau
        self.mult = mult
        self.y = y
        self.rr = rr
        self.wt = wt
        self.wb = wb
        self.part = part

    @always_inline
    def fg(mut self, th: FP, g: FP) -> Float32:
        var S = self.S
        var K = self.K
        var N = self.N
        var P = 3 + S + K
        var tt = self.t
        var cp = self.cp
        var X = self.X
        var tid = self.team.tid
        var nt = self.team.nt
        # th's coordinates were stored by the simdgroup's lanes
        self.team.sync()
        var k = ld(th, 0)
        var m = ld(th, 1)
        var u = ld(th, 2 + S)
        var sigma = ftz(identical_exp(u))
        var s2 = mul(sigma, sigma)
        # pass A: point i's residual and weights
        for i in range(tid, N, nt):
            var ti = ld(tt, i)
            var tr = fma3(k, ti, m)
            for j in range(S):
                var c = ld(cp, j)
                if ti >= c:
                    tr = fma3(ld(th, 2 + j), sub(ti, c), tr)
            var se = Float32(0.0)
            for q in range(K):
                se = fma3(ld(X, i * K + q), ld(th, 3 + S + q), se)
            var yhat: Float32
            if self.mult:
                yhat = fma3(tr, se, tr)
            else:
                yhat = add(tr, se)
            var r = sub(ld(self.y, i), yhat)
            var w = ftz(identical_div(-r, s2))
            var wt = mul(w, add(Float32(1.0), se)) if self.mult else w
            var wb = mul(w, tr) if self.mult else w
            st(self.rr, i, r)
            st(self.wt, i, wt)
            st(self.wb, i, wb)
        self.team.sync()
        # pass B: every accumulator over the whole block
        var prow = self.part + self.wid * (P + 1)
        for j in range(P + 1):
            var acc = Float32(0.0)
            if j == 0:
                for i in range(tid, N, nt):
                    acc = fma3(ld(self.wt, i), ld(tt, i), acc)
            elif j == 1:
                for i in range(tid, N, nt):
                    acc = add(acc, ld(self.wt, i))
            elif j < 2 + S:
                var c = ld(cp, j - 2)
                for i in range(tid, N, nt):
                    var ti = ld(tt, i)
                    if ti >= c:
                        acc = fma3(ld(self.wt, i), sub(ti, c), acc)
            elif j == 2 + S:
                pass
            elif j < P:
                var q = j - 3 - S
                for i in range(tid, N, nt):
                    acc = fma3(ld(self.wb, i), ld(X, i * K + q), acc)
            else:
                for i in range(tid, N, nt):
                    var r = ld(self.rr, i)
                    acc = fma3(r, r, acc)
            acc = _wsum(acc)
            if self.lane == 0:
                st(prow, j, acc)
        self.team.sync()
        var sse = Float32(0.0)
        for w in range(self.nw):
            sse = add(sse, ld(self.part, w * (P + 1) + P))
        # the lane's coordinates: likelihood gradient plus priors
        var fc = Float32(0.0)
        for i in range(self.lane, P, PW):
            var gi = Float32(0.0)
            for w in range(self.nw):
                gi = add(gi, ld(self.part, w * (P + 1) + i))
            var x = ld(th, i)
            if i == 0 or i == 1:
                fc = add(fc, ftz(identical_div(mul(x, x), Float32(50.0))))
                gi = add(gi, ftz(identical_div(x, Float32(25.0))))
            elif i < 2 + S:
                fc = add(fc, ftz(identical_div(abs(x), self.tau)))
                var sg = Float32(0.0)
                if x > Float32(0.0):
                    sg = Float32(1.0)
                elif x < Float32(0.0):
                    sg = Float32(-1.0)
                gi = add(gi, ftz(identical_div(sg, self.tau)))
            elif i == 2 + S:
                gi = sub(add(ftz(identical_div(s2, Float32(0.25))), Float32(N)),
                         ftz(identical_div(sse, s2)))
            else:
                var sq = mul(ld(self.sig, i - 3 - S), ld(self.sig, i - 3 - S))
                fc = add(fc, ftz(identical_div(mul(x, x), mul(Float32(2.0), sq))))
                gi = add(gi, ftz(identical_div(x, sq)))
            st(g, i, gi)
        var f = _wsum(fc)
        # sigma: prior N(0, 0.5), likelihood N log sigma + sse / (2 sigma^2)
        f = add(f, ftz(identical_div(s2, Float32(0.5))))
        f = fma3(Float32(N), u, f)
        f = add(f, ftz(identical_div(sse, mul(Float32(2.0), s2))))
        return f


struct CoopState(ImplicitlyCopyable, Movable):
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


@always_inline
def lbfgs_coop_steps(mut fg: CoopFG, mut s: CoopState, P: Int, th: FP, w: FP,
                     max_iter: Int, budget: Int) -> Int:
    """lbfgs_steps (sequence/prophet.mojo) over a simdgroup's vector row:
    the same iteration, lane-owned coordinates, butterfly dot products."""
    var lane = fg.lane
    var g = w
    var dvec = g + P
    var thn = dvec + P
    var gn = thn + P
    var q = gn + P
    var sm = q + P
    var ym = sm + MEM * P
    var rho = ym + MEM * P
    var al = rho + MEM * PW
    var f = s.f
    var npairs = s.npairs
    var head = s.head
    var small = s.small
    var it = s.it
    var steps = 0
    var finished = True
    while it < max_iter:
        if budget >= 0 and steps >= budget:
            finished = False
            break
        steps += 1
        for i in range(lane, P, PW):
            st(q, i, ld(g, i))
        var idx = head
        for _ in range(npairs):
            idx = (idx - 1 + MEM) % MEM
            var aa = mul(ld(rho, idx * PW + lane), _cdot(sm + idx * P, q, P, lane))
            st(al, idx * PW + lane, aa)
            for i in range(lane, P, PW):
                st(q, i, sub(ld(q, i), mul(aa, ld(ym + idx * P, i))))
        var gamma = Float32(1.0)
        if npairs > 0:
            var last = (head - 1 + MEM) % MEM
            var yy = _cdot(ym + last * P, ym + last * P, P, lane)
            if yy > Float32(0.0):
                gamma = ftz(identical_div(_cdot(sm + last * P, ym + last * P, P, lane), yy))
        else:
            var gg = ftz(identical_sqrt(_cdot(g, g, P, lane)))
            if gg > Float32(1.0):
                gamma = ftz(identical_div(Float32(1.0), gg))
        for i in range(lane, P, PW):
            st(q, i, mul(gamma, ld(q, i)))
        idx = (head - npairs + MEM) % MEM
        for _ in range(npairs):
            var bb = mul(ld(rho, idx * PW + lane), _cdot(ym + idx * P, q, P, lane))
            var coef = sub(ld(al, idx * PW + lane), bb)
            for i in range(lane, P, PW):
                st(q, i, fma3(coef, ld(sm + idx * P, i), ld(q, i)))
            idx = (idx + 1) % MEM
        for i in range(lane, P, PW):
            st(dvec, i, -ld(q, i))
        var gd = _cdot(g, dvec, P, lane)
        if not (gd < Float32(0.0)):
            for i in range(lane, P, PW):
                st(dvec, i, -ld(g, i))
            gd = -_cdot(g, g, P, lane)
            npairs = 0
        var step = Float32(1.0)
        var fnew = Float32(0.0)
        var ok = False
        for _ in range(40):
            for i in range(lane, P, PW):
                st(thn, i, fma3(step, ld(dvec, i), ld(th, i)))
            fnew = fg.fg(thn, gn)
            if fnew <= fma3(mul(Float32(1e-4), step), gd, f):
                ok = True
                break
            step = mul(step, Float32(0.5))
        it += 1
        if not ok:
            break
        for i in range(lane, P, PW):
            st(q, i, sub(ld(thn, i), ld(th, i)))
            st(dvec, i, sub(ld(gn, i), ld(g, i)))
        var sy = _cdot(q, dvec, P, lane)
        if sy > Float32(1e-12):
            var slot = head
            for i in range(lane, P, PW):
                st(sm + slot * P, i, ld(q, i))
                st(ym + slot * P, i, ld(dvec, i))
            st(rho, slot * PW + lane, ftz(identical_div(Float32(1.0), sy)))
            head = (head + 1) % MEM
            if npairs < MEM:
                npairs += 1
        var fscale = abs(f)
        if abs(fnew) > fscale:
            fscale = abs(fnew)
        if fscale < Float32(1.0):
            fscale = Float32(1.0)
        var df = sub(f, fnew)
        var gl = Float32(0.0)
        for i in range(lane, P, PW):
            st(th, i, ld(thn, i))
            var gv = ld(gn, i)
            st(g, i, gv)
            if abs(gv) > gl:
                gl = abs(gv)
        f = fnew
        var gmax = _wmax(gl)
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
def prophet_fit_coop(slot: Int, team: SeqTeam, a: Args):
    """prophet_fit_team's contract (arguments, outputs, slicing, done flags)
    on the cooperative path: p7 shared rows [G, i11]
    (prophet_coop_shared), p8 private areas [G, i10] (prophet_coop_priv,
    zero before a group's first launch). Thread record: 0 phase, 1 y scale,
    2 f, 3 npairs, 4 head, 5 small, 6 iterations."""
    var b = a.i8 + slot
    var N = a.i0
    var K = a.i1
    var S = a.i2
    var P = 3 + S + K
    var nt = team.nt
    var lane = team.tid % PW
    var wid = team.tid // PW
    var nw = nt // PW
    var y = a.p0 + b * N
    var sh = a.p7 + slot * a.i11
    var ys = sh
    var rr = ys + N
    var wt = rr + N
    var wb = wt + N
    var part = wb + N
    var area = a.p8 + slot * a.i10
    var rec = area + team.tid * TEAM_REC
    var th = area + nt * TEAM_REC + wid * prophet_coop_vrow(P)
    var w = th + P
    var fg = CoopFG(team, a.p1, a.p2, a.p3, a.p4, N, K, S, a.f0, a.i3 != 0, ys, rr, wt, wb, part)
    var phase = _ldi(rec, 0)
    var left = a.i9
    while phase < PT_DONE and left != 0:
        if phase == 0:
            var mx = Float32(0.0)
            for i in range(team.tid, N, nt):
                var v = abs(ld(y, i))
                if v > mx:
                    mx = v
            mx = _wmax(mx)
            if lane == 0:
                st(part, wid, mx)
            team.sync()
            var scale = Float32(0.0)
            for ww in range(nw):
                var v = ld(part, ww)
                if v > scale:
                    scale = v
            if scale == Float32(0.0):
                scale = Float32(1.0)
            for i in range(team.tid, N, nt):
                st(ys, i, ftz(identical_div(ld(y, i), scale)))
            team.sync()
            var t0 = ld(a.p1, 0)
            var t1 = ld(a.p1, N - 1)
            var k = ftz(identical_div(sub(ld(ys, N - 1), ld(ys, 0)), sub(t1, t0))) if t1 != t0 else Float32(0.0)
            for j in range(lane, P, PW):
                var v = Float32(0.0)
                if j == 0:
                    v = k
                elif j == 1:
                    v = sub(ld(ys, 0), mul(k, t0))
                st(th, j, v)
            var s = CoopState(fg.fg(th, w))
            rec.unsafe_store(1, scale)
            rec.unsafe_store(2, s.f)
            _sti(rec, 3, s.npairs)
            _sti(rec, 4, s.head)
            _sti(rec, 5, s.small)
            _sti(rec, 6, s.it)
            phase = 1
            left = _spend(left, 1)
        elif phase == 1:
            var s = CoopState(rec.unsafe_load(2))
            s.npairs = _ldi(rec, 3)
            s.head = _ldi(rec, 4)
            s.small = _ldi(rec, 5)
            s.it = _ldi(rec, 6)
            var used = lbfgs_coop_steps(fg, s, P, th, w, a.i5, left)
            rec.unsafe_store(2, s.f)
            _sti(rec, 3, s.npairs)
            _sti(rec, 4, s.head)
            _sti(rec, 5, s.small)
            _sti(rec, 6, s.it)
            left = _spend(left, used)
            if s.done:
                phase = 2
            else:
                left = 0
        else:
            # th's coordinates were stored by the lead simdgroup's lanes
            team.sync()
            if team.lead():
                for j in range(P):
                    st(a.p5, b * P + j, ld(th, j))
                var info = a.p6 + b * 4
                st(info, 0, rec.unsafe_load(1))
                st(info, 1, rec.unsafe_load(2))
                st(info, 2, Float32(_ldi(rec, 6)))
                st(info, 3, Float32(0.0))
            phase = PT_DONE
    _sti(rec, 0, phase)
    if team.lead():
        a.p9.unsafe_store(slot, Float32(1.0) if phase == PT_DONE else Float32(0.0))
