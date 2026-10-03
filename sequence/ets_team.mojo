# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ETS(A, A|Ad, N) with ONE SERIES PER BLOCK and the likelihood over the
block (Apple FAST, `-D MOJOLEARN_ETS_TEAM`; lane apple-fast-ets).

`op_ets` (sequence/ets.mojo) fits each series inside one GPU thread: the
Nelder-Mead walks the series serially once or twice per iteration, up to
1000 iterations, and a batch of 64 series keeps 64 threads of the GPU
busy (damped-ets taxi-hourly on the M3 Ultra: 612 ms FAST against
statsforecast's 185 ms on the CPU).

Here a block of SEQ_TEAM_TPB threads runs one series, the GARCH / Prophet
shape of sequence/fit_team.mojo: the optimizer's control runs in EVERY
thread over that thread's private copy of its state, from values every
thread computes with the same bits, so control flow is uniform and every
barrier is reached by the whole block. The objective is the block's. With
additive errors, no season and an additive (damped) trend the state
(l, b) is an AFFINE function of the previous one:

    q  = l + phi b,  e = y - q,
    l' = q + alpha e = (1 - alpha) l + (1 - alpha) phi b + alpha y,
    b' = phi b + beta e = -beta l + phi (1 - beta) b + beta y,

so the series splits into T chunks of L points. Thread k composes its
chunk's map (M_k, v_k) with the start state unknown (pass 1), takes its
start state from the maps of the chunks before it (the exclusive prefix,
k compositions of a 2 x 2 affine map), then walks its chunk from that
state and sums its squared errors (pass 2). The likelihood n log(sum e^2)
is every thread's ascending fold over the T partial sums, so every thread
sees the same value. Same model, same bounds, same Nelder-Mead; the
error sum's fold order differs from op_ets's and the trend update is
beta e in place of (beta / alpha) ((l' - l) - phi b), so the bits differ:
FAST only, and only on Apple (NVIDIA and AMD keep op_ets).

The multiplicative-error and the seasonal models keep `op_ets`."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from sequence.ets import SEAS_N, _log, ets_init_state
from sequence.fit_team import SEQ_TEAM_TPB, TEAM_REC, SeqTeam, _ldi, _nm_load, _nm_save, _spend, _sti
from sequence.nm import Objective, nm_finish, nm_start, nm_steps
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub

#: the switch: FAST on Apple, default ON since the M3 A/B (damped-ets
#: taxi-hourly, lane/apple-fast-ets e3369d119, n=1: 611.9 -> 48.0 ms,
#: forecast_rmse 96.69 -> 96.68; docs/apple-fast/ab/ets.md).
#: -D MOJOLEARN_ETS_TEAM_OFF turns it off; the old -D MOJOLEARN_ETS_TEAM is
#: harmless. IDENTICAL and every other vendor compile op_ets unchanged.
comptime ETS_TEAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ETS_TEAM_OFF"]()
)

#: ETS phases: 0 setup (the initial state, the bounds, the simplex);
#: 1 the Nelder-Mead iterations; 2 the outputs; 3 done
comptime ET_DONE = 3
#: words of x / lo / hi in a private row (k <= 5 coordinates)
comptime ETS_TEAM_X = 16
#: words of the Nelder-Mead scratch ((k + 1) k + (k + 1) + 4 k = 56 at k 5)
comptime ETS_TEAM_NM = 64


def ets_team_priv() -> Int:
    """Words of one thread's private row: the record, x / lo / hi, the
    Nelder-Mead scratch."""
    return TEAM_REC + 3 * ETS_TEAM_X + ETS_TEAM_NM


def ets_team_shared() -> Int:
    """Words of a series' shared row: the T chunk maps (M, v: 6 words each),
    the T partial sums, the final state (l, b)."""
    return 7 * SEQ_TEAM_TPB + 2


struct EtsTeamObj(Objective):
    """ets_lik for ETS(A, A|Ad, N) over a block (the module docstring)."""
    var team: SeqTeam
    var y: FP
    var n: Int
    var L: Int
    var maps: FP
    var parts: FP
    var fin: FP
    var oa: Bool
    var ob: Bool
    var op: Bool
    var alpha: Float32
    var beta: Float32
    var phi: Float32

    @always_inline
    def __init__(out self, team: SeqTeam, y: FP, n: Int, L: Int, maps: FP, parts: FP, fin: FP,
                 oa: Bool, ob: Bool, op: Bool, alpha: Float32, beta: Float32, phi: Float32):
        self.team = team
        self.y = y
        self.n = n
        self.L = L
        self.maps = maps
        self.parts = parts
        self.fin = fin
        self.oa = oa
        self.ob = ob
        self.op = op
        self.alpha = alpha
        self.beta = beta
        self.phi = phi

    @always_inline
    def unpack(self, x: FP) -> Tuple[Float32, Float32, Float32, Float32, Float32]:
        """(alpha, beta, phi, l0, b0): the free coordinates in order, the
        fixed ones from the caller (op_ets's `EtsObj.unpack`)."""
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
        return (a, b, p, ld(x, j), ld(x, j + 1))

    @always_inline
    def eval(mut self, x: FP) -> Float32:
        var u = self.unpack(x)
        var alpha = u[0]
        var beta = u[1]
        var phi = u[2]
        var tid = self.team.tid
        var t0 = tid * self.L
        var t1 = t0 + self.L
        if t1 > self.n:
            t1 = self.n
        # the step map: l' = a00 l + a01 b + alpha y; b' = a10 l + a11 b + beta y
        var a00 = sub(Float32(1.0), alpha)
        var a01 = mul(a00, phi)
        var a10 = sub(Float32(0.0), beta)
        var a11 = mul(phi, sub(Float32(1.0), beta))
        # pass 1: this chunk's map from an unknown start, (M, v) <- (A M, A v + c y)
        var m00 = Float32(1.0)
        var m01 = Float32(0.0)
        var m10 = Float32(0.0)
        var m11 = Float32(1.0)
        var v0 = Float32(0.0)
        var v1 = Float32(0.0)
        for t in range(t0, t1):
            var yt = ld(self.y, t)
            var n00 = fma3(a00, m00, mul(a01, m10))
            var n01 = fma3(a00, m01, mul(a01, m11))
            var n10 = fma3(a10, m00, mul(a11, m10))
            var n11 = fma3(a10, m01, mul(a11, m11))
            var w0 = fma3(a00, v0, fma3(a01, v1, mul(alpha, yt)))
            var w1 = fma3(a10, v0, fma3(a11, v1, mul(beta, yt)))
            m00 = n00
            m01 = n01
            m10 = n10
            m11 = n11
            v0 = w0
            v1 = w1
        var mp = self.maps + 6 * tid
        st(mp, 0, m00)
        st(mp, 1, m01)
        st(mp, 2, m10)
        st(mp, 3, m11)
        st(mp, 4, v0)
        st(mp, 5, v1)
        self.team.sync()
        # the exclusive prefix: this chunk's start state through the maps before it
        var l = u[3]
        var b = u[4]
        for j in range(tid):
            var mj = self.maps + 6 * j
            var nl = fma3(ld(mj, 0), l, fma3(ld(mj, 1), b, ld(mj, 4)))
            var nb = fma3(ld(mj, 2), l, fma3(ld(mj, 3), b, ld(mj, 5)))
            l = nl
            b = nb
        # pass 2: the chunk from its start state, its squared errors
        var sse = Float32(0.0)
        for t in range(t0, t1):
            var q = fma3(phi, b, l)
            var e = sub(ld(self.y, t), q)
            sse = fma3(e, e, sse)
            l = fma3(alpha, e, q)
            b = fma3(beta, e, mul(phi, b))
        st(self.parts, tid, sse)
        if t0 < self.n and t1 == self.n:
            # the chunk holding the last point: the final state
            st(self.fin, 0, l)
            st(self.fin, 1, b)
        self.team.sync()
        var tot = Float32(0.0)
        for j in range(self.team.nt):
            tot = add(tot, ld(self.parts, j))
        var lik: Float32
        if tot > Float32(0.0):
            lik = mul(Float32(self.n), _log(tot))
        else:
            lik = mul(Float32(self.n), _log(add(tot, Float32(1e-8))))
        return lik if lik > Float32(-1e10) else Float32(-1e10)


def ets_team(slot: Int, team: SeqTeam, a: Args):
    """op_ets for series a.i8 + slot over a block, ETS(A, A|Ad, N) only.
    p0 y [B, n]; p1 forecast [B, h] out; p2 info [B, 10] out (op_ets's
    row: alpha, beta, phi, l0, b0, lik, iterations, parameter count, 0,
    0); p5 the shared rows [G, i11]; p6 the private rows [G, nt, i10] (zero
    before a group's first launch); p7 the done flags [G]. i0 n, i1 h, i2
    the FAST stall iterations (sequence/nm.mojo), i4 damped, i5 fixed mask
    (1 alpha, 2 beta, 4 phi), i8 the group's first series, i9 the launch's
    budget (Nelder-Mead iterations, < 0 none), i10 / i11 the row strides;
    f0 alpha, f1 beta, f2 phi (read where fixed), f4 the stall's relative
    drop. Private record: 0 phase, 1 iterations, 3.. NMState."""
    var b = a.i8 + slot
    var n = a.i0
    var h = a.i1
    var damped = a.i4 != 0
    var y = a.p0 + b * n
    var sh = a.p5 + slot * a.i11
    var maps = sh
    var parts = maps + 6 * team.nt
    var fin = parts + team.nt
    var pv = a.p6 + (slot * team.nt + team.tid) * a.i10
    var rec = pv
    var x = pv + TEAM_REC
    var lo = x + ETS_TEAM_X
    var hi = lo + ETS_TEAM_X
    var nm_scr = hi + ETS_TEAM_X
    var L = (n + team.nt - 1) // team.nt
    var fa = (a.i5 & 1) != 0
    var fb = (a.i5 & 2) != 0
    var fp = (a.i5 & 4) != 0
    # initparam, the usual bounds without a season (op_ets)
    var lo_a = Float32(1e-4)
    var hi_a = Float32(0.9999)
    var alpha = a.f0 if fa else fma3(Float32(0.2), sub(hi_a, lo_a), lo_a)
    var hi_b = hi_a if hi_a < alpha else alpha
    var beta = a.f1 if fb else fma3(Float32(0.1), sub(hi_b, lo_a), lo_a)
    var phi = Float32(1.0)
    if damped:
        phi = a.f2 if fp else fma3(Float32(0.99), sub(Float32(0.98), Float32(0.8)), Float32(0.8))
    var oa = not fa
    var ob = not fb
    var op = damped and not fp
    var k = Int(oa) + Int(ob) + Int(op) + 2
    var obj = EtsTeamObj(team, y, n, L, maps, parts, fin, oa, ob, op, alpha, beta, phi)
    var phase = _ldi(rec, 0)
    var left = a.i9
    while phase < ET_DONE and left != 0:
        if phase == 0:
            # the initial level and trend (every thread, the same bits);
            # without a season initstate touches none of its scratch
            var st0 = ets_init_state(y, n, True, SEAS_N, 1, nm_scr, nm_scr, nm_scr)
            var j = 0
            if oa:
                st(x, j, alpha)
                st(lo, j, lo_a)
                st(hi, j, hi_a)
                j += 1
            if ob:
                st(x, j, beta)
                st(lo, j, lo_a)
                st(hi, j, hi_b)
                j += 1
            if op:
                st(x, j, phi)
                st(lo, j, Float32(0.8))
                st(hi, j, Float32(0.98))
                j += 1
            st(x, j, st0[0])
            st(lo, j, Float32(-3.0e38))
            st(hi, j, Float32(3.0e38))
            j += 1
            st(x, j, st0[1])
            st(lo, j, Float32(-3.0e38))
            st(hi, j, Float32(3.0e38))
            var s = nm_start(obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4))
            _nm_save(rec, s)
            _sti(rec, 1, 0)
            phase = 1
            left = _spend(left, k + 1)
        elif phase == 1:
            var s = _nm_load(rec)
            # i2 / f4: the FAST stall stop (sequence/nm.mojo), op_ets's call
            var used = nm_steps(obj, s, lo, hi, k, nm_scr, 1000, Float32(1e-4),
                                stall_iters=a.i2, stall_rel=a.f4, budget=left)
            _nm_save(rec, s)
            left = _spend(left, used)
            if s.done:
                _sti(rec, 1, nm_finish(x, k, nm_scr, s))
                phase = 2
            else:
                left = 0
        else:
            # the last evaluation at the best point leaves the final state in fin
            var lik = obj.eval(x)
            if team.lead():
                var u = obj.unpack(x)
                var l = ld(fin, 0)
                var bb = ld(fin, 1)
                var pk = u[2]
                var phistar = u[2]
                for i in range(h):
                    st(a.p1, b * h + i, fma3(phistar, bb, l))
                    pk = mul(pk, u[2])
                    phistar = add(phistar, pk)
                var info = a.p2 + b * 10
                st(info, 0, u[0])
                st(info, 1, u[1])
                st(info, 2, u[2])
                st(info, 3, u[3])
                st(info, 4, u[4])
                st(info, 5, lik)
                st(info, 6, Float32(_ldi(rec, 1)))
                st(info, 7, Float32(k))
                st(info, 8, Float32(0.0))
                st(info, 9, Float32(0.0))
            phase = ET_DONE
    _sti(rec, 0, phase)
    if team.lead():
        a.p7.unsafe_store(slot, Float32(1.0) if phase == ET_DONE else Float32(0.0))
