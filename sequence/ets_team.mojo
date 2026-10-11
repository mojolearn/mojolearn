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
FAST on Apple by default, and IDENTICAL behind `-D MOJOLEARN_IDN_ETS_TEAM`
(below: a fixed tree prefix in place of the serial one, replayed on the
host column by `ets_team_replay`).

The multiplicative-error and the seasonal models keep `op_ets`."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from sequence.ets import SEAS_N, _log, ets_init_state
from sequence.fit_team import SEQ_TEAM_TPB, TEAM_REC, SeqTeam, _ldi, _nm_load, _nm_save, _spend, _sti
from sequence.nm import Objective, nelder_mead, nm_finish, nm_start, nm_steps
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

#: lane ets-team-idn (2026-10-11, idea E2). IDENTICAL, default OFF:
#: `-D MOJOLEARN_IDN_ETS_TEAM` runs ETS(A, A|Ad, N) on NVIDIA and AMD (and
#: Apple) with the team above, one block of SEQ_TEAM_TPB threads per series,
#: in place of op_ets's one thread per series. Structural reason: op_ets
#: walks the whole series serially in one thread per Nelder-Mead evaluation
#: and keeps only B threads of the GPU busy (damped-ets taxi-hourly: AMD
#: 1771 ms, NV 744 ms); the team cuts the per-evaluation chain from n steps
#: to 2 n / 64 + log2(64) compositions. Expected AMD 1771 -> ~150 ms, NV 744
#: -> ~80 ms (Apple FAST measured 612 -> 48 ms with the serial prefix).
#: The chunk count is the constant ETS_TEAM_CHUNKS (= SEQ_TEAM_TPB, the
#: launch's block size on every vendor; L = ceil(n / 64) points a chunk), so
#: the bits never depend on the device. The exclusive prefix of the chunk
#: maps is a fixed Hillis-Steele tree (6 levels, offsets 1, 2, .., 32, each
#: level `_ets_compose`), not the serial walk of FAST, and the chunk SSEs
#: fold ascending from 0. BITS CHANGE against op_ets (composed maps, beta e
#: trend form, chunked SSE fold, the forecast's running phi power) on NVIDIA,
#: AMD and the host column together: the host binding replays the same chunk
#: maps, tree and fold in `ets_team_replay` (sequence/exec.mojo). Every other
#: ETS model keeps op_ets. MOJOLEARN_IDN_ALL_OFF turns it off.
comptime IDN_ETS_TEAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_ETS_TEAM"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: IDN_ETS_TEAM's chunk count: the team's threads (the host replay runs
#: them one after another)
comptime ETS_TEAM_CHUNKS = 64
#: IDN_ETS_TEAM's tree levels: log2(ETS_TEAM_CHUNKS)
comptime ETS_TEAM_LEVELS = 6

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


@always_inline
def _ets_chunk_map(y: FP, t0: Int, t1: Int, alpha: Float32, beta: Float32, phi: Float32
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    """Pass 1: chunk [t0, t1)'s affine map (m00, m01, m10, m11, v0, v1) from
    an unknown start, (M, v) <- (A M, A v + c y) per point. The device team
    and the host replay run these statements."""
    # the step map: l' = a00 l + a01 b + alpha y; b' = a10 l + a11 b + beta y
    var a00 = sub(Float32(1.0), alpha)
    var a01 = mul(a00, phi)
    var a10 = sub(Float32(0.0), beta)
    var a11 = mul(phi, sub(Float32(1.0), beta))
    var m00 = Float32(1.0)
    var m01 = Float32(0.0)
    var m10 = Float32(0.0)
    var m11 = Float32(1.0)
    var v0 = Float32(0.0)
    var v1 = Float32(0.0)
    for t in range(t0, t1):
        var yt = ld(y, t)
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
    return (m00, m01, m10, m11, v0, v1)


@always_inline
def _ets_compose(hi: FP, lo: FP) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    """IDN_ETS_TEAM's tree step: the map `lo` (6 words) then the map `hi`,
    (Mh Ml, Mh vl + vh). One spelling for the device tree and the host
    replay."""
    var h00 = ld(hi, 0)
    var h01 = ld(hi, 1)
    var h10 = ld(hi, 2)
    var h11 = ld(hi, 3)
    var l00 = ld(lo, 0)
    var l01 = ld(lo, 1)
    var l10 = ld(lo, 2)
    var l11 = ld(lo, 3)
    return (
        fma3(h00, l00, mul(h01, l10)),
        fma3(h00, l01, mul(h01, l11)),
        fma3(h10, l00, mul(h11, l10)),
        fma3(h10, l01, mul(h11, l11)),
        fma3(h00, ld(lo, 4), fma3(h01, ld(lo, 5), ld(hi, 4))),
        fma3(h10, ld(lo, 4), fma3(h11, ld(lo, 5), ld(hi, 5))),
    )


@always_inline
def _ets_put_map(p: FP, m: Tuple[Float32, Float32, Float32, Float32, Float32, Float32]):
    st(p, 0, m[0])
    st(p, 1, m[1])
    st(p, 2, m[2])
    st(p, 3, m[3])
    st(p, 4, m[4])
    st(p, 5, m[5])


@always_inline
def _ets_apply(mj: FP, l: Float32, b: Float32) -> Tuple[Float32, Float32]:
    """The state (l, b) through the map mj (the FAST prefix's step)."""
    var nl = fma3(ld(mj, 0), l, fma3(ld(mj, 1), b, ld(mj, 4)))
    var nb = fma3(ld(mj, 2), l, fma3(ld(mj, 3), b, ld(mj, 5)))
    return (nl, nb)


@always_inline
def _ets_chunk_sse(y: FP, t0: Int, t1: Int, alpha: Float32, beta: Float32, phi: Float32,
                   l_in: Float32, b_in: Float32) -> Tuple[Float32, Float32, Float32]:
    """Pass 2: chunk [t0, t1) from its start state; (sse, l, b) at its end."""
    var l = l_in
    var b = b_in
    var sse = Float32(0.0)
    for t in range(t0, t1):
        var q = fma3(phi, b, l)
        var e = sub(ld(y, t), q)
        sse = fma3(e, e, sse)
        l = fma3(alpha, e, q)
        b = fma3(beta, e, mul(phi, b))
    return (sse, l, b)


@always_inline
def _ets_lik(n: Int, tot: Float32) -> Float32:
    """n log(sum e^2), clamped below as op_ets's objective."""
    var lik: Float32
    if tot > Float32(0.0):
        lik = mul(Float32(n), _log(tot))
    else:
        lik = mul(Float32(n), _log(add(tot, Float32(1e-8))))
    return lik if lik > Float32(-1e10) else Float32(-1e10)


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
        var mp = self.maps + 6 * tid
        _ets_put_map(mp, _ets_chunk_map(self.y, t0, t1, alpha, beta, phi))
        self.team.sync()
        # the exclusive prefix: this chunk's start state through the maps before it
        var l = u[3]
        var b = u[4]
        comptime if IDN_ETS_TEAM:
            # a fixed Hillis-Steele tree (module comment at IDN_ETS_TEAM): at
            # offset d, thread k >= d composes its map after map k - d; the
            # reads land before the barrier, the writes after it. Map k then
            # holds chunks 0 .. k, and chunk k starts from map k - 1.
            comptime for lev in range(ETS_TEAM_LEVELS):
                comptime d = 1 << lev
                var cm = (ld(mp, 0), ld(mp, 1), ld(mp, 2), ld(mp, 3), ld(mp, 4), ld(mp, 5))
                if tid >= d:
                    cm = _ets_compose(mp, mp - 6 * d)
                self.team.sync()
                if tid >= d:
                    _ets_put_map(mp, cm)
                self.team.sync()
            if tid > 0:
                var s0 = _ets_apply(mp - 6, l, b)
                l = s0[0]
                b = s0[1]
        else:
            for j in range(tid):
                var s0 = _ets_apply(self.maps + 6 * j, l, b)
                l = s0[0]
                b = s0[1]
        # pass 2: the chunk from its start state, its squared errors
        var r = _ets_chunk_sse(self.y, t0, t1, alpha, beta, phi, l, b)
        st(self.parts, tid, r[0])
        if t0 < self.n and t1 == self.n:
            # the chunk holding the last point: the final state
            st(self.fin, 0, r[1])
            st(self.fin, 1, r[2])
        self.team.sync()
        var tot = Float32(0.0)
        for j in range(self.team.nt):
            tot = add(tot, ld(self.parts, j))
        return _ets_lik(self.n, tot)


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


# ---------------------------------------------------------------------------
# IDN_ETS_TEAM's host column (lane ets-team-idn): the team's likelihood with
# its ETS_TEAM_CHUNKS threads run one after another over the same helpers.

#: words of the replay's map buffer: the chunk maps, then the final state
comptime ETS_REPLAY_W = 6 * ETS_TEAM_CHUNKS + 2


struct EtsReplayObj(Objective):
    """EtsTeamObj.eval under IDN_ETS_TEAM on one host thread: pass 1 of every
    chunk, the same tree levels (each level's compositions in descending
    chunk order, so each reads the level's old map k - d, as the team's
    barrier-separated reads do), pass 2 of every chunk from map k - 1, the
    per-chunk sums folded ascending from 0."""
    var y: FP
    var n: Int
    var L: Int
    var maps: FP
    var oa: Bool
    var ob: Bool
    var op: Bool
    var alpha: Float32
    var beta: Float32
    var phi: Float32

    @always_inline
    def __init__(out self, y: FP, n: Int, L: Int, maps: FP, oa: Bool, ob: Bool, op: Bool,
                 alpha: Float32, beta: Float32, phi: Float32):
        self.y = y
        self.n = n
        self.L = L
        self.maps = maps
        self.oa = oa
        self.ob = ob
        self.op = op
        self.alpha = alpha
        self.beta = beta
        self.phi = phi

    @always_inline
    def unpack(self, x: FP) -> Tuple[Float32, Float32, Float32, Float32, Float32]:
        """EtsTeamObj.unpack."""
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
        comptime nt = ETS_TEAM_CHUNKS
        for tid in range(nt):
            var t0 = tid * self.L
            var t1 = t0 + self.L
            if t1 > self.n:
                t1 = self.n
            _ets_put_map(self.maps + 6 * tid, _ets_chunk_map(self.y, t0, t1, alpha, beta, phi))
        comptime for lev in range(ETS_TEAM_LEVELS):
            comptime d = 1 << lev
            for kk in range(nt - d):
                var k = nt - 1 - kk
                var mp = self.maps + 6 * k
                _ets_put_map(mp, _ets_compose(mp, mp - 6 * d))
        var tot = Float32(0.0)
        for tid in range(nt):
            var t0 = tid * self.L
            var t1 = t0 + self.L
            if t1 > self.n:
                t1 = self.n
            var l = u[3]
            var b = u[4]
            if tid > 0:
                var s0 = _ets_apply(self.maps + 6 * (tid - 1), l, b)
                l = s0[0]
                b = s0[1]
            var r = _ets_chunk_sse(self.y, t0, t1, alpha, beta, phi, l, b)
            tot = add(tot, r[0])
            if t0 < self.n and t1 == self.n:
                st(self.maps, 6 * nt, r[1])
                st(self.maps, 6 * nt + 1, r[2])
        return _ets_lik(self.n, tot)


def ets_team_replay_applies(a: Args) -> Bool:
    """Whether op_ets's arguments name the model the team fits (the host
    twin of ets_team_py.ets_team_applies): additive errors (i2 0), an
    additive trend (i3 1), no season (i6 0), n >= 4 and h >= 1."""
    return a.i2 == 0 and a.i3 == 1 and a.i6 == 0 and a.i0 >= 4 and a.i1 >= 1


def ets_team_replay(t: Int, a: Args):
    """IDN_ETS_TEAM's host column for series t, op_ets's arguments (p0 y,
    p1 forecast, p2 info, p3 scratch row of i8 floats; i0 n, i1 h, i4
    damped, i5 fixed mask; f0 alpha, f1 beta, f2 phi): ets_team's setup,
    statsforecast's Nelder-Mead over EtsReplayObj (nelder_mead runs
    nm_start, nm_steps and nm_finish, the team's phases; the IDENTICAL build
    compiles the stall stop out of both), the last evaluation's final
    state, the team's forecast and info row."""
    comptime assert ETS_TEAM_CHUNKS == SEQ_TEAM_TPB, "ETS_TEAM_CHUNKS must be the team's threads"
    comptime assert (1 << ETS_TEAM_LEVELS) == ETS_TEAM_CHUNKS, "ETS_TEAM_LEVELS = log2(ETS_TEAM_CHUNKS)"
    var n = a.i0
    var h = a.i1
    var damped = a.i4 != 0
    var y = a.p0 + t * n
    var sc = a.p3 + t * a.i8
    # op_ets's scratch row: x / lo / hi (8 words each), then the Nelder-Mead
    # scratch ((k + 1) k + (k + 1) + 4 k <= 56 at k 5)
    var x = sc
    var lo = x + 8
    var hi = lo + 8
    var nm_scr = hi + 8
    var buf = InlineArray[Float32, ETS_REPLAY_W](fill=Float32(0.0))
    var maps = FP(unsafe_from_address=Int(buf.unsafe_ptr()))
    var L = (n + ETS_TEAM_CHUNKS - 1) // ETS_TEAM_CHUNKS
    var fa = (a.i5 & 1) != 0
    var fb = (a.i5 & 2) != 0
    var fp = (a.i5 & 4) != 0
    # ets_team's initparam
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
    var obj = EtsReplayObj(y, n, L, maps, oa, ob, op, alpha, beta, phi)
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
    var it = nelder_mead(obj, x, lo, hi, k, nm_scr, Float32(0.05), Float32(1e-4), 1000, Float32(1e-4))
    # the team's phase 2: the last evaluation at the best point
    var lik = obj.eval(x)
    var u = obj.unpack(x)
    var l = ld(maps, 6 * ETS_TEAM_CHUNKS)
    var bb = ld(maps, 6 * ETS_TEAM_CHUNKS + 1)
    var pk = u[2]
    var phistar = u[2]
    for i in range(h):
        st(a.p1, t * h + i, fma3(phistar, bb, l))
        pk = mul(pk, u[2])
        phistar = add(phistar, pk)
    var info = a.p2 + t * 10
    st(info, 0, u[0])
    st(info, 1, u[1])
    st(info, 2, u[2])
    st(info, 3, u[3])
    st(info, 4, u[4])
    st(info, 5, lik)
    st(info, 6, Float32(it))
    st(info, 7, Float32(k))
    st(info, 8, Float32(0.0))
    st(info, 9, Float32(0.0))
    # the map buffer lives to here (Mojo ends a local at its last named use)
    _ = buf[0]
