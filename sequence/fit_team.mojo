# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH and the Prophet fit with ONE SERIES PER BLOCK (lane neural-pass143).

The one-thread-per-series device path (`op_garch`, `op_prophet_fit` under
`seq_kernel`) ran each fit's whole optimizer inside one GPU thread, in one
launch: every objective walked the series serially from device memory, and a
batch of 64 series kept 64 threads of the GPU busy. The binding sent every
batch of up to 4,096 series to the host executor instead.

Here a block of SEQ_TEAM_TPB threads runs one series, and the fit is sliced
over launches only where the platform needs it (Apple: macOS aborts a long
command buffer silently). Bits do not move, by the team rules of
x_linear/team.mojo:

  * every stored value is computed by exactly ONE thread, by the same
    sequence of operations the one-thread body uses for it: a fold over the
    points stays one thread's loop over ascending points; only INDEPENDENT
    values (the residuals, the log-likelihood terms, the gradient's
    accumulators, the points' weights) are dealt out across threads;
  * a value one thread stored is read by another only after `sync()`;
  * the optimizer's control (`sequence/nm.mojo`, `sequence/prophet.mojo`
    `lbfgs_steps`) runs in EVERY thread over that thread's PRIVATE copy of
    its state, from values every thread computes with the same bits, so
    control flow is uniform and every barrier is reached by the whole block.

GARCH: the variance recursion is the lead thread's (it is sequential in
time); the residuals and the n log-likelihood terms are dealt out; the sum of
the terms is every thread's own ascending fold (the same words in the same
order as op_garch's fused loop; IDENTICAL: sequence/fold32.mojo's slots and
tree, GARCH_FOLD32 in sequence/garch.mojo). Prophet: the points' residuals and weights
are dealt out; each of the P + 1 gradient and sse accumulators is ONE
thread's ascending chain over the points, exactly op_prophet_fit's chain for
that accumulator.

Slicing: the optimizers stop at an iteration boundary (`nm_steps`,
`lbfgs_steps` with a budget) and keep their state in each thread's private
row, so a resumed fit runs the same iterations as one call."""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator, is_amd_gpu, is_apple_gpu, is_nvidia_gpu
from x_linear.team import team_barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_exp, identical_log, identical_sqrt
from sequence.fold32 import FOLD_L, tree32
from sequence.garch import GARCH_COOP_IDN, GARCH_FOLD32, GARCH_SNAP, LOG_2PI, _backcast, _garch_step_reg, _grid, _pers, _var_bounds, garch_sigma2
from sequence.nm import NMState, Objective, nm_finish, nm_start, nm_steps
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from sequence.prophet import MEM, LBState, ProphetFG, _fg_prior_v, lbfgs_start, lbfgs_steps

#: threads of a series' block (one AMD wave, two NVIDIA warps / Apple
#: simdgroups)
comptime SEQ_TEAM_TPB = 64
#: words at the start of every private row: the fit's phase and the
#: optimizer's state between launches
comptime TEAM_REC = 16


struct SeqTeam(ImplicitlyCopyable, Movable):
    var tid: Int
    var nt: Int

    @always_inline
    def __init__(out self, tid: Int, nt: Int):
        self.tid = tid
        self.nt = nt

    @always_inline
    def lead(self) -> Bool:
        return self.tid == 0

    @always_inline
    def sync(self):
        """A block barrier that orders DEVICE memory (team_barrier: on Apple
        `air.wg.barrier(3, 1)`); nothing in a team of one."""
        comptime if is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu():
            if self.nt > 1:
                team_barrier()


@always_inline
def _ldi(p: FP, i: Int) -> Int:
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _sti(p: FP, i: Int, v: Int):
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def _spend(left: Int, c: Int) -> Int:
    """The budget left after c units (a negative budget is unlimited)."""
    if left < 0:
        return left
    return left - c if left > c else 0


# ---------------------------------------------------------------- GARCH
#: GARCH phases: 0 setup (backcast, bounds, the 64-point start grid);
#: 1 / 3 the first / second Nelder-Mead's start; 2 / 4 its iterations;
#: 5 the outputs; 6 done
comptime GT_DONE = 6

#: lane apple-fast-garchspeed (2026-10-03). `garch_nll_team` runs the
#: variance recursion on the lead thread alone (n dependent steps through
#: device memory: r, s2 and the bounds read back, s2 stored), then deals
#: out the n terms and has every thread fold all n of them: on the M3
#: (FAST, GARCH_REG + GARCH_GRID default) garch synthetic 384 ms, taxi-hourly
#: 581 ms against arch's 106 / 125 ms. For p, o, q <= 1 (the board's
#: GARCH(1, 1)) the recursion without its bounds is AFFINE in the previous
#: variance, sigma2_t = beta sigma2_{t-1} + c_t with c_t = omega +
#: alpha r_{t-1}^2 + gamma r_{t-1}^2 [r_{t-1} < 0], so
#: GARCH_COOP splits the series into one chunk of L points
#: per thread, the ETS_TEAM shape (sequence/ets_team.mojo): pass 1 composes
#: the chunk's map (A, C) from an unknown start, each thread takes its
#: start from the maps before it (the exclusive prefix), and pass 2 walks
#: the chunk from that start with the bounds (`_garch_step_reg`), summing
#: its log-likelihood terms; the total is every thread's ascending fold over
#: the per-chunk sums. A bound that fires (or a NaN) anywhere makes the map
#: wrong past it, so the block then scores the point with the lead's exact
#: recursion (`garch_nll_team`), the same for every thread. The 64-point
#: start grid scores its candidates the same way; the final evaluation keeps
#: the stored recursion (the sigma output and the forecast read r and s2).
#: The fold orders differ, so the bits differ: FAST only, Apple only.
#: Default on FAST + Apple since the M3 A/B (lane/apple-fast-garchspeed
#: 5bc97d9d7, n=1): garch synthetic 383 -> 15.1 ms, taxi-hourly 582 -> 31.0
#: ms; mean_llf -1938.2249 -> -1938.2239 and -1132.955 -> -1131.911 (equal
#: or better). -D MOJOLEARN_GARCH_COOP_OFF restores the lead's recursion;
#: the old -D MOJOLEARN_GARCH_COOP is harmless.
comptime GARCH_COOP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_GARCH_COOP_OFF"]()
)


def garch_team_priv(h: Int, m: Int) -> Int:
    """Words of one thread's private row: the record, x / lo / hi / cand
    (16 each), op_garch's Nelder-Mead and forecast scratch, the cycle
    snapshot."""
    return TEAM_REC + 64 + max(128, 3 * (m + h)) + GARCH_SNAP


def garch_team_shared(n: Int) -> Int:
    """Words of a series' shared row: r, s2, the variance bounds (2n), the
    log-likelihood terms; GARCH_COOP: then the chunk maps (2 words per
    thread), the per-chunk sums and the bound flags."""
    var w = 5 * n
    comptime if GARCH_COOP:
        w += 4 * SEQ_TEAM_TPB
    comptime if GARCH_FOLD32:
        # sequence/fold32.mojo's slots (garch_nll_team), after the maps:
        # never the maps' words, which the next chunked pass writes unsynced
        w += FOLD_L
    return w


@always_inline
def garch_slots_off(n: Int) -> Int:
    """GARCH_FOLD32's slots, in words past the terms row: after the n terms
    and, under GARCH_COOP, the chunk maps, sums and flags."""
    comptime if GARCH_COOP:
        return n + 4 * SEQ_TEAM_TPB
    return n


@always_inline
def garch_nll_team(team: SeqTeam, par: FP, r: FP, n: Int, p: Int, o: Int, q: Int, backcast: Float32,
                   vb: FP, s2: FP, terms: FP) -> Float32:
    """garch_nll over a block: the recursion on the lead, the n terms dealt
    out, their sum every thread's own ascending fold (garch_nll's words)."""
    if team.lead():
        garch_sigma2(par, r, n, p, o, q, backcast, vb, s2)
    team.sync()
    var ll = Float32(0.0)
    comptime if GARCH_FOLD32:
        # sequence/fold32.mojo: slot s (on thread s mod nt) folds the terms
        # s, s + 32, ... ascending from 0; every thread runs the tree on the
        # stored slots (garch_nll's host order). The slots sit past the
        # chunk maps (garch_team_shared); a slot is rewritten only after the
        # next call's sync above, which every reader reaches first.
        var slots = terms + garch_slots_off(n)
        for s in range(team.tid, FOLD_L, team.nt):
            var acc = Float32(0.0)
            for t in range(s, n, FOLD_L):
                var v = ld(s2, t)
                var x = ld(r, t)
                acc = add(acc, add(add(LOG_2PI, ftz(identical_log(v))), ftz(identical_div(mul(x, x), v))))
            st(slots, s, acc)
        team.sync()
        var sl = InlineArray[Float32, FOLD_L](fill=Float32(0.0))
        comptime for j in range(FOLD_L):
            sl[j] = ld(slots, j)
        ll = tree32(sl)
    else:
        for t in range(team.tid, n, team.nt):
            var v = ld(s2, t)
            var x = ld(r, t)
            st(terms, t, add(add(LOG_2PI, ftz(identical_log(v))), ftz(identical_div(mul(x, x), v))))
        team.sync()
        for t in range(n):
            ll = add(ll, ld(terms, t))
    ll = mul(Float32(0.5), ll)
    if not (ll <= Float32(3.0e38)):
        return Float32(3.0e38)
    return ll


@always_inline
def garch_nll_coop(team: SeqTeam, w: Float32, pa: Float32, pg: Float32, pb: Float32, y: FP, mu: Float32,
                   n: Int, p: Int, o: Int, q: Int, backcast: Float32, vb: FP, cs: FP) -> Tuple[Float32, Bool]:
    """GARCH_COOP: -loglik of (omega, alpha, gamma, beta) = (w, pa, pg, pb)
    (unused ones 0), p, o, q <= 1, r_t = y_t - mu, over the block as one
    chunk per thread (the GARCH_COOP comment). cs: the maps [2 nt], the sums
    [nt], the flags [nt]. Returns (value, ok); ok False (the same in every
    thread) when a bound or a NaN fired, and the value is then unset."""
    var tid = team.tid
    var nt = team.nt
    var L = (n + nt - 1) // nt
    var t0 = tid * L
    var t1 = t0 + L
    if t1 > n:
        t1 = n
    if t0 > n:
        t0 = n
    # pass 1: sigma2_{t1 - 1} = A sigma2_{t0 - 1} + C, the bounds left out
    var A = Float32(1.0)
    var C = Float32(0.0)
    for t in range(t0, t1):
        if t < 1:
            var v0 = w
            if p > 0:
                v0 = fma3(pa, backcast, v0)
            if o > 0:
                v0 = fma3(pg, mul(Float32(0.5), backcast), v0)
            if q > 0:
                v0 = fma3(pb, backcast, v0)
            A = Float32(0.0)
            C = v0
        else:
            var x = sub(ld(y, t - 1), mu)
            var x2 = mul(x, x)
            var c = fma3(pa, x2, w)
            if x < Float32(0.0):
                c = fma3(pg, x2, c)
            A = mul(pb, A)
            C = fma3(pb, C, c)
    st(cs, 2 * tid, A)
    st(cs, 2 * tid + 1, C)
    team.sync()
    # the exclusive prefix: sigma2_{t0 - 1} through the maps before this
    # chunk (chunk 0's map starts at t = 0 with A = 0, so the seed is moot)
    var sp = Float32(0.0)
    for j in range(tid):
        sp = fma3(ld(cs, 2 * j), sp, ld(cs, 2 * j + 1))
    var rp = Float32(0.0)
    if t0 > 0 and t0 < n:
        rp = sub(ld(y, t0 - 1), mu)
    # pass 2: the chunk with its bounds, its terms' sum
    var part = Float32(0.0)
    var bad = Float32(0.0)
    for t in range(t0, t1):
        var lo = ld(vb, 2 * t)
        var hi = ld(vb, 2 * t + 1)
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
        if not (v >= lo and v <= hi):
            # a bound (or a NaN): the chunks past this point started wrong
            bad = Float32(1.0)
            v = _garch_step_reg(w, pa, pg, pb, p, o, q, t, rp, sp, backcast, lo, hi)
        var x = sub(ld(y, t), mu)
        part = add(part, add(add(LOG_2PI, ftz(identical_log(v))), ftz(identical_div(mul(x, x), v))))
        rp = x
        sp = v
    st(cs, 2 * nt + tid, part)
    st(cs, 3 * nt + tid, bad)
    team.sync()
    var ll = Float32(0.0)
    var nbad = Float32(0.0)
    for j in range(nt):
        ll = add(ll, ld(cs, 2 * nt + j))
        nbad = add(nbad, ld(cs, 3 * nt + j))
    # the sums and flags are read again only after the next pass 1's sync
    if nbad != Float32(0.0):
        return (Float32(0.0), False)
    ll = mul(Float32(0.5), ll)
    if not (ll <= Float32(3.0e38)):
        return (Float32(3.0e38), True)
    return (ll, True)


struct GarchTeamObj(Objective):
    """GarchObj over a block (sequence/garch.mojo `GarchObj.eval`)."""
    var team: SeqTeam
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
    var terms: FP
    var cs: FP

    @always_inline
    def __init__(out self, team: SeqTeam, y: FP, r: FP, n: Int, p: Int, o: Int, q: Int, has_mean: Bool,
                 backcast: Float32, vb: FP, s2: FP, terms: FP):
        self.team = team
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
        self.terms = terms
        self.cs = terms + n

    @always_inline
    def eval(mut self, x: FP) -> Float32:
        comptime if GARCH_COOP:
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
                var res = garch_nll_coop(self.team, ld(vol, 0), pa, pg, pb, self.y, mu, self.n, self.p, self.o,
                                         self.q, self.backcast, self.vb, self.cs)
                if res[1]:
                    return res[0]
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
        for t in range(self.team.tid, self.n, self.team.nt):
            st(self.r, t, sub(ld(self.y, t), mu))
        self.team.sync()
        return garch_nll_team(self.team, vol, self.r, self.n, self.p, self.o, self.q, self.backcast,
                              self.vb, self.s2, self.terms)


@always_inline
def _nm_save(rec: FP, s: NMState):
    _sti(rec, 3, 1 if s.changed else 0)
    _sti(rec, 4, 1 if s.have_snap else 0)
    _sti(rec, 5, s.snap_it)
    _sti(rec, 6, s.power)
    _sti(rec, 7, s.stop_at)
    _sti(rec, 8, s.it)
    _sti(rec, 9, s.best)
    rec.unsafe_store(10, s.stall_ref)
    _sti(rec, 11, s.stall_at)


@always_inline
def _nm_load(rec: FP) -> NMState:
    var s = NMState()
    s.changed = _ldi(rec, 3) != 0
    s.have_snap = _ldi(rec, 4) != 0
    s.snap_it = _ldi(rec, 5)
    s.power = _ldi(rec, 6)
    s.stop_at = _ldi(rec, 7)
    s.it = _ldi(rec, 8)
    s.best = _ldi(rec, 9)
    s.stall_ref = rec.unsafe_load(10)
    s.stall_at = _ldi(rec, 11)
    return s


def garch_team(slot: Int, team: SeqTeam, a: Args):
    """op_garch for series a.i8 + slot over a block. Arguments as op_garch
    (p0..p4, i0..i7, f0) and: p5 the shared rows [G, i11]; p6 the private
    rows [G, nt, i10] (zero before a group's first launch); p7 the done
    flags [G]; i9 the launch's budget (Nelder-Mead iterations, < 0 none).
    Private record: 0 phase, 1 backcast, 2 iterations so far, 3.. NMState."""
    var b = a.i8 + slot
    var n = a.i0
    var h = a.i1
    var p = a.i2
    var o = a.i3
    var q = a.i4
    var has_mean = a.i5 != 0
    var k = 1 + p + o + q
    var np_ = k + (1 if has_mean else 0)
    var off = 1 if has_mean else 0
    var y = a.p0 + b * n
    var sh = a.p5 + slot * a.i11
    var r = sh
    var s2 = r + n
    var vb = s2 + n
    var terms = vb + 2 * n
    var pv = a.p6 + (slot * team.nt + team.tid) * a.i10
    var rec = pv
    var x = pv + TEAM_REC
    var lo = x + 16
    var hi = lo + 16
    var cand = hi + 16
    var nm_scr = cand + 16
    var snap = pv + (a.i10 - GARCH_SNAP)
    var phase = _ldi(rec, 0)
    var left = a.i9
    var obj = GarchTeamObj(team, y, r, n, p, o, q, has_mean, rec.unsafe_load(1), vb, s2, terms)
    while phase < GT_DONE and left != 0:
        if phase == 0:
            var mu0 = Float32(0.0)
            if has_mean:
                var s = Float32(0.0)
                for i in range(n):
                    s = add(s, ld(y, i))
                mu0 = ftz(identical_div(s, Float32(n)))
            for i in range(team.tid, n, team.nt):
                st(r, i, sub(ld(y, i), mu0))
            team.sync()
            var backcast = _backcast(r, n)
            if team.lead():
                _var_bounds(r, n, vb)
            var target = Float32(0.0)
            for i in range(n):
                var v = ld(r, i)
                target = fma3(v, v, target)
            target = ftz(identical_div(target, Float32(n)))
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
                                st(cand, 1 + j, ftz(identical_div(_grid(ia), Float32(p))))
                            agb = sub(agb, _grid(ia))
                        if o > 0:
                            for j in range(o):
                                st(cand, 1 + p + j, ftz(identical_div(_grid(ig), Float32(o))))
                            agb = sub(agb, ftz(identical_div(_grid(ig), Float32(2.0))))
                        if q > 0:
                            for j in range(q):
                                st(cand, 1 + p + o + j, ftz(identical_div(agb, Float32(q))))
                        var nll = Float32(0.0)
                        var done = False
                        comptime if GARCH_COOP:
                            if p <= 1 and o <= 1 and q <= 1:
                                var res = garch_nll_coop(
                                    team, ld(cand, 0), ld(cand, 1) if p > 0 else Float32(0.0),
                                    ld(cand, 1 + p) if o > 0 else Float32(0.0),
                                    ld(cand, 1 + p + o) if q > 0 else Float32(0.0),
                                    y, mu0, n, p, o, q, backcast, vb, obj.cs)
                                if res[1]:
                                    nll = res[0]
                                    done = True
                        if not done:
                            nll = garch_nll_team(team, cand, r, n, p, o, q, backcast, vb, s2, terms)
                        if nll < best:
                            best = nll
                            for j in range(k):
                                st(x, j + off, ld(cand, j))
            if has_mean:
                var mx = Float32(0.0)
                for i in range(n):
                    var v = abs(ld(y, i))
                    if v > mx:
                        mx = v
                st(x, 0, mu0)
                st(lo, 0, sub(Float32(0.0), fma3(Float32(10.0), mx, Float32(1.0))))
                st(hi, 0, fma3(Float32(10.0), mx, Float32(1.0)))
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
            obj.backcast = backcast
            rec.unsafe_store(1, backcast)
            _sti(rec, 2, 0)
            phase = 1
            left = _spend(left, 64)
        elif phase == 1 or phase == 3:
            var s = nm_start(obj, x, lo, hi, np_, nm_scr, Float32(0.05), Float32(1e-4))
            _nm_save(rec, s)
            phase += 1
            left = _spend(left, np_ + 1)
        elif phase == 2 or phase == 4:
            var s = _nm_load(rec)
            # i7 / f0: the FAST stall stop (sequence/nm.mojo; compiled out of IDENTICAL)
            var used = nm_steps(obj, s, lo, hi, np_, nm_scr, 2000, Float32(1e-6), snap,
                                stall_iters=a.i7, stall_rel=a.f0, budget=left)
            _nm_save(rec, s)
            left = _spend(left, used)
            if s.done:
                _sti(rec, 2, _ldi(rec, 2) + nm_finish(x, np_, nm_scr, s))
                phase += 1
            else:
                left = 0
        else:
            # the sigma output and the forecast read r and s2
            var nll = obj.eval_stored(x)
            if team.lead():
                var outp = a.p1 + b * (1 + k)
                st(outp, 0, ld(x, 0) if has_mean else Float32(0.0))
                for j in range(k):
                    st(outp, 1 + j, ld(x, off + j))
                var info = a.p2 + b * 4
                st(info, 0, -nll)
                st(info, 1, Float32(_ldi(rec, 2)))
            for i in range(team.tid, n, team.nt):
                st(a.p3, b * n + i, ftz(identical_sqrt(ld(s2, i))))
            if team.lead():
                # _analytic_forecast from the last observation (op_garch)
                var par = x + off
                var m = p if p > o else o
                if q > m:
                    m = q
                var rr = nm_scr
                var ar = rr + (m + h)
                var sg = ar + (m + h)
                for j in range(m):
                    var i = n - m + j
                    if i >= 0:
                        var v = ld(r, i)
                        st(rr, j, mul(v, v))
                        st(ar, j, mul(v, v) if v < Float32(0.0) else Float32(0.0))
                        st(sg, j, ld(s2, i))
                    else:
                        st(rr, j, obj.backcast)
                        st(ar, j, mul(Float32(0.5), obj.backcast))
                        st(sg, j, obj.backcast)
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
                    st(a.p4, b * h + hh, v)
            phase = GT_DONE
    _sti(rec, 0, phase)
    if team.lead():
        a.p7.unsafe_store(slot, Float32(1.0) if phase == GT_DONE else Float32(0.0))


# ---------------------------------------------------------------- Prophet
#: Prophet phases: 0 setup and the first objective; 1 L-BFGS iterations;
#: 2 the outputs; 3 done
comptime PT_DONE = 3


def prophet_team_priv(P: Int) -> Int:
    """Words of one thread's private row: the record, th, lbfgs scratch."""
    return TEAM_REC + P + (6 + 2 * MEM) * P + 2 * MEM


def prophet_team_shared(N: Int, P: Int) -> Int:
    """Words of a series' shared row: scaled y, then the points' residuals
    and the two weights (N each), then the P + 1 accumulators."""
    return 4 * N + P + 1


struct TeamFG(ProphetFG):
    """prophet_fg over a block: the points' residuals and weights dealt out
    (pass A), then each of the P gradient accumulators and the sse ONE
    thread's ascending chain over the points (pass B): op_prophet_fit's
    `_fg_data` chain for that accumulator, word for word; the priors
    (`_fg_prior`) in every thread on its private g."""
    # the data pointers and scalars as plain fields, taken straight from
    # Args (no ProphetData struct on the Metal team path: lane
    # apple-fast-prophetfix, the team fit returned the initial line on Apple)
    var team: SeqTeam
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
    var gs: FP

    @always_inline
    def __init__(out self, team: SeqTeam, t: FP, X: FP, cp: FP, sig: FP, N: Int, K: Int, S: Int,
                 tau: Float32, mult: Bool, y: FP, rr: FP, wt: FP, wb: FP, gs: FP):
        self.team = team
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
        self.gs = gs

    @always_inline
    def fg(mut self, th: FP, g: FP) -> Float32:
        var S = self.S
        var K = self.K
        var N = self.N
        var P = 3 + S + K
        var tt = self.t
        var cp = self.cp
        var X = self.X
        var k = ld(th, 0)
        var m = ld(th, 1)
        var u = ld(th, 2 + S)
        var sigma = ftz(identical_exp(u))
        var s2 = mul(sigma, sigma)
        # pass A: point i's residual and weights (`_fg_data`'s per-point values)
        for i in range(self.team.tid, N, self.team.nt):
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
        # pass B: accumulator j over the points in ascending order
        for j in range(self.team.tid, P + 1, self.team.nt):
            var acc = Float32(0.0)
            if j == 0:
                for i in range(N):
                    acc = fma3(ld(self.wt, i), ld(tt, i), acc)
            elif j == 1:
                for i in range(N):
                    acc = add(acc, ld(self.wt, i))
            elif j < 2 + S:
                var c = ld(cp, j - 2)
                for i in range(N):
                    var ti = ld(tt, i)
                    if ti >= c:
                        acc = fma3(ld(self.wt, i), sub(ti, c), acc)
            elif j == 2 + S:
                pass
            elif j < P:
                var q = j - 3 - S
                for i in range(N):
                    acc = fma3(ld(self.wb, i), ld(X, i * K + q), acc)
            else:
                for i in range(N):
                    var r = ld(self.rr, i)
                    acc = fma3(r, r, acc)
            st(self.gs, j, acc)
        self.team.sync()
        for j in range(P):
            st(g, j, ld(self.gs, j))
        return _fg_prior_v(self.sig, self.tau, N, S, K, th, g, ld(self.gs, P))


@always_inline
def prophet_fit_team(slot: Int, team: SeqTeam, a: Args):
    """op_prophet_fit for series a.i8 + slot over a block. Arguments as
    op_prophet_fit (p0..p6, i0..i3, i5, f0) and: p7 the shared rows [G, i11];
    p8 the private rows [G, nt, i10] (zero before a group's first launch);
    p9 the done flags [G]; i9 the launch's budget (L-BFGS iterations, < 0
    none). Private record: 0 phase, 1 y scale, 2 f, 3 npairs, 4 head,
    5 small, 6 iterations."""
    var b = a.i8 + slot
    var N = a.i0
    var K = a.i1
    var S = a.i2
    var P = 3 + S + K
    var y = a.p0 + b * N
    var sh = a.p7 + slot * a.i11
    var ys = sh
    var rr = ys + N
    var wt = rr + N
    var wb = wt + N
    var gs = wb + N
    var pv = a.p8 + (slot * team.nt + team.tid) * a.i10
    var rec = pv
    var th = pv + TEAM_REC
    var w = th + P
    var fg = TeamFG(team, a.p1, a.p2, a.p3, a.p4, N, K, S, a.f0, a.i3 != 0, ys, rr, wt, wb, gs)
    var phase = _ldi(rec, 0)
    var left = a.i9
    while phase < PT_DONE and left != 0:
        if phase == 0:
            var scale = Float32(0.0)
            for i in range(N):
                var v = abs(ld(y, i))
                if v > scale:
                    scale = v
            if scale == Float32(0.0):
                scale = Float32(1.0)
            for i in range(team.tid, N, team.nt):
                st(ys, i, ftz(identical_div(ld(y, i), scale)))
            team.sync()
            # linear_growth_init: through the first and last points
            var t0 = ld(a.p1, 0)
            var t1 = ld(a.p1, N - 1)
            var k = ftz(identical_div(sub(ld(ys, N - 1), ld(ys, 0)), sub(t1, t0))) if t1 != t0 else Float32(0.0)
            st(th, 0, k)
            st(th, 1, sub(ld(ys, 0), mul(k, t0)))
            for j in range(S + 1 + K):
                st(th, 2 + j, Float32(0.0))
            var s = lbfgs_start(fg, th, w)
            rec.unsafe_store(1, scale)
            rec.unsafe_store(2, s.f)
            _sti(rec, 3, s.npairs)
            _sti(rec, 4, s.head)
            _sti(rec, 5, s.small)
            _sti(rec, 6, s.it)
            phase = 1
            left = _spend(left, 1)
        elif phase == 1:
            var s = LBState(rec.unsafe_load(2))
            s.npairs = _ldi(rec, 3)
            s.head = _ldi(rec, 4)
            s.small = _ldi(rec, 5)
            s.it = _ldi(rec, 6)
            var used = lbfgs_steps(fg, s, P, th, w, a.i5, left)
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
