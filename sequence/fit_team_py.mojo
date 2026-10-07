# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's GARCH and Prophet fit entries (lane neural-pass143): one
series per block (`sequence/fit_team.mojo`), the whole fit in one launch on
NVIDIA and AMD, and on Apple sliced over launches of bounded work (macOS
aborts a long command buffer silently: memory MACOS ABORTS LONG METAL
LAUNCHES). Same arguments and outputs as `garch_py` / `prophet_fit_py`.

Every slice ends each block with the completion witness
(x_linear/witness.mojo). On Apple a slice is one idempotent unit: the
group's private rows (each thread's phase, optimizer state and scratch) and
shared rows are copied aside before it, and a slice whose words do not all
read the nonce (cut by macOS, or its copies dropped) gets them back and runs
again, the same iterations from the same state, up to WITNESS_TRIES times;
then the fit raises. NVIDIA and AMD run the fit in one launch with no check
(witness_end compiles to nothing there).
`-D MOJOLEARN_WITNESS_SABOTAGE=1` makes every Apple GARCH / Prophet fit
raise (the test that the check is wired)."""
from std.python import PythonObject
from experiments.classical_identical_ideas.stats_controls import C58_TEAM_MIB
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec_device import DeviceExec
from x_linear.witness import WITNESS_TRIES, Witness
from sequence.fit_team import (
    SEQ_TEAM_TPB,
    garch_team_priv,
    garch_team_shared,
    prophet_team_priv,
    prophet_team_shared,
)
from sequence.prophet_coop import PROPHET_COOP, PROPHET_COOP_TPB, prophet_coop_priv, prophet_coop_shared
from sequence.ops import FP, OP_FILL, OP_GARCH, OP_PROPHET_FIT, Args
from sequence.pyapi import (
    GARCH_FAST_STALL_ITERS,
    GARCH_FAST_STALL_REL,
    PROPHET_FAST_MIN_N,
    _prophet_X,
    fptr,
    fptr_of,
    fval,
    ival,
    prophet_fit_py,
)

#: device words a group of series may hold in shared and private rows; a
#: larger batch runs as several groups over the same buffers
# C58 independent classical-series grouping: cap live optimizer/rollback state
# at C58_TEAM_MIB (int sweep 64|256, absent = incumbent 256 MiB). Grouping follows actual state bytes,
# including Apple rollback copies, rather than series count or dataset shape.
# Same kernels, same series-local trial order; all groups complete before return.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime SEQ_TEAM_BYTES = C58_TEAM_MIB << 20
#: Apple: the lead-thread steps one launch may take over all of its series
#: (a bound on the command buffer's length, not on any value)
comptime SEQ_TEAM_APPLE_STEPS = 1 << 27


def _team_budget(g: Int, steps_per_iter: Int) -> Int:
    """Optimizer iterations per launch: unbounded (< 0) on NVIDIA and AMD; on
    Apple at most SEQ_TEAM_APPLE_STEPS lead steps over the group."""
    comptime if has_apple_gpu_accelerator():
        return max(1, SEQ_TEAM_APPLE_STEPS // max(1, g * steps_per_iter))
    else:
        return -1


def _run_team[OP: Int](mut ex: DeviceExec, a: Args, g: Int, flags: FP, mut wit: Witness, what: String,
                       sh: FP, pv: FP, nsh: Int, npv: Int, sv_sh: FP, sv_pv: FP,
                       tpb: Int = SEQ_TEAM_TPB) raises:
    """Launches the group until every series is done: one launch where the
    budget is unbounded (NVIDIA, AMD), else (Apple) slice after slice until
    the done flags all read 1, each slice witness-checked and rerun from
    its saved starting state: the group's nsh shared words at sh and npv
    private words at pv, kept at sv_sh / sv_pv."""
    if a.i9 < 0:
        var nonce0 = wit.begin()
        ex.launch_team[OP](a, g, tpb, wit.p(), 0, nonce0)
        return
    var fl = List[Float32](length=g, fill=Float32(0.0))
    while True:
        # the slice's starting state (the rows are all a slice reads that
        # an earlier slice wrote; the outputs are rewritten from them)
        ex.copy(sv_pv, pv, npv)
        ex.copy(sv_sh, sh, nsh)
        var tries = 0
        while True:
            var nonce = wit.begin()
            ex.launch_team[OP](a, g, tpb, wit.p(), 0, nonce)
            if wit.ok(ex.ctx, g, what):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
            ex.copy(pv, sv_pv, npv)
            ex.copy(sh, sv_sh, nsh)
        ex.download(fptr_of(fl), flags, g)
        var all_done = True
        for i in range(g):
            if fl[i] != Float32(1.0):
                all_done = False
                break
        if all_done:
            return


def _saves(mut ex: DeviceExec, nsh: Int, npv: Int, sh: FP, pv: FP) raises -> Tuple[FP, FP]:
    """The slice-start copies of a group's shared and private rows: Apple
    only (elsewhere the fit is one launch and the rows stand in)."""
    comptime if has_apple_gpu_accelerator():
        return (ex._alloc(nsh, False), ex._alloc(npv, False))
    else:
        return (sh, pv)


def _zero(mut ex: DeviceExec, p: FP, n: Int) raises:
    var f = Args()
    f.p0 = p
    f.f0 = Float32(0.0)
    ex.launch[OP_FILL](f, n)


def _group(B: Int, shared: Int, priv: Int) -> Int:
    """Series per group: as many as SEQ_TEAM_BYTES holds, at least one."""
    var per = (shared + SEQ_TEAM_TPB * priv) * 4
    comptime if has_apple_gpu_accelerator():
        per *= 2  # the slice-start copies (_saves)
    return min(B, max(1, SEQ_TEAM_BYTES // per))


def garch_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """garch_py's contract (sequence/pyapi.mojo) over one block per series."""
    if len(addrs) != 5 or (len(ip) != 7 and len(ip) != 9):
        raise Error("garch: requires 5 addresses and 7 (or 9) integer parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var h = ival(ip, 2)
    var p = ival(ip, 3)
    var o = ival(ip, 4)
    var q = ival(ip, 5)
    var cm = ival(ip, 6)
    if B < 1 or n < 10 or h < 1 or p < 0 or o < 0 or q < 0 or p + o + q < 1 or 1 + p + o + q + cm > 8:
        raise Error("garch: B >= 1, n >= 10, h >= 1, p + o + q >= 1 and at most 8 parameters")
    var m = max(p, max(o, q))
    var R = garch_team_priv(h, m)
    var SH = garch_team_shared(n)
    var G = _group(B, SH, R)
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var Pp = ex.alloc(B * (1 + 1 + p + o + q))
    var I = ex.alloc(B * 4)
    var Sg = ex.alloc(B * n)
    var F = ex.alloc(B * h)
    var Sh = ex.alloc(G * SH)
    var Pv = ex.alloc(G * SEQ_TEAM_TPB * R)
    var Fl = ex.alloc(G)
    var sv = _saves(ex, G * SH, G * SEQ_TEAM_TPB * R, Sh, Pv)
    var wit = Witness(ex.ctx, G)
    var a = Args()
    a.p0 = Y
    a.p1 = Pp
    a.p2 = I
    a.p3 = Sg
    a.p4 = F
    a.p5 = Sh
    a.p6 = Pv
    a.p7 = Fl
    a.i0 = n
    a.i1 = h
    a.i2 = p
    a.i3 = o
    a.i4 = q
    a.i5 = cm
    a.i7 = ival(ip, 7) if len(ip) == 9 else GARCH_FAST_STALL_ITERS
    a.f0 = Float32(Float64(ival(ip, 8)) * 1e-9) if len(ip) == 9 else GARCH_FAST_STALL_REL
    a.i10 = R
    a.i11 = SH
    var s0 = 0
    while s0 < B:
        var g = min(G, B - s0)
        if s0 > 0:
            _zero(ex, Pv, g * SEQ_TEAM_TPB * R)
        a.i8 = s0
        a.i9 = _team_budget(g, 2 * n)
        _run_team[OP_GARCH](ex, a, g, Fl, wit, "GARCH fit slice", Sh, Pv, g * SH, g * SEQ_TEAM_TPB * R,
                            sv[0], sv[1])
        s0 += g
    ex.sync()
    ex.download(fptr(addrs[1], "params"), Pp, B * (1 + 1 + p + o + q))
    ex.download(fptr(addrs[2], "info"), I, B * 4)
    ex.download(fptr(addrs[3], "sigma"), Sg, B * n)
    ex.download(fptr(addrs[4], "forecast"), F, B * h)
    return PythonObject(B)


def prophet_fit_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """prophet_fit_py's contract over one block per series (FAST's long
    series keep prophet_fit_py's chunked likelihood)."""
    if len(addrs) != 9 or len(ip) != 8 or len(fp) != 1:
        raise Error("prophet_fit: requires 9 addresses, 8 integer and 1 float parameters")
    var B = ival(ip, 0)
    var N = ival(ip, 1)
    var ns = ival(ip, 2)
    var nh = ival(ip, 3)
    var K = ival(ip, 4)
    var S = ival(ip, 5)
    if B < 1 or N < 2 or ns < 0 or nh < 0 or K < 0 or S < 0:
        raise Error("prophet_fit: B >= 1, N >= 2 and nonnegative counts")
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if N >= PROPHET_FAST_MIN_N:
            return prophet_fit_py(ex, addrs, ip, fp)
    var P = 3 + S + K
    # rows per series: R words per thread (NT threads), SH shared; the
    # cooperative path (sequence/prophet_coop.mojo) keeps one private area
    # of R words per series
    var NT = SEQ_TEAM_TPB
    var R = prophet_team_priv(P)
    var SH = prophet_team_shared(N, P)
    var G = _group(B, SH, R)
    var RT = NT * R
    comptime if PROPHET_COOP:
        NT = PROPHET_COOP_TPB
        R = prophet_coop_priv(P, NT)
        SH = prophet_coop_shared(N, P, NT)
        G = _group(B, SH + R, 0)
        RT = R
    var X = _prophet_X(ex, addrs[2], addrs[3], addrs[4], N, ns, nh, K)
    var Y = ex.alloc(B * N)
    ex.upload(Y, fptr(addrs[0], "y"), B * N)
    var T = ex.alloc(N)
    ex.upload(T, fptr(addrs[1], "t"), N)
    var C_ = ex.alloc(max(S, 1))
    if S > 0:
        ex.upload(C_, fptr(addrs[5], "changepoints"), S)
    var Sg = ex.alloc(max(K, 1))
    if K > 0:
        ex.upload(Sg, fptr(addrs[6], "prior scales"), K)
    var Pm = ex.alloc(B * P)
    var I = ex.alloc(B * 4)
    var Sh = ex.alloc(G * SH)
    var Pv = ex.alloc(G * RT)
    var Fl = ex.alloc(G)
    var sv = _saves(ex, G * SH, G * RT, Sh, Pv)
    var wit = Witness(ex.ctx, G)
    var a = Args()
    a.p0 = Y
    a.p1 = T
    a.p2 = X
    a.p3 = C_
    a.p4 = Sg
    a.p5 = Pm
    a.p6 = I
    a.p7 = Sh
    a.p8 = Pv
    a.p9 = Fl
    a.i0 = N
    a.i1 = K
    a.i2 = S
    a.i3 = ival(ip, 6)
    a.i5 = ival(ip, 7)
    a.f0 = fval(fp, 0)
    a.i10 = R
    a.i11 = SH
    # a point's pass A and pass B work, against the lead's GARCH step
    var per_eval = N * (2 + (S + K + SEQ_TEAM_TPB - 1) // SEQ_TEAM_TPB)
    comptime if PROPHET_COOP:
        # a thread's share: its N / NT points in pass A and in each of the
        # P + 1 block-wide accumulators
        per_eval = (N + NT - 1) // NT * (S + K + P + 1) + 2 * (P + 1)
    var s0 = 0
    while s0 < B:
        var g = min(G, B - s0)
        if s0 > 0:
            _zero(ex, Pv, g * RT)
        a.i8 = s0
        a.i9 = _team_budget(g, 2 * per_eval)
        _run_team[OP_PROPHET_FIT](ex, a, g, Fl, wit, "Prophet fit slice", Sh, Pv, g * SH, g * RT,
                                  sv[0], sv[1], NT)
        s0 += g
    ex.sync()
    ex.download(fptr(addrs[7], "params"), Pm, B * P)
    ex.download(fptr(addrs[8], "info"), I, B * 4)
    return PythonObject(B)
