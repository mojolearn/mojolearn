# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's GARCH and Prophet fit entries (lane neural-pass143): one
series per block (`sequence/fit_team.mojo`), the whole fit in one launch on
NVIDIA and AMD, and on Apple sliced over launches of bounded work (macOS
aborts a long command buffer silently: memory MACOS ABORTS LONG METAL
LAUNCHES). Same arguments and outputs as `garch_py` / `prophet_fit_py`."""
from std.python import PythonObject
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec_device import DeviceExec
from sequence.fit_team import (
    SEQ_TEAM_TPB,
    garch_team_priv,
    garch_team_shared,
    prophet_team_priv,
    prophet_team_shared,
)
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
comptime SEQ_TEAM_BYTES = 1 << 28
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


def _run_team[OP: Int](mut ex: DeviceExec, a: Args, g: Int, flags: FP) raises:
    """Launches the group until every series is done: one launch where the
    budget is unbounded, else until the done flags all read 1."""
    var fl = List[Float32](length=g, fill=Float32(0.0))
    while True:
        ex.launch_team[OP](a, g, SEQ_TEAM_TPB)
        if a.i9 < 0:
            return
        ex.download(fptr_of(fl), flags, g)
        var all_done = True
        for i in range(g):
            if fl[i] != Float32(1.0):
                all_done = False
                break
        if all_done:
            return


def _zero(mut ex: DeviceExec, p: FP, n: Int) raises:
    var f = Args()
    f.p0 = p
    f.f0 = Float32(0.0)
    ex.launch[OP_FILL](f, n)


def _group(B: Int, shared: Int, priv: Int) -> Int:
    """Series per group: as many as SEQ_TEAM_BYTES holds, at least one."""
    var per = (shared + SEQ_TEAM_TPB * priv) * 4
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
        _run_team[OP_GARCH](ex, a, g, Fl)
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
    var R = prophet_team_priv(P)
    var SH = prophet_team_shared(N, P)
    var G = _group(B, SH, R)
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
    var Pv = ex.alloc(G * SEQ_TEAM_TPB * R)
    var Fl = ex.alloc(G)
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
    var s0 = 0
    while s0 < B:
        var g = min(G, B - s0)
        if s0 > 0:
            _zero(ex, Pv, g * SEQ_TEAM_TPB * R)
        a.i8 = s0
        a.i9 = _team_budget(g, 2 * per_eval)
        _run_team[OP_PROPHET_FIT](ex, a, g, Fl)
        s0 += g
    ex.sync()
    ex.download(fptr(addrs[7], "params"), Pm, B * P)
    ex.download(fptr(addrs[8], "info"), I, B * 4)
    return PythonObject(B)
