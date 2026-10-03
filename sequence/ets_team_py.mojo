# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's ETS entry over one block per series
(sequence/ets_team.mojo; Apple FAST, `-D MOJOLEARN_ETS_TEAM`): `ets_py`'s
contract (sequence/pyapi.mojo) for ETS(A, A|Ad, N). The fit is one group
launch (sliced on Apple only where the command-buffer bound needs it,
`sequence/fit_team_py.mojo::_run_team`) and one wait; the forecast and the
info rows come back together. Every other model, and every other build,
keeps `ets_py`."""
from std.python import PythonObject

from sequence.ets_team import ets_team_priv, ets_team_shared
from sequence.exec_device import DeviceExec
from sequence.fit_team import SEQ_TEAM_TPB
from sequence.fit_team_py import _group, _run_team, _saves, _team_budget, _zero
from sequence.ops import OP_ETS, Args
from sequence.pyapi import ETS_FAST_STALL_ITERS, ETS_FAST_STALL_REL, fptr, fval, ival
from x_linear.witness import Witness


def ets_team_applies(ip: PythonObject) raises -> Bool:
    """Whether ets_py's integer parameters name the model the team fits:
    additive errors (ip[3] 0), an additive trend (ip[4] 1), no season
    (ip[7] 0)."""
    if len(ip) != 9 and len(ip) != 10:
        return False
    return ival(ip, 3) == 0 and ival(ip, 4) == 1 and ival(ip, 7) == 0


def ets_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """ets_py's contract over one block per series (ETS(A, A|Ad, N))."""
    if len(addrs) != 4 or (len(ip) != 9 and len(ip) != 10) or (len(fp) != 4 and len(fp) != 5):
        raise Error("ets: requires 4 addresses, 9 (or 10) integer and 4 (or 5) float parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var h = ival(ip, 2)
    if B < 1 or n < 4 or h < 1:
        raise Error("ets: B >= 1, n >= 4 and h >= 1")
    var R = ets_team_priv()
    var SH = ets_team_shared()
    var G = _group(B, SH, R)
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var F = ex.alloc(B * h)
    var I = ex.alloc(B * 10)
    var Sh = ex.alloc(G * SH)
    var Pv = ex.alloc(G * SEQ_TEAM_TPB * R)
    var Fl = ex.alloc(G)
    var sv = _saves(ex, G * SH, G * SEQ_TEAM_TPB * R, Sh, Pv)
    var wit = Witness(ex.ctx, G)
    var a = Args()
    a.p0 = Y
    a.p1 = F
    a.p2 = I
    a.p5 = Sh
    a.p6 = Pv
    a.p7 = Fl
    a.i0 = n
    a.i1 = h
    a.i2 = ival(ip, 9) if len(ip) == 10 else ETS_FAST_STALL_ITERS
    a.i4 = ival(ip, 5)
    a.i5 = ival(ip, 6)
    a.i10 = R
    a.i11 = SH
    a.f0 = fval(fp, 0)
    a.f1 = fval(fp, 1)
    a.f2 = fval(fp, 2)
    a.f4 = fval(fp, 4) if len(fp) == 5 else ETS_FAST_STALL_REL
    # a thread's two chunk walks and its prefix per evaluation, against the
    # lead's GARCH step (two evaluations per Nelder-Mead iteration)
    var L = (n + SEQ_TEAM_TPB - 1) // SEQ_TEAM_TPB
    var per_eval = 2 * L + 2 * SEQ_TEAM_TPB
    var s0 = 0
    while s0 < B:
        var g = min(G, B - s0)
        if s0 > 0:
            _zero(ex, Pv, g * SEQ_TEAM_TPB * R)
        a.i8 = s0
        a.i9 = _team_budget(g, 2 * per_eval)
        _run_team[OP_ETS](ex, a, g, Fl, wit, "ETS fit slice", Sh, Pv, g * SH, g * SEQ_TEAM_TPB * R,
                          sv[0], sv[1])
        s0 += g
    ex.sync()
    ex.download(fptr(addrs[1], "forecast"), F, B * h)
    ex.download(fptr(addrs[2], "info"), I, B * 10)
    return PythonObject(B * h)
