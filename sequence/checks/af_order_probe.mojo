# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Metal Adafactor FAIL, launch order: through the real DeviceExec.launch,
fresh buffers per case (P = 256 ones, sc pre-filled with 7), which one-thread
stores land. On Metal, OP_AF_ALPHA's store was lost as the first launch
(af_alpha_probe.mojo) while OP_AF_RMEAN / OP_AF_DENOM later in the step
landed; af_kb/ showed even a constant store lost as a process's only launch.

    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/af_order_probe.mojo
"""
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_AF_ALPHA, OP_AF_RMEAN, OP_SCALE


def _fp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _alpha[E: Exec](mut ex: E, P: FP, sc: FP):
    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = 256
    a.f0 = Float32(0.5)
    a.f1 = Float32(3.0)
    try:
        ex.launch[OP_AF_ALPHA](a, 1)
    except:
        print("launch raised")


def _rmean[E: Exec](mut ex: E, P: FP, sc: FP):
    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = 256
    a.f0 = Float32(0.5)
    try:
        ex.launch[OP_AF_RMEAN](a, 1)
    except:
        print("launch raised")


def _scale[E: Exec](mut ex: E, P: FP):
    var a = Args()
    a.p0 = P
    a.f0 = Float32(1.0)
    try:
        ex.launch[OP_SCALE](a, 256)
    except:
        print("launch raised")


def _case[E: Exec](mut ex: E, c: Int) raises -> List[Float32]:
    var p = List[Float32](length=256, fill=Float32(1.0))
    var s = List[Float32](length=4, fill=Float32(7.0))
    var P = ex.alloc(256)
    var sc = ex.alloc(4)
    ex.upload(P, _fp(p), 256)
    ex.upload(sc, _fp(s), 4)
    if c == 0:
        _alpha(ex, P, sc)
    elif c == 1:
        _alpha(ex, P, sc)
        _alpha(ex, P, sc)
    elif c == 2:
        _rmean(ex, P, sc)
    elif c == 3:
        _rmean(ex, P, sc)
        _alpha(ex, P, sc)
    elif c == 4:
        _alpha(ex, P, sc)
        _rmean(ex, P, sc)
    elif c == 5:
        _scale(ex, P)
        _alpha(ex, P, sc)
    elif c == 6:
        _alpha(ex, P, sc)
        ex.sync()
        _alpha(ex, P, sc)
    ex.sync()
    var out = List[Float32](length=4, fill=Float32(-1.0))
    ex.download(_fp(out), sc, 4)
    _ = p^
    _ = s^
    return out^


def main() raises:
    var names: List[String] = ["alpha once", "alpha twice", "rmean once", "rmean then alpha", "alpha then rmean",
                               "scale(n threads) then alpha", "alpha, sync, alpha"]
    var hx = HostExec()
    var dx = DeviceExec()
    for c in range(len(names)):
        var h = _case(hx, c)
        var d = _case(dx, c)
        print("case " + names[c] + ": host " + String(h[1]) + " " + String(h[2]) + " dev " + String(d[1]) + " " + String(d[2])
              + (" SAME" if h[1] == d[1] and h[2] == d[2] else " DIFF"))
    _ = dx^
    _ = hx^
