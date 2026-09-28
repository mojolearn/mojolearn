# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OP_AF_ALPHA alone, device vs host, with controlled inputs (the Apple
Adafactor FAIL: af_probe.mojo showed sc[1] = 0 on Metal after this launch,
every other stage equal). sc is pre-filled with 7 so a missing store and a
stored zero differ. Cases separate the float slots (f0 low half, f1 high
half of one Int64 word) from the sum of squares.

    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/af_alpha_probe.mojo
"""
from std.memory import bitcast

from checks.fixture_rng import hashed_signed_f32
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_AF_ALPHA


def _fp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _case[E: Exec](mut ex: E, n: Int, kind: Int, f0: Float32, f1: Float32) raises -> List[Float32]:
    var p = List[Float32](capacity=n)
    for i in range(n):
        if kind == 0:
            p.append(Float32(1.0))
        elif kind == 1:
            p.append(Float32(0.0))
        else:
            p.append(hashed_signed_f32(UInt64(71), i))
    var s = List[Float32](length=4, fill=Float32(7.0))
    var P = ex.alloc(n)
    var sc = ex.alloc(4)
    ex.upload(P, _fp(p), n)
    ex.upload(sc, _fp(s), 4)
    ex.sync()
    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = n
    a.f0 = f0
    a.f1 = f1
    ex.launch[OP_AF_ALPHA](a, 1)
    ex.sync()
    var out = List[Float32](length=4, fill=Float32(-1.0))
    ex.download(_fp(out), sc, 4)
    ex.sync()
    _ = p^
    _ = s^
    return out^


def main() raises:
    var hx = HostExec()
    var dx = DeviceExec()
    var names: List[String] = ["ones4 f0=0 f1=1 (want 1)", "ones4 f0=0 f1=3 (want 3)", "ones4 f0=5 f1=1 (want 5)",
                              "zeros4 f0=5 f1=3 (want 15)", "ones4 f0=0.5 f1=0.25 (want 0.25)",
                              "rand256 f0=1e-3 f1=0.01 (the FAIL)", "rand256 f0=0 f1=1 (rms)", "ones256 f0=0 f1=1 (want 1)"]
    var ns: List[Int] = [4, 4, 4, 4, 4, 256, 256, 256]
    var ks: List[Int] = [0, 0, 0, 1, 0, 2, 2, 0]
    var f0s: List[Float32] = [0.0, 0.0, 5.0, 5.0, 0.5, 1e-3, 0.0, 0.0]
    var f1s: List[Float32] = [1.0, 3.0, 1.0, 3.0, 0.25, 0.01, 1.0, 1.0]
    var bad = 0
    for c in range(len(ns)):
        var h = _case(hx, ns[c], ks[c], f0s[c], f1s[c])
        var d = _case(dx, ns[c], ks[c], f0s[c], f1s[c])
        var line = String("case ") + names[c] + ":"
        for j in range(4):
            line += " [" + String(j) + "] host " + String(h[j]) + " dev " + String(d[j])
            if bitcast[DType.uint32](h[j]) != bitcast[DType.uint32](d[j]):
                line += " DIFF"
                bad += 1
        print(line)
    print("DIFFERING SLOTS", bad)
    _ = dx^
    _ = hx^
