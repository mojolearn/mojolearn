# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Adafactor device vs host, stage by stage (the Apple FAIL on
sequence-adafactor, steward request 1790537359124-sequence-3607aa10ee:
state equal, params differ). One step of a 32 x 8 matrix and an 8-vector,
the scalars, the update and the params downloaded after every launch.

    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/af_probe.mojo
"""
from std.memory import bitcast
from std.math import sqrt

from checks.fixture_rng import hashed_signed_f32
from checks.numerics import ftz, identical_mul, identical_pow64
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import (
    FP, Args, OP_AF_ALPHA, OP_AF_ROW, OP_AF_COL, OP_AF_RMEAN, OP_AF_UPDATE_MAT, OP_AF_VEC, OP_AF_DENOM,
    OP_AF_APPLY,
)


def _fp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _get[E: Exec](mut ex: E, src: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    ex.sync()
    ex.download(_fp(out), src, n)
    return out^


def _run[E: Exec](mut ex: E, R: Int, C: Int) raises -> List[List[Float32]]:
    """Snapshots [sc, U, P] after alpha, the second moment, rmean, update,
    denom and apply."""
    var n = R * C if C > 0 else R
    var p = List[Float32](capacity=n)
    var g = List[Float32](capacity=n)
    for i in range(n):
        p.append(hashed_signed_f32(UInt64(71), i))
        g.append(hashed_signed_f32(UInt64(72), i) * Float32(0.5))
    var lr = Float64(1e-2)
    var t = 1
    var w = Float32(identical_pow64(Float64(t), Float64(-0.8)))
    var rho = Float32(min(lr, Float64(1.0) / sqrt(Float64(t))))
    var eps1 = Float32(1.1920928955078125e-07)
    var eps1sq = ftz(identical_mul(eps1, eps1))
    var P = ex.alloc(n)
    var G = ex.alloc(n)
    var S1 = ex.alloc(n if C == 0 else R)
    var S2 = ex.alloc(C if C > 0 else 1)
    var U = ex.alloc(n)
    var sc = ex.alloc(4)
    ex.upload(P, _fp(p), n)
    ex.upload(G, _fp(g), n)
    var snaps = List[List[Float32]]()

    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = n
    a.f0 = Float32(1e-3)
    a.f1 = rho
    ex.launch[OP_AF_ALPHA](a, 1)
    snaps.append(_get(ex, sc, 4))
    if C > 0:
        var r = Args()
        r.p0 = G
        r.p1 = S1
        r.i0 = C
        r.f0 = w
        ex.launch[OP_AF_ROW](r, R)
        var c = Args()
        c.p0 = G
        c.p1 = S2
        c.i0 = R
        c.i1 = C
        c.f0 = w
        ex.launch[OP_AF_COL](c, C)
        snaps.append(_get(ex, S1, R))
        var m = Args()
        m.p0 = S1
        m.p1 = sc
        m.i0 = R
        m.f0 = eps1
        ex.launch[OP_AF_RMEAN](m, 1)
        snaps.append(_get(ex, sc, 4))
        var u = Args()
        u.p0 = G
        u.p1 = S1
        u.p2 = S2
        u.p3 = sc
        u.p4 = U
        u.i0 = C
        u.f0 = eps1sq
        ex.launch[OP_AF_UPDATE_MAT](u, n)
    else:
        var v = Args()
        v.p0 = G
        v.p1 = S1
        v.p2 = U
        v.f0 = w
        v.f1 = eps1sq
        ex.launch[OP_AF_VEC](v, n)
        snaps.append(_get(ex, S1, n))
    snaps.append(_get(ex, U, n))
    var d = Args()
    d.p0 = U
    d.p1 = sc
    d.i0 = n
    d.f0 = Float32(1.0)
    ex.launch[OP_AF_DENOM](d, 1)
    snaps.append(_get(ex, sc, 4))
    var ap = Args()
    ap.p0 = P
    ap.p1 = U
    ap.p2 = sc
    ex.launch[OP_AF_APPLY](ap, n)
    snaps.append(_get(ex, P, n))
    snaps.append(p.copy())
    return snaps^


def _show(name: String, h: List[Float32], d: List[Float32]):
    var nd = 0
    for i in range(len(h)):
        if bitcast[DType.uint32](h[i]) != bitcast[DType.uint32](d[i]):
            nd += 1
    print(name, "n", len(h), "differ", nd)
    var k = min(len(h), 4)
    for i in range(k):
        print("   ", i, "host", h[i], bitcast[DType.uint32](h[i]), "dev", d[i], bitcast[DType.uint32](d[i]))


def main() raises:
    var names = List[String]()
    for shape in [(32, 8), (8, 0)]:
        var R = shape[0]
        var C = shape[1]
        var hx = HostExec()
        var dx = DeviceExec()
        var h = _run(hx, R, C)
        var d = _run(dx, R, C)
        print("== shape", R, C)
        var labels = List[String]()
        labels.append("alpha sc")
        if C > 0:
            labels.append("row_var")
            labels.append("rmean sc")
        else:
            labels.append("variance")
        labels.append("U")
        labels.append("denom sc")
        labels.append("P after")
        labels.append("P before (host copy)")
        for i in range(len(h)):
            _show(labels[i], h[i], d[i])
