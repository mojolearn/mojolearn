# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MDS's SMACOF in Mojo (lane py-runtime-b, 2026-10-05): `_expansion_decomp.
MDS._single` (metric and non-metric, the latter's disparities through
x_decomp/mds_iso.mojo's setup and isotonic step, `_nm_native`'s host form)
and `fit_transform`'s restart pick (the first strictly smaller stress),
statement for statement on `Kit[E]`: the same cells and float32 scalars and
Python's float64 stress tests in the same order, so the IDENTICAL words are
the Python driver's. x_decomp/mds_dev.mojo runs the same on resident
matrices with x_decomp/mds_iso_dev.mojo's kernels."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ADD, OP_DIV, OP_FMA, OP_SCALE, OP_SELECT, OP_SQ, OP_SQDIFF, OP_SQRT, mat_const, mat_eye,
    mat_from, mat_rows, mat_t,
)
from x_decomp.mds_iso import mds_disp_host, mds_setup_host


struct MdsArgs(Copyable, Movable):
    var max_iter: Int
    var metric: Bool
    var norm: Bool
    var eps: Float64

    def __init__(out self, max_iter: Int, metric: Bool, norm: Bool, eps: Float64):
        self.max_iter = max_iter
        self.metric = metric
        self.norm = norm
        self.eps = eps


struct _NmHost(Movable):
    """`_nm_native`'s host buffers and its (m, G)."""
    var keys: List[Float32]
    var ints: List[List[Int32]]
    var floats: List[List[Float32]]
    var m: Int
    var G: Int

    def __init__(out self, Dis: Mat) raises:
        var n = Dis.r
        var cap = max(n * (n - 1) // 2, 1)
        self.keys = List[Float32](length=cap, fill=Float32(0))
        self.ints = List[List[Int32]]()
        for _ in range(3):  # small-loop(cap: three pair-sized int buffers): idx, gid, gst
            self.ints.append(List[Int32](length=cap + 1, fill=Int32(0)))
        var r = mds_setup_host(
            Dis.p(), F32Ptr(unsafe_from_address=Int(self.keys.unsafe_ptr())),
            I32Ptr(unsafe_from_address=Int(self.ints[0].unsafe_ptr())),
            I32Ptr(unsafe_from_address=Int(self.ints[1].unsafe_ptr())),
            I32Ptr(unsafe_from_address=Int(self.ints[2].unsafe_ptr())), n,
        )
        self.m = r[0]
        self.G = r[1]
        var gw = max(self.G, 1)
        for _ in range(5):  # small-loop(gw: five group-sized int buffers): wt, end, prv, last, hf
            self.ints.append(List[Int32](length=gw, fill=Int32(0)))
        self.floats = List[List[Float32]]()
        for _ in range(2):  # small-loop(gw: two group-sized float buffers): sm, gv
            self.floats.append(List[Float32](length=gw, fill=Float32(0)))

    def addrs(self) -> List[Int]:
        """[keys, idx, gid, gst, sm, wt, end, prv, last, hf, gv]."""
        var a = List[Int]()
        a.append(Int(self.keys.unsafe_ptr()))
        for i in range(3):  # small-loop(i: three buffer addresses): idx, gid, gst
            a.append(Int(self.ints[i].unsafe_ptr()))
        a.append(Int(self.floats[0].unsafe_ptr()))
        for i in range(3, 8):  # small-loop(i: five buffer addresses): wt .. hf
            a.append(Int(self.ints[i].unsafe_ptr()))
        a.append(Int(self.floats[1].unsafe_ptr()))
        return a^


def _dist[E: Exec](k: Kit[E], Y: Mat) raises -> Mat:
    """`MDS._dist`: sqrt(sqdist(Y, Y))."""
    return k.ew1(OP_SQRT, k.sqdist(Y, Y), 0.0)


def mds_single[E: Exec](k: Kit[E], Dis: Mat, var Y: Mat, a: MdsArgs, mut stress: Float64) raises -> Tuple[Mat, Int]:
    """`MDS._single`: (Y, it), stress set."""
    var n = Dis.r
    var native = not a.metric
    var nm = _NmHost(Dis) if native else _NmHost(Mat(1, 1))
    var disp = Dis.copy()
    var d = _dist(k, Y)
    var old = 0.0
    var have_old = False
    var it = 0
    var eye = mat_eye(n)
    var floor = mat_const(1e-5, 1, 1)
    stress = 0.0
    for i in range(1, a.max_iter + 1):
        it = i
        if native:
            var P = Mat(n, n)
            mds_disp_host(d.p(), P.p(), nm.addrs(), n, nm.m, nm.G, i == 1)
            var ss = k.word(k.total(k.ew1(OP_SQ, P, 0.0)))
            P = k.ew1(OP_SCALE, P, sqrt((Float64(n * (n - 1)) / 2) / ss))
            disp = k.ew2(OP_ADD, P, mat_t(P))
        var dz = k.ew3(OP_SELECT, d, d, floor, 0.0)
        var ratio = k.ew2(OP_DIV, disp, dz)
        var B = k.ew1(OP_SCALE, ratio, -1.0)
        var rs = k.rowsum(ratio)
        B = k.ew3(OP_FMA, eye, rs, B, 0.0)
        Y = k.ew1(OP_SCALE, k.mm(B, Y, False, False), 1.0 / Float64(n))
        d = _dist(k, Y)
        stress = k.word(k.total(k.ew2(OP_SQDIFF, d, disp))) / 2
        if have_old:
            var ssd = k.word(k.total(k.ew1(OP_SQ, d, 0.0)))
            if (old - stress) / (ssd / 2) < a.eps:
                break
        old = stress
        have_old = True
    if a.norm:
        var ssd = k.word(k.total(k.ew1(OP_SQ, d, 0.0)))
        stress = sqrt(stress / (ssd / 2)) if ssd != 0.0 else 0.0
    return (Y^, it)


def mds_fit[E: Exec](Dis: Mat, starts: Mat, nc: Int, a: MdsArgs, mut best_y: Mat, mut best_s: Float64) raises -> Int:
    """`fit_transform`'s restarts: each start's `_single`, the first with the
    strictly smallest stress kept. Returns its iteration count."""
    var k = Kit[E]()
    var n = Dis.r
    var ns = starts.r // n
    var best_it = 0
    for r in range(ns):  # small-loop(ns: restarts): one SMACOF run per start
        var s = 0.0
        var got = mds_single(k, Dis, mat_rows(starts, r * n, (r + 1) * n), a, s)
        if r == 0 or s < best_s:
            best_s = s
            best_it = got[1]
            best_y = got[0].copy()
    return best_it


def _mds_args(p: PythonObject, f: PythonObject) raises -> MdsArgs:
    """p = [n, nc, n_starts, max_iter, metric, norm]; f = [eps]."""
    return MdsArgs(Int(py=p[3]), Int(py=p[4]) != 0, Int(py=p[5]) != 0, Float64(py=f[0]))


def mds_fit_py[E: Exec](
    dis: PythonObject, starts: PythonObject, y: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """dis (n x n), starts (n_starts * n x nc), y (n x nc, out). Returns
    (stress, n_iter)."""
    var n = _n(p, 0)
    var nc = _n(p, 1)
    var ns = _n(p, 2)
    if n < 1 or ns < 1 or n * n > 2147483647 or ns * n * nc > 2147483647:
        raise Error("x_decomp: mds shape out of range")
    var a = _mds_args(p, f)
    var pd = _f(dis)
    var ps = _f(starts)
    var py_ = _f(y)
    var it = 0
    var st = 0.0
    with GILReleased(Python()):
        var Y = Mat(n, nc)
        it = mds_fit[E](mat_from(pd, n, n), mat_from(ps, ns * n, nc), nc, a, Y, st)
        for i in range(n * nc):
            py_.unsafe_store(i, Y.d[i])
    return Python.tuple(st, it)
