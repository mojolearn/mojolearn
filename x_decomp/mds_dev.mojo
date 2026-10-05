# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/mds.mojo's SMACOF on resident device matrices (lane
py-runtime-b): the same statements on `DKit`, non-metric disparities by
x_decomp/mds_iso_dev.mojo's setup and isotonic kernels (`_nm_native`'s
resident form), each stress word home where Python read it. GPU binding only."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.kit import (
    Mat, OP_ADD, OP_DIV, OP_FMA, OP_SCALE, OP_SELECT, OP_SQ, OP_SQDIFF, OP_SQRT, mat_const, mat_eye, mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.mds import MdsArgs, _mds_args
from x_decomp.mds_iso_dev import mds_disp_dev, mds_setup_dev


struct _NmDev(Movable):
    var bufs: List[DMat]
    var m: Int
    var G: Int

    def __init__(out self, Dis: DMat, native: Bool) raises:
        self.bufs = List[DMat]()
        self.m = 0
        self.G = 0
        if not native:
            return
        var n = Dis.r
        var N2 = 1
        for _ in range(62):  # small-loop(N2: doublings to the pair count's power of two): Python's 1 << bit_length
            if N2 >= n * n:
                break
            N2 *= 2
        # keys, idx, gid, tmp (N2), gst (N2 + 1), word (1)
        var setup = List[DMat]()
        for _ in range(4):  # small-loop(N2: four pair-sized buffers): keys, idx, gid, tmp
            setup.append(DMat(1, N2))
        setup.append(DMat(1, N2 + 1))
        setup.append(DMat(1, 1))
        var ids = List[Int]()
        for i in range(6):  # small-loop(i: six buffer ids): the setup's buffers
            ids.append(setup[i].id)
        var r = mds_setup_dev(Dis.id, ids, n, N2)
        self.m = r[0]
        self.G = r[1]
        # keep keys, idx, gid, gst; then sm, wt, end, prv, last, hf, hd0, hd1, gv
        self.bufs.append(setup.pop(0))
        self.bufs.append(setup.pop(0))
        self.bufs.append(setup.pop(0))
        _ = setup.pop(0)
        self.bufs.append(setup.pop(0))
        for _ in range(9):  # small-loop(G: nine group-sized buffers): the isotonic step's scratch
            self.bufs.append(DMat(1, max(self.G, 1)))

    def ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.bufs)):  # small-loop(i: thirteen buffer ids): the disparity step's buffers
            out.append(self.bufs[i].id)
        return out^


def _dist_dev(k: DKit, Y: DMat) raises -> DMat:
    return k.ew1(OP_SQRT, k.sqdist(Y, Y), 0.0)


def mds_single_dev(mut k: DKit, Dis: DMat, var Y: DMat, a: MdsArgs, mut stress: Float64, mut it: Int) raises -> DMat:
    var n = Dis.r
    var native = not a.metric
    var nm = _NmDev(Dis, native)
    var disp = k.copy(Dis)
    var d = _dist_dev(k, Y)
    var old = 0.0
    var have_old = False
    it = 0
    var eye = k.upload(mat_eye(n))
    var floor = k.upload(mat_const(1e-5, 1, 1))
    stress = 0.0
    for i in range(1, a.max_iter + 1):
        it = i
        if native:
            var P = DMat(n, n)
            mds_disp_dev(P.id if i == 1 else d.id, P.id, nm.ids(), n, nm.m, nm.G, i == 1)
            var ss = k.word(k.total(k.ew1(OP_SQ, P, 0.0)))
            P = k.ew1(OP_SCALE, P, sqrt((Float64(n * (n - 1)) / 2) / ss))
            disp = k.ew2(OP_ADD, P, k.t(P))
        var dz = k.ew3(OP_SELECT, d, d, floor, 0.0)
        var ratio = k.ew2(OP_DIV, disp, dz)
        var B = k.ew1(OP_SCALE, ratio, -1.0)
        var rs = k.rowsum(ratio)
        B = k.ew3(OP_FMA, eye, rs, B, 0.0)
        Y = k.ew1(OP_SCALE, k.mm(B, Y, False, False), 1.0 / Float64(n))
        d = _dist_dev(k, Y)
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
    return Y^


def mds_fit_dev_py(
    dis: PythonObject, starts: PythonObject, y: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`mds_fit_py` on the resident kit."""
    var n = _n(p, 0)
    var nc = _n(p, 1)
    var ns = _n(p, 2)
    if n < 1 or ns < 1 or n * n > 2147483647 or ns * n * nc > 2147483647:
        raise Error("x_decomp: mds shape out of range")
    var a = _mds_args(p, f)
    var pd = _f(dis)
    var ps = _f(starts)
    var py_ = _f(y)
    var best_it = 0
    var best_s = 0.0
    with GILReleased(Python()):
        var k = DKit()
        var D = k.upload(mat_from(pd, n, n))
        var S = k.upload(mat_from(ps, ns * n, nc))
        var best = DMat(n, nc)
        for r in range(ns):  # small-loop(ns: restarts): one SMACOF run per start
            var s = 0.0
            var it = 0
            var Y0 = k.rows(S, r * n, (r + 1) * n)
            var Y = mds_single_dev(k, D, Y0^, a, s, it)
            if r == 0 or s < best_s:
                best_s = s
                best_it = it
                best = Y^
        var h = k.get(best)
        k.sync()
        for i in range(n * nc):
            py_.unsafe_store(i, h.d[i])
    return Python.tuple(best_s, best_it)
