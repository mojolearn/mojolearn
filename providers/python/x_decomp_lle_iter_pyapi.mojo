# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of LocallyLinearEmbedding's subspace iteration
(x_decomp/lle_iter.mojo): the door `bindings/_mojolearn_x_decomp_host.mojo`
registers (`lle_iterate_py[HostExec]`) and the argument unpacker the device
door shares. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/lle_iter.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.lle_iter import LleArgs, lle_iterate_run


def _lle_args(p: PythonObject, f: PythonObject) raises -> LleArgs:
    """p = [n, n1, nc, p, max_iter, seed]; f = [floor, sub_tol, stall_tol]."""
    return LleArgs(_n(p, 0), _n(p, 1), _n(p, 2), _n(p, 3), Int(py=p[4]), Int(py=p[5]), Float64(py=f[0]),
                   Float64(py=f[1]), Float64(py=f[2]))


def lle_iterate_py[E: Exec](
    f0: PythonObject, lu: PythonObject, pm: PythonObject, im: PythonObject, z: PythonObject, outs: PythonObject,
    p: PythonObject, f: PythonObject,
) raises -> PythonObject:
    """out = [X (n1 x p), Y (n1 x nc), S (p)] addresses. Returns (status, e)."""
    var a = _lle_args(p, f)
    var pf = _f(f0)
    var pl = _f(lu)
    var ppm = _f(pm)
    var pim = _f(im)
    var pz = _f(z)
    var px = _f(outs[0])
    var py_ = _f(outs[1])
    var ps = _f(outs[2])
    var st = 0
    var e = 0.0
    with GILReleased(Python()):
        st = lle_iterate_run[E](pf, pl, ppm, pim, pz, px, py_, ps, a, e)
    return Python.tuple(st, e)
