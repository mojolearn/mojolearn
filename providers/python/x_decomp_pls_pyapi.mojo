# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the PLS component loop (x_decomp/pls.mojo): the door
`bindings/_mojolearn_x_decomp_host.mojo` registers (`pls_fit_py[HostExec]`)
and the argument unpacker the device door shares. Each door unpacks
addresses and integers, releases the GIL and calls one typed entry in
x_decomp/pls.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.pls import PlsArgs, pls_fit_run


def _pls_args(p: PythonObject, f: PythonObject) raises -> PlsArgs:
    """p = [n, px, q, nc, max_iter, mode_b, norm_y, canonical, svd_algo]; f = [tol]."""
    return PlsArgs(_n(p, 3), Int(py=p[4]), Float64(py=f[0]), Int(py=p[5]) != 0, Int(py=p[6]) != 0,
                   Int(py=p[7]) != 0, Int(py=p[8]) != 0)


def _pls_dst(outs: PythonObject) raises -> List[Int]:
    """The six output addresses of `outs`."""
    var dst = List[Int]()
    dst.append(Int(py=outs[0]))
    dst.append(Int(py=outs[1]))
    dst.append(Int(py=outs[2]))
    dst.append(Int(py=outs[3]))
    dst.append(Int(py=outs[4]))
    dst.append(Int(py=outs[5]))
    return dst^


def pls_fit_py[E: Exec](
    x: PythonObject, y: PythonObject, outs: PythonObject, its: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """x (n x px) and y (n x q) the centred, scaled data; outs = six
    addresses of [xw px x nc, yw q x nc, xs n x nc, ys n x nc, xl px x nc,
    yl q x nc] (column c written for each fitted component); its (nc int32)
    the power iteration counts. Returns (fitted, iteration count entries)."""
    var n = _n(p, 0)
    var px = _n(p, 1)
    var q = _n(p, 2)
    var a = _pls_args(p, f)
    var pxa = _f(x)
    var pya = _f(y)
    var dst = _pls_dst(outs)
    var pit = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=its))
    var done = 0
    var nits = 0
    with GILReleased(Python()):
        done = pls_fit_run[E](pxa, pya, dst, pit, n, px, q, a, nits)
    return Python.tuple(done, nits)
