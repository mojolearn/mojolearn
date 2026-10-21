# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the NMF solver loops on resident device matrices
(x_decomp/nmf_dev.mojo): the doors `bindings/_mojolearn_x_decomp.mojo`
registers. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/nmf_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _n
from x_decomp.nmf_dev import nmf_nndsvd_dev_run, nmf_solve_dev_run
from x_decomp.nmf_pyapi import _nmf_args


def nmf_solve_dev_py(
    x: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`nmf_solve_py` on the resident kit."""
    var n = _n(p, 0)
    var d = _n(p, 1)
    var nc = _n(p, 2)
    var solver = Int(py=p[3])
    var a = _nmf_args(p, f)
    var px = _f(x)
    var pw = _f(w)
    var ph = _f(h)
    var it = 0
    with GILReleased(Python()):
        it = nmf_solve_dev_run(px, pw, ph, n, d, nc, solver, a)
    return PythonObject(it)


def nmf_nndsvd_dev_py(
    u: PythonObject, s: PythonObject, vt: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject
) raises -> PythonObject:
    """`nmf_nndsvd_py` on the resident kit."""
    var n = _n(p, 0)
    var d = _n(p, 1)
    var r = _n(p, 2)
    var nc = _n(p, 3)
    var pu = _f(u)
    var ps = _f(s)
    var pv = _f(vt)
    var pw = _f(w)
    var ph = _f(h)
    var res = 0
    with GILReleased(Python()):
        res = nmf_nndsvd_dev_run(pu, ps, pv, pw, ph, n, d, r, nc)
    return PythonObject(res)
