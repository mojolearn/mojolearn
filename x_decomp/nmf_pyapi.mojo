# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the NMF solver loops (x_decomp/nmf.mojo): the doors
`bindings/_mojolearn_x_decomp_host.mojo` registers (`*_py[HostExec]`) and the
argument unpacker the device doors share. Each door unpacks addresses and
integers, releases the GIL and calls one typed entry in x_decomp/nmf.mojo;
no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.nmf import NmfArgs, nmf_args_checked, nmf_nndsvd_run, nmf_solve_run


def _nmf_args(p: PythonObject, f: PythonObject) raises -> NmfArgs:
    """p = [n, d, nc, solver (0 cd, 1 mu), update_H, max_iter, shuffle,
    seed & 0xFFFFFFFF]; f = [beta, tol, l1W, l1H, l2W, l2H]."""
    var beta = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var l1w = Float64(py=f[2])
    var l1h = Float64(py=f[3])
    var l2w = Float64(py=f[4])
    var l2h = Float64(py=f[5])
    var update_h = Int(py=p[4]) != 0
    var max_iter = Int(py=p[5])
    var shuffle = Int(py=p[6]) != 0
    var seed = Int(py=p[7])
    return nmf_args_checked(beta, update_h, max_iter, tol, l1w, l1h, l2w, l2h, shuffle, seed)


def nmf_solve_py[E: Exec](
    x: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`NMF._cd` / `_mu` over x (n x d): w (n x nc) and h (nc x d) replaced
    in place (h only when update_H). Returns the iteration count, or
    NMF_ZERO_ERR0 (the caller raises ZeroDivisionError)."""
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
        it = nmf_solve_run[E](px, pw, ph, n, d, nc, solver, a)
    return PythonObject(it)


def nmf_nndsvd_py[E: Exec](
    u: PythonObject, s: PythonObject, vt: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject
) raises -> PythonObject:
    """`nmf_nndsvd` over u (n x r), s (r), vt (r x d); w (n x nc) and h
    (nc x d) written. p = [n, d, r, nc]."""
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
        res = nmf_nndsvd_run[E](pu, ps, pv, pw, ph, n, d, r, nc)
    return PythonObject(res)
