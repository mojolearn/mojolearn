# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mds.mojo: the PythonObject <-> typed-value
door that `bindings/_mojolearn_x_decomp_host.mojo` registers under the same
name. It unpacks addresses and integers, releases the GIL and calls one
typed entry in x_decomp/mds.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.mds import MdsArgs, mds_fit_check, mds_fit_run


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
    mds_fit_check(n, nc, ns)
    var a = _mds_args(p, f)
    var pd = _f(dis)
    var ps = _f(starts)
    var py_ = _f(y)
    var it = 0
    var st = 0.0
    with GILReleased(Python()):
        it = mds_fit_run[E](pd, ps, py_, n, nc, ns, a, st)
    return Python.tuple(st, it)
