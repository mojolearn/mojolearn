# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/als.mojo: the PythonObject <-> typed-value door
that the host binding registers under the same name, and the `AlsArgs`
unpacker providers/python/x_decomp_als_dev_pyapi.mojo shares. Each door unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/als.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.als import AlsArgs, als_fit_run
from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.exec_trait import Exec


def _als_args(p: PythonObject, f: PythonObject) raises -> AlsArgs:
    """p = [n, m, f, iterations, use_cg, cg_steps, calc_loss]; f = [reg]."""
    return AlsArgs(Int(py=p[3]), Int(py=p[4]) != 0, Int(py=p[5]), Int(py=p[6]) != 0, Float64(py=f[0]))


def als_fit_py[E: Exec](
    c: PythonObject, x: PythonObject, y: PythonObject, ls: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """c (n x m) the confidences; x (n x f) and y (m x f) the starts,
    replaced; ls (iterations float64) the losses when asked. Returns the
    number of losses written."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nf = _n(p, 2)
    var a = _als_args(p, f)
    var pc = _f(c)
    var px = _f(x)
    var py_ = _f(y)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=ls))
    var cnt = 0
    with GILReleased(Python()):
        cnt = als_fit_run[E](pc, px, py_, pl, n, m, nf, a)
    return PythonObject(cnt)
