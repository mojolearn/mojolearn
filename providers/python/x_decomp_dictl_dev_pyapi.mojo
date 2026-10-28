# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the dictionary learning loops on resident device
matrices (x_decomp/dictl_dev.mojo): the doors `bindings/_mojolearn_x_decomp.mojo`
registers. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/dictl_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.dictl_dev import dict_learning_dev_run, minibatch_dev_run
from providers.python.x_decomp_dictl_pyapi import _dl_args


def dict_learning_dev_py(
    x: PythonObject, code: PythonObject, d: PythonObject, errs: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`dict_learning_py` on the resident kit."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    var a = _dl_args(p, f)
    var px = _f(x)
    var pc = _f(code)
    var pd = _f(d)
    var pe = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=errs))
    var ii = 0
    var ne = 0
    with GILReleased(Python()):
        ii = dict_learning_dev_run(px, pc, pd, pe, n, m, nc, a, ne)
    return Python.tuple(ii, ne)


def minibatch_dev_py(x: PythonObject, d: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`minibatch_py` on the resident kit."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 11)
    var n_steps = _n(p, 12)
    var a = _dl_args(p, f)
    var px = _f(x)
    var pd = _f(d)
    var steps = 0
    with GILReleased(Python()):
        steps = minibatch_dev_run(px, pd, n, m, nc, bs, n_steps, a)
    return PythonObject(steps)
