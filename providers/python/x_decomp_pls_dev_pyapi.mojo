# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the PLS component loop on resident device matrices
(x_decomp/pls_dev.mojo): the door `bindings/_mojolearn_x_decomp.mojo`
registers. It unpacks addresses and integers, releases the GIL and calls one
typed entry in x_decomp/pls_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.pls_dev import pls_fit_dev_run
from providers.python.x_decomp_pls_pyapi import _pls_args, _pls_dst


def pls_fit_dev_py(
    x: PythonObject, y: PythonObject, outs: PythonObject, its: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`pls_fit_py` on the resident kit."""
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
        done = pls_fit_dev_run(pxa, pya, dst, pit, n, px, q, a, nits)
    return Python.tuple(done, nits)
