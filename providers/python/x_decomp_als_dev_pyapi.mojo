# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/als_dev.mojo: the PythonObject <-> typed-value
door that the GPU binding registers under the same name. It unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/als_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.als_dev import als_fit_dev_run
from providers.python.x_decomp_als_pyapi import _als_args
from providers.python.x_decomp_api_pyapi import _f, _n


def als_fit_dev_py(
    c: PythonObject, x: PythonObject, y: PythonObject, ls: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`als_fit_py` on the resident kit."""
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
        cnt = als_fit_dev_run(pc, px, py_, pl, n, m, nf, a)
    return PythonObject(cnt)
