# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lda_fit_dev.mojo: the PythonObject <-> typed-value
door that the GPU binding registers under the same name. It unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/lda_fit_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api_pyapi import _f, _n
from x_decomp.lda_fit_dev import lda_fit_dev_run
from x_decomp.lda_fit_pyapi import _lda_args


def lda_fit_dev_py(
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`lda_fit_py` on the resident kit."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var draw = Int(py=p[8])
    var nbi = Int(py=p[9])
    var a = _lda_args(p, f)
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    var it = 0
    with GILReleased(Python()):
        it = lda_fit_dev_run(px, pc, pe, n, v, nc, a, draw, nbi)
    return Python.tuple(it, draw, nbi)
