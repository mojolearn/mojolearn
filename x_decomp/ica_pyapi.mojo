# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/ica.mojo: the PythonObject <-> typed-value door
that the host binding registers under the same name. It unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/ica.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.ica import ica_solve_run


def ica_solve_py[E: Exec](x1: PythonObject, w: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`FastICA._par` / `_def` over X1 (xr x xc): w (nc x nc, the initial
    unmixing matrix) replaced in place. p = [nc, xr, xc, algorithm (0
    parallel, 1 deflation), fun (0 logcosh, 1 exp, 2 cube), max_iter];
    f = [alpha, tol]. Returns the iteration count."""
    var nc = _n(p, 0)
    var xr = _n(p, 1)
    var xc = _n(p, 2)
    var algo = Int(py=p[3])
    var fun = Int(py=p[4])
    var max_iter = Int(py=p[5])
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var px = _f(x1)
    var pw = _f(w)
    var it = 0
    with GILReleased(Python()):
        it = ica_solve_run[E](px, pw, nc, xr, xc, algo, fun, max_iter, alpha, tol)
    return PythonObject(it)
