# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/rotation.mojo: the PythonObject <-> typed-value
door that the decomp bindings register under the same name. It unpacks
addresses, integers and scalars, releases the GIL and calls one typed entry in
x_decomp/rotation.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.rotation import ortho_rotation_run


def ortho_rotation_py[E: Exec](c: PythonObject, out_t: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """c (nrow x ncol); out_t (ncol x nrow) = (C R)^T. p = [nrow, ncol,
    varimax, max_iter]; f = [tol]."""
    var nrow = _n(p, 0)
    var ncol = _n(p, 1)
    var vm = Int(py=p[2]) != 0
    var max_iter = Int(py=p[3])
    var tol = Float64(py=f[0])
    var pc = _f(c)
    var po = _f(out_t)
    with GILReleased(Python()):
        ortho_rotation_run[E](pc, po, nrow, ncol, vm, max_iter, tol)
    return PythonObject(ncol)
