# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/rotation_dev.mojo: the PythonObject <-> typed-value
door that the GPU binding registers under the same name. It unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/rotation_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api_pyapi import _f, _n
from x_decomp.rotation_dev import ortho_rotation_dev_run


def ortho_rotation_dev_py(c: PythonObject, out_t: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`ortho_rotation_py` on the resident kit."""
    var nrow = _n(p, 0)
    var ncol = _n(p, 1)
    var vm = Int(py=p[2]) != 0
    var max_iter = Int(py=p[3])
    var tol = Float64(py=f[0])
    var pc = _f(c)
    var po = _f(out_t)
    with GILReleased(Python()):
        ortho_rotation_dev_run(pc, po, nrow, ncol, vm, max_iter, tol)
    return PythonObject(ncol)
