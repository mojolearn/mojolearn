# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lda_fast.mojo: the PythonObject <-> typed-value
door that the GPU binding registers under the same name. It unpacks resident
matrix ids, integers and scalars and calls one typed entry in
x_decomp/lda_fast.mojo; no data work happens here."""
from std.python import PythonObject
from x_decomp.lda_fast import dev_lda_estep_ss_run
from x_decomp.resident_pyapi import _id, _n


def dev_lda_estep_ss_py(
    x: PythonObject, ew: PythonObject, d: PythonObject, e: PythonObject, ss: PythonObject, p: PythonObject,
    f: PythonObject,
) raises -> PythonObject:
    """`x_decomp_dev_lda_estep_ss` on device matrices: d and e (n x k)
    updated in place as `x_decomp_dev_lda_rows` does, and ss (k x v) = the
    E-step's sufficient statistics. p = [n, k, v, max_iter], f = [prior,
    tol]. Returns 1, or 0 when the shape is past the kernel's caps (the
    caller then runs main's chain)."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var max_iter = _n(p, 3)
    var prior = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var xi = _id(x)
    var wi = _id(ew)
    var di = _id(d)
    var ei = _id(e)
    var si = _id(ss)
    var ok = dev_lda_estep_ss_run(xi, wi, di, ei, si, n, k, v, max_iter, prior, tol)
    return PythonObject(ok)
