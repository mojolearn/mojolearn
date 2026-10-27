# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/chi2_dev.mojo: the chi-squared doors the GPU
binding registers. Each unpacks floats, releases the GIL and calls one typed
`*_run` entry in x_decomp/chi2_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.chi2_dev import chi2_quantile_dev_run, consistency_factor_dev_run


def consistency_factor_dev_py(p: PythonObject, num: PythonObject, den: PythonObject) raises -> PythonObject:
    var pp = Float64(py=p)
    var dn = Float64(py=den)
    var nm = Float64(py=num)
    var r = 0.0
    with GILReleased(Python()):
        r = consistency_factor_dev_run(pp, nm, dn)
    return PythonObject(r)


def chi2_quantile_dev_py(dof: PythonObject, upper: PythonObject) raises -> PythonObject:
    var d = Float64(py=dof)
    var u = Float64(py=upper)
    var r = 0.0
    with GILReleased(Python()):
        r = chi2_quantile_dev_run(d, u)
    return PythonObject(r)
