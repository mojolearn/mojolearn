# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/chi2.mojo: the chi-squared and LDA bound
doors the host binding registers. Each unpacks floats, addresses and
integers and calls one typed entry in x_decomp/chi2.mojo; no data work
happens here."""
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.chi2 import chi2_cdf, chi2_quantile, consistency_factor_run, lda_bound_host_run


def _fp(a: PythonObject) raises -> F32Ptr:
    var v = Int(py=a)
    if v == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=v)


def chi2_cdf_py(dof: PythonObject, m: PythonObject) raises -> PythonObject:
    return PythonObject(chi2_cdf(Float64(py=dof), Float64(py=m)))


def consistency_factor_py(p: PythonObject, num: PythonObject, den: PythonObject) raises -> PythonObject:
    """`consistency_factor(p, num / den)`: MinCovDet passes its support count
    over the row count, or 0.975 over 1."""
    var dn = Float64(py=den)
    var pp = Float64(py=p)
    var nm = Float64(py=num)
    return PythonObject(consistency_factor_run(pp, nm, dn))


def chi2_quantile_py(dof: PythonObject, upper: PythonObject) raises -> PythonObject:
    return PythonObject(chi2_quantile(Float64(py=dof), Float64(py=upper)))


def lda_bound_host_py(
    x: PythonObject, ddt: PythonObject, dcomp: PythonObject, dst: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """LatentDirichletAllocation's `_approx_bound` term matrix on the host
    column (see x_decomp/chi2.mojo `lda_bound_host`). p = [n, k, v];
    f = [floor]. The caller folds dst with the kit's total."""
    var n = Int(py=p[0])
    var kk = Int(py=p[1])
    var vv = Int(py=p[2])
    var fl = Float64(py=f[0])
    var px = _fp(x)
    var pd = _fp(ddt)
    var pc = _fp(dcomp)
    var po = _fp(dst)
    return PythonObject(lda_bound_host_run(px, pd, pc, po, n, kk, vv, fl))
