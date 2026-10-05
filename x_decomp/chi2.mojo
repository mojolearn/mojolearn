# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The chi-squared CDF and quantile of MinCovDet in Mojo (lane py-runtime-b,
2026-10-05): `_expansion_decomp._chi2_cdf` and `_chi2_quantile`, parameter
math only (dof and a probability, no data). The same steps as the Python:
the prefactor x^a e^-x / Gamma(a + 1) through the cells' logs, lgamma and exp
on float32 words (`ew_cell`, the cell every executor applies), the series
in float64, the 80 bisections in float64 and the result rounded once to
float32. So the same words on every column."""
from std.python import PythonObject
from x_decomp.cells import ew_cell

comptime _OP_EXP = 9
comptime _OP_LOGS = 10
comptime _OP_LGAMMA = 37


def chi2_cdf(dof: Float64, m: Float64) -> Float64:
    """P(chi2_dof <= m): the regularized lower incomplete gamma P(dof/2, m/2)."""
    var a = dof / 2.0
    var x = m / 2.0
    if x <= 0:
        return 0.0
    var z = Float32(0)
    var lx = Float64(ew_cell(_OP_LOGS, Float32(x), z, z, Float32(1e-30)))
    var lg = Float64(ew_cell(_OP_LGAMMA, Float32(a + 1), z, z, z))
    var pref = Float64(ew_cell(_OP_EXP, Float32(a * lx - x - lg), z, z, z))
    var term = 1.0
    var tot = 1.0
    for n in range(1, 4000):
        term *= x / (a + Float64(n))
        tot += term
        if term < 1e-17 * tot:
            break
    return pref * tot


def chi2_quantile(dof: Float64, upper: Float64) -> Float64:
    """The point m with P(chi2_dof > m) = upper, by 80 bisections, rounded
    once to float32."""
    var target = 1.0 - upper
    var lo = 0.0
    var hi = 4.0 * dof + 40.0
    if not (hi > 1.0):
        hi = 1.0
    for _ in range(80):
        var mid = 0.5 * (lo + hi)
        if chi2_cdf(dof, mid) < target:
            lo = mid
        else:
            hi = mid
    return Float64(Float32(0.5 * (lo + hi)))


def chi2_cdf_py(dof: PythonObject, m: PythonObject) raises -> PythonObject:
    return PythonObject(chi2_cdf(Float64(py=dof), Float64(py=m)))


def chi2_quantile_py(dof: PythonObject, upper: PythonObject) raises -> PythonObject:
    return PythonObject(chi2_quantile(Float64(py=dof), Float64(py=upper)))
