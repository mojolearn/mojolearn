# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/chi2.mojo's chi-squared CDF and quantile with the prefactor's
cells launched on the device (lane py-runtime-b), as Python's resident kit
launched them: logs, lgamma and exp of 1 x 1 float32 words through `DKit`
(the same `ew_cell` per element), each read home; the series and the
bisection in float64 as on the host column. GPU binding only."""
from std.python import PythonObject
from std.python._cpython import GILReleased
from std.python import Python
from x_decomp.chi2 import CHI2_BISECT, CHI2_TERMS
from x_decomp.kit import OP_EXP, OP_LOGS, mat_const
from x_decomp.kit_device import DKit

comptime _OP_LGAMMA = 37


def _cell1(mut k: DKit, op: Int, v: Float64, s: Float64) raises -> Float64:
    """`k.ew(op, k.const(v), s=s).s[0]`: one cell of one float32 word."""
    var a = k.upload(mat_const(v, 1, 1))
    var r = k.ew1(op, a, s)
    return k.word(r)


def chi2_cdf_dev(mut k: DKit, dof: Float64, m: Float64) raises -> Float64:
    var a = dof / 2.0
    var x = m / 2.0
    if x <= 0:
        return 0.0
    var lx = _cell1(k, OP_LOGS, x, 1e-30)
    var lg = _cell1(k, _OP_LGAMMA, a + 1, 0.0)
    var pref = _cell1(k, OP_EXP, Float64(Float32(a * lx - x - lg)), 0.0)
    var term = 1.0
    var tot = 1.0
    for n in range(1, CHI2_TERMS):  # small-loop(CHI2_TERMS: series terms): float64 scalar series, no data
        term *= x / (a + Float64(n))
        tot += term
        if term < 1e-17 * tot:
            break
    return pref * tot


def chi2_quantile_dev(mut k: DKit, dof: Float64, upper: Float64) raises -> Float64:
    var target = 1.0 - upper
    var lo = 0.0
    var hi = 4.0 * dof + 40.0
    if not (hi > 1.0):
        hi = 1.0
    for _ in range(CHI2_BISECT):  # small-loop(CHI2_BISECT: bisection steps): float64 scalar search, no data
        var mid = 0.5 * (lo + hi)
        if chi2_cdf_dev(k, dof, mid) < target:
            lo = mid
        else:
            hi = mid
    return Float64(Float32(0.5 * (lo + hi)))


def chi2_cdf_dev_py(dof: PythonObject, m: PythonObject) raises -> PythonObject:
    var d = Float64(py=dof)
    var x = Float64(py=m)
    var r = 0.0
    with GILReleased(Python()):
        var k = DKit()
        r = chi2_cdf_dev(k, d, x)
    return PythonObject(r)


def chi2_quantile_dev_py(dof: PythonObject, upper: PythonObject) raises -> PythonObject:
    var d = Float64(py=dof)
    var u = Float64(py=upper)
    var r = 0.0
    with GILReleased(Python()):
        var k = DKit()
        r = chi2_quantile_dev(k, d, u)
    return PythonObject(r)
