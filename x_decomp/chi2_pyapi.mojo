# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/chi2.mojo: the chi-squared and LDA bound
doors the host binding registers. Each unpacks floats, addresses and
integers and calls one typed entry in x_decomp/chi2.mojo; no data work
happens here."""
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.chi2 import chi2_cdf, chi2_quantile, consistency_factor_run, lda_bound_host_run


# TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_decomp.patch


def consistency_factor_py(p: PythonObject, num: PythonObject, den: PythonObject) raises -> PythonObject:
    """`consistency_factor(p, num / den)`: MinCovDet passes its support count
    over the row count, or 0.975 over 1."""
    var dn = Float64(py=den)
    var pp = Float64(py=p)
    var nm = Float64(py=num)
    return PythonObject(consistency_factor_run(pp, nm, dn))


def chi2_quantile_py(dof: PythonObject, upper: PythonObject) raises -> PythonObject:
    return PythonObject(chi2_quantile(Float64(py=dof), Float64(py=upper)))

