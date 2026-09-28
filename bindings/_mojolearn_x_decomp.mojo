# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S GPU BINDING (lane/algos-decomp, 2026-09-27): the entry
points of x_decomp/api.mojo on the device executor (x_decomp/device.mojo).
bindings/_mojolearn_x_decomp_host.mojo registers the same names on the host
executor."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from x_decomp.api import (
    cd_rows_py, chol_py, colsum_py, eigh_py, ew_py, gemm_py, lu_py, lu_solve_py, numeric_mode_py, orth_py, rand_py, svd_py, lasso_rows_py, omp_rows_py, rand_gamma_py, lda_rows_py, dijkstra_rows_py, barycenter_rows_py, als_rows_py, absmax_sign_py, qr_r_py,
    geqrf_py, orgqr_py, als_cg_rows_py,
    rowsum_py, sqdist_py, vendor_py,
)
from x_decomp.device import DevExec


@export
def PyInit__mojolearn_x_decomp() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_decomp")
        m.def_function[gemm_py[DevExec]]("x_decomp_gemm")
        m.def_function[ew_py[DevExec]]("x_decomp_ew")
        m.def_function[colsum_py[DevExec]]("x_decomp_colsum")
        m.def_function[rowsum_py[DevExec]]("x_decomp_rowsum")
        m.def_function[sqdist_py[DevExec]]("x_decomp_sqdist")
        m.def_function[rand_py[DevExec]]("x_decomp_rand")
        m.def_function[lu_py[DevExec]]("x_decomp_lu")
        m.def_function[lu_solve_py[DevExec]]("x_decomp_lu_solve")
        m.def_function[chol_py[DevExec]]("x_decomp_chol")
        m.def_function[eigh_py[DevExec]]("x_decomp_eigh")
        m.def_function[cd_rows_py[DevExec]]("x_decomp_cd_rows")
        m.def_function[orth_py[DevExec]]("x_decomp_orth")
        m.def_function[svd_py[DevExec]]("x_decomp_svd")
        m.def_function[lasso_rows_py[DevExec]]("x_decomp_lasso_rows")
        m.def_function[omp_rows_py[DevExec]]("x_decomp_omp_rows")
        m.def_function[rand_gamma_py[DevExec]]("x_decomp_rand_gamma")
        m.def_function[lda_rows_py[DevExec]]("x_decomp_lda_rows")
        m.def_function[dijkstra_rows_py[DevExec]]("x_decomp_dijkstra_rows")
        m.def_function[barycenter_rows_py[DevExec]]("x_decomp_barycenter_rows")
        m.def_function[als_rows_py[DevExec]]("x_decomp_als_rows")
        m.def_function[absmax_sign_py[DevExec]]("x_decomp_absmax_sign")
        m.def_function[qr_r_py[DevExec]]("x_decomp_qr_r")
        m.def_function[geqrf_py[DevExec]]("x_decomp_geqrf")
        m.def_function[orgqr_py[DevExec]]("x_decomp_orgqr")
        m.def_function[als_cg_rows_py[DevExec]]("x_decomp_als_cg_rows")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[DevExec]]("x_decomp_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp: ", e))
