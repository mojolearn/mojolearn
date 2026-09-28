# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_decomp` (lane/algos-decomp, 2026-09-27).
HOST ONLY: the entry points of x_decomp/api.mojo on the host executor
(x_decomp/host.mojo), under the GPU binding's names and address contract,
plus the family read-backs."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.api import (
    cd_rows_py, chol_py, colsum_py, eigh_py, ew_py, gemm_py, lu_py, lu_solve_py, numeric_mode_py, orth_py, rand_py, svd_py, lasso_rows_py, omp_rows_py, rand_gamma_py, lda_rows_py, dijkstra_rows_py, barycenter_rows_py, als_rows_py, absmax_sign_py, qr_r_py,
    geqrf_py, orgqr_py, als_cg_rows_py, mcd_py, lda_online_py, gather_py, scatter_py, triu_nonzero_py, argsort_f32_py, iso_order_py,
    rowsum_py, sqdist_py, vendor_py,
)
from x_decomp.host import HostExec, X_DECOMP_HOST_SABOTAGE


def x_decomp_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_decomp_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_decomp_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_decomp host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_decomp_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_DECOMP_HOST_SABOTAGE)


@export
def PyInit__mojolearn_x_decomp_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_decomp_host")
        m.def_function[x_decomp_host_numeric_mode_binding]("x_decomp_host_numeric_mode")
        m.def_function[x_decomp_host_vendor_binding]("x_decomp_host_vendor")
        m.def_function[x_decomp_host_column_binding]("x_decomp_host_column")
        m.def_function[x_decomp_host_sabotage_binding]("x_decomp_host_sabotage")
        m.def_function[gemm_py[HostExec]]("x_decomp_gemm")
        m.def_function[ew_py[HostExec]]("x_decomp_ew")
        m.def_function[colsum_py[HostExec]]("x_decomp_colsum")
        m.def_function[rowsum_py[HostExec]]("x_decomp_rowsum")
        m.def_function[sqdist_py[HostExec]]("x_decomp_sqdist")
        m.def_function[rand_py[HostExec]]("x_decomp_rand")
        m.def_function[lu_py[HostExec]]("x_decomp_lu")
        m.def_function[lu_solve_py[HostExec]]("x_decomp_lu_solve")
        m.def_function[chol_py[HostExec]]("x_decomp_chol")
        m.def_function[eigh_py[HostExec]]("x_decomp_eigh")
        m.def_function[cd_rows_py[HostExec]]("x_decomp_cd_rows")
        m.def_function[orth_py[HostExec]]("x_decomp_orth")
        m.def_function[svd_py[HostExec]]("x_decomp_svd")
        m.def_function[lasso_rows_py[HostExec]]("x_decomp_lasso_rows")
        m.def_function[omp_rows_py[HostExec]]("x_decomp_omp_rows")
        m.def_function[rand_gamma_py[HostExec]]("x_decomp_rand_gamma")
        m.def_function[lda_rows_py[HostExec]]("x_decomp_lda_rows")
        m.def_function[dijkstra_rows_py[HostExec]]("x_decomp_dijkstra_rows")
        m.def_function[barycenter_rows_py[HostExec]]("x_decomp_barycenter_rows")
        m.def_function[als_rows_py[HostExec]]("x_decomp_als_rows")
        m.def_function[absmax_sign_py[HostExec]]("x_decomp_absmax_sign")
        m.def_function[qr_r_py[HostExec]]("x_decomp_qr_r")
        m.def_function[geqrf_py[HostExec]]("x_decomp_geqrf")
        m.def_function[orgqr_py[HostExec]]("x_decomp_orgqr")
        m.def_function[als_cg_rows_py[HostExec]]("x_decomp_als_cg_rows")
        m.def_function[mcd_py[HostExec, HostExec]]("x_decomp_mcd")
        m.def_function[lda_online_py[HostExec, HostExec]]("x_decomp_lda_online")
        m.def_function[gather_py]("x_decomp_gather")
        m.def_function[scatter_py]("x_decomp_scatter")
        m.def_function[triu_nonzero_py]("x_decomp_triu_nonzero")
        m.def_function[argsort_f32_py]("x_decomp_argsort_f32")
        m.def_function[iso_order_py]("x_decomp_iso_order")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[HostExec]]("x_decomp_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp_host: ", e))
