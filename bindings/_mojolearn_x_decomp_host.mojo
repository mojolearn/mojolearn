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
    cd_rows_py, chol_py, colsum_py, eigh_py, eigh_batch_py, lle_local_py, lle_apply_py, ew_py, gemm_py, lu_py, lu_solve_py, trisolve_py, knn_select_py, numeric_mode_py, orth_py, orth_diag_py, rand_py, svd_py, lasso_rows_py, lars_rows_py, lu_aux_py, omp_rows_py, rand_gamma_py, lda_rows_py, dijkstra_rows_py, barycenter_rows_py, als_rows_py, absmax_sign_py, qr_r_py,
    geqrf_py, orgqr_py, tsqr_r_py, tsqr_q_py, r_signs_py, rank_above_py, als_cg_rows_py, mcd_py, lda_online_py, gather_py, scatter_py, triu_nonzero_py, argsort_f32_py, iso_order_py,
    py2mojo_py, move_py, dsum_sq_py, order_f_py, select_smallest_py, argmin_all_py, sign_labels_py, accuracy_py, pca_mle_rank_host_py, topn_desc_py,
    rowsum_py, sqdist_py, vendor_py,
    idn_flags_py, lu_gesv_py, ols_tsqr_r_py,
)
from x_decomp.host import HostExec, X_DECOMP_HOST_SABOTAGE
from x_decomp.select_ops import order_small_py, reduce_py
from x_decomp.nmf import nmf_nndsvd_py, nmf_solve_py
from x_decomp.ica import ica_solve_py
from x_decomp.fa_em import dsum_f32_py, fa_em_main_py
from x_decomp.chi2 import chi2_cdf_py, chi2_quantile_py, lda_bound_host_py
from x_decomp.als import als_fit_py
from x_decomp.mds import mds_fit_py
from x_decomp.lle_iter import lle_iterate_py
from x_decomp.pls import pls_fit_py
from x_decomp.lda_fit import lda_fit_py
from x_decomp.dictl import dict_learning_py, minibatch_py
from x_decomp.rotation import ortho_rotation_py
from x_decomp.lanczos_host import lanczos_py
from x_decomp.mds_iso import mds_disp_host_py, mds_setup_host_py
from x_decomp.graph_host import (
    graph_knn_py, graph_knn_dense_py, graph_radius_py, graph_radius_geo_py, graph_lle_iw_py, graph_components_py, graph_join_py,
    graph_dijkstra_py,
)


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
        m.def_function[trisolve_py[HostExec]]("x_decomp_trisolve")
        m.def_function[knn_select_py[HostExec]]("x_decomp_knn_select")
        m.def_function[chol_py[HostExec]]("x_decomp_chol")
        m.def_function[eigh_py[HostExec]]("x_decomp_eigh")
        m.def_function[eigh_batch_py[HostExec]]("x_decomp_eigh_batch")
        m.def_function[lle_local_py[HostExec]]("x_decomp_lle_local")
        m.def_function[lle_apply_py[HostExec]]("x_decomp_lle_apply")
        m.def_function[cd_rows_py[HostExec]]("x_decomp_cd_rows")
        m.def_function[orth_py[HostExec]]("x_decomp_orth")
        m.def_function[orth_diag_py[HostExec]]("x_decomp_orth_diag")
        m.def_function[svd_py[HostExec]]("x_decomp_svd")
        m.def_function[lasso_rows_py[HostExec]]("x_decomp_lasso_rows")
        m.def_function[omp_rows_py[HostExec]]("x_decomp_omp_rows")
        m.def_function[lars_rows_py[HostExec]]("x_decomp_lars_rows")
        m.def_function[lu_aux_py[HostExec]]("x_decomp_lu_aux")
        m.def_function[rand_gamma_py[HostExec]]("x_decomp_rand_gamma")
        m.def_function[lda_rows_py[HostExec]]("x_decomp_lda_rows")
        m.def_function[dijkstra_rows_py[HostExec]]("x_decomp_dijkstra_rows")
        m.def_function[barycenter_rows_py[HostExec]]("x_decomp_barycenter_rows")
        m.def_function[als_rows_py[HostExec]]("x_decomp_als_rows")
        m.def_function[absmax_sign_py[HostExec]]("x_decomp_absmax_sign")
        m.def_function[qr_r_py[HostExec]]("x_decomp_qr_r")
        m.def_function[geqrf_py[HostExec]]("x_decomp_geqrf")
        m.def_function[orgqr_py[HostExec]]("x_decomp_orgqr")
        m.def_function[tsqr_r_py[HostExec]]("x_decomp_tsqr_r")
        m.def_function[tsqr_q_py[HostExec]]("x_decomp_tsqr_q")
        m.def_function[r_signs_py]("x_decomp_r_signs")
        m.def_function[rank_above_py]("x_decomp_rank_above")
        # lane idn-dense-linalg: the one-entry OLS and solve routes
        m.def_function[idn_flags_py]("x_decomp_idn_flags")
        m.def_function[ols_tsqr_r_py[HostExec]]("x_decomp_ols_tsqr_r")
        m.def_function[lu_gesv_py[HostExec]]("x_decomp_lu_gesv")
        m.def_function[als_cg_rows_py[HostExec]]("x_decomp_als_cg_rows")
        m.def_function[mcd_py[HostExec]]("x_decomp_mcd")
        m.def_function[lda_online_py[HostExec]]("x_decomp_lda_online")
        m.def_function[nmf_solve_py[HostExec]]("x_decomp_nmf_solve")
        m.def_function[nmf_nndsvd_py[HostExec]]("x_decomp_nmf_nndsvd")
        m.def_function[ica_solve_py[HostExec]]("x_decomp_ica_solve")
        m.def_function[fa_em_main_py[HostExec]]("x_decomp_fa_em_main")
        m.def_function[dsum_f32_py]("x_decomp_dsum_f32")
        m.def_function[dict_learning_py[HostExec]]("x_decomp_dict_learning")
        m.def_function[minibatch_py[HostExec]]("x_decomp_dict_minibatch")
        m.def_function[lda_fit_py[HostExec]]("x_decomp_lda_fit")
        m.def_function[pls_fit_py[HostExec]]("x_decomp_pls_fit")
        m.def_function[lle_iterate_py[HostExec]]("x_decomp_lle_iterate")
        m.def_function[mds_fit_py[HostExec]]("x_decomp_mds_fit")
        m.def_function[als_fit_py[HostExec]]("x_decomp_als_fit")
        m.def_function[chi2_cdf_py]("x_decomp_chi2_cdf")
        m.def_function[chi2_quantile_py]("x_decomp_chi2_quantile")
        m.def_function[lda_bound_host_py]("x_decomp_lda_bound")
        m.def_function[ortho_rotation_py[HostExec]]("x_decomp_ortho_rotation")
        m.def_function[gather_py]("x_decomp_gather")
        m.def_function[scatter_py]("x_decomp_scatter")
        m.def_function[triu_nonzero_py]("x_decomp_triu_nonzero")
        m.def_function[argsort_f32_py]("x_decomp_argsort_f32")
        m.def_function[iso_order_py]("x_decomp_iso_order")
        m.def_function[py2mojo_py]("x_decomp_py2mojo")
        m.def_function[move_py]("x_decomp_move")
        m.def_function[reduce_py]("x_decomp_reduce")
        m.def_function[order_small_py]("x_decomp_order_small")
        m.def_function[lanczos_py]("x_decomp_lanczos")
        m.def_function[mds_setup_host_py]("x_decomp_mds_setup")
        m.def_function[mds_disp_host_py]("x_decomp_mds_disp")
        m.def_function[dsum_sq_py]("x_decomp_dsum_sq")
        m.def_function[order_f_py]("x_decomp_order_f")
        m.def_function[select_smallest_py]("x_decomp_select_smallest")
        m.def_function[argmin_all_py]("x_decomp_argmin_all")
        m.def_function[sign_labels_py]("x_decomp_sign_labels")
        m.def_function[accuracy_py]("x_decomp_accuracy")
        m.def_function[pca_mle_rank_host_py]("x_decomp_pca_mle_rank")
        m.def_function[topn_desc_py]("x_decomp_topn_desc")
        m.def_function[graph_knn_py]("x_decomp_graph_knn")
        m.def_function[graph_knn_dense_py]("x_decomp_graph_knn_dense")
        m.def_function[graph_radius_py]("x_decomp_graph_radius")
        m.def_function[graph_radius_geo_py]("x_decomp_graph_radius_geo")
        m.def_function[graph_lle_iw_py]("x_decomp_graph_lle_iw")
        m.def_function[graph_components_py]("x_decomp_graph_components")
        m.def_function[graph_join_py]("x_decomp_graph_join")
        m.def_function[graph_dijkstra_py]("x_decomp_graph_dijkstra")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[HostExec]]("x_decomp_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp_host: ", e))
