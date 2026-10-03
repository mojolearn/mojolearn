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
    cd_rows_py, chol_py, colsum_py, eigh_py, eigh_batch_py, lle_local_py, lle_apply_py, ew_py, gemm_py, lu_py, lu_solve_py, trisolve_py, knn_select_py, numeric_mode_py, orth_py, orth_diag_py, rand_py, svd_py, lasso_rows_py, lars_rows_py, lu_aux_py, omp_rows_py, rand_gamma_py, lda_rows_py, dijkstra_rows_py, barycenter_rows_py, als_rows_py, absmax_sign_py, qr_r_py,
    geqrf_py, orgqr_py, tsqr_r_py, tsqr_q_py, als_cg_rows_py, gather_py, scatter_py, triu_nonzero_py, argsort_f32_py, iso_order_py,
    py2mojo_py, move_py, dsum_sq_py, order_f_py, select_smallest_py, argmin_all_py, sign_labels_py, accuracy_py, pca_mle_rank_terms_py, pca_mle_pa_py,
    rowsum_py, sqdist_py, vendor_py,
)
from x_decomp.device import DevExec
from x_decomp.kit_device import lda_online_dev_py, mcd_dev_py
from x_decomp.lda_fast import LDA_FUSED_SS, dev_lda_estep_ss_py
from x_decomp.graph_device import (
    dev_graph_knn_py, dev_graph_knn_dense_py, dev_graph_radius_py, dev_graph_radius_geo_py, dev_graph_lle_iw_py, dev_graph_components_py,
    dev_graph_join_py, dev_graph_dijkstra_py,
)
from x_decomp.resident import (
    dev_alloc_py, dev_colsum_py, dev_download_py, dev_ew_py, dev_free_py, dev_gemm_py, dev_project_py, dev_rand_py, dev_trisolve_py, dev_knn_select_py, dev_rowsum_py,
    dev_sqdist_py, dev_upload_py, dev_absmax_py, dev_orth_py, dev_orth_diag_py, dev_lda_rows_py,
    dev_lda_bound_py, dev_als_rows_py, dev_move_py,
)


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
        m.def_function[trisolve_py[DevExec]]("x_decomp_trisolve")
        m.def_function[knn_select_py[DevExec]]("x_decomp_knn_select")
        m.def_function[chol_py[DevExec]]("x_decomp_chol")
        m.def_function[eigh_py[DevExec]]("x_decomp_eigh")
        m.def_function[eigh_batch_py[DevExec]]("x_decomp_eigh_batch")
        m.def_function[lle_local_py[DevExec]]("x_decomp_lle_local")
        m.def_function[lle_apply_py[DevExec]]("x_decomp_lle_apply")
        m.def_function[cd_rows_py[DevExec]]("x_decomp_cd_rows")
        m.def_function[orth_py[DevExec]]("x_decomp_orth")
        m.def_function[orth_diag_py[DevExec]]("x_decomp_orth_diag")
        m.def_function[svd_py[DevExec]]("x_decomp_svd")
        m.def_function[lasso_rows_py[DevExec]]("x_decomp_lasso_rows")
        m.def_function[omp_rows_py[DevExec]]("x_decomp_omp_rows")
        m.def_function[lars_rows_py[DevExec]]("x_decomp_lars_rows")
        m.def_function[lu_aux_py[DevExec]]("x_decomp_lu_aux")
        m.def_function[rand_gamma_py[DevExec]]("x_decomp_rand_gamma")
        m.def_function[lda_rows_py[DevExec]]("x_decomp_lda_rows")
        m.def_function[dijkstra_rows_py[DevExec]]("x_decomp_dijkstra_rows")
        m.def_function[barycenter_rows_py[DevExec]]("x_decomp_barycenter_rows")
        m.def_function[als_rows_py[DevExec]]("x_decomp_als_rows")
        m.def_function[absmax_sign_py[DevExec]]("x_decomp_absmax_sign")
        m.def_function[qr_r_py[DevExec]]("x_decomp_qr_r")
        m.def_function[geqrf_py[DevExec]]("x_decomp_geqrf")
        m.def_function[orgqr_py[DevExec]]("x_decomp_orgqr")
        m.def_function[tsqr_r_py[DevExec]]("x_decomp_tsqr_r")
        m.def_function[tsqr_q_py[DevExec]]("x_decomp_tsqr_q")
        m.def_function[als_cg_rows_py[DevExec]]("x_decomp_als_cg_rows")
        # MinCovDet's fast_mcd and online LDA on the resident kit
        # (x_decomp/kit_device.mojo)
        m.def_function[mcd_dev_py]("x_decomp_mcd")
        m.def_function[lda_online_dev_py]("x_decomp_lda_online")
        m.def_function[gather_py]("x_decomp_gather")
        m.def_function[scatter_py]("x_decomp_scatter")
        m.def_function[triu_nonzero_py]("x_decomp_triu_nonzero")
        m.def_function[argsort_f32_py]("x_decomp_argsort_f32")
        m.def_function[iso_order_py]("x_decomp_iso_order")
        m.def_function[py2mojo_py]("x_decomp_py2mojo")
        m.def_function[move_py]("x_decomp_move")
        m.def_function[dsum_sq_py]("x_decomp_dsum_sq")
        m.def_function[order_f_py]("x_decomp_order_f")
        m.def_function[select_smallest_py]("x_decomp_select_smallest")
        m.def_function[argmin_all_py]("x_decomp_argmin_all")
        m.def_function[sign_labels_py]("x_decomp_sign_labels")
        m.def_function[accuracy_py]("x_decomp_accuracy")
        m.def_function[pca_mle_rank_terms_py]("x_decomp_pca_mle_terms")
        m.def_function[pca_mle_pa_py]("x_decomp_pca_mle_pa")
        m.def_function[dev_move_py]("x_decomp_dev_move")
        # device-resident matrices (x_decomp/resident.mojo; GPU binding only)
        m.def_function[dev_alloc_py]("x_decomp_dev_alloc")
        m.def_function[dev_free_py]("x_decomp_dev_free")
        m.def_function[dev_upload_py]("x_decomp_dev_upload")
        m.def_function[dev_download_py]("x_decomp_dev_download")
        m.def_function[dev_ew_py]("x_decomp_dev_ew")
        m.def_function[dev_gemm_py]("x_decomp_dev_gemm")
        m.def_function[dev_project_py]("x_decomp_dev_project")
        m.def_function[dev_rand_py]("x_decomp_dev_rand")
        m.def_function[dev_trisolve_py]("x_decomp_dev_trisolve")
        m.def_function[dev_knn_select_py]("x_decomp_dev_knn_select")
        m.def_function[dev_colsum_py]("x_decomp_dev_colsum")
        m.def_function[dev_rowsum_py]("x_decomp_dev_rowsum")
        m.def_function[dev_sqdist_py]("x_decomp_dev_sqdist")
        m.def_function[dev_absmax_py]("x_decomp_dev_absmax")
        m.def_function[dev_orth_py]("x_decomp_dev_orth")
        m.def_function[dev_orth_diag_py]("x_decomp_dev_orth_diag")
        m.def_function[dev_lda_rows_py]("x_decomp_dev_lda_rows")
        m.def_function[dev_lda_bound_py]("x_decomp_dev_lda_bound")
        comptime if LDA_FUSED_SS:
            # lane apple-fast-nb: FAST + Apple default (off: -D MOJOLEARN_LDA_FUSED_SS_OFF) (x_decomp/lda_fast.mojo)
            m.def_function[dev_lda_estep_ss_py]("x_decomp_dev_lda_estep_ss")
        m.def_function[dev_als_rows_py]("x_decomp_dev_als_rows")
        # Isomap / LLE graph builds (x_decomp/graph_device.mojo, lane hr2-graph-embed)
        m.def_function[dev_graph_knn_py]("x_decomp_dev_graph_knn")
        m.def_function[dev_graph_knn_dense_py]("x_decomp_dev_graph_knn_dense")
        m.def_function[dev_graph_radius_py]("x_decomp_dev_graph_radius")
        m.def_function[dev_graph_radius_geo_py]("x_decomp_dev_graph_radius_geo")
        m.def_function[dev_graph_lle_iw_py]("x_decomp_dev_graph_lle_iw")
        m.def_function[dev_graph_components_py]("x_decomp_dev_graph_components")
        m.def_function[dev_graph_join_py]("x_decomp_dev_graph_join")
        m.def_function[dev_graph_dijkstra_py]("x_decomp_dev_graph_dijkstra")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[DevExec]]("x_decomp_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp: ", e))
