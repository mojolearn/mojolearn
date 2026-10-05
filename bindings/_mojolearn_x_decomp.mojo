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
    geqrf_py, orgqr_py, tsqr_r_py, tsqr_q_py, r_signs_py, rank_above_py, als_cg_rows_py, gather_py, scatter_py, triu_nonzero_py, argsort_f32_py, iso_order_py,
    py2mojo_py, move_py, dsum_sq_py, order_f_py, select_smallest_py, argmin_all_py, sign_labels_py, accuracy_py, pca_mle_rank_host_py, topn_desc_py,
    rowsum_py, sqdist_py, vendor_py, fast_defines_py,
    idn_flags_py, lu_gesv_py, ols_tsqr_r_py,
)
from x_decomp.device import DevExec
from x_decomp.fa_fast import FA_FAST_APPLE, fa_defines_py, fa_em_py, fa_gram_py, fa_transform_py
from std.sys.info import has_apple_gpu_accelerator
from x_decomp.mcd_bmma import MCD_G1_GRAM, MCD_G1_AUDIT, mcd_g1_count, mcd_g1_last

def mcd_g1_gram_on_py() raises -> PythonObject:
    return PythonObject(Int(MCD_G1_GRAM))

def mcd_g1_gram_count_py(index: PythonObject) raises -> PythonObject:
    return PythonObject(mcd_g1_count(Int(py=index)))

def mcd_g1_gram_last_py(index: PythonObject) raises -> PythonObject:
    return PythonObject(mcd_g1_last(Int(py=index)))

from x_decomp.kit_device import lda_online_dev_py, mcd_dev_py
from x_decomp.nmf_dev import nmf_nndsvd_dev_py, nmf_solve_dev_py
from x_decomp.ica_dev import ica_solve_dev_py
from x_decomp.lda_fast import LDA_FUSED_SS, dev_lda_estep_ss_py
from x_decomp.dict_fast import DECOMP_FAST_DICT_DEV, dev_dict_update_py
from x_decomp.select_ops import order_small_py, reduce_py
from x_decomp.select_dev import (
    dev_argmin_all_py, dev_dsum_sq_py, dev_order_f_py, dev_order_small_py, dev_pca_mle_rank_py, dev_reduce_py,
    dev_select_smallest_py,
)
from x_decomp.mds_iso_dev import dev_mds_disp_py, dev_mds_setup_py
from x_decomp.graph_device import (
    dev_graph_knn_py, dev_graph_knn_dense_py, dev_graph_radius_py, dev_graph_radius_geo_py, dev_graph_lle_iw_py, dev_graph_components_py,
    dev_graph_join_py, dev_graph_dijkstra_py,
)
from x_decomp.mcd_bmma import MCD_ORDERED_COV, mcd_cov_reach_py
from x_decomp.resident import (
    mcd_cov_probe_py, dev_mcd_cov_py, dev_alloc_py, dev_colsum_py, dev_download_py, dev_ew_py, dev_free_py, dev_gemm_py, dev_project_py, dev_rand_py, dev_trisolve_py, dev_knn_select_py, dev_rowsum_py,
    dev_sqdist_py, dev_upload_py, dev_absmax_py, dev_orth_py, dev_orth_diag_py, dev_lda_rows_py,
    dev_lda_bound_py, dev_als_rows_py, dev_move_py,
)
from x_decomp.resident import GRP_CLS2_ANY, GRP_CLS2_DEVSCAN, GRP_FAST_FUSED, grp_cls2_py, dev_first_nonfinite_py, grp_fit_fused_py
from x_decomp.lanczos_dev import dev_lanczos_py, ipca_dev_on_py, kpca_lanczos_dev_on_py
from x_decomp.resident import IDN_CD_RESIDENT, dev_cd_rows_py
from x_decomp.resident import IDN_SVD_RESIDENT, dev_svd_py
from x_decomp.resident import IDN_LU_RESIDENT, dev_lu_py, dev_lu_aux_py
from x_decomp.resident import IDN_QR_R_RESIDENT, dev_qr_r_py
from x_decomp.resident import dev_maxabs_py
from x_decomp.api import IDN_DEV_MAXABS
from x_decomp.resident import IDN_CODE_RESIDENT, dev_code_rows_py
from x_decomp.resident import IDN_EIGH_RESIDENT, dev_eigh_py
# merge 2026-10-05: both sides export `dev_lu_aux_py`; the w4 (FAST + Apple) one is
# aliased here. Same Python name "x_decomp_dev_lu_aux" for both: LLE_FAST_DEV_LU is
# FAST-only and IDN_LU_RESIDENT is IDENTICAL-only, so one build registers at most one.
from x_decomp.w4_fast import LLE_FAST_DEV_LU, dev_lu_aux_py as w4_dev_lu_aux_py, w4_flags_py
from x_decomp.qfix import LU_QFIX, lu_resid_py, qfix_flags_py
from x_decomp.tsvd_fast import TSVD_FAST_CHOLQR3, tsvd_cholqr_r_py
from x_decomp.s_linalg_fast import (
    DECOMP_FAST_ORTH_WS, LU_FAST_RESIDENT, RSVD_FAST_DEVSCAN, dev_lu_factor_py, dev_lu_solve_py, dev_orth_ws_py,
    dev_upload_scan_py, s_flags_py,
)


@export
def PyInit__mojolearn_x_decomp() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_decomp")
        comptime if MCD_G1_AUDIT:
            m.def_function[mcd_g1_gram_on_py]("mcd_g1_gram_on")
            m.def_function[mcd_g1_gram_count_py]("mcd_g1_gram_count")
            m.def_function[mcd_g1_gram_last_py]("mcd_g1_gram_last")
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
        m.def_function[r_signs_py]("x_decomp_r_signs")
        m.def_function[rank_above_py]("x_decomp_rank_above")
        # lane idn-dense-linalg: the one-entry OLS and solve routes
        m.def_function[idn_flags_py]("x_decomp_idn_flags")
        m.def_function[ols_tsqr_r_py[DevExec]]("x_decomp_ols_tsqr_r")
        m.def_function[lu_gesv_py[DevExec]]("x_decomp_lu_gesv")
        m.def_function[als_cg_rows_py[DevExec]]("x_decomp_als_cg_rows")
        # MinCovDet's fast_mcd and online LDA on the resident kit
        # (x_decomp/kit_device.mojo)
        m.def_function[mcd_dev_py]("x_decomp_mcd")
        m.def_function[lda_online_dev_py]("x_decomp_lda_online")
        m.def_function[nmf_solve_dev_py]("x_decomp_nmf_solve")
        m.def_function[nmf_nndsvd_dev_py]("x_decomp_nmf_nndsvd")
        m.def_function[ica_solve_dev_py]("x_decomp_ica_solve")
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
        m.def_function[pca_mle_rank_host_py]("x_decomp_pca_mle_rank")
        m.def_function[topn_desc_py]("x_decomp_topn_desc")
        m.def_function[dev_move_py]("x_decomp_dev_move")
        # lane cpu2-l8-decomp: exact select reductions and the small stable order (x_decomp/select_*.mojo)
        m.def_function[reduce_py]("x_decomp_reduce")
        m.def_function[order_small_py]("x_decomp_order_small")
        m.def_function[dev_reduce_py]("x_decomp_dev_reduce")
        m.def_function[dev_order_small_py]("x_decomp_dev_order_small")
        m.def_function[dev_pca_mle_rank_py]("x_decomp_dev_pca_mle_rank")
        m.def_function[dev_order_f_py]("x_decomp_dev_order_f")
        m.def_function[dev_argmin_all_py]("x_decomp_dev_argmin_all")
        m.def_function[dev_select_smallest_py]("x_decomp_dev_select_smallest")
        m.def_function[dev_dsum_sq_py]("x_decomp_dev_dsum_sq")
        m.def_function[dev_mds_setup_py]("x_decomp_dev_mds_setup")
        m.def_function[dev_mds_disp_py]("x_decomp_dev_mds_disp")
        # device-resident matrices (x_decomp/resident.mojo; GPU binding only)
        m.def_function[dev_alloc_py]("x_decomp_dev_alloc")
        m.def_function[dev_free_py]("x_decomp_dev_free")
        m.def_function[dev_upload_py]("x_decomp_dev_upload")
        m.def_function[dev_download_py]("x_decomp_dev_download")
        m.def_function[dev_ew_py]("x_decomp_dev_ew")
        m.def_function[dev_gemm_py]("x_decomp_dev_gemm")
        m.def_function[mcd_cov_reach_py]("x_decomp_mcd_cov_reach")
        # Apple simdgroup MMA probe (tools/mcd_ordered_quality.py): NVIDIA/AMD
        # cannot link air.simdgroup_* (box-run-2 compile fix)
        comptime if has_apple_gpu_accelerator():
            m.def_function[mcd_cov_probe_py]("x_decomp_mcd_cov_probe")
        comptime if MCD_ORDERED_COV:
            m.def_function[dev_mcd_cov_py]("x_decomp_dev_mcd_cov")
        m.def_function[dev_lanczos_py]("x_decomp_dev_lanczos")
        m.def_function[kpca_lanczos_dev_on_py]("x_decomp_lanczos_dev_on")
        m.def_function[ipca_dev_on_py]("x_decomp_ipca_dev_on")
        m.def_function[dev_project_py]("x_decomp_dev_project")
        m.def_function[dev_rand_py]("x_decomp_dev_rand")
        m.def_function[dev_trisolve_py]("x_decomp_dev_trisolve")
        # lane/apple-fast-w4-decomp (x_decomp/w4_fast.mojo): the build's w4 flags
        # (LLE_FAST_DEV_LU, RSVD_FAST_DIRECT_IN); the resident LU only when compiled in
        m.def_function[w4_flags_py]("x_decomp_w4_flags")
        # lane/apple-fast-q-linalg (x_decomp/qfix.mojo): FAST quality repairs
        # (bits SVD_QFIX 1, TSVD_QFIX 2, LU_QFIX 4; QOLD defines restore)
        m.def_function[qfix_flags_py]("x_decomp_qfix_flags")
        comptime if LU_QFIX:
            m.def_function[lu_resid_py]("x_decomp_lu_resid")
        # lane/apple-fast-s-linalg (x_decomp/tsvd_fast.mojo): -D MOJOLEARN_TSVD_FAST_CHOLQR3
        # (default off, FAST + Apple): TSVD_QFIX's R by shifted CholeskyQR3
        comptime if TSVD_FAST_CHOLQR3:
            m.def_function[tsvd_cholqr_r_py]("x_decomp_tsvd_cholqr_r")
        # lane/apple-fast-s-linalg (x_decomp/s_linalg_fast.mojo): bit 1 RSVD_FAST_DEVSCAN,
        # bit 2 DECOMP_FAST_ORTH_WS, bit 4 LU_FAST_RESIDENT (FAST + Apple defaults, _OFF rollbacks)
        m.def_function[s_flags_py]("x_decomp_s_flags")
        comptime if RSVD_FAST_DEVSCAN:
            m.def_function[dev_upload_scan_py]("x_decomp_dev_upload_scan")
        comptime if DECOMP_FAST_ORTH_WS:
            m.def_function[dev_orth_ws_py]("x_decomp_dev_orth_ws")
        comptime if LU_FAST_RESIDENT:
            m.def_function[dev_lu_factor_py]("x_decomp_dev_lu_factor")
            m.def_function[dev_lu_solve_py]("x_decomp_dev_lu_solve")
        comptime if LLE_FAST_DEV_LU:
            m.def_function[w4_dev_lu_aux_py]("x_decomp_dev_lu_aux")
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
        comptime if IDN_CD_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_CD_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_cd_rows_py]("x_decomp_dev_cd_rows")
        comptime if IDN_SVD_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_SVD_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_svd_py]("x_decomp_dev_svd")
        comptime if IDN_LU_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_LU_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_lu_py]("x_decomp_dev_lu")
            m.def_function[dev_lu_aux_py]("x_decomp_dev_lu_aux")
        comptime if IDN_QR_R_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_QR_R_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_qr_r_py]("x_decomp_dev_qr_r")
        comptime if IDN_DEV_MAXABS:
            # lane fix-d1-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_ICA_LIM_DEV_OFF and
            # -D MOJOLEARN_IDN_POLAR_MAX_DEV_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_maxabs_py]("x_decomp_dev_maxabs")
        comptime if IDN_CODE_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_CODE_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_code_rows_py]("x_decomp_dev_code_rows")
        comptime if IDN_EIGH_RESIDENT:
            # lane fam-decomp: IDENTICAL default (off: -D MOJOLEARN_IDN_EIGH_RESIDENT_OFF) (x_decomp/resident.mojo)
            m.def_function[dev_eigh_py]("x_decomp_dev_eigh")
        comptime if DECOMP_FAST_DICT_DEV:
            # lane/apple-fast-gap-clus3: FAST + Apple default (off: -D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF) (x_decomp/dict_fast.mojo)
            m.def_function[dev_dict_update_py]("x_decomp_dev_dict_update")
        # Isomap / LLE graph builds (x_decomp/graph_device.mojo, lane hr2-graph-embed)
        m.def_function[dev_graph_knn_py]("x_decomp_dev_graph_knn")
        m.def_function[dev_graph_knn_dense_py]("x_decomp_dev_graph_knn_dense")
        m.def_function[dev_graph_radius_py]("x_decomp_dev_graph_radius")
        m.def_function[dev_graph_radius_geo_py]("x_decomp_dev_graph_radius_geo")
        m.def_function[dev_graph_lle_iw_py]("x_decomp_dev_graph_lle_iw")
        m.def_function[dev_graph_components_py]("x_decomp_dev_graph_components")
        m.def_function[dev_graph_join_py]("x_decomp_dev_graph_join")
        m.def_function[dev_graph_dijkstra_py]("x_decomp_dev_graph_dijkstra")
        # lane/apple-fast-gap-cls2: the random projections' FAST Apple fit
        # switches (x_decomp/resident.mojo GRP_CLS2_*; DEVSCAN FAST + Apple default)
        comptime if GRP_CLS2_ANY:
            m.def_function[grp_cls2_py]("x_decomp_grp_cls2")
        comptime if GRP_CLS2_DEVSCAN:
            m.def_function[dev_first_nonfinite_py]("x_decomp_dev_first_nonfinite")
        comptime if GRP_FAST_FUSED:
            m.def_function[grp_fit_fused_py]("x_decomp_grp_fit_fused")
        # FactorAnalysis on the Apple GPU, FAST only (x_decomp/fa_fast.mojo,
        # lane/apple-fast-fa recovered by lane/apple-fast-rec-fa-robust):
        # registered only in a FAST build for Apple; each route also needs its
        # -D MOJOLEARN_FA_<NAME> (reported by x_decomp_fa_defines, empty when none)
        comptime if FA_FAST_APPLE:
            m.def_function[fa_defines_py]("x_decomp_fa_defines")
            m.def_function[fa_gram_py]("x_decomp_fa_gram")
            m.def_function[fa_em_py]("x_decomp_fa_em")
            m.def_function[fa_transform_py]("x_decomp_fa_transform")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[DevExec]]("x_decomp_vendor")
        m.def_function[fast_defines_py]("x_decomp_fast_defines")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp: ", e))
