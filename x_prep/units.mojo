# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The op table: op id -> unit. python/mojolearn/_expansion_prep.py `_OPS` carries
the same ids; the binding refuses an id outside the table."""
from x_prep.common import FP, IP
from x_prep.prims import (
    sort_cols_unit, col_stats_unit, quantile_unit, affine_unit, scale_params_unit,
    unique_cols_unit, mode_cols_unit, lookup_unit, count_neg_unit, onehot_unit,
    i2f_unit, f2i_unit, binarize_unit, matmul_unit, row_softmax_unit, row_argmax_unit,
    class_stats_unit, center_rows_unit, where_neg_unit, mark_missing_unit, fill_unit,
    label_binarize_unit, scatter_ones_unit, gather_cols_unit, var_ptp_unit,
    sqsum_cols_unit, block_argmax_unit, ord_inverse_unit, cat_gather_unit, where_code_unit,
    class_stats_w_unit, indicator_unit, code_counts_unit, remap_codes_unit, add_arrays_unit,
    scaler_stats_unit, std_scale_unit, nan_keep_unit, hcat_unit, colblock_unit,
)
from x_prep.eigh import eigh_unit
from x_prep.target import (
    te_global_unit, te_enc_unit, te_apply_unit, te_bucket_unit, te_gather_unit,
    te_hist_unit, te_hsum_unit, te_hstart_unit, te_hscatter_unit,
)
from x_prep.kbins import kbins_edges_unit, kbins_codes_unit, kbins_inverse_unit, kbins_gw_unit, kbins_wq_unit, kbins_wkm_unit
from naive_bayes.nb import (
    gnb_eps_unit, gnb_params_unit, gnb_jll_unit, class_log_prior_unit, mnb_params_unit,
    bnb_params_unit, cnb_params_unit, cat_params_unit, cat_jll_unit, log_unit,
    gnb_merge_unit, cat_counts_unit, cat_flp_unit,
)
from x_prep.transform import (
    qt_apply_unit, pt_fit_unit, pt_apply_unit, std_params_unit, normalize_unit, poly_unit, robust_uv_unit,
    qt_inverse_unit, pt_inverse_unit, pt_init_unit, pt_map_unit, pt_fold_unit, pt_log_unit,
    pt_spts_unit, pt_smap_unit, pt_sfold_unit, pt_sres_unit,
)
from x_prep.spline import spline_knots_unit, spline_apply_unit
from x_prep.iterative import (
    ii_mean_unit, ii_gram_unit, ii_sub_unit, ii_br_unit, ii_predict_unit, ii_snapshot_unit, ii_conv_unit,
    nan_mask_unit, ii_sigma_unit, ii_post_unit, ii_rowabs_unit,
    ii_rcount_unit, ii_rwrite_unit, ii_gather_unit, ii_scatter_unit,
)
from x_prep.labels import lab_load_unit, uniq_count_unit, uniq_scan_unit, uniq_write_unit, chunk_neg_unit, row_ones_unit
from x_prep.stats import f_classif_unit, f_regression_unit, chi2_unit
from x_prep.mutual_info import mi_colscale_unit, mi_noise_unit, mi_cc_unit, mi_cd_unit, mi_reduce_unit, mi_dc_unit, mi_dd_unit
from x_prep.blocked import (
    colb_part_unit, colb_fold_unit, colb_ss_unit, colb_var_unit, maxabs_fold_unit,
    csb_part_unit, csb_fold_unit, csb_ss_unit, csb_var_unit, cat_hpart_unit, cat_hfold_unit,
)
from naive_bayes.da import (
    lda_prep_unit, lda_w_unit, lda_stage2_unit, lda_stage3_unit, qda_cov_unit, qda_prep_unit, qda_dec_unit,
    da_shrink_unit, da_pool_unit, sym_fn_unit, da_intercept_unit, evr_unit,
)
from x_prep.calib import (
    CALIB_FOLDS, cal_fold_part_unit, cal_fold_scan_unit, cal_fold_rank_unit, cal_fold_assign_unit,
    cal_lofo_merge_unit, cal_eps_folds_unit, cal_params_folds_unit, cal_jll_folds_unit, cal_platt_init_unit,
    cal_platt_setup_unit, cal_platt_part_unit, cal_platt_step_unit, cal_platt_ls_part_unit, cal_platt_ls_pick_unit,
    cal_sigmoid_avg_unit,
)

#: ops 142-156 (x_prep/calib.mojo) exist only under CALIB_FOLDS (FAST + Apple,
#: default on, -D MOJOLEARN_CALIB_GNB_FOLDS_OFF off); every other binding keeps the 142-op table.
comptime N_OPS = 157 if CALIB_FOLDS else 142


@always_inline
def run_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == 0:
        sort_cols_unit(t, f, q)
    comptime if OP == 1:
        col_stats_unit(t, f, q)
    comptime if OP == 2:
        quantile_unit(t, f, q)
    comptime if OP == 3:
        affine_unit(t, f, q)
    comptime if OP == 4:
        scale_params_unit(t, f, q)
    comptime if OP == 5:
        unique_cols_unit(t, f, q)
    comptime if OP == 6:
        mode_cols_unit(t, f, q)
    comptime if OP == 7:
        lookup_unit(t, f, q)
    comptime if OP == 8:
        count_neg_unit(t, f, q)
    comptime if OP == 9:
        onehot_unit(t, f, q)
    comptime if OP == 10:
        i2f_unit(t, f, q)
    comptime if OP == 11:
        f2i_unit(t, f, q)
    comptime if OP == 12:
        binarize_unit(t, f, q)
    comptime if OP == 13:
        matmul_unit(t, f, q)
    comptime if OP == 14:
        row_softmax_unit(t, f, q)
    comptime if OP == 15:
        row_argmax_unit(t, f, q)
    comptime if OP == 16:
        class_stats_unit(t, f, q)
    comptime if OP == 17:
        center_rows_unit(t, f, q)
    comptime if OP == 18:
        eigh_unit(t, f, q)
    comptime if OP == 19:
        where_neg_unit(t, f, q)
    comptime if OP == 20:
        te_global_unit(t, f, q)
    comptime if OP == 21:
        te_enc_unit(t, f, q)
    comptime if OP == 22:
        te_apply_unit(t, f, q)
    comptime if OP == 23:
        mark_missing_unit(t, f, q)
    comptime if OP == 24:
        fill_unit(t, f, q)
    comptime if OP == 25:
        kbins_edges_unit(t, f, q)
    comptime if OP == 26:
        kbins_codes_unit(t, f, q)
    comptime if OP == 27:
        gnb_eps_unit(t, f, q)
    comptime if OP == 28:
        gnb_params_unit(t, f, q)
    comptime if OP == 29:
        gnb_jll_unit(t, f, q)
    comptime if OP == 30:
        class_log_prior_unit(t, f, q)
    comptime if OP == 31:
        mnb_params_unit(t, f, q)
    comptime if OP == 32:
        bnb_params_unit(t, f, q)
    comptime if OP == 33:
        cnb_params_unit(t, f, q)
    comptime if OP == 34:
        cat_params_unit(t, f, q)
    comptime if OP == 35:
        cat_jll_unit(t, f, q)
    comptime if OP == 36:
        lda_prep_unit(t, f, q)
    comptime if OP == 37:
        lda_w_unit(t, f, q)
    comptime if OP == 38:
        lda_stage2_unit(t, f, q)
    comptime if OP == 39:
        lda_stage3_unit(t, f, q)
    comptime if OP == 40:
        qda_cov_unit(t, f, q)
    comptime if OP == 41:
        qda_prep_unit(t, f, q)
    comptime if OP == 42:
        qda_dec_unit(t, f, q)
    comptime if OP == 43:
        qt_apply_unit(t, f, q)
    comptime if OP == 44:
        pt_fit_unit(t, f, q)
    comptime if OP == 45:
        pt_apply_unit(t, f, q)
    comptime if OP == 46:
        std_params_unit(t, f, q)
    comptime if OP == 47:
        normalize_unit(t, f, q)
    comptime if OP == 48:
        poly_unit(t, f, q)
    comptime if OP == 49:
        spline_knots_unit(t, f, q)
    comptime if OP == 50:
        spline_apply_unit(t, f, q)
    comptime if OP == 51:
        label_binarize_unit(t, f, q)
    comptime if OP == 52:
        scatter_ones_unit(t, f, q)
    comptime if OP == 53:
        ii_mean_unit(t, f, q)
    comptime if OP == 54:
        ii_gram_unit(t, f, q)
    comptime if OP == 55:
        ii_sub_unit(t, f, q)
    comptime if OP == 56:
        ii_br_unit(t, f, q)
    comptime if OP == 57:
        ii_predict_unit(t, f, q)
    comptime if OP == 58:
        ii_snapshot_unit(t, f, q)
    comptime if OP == 59:
        ii_conv_unit(t, f, q)
    comptime if OP == 60:
        nan_mask_unit(t, f, q)
    comptime if OP == 61:
        gather_cols_unit(t, f, q)
    comptime if OP == 62:
        var_ptp_unit(t, f, q)
    comptime if OP == 63:
        f_classif_unit(t, f, q)
    comptime if OP == 64:
        f_regression_unit(t, f, q)
    comptime if OP == 65:
        chi2_unit(t, f, q)
    comptime if OP == 66:
        mi_colscale_unit(t, f, q)
    comptime if OP == 67:
        mi_noise_unit(t, f, q)
    comptime if OP == 68:
        mi_cc_unit(t, f, q)
    comptime if OP == 69:
        mi_cd_unit(t, f, q)
    comptime if OP == 70:
        mi_reduce_unit(t, f, q)
    comptime if OP == 71:
        sqsum_cols_unit(t, f, q)
    comptime if OP == 72:
        log_unit(t, f, q)
    comptime if OP == 73:
        robust_uv_unit(t, f, q)
    comptime if OP == 74:
        qt_inverse_unit(t, f, q)
    comptime if OP == 75:
        pt_inverse_unit(t, f, q)
    comptime if OP == 76:
        block_argmax_unit(t, f, q)
    comptime if OP == 77:
        ord_inverse_unit(t, f, q)
    comptime if OP == 78:
        cat_gather_unit(t, f, q)
    comptime if OP == 79:
        where_code_unit(t, f, q)
    comptime if OP == 80:
        kbins_inverse_unit(t, f, q)
    comptime if OP == 81:
        da_shrink_unit(t, f, q)
    comptime if OP == 82:
        da_pool_unit(t, f, q)
    comptime if OP == 83:
        sym_fn_unit(t, f, q)
    comptime if OP == 84:
        da_intercept_unit(t, f, q)
    comptime if OP == 85:
        evr_unit(t, f, q)
    comptime if OP == 86:
        class_stats_w_unit(t, f, q)
    comptime if OP == 87:
        indicator_unit(t, f, q)
    comptime if OP == 88:
        code_counts_unit(t, f, q)
    comptime if OP == 89:
        remap_codes_unit(t, f, q)
    comptime if OP == 90:
        add_arrays_unit(t, f, q)
    comptime if OP == 91:
        gnb_merge_unit(t, f, q)
    comptime if OP == 92:
        cat_counts_unit(t, f, q)
    comptime if OP == 93:
        cat_flp_unit(t, f, q)
    comptime if OP == 94:
        mi_dc_unit(t, f, q)
    comptime if OP == 95:
        mi_dd_unit(t, f, q)
    comptime if OP == 96:
        kbins_gw_unit(t, f, q)
    comptime if OP == 97:
        kbins_wq_unit(t, f, q)
    comptime if OP == 98:
        kbins_wkm_unit(t, f, q)
    comptime if OP == 99:
        ii_sigma_unit(t, f, q)
    comptime if OP == 100:
        ii_post_unit(t, f, q)
    comptime if OP == 101:
        scaler_stats_unit(t, f, q)
    comptime if OP == 102:
        std_scale_unit(t, f, q)
    comptime if OP == 103:
        nan_keep_unit(t, f, q)
    comptime if OP == 104:
        pt_init_unit(t, f, q)
    comptime if OP == 105:
        pt_map_unit(t, f, q)
    comptime if OP == 106:
        pt_fold_unit(t, f, q)
    comptime if OP == 107:
        ii_rowabs_unit(t, f, q)
    comptime if OP == 108:
        te_bucket_unit(t, f, q)
    comptime if OP == 109:
        pt_log_unit(t, f, q)
    comptime if OP == 110:
        pt_spts_unit(t, f, q)
    comptime if OP == 111:
        pt_smap_unit(t, f, q)
    comptime if OP == 112:
        pt_sfold_unit(t, f, q)
    comptime if OP == 113:
        pt_sres_unit(t, f, q)
    comptime if OP == 114:
        te_gather_unit(t, f, q)
    comptime if OP == 115:
        te_hist_unit(t, f, q)
    comptime if OP == 116:
        te_hsum_unit(t, f, q)
    comptime if OP == 117:
        te_hstart_unit(t, f, q)
    comptime if OP == 118:
        te_hscatter_unit(t, f, q)
    comptime if OP == 119:
        lab_load_unit(t, f, q)
    comptime if OP == 120:
        uniq_count_unit(t, f, q)
    comptime if OP == 121:
        uniq_scan_unit(t, f, q)
    comptime if OP == 122:
        uniq_write_unit(t, f, q)
    comptime if OP == 123:
        chunk_neg_unit(t, f, q)
    comptime if OP == 124:
        colb_part_unit(t, f, q)
    comptime if OP == 125:
        colb_fold_unit(t, f, q)
    comptime if OP == 126:
        colb_ss_unit(t, f, q)
    comptime if OP == 127:
        colb_var_unit(t, f, q)
    comptime if OP == 128:
        maxabs_fold_unit(t, f, q)
    comptime if OP == 129:
        csb_part_unit(t, f, q)
    comptime if OP == 130:
        csb_fold_unit(t, f, q)
    comptime if OP == 131:
        csb_ss_unit(t, f, q)
    comptime if OP == 132:
        csb_var_unit(t, f, q)
    comptime if OP == 133:
        cat_hpart_unit(t, f, q)
    comptime if OP == 134:
        cat_hfold_unit(t, f, q)
    comptime if OP == 135:
        row_ones_unit(t, f, q)
    comptime if OP == 136:
        ii_rcount_unit(t, f, q)
    comptime if OP == 137:
        ii_rwrite_unit(t, f, q)
    comptime if OP == 138:
        ii_gather_unit(t, f, q)
    comptime if OP == 139:
        ii_scatter_unit(t, f, q)
    comptime if OP == 140:
        hcat_unit(t, f, q)
    comptime if OP == 141:
        colblock_unit(t, f, q)
    comptime if CALIB_FOLDS:
        comptime if OP == 142:
            cal_fold_part_unit(t, f, q)
        comptime if OP == 143:
            cal_fold_scan_unit(t, f, q)
        comptime if OP == 144:
            cal_fold_rank_unit(t, f, q)
        comptime if OP == 145:
            cal_fold_assign_unit(t, f, q)
        comptime if OP == 146:
            cal_lofo_merge_unit(t, f, q)
        comptime if OP == 147:
            cal_eps_folds_unit(t, f, q)
        comptime if OP == 148:
            cal_params_folds_unit(t, f, q)
        comptime if OP == 149:
            cal_jll_folds_unit(t, f, q)
        comptime if OP == 150:
            cal_platt_init_unit(t, f, q)
        comptime if OP == 151:
            cal_platt_setup_unit(t, f, q)
        comptime if OP == 152:
            cal_platt_part_unit(t, f, q)
        comptime if OP == 153:
            cal_platt_step_unit(t, f, q)
        comptime if OP == 154:
            cal_platt_ls_part_unit(t, f, q)
        comptime if OP == 155:
            cal_platt_ls_pick_unit(t, f, q)
        comptime if OP == 156:
            cal_sigmoid_avg_unit(t, f, q)
