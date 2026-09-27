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
)
from x_prep.eigh import eigh_unit
from x_prep.target import te_global_unit, te_enc_unit, te_apply_unit
from x_prep.kbins import kbins_edges_unit, kbins_codes_unit
from naive_bayes.nb import (
    gnb_eps_unit, gnb_params_unit, gnb_jll_unit, class_log_prior_unit, mnb_params_unit,
    bnb_params_unit, cnb_params_unit, cat_params_unit, cat_jll_unit,
)
from x_prep.transform import qt_apply_unit, pt_fit_unit, pt_apply_unit, std_params_unit, normalize_unit, poly_unit
from x_prep.spline import spline_knots_unit, spline_apply_unit
from x_prep.iterative import (
    ii_mean_unit, ii_gram_unit, ii_sub_unit, ii_br_unit, ii_predict_unit, ii_snapshot_unit, ii_conv_unit,
    nan_mask_unit,
)
from x_prep.stats import f_classif_unit, f_regression_unit, chi2_unit
from naive_bayes.da import (
    lda_prep_unit, lda_w_unit, lda_stage2_unit, lda_stage3_unit, qda_cov_unit, qda_prep_unit, qda_dec_unit,
)

comptime N_OPS = 66


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
