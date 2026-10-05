# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The op table: op id -> unit. python/mojolearn/_expansion_metrics.py `_OPS`
carries the same ids; the runners refuse an id outside the table."""
from x_metrics.common import FP, IP
from x_metrics.group import group_sort_unit, group_sum_unit, pair_key_unit
from x_metrics.ranking import bin_curve_unit, row_metric_unit
from x_metrics.cluster import row_centroid_dist_unit
from x_metrics.onehot import onehot_unit, rep_rows_unit, pair_cols_unit
from x_metrics.split import permute_unit, fold_rows_unit, rows64_unit, strat_codes_unit, approx_mode_unit
from x_metrics.regression import reg_term_unit, col_sort_unit, wpercentile_unit, col_max_unit, wpct_select
from x_metrics.par import (
    cs_hist_unit, cs_scan_rows_unit, cs_scan_groups_unit, cs_place_unit,
    fold_leaf_unit, fold_level_unit, fold_final_unit,
    sort_key_unit, sort_runs_unit, sort_merge_unit, sort_emit_unit,
    curve_gather_unit, curve_emit_unit, wpct_gather_unit, copy_unit,
    cm_chunk_unit, cm_final_unit, wpct_iota_unit, curve_cnt_unit, curve_off_unit, curve_fill_unit,
    curve_keep_unit, fr_scatter_unit, fr_cnt_unit, fr_off_unit, fr_fill_unit,
    ck_cnt_unit, ck_off_unit, ck_fill_unit,
    curve_fold_unit, cf_chunk_unit, cf_final_unit, wpct_csum_unit, wpct_coff_unit, wpct_cfill_unit,
)
from x_metrics.cls_epi import cls_epi_unit
from x_metrics.tail import off_diff_unit, flag_scan_unit, proba_rows_unit
from x_metrics.cm_epi import cm_epi_unit
from x_metrics.reg_epi import reg_epi_unit
from x_metrics.rank_epi import rank_epi_unit, cl_epi_unit
from x_metrics.curve_out import (
    curve_out_unit, co_keep_unit, co_emit_unit, co_det_unit, auc_xy_unit, ax_chunk_unit, ax_final_unit,
)
from x_metrics.contingency import (
    cont_stats_unit, ct_cell_unit, ct_row_unit, ct_col_unit, ct_isum_unit, ct_pairs_unit, ct_ent_unit,
    ff_chunk_unit, ff_fin_unit, mi_cell_unit, emi_cell_unit,
)

#: ops 0..10, 36 (fold_rows), 41 (rows64), 45 (strat_codes) (lane metrics-apple2) and 46
#: (curve_fold, lane cgr2-metrics-shap) are the caller's (x_metrics/plan.mojo
#: `is_user_op`); the others are the planner's parallel schedules
#: (x_metrics/par.mojo). 23 and 25 (the retired sequential prefixes) run nothing.
#: 52..54 (onehot, rep_rows, pair_cols; lane apple-fast-py2mojo-core) are the caller's too.
#: 55 (cls_epi, x_metrics/cls_epi.mojo; lane fam2-prep-metrics) is the caller's, ON in both
#: numeric modes since lane cpu2-l7-metrics (2026-10-04: the Python epilogue left the GPU route).
#: 56..62 (lane cpu2-l7-metrics) are the caller's: off_diff, flag_scan, proba_rows
#: (x_metrics/tail.mojo), cm_epi (x_metrics/cm_epi.mojo), reg_epi (x_metrics/reg_epi.mojo),
#: rank_epi and cl_epi (x_metrics/rank_epi.mojo): the metric tails and scans on the device.
#: 63 (curve_out, lane cpu4-python) is the caller's; 64..66 (co_keep, co_emit, co_det) are
#: its planned schedule (x_metrics/curve_out.mojo): the curve arrays on the device.
#: 67 (auc_xy, the public auc's trapezoid sum) is the caller's; 68, 69 (ax_chunk, ax_final) its schedule.
#: 70 (cont_stats, x_metrics/contingency.mojo) is the caller's; 71..80 its schedule: the
#: contingency counts, sums, pair confusion, entropies, MI and expected MI on the device.
#: 81 (approx_mode, x_metrics/split.mojo; lane py-runtime-b) is the caller's: StratifiedShuffleSplit's
#: per-class draw counts with the counter-RNG tie draw.
comptime N_OPS = 82


@always_inline
def run_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == 0:
        group_sort_unit(t, f, q)
    comptime if OP == 1:
        group_sum_unit(t, f, q)
    comptime if OP == 2:
        pair_key_unit(t, f, q)
    comptime if OP == 3:
        reg_term_unit(t, f, q)
    comptime if OP == 4:
        col_sort_unit(t, f, q)
    comptime if OP == 5:
        wpercentile_unit(t, f, q)
    comptime if OP == 6:
        col_max_unit(t, f, q)
    comptime if OP == 7:
        bin_curve_unit(t, f, q)
    comptime if OP == 8:
        row_metric_unit(t, f, q)
    comptime if OP == 9:
        row_centroid_dist_unit(t, f, q)
    comptime if OP == 10:
        permute_unit(t, f, q)
    comptime if OP == 11:
        cs_hist_unit(t, f, q)
    comptime if OP == 12:
        cs_scan_rows_unit(t, f, q)
    comptime if OP == 13:
        cs_scan_groups_unit(t, f, q)
    comptime if OP == 14:
        cs_place_unit(t, f, q)
    comptime if OP == 15:
        fold_leaf_unit(t, f, q)
    comptime if OP == 16:
        fold_level_unit(t, f, q)
    comptime if OP == 17:
        fold_final_unit(t, f, q)
    comptime if OP == 18:
        sort_key_unit(t, f, q)
    comptime if OP == 19:
        sort_runs_unit(t, f, q)
    comptime if OP == 20:
        sort_merge_unit(t, f, q)
    comptime if OP == 21:
        sort_emit_unit(t, f, q)
    comptime if OP == 22:
        curve_gather_unit(t, f, q)
    comptime if OP == 24:
        wpct_gather_unit(t, f, q)
    comptime if OP == 26:
        wpct_select(t, f, q)
    comptime if OP == 27:
        curve_emit_unit(t, f, q)
    comptime if OP == 28:
        copy_unit(t, f, q)
    comptime if OP == 29:
        cm_chunk_unit(t, f, q)
    comptime if OP == 30:
        cm_final_unit(t, f, q)
    comptime if OP == 31:
        wpct_iota_unit(t, f, q)
    comptime if OP == 32:
        curve_cnt_unit(t, f, q)
    comptime if OP == 33:
        curve_off_unit(t, f, q)
    comptime if OP == 34:
        curve_fill_unit(t, f, q)
    comptime if OP == 35:
        curve_keep_unit(t, f, q)
    comptime if OP == 36:
        fold_rows_unit(t, f, q)
    comptime if OP == 37:
        fr_scatter_unit(t, f, q)
    comptime if OP == 38:
        fr_cnt_unit(t, f, q)
    comptime if OP == 39:
        fr_off_unit(t, f, q)
    comptime if OP == 40:
        fr_fill_unit(t, f, q)
    comptime if OP == 41:
        rows64_unit(t, f, q)
    comptime if OP == 42:
        ck_cnt_unit(t, f, q)
    comptime if OP == 43:
        ck_off_unit(t, f, q)
    comptime if OP == 44:
        ck_fill_unit(t, f, q)
    comptime if OP == 45:
        strat_codes_unit(t, f, q)
    comptime if OP == 46:
        curve_fold_unit(t, f, q)
    comptime if OP == 47:
        cf_chunk_unit(t, f, q)
    comptime if OP == 48:
        cf_final_unit(t, f, q)
    comptime if OP == 49:
        wpct_csum_unit(t, f, q)
    comptime if OP == 50:
        wpct_coff_unit(t, f, q)
    comptime if OP == 51:
        wpct_cfill_unit(t, f, q)
    comptime if OP == 52:
        onehot_unit(t, f, q)
    comptime if OP == 53:
        rep_rows_unit(t, f, q)
    comptime if OP == 54:
        pair_cols_unit(t, f, q)
    comptime if OP == 55:
        cls_epi_unit(t, f, q)
    comptime if OP == 56:
        off_diff_unit(t, f, q)
    comptime if OP == 57:
        flag_scan_unit(t, f, q)
    comptime if OP == 58:
        proba_rows_unit(t, f, q)
    comptime if OP == 59:
        cm_epi_unit(t, f, q)
    comptime if OP == 60:
        reg_epi_unit(t, f, q)
    comptime if OP == 61:
        rank_epi_unit(t, f, q)
    comptime if OP == 62:
        cl_epi_unit(t, f, q)
    comptime if OP == 63:
        curve_out_unit(t, f, q)
    comptime if OP == 64:
        co_keep_unit(t, f, q)
    comptime if OP == 65:
        co_emit_unit(t, f, q)
    comptime if OP == 66:
        co_det_unit(t, f, q)
    comptime if OP == 67:
        auc_xy_unit(t, f, q)
    comptime if OP == 68:
        ax_chunk_unit(t, f, q)
    comptime if OP == 69:
        ax_final_unit(t, f, q)
    comptime if OP == 70:
        cont_stats_unit(t, f, q)
    comptime if OP == 71:
        ct_cell_unit(t, f, q)
    comptime if OP == 72:
        ct_row_unit(t, f, q)
    comptime if OP == 73:
        ct_col_unit(t, f, q)
    comptime if OP == 74:
        ct_isum_unit(t, f, q)
    comptime if OP == 75:
        ct_pairs_unit(t, f, q)
    comptime if OP == 76:
        ct_ent_unit(t, f, q)
    comptime if OP == 77:
        ff_chunk_unit(t, f, q)
    comptime if OP == 78:
        ff_fin_unit(t, f, q)
    comptime if OP == 79:
        mi_cell_unit(t, f, q)
    comptime if OP == 80:
        emi_cell_unit(t, f, q)
    comptime if OP == 81:
        approx_mode_unit(t, f, q)
