# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The op table: op id -> unit. python/mojolearn/_expansion_metrics.py `_OPS`
carries the same ids; the runners refuse an id outside the table."""
from x_metrics.common import FP, IP
from x_metrics.group import group_sort_unit, group_sum_unit, pair_key_unit
from x_metrics.ranking import bin_curve_unit, row_metric_unit
from x_metrics.cluster import row_centroid_dist_unit
from x_metrics.split import permute_unit
from x_metrics.regression import reg_term_unit, col_sort_unit, wpercentile_unit, col_max_unit, wpct_select
from x_metrics.par import (
    cs_hist_unit, cs_scan_rows_unit, cs_scan_groups_unit, cs_place_unit,
    fold_leaf_unit, fold_level_unit, fold_final_unit,
    sort_key_unit, sort_runs_unit, sort_merge_unit, sort_emit_unit,
    curve_gather_unit, curve_scan_unit, wpct_gather_unit, wpct_prefix_unit,
)

#: ops 0..10 are the caller's (x_metrics/plan.mojo N_USER_OPS); 11.. are the
#: planner's parallel schedules (x_metrics/par.mojo)
comptime N_OPS = 27


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
    comptime if OP == 23:
        curve_scan_unit(t, f, q)
    comptime if OP == 24:
        wpct_gather_unit(t, f, q)
    comptime if OP == 25:
        wpct_prefix_unit(t, f, q)
    comptime if OP == 26:
        wpct_select(t, f, q)
