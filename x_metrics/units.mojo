# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The op table: op id -> unit. python/mojolearn/_expansion_metrics.py `_OPS`
carries the same ids; the runners refuse an id outside the table."""
from x_metrics.common import FP, IP
from x_metrics.group import group_sort_unit, group_sum_unit, pair_key_unit
from x_metrics.ranking import bin_curve_unit, row_metric_unit
from x_metrics.regression import reg_term_unit, col_sort_unit, wpercentile_unit, col_max_unit

comptime N_OPS = 9


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
