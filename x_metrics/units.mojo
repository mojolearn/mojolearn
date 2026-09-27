# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The op table: op id -> unit. python/mojolearn/_expansion_metrics.py `_OPS`
carries the same ids; the runners refuse an id outside the table."""
from x_metrics.common import FP, IP
from x_metrics.group import group_sort_unit, group_sum_unit, pair_key_unit

comptime N_OPS = 3


@always_inline
def run_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == 0:
        group_sort_unit(t, f, q)
    comptime if OP == 1:
        group_sum_unit(t, f, q)
    comptime if OP == 2:
        pair_key_unit(t, f, q)
