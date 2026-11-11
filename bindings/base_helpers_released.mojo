# SPDX-License-Identifier: Apache-2.0
"""The base helper table calls of `core.base_helpers_calls` with the GIL released
(lane perf-audit-20261111). The former Python exports these calls replaced
released the GIL around their data-sized work; the table calls in `core/`
cannot (core is Python-free), so the doors import these same-named wrappers
instead and CV unit threads and other Python threads keep running during the
work. Every argument is a plain word: nothing here reads a Python object
while the GIL is released. The caller holds the GIL on entry."""
from std.python import Python
from std.python._cpython import GILReleased
from core.hotpath_impl import HpStat
from core.base_entry_table import BaseEntryKey
from core.base_helpers_calls import (
    base_check_indices_i64 as _base_check_indices_i64,
    base_bincount_i64 as _base_bincount_i64,
    base_reduce_stat as _base_reduce_stat,
    base_reduce_stat_int as _base_reduce_stat_int,
    base_equal_elements as _base_equal_elements,
    base_bincount2_i32 as _base_bincount2_i32,
    base_row_means_f64 as _base_row_means_f64,
    base_assign_fold_i64 as _base_assign_fold_i64,
    base_count_fold_hits_i64 as _base_count_fold_hits_i64,
    base_mark_split_roles_f32 as _base_mark_split_roles_f32,
)


def base_check_indices_i64(key: BaseEntryKey, idx: Int, n: Int, bound: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_check_indices_i64(key, idx, n, bound)
    return result


def base_bincount_i64(key: BaseEntryKey, src: Int, code: Int, n: Int, k: Int, counts: Int, accumulate: Bool) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_bincount_i64(key, src, code, n, k, counts, accumulate)
    return result


def base_reduce_stat(key: BaseEntryKey, src: Int, code: Int, n: Int, what: Int) raises -> HpStat:
    var result = HpStat(0, 0, 0.0, 0, 0, 0)
    with GILReleased(Python()):
        result = _base_reduce_stat(key, src, code, n, what)
    return result^


def base_reduce_stat_int(key: BaseEntryKey, src: Int, code: Int, n: Int, what: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_reduce_stat_int(key, src, code, n, what)
    return result


def base_equal_elements(key: BaseEntryKey, a: Int, b: Int, code: Int, n: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_equal_elements(key, a, b, code, n, dst)
    return result


def base_bincount2_i32(key: BaseEntryKey, a: Int, b: Int, n: Int, ka: Int, kb: Int, counts: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_bincount2_i32(key, a, b, n, ka, kb, counts)
    return result


def base_row_means_f64(key: BaseEntryKey, src: Int, rows: Int, cols: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_row_means_f64(key, src, rows, cols, dst)
    return result


def base_assign_fold_i64(key: BaseEntryKey, idx: Int, m: Int, n: Int, fold: Int, folds: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_assign_fold_i64(key, idx, m, n, fold, folds)
    return result


def base_count_fold_hits_i64(key: BaseEntryKey, idx: Int, m: Int, folds: Int, n: Int, fold: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_count_fold_hits_i64(key, idx, m, folds, n, fold)
    return result


def base_mark_split_roles_f32(key: BaseEntryKey, idx: Int, m: Int, n: Int, split: Int, role: Int, roles: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _base_mark_split_roles_f32(key, idx, m, n, split, role, roles)
    return result
