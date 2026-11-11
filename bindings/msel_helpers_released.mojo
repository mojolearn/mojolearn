# SPDX-License-Identifier: Apache-2.0
"""The model-selection helper table calls of `core.msel_helpers_calls` with the GIL released
(lane perf-audit-20261111). The former Python exports these calls replaced
released the GIL around their data-sized work; the table calls in `core/`
cannot (core is Python-free), so the doors import these same-named wrappers
instead and CV unit threads and other Python threads keep running during the
work. Every argument is a plain word: nothing here reads a Python object
while the GIL is released. The caller holds the GIL on entry."""
from std.python import Python
from std.python._cpython import GILReleased
from core.entry_table import EntryRef
from core.msel_helpers_calls import (
    msel_arange_i64 as _msel_arange_i64,
    msel_count_mask_u8 as _msel_count_mask_u8,
    msel_select_mask_u8_i64 as _msel_select_mask_u8_i64,
    msel_mask_from_indices_u8 as _msel_mask_from_indices_u8,
    msel_leave_range_i64 as _msel_leave_range_i64,
    msel_indices_overlap_i64 as _msel_indices_overlap_i64,
    msel_fold_ids as _msel_fold_ids,
    msel_select_fold_i64 as _msel_select_fold_i64,
    msel_select_folds_i64 as _msel_select_folds_i64,
    msel_gather_i32 as _msel_gather_i32,
    msel_gather_i64 as _msel_gather_i64,
    msel_first_seen_i32 as _msel_first_seen_i32,
    msel_strat_fold_assign_i32 as _msel_strat_fold_assign_i32,
    msel_store_alloc as _msel_store_alloc,
    msel_store_read as _msel_store_read,
    msel_store_free as _msel_store_free,
    msel_scatter_rows as _msel_scatter_rows,
    msel_proba_column as _msel_proba_column,
    msel_split_table_i32 as _msel_split_table_i32,
    msel_group_fold_perm_i32 as _msel_group_fold_perm_i32,
    msel_store_free_bytes as _msel_store_free_bytes,
)


def msel_arange_i64(entry: EntryRef, dst: Int, start: Int, n: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_arange_i64(entry, dst, start, n)
    return result


def msel_count_mask_u8(entry: EntryRef, mask: Int, n: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_count_mask_u8(entry, mask, n)
    return result


def msel_select_mask_u8_i64(entry: EntryRef, mask: Int, n: Int, test: Int, train: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_select_mask_u8_i64(entry, mask, n, test, train)
    return result


def msel_mask_from_indices_u8(entry: EntryRef, idx: Int, k: Int, n: Int, mask: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_mask_from_indices_u8(entry, idx, k, n, mask)
    return result


def msel_leave_range_i64(entry: EntryRef, n: Int, lo: Int, hi: Int, train: Int, test: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_leave_range_i64(entry, n, lo, hi, train, test)
    return result


def msel_indices_overlap_i64(entry: EntryRef, a: Int, na: Int, b: Int, nb: Int, bound: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_indices_overlap_i64(entry, a, na, b, nb, bound)
    return result


def msel_fold_ids(entry: EntryRef, codes: Int, n: Int, n_classes: Int, n_splits: Int, counts: Int, folds: Int,
                  fold_counts: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_fold_ids(entry, codes, n, n_classes, n_splits, counts, folds, fold_counts)
    return result


def msel_select_fold_i64(entry: EntryRef, folds: Int, n: Int, fold: Int, test: Int, train: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_select_fold_i64(entry, folds, n, fold, test, train)
    return result


def msel_select_folds_i64(entry: EntryRef, folds: Int, n: Int, k: Int, sizes: Int, tests: Int, trains: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_select_folds_i64(entry, folds, n, k, sizes, tests, trains)
    return result


def msel_gather_i32(entry: EntryRef, table: Int, n_table: Int, codes: Int, n: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_gather_i32(entry, table, n_table, codes, n, dst)
    return result


def msel_gather_i64(entry: EntryRef, table: Int, n_table: Int, codes: Int, n: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_gather_i64(entry, table, n_table, codes, n, dst)
    return result


def msel_first_seen_i32(entry: EntryRef, codes: Int, n: Int, k: Int, enc: Int, counts: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_first_seen_i32(entry, codes, n, k, enc, counts)
    return result


def msel_strat_fold_assign_i32(entry: EntryRef, enc: Int, n: Int, k: Int, n_folds: Int, alloc: Int, perms: Int,
                               counts: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_strat_fold_assign_i32(entry, enc, n, k, n_folds, alloc, perms, counts, dst)
    return result


def msel_store_alloc(entry: EntryRef, nbytes: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_store_alloc(entry, nbytes)
    return result


def msel_store_read(entry: EntryRef, id: Int, dst: Int, nbytes: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_store_read(entry, id, dst, nbytes)
    return result


def msel_store_free(entry: EntryRef, id: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_store_free(entry, id)
    return result


def msel_scatter_rows(entry: EntryRef, dst_id: Int, n_dst: Int, dst_code: Int, src: Int, src_code: Int,
                      n_src: Int, width: Int, idx_id: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_scatter_rows(entry, dst_id, n_dst, dst_code, src, src_code, n_src, width, idx_id)
    return result


def msel_proba_column(entry: EntryRef, src: Int, code: Int, n: Int, k: Int, col: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_proba_column(entry, src, code, n, k, col, dst)
    return result


def msel_split_table_i32(entry: EntryRef, perm: Int, m: Int, n_test: Int, n_train: Int, counts: Int, table: Int,
                         sums: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_split_table_i32(entry, perm, m, n_test, n_train, counts, table, sums)
    return result


def msel_group_fold_perm_i32(entry: EntryRef, counts: Int, m: Int, n_folds: Int, perm: Int, dst: Int, sizes: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_group_fold_perm_i32(entry, counts, m, n_folds, perm, dst, sizes)
    return result


def msel_store_free_bytes(entry: EntryRef) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_store_free_bytes(entry)
    return result
