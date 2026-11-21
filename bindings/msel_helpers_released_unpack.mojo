# SPDX-License-Identifier: Apache-2.0
"""The model-selection helper table calls of `core.msel_helpers_calls` with the GIL released
(lane perf-audit-20261111). The former Python exports these calls replaced
released the GIL around their data-sized work; the table calls in `core/`
cannot (core is Python-free), so the doors import these same-named wrappers
instead and CV unit threads and other Python threads keep running during the
work. Every argument is a plain word: nothing here reads a Python object
while the GIL is released. The caller holds the GIL on entry.

The splitters' twelve wrappers (fold ids, the fold and mask selections, the
gathers, first-seen codes, the stratified and group fold plans, the split
table, the leave range) were removed 2026-11-21: bindings/msel_split_plan_run.mojo
calls `core.msel_helpers_calls` directly inside one GIL release per split
operation (bindings/msel_split_door.mojo)."""
from std.python import Python
from std.python._cpython import GILReleased
from core.entry_table import EntryRef
from core.msel_helpers_calls import (
    msel_arange_i64 as _msel_arange_i64,
    msel_indices_overlap_i64 as _msel_indices_overlap_i64,
    msel_gather_i64 as _msel_gather_i64,
    msel_store_alloc as _msel_store_alloc,
    msel_store_read as _msel_store_read,
    msel_store_free as _msel_store_free,
    msel_scatter_rows as _msel_scatter_rows,
    msel_proba_column as _msel_proba_column,
    msel_store_free_bytes as _msel_store_free_bytes,
)


def msel_arange_i64(entry: EntryRef, dst: Int, start: Int, n: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_arange_i64(entry, dst, start, n)
    return result


def msel_indices_overlap_i64(entry: EntryRef, a: Int, na: Int, b: Int, nb: Int, bound: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_indices_overlap_i64(entry, a, na, b, nb, bound)
    return result


def msel_gather_i64(entry: EntryRef, table: Int, n_table: Int, codes: Int, n: Int, dst: Int) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_gather_i64(entry, table, n_table, codes, n, dst)
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


def msel_store_free_bytes(entry: EntryRef) raises -> Int:
    var result = 0
    with GILReleased(Python()):
        result = _msel_store_free_bytes(entry)
    return result
