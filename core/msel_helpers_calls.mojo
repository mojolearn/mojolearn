# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the model-selection helpers through the
model-selection helper table (core/msel_entry_table.mojo; lane
calltable-msel, 2026-11-11). No Python: a caller passes buffer addresses and
plain words and keeps the buffers alive until the call returns. The key is
the one the model-selection bridge recorded at import
(`python/mojolearn/_native_msel_inputs.py`, read at UNPACK by
`bindings/msel_helpers_key.mojo`).

The table is resolved once, at UNPACK (`msel_helpers_ref`, called by
`bindings/msel_helpers_key.mojo`), into an `EntryRef`; a missing or mismatched
publication refuses there with the table's error, and `msel_helper` is one
indirect call through the ref (lane entry-table-zero-alloc, 2026-11-11). A run's refusal text is the
text its former Python export raised. These calls do not touch the GIL.
"""
from core.msel_entry_table import (
    MselEntryKey, msel_table_entry, MSEL_HELPERS_ABI, MH_OP_COUNT,
    MH_ARANGE_I64, MH_COUNT_MASK_U8, MH_SELECT_MASK_U8_I64, MH_MASK_FROM_INDICES_U8, MH_LEAVE_RANGE_I64,
    MH_INDICES_OVERLAP_I64, MH_FOLD_IDS, MH_SELECT_FOLD_I64, MH_SELECT_FOLDS_I64, MH_GATHER_I32, MH_GATHER_I64,
    MH_FIRST_SEEN_I32, MH_STRAT_FOLD_ASSIGN_I32, MH_STORE_ALLOC, MH_STORE_READ, MH_STORE_FREE, MH_SCATTER_ROWS,
    MH_PROBA_COLUMN, MH_SPLIT_TABLE_I32, MH_GROUP_FOLD_PERM_I32, MH_STORE_FREE_BYTES,
)
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word


def msel_helpers_ref(key: MselEntryKey) raises -> EntryRef:
    """The model-selection helper entry the key names, resolved once."""
    return EntryRef(msel_table_entry(key, 0, String(MSEL_HELPERS_ABI), MH_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, MH_OP_COUNT))


def msel_helper[NA: Int, NP: Int](entry: EntryRef, op: Int, addrs: Array[Int, NA], ints: Array[Int, NP]) raises -> Int:
    """One model-selection helper operation through the resolved entry: one
    indirect call over the caller's inline request envelopes, no heap."""
    var floats: Array[Float64, 0] = []
    return entry_ref_call(entry, op, addrs, ints, floats)


def msel_arange_i64(entry: EntryRef, dst: Int, start: Int, n: Int) raises -> Int:
    """dst[i] = start + i for i < n, int64."""
    var addrs: Array[Int, 1] = [dst]
    var ints: Array[Int, 2] = [start, n]
    return msel_helper(entry, MH_ARANGE_I64, addrs, ints)


def msel_count_mask_u8(entry: EntryRef, mask: Int, n: Int) raises -> Int:
    """The number of nonzero bytes of the n-byte mask."""
    var addrs: Array[Int, 1] = [mask]
    var ints: Array[Int, 1] = [n]
    return msel_helper(entry, MH_COUNT_MASK_U8, addrs, ints)


def msel_select_mask_u8_i64(entry: EntryRef, mask: Int, n: Int, test: Int, train: Int) raises -> Int:
    """Ascending int64 rows whose mask byte is nonzero into `test`, the rest
    into `train` (a zero address for an empty side); returns the test
    count."""
    var addrs: Array[Int, 3] = [mask, test, train]
    var ints: Array[Int, 1] = [n]
    return msel_helper(entry, MH_SELECT_MASK_U8_I64, addrs, ints)


def msel_mask_from_indices_u8(entry: EntryRef, idx: Int, k: Int, n: Int, mask: Int) raises -> Int:
    """mask[r] = 1 for every int64 index r of the k at `idx`; returns the
    number of distinct rows set."""
    var addrs: Array[Int, 2] = [idx, mask]
    var ints: Array[Int, 2] = [k, n]
    return msel_helper(entry, MH_MASK_FROM_INDICES_U8, addrs, ints)


def msel_leave_range_i64(entry: EntryRef, n: Int, lo: Int, hi: Int, train: Int, test: Int) raises -> Int:
    """test = rows [lo, hi), train = rows [0, lo) then [hi, n), ascending
    int64. Returns hi - lo."""
    var addrs: Array[Int, 2] = [train, test]
    var ints: Array[Int, 3] = [n, lo, hi]
    return msel_helper(entry, MH_LEAVE_RANGE_I64, addrs, ints)


def msel_indices_overlap_i64(entry: EntryRef, a: Int, na: Int, b: Int, nb: Int, bound: Int) raises -> Int:
    """1 when two int64 index sets share a member, else 0."""
    var addrs: Array[Int, 2] = [a, b]
    var ints: Array[Int, 3] = [na, nb, bound]
    return msel_helper(entry, MH_INDICES_OVERLAP_I64, addrs, ints)


def msel_fold_ids(entry: EntryRef, codes: Int, n: Int, n_classes: Int, n_splits: Int, counts: Int,
                  folds: Int, fold_counts: Int) raises -> Int:
    """The unshuffled default folds as one int32 fold id per row (`codes` 0:
    KFold; else the stratified dealing with int64 class counts)."""
    var addrs: Array[Int, 4] = [codes, counts, folds, fold_counts]
    var ints: Array[Int, 3] = [n, n_classes, n_splits]
    return msel_helper(entry, MH_FOLD_IDS, addrs, ints)


def msel_select_fold_i64(entry: EntryRef, folds: Int, n: Int, fold: Int, test: Int, train: Int) raises -> Int:
    """Ascending int64 rows with fold id `fold` into `test`, every other row
    into `train`; returns the test count."""
    var addrs: Array[Int, 3] = [folds, test, train]
    var ints: Array[Int, 2] = [n, fold]
    return msel_helper(entry, MH_SELECT_FOLD_I64, addrs, ints)


def msel_select_folds_i64(entry: EntryRef, folds: Int, n: Int, k: Int, sizes: Int, tests: Int,
                          trains: Int) raises -> Int:
    """`select_fold_i64` of every fold 0..k-1 into the arrays whose int64
    addresses are tests[f] / trains[f]; -1 when every test count equals
    sizes[f], else the first fold that differs."""
    var addrs: Array[Int, 4] = [folds, sizes, tests, trains]
    var ints: Array[Int, 2] = [n, k]
    return msel_helper(entry, MH_SELECT_FOLDS_I64, addrs, ints)


def msel_gather_i32(entry: EntryRef, table: Int, n_table: Int, codes: Int, n: Int, dst: Int) raises -> Int:
    """dst[i] = table[codes[i]], int32 codes into an int32 table."""
    var addrs: Array[Int, 3] = [table, codes, dst]
    var ints: Array[Int, 2] = [n_table, n]
    return msel_helper(entry, MH_GATHER_I32, addrs, ints)


def msel_gather_i64(entry: EntryRef, table: Int, n_table: Int, codes: Int, n: Int, dst: Int) raises -> Int:
    """dst[i] = table[codes[i]], int64 codes into an int64 table."""
    var addrs: Array[Int, 3] = [table, codes, dst]
    var ints: Array[Int, 2] = [n_table, n]
    return msel_helper(entry, MH_GATHER_I64, addrs, ints)


def msel_first_seen_i32(entry: EntryRef, codes: Int, n: Int, k: Int, enc: Int, counts: Int) raises -> Int:
    """Renumber n int32 class codes in [0, k) by first appearance; returns
    the number of classes present."""
    var addrs: Array[Int, 3] = [codes, enc, counts]
    var ints: Array[Int, 2] = [n, k]
    return msel_helper(entry, MH_FIRST_SEEN_I32, addrs, ints)


def msel_strat_fold_assign_i32(entry: EntryRef, enc: Int, n: Int, k: Int, n_folds: Int, alloc: Int,
                               perms: Int, counts: Int, dst: Int) raises -> Int:
    """StratifiedKFold's test fold of every row (`perms` 0: unshuffled)."""
    var addrs: Array[Int, 5] = [enc, alloc, perms, counts, dst]
    var ints: Array[Int, 3] = [n, k, n_folds]
    return msel_helper(entry, MH_STRAT_FOLD_ASSIGN_I32, addrs, ints)


def msel_store_alloc(entry: EntryRef, nbytes: Int) raises -> Int:
    """A zeroed block of the publisher's fold store; returns its id."""
    var addrs: Array[Int, 0] = []
    var ints: Array[Int, 1] = [nbytes]
    return msel_helper(entry, MH_STORE_ALLOC, addrs, ints)


def msel_store_read(entry: EntryRef, id: Int, dst: Int, nbytes: Int) raises -> Int:
    """Copy nbytes of block `id` to host memory at `dst`."""
    var addrs: Array[Int, 1] = [dst]
    var ints: Array[Int, 2] = [id, nbytes]
    return msel_helper(entry, MH_STORE_READ, addrs, ints)


def msel_store_free(entry: EntryRef, id: Int) raises -> Int:
    """Release block `id`."""
    var addrs: Array[Int, 0] = []
    var ints: Array[Int, 1] = [id]
    return msel_helper(entry, MH_STORE_FREE, addrs, ints)


def msel_scatter_rows(entry: EntryRef, dst_id: Int, n_dst: Int, dst_code: Int, src: Int, src_code: Int,
                      n_src: Int, width: Int, idx_id: Int) raises -> Int:
    """Rows of a host buffer cast into rows idx of a stored int64 or float64
    block."""
    var addrs: Array[Int, 1] = [src]
    var ints: Array[Int, 7] = [dst_id, n_dst, dst_code, src_code, n_src, width, idx_id]
    return msel_helper(entry, MH_SCATTER_ROWS, addrs, ints)


def msel_proba_column(entry: EntryRef, src: Int, code: Int, n: Int, k: Int, col: Int, dst: Int) raises -> Int:
    """Column `col` of an n x k float32/float64 buffer as float32."""
    var addrs: Array[Int, 2] = [src, dst]
    var ints: Array[Int, 4] = [code, n, k, col]
    return msel_helper(entry, MH_PROBA_COLUMN, addrs, ints)


def msel_split_table_i32(entry: EntryRef, perm: Int, m: Int, n_test: Int, n_train: Int, counts: Int,
                         table: Int, sums: Int) raises -> Int:
    """GroupShuffleSplit's per-group side table and side row sums."""
    var addrs: Array[Int, 4] = [perm, counts, table, sums]
    var ints: Array[Int, 3] = [m, n_test, n_train]
    return msel_helper(entry, MH_SPLIT_TABLE_I32, addrs, ints)


def msel_group_fold_perm_i32(entry: EntryRef, counts: Int, m: Int, n_folds: Int, perm: Int, dst: Int,
                             sizes: Int) raises -> Int:
    """Shuffled GroupKFold: the permuted groups in n_folds nearly equal runs,
    int32 dst[g] and int64 sizes[f]."""
    var addrs: Array[Int, 4] = [counts, perm, dst, sizes]
    var ints: Array[Int, 2] = [m, n_folds]
    return msel_helper(entry, MH_GROUP_FOLD_PERM_I32, addrs, ints)


def msel_store_free_bytes(entry: EntryRef) raises -> Int:
    """Free bytes of the device the publisher's fold store lives on; -1 for
    the host column."""
    var none: Array[Int, 0] = []
    return msel_helper(entry, MH_STORE_FREE_BYTES, none, none)
