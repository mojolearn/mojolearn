# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the base helpers through the base helper table
(core/base_entry_table.mojo; lane calltable-base-helpers, 2026-11-11). No
Python: a caller passes buffer addresses and plain words and keeps the
buffers alive until the call returns. The key is the one the caller's
bridge recorded at import (`python/mojolearn/_buffer.py::base_native_key`,
read at UNPACK by `bindings/base_helpers_key.mojo`).

The table is resolved once, at UNPACK (`base_helpers_ref`, called by
`bindings/base_helpers_key.mojo`), into an `EntryRef`; a missing or mismatched
publication refuses there with the table's error, and `base_helper` is one
indirect call through the ref (lane entry-table-zero-alloc, 2026-11-11). A run's refusal text is the
text its former Python export raised. These calls do not touch the GIL; a
door that wants other Python threads to run during a device helper
releases it around the call.
"""
from std.memory import bitcast
from core.base_entry_table import (
    BaseEntryKey, base_table_entry, BASE_HELPERS_ABI, BH_OP_COUNT, BH_STAT_WORDS,
    BH_CHECK_INDICES_I64, BH_BINCOUNT_I64, BH_REDUCE_STAT, BH_EQUAL_ELEMENTS, BH_BINCOUNT2_I32,
    BH_ROW_MEANS_F64, BH_ASSIGN_FOLD_I64, BH_COUNT_FOLD_HITS_I64, BH_MARK_SPLIT_ROLES_F32,
)
from core.hotpath_impl import HpStat
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word, entry_ref_unresolved
from core.shared_slot import SharedSlot


def base_helpers_ref(key: BaseEntryKey) raises -> EntryRef:
    """The base helper entry of the base binding the key names, resolved once."""
    return EntryRef(base_table_entry(key, 0, String(BASE_HELPERS_ABI), BH_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, BH_OP_COUNT))


def base_helper[NA: Int, NP: Int](entry: EntryRef, op: Int, addrs: Array[Int, NA], ints: Array[Int, NP]) raises -> Int:
    """One base helper operation through the resolved entry (no floats): one
    indirect call over the caller's inline request envelopes, no heap."""
    var floats: Array[Float64, 0] = []
    return entry_ref_call(entry, op, addrs, ints, floats)


def base_check_indices_i64(entry: EntryRef, idx: Int, n: Int, bound: Int) raises -> Int:
    """0 when the n int64 indices are in [0, bound) and distinct, 1 when one
    is out of range, else 2 (a duplicate)."""
    var addrs: Array[Int, 1] = [idx]
    var ints: Array[Int, 2] = [n, bound]
    return base_helper(entry, BH_CHECK_INDICES_I64, addrs, ints)


def base_bincount_i64(entry: EntryRef, src: Int, code: Int, n: Int, k: Int, counts: Int,
                      accumulate: Bool) raises -> Int:
    """counts[v] += 1 for the n int32 (code 2) or int64 (code 3) values
    (counts zeroed first unless accumulate); a value outside [0, k) raises."""
    var addrs: Array[Int, 2] = [src, counts]
    var ints: Array[Int, 4] = [code, n, k, 1 if accumulate else 0]
    return base_helper(entry, BH_BINCOUNT_I64, addrs, ints)


def base_reduce_stat(entry: EntryRef, src: Int, code: Int, n: Int, what: Int) raises -> HpStat:
    """`reduce_stat` (min 0, max 1, float sum 2, argmax 3, integral test 4,
    exact integer sum 5) as its typed answer."""
    var words = List[Int64](length=BH_STAT_WORDS, fill=0)
    var addrs: Array[Int, 2] = [src, Int(words.unsafe_ptr())]
    var ints: Array[Int, 3] = [code, n, what]
    var kind = base_helper(entry, BH_REDUCE_STAT, addrs, ints)
    var stat = HpStat(kind, Int(words[0]), bitcast[DType.float64](words[1]), Int(words[2]), Int(words[3]),
                      Int(words[4]))
    _ = words  # keep the out buffer alive past the call and the reads above
    return stat^


def base_reduce_stat_int(entry: EntryRef, src: Int, code: Int, n: Int, what: Int) raises -> Int:
    """`reduce_stat` for an integer answer (min/max of integers, argmax,
    the integral test)."""
    return base_reduce_stat(entry, src, code, n, what).i


def base_equal_elements(entry: EntryRef, a: Int, b: Int, code: Int, n: Int, dst: Int) raises -> Int:
    """dst[i] (uint8) = 1 where a[i] == b[i] (IEEE), one dtype code."""
    var addrs: Array[Int, 3] = [a, b, dst]
    var ints: Array[Int, 2] = [code, n]
    return base_helper(entry, BH_EQUAL_ELEMENTS, addrs, ints)


def base_bincount2_i32(entry: EntryRef, a: Int, b: Int, n: Int, ka: Int, kb: Int, counts: Int) raises -> Int:
    """int64 counts[b * ka + a] over the n rows of two int32 code columns."""
    var addrs: Array[Int, 3] = [a, b, counts]
    var ints: Array[Int, 3] = [n, ka, kb]
    return base_helper(entry, BH_BINCOUNT2_I32, addrs, ints)


def base_row_means_f64(entry: EntryRef, src: Int, rows: Int, cols: Int, dst: Int) raises -> Int:
    """fsum(row) / cols per row of a C-order float64 table; returns the
    number of rows holding +inf and -inf."""
    var addrs: Array[Int, 2] = [src, dst]
    var ints: Array[Int, 2] = [rows, cols]
    return base_helper(entry, BH_ROW_MEANS_F64, addrs, ints)


def base_assign_fold_i64(entry: EntryRef, idx: Int, m: Int, n: Int, fold: Int, folds: Int) raises -> Int:
    var addrs: Array[Int, 2] = [idx, folds]
    var ints: Array[Int, 3] = [m, n, fold]
    return base_helper(entry, BH_ASSIGN_FOLD_I64, addrs, ints)


def base_count_fold_hits_i64(entry: EntryRef, idx: Int, m: Int, folds: Int, n: Int, fold: Int) raises -> Int:
    var addrs: Array[Int, 2] = [idx, folds]
    var ints: Array[Int, 3] = [m, n, fold]
    return base_helper(entry, BH_COUNT_FOLD_HITS_I64, addrs, ints)


def base_mark_split_roles_f32(entry: EntryRef, idx: Int, m: Int, n: Int, split: Int, role: Int,
                              roles: Int) raises -> Int:
    var addrs: Array[Int, 2] = [idx, roles]
    var ints: Array[Int, 4] = [m, n, split, role]
    return base_helper(entry, BH_MARK_SPLIT_ROLES_F32, addrs, ints)


struct _BaseHelpersHeld(Defaultable, Movable):
    """The base helper entry of the process's one base binary, resolved once
    (round2-fixes-2): the key it was resolved for and the ref."""
    var ready: Bool
    var mode: Int
    var vendor: Int
    var entry: EntryRef

    def __init__(out self):
        self.ready = False
        self.mode = -1
        self.vendor = -1
        self.entry = entry_ref_unresolved()


comptime _BASE_HELPERS_HELD = SharedSlot[_BaseHelpersHeld, "MojolearnBaseHelpersHeldV1"]


def base_helpers_held(key: BaseEntryKey) raises -> EntryRef:
    """`base_helpers_ref(key)` resolved once per process and held: every
    bridge records the same key (`python/mojolearn/_buffer.py::base_native_key`,
    the one base binary), so the first door call resolves and admits the
    table (publication, key, ABI schema) and later calls take the held ref.
    The caller holds the GIL (it has just read the key from its helper dict),
    which serializes the first fill. A refusal is not held: the next call
    refuses again with the same text. perf audit 3: each public operation
    re-resolved the table (a slot read, a schema String built and compared)."""
    var held = _BASE_HELPERS_HELD.get_or_create_ptr()
    if held[].ready and held[].mode == key.mode and held[].vendor == key.vendor:
        return held[].entry
    var entry = base_helpers_ref(key)
    held[].entry = entry
    held[].mode = key.mode
    held[].vendor = key.vendor
    held[].ready = True
    return entry
