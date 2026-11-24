# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The model-selection helper table (core/msel_entry_table.mojo), reached
natively (lane calltable-msel, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication the core host binding makes
at module init (`publish_msel_helpers_host`), a key, a schema and a span
mismatch refuse. Then one case per published operation, each through
`core/msel_helpers_calls.mojo` and the published pointer, with literal
expected values or the run's own refusal text (the text its former Python
export raised). No Python participates.

The cases are independent units and run across the native host worker pool
(the host column publishes the host entry; the device entry's runs are
exercised by the GPU columns' base binding, and the layout V3 stored-code
operations on both columns by `msel_split_stratified_res`). The fold-store
case and the layout V3 cases (10, 11, 18: each puts and frees its own
blocks) touch the process store, whose bookkeeping is locked. Run in a fresh
process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from checks.numerics import GLOBAL_NUMERIC_MODE
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.msel_convert import MSEL_F32
from core.msel_entry_table import (
    MselEntryKey, msel_table_entry, MSEL_HELPERS_ABI, MH_OP_COUNT, MH_ARANGE_I64, MH_RETIRED_5, MH_RETIRED_10,
    MH_RETIRED_11, MH_RETIRED_12, MH_RETIRED_13, MH_RETIRED_14, MH_RETIRED_16, MH_RETIRED_20,
)
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import (
    msel_helpers_ref, msel_helper, msel_arange_i64, msel_count_mask_u8, msel_select_mask_u8_i64, msel_mask_from_indices_u8,
    msel_leave_range_i64, msel_fold_ids, msel_select_fold_i64, msel_select_folds_i64,
    msel_gather_i32, msel_store_free, msel_proba_column, msel_split_table_i32,
    msel_group_fold_perm_i32, msel_store_put, msel_store_take_rows,
    msel_first_seen_res, msel_strat_fold_assign_res, msel_select_folds_res, msel_fold_ids_res,
)

comptime MSEL_HELPER_CASES = 19


def _key() -> MselEntryKey:
    return MselEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _a(values: List[Int64]) -> Int:
    return Int(values.unsafe_ptr())


def _a32(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def precondition_unpublished() raises:
    var out: List[Int64] = [0]
    try:
        _ = msel_arange_i64(msel_helpers_ref(_key()), _a(out), 0, 1)
    except problem:
        _refuses(problem, "provider is not published")
        _ = out
        return
    raise Error("an unpublished model-selection helper table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = msel_helpers_ref(_key())
    var second = msel_helpers_ref(_key())
    require(entry_ref_same(first, second), "model-selection helpers: a second resolution returned another entry")
    try:
        _ = msel_table_entry(_key(), 0, String(MSEL_HELPERS_ABI), MH_OP_COUNT + 1)
        raise Error("model-selection helpers: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "model-selection helpers: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    var out: List[Int64] = [0]
    try:
        _ = msel_arange_i64(msel_helpers_ref(MselEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)), _a(out), 0, 1)
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = msel_table_entry(_key(), 0, String("mojolearn.msel.other.v1"), MH_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        var addrs: Array[Int, 1] = [_a(out)]
        var ints: Array[Int, 1] = [0]
        _ = msel_helper(msel_helpers_ref(_key()), MH_ARANGE_I64, addrs, ints)
        raise Error("a short request did not refuse")
    except problem:
        _refuses(problem, "request spans do not match")
    try:
        var none: Array[Int, 0] = []
        _ = msel_helper(msel_helpers_ref(_key()), MH_OP_COUNT, none, none)
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    _ = out


def _operation(unit: Int) raises:
    var k = msel_helpers_ref(_key())
    if unit == 0:
        var out: List[Int64] = [0, 0, 0]
        require(msel_arange_i64(k, _a(out), 5, 3) == 0, "arange status")
        require(out[0] == 5 and out[1] == 6 and out[2] == 7, "arange_i64")
        _ = out
    elif unit == 1:
        var mask: List[UInt8] = [0, 3, 0, 1]
        require(msel_count_mask_u8(k, Int(mask.unsafe_ptr()), 4) == 2, "count_mask_u8")
        var test: List[Int64] = [9, 9]
        var train: List[Int64] = [9, 9]
        require(msel_select_mask_u8_i64(k, Int(mask.unsafe_ptr()), 4, _a(test), _a(train)) == 2, "select_mask count")
        require(test[0] == 1 and test[1] == 3 and train[0] == 0 and train[1] == 2, "select_mask_u8_i64")
        _ = mask
        _ = test
        _ = train
    elif unit == 2:
        var idx: List[Int64] = [2, 0, 2]
        var mask: List[UInt8] = [7, 7, 7, 7]
        require(msel_mask_from_indices_u8(k, _a(idx), 3, 4, Int(mask.unsafe_ptr())) == 2, "mask rows set")
        require(mask[0] == 1 and mask[1] == 0 and mask[2] == 1 and mask[3] == 0, "mask_from_indices_u8")
        var bad: List[Int64] = [4]
        try:
            _ = msel_mask_from_indices_u8(k, _a(bad), 1, 4, Int(mask.unsafe_ptr()))
            raise Error("mask_from_indices_u8 out of range did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
        _ = idx
        _ = bad
        _ = mask
    elif unit == 3:
        var train: List[Int64] = [9, 9]
        var test: List[Int64] = [9]
        require(msel_leave_range_i64(k, 3, 1, 2, _a(train), _a(test)) == 1, "leave_range width")
        require(test[0] == 1 and train[0] == 0 and train[1] == 2, "leave_range_i64")
        _ = train
        _ = test
    elif unit == 4:
        # Operation 5 (the train/test overlap test) is retired: explicit folds
        # whose sides share rows are accepted, as scikit-learn accepts them.
        try:
            var a: List[Int64] = [0, 1]
            var addrs: Array[Int, 2] = [_a(a), _a(a)]
            var ints: Array[Int, 3] = [2, 2, 3]
            _ = msel_helper(k, MH_RETIRED_5, addrs, ints)
            _ = a^  # lastuse-audit: the call reads it through an untracked address; Mojo frees a local at its last named use
            raise Error("the retired operation 5 did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
    elif unit == 5:
        var folds: List[Int32] = [9, 9, 9, 9, 9]
        var fold_counts: List[Int64] = [0, 0]
        var unused: List[Int64] = [0]
        require(msel_fold_ids(k, 0, 5, 0, 2, _a(unused), _a32(folds), _a(fold_counts)) == 0, "kfold status")
        require(folds[0] == 0 and folds[1] == 0 and folds[2] == 0 and folds[3] == 1 and folds[4] == 1, "kfold ids")
        require(fold_counts[0] == 3 and fold_counts[1] == 2, "kfold counts")
        var codes: List[Int32] = [0, 0, 1, 1]
        var strat: List[Int32] = [9, 9, 9, 9]
        var class_counts: List[Int64] = [0, 0]
        require(msel_fold_ids(k, _a32(codes), 4, 2, 2, _a(class_counts), _a32(strat), _a(fold_counts)) == 0,
                "stratified status")
        require(strat[0] == 0 and strat[1] == 1 and strat[2] == 0 and strat[3] == 1, "stratified ids")
        require(class_counts[0] == 2 and class_counts[1] == 2 and fold_counts[0] == 2 and fold_counts[1] == 2,
                "stratified counts")
        _ = folds
        _ = fold_counts
        _ = unused
        _ = codes
        _ = strat
        _ = class_counts
    elif unit == 6:
        var folds: List[Int32] = [0, 1, 0, 1]
        var test: List[Int64] = [9, 9]
        var train: List[Int64] = [9, 9]
        require(msel_select_fold_i64(k, _a32(folds), 4, 1, _a(test), _a(train)) == 2, "select_fold count")
        require(test[0] == 1 and test[1] == 3 and train[0] == 0 and train[1] == 2, "select_fold_i64")
        _ = folds
        _ = test
        _ = train
    elif unit == 7:
        var folds: List[Int32] = [0, 1, 0, 1]
        var t0: List[Int64] = [9, 9]
        var r0: List[Int64] = [9, 9]
        var t1: List[Int64] = [9, 9]
        var r1: List[Int64] = [9, 9]
        var tests: List[Int64] = [Int64(_a(t0)), Int64(_a(t1))]
        var trains: List[Int64] = [Int64(_a(r0)), Int64(_a(r1))]
        var sizes: List[Int64] = [2, 2]
        require(msel_select_folds_i64(k, _a32(folds), 4, 2, _a(sizes), _a(tests), _a(trains)) == -1, "plan agrees")
        require(t0[0] == 0 and t0[1] == 2 and r0[0] == 1 and r0[1] == 3, "fold 0 split")
        require(t1[0] == 1 and t1[1] == 3 and r1[0] == 0 and r1[1] == 2, "fold 1 split")
        var wrong: List[Int64] = [1, 3]
        require(msel_select_folds_i64(k, _a32(folds), 4, 2, _a(wrong), _a(tests), _a(trains)) == 0, "plan disagrees")
        _ = folds
        _ = t0
        _ = r0
        _ = t1
        _ = r1
        _ = tests
        _ = trains
        _ = sizes
        _ = wrong
    elif unit == 8:
        var table: List[Int32] = [7, -1, 5]
        var codes: List[Int32] = [2, 0, 1]
        var dst: List[Int32] = [0, 0, 0]
        _ = msel_gather_i32(k, _a32(table), 3, _a32(codes), 3, _a32(dst))
        require(dst[0] == 5 and dst[1] == 7 and dst[2] == -1, "gather_i32")
        var bad: List[Int32] = [3]
        try:
            _ = msel_gather_i32(k, _a32(table), 3, _a32(bad), 1, _a32(dst))
            raise Error("gather_i32 out of range did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
        _ = table
        _ = codes
        _ = bad
        _ = dst
    elif unit == 9:
        # 10 (gather_i64), 13 (store_alloc), 14 (store_read) and 16 (scatter_rows) were retired with
        # cross_val_predict's identity placement and learning_curve's per-fold gather (lane vdf-scope-cv),
        # 11 (first_seen_i32) and 12 (strat_fold_assign_i32) with the splitters' move to the stored-code
        # forms (lane vdf-scope-split): the reserved numbers refuse. Fail-proof: drop one of them from
        # msel_check_spans: that call answers.
        var retired: List[Int] = [MH_RETIRED_10, MH_RETIRED_11, MH_RETIRED_12, MH_RETIRED_13, MH_RETIRED_14,
                                  MH_RETIRED_16]
        for i in range(len(retired)):  # small-loop(the six retired operation numbers)
            try:
                var none: Array[Int, 0] = []
                _ = msel_helper(k, retired[i], none, none)
                raise Error("the retired operation " + String(retired[i]) + " did not refuse")
            except problem:
                require("did not refuse" not in String(problem), String(problem))
    elif unit == 10:
        # Layout V3: the stored codes' first-seen sizes and rank table (code 2 first, then 0, then 1).
        var codes: List[Int32] = [2, 0, 2, 1]
        var held = msel_store_put(k, _a32(codes), 16)
        var counts: List[Int64] = [0, 0, 0]
        var rank: List[Int32] = [9, 9, 9]
        require(msel_first_seen_res(k, held, 4, 3, _a(counts), _a32(rank)) == 3, "first_seen_res classes")
        require(counts[0] == 2 and counts[1] == 1 and counts[2] == 1, "first_seen_res counts")
        require(rank[0] == 1 and rank[1] == 2 and rank[2] == 0, "first_seen_res rank table")
        var absent: List[Int32] = [2, 2, 0]
        var absent_id = msel_store_put(k, _a32(absent), 12)
        require(msel_first_seen_res(k, absent_id, 3, 3, _a(counts), _a32(rank)) == 2, "first_seen_res present")
        # code 1 has no row: it takes the position after the present codes
        require(rank[0] == 1 and rank[1] == 2 and rank[2] == 0 and counts[2] == 0, "first_seen_res absent code")
        _ = msel_store_free(k, absent_id)
        _ = msel_store_free(k, held)
        _ = codes
        _ = absent
        _ = counts
        _ = rank
    elif unit == 11:
        # Layout V3: one class of two rows dealt over two folds, then each fold's rows out of the stored block;
        # the stored codes [1, 1] renumbered by the rank table [1, 0] are the first-seen codes [0, 0].
        var codes: List[Int32] = [1, 1]
        var held = msel_store_put(k, _a32(codes), 8)
        var rank: List[Int32] = [1, 0]
        var alloc: List[Int64] = [1, 1]
        var counts: List[Int64] = [2]
        var sizes: List[Int64] = [0, 0]
        # two words a side: the disagreeing plan below sizes fold 0's test side 2
        var t0: List[Int64] = [9, 9]
        var r0: List[Int64] = [9, 9]
        var t1: List[Int64] = [9, 9]
        var r1: List[Int64] = [9, 9]
        var tests: List[Int64] = [Int64(_a(t0)), Int64(_a(t1))]
        var trains: List[Int64] = [Int64(_a(r0)), Int64(_a(r1))]
        var folds = msel_strat_fold_assign_res(k, held, 2, 1, 2, _a32(rank), 2, _a(alloc), 0, _a(counts), _a(sizes))
        require(sizes[0] == 1 and sizes[1] == 1, "strat_fold_assign_res sizes")
        require(msel_select_folds_res(k, folds, 2, 2, _a(sizes), _a(tests), _a(trains)) == -1, "stored plan agrees")
        require(t0[0] == 0 and r0[0] == 1 and t1[0] == 1 and r1[0] == 0, "strat_fold_assign_res folds")
        var wrong: List[Int64] = [2, 0]
        require(msel_select_folds_res(k, folds, 2, 2, _a(wrong), _a(tests), _a(trains)) == 0, "stored plan disagrees")
        _ = msel_store_free(k, folds)
        var perm: List[Int64] = [1, 0]
        folds = msel_strat_fold_assign_res(k, held, 2, 1, 2, _a32(rank), 2, _a(alloc), _a(perm), _a(counts), _a(sizes))
        _ = msel_select_folds_res(k, folds, 2, 2, _a(sizes), _a(tests), _a(trains))
        require(t0[0] == 1 and t1[0] == 0, "strat_fold_assign_res permuted")
        _ = msel_store_free(k, folds)
        var short: List[Int64] = [3]
        try:
            _ = msel_strat_fold_assign_res(k, held, 2, 1, 2, _a32(rank), 2, _a(alloc), 0, _a(short), _a(sizes))
            raise Error("strat_fold_assign_res with counts that miss the rows did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
        _ = msel_store_free(k, held)
        _ = codes
        _ = rank
        _ = alloc
        _ = counts
        _ = sizes
        _ = t0
        _ = r0
        _ = t1
        _ = r1
        _ = tests
        _ = trains
        _ = wrong
        _ = perm
        _ = short
    elif unit == 12:
        # The only unit that touches the process fold store.
        # Layout V2: a resident copy of host rows and a gather of two of them.
        var rows: List[Float64] = [1.0, 2.0, 3.0]
        var picks: List[Int64] = [2, 0]
        var rows_id = msel_store_put(k, Int(rows.unsafe_ptr()), 24)
        var picks_id = msel_store_put(k, Int(picks.unsafe_ptr()), 16)
        var taken: List[Float64] = [9.0, 9.0]
        _ = msel_store_take_rows(k, rows_id, 3, 8, picks_id, 2, Int(taken.unsafe_ptr()))
        require(taken[0] == 3.0 and taken[1] == 1.0, "store put and take rows")
        try:
            _ = msel_store_take_rows(k, rows_id, 2, 8, picks_id, 2, Int(taken.unsafe_ptr()))
            raise Error("a row index past the source did not refuse")
        except problem:
            _refuses(problem, "row index out of bounds")
        _ = taken^  # keep alive: the call in the try above reads it through an untracked pointer
        _ = msel_store_free(k, picks_id)
        _ = msel_store_free(k, rows_id)
        _ = rows
        _ = picks
        try:
            _ = msel_store_free(k, rows_id)
            raise Error("a freed block did not refuse")
        except problem:
            _refuses(problem, "is not live")
    elif unit == 13:
        var src: List[Float32] = [0.25, 0.75, 0.5, 0.5]
        var dst: List[Float32] = [9.0, 9.0]
        _ = msel_proba_column(k, Int(src.unsafe_ptr()), MSEL_F32, 2, 2, 1, Int(dst.unsafe_ptr()))
        require(dst[0] == 0.75 and dst[1] == 0.5, "proba_column")
        try:
            _ = msel_proba_column(k, Int(src.unsafe_ptr()), MSEL_F32, 2, 2, 2, Int(dst.unsafe_ptr()))
            raise Error("a column outside k did not refuse")
        except problem:
            _refuses(problem, "msel_proba_column: bad sizes")
        _ = src
        _ = dst
    elif unit == 14:
        var perm: List[Int64] = [2, 0, 1]
        var counts: List[Int64] = [5, 6, 7]
        var table: List[Int32] = [9, 9, 9]
        var sums: List[Int64] = [0, 0]
        _ = msel_split_table_i32(k, _a(perm), 3, 1, 1, _a(counts), _a32(table), _a(sums))
        require(table[0] == 0 and table[1] == 2 and table[2] == 1, "split_table sides")
        require(sums[0] == 5 and sums[1] == 7, "split_table sums")
        _ = perm
        _ = counts
        _ = table
        _ = sums
    elif unit == 15:
        var counts: List[Int64] = [4, 5, 6]
        var perm: List[Int64] = [1, 2, 0]
        var dst: List[Int32] = [9, 9, 9]
        var sizes: List[Int64] = [0, 0]
        _ = msel_group_fold_perm_i32(k, _a(counts), 3, 2, _a(perm), _a32(dst), _a(sizes))
        require(dst[0] == 1 and dst[1] == 0 and dst[2] == 0, "group_fold_perm folds")
        require(sizes[0] == 11 and sizes[1] == 4, "group_fold_perm sizes")
        var bad: List[Int64] = [3, 0, 1]
        try:
            _ = msel_group_fold_perm_i32(k, _a(counts), 3, 2, _a(bad), _a32(dst), _a(sizes))
            raise Error("a group outside m did not refuse")
        except problem:
            _refuses(problem, "a permuted group is out of range")
        _ = counts
        _ = perm
        _ = bad
        _ = dst
        _ = sizes
    elif unit == 16:
        # 20 was MH_STORE_FREE_BYTES (retired with the cv units, lane mf-r9b): the reserved number refuses.
        # Fail-proof: drop `or operation == MH_RETIRED_20` from msel_check_spans: the call answers.
        try:
            var none: Array[Int, 0] = []
            _ = msel_helper(k, MH_RETIRED_20, none, none)
            raise Error("the retired operation 20 did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
    elif unit == 17:
        try:
            var dst: List[Int64] = [0]
            _ = msel_leave_range_i64(k, 3, 2, 1, _a(dst), _a(dst))
            _ = dst^  # lastuse-audit: the call reads it through an untracked address; Mojo frees a local at its last named use
            raise Error("an inverted leave range did not refuse")
        except problem:
            require("did not refuse" not in String(problem), String(problem))
    elif unit == 18:
        # Layout V3: the stratified default folds of stored codes into a stored fold block.
        var codes: List[Int32] = [0, 0, 1, 1]
        var held = msel_store_put(k, _a32(codes), 16)
        var class_counts: List[Int64] = [0, 0]
        var fold_counts: List[Int64] = [0, 0]
        var folds = msel_fold_ids_res(k, held, 4, 2, 2, _a(class_counts), _a(fold_counts))
        _ = msel_store_free(k, held)
        require(class_counts[0] == 2 and class_counts[1] == 2 and fold_counts[0] == 2 and fold_counts[1] == 2,
                "fold_ids_res counts")
        var t0: List[Int64] = [9, 9]
        var r0: List[Int64] = [9, 9]
        var t1: List[Int64] = [9, 9]
        var r1: List[Int64] = [9, 9]
        var tests: List[Int64] = [Int64(_a(t0)), Int64(_a(t1))]
        var trains: List[Int64] = [Int64(_a(r0)), Int64(_a(r1))]
        require(msel_select_folds_res(k, folds, 4, 2, _a(fold_counts), _a(tests), _a(trains)) == -1,
                "fold_ids_res plan agrees")
        require(t0[0] == 0 and t0[1] == 2 and r0[0] == 1 and r0[1] == 3, "fold_ids_res fold 0")
        require(t1[0] == 1 and t1[1] == 3 and r1[0] == 0 and r1[1] == 2, "fold_ids_res fold 1")
        _ = msel_store_free(k, folds)
        try:
            _ = msel_select_folds_res(k, folds, 4, 2, _a(fold_counts), _a(tests), _a(trains))
            raise Error("a freed fold block did not refuse")
        except problem:
            _refuses(problem, "is not live")
        _ = codes
        _ = class_counts
        _ = fold_counts
        _ = t0
        _ = r0
        _ = t1
        _ = r1
        _ = tests
        _ = trains
    else:
        raise Error("no model-selection helper case " + String(unit))


def test_unpublished_then_published_operations() raises:
    """One test, in this order: the first call needs the process's table unpublished, so the preconditions,
    the publication and the operations cannot be separate tests (TestSuite may run any one alone)."""
    precondition_unpublished()
    # The publication _mojolearn_core_host makes at module init.
    publish_msel_helpers_host()
    preconditions_published()
    check_cases("operation", MSEL_HELPER_CASES, _operation)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
