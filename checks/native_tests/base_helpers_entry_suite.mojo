# SPDX-License-Identifier: Apache-2.0
"""The base helper table (core/base_entry_table.mojo), reached natively
(lane calltable-base-helpers, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication the core host binding makes
at module init (`publish_base_helpers_host`), a key, a schema and a span
mismatch refuse. Then one case per published operation, each through
`core/base_helpers_calls.mojo` and the published pointer, with literal
expected values or the run's own refusal text (the text its former Python
export raised). No Python participates.

The cases are independent units and run across the native host worker pool
(the host column publishes the host entry; the device entry's runs are
exercised by the GPU columns' base binding). Run in a fresh process (the
first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.base_entry_table import BaseEntryKey, base_table_entry, BASE_HELPERS_ABI, BH_OP_COUNT, BH_CHECK_INDICES_I64
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import (
    base_helpers_ref, base_helper, base_check_indices_i64, base_bincount_i64, base_reduce_stat, base_reduce_stat_int,
    base_equal_elements, base_bincount2_i32, base_row_means_f64, base_assign_fold_i64,
    base_count_fold_hits_i64, base_mark_split_roles_f32,
)
from core.hotpath_impl import HP_STAT_INT, HP_STAT_FLOAT, HP_STAT_TUPLE

comptime BASE_HELPER_CASES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> BaseEntryKey:
    return BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def precondition_unpublished() raises:
    var idx: List[Int64] = [0]
    try:
        _ = base_check_indices_i64(base_helpers_ref(_key()), Int(idx.unsafe_ptr()), 1, 1)
    except problem:
        _refuses(problem, "provider is not published")
        _ = idx
        return
    raise Error("an unpublished base helper table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = base_helpers_ref(_key())
    var second = base_helpers_ref(_key())
    require(entry_ref_same(first, second), "base helpers: a second resolution returned another entry")
    try:
        _ = base_table_entry(_key(), 0, String(BASE_HELPERS_ABI), BH_OP_COUNT + 1)
        raise Error("base helpers: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "base helpers: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    var idx: List[Int64] = [0]
    try:
        _ = base_check_indices_i64(base_helpers_ref(BaseEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)), Int(idx.unsafe_ptr()), 1, 1)
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = base_table_entry(_key(), 0, String("mojolearn.base.other.v1"), BH_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        var addrs: Array[Int, 1] = [Int(idx.unsafe_ptr())]
        var ints: Array[Int, 1] = [1]
        _ = base_helper(base_helpers_ref(_key()), BH_CHECK_INDICES_I64, addrs, ints)
        raise Error("a short request did not refuse")
    except problem:
        _refuses(problem, "request spans do not match")
    try:
        var none: Array[Int, 0] = []
        _ = base_helper(base_helpers_ref(_key()), BH_OP_COUNT, none, none)
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    _ = idx


def run_case(unit: Int) raises:
    var k = base_helpers_ref(_key())
    if unit == 0:
        var ok: List[Int64] = [0, 2, 1]
        var out: List[Int64] = [0, 3]
        var dup: List[Int64] = [1, 1]
        require(base_check_indices_i64(k, Int(ok.unsafe_ptr()), 3, 3) == 0, "check_indices distinct")
        require(base_check_indices_i64(k, Int(out.unsafe_ptr()), 2, 3) == 1, "check_indices range")
        require(base_check_indices_i64(k, Int(dup.unsafe_ptr()), 2, 3) == 2, "check_indices duplicate")
        _ = ok
        _ = out
        _ = dup
    elif unit == 1:
        var src: List[Int64] = [0, 2, 2]
        var counts: List[Int64] = [9, 9, 9]
        _ = base_bincount_i64(k, Int(src.unsafe_ptr()), 3, 3, 3, Int(counts.unsafe_ptr()), False)
        require(counts[0] == 1 and counts[1] == 0 and counts[2] == 2, "bincount")
        _ = base_bincount_i64(k, Int(src.unsafe_ptr()), 3, 3, 3, Int(counts.unsafe_ptr()), True)
        require(counts[0] == 2 and counts[1] == 0 and counts[2] == 4, "bincount accumulate")
        var bad: List[Int64] = [3]
        try:
            _ = base_bincount_i64(k, Int(bad.unsafe_ptr()), 3, 1, 3, Int(counts.unsafe_ptr()), True)
            raise Error("bincount out of range did not refuse")
        except problem:
            _refuses(problem, "bincount_i64: value out of range")
        _ = src
        _ = bad
        _ = counts
    elif unit == 2:
        var ints: List[Int64] = [5, -1, 3]
        var floats: List[Float64] = [0.5, 0.25]
        var pair: List[Int64] = [1, 2]
        var a = Int(ints.unsafe_ptr())
        require(base_reduce_stat_int(k, a, 3, 3, 0) == -1, "reduce_stat min")
        require(base_reduce_stat_int(k, a, 3, 3, 1) == 5, "reduce_stat max")
        require(base_reduce_stat_int(k, a, 3, 3, 3) == 0, "reduce_stat argmax")
        var total = base_reduce_stat(k, Int(floats.unsafe_ptr()), 1, 2, 2)
        require(total.kind == HP_STAT_FLOAT and total.f == 0.75, "reduce_stat float sum")
        var exact = base_reduce_stat(k, Int(pair.unsafe_ptr()), 3, 2, 5)
        require(exact.kind == HP_STAT_TUPLE and exact.hi == 0 and exact.mid == 0 and exact.low == 3,
                "reduce_stat exact integer sum")
        var low = base_reduce_stat(k, a, 3, 3, 0)
        require(low.kind == HP_STAT_INT, "reduce_stat integer kind")
        _ = ints
        _ = floats
        _ = pair
    elif unit == 3:
        var x: List[Float64] = [1.0, bitcast[DType.float64](Int64(0x7FF8000000000000)), -0.0]
        var y: List[Float64] = [1.0, bitcast[DType.float64](Int64(0x7FF8000000000000)), 0.0]
        var dst: List[UInt8] = [7, 7, 7]
        _ = base_equal_elements(k, Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), 1, 3, Int(dst.unsafe_ptr()))
        require(dst[0] == 1 and dst[1] == 0 and dst[2] == 1, "equal_elements IEEE")
        _ = x
        _ = y
        _ = dst
    elif unit == 4:
        var a: List[Int32] = [0, 1, 1]
        var b: List[Int32] = [0, 0, 1]
        var counts: List[Int64] = [9, 9, 9, 9]
        _ = base_bincount2_i32(k, Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), 3, 2, 2, Int(counts.unsafe_ptr()))
        require(counts[0] == 1 and counts[1] == 1 and counts[2] == 0 and counts[3] == 1, "bincount2")
        _ = a
        _ = b
        _ = counts
    elif unit == 5:
        var src: List[Float64] = [1.0, 2.0, 3.0, 4.0]
        var dst: List[Float64] = [0.0, 0.0]
        require(base_row_means_f64(k, Int(src.unsafe_ptr()), 2, 2, Int(dst.unsafe_ptr())) == 0, "row_means status")
        require(dst[0] == 1.5 and dst[1] == 3.5, "row_means")
        _ = src
        _ = dst
    elif unit == 6:
        var idx: List[Int64] = [0, 2]
        var again: List[Int64] = [2]
        var folds: List[Int32] = [-1, -1, -1]
        require(base_assign_fold_i64(k, Int(idx.unsafe_ptr()), 2, 3, 1, Int(folds.unsafe_ptr())) == 0, "assign_fold")
        require(folds[0] == 1 and folds[1] == -1 and folds[2] == 1, "assign_fold words")
        require(base_assign_fold_i64(k, Int(again.unsafe_ptr()), 1, 3, 0, Int(folds.unsafe_ptr())) == 2,
                "assign_fold overlap")
        _ = idx
        _ = again
        _ = folds
    elif unit == 7:
        var idx: List[Int64] = [0, 1, 2]
        var folds: List[Int32] = [1, -1, 1]
        require(base_count_fold_hits_i64(k, Int(idx.unsafe_ptr()), 3, Int(folds.unsafe_ptr()), 3, 1) == 2,
                "count_fold_hits")
        _ = idx
        _ = folds
    elif unit == 8:
        var idx: List[Int64] = [1]
        var roles: List[Float32] = [0.0, 0.0, 0.0, 0.0]
        require(base_mark_split_roles_f32(k, Int(idx.unsafe_ptr()), 1, 2, 1, 2, Int(roles.unsafe_ptr())) == 0,
                "mark_split_roles status")
        require(roles[3] == 2.0 and roles[0] == 0.0 and roles[1] == 0.0 and roles[2] == 0.0, "mark_split_roles")
        _ = idx
        _ = roles
    else:
        raise Error("no base helper case " + String(unit))


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_core_host makes at module init.
    publish_base_helpers_host()
    preconditions_published()
    var failures = List[Int](length=BASE_HELPER_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL base helper entry case", unit, String(problem))
    host_parallelize(task, BASE_HELPER_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "base helper entry cases failed: " + String(failed))
    print("PASS base helper entry", BASE_HELPER_CASES)
