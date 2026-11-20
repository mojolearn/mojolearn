# SPDX-License-Identifier: Apache-2.0
"""The host column of the base helper table (core/base_entry_table.mojo):
the core host binding's (`bindings/_mojolearn_core_host.mojo`) helpers as
typed runs, its one `abi("C")` entry, and its publication (lane
calltable-base-helpers, 2026-11-11).

Each run is the body the host Python export of the same name ran
(`bindings/hotpath_helpers.mojo`, `bindings/array_helpers.mojo`) after its
arguments were unpacked: the same core function, checks and refusals. The
GPU base binding's runs (`core/base_helpers_device.mojo`) fall back to these
wherever their device twin does not cover a call, as the device Python
exports fell back to the host exports. No Python object crosses; the entry
is called with the GIL released by its caller.

The dispatch is a fixed comparison chain on the operation code, no data
work; each run's own parallelism is its core function's.
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.array_helpers_impl import row_means_f64
from core.base_entry_table import (
    BaseEntryKey, base_table_publish, base_check_spans, base_word, base_entry_vendor,
    BASE_HELPERS_ABI, BH_OP_COUNT,
    BH_CHECK_INDICES_I64, BH_BINCOUNT_I64, BH_REDUCE_STAT, BH_EQUAL_ELEMENTS, BH_BINCOUNT2_I32,
    BH_ROW_MEANS_F64, BH_ASSIGN_FOLD_I64, BH_COUNT_FOLD_HITS_I64, BH_MARK_SPLIT_ROLES_F32,
)
from core.hotpath_impl import HpStat, check_indices_i64, equal_elements, reduce_stat
from core.hotpath_impl_rows import (
    assign_fold_i64, bincount2_i32, bincount_i64, count_fold_hits_i64, mark_split_roles_f32,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot


def base_stat_store(stat: HpStat, out_addr: Int) raises -> Int:
    """Write a `reduce_stat` answer to the five-word out buffer of
    BH_REDUCE_STAT and return its kind."""
    if out_addr == 0:
        raise Error("reduce_stat: null result buffer")
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_addr)
    out[0] = Int64(stat.i)
    out[1] = bitcast[DType.int64](stat.f)
    out[2] = Int64(stat.hi)
    out[3] = Int64(stat.mid)
    out[4] = Int64(stat.low)
    return stat.kind


def bincount2_i32_host_run(a: Int, b: Int, count: Int, ka: Int, kb: Int, ca: Int) raises -> Int:
    """`bincount2_i32` on the host: dims [n, ka, kb], int64
    counts[b * ka + a]; raises when a code is outside [0, ka) x [0, kb)."""
    if ca == 0:
        raise Error("bincount2_i32: null counts address")
    if count > 0 and (a == 0 or b == 0):
        raise Error("bincount2_i32: null code address")
    return bincount2_i32(a, b, count, ka, kb, ca)


def row_means_f64_run(src: Int, rows: Int, cols: Int, dst: Int) raises -> Int:
    """`fsum(row) / cols` per row of a C-order float64 table; returns the
    number of rows holding +inf and -inf (the caller raises fsum's error)."""
    if rows <= 0 or cols <= 0:
        return 0
    if src == 0 or dst == 0:
        raise Error("mojolearn: null buffer address")
    return row_means_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=src),
                         MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst), rows, cols)


def base_helpers_host_dispatch(op: Int, aa: Int, pp: Int) raises -> Int:
    """One base helper by its `BH_*` code over admitted spans."""
    if op == BH_CHECK_INDICES_I64:
        return check_indices_i64(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1))
    if op == BH_BINCOUNT_I64:
        return bincount_i64(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), base_word(pp, 2),
                            base_word(aa, 1), base_word(pp, 3) != 0)
    if op == BH_REDUCE_STAT:
        return base_stat_store(reduce_stat(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), base_word(pp, 2)),
                               base_word(aa, 1))
    if op == BH_EQUAL_ELEMENTS:
        return equal_elements(base_word(aa, 0), base_word(aa, 1), base_word(pp, 0), base_word(pp, 1),
                              base_word(aa, 2))
    if op == BH_BINCOUNT2_I32:
        return bincount2_i32_host_run(base_word(aa, 0), base_word(aa, 1), base_word(pp, 0), base_word(pp, 1),
                                      base_word(pp, 2), base_word(aa, 2))
    if op == BH_ROW_MEANS_F64:
        return row_means_f64_run(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), base_word(aa, 1))
    if op == BH_ASSIGN_FOLD_I64:
        return assign_fold_i64(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), Int32(base_word(pp, 2)),
                               base_word(aa, 1))
    if op == BH_COUNT_FOLD_HITS_I64:
        return count_fold_hits_i64(base_word(aa, 0), base_word(pp, 0), base_word(aa, 1), base_word(pp, 1),
                                   Int32(base_word(pp, 2)))
    if op == BH_MARK_SPLIT_ROLES_F32:
        return mark_split_roles_f32(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), base_word(pp, 2),
                                    base_word(pp, 3), base_word(aa, 1))
    raise Error("base helpers: unknown operation " + String(op))


def base_helpers_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host base helper entry (`BASE_HELPERS_ABI`): addresses aa[0:na],
    Int words pp[0:np], no float words. Writes the run's answer to result; a
    refusal goes to sink."""
    try:
        base_check_spans(operation, aa, na, pp, np, nf, result)
        var value = base_helpers_host_dispatch(operation, aa, pp)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct BaseHelpersHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host base helper table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _HOST_OWNER_NAME = "MojolearnBaseHelpersHostOwnerV1"
comptime BASE_HELPERS_HOST_OWNER = SharedSlot[BaseHelpersHostOwner, _HOST_OWNER_NAME]


def publish_base_helpers_host() raises:
    """Publish the core host binding's base helper entry at module init. A
    repeated init publishes the same pointer; another image with the same
    key and another pointer refuses (core/entry_table.mojo)."""
    var owner = BASE_HELPERS_HOST_OWNER.get_or_create_ptr()
    var key = BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), base_entry_vendor("cpu"))
    base_table_publish(key, Int(owner), [String(BASE_HELPERS_ABI)], [BH_OP_COUNT], [base_helpers_host_v1])
