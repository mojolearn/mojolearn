# SPDX-License-Identifier: Apache-2.0
"""The device column of the base helper table (core/base_entry_table.mojo):
the GPU base binding's (`bindings/_mojolearn.mojo`) helpers as typed runs,
its one `abi("C")` entry, and its publication (lane calltable-base-helpers,
2026-11-11).

Each run is the body the device Python export of the same name ran
(`bindings/hotpath_device.mojo`, which now unpacks its arguments and calls
the run): the device twin of `core/hotpath_device.mojo` when its switch is
on and the call is one the twin covers, otherwise the host run of
`core/base_helpers_host.mojo`, which raises its own words. Same bytes either
way. The operations the GPU binding has always served from the host helpers
(row means, fold assignment, fold hits, split roles) run the host runs.

No Python object crosses; the entry is called with the GIL released by its
caller (the Python exports release it around the same runs).
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.base_entry_table import (
    BaseEntryKey, base_table_publish, base_check_spans, base_word, base_entry_vendor,
    BASE_HELPERS_ABI, BH_OP_COUNT,
    BH_CHECK_INDICES_I64, BH_BINCOUNT_I64, BH_REDUCE_STAT, BH_EQUAL_ELEMENTS, BH_BINCOUNT2_I32,
)
from core.base_helpers_host import base_helpers_host_dispatch, base_stat_store
from core.hotpath_device import (
    HPD_F32, HPD_I32, HPD_I64, HPD_MAX_N, HPD_U8,
    device_bincount_i64, device_bincount2_i32, device_check_indices_i64, device_equal_elements,
)
from core.hotpath_impl import check_indices_i64, equal_elements
from core.hotpath_impl_rows import bincount_i64
from core.label_device_steps import (
    LABELS_DEVICE_BASE, LABELS_DEVICE_ELEM, LABELS_DEVICE_CONTEXT, labels_device_reduce,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot

#: IDN_HPDEV_FOLDS (`-D MOJOLEARN_IDN_HPDEV_FOLDS_OFF`): the fold and index
#: helpers on the device (see `bindings/hotpath_device.mojo`, which imports
#: it from here).
comptime IDN_HPDEV_FOLDS = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_FOLDS_OFF"]()


def check_indices_i64_device_run(a: Int, count: Int, limit: Int) raises -> Int:
    """`check_indices_i64`: 0, 1 (out of range) or 2 (duplicate)."""
    comptime if IDN_HPDEV_FOLDS:
        if count >= 1 and count <= HPD_MAX_N and limit >= 0 and limit <= HPD_MAX_N and a != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            return device_check_indices_i64(ctx, a, count, limit)
    return check_indices_i64(a, count, limit)


def equal_elements_device_run(a: Int, b: Int, c: Int, count: Int, d: Int) raises -> Int:
    """`equal_elements`: dst[i] = 1 where a[i] == b[i] (IEEE), one dtype."""
    comptime if LABELS_DEVICE_ELEM:
        if count >= 1 and count <= HPD_MAX_N and c >= HPD_F32 and c <= HPD_U8 and a != 0 and b != 0 and d != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            device_equal_elements(ctx, a, b, c, count, d)
            return 0
    return equal_elements(a, b, c, count, d)


def bincount_i64_device_run(s: Int, c: Int, count: Int, kk: Int, ca: Int, acc: Bool) raises -> Int:
    """`bincount_i64`; a value out of range reruns the host helper, which
    raises."""
    comptime if LABELS_DEVICE_ELEM:
        if (
            count >= 1 and count <= HPD_MAX_N and kk >= 1 and kk <= HPD_MAX_N
            and (c == HPD_I32 or c == HPD_I64) and s != 0 and ca != 0
        ):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if device_bincount_i64(ctx, s, c, count, kk, ca, acc):
                return 0
    return bincount_i64(s, c, count, kk, ca, acc)


def bincount2_i32_device_run(a: Int, b: Int, count: Int, ka: Int, kb: Int, ca: Int) raises -> Int:
    """dims [n, ka, kb]: int64 counts[b * ka + a] over two int32 code
    columns, on the device (this binding has no host route). Raises when a
    code is outside [0, ka) x [0, kb)."""
    if count < 0 or ka < 1 or kb < 1 or ka > HPD_MAX_N // kb or count > HPD_MAX_N:
        raise Error("bincount2_i32: 0 <= n, 1 <= ka, kb and ka * kb within the device bound")
    if ca == 0:
        raise Error("bincount2_i32: null counts address")
    if count == 0:
        var cp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ca)
        for j in range(ka * kb):  # small-loop(ka * kb: zero rows, the table is groups x classes): zeroes the empty table, no row data
            cp.unsafe_store(j, Int64(0))
        return 0
    if a == 0 or b == 0:
        raise Error("bincount2_i32: null code address")
    var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
    if not device_bincount2_i32(ctx, a, b, count, ka, kb, ca):
        raise Error("bincount2_i32: a code is out of range")
    return 0


def base_helpers_device_dispatch(op: Int, aa: Int, pp: Int) raises -> Int:
    """One base helper by its `BH_*` code over admitted spans."""
    if op == BH_CHECK_INDICES_I64:
        return check_indices_i64_device_run(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1))
    if op == BH_BINCOUNT_I64:
        return bincount_i64_device_run(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1), base_word(pp, 2),
                                       base_word(aa, 1), base_word(pp, 3) != 0)
    if op == BH_REDUCE_STAT:
        return base_stat_store(labels_device_reduce(base_word(aa, 0), base_word(pp, 0), base_word(pp, 1),
                                                    base_word(pp, 2)), base_word(aa, 1))
    if op == BH_EQUAL_ELEMENTS:
        return equal_elements_device_run(base_word(aa, 0), base_word(aa, 1), base_word(pp, 0), base_word(pp, 1),
                                         base_word(aa, 2))
    if op == BH_BINCOUNT2_I32:
        return bincount2_i32_device_run(base_word(aa, 0), base_word(aa, 1), base_word(pp, 0), base_word(pp, 1),
                                        base_word(pp, 2), base_word(aa, 2))
    # Row means, fold assignment, fold hits and split roles: the host runs,
    # as the GPU binding's Python exports always were.
    return base_helpers_host_dispatch(op, aa, pp)


def base_helpers_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The GPU base binding's base helper entry (`BASE_HELPERS_ABI`):
    addresses aa[0:na], Int words pp[0:np], no float words. Writes the run's
    answer to result; a refusal goes to sink."""
    try:
        base_check_spans(operation, aa, na, pp, np, nf, result)
        var value = base_helpers_device_dispatch(operation, aa, pp)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct BaseHelpersDeviceOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the device base helper table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _DEVICE_OWNER_NAME = "MojolearnBaseHelpersDeviceOwnerV1"
comptime BASE_HELPERS_DEVICE_OWNER = SharedSlot[BaseHelpersDeviceOwner, _DEVICE_OWNER_NAME]


def publish_base_helpers_device() raises:
    """Publish the GPU base binding's base helper entry at module init. A
    build with no GPU vendor ("none") publishes nothing: its callers refuse
    with the table's "provider is not published"."""
    var word = base_entry_vendor(COMPILED_VENDOR)
    if word < 1:
        return
    var owner = BASE_HELPERS_DEVICE_OWNER.get_or_create_ptr()
    var key = BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    base_table_publish(key, Int(owner), [String(BASE_HELPERS_ABI)], [BH_OP_COUNT], [base_helpers_device_v1])
