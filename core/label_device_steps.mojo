# SPDX-License-Identifier: Apache-2.0
"""Existing label-related hotpath dispatch, with no Python envelopes.

The measured switches, dtype/size coverage, kernels and host refusal replay
are unchanged from bindings.hotpath_device. Callers own GIL/owner admission.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hotpath_impl import (
    HOTPATH_SABOTAGE, HpStat, HP_STAT_INT, HP_STAT_FLOAT, HP_STAT_TUPLE,
    cast_elements, reduce_stat, gather_i32, threshold_labels_i64,
)
from core.hotpath_device import (
    HPD_MAX_N, HPD_F32, HPD_F64, HPD_I32, HPD_I64, HPD_U32, HPD_U8,
    device_gather_i32, device_isum, device_all_integral, device_reduce_arg, device_cast_elements,
    device_threshold_labels_i64,
)
from core.hotpath_device_impl import device_all_finite_span
from core.neural_context import process_ctx

comptime LABELS_DEVICE_BASE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not HOTPATH_SABOTAGE
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime LABELS_DEVICE_ELEM = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_ELEM_OFF"]()
comptime LABELS_DEVICE_REDUCE = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_REDUCE_OFF"]()
comptime LABELS_DEVICE_ISUM = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_ISUM_OFF"]()
comptime LABELS_DEVICE_CAST = LABELS_DEVICE_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_CAST_OFF"]()
comptime LABELS_DEVICE_INPUT = not HOTPATH_SABOTAGE
comptime LABELS_DEVICE_CONTEXT = (
    "MojolearnCoreContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV3"
)


def labels_device_threshold_eligible(source: Int, code: Int, count: Int, dst: Int) -> Bool:
    return Bool(LABELS_DEVICE_ELEM) and count >= 1 and count <= HPD_MAX_N and (code == HPD_F32 or code == HPD_F64) and source != 0 and dst != 0


def labels_device_threshold(source: Int, code: Int, count: Int, threshold: Float64,
    strict: Bool, below: Int64, above: Int64, dst: Int) raises -> Int:
    if labels_device_threshold_eligible(source, code, count, dst):
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        device_threshold_labels_i64(ctx, source, code, count, threshold, strict, below, above, dst)
        return 0
    return threshold_labels_i64(source, code, count, threshold, strict, below, above, dst)


def labels_device_gather_i32(table: Int, classes: Int, codes: Int, count: Int, dst: Int) raises -> Int:
    comptime if LABELS_DEVICE_ELEM:
        if count >= 1 and count <= HPD_MAX_N and classes >= 1 and classes <= HPD_MAX_N and table != 0 and codes != 0 and dst != 0:
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if device_gather_i32(ctx, table, classes, codes, count, dst):
                return 0
    return gather_i32(table, classes, codes, count, dst)


def _peek[dt: DType](address: Int, index: Int) -> HpStat:
    var pointer = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=address)
    var value = pointer.unsafe_load(index)
    comptime if dt.is_floating_point():
        return HpStat(HP_STAT_FLOAT, 0, value.cast[DType.float64](), 0, 0, 0)
    else:
        return HpStat(HP_STAT_INT, Int(value), 0, 0, 0, 0)


def _first_is_nan(address: Int, code: Int) -> Bool:
    if code == HPD_F32:
        var value = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load(0)
        return value != value
    if code == HPD_F64:
        var value = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load(0)
        return value != value
    return False


def labels_device_reduce(address: Int, code: Int, count: Int, operation: Int) raises -> HpStat:
    comptime if LABELS_DEVICE_ISUM:
        if (operation == 5 and count >= 1 and count <= HPD_MAX_N and address != 0
                and (code == HPD_I32 or code == HPD_I64 or code == HPD_U32 or code == HPD_U8)):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            var words = device_isum(ctx, address, code, count)
            return HpStat(HP_STAT_TUPLE, 0, 0, Int(bitcast[DType.int64](words[0])),
                Int(words[1] >> 32), Int(words[1] & UInt64(0xFFFFFFFF)))
    comptime if LABELS_DEVICE_REDUCE:
        if count >= 1 and count <= HPD_MAX_N and address != 0 and code >= HPD_F32 and code <= HPD_U8:
            if operation == 4 and (code == HPD_F32 or code == HPD_F64):
                var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
                var ok = device_all_integral(ctx, address, code, count)
                return HpStat(HP_STAT_INT, 1 if ok else 0, 0, 0, 0, 0)
            if (operation == 0 or operation == 1 or operation == 3) and not _first_is_nan(address, code):
                var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
                var index = device_reduce_arg(ctx, address, code, count, operation != 0)
                if operation == 3:
                    return HpStat(HP_STAT_INT, index, 0, 0, 0, 0)
                if code == HPD_F32:
                    return _peek[DType.float32](address, index)
                if code == HPD_F64:
                    return _peek[DType.float64](address, index)
                if code == HPD_I32:
                    return _peek[DType.int32](address, index)
                if code == HPD_I64:
                    return _peek[DType.int64](address, index)
                if code == HPD_U32:
                    return _peek[DType.uint32](address, index)
                return _peek[DType.uint8](address, index)
    return reduce_stat(address, code, count, operation)


def labels_device_cast(source: Int, source_code: Int, dst: Int, dst_code: Int, count: Int) raises -> Int:
    comptime if LABELS_DEVICE_CAST:
        if (count >= 1 and count <= HPD_MAX_N and source != 0 and dst != 0 and source_code != dst_code
                and source_code >= 0 and source_code <= 5 and dst_code >= 0 and dst_code <= 5):
            var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
            if device_cast_elements(ctx, source, source_code, dst, dst_code, count):
                return 0
    return cast_elements(source, source_code, dst, dst_code, count)


def labels_device_try_finite(address: Int, count: Int, wide: Bool) raises -> Int:
    comptime if LABELS_DEVICE_INPUT:
        if count < 1 or address == 0:
            return -1
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        return 1 if device_all_finite_span(ctx, address, count, wide) else 0
    return -1
