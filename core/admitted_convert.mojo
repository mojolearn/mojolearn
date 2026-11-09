# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host conversion of an admitted strided buffer into a C-order store
of any Array dtype (no Python import): Array.astype's rules (float to float32
round to nearest even, integers range checked, float to integer truncated
toward zero). The native linalg / metrics admission
(bindings/linalg_ops_buffers.mojo `lin_convert`) reads it.

    admitted_convert(src, shape, strides, count, item, kind, titem, tkind,
                     typestr, dst)
        every element of the view at byte address `src` (shape / byte
        strides, `count` elements, items of `item` bytes of CastIn kind
        `kind`) in C order into the `typestr` store at `dst` (items of
        `titem` bytes of kind `tkind`); a same-dtype view with a unit-stride
        last axis is copied one contiguous run at a time

The elements (or runs) are split into contiguous C-order ranges across the
host's cores (core/host_predict_threads.mojo, with its grain floor). Each
element writes only its own cell, so no word depends on the split; a refusal
is the one the first failing element in C order raises, as the serial walk
raised it (each task records its first failing element; the lowest is
converted again on the calling thread to raise). Moved here from the binding
(lane a3-admit-decomp-prep-metrics, audit R10: it was a serial element walk).
"""
from std.math import isinf
from std.memory import bitcast, memcpy
from core.admitted_input import admitted_raw_word
from core.cast_in import cast_f32_bits, f16_to_f32_bits, CAST_KIND_UNSIGNED, CAST_KIND_SIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime LIN_IP = MutPointer[Int, MutUntrackedOrigin]
comptime LIN_BP = MutPointer[UInt8, MutUntrackedOrigin]


def _lin_ctype(typestr: String) -> String:
    """The C type `array.array` names in its range refusal."""
    if typestr == "<i1":
        return "signed char"
    if typestr == "<i4":
        return "signed integer"
    if typestr == "<u2":
        return "unsigned short"
    if typestr == "<u1":
        return "unsigned byte integer"
    if typestr == "<u4":
        return "unsigned int"
    return "signed long long"


def _lin_bounds(typestr: String, mut low: Int, mut high: Int):
    if typestr == "<i1":
        low = -128
        high = 127
    elif typestr == "<u1":
        low = 0
        high = 255
    elif typestr == "<u2":
        low = 0
        high = 65535
    elif typestr == "<i4":
        low = -(1 << 31)
        high = (1 << 31) - 1
    elif typestr == "<u4":
        low = 0
        high = (1 << 32) - 1
    else:
        low = Int.MIN
        high = Int.MAX


@always_inline
def _lin_raw(base: Int, off: Int, item: Int) -> UInt64:
    """The item at byte `off` from `base`, zero-extended (little-endian): one
    typed load (`admitted_raw_word`)."""
    return admitted_raw_word(base + off, item)


@always_inline
def _lin_store(dst: Int, t: Int, item: Int, raw: UInt64):
    """Cell `t` of the store at `dst` (items of `item` bytes): one typed
    store of the low `item` bytes of `raw`."""
    var address = dst + t * item
    if item == 4:
        MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address).unsafe_store[alignment=1](0, UInt32(raw))
    elif item == 8:
        MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address).unsafe_store[alignment=1](0, raw)
    elif item == 2:
        MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address).unsafe_store[alignment=1](0, UInt16(raw))
    else:
        var p = LIN_BP(unsafe_from_address=address)
        for j in range(item):  # small-loop(bytes of one item): at most 8
            p[j] = UInt8((raw >> UInt64(8 * j)) & UInt64(0xFF))


def _lin_float(raw: UInt64, item: Int) -> Float64:
    if item == 2:
        return bitcast[DType.float32](f16_to_f32_bits(raw)).cast[DType.float64]()
    if item == 4:
        return bitcast[DType.float32](UInt32(raw)).cast[DType.float64]()
    return bitcast[DType.float64](raw)


def _lin_integer(raw: UInt64, kind: Int, item: Int, typestr: String, low: Int, high: Int) raises -> Int:
    """One item as an exact integer for an integer target, refused as
    Array.astype refuses (a NaN, an infinity, a value outside the target).
    `low`/`high` are `_lin_bounds(typestr)`, read once per conversion by the
    caller; the C type name is built only for a refusal (both were rebuilt
    for every element)."""
    var value: Int
    if kind == CAST_KIND_BOOL:
        value = 1 if raw != 0 else 0
    elif kind == CAST_KIND_SIGNED:
        var bits = raw
        if item < 8 and (raw >> UInt64(8 * item - 1)) & UInt64(1) != 0:
            bits = raw | (~UInt64(0) << UInt64(8 * item))
        value = Int(bitcast[DType.int64](bits))
    elif kind == CAST_KIND_UNSIGNED:
        if raw > UInt64(Int.MAX):
            raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + _lin_ctype(typestr) + " is greater than maximum")
        value = Int(raw)
    else:
        var x = _lin_float(raw, item)
        if x != x:
            raise Error("cannot convert float NaN to integer")
        if isinf(x):
            raise Error("OverflowError: cannot convert float infinity to integer")
        if x >= 9223372036854775808.0 or x < -9223372036854775808.0:
            raise Error("OverflowError: mojolearn: value does not fit " + typestr)
        value = Int(x)  # truncation toward zero
    if value > high:
        raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + _lin_ctype(typestr) + " is greater than maximum")
    if value < low:
        raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + _lin_ctype(typestr) + " is less than minimum")
    return value


@always_inline
def _lin_offset(linear: Int, shape: LIN_IP, strides: LIN_IP, rank: Int) -> Int:
    """The byte offset of C-order element `linear` of the strided view."""
    var remainder = linear
    var offset = 0
    for k in range(rank):  # small-loop(the view's rank)
        var axis = rank - 1 - k
        var extent = shape.unsafe_load(axis)
        offset += (remainder % extent) * strides.unsafe_load(axis)
        remainder //= extent
    return offset


def _lin_one(raw: UInt64, kind: Int, item: Int, tkind: Int, titem: Int, same: Bool, typestr: String,
             low: Int, high: Int, wide_low: Int, wide_high: Int, dst: Int, t: Int) raises:
    """Element `t` (its raw item) converted into the C-order store at `dst`
    by Array.astype's rules; raises Array.astype's refusal."""
    if same:
        _lin_store(dst, t, titem, raw)
    elif tkind == CAST_KIND_FLOAT and titem == 4:
        _lin_store(dst, t, 4, UInt64(cast_f32_bits(raw, kind, item)))
    elif tkind == CAST_KIND_FLOAT and titem == 8:
        var x = Float64(0.0)
        if kind == CAST_KIND_FLOAT:
            x = _lin_float(raw, item)
        else:
            x = Float64(_lin_integer(raw, kind, item, "<i8", wide_low, wide_high))
        _lin_store(dst, t, 8, bitcast[DType.uint64](x))
    else:
        var value = _lin_integer(raw, kind, item, typestr, low, high)
        _lin_store(dst, t, titem, UInt64(bitcast[DType.uint64](Int64(value))))


def admitted_convert(
    src: Int, shape_in: List[Int], strides_in: List[Int], count: Int, item: Int, kind: Int, titem: Int, tkind: Int,
    typestr: String, dst: Int,
) raises:
    """See the module note. `shape_in` / `strides_in` hold at least one
    axis (a rank-0 view is one axis of 1)."""
    # parallel: host input layout copy / conversion over element ranges across host cores (was serial, audit R10)
    var same = kind == tkind and item == titem
    var nd = len(shape_in)
    var shape = shape_in.copy()
    var strides = strides_in.copy()
    if count <= 0:
        return
    var rank = max(nd, 1)
    var shape_ptr = rebind[LIN_IP](shape.unsafe_ptr())
    var stride_ptr = rebind[LIN_IP](strides.unsafe_ptr())
    if same and nd >= 1 and strides[nd - 1] == item:
        # Same dtype, unit-stride last axis (a row slice, a transposed outer
        # axis): one memcpy per contiguous run (p3 rank 99), runs split
        # across the host's cores. Same bytes.
        var run = shape[nd - 1]
        var runs = count // run
        var run_tasks = host_predict_task_count_work(runs, count * item)
        var run_chunk = host_predict_chunk(runs, run_tasks)

        def _runs(t: Int) {imm src, imm dst, imm run, imm runs, imm item, imm run_chunk, imm shape_ptr, imm stride_ptr, imm rank}:
            var lo = t * run_chunk
            var hi = min(lo + run_chunk, runs)
            for r in range(lo, hi):  # admitted input: this task's contiguous runs
                memcpy(dest=LIN_BP(unsafe_from_address=dst + r * run * item),
                       src=LIN_BP(unsafe_from_address=src + _lin_offset(r * run, shape_ptr, stride_ptr, rank)),
                       count=run * item)

        if run_tasks <= 1:
            _runs(0)
        else:
            host_parallelize(_runs, run_tasks)
        _ = shape^
        _ = strides^
        return
    var low = 0
    var high = 0
    _lin_bounds(typestr, low, high)
    var wide_low = 0
    var wide_high = 0
    _lin_bounds("<i8", wide_low, wide_high)
    var tasks = host_predict_task_count_work(count, count * max(item, 1))
    var chunk = host_predict_chunk(count, tasks)
    # The first refused element of each task (count when none).
    var failed = List[Int](length=tasks, fill=count)
    var failed_ptr = rebind[LIN_IP](failed.unsafe_ptr())

    def _elements(t: Int) {imm src, imm dst, imm kind, imm item, imm tkind, imm titem, imm same, imm low, imm high, imm wide_low, imm wide_high, imm count, imm chunk, imm failed_ptr, imm shape_ptr, imm stride_ptr, imm rank}:
        var lo = t * chunk
        var hi = min(lo + chunk, count)
        for e in range(lo, hi):  # admitted input: this task's elements, converted one by one
            var raw = _lin_raw(src, _lin_offset(e, shape_ptr, stride_ptr, rank), item)
            try:
                # The target's name only spells a refusal, which is raised
                # again below with it; the bounds are the target's own.
                _lin_one(raw, kind, item, tkind, titem, same, String(), low, high, wide_low, wide_high, dst, e)
            except:
                failed_ptr.unsafe_store(t, e)
                return

    if tasks <= 1:
        _elements(0)
    else:
        host_parallelize(_elements, tasks)
    var first = count
    for t in range(tasks):  # driver-loop: over the conversion tasks' first refusals (at most one per core)
        first = min(first, failed[t])
    if first < count:
        # Raise the refusal of the first failing element in C order.
        var raw = _lin_raw(src, _lin_offset(first, shape_ptr, stride_ptr, rank), item)
        _lin_one(raw, kind, item, tkind, titem, same, typestr, low, high, wide_low, wide_high, dst, first)
    _ = failed^
    _ = shape^
    _ = strides^
