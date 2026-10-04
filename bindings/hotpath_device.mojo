# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU base binding's shared helpers on the device (lane fam2-shared,
2026-10-04; cpu-gpu audit case 10).

Each `<name>_binding` here has the name and the Python signature of the host
helper in `bindings/hotpath_helpers.mojo` it replaces in
`bindings/_mojolearn.mojo` (the GPU base binding imports these instead; the
core host binding keeps the host helpers, which are the host column). A
binding runs the device twin of `core/hotpath_device.mojo` when its switch is
on and the call is one the twin covers; otherwise, and whenever the twin
reports an input the host helper refuses, it calls the host helper, which
raises its own words. Same bytes either way: integers, comparisons and bit
moves.

Switches (IDENTICAL builds, default ON; `-D MOJOLEARN_IDN_ALL_OFF` turns all
five off):
  IDN_HPDEV_ELEM    -D MOJOLEARN_IDN_HPDEV_ELEM_OFF    equal_elements,
                    gather_i32, threshold_labels_i64, bincount_i64,
                    count_mask_u8, fold_pair_f32, gather_i64 and gather_f64
                    (`hpdev_try_gather_u64`, called by the base binding)
  IDN_HPDEV_LABELS  -D MOJOLEARN_IDN_HPDEV_LABELS_OFF  encode_labels_<dtype>
                    (`hpdev_try_encode_labels`, called by the base binding)
  IDN_HPDEV_FOLDS   -D MOJOLEARN_IDN_HPDEV_FOLDS_OFF   fold_ids,
                    select_fold_i64, select_mask_u8_i64,
                    mask_from_indices_u8, arange_i64, leave_range_i64,
                    check_indices_i64, indices_overlap_i64, first_seen_i32,
                    strat_fold_assign_i32
  IDN_HPDEV_REDUCE  -D MOJOLEARN_IDN_HPDEV_REDUCE_OFF  reduce_stat's min,
                    max, argmax and integral test (the sequential float
                    sum stays the host helper's)
  IDN_HPDEV_INIT    -D MOJOLEARN_IDN_HPDEV_INIT_OFF    uniform_init_f32
  IDN_HPDEV_ISUM    -D MOJOLEARN_IDN_HPDEV_ISUM_OFF    reduce_stat's exact
                    integer sum (lane fix-s1-shared)
  IDN_HPDEV_CAST_F64  CANDIDATE, default OFF, -D MOJOLEARN_IDN_HPDEV_CAST_F64
                    turns it on: cast_f64_to_f32 (`hpdev_try_cast_f64_to_f32`)
The sabotage builds (`HOTPATH_SABOTAGE`) keep every host helper, so the
negative control still answers wrong on purpose.
"""

from std.math import isfinite
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined

from bindings.hotpath_helpers import (
    HOTPATH_SABOTAGE,
    arange_i64_binding as host_arange_i64_binding,
    bincount_i64_binding as host_bincount_i64_binding,
    check_indices_i64_binding as host_check_indices_i64_binding,
    count_mask_u8_binding as host_count_mask_u8_binding,
    equal_elements_binding as host_equal_elements_binding,
    first_seen_i32_binding as host_first_seen_i32_binding,
    fold_ids_binding as host_fold_ids_binding,
    fold_pair_f32_binding as host_fold_pair_f32_binding,
    gather_i32_binding as host_gather_i32_binding,
    indices_overlap_i64_binding as host_indices_overlap_i64_binding,
    leave_range_i64_binding as host_leave_range_i64_binding,
    mask_from_indices_u8_binding as host_mask_from_indices_u8_binding,
    select_fold_i64_binding as host_select_fold_i64_binding,
    select_mask_u8_i64_binding as host_select_mask_u8_i64_binding,
    strat_fold_assign_i32_binding as host_strat_fold_assign_i32_binding,
    reduce_stat_binding as host_reduce_stat_binding,
    threshold_labels_i64_binding as host_threshold_labels_i64_binding,
    uniform_init_f32_binding as host_uniform_init_f32_binding,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hotpath_device import (
    HPD_F32,
    HPD_F64,
    HPD_I32,
    HPD_I64,
    HPD_MAX_N,
    HPD_U32,
    HPD_U8,
    device_all_integral,
    device_arange_skip_i64,
    device_bincount_i64,
    device_cast_f64_to_f32,
    device_check_indices_i64,
    device_count_mask_u8,
    device_encode_labels,
    device_equal_elements,
    device_first_seen_i32,
    device_fold_pair_f32,
    device_gather_i32,
    device_gather_u64,
    device_indices_overlap_i64,
    device_isum,
    device_kfold_ids,
    device_mask_from_indices_u8,
    device_reduce_arg,
    device_select_fold_i64,
    device_select_mask_u8_i64,
    device_strat_fold_assign_i32,
    device_stratified_fold_ids,
    device_threshold_labels_i64,
    device_uniform_init_f32,
)
from core.neural_context import process_ctx


comptime _HPDEV_BASE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not HOTPATH_SABOTAGE
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime IDN_HPDEV_ELEM = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_ELEM_OFF"]()
comptime IDN_HPDEV_LABELS = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_LABELS_OFF"]()
comptime IDN_HPDEV_FOLDS = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_FOLDS_OFF"]()
comptime IDN_HPDEV_REDUCE = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_REDUCE_OFF"]()
comptime IDN_HPDEV_INIT = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_INIT_OFF"]()
#: lane fix-s1-shared: reduce_stat's exact integer sum as a device tile fold.
comptime IDN_HPDEV_ISUM = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_ISUM_OFF"]()
#: CANDIDATE ARM, default OFF: `-D MOJOLEARN_IDN_HPDEV_CAST_F64` narrows a
#: float64 input on the device (`cast_f64_to_f32`). The words cross the bus
#: twice more than the host cast's, so it is on only when measured to win, or
#: as the first half of a resident handoff.
comptime IDN_HPDEV_CAST_F64 = _HPDEV_BASE and is_defined["MOJOLEARN_IDN_HPDEV_CAST_F64"]()

#: The base binding's process-lifetime context slot
#: (`bindings/_mojolearn.mojo::_DEVCTX_SLOT`): the same name, so the same
#: DeviceContext.
comptime _HPDEV_SLOT = (
    "MojoCoreContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoCoreContextFast"
)

#: Largest k x n_folds table the stratified helpers build.
comptime _HPDEV_MAX_TABLE = 1 << 26


# ---------------------------------------------------------------------------
# IDN_HPDEV_ELEM
# ---------------------------------------------------------------------------


def equal_elements_binding(
    a_addr: PythonObject, b_addr: PythonObject, code: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`equal_elements` (see `bindings/hotpath_helpers.mojo`), on the device."""
    comptime if IDN_HPDEV_ELEM:
        var count = Int(py=n)
        var c = Int(py=code)
        var a = Int(py=a_addr)
        var b = Int(py=b_addr)
        var d = Int(py=dst_addr)
        if count >= 1 and count <= HPD_MAX_N and c >= HPD_F32 and c <= HPD_U8 and a != 0 and b != 0 and d != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                device_equal_elements(ctx, a, b, c, count, d)
            return PythonObject(0)
    return host_equal_elements_binding(a_addr, b_addr, code, n, dst_addr)


def gather_i32_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`gather_i32`, on the device; an out-of-range code reruns the host
    helper, which raises."""
    comptime if IDN_HPDEV_ELEM:
        var count = Int(py=n)
        var nt = Int(py=n_table)
        var t = Int(py=table_addr)
        var c = Int(py=codes_addr)
        var d = Int(py=dst_addr)
        if count >= 1 and count <= HPD_MAX_N and nt >= 1 and nt <= HPD_MAX_N and t != 0 and c != 0 and d != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            var ok = False
            with GILReleased(Python()):
                ok = device_gather_i32(ctx, t, nt, c, count, d)
            if ok:
                return PythonObject(0)
    return host_gather_i32_binding(table_addr, n_table, codes_addr, n, dst_addr)


def hpdev_try_gather_u64(
    table_addr: Int, nt: Int, codes_addr: Int, n: Int, dst_addr: Int,
) raises -> Bool:
    """The device `gather_i64` / `gather_f64` for the base binding (64-bit
    table words, int64 codes): True when the device wrote `dst`; False when
    it did not (switch off, a size it does not cover, a code out of range)
    and the caller runs its host loop, whose refusal is the definition."""
    comptime if IDN_HPDEV_ELEM:
        if n < 1 or n > HPD_MAX_N or nt < 1 or nt > HPD_MAX_N:
            return False
        if table_addr == 0 or codes_addr == 0 or dst_addr == 0:
            return False
        var ctx = process_ctx[_HPDEV_SLOT]()
        var ok = False
        with GILReleased(Python()):
            ok = device_gather_u64(ctx, table_addr, nt, codes_addr, n, dst_addr)
        return ok
    return False


def threshold_labels_i64_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, thr: PythonObject,
    strict: PythonObject, below: PythonObject, above: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`threshold_labels_i64`, on the device (binary64 words)."""
    comptime if IDN_HPDEV_ELEM:
        var count = Int(py=n)
        var c = Int(py=code)
        var s = Int(py=src_addr)
        var d = Int(py=dst_addr)
        if count >= 1 and count <= HPD_MAX_N and (c == HPD_F32 or c == HPD_F64) and s != 0 and d != 0:
            var t = Float64(py=thr)
            var st = Int(py=strict) != 0
            var lo = Int64(Int(py=below))
            var hi = Int64(Int(py=above))
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                device_threshold_labels_i64(ctx, s, c, count, t, st, lo, hi, d)
            return PythonObject(0)
    return host_threshold_labels_i64_binding(src_addr, code, n, thr, strict, below, above, dst_addr)


def bincount_i64_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, k: PythonObject,
    counts_addr: PythonObject, accumulate: PythonObject,
) raises -> PythonObject:
    """`bincount_i64`, on the device; a value out of range reruns the host
    helper, which raises."""
    comptime if IDN_HPDEV_ELEM:
        var count = Int(py=n)
        var kk = Int(py=k)
        var c = Int(py=code)
        var s = Int(py=src_addr)
        var ca = Int(py=counts_addr)
        if (
            count >= 1 and count <= HPD_MAX_N and kk >= 1 and kk <= HPD_MAX_N
            and (c == HPD_I32 or c == HPD_I64) and s != 0 and ca != 0
        ):
            var acc = Int(py=accumulate) != 0
            var ctx = process_ctx[_HPDEV_SLOT]()
            var ok = False
            with GILReleased(Python()):
                ok = device_bincount_i64(ctx, s, c, count, kk, ca, acc)
            if ok:
                return PythonObject(0)
    return host_bincount_i64_binding(src_addr, code, n, k, counts_addr, accumulate)


def count_mask_u8_binding(mask_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`count_mask_u8`, on the device."""
    comptime if IDN_HPDEV_ELEM:
        var rows = Int(py=n)
        var m = Int(py=mask_addr)
        if rows >= 1 and rows <= HPD_MAX_N and m != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            var c = 0
            with GILReleased(Python()):
                c = device_count_mask_u8(ctx, m, rows)
            return PythonObject(c)
    return host_count_mask_u8_binding(mask_addr, n)


def fold_pair_f32_binding(
    dst_addr: PythonObject, src_addr: PythonObject, n: PythonObject,
) raises -> PythonObject:
    """`fold_pair_f32`, on the device."""
    comptime if IDN_HPDEV_ELEM:
        var count = Int(py=n)
        var d = Int(py=dst_addr)
        var s = Int(py=src_addr)
        if count >= 1 and count <= HPD_MAX_N and d != 0 and s != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                device_fold_pair_f32(ctx, d, s, count)
            return PythonObject(0)
    return host_fold_pair_f32_binding(dst_addr, src_addr, n)


# ---------------------------------------------------------------------------
# IDN_HPDEV_LABELS
# ---------------------------------------------------------------------------


def hpdev_try_encode_labels[dt: DType](
    src_addr: Int, n: Int, classes_addr: Int, max_classes: Int, codes_addr: Int,
) raises -> Int:
    """The device `encode_labels_<dt>` for the base binding's
    `_encode_labels_binding[dt]`: the class count when the device encoded
    the labels, or -1 when it did not (switch off, a size it does not cover,
    more than `max_classes` classes, a NaN label): the caller then runs its
    host encoder, whose answer or refusal is the definition."""
    comptime if IDN_HPDEV_LABELS:
        var code = -1
        comptime if dt == DType.float32:
            code = HPD_F32
        comptime if dt == DType.float64:
            code = HPD_F64
        comptime if dt == DType.int32:
            code = HPD_I32
        comptime if dt == DType.int64:
            code = HPD_I64
        comptime if dt == DType.uint32:
            code = HPD_U32
        comptime if dt == DType.uint8:
            code = HPD_U8
        if code < 0 or n < 1 or n > HPD_MAX_N or max_classes < 1:
            return -1
        if src_addr == 0 or classes_addr == 0 or codes_addr == 0:
            return -1
        var ctx = process_ctx[_HPDEV_SLOT]()
        var k = -1
        with GILReleased(Python()):
            k = device_encode_labels(ctx, src_addr, code, n, classes_addr, max_classes, codes_addr)
        return k
    return -1


# ---------------------------------------------------------------------------
# IDN_HPDEV_FOLDS
# ---------------------------------------------------------------------------


def fold_ids_binding(
    codes_addr: PythonObject, n: PythonObject, n_classes: PythonObject,
    n_splits: PythonObject, counts_addr: PythonObject, fold_addr: PythonObject,
    fold_counts_addr: PythonObject,
) raises -> PythonObject:
    """`fold_ids`, on the device: KFold's fold of a row is a closed form of
    its index; the stratified dealing ranks each row within its class by a
    stable sort. The per-fold counts of KFold are its `n_splits` sizes,
    written here (control data)."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var k = Int(py=n_classes)
        var splits = Int(py=n_splits)
        var ca = Int(py=codes_addr)
        var fa = Int(py=fold_addr)
        var fca = Int(py=fold_counts_addr)
        if rows >= 1 and rows <= HPD_MAX_N and splits >= 2 and splits <= HPD_MAX_N and fa != 0 and fca != 0:
            if ca == 0:
                var ctx = process_ctx[_HPDEV_SLOT]()
                with GILReleased(Python()):
                    device_kfold_ids(ctx, rows, splits, fa)
                var fcp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=fca)
                for fold in range(splits):
                    var size = rows // splits
                    if fold < rows % splits:
                        size += 1
                    fcp.unsafe_store(fold, Int64(size))
                return PythonObject(0)
            var na = Int(py=counts_addr)
            if k >= 1 and na != 0 and k * splits <= _HPDEV_MAX_TABLE:
                var ctx = process_ctx[_HPDEV_SLOT]()
                var ok = False
                with GILReleased(Python()):
                    ok = device_stratified_fold_ids(ctx, ca, rows, k, splits, na, fa, fca)
                if ok:
                    return PythonObject(0)
    return host_fold_ids_binding(
        codes_addr, n, n_classes, n_splits, counts_addr, fold_addr, fold_counts_addr
    )


def select_fold_i64_binding(
    fold_addr: PythonObject, n: PythonObject, fold: PythonObject,
    test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """`select_fold_i64`, on the device (flag, exclusive scan, scatter)."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var fa = Int(py=fold_addr)
        var ta = Int(py=test_addr)
        var ra = Int(py=train_addr)
        var want = Int(py=fold)
        if (
            rows >= 1 and rows <= HPD_MAX_N and fa != 0 and ta != 0 and ra != 0
            and want >= -2147483648 and want <= 2147483647
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var n_test = 0
            with GILReleased(Python()):
                n_test = device_select_fold_i64(ctx, fa, rows, want, ta, ra)
            return PythonObject(n_test)
    return host_select_fold_i64_binding(fold_addr, n, fold, test_addr, train_addr)


def select_mask_u8_i64_binding(
    mask_addr: PythonObject, n: PythonObject, test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """`select_mask_u8_i64`, on the device."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var ma = Int(py=mask_addr)
        if rows >= 1 and rows <= HPD_MAX_N and ma != 0:
            var ta = Int(py=test_addr)
            var ra = Int(py=train_addr)
            var ctx = process_ctx[_HPDEV_SLOT]()
            var n_test = 0
            with GILReleased(Python()):
                n_test = device_select_mask_u8_i64(ctx, ma, rows, ta, ra)
            return PythonObject(n_test)
    return host_select_mask_u8_i64_binding(mask_addr, n, test_addr, train_addr)


def mask_from_indices_u8_binding(
    idx_addr: PythonObject, k: PythonObject, n: PythonObject, mask_addr: PythonObject,
) raises -> PythonObject:
    """`mask_from_indices_u8`, on the device; an index out of range reruns
    the host helper, which raises."""
    comptime if IDN_HPDEV_FOLDS:
        var count = Int(py=k)
        var rows = Int(py=n)
        var ia = Int(py=idx_addr)
        var ma = Int(py=mask_addr)
        if (
            rows >= 1 and rows <= HPD_MAX_N and count >= 0 and count <= HPD_MAX_N
            and ma != 0 and (count == 0 or ia != 0)
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var set = -1
            with GILReleased(Python()):
                set = device_mask_from_indices_u8(ctx, ia, count, rows, ma)
            if set >= 0:
                return PythonObject(set)
    return host_mask_from_indices_u8_binding(idx_addr, k, n, mask_addr)


def arange_i64_binding(
    dst_addr: PythonObject, start: PythonObject, n: PythonObject,
) raises -> PythonObject:
    """`arange_i64`, on the device."""
    comptime if IDN_HPDEV_FOLDS:
        var count = Int(py=n)
        var d = Int(py=dst_addr)
        if count >= 1 and count <= HPD_MAX_N and d != 0:
            var lo = Int(py=start)
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                device_arange_skip_i64(ctx, d, lo, count, 0, 0)
            return PythonObject(0)
    return host_arange_i64_binding(dst_addr, start, n)


def leave_range_i64_binding(
    n: PythonObject, lo: PythonObject, hi: PythonObject,
    train_addr: PythonObject, test_addr: PythonObject,
) raises -> PythonObject:
    """`leave_range_i64`, on the device: test = [lo, hi), train = the other
    rows, each a range with at most one gap."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var a = Int(py=lo)
        var b = Int(py=hi)
        var ta = Int(py=test_addr)
        var ra = Int(py=train_addr)
        var held = b - a
        if (
            rows >= 1 and rows <= HPD_MAX_N and a >= 0 and b >= a and b <= rows
            and (held == 0 or ta != 0) and (held == rows or ra != 0)
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                if held > 0:
                    device_arange_skip_i64(ctx, ta, a, held, 0, 0)
                if rows - held > 0:
                    device_arange_skip_i64(ctx, ra, 0, rows - held, a, held)
            return PythonObject(held)
    return host_leave_range_i64_binding(n, lo, hi, train_addr, test_addr)


def check_indices_i64_binding(
    addr: PythonObject, n: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """`check_indices_i64`, on the device."""
    comptime if IDN_HPDEV_FOLDS:
        var count = Int(py=n)
        var limit = Int(py=bound)
        var a = Int(py=addr)
        if count >= 1 and count <= HPD_MAX_N and limit >= 0 and limit <= HPD_MAX_N and a != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            var status = 0
            with GILReleased(Python()):
                status = device_check_indices_i64(ctx, a, count, limit)
            return PythonObject(status)
    return host_check_indices_i64_binding(addr, n, bound)


def indices_overlap_i64_binding(
    a_addr: PythonObject, n_a: PythonObject, b_addr: PythonObject,
    n_b: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """`indices_overlap_i64`, on the device; an index out of range reruns
    the host helper."""
    comptime if IDN_HPDEV_FOLDS:
        var na = Int(py=n_a)
        var nb = Int(py=n_b)
        var limit = Int(py=bound)
        var a = Int(py=a_addr)
        var b = Int(py=b_addr)
        if (
            na >= 1 and na <= HPD_MAX_N and nb >= 1 and nb <= HPD_MAX_N
            and limit >= 1 and limit <= HPD_MAX_N and a != 0 and b != 0
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var hit = -1
            with GILReleased(Python()):
                hit = device_indices_overlap_i64(ctx, a, na, b, nb, limit)
            if hit >= 0:
                return PythonObject(hit)
    return host_indices_overlap_i64_binding(a_addr, n_a, b_addr, n_b, bound)


def first_seen_i32_binding(
    codes_addr: PythonObject, n: PythonObject, k: PythonObject, enc_addr: PythonObject,
    counts_addr: PythonObject,
) raises -> PythonObject:
    """`first_seen_i32`, on the device; a code out of range reruns the host
    helper, which raises."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var kk = Int(py=k)
        var ca = Int(py=codes_addr)
        var ea = Int(py=enc_addr)
        var na = Int(py=counts_addr)
        if (
            rows >= 1 and rows <= HPD_MAX_N and kk >= 1 and kk <= HPD_MAX_N
            and ca != 0 and ea != 0 and na != 0
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var m = -1
            with GILReleased(Python()):
                m = device_first_seen_i32(ctx, ca, rows, kk, ea, na)
            if m >= 0:
                return PythonObject(m)
    return host_first_seen_i32_binding(codes_addr, n, k, enc_addr, counts_addr)


def strat_fold_assign_i32_binding(
    enc_addr: PythonObject, n: PythonObject, k: PythonObject, n_folds: PythonObject,
    alloc_addr: PythonObject, perms_addr: PythonObject, counts_addr: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """`strat_fold_assign_i32`, on the device; tables the device twin does
    not accept rerun the host helper."""
    comptime if IDN_HPDEV_FOLDS:
        var rows = Int(py=n)
        var kk = Int(py=k)
        var K = Int(py=n_folds)
        var ea = Int(py=enc_addr)
        var aa = Int(py=alloc_addr)
        var pa = Int(py=perms_addr)
        var ca = Int(py=counts_addr)
        var da = Int(py=dst_addr)
        if (
            rows >= 1 and rows <= HPD_MAX_N and kk >= 1 and K >= 1 and K <= HPD_MAX_N
            and kk * K <= _HPDEV_MAX_TABLE and ea != 0 and aa != 0 and ca != 0 and da != 0
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var ok = False
            with GILReleased(Python()):
                ok = device_strat_fold_assign_i32(ctx, ea, rows, kk, K, aa, pa, ca, da)
            if ok:
                return PythonObject(0)
    return host_strat_fold_assign_i32_binding(
        enc_addr, n, k, n_folds, alloc_addr, perms_addr, counts_addr, dst_addr
    )


# ---------------------------------------------------------------------------
# IDN_HPDEV_REDUCE
# ---------------------------------------------------------------------------

#: `reduce_stat`'s reductions (`bindings/hotpath_helpers.mojo` HP_MIN ...).
comptime _RS_MIN = 0
comptime _RS_MAX = 1
comptime _RS_ARGMAX = 3
comptime _RS_INTEGRAL = 4
comptime _RS_ISUM = 5


def _peek[dt: DType](addr: Int, i: Int) -> PythonObject:
    """Element i as `reduce_stat` returns a minimum or maximum: a float as
    float64, an integer as an int. One scalar read of the caller's buffer."""
    var p = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)
    var v = p.unsafe_load(i)
    comptime if dt.is_floating_point():
        return PythonObject(v.cast[DType.float64]())
    else:
        return PythonObject(Int(v))


def _first_is_nan(addr: Int, code: Int) -> Bool:
    if code == HPD_F32:
        var v = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr).unsafe_load(0)
        return v != v
    if code == HPD_F64:
        var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr).unsafe_load(0)
        return w != w
    return False


def reduce_stat_binding(
    addr: PythonObject, code: PythonObject, n: PythonObject, what: PythonObject,
) raises -> PythonObject:
    """`reduce_stat`: min, max and argmax as a device tile reduction over
    (ordered key, index) pairs, and the integral test as a device predicate.
    A NaN first element (Python's answer is then that NaN, or index 0) and
    the float sum take the host helper. The exact integer sum (what = 5) is a
    device tile fold under IDN_HPDEV_ISUM (any order is exact)."""
    comptime if IDN_HPDEV_ISUM:
        var count = Int(py=n)
        var c = Int(py=code)
        var a = Int(py=addr)
        if (
            Int(py=what) == _RS_ISUM and count >= 1 and count <= HPD_MAX_N and a != 0
            and (c == HPD_I32 or c == HPD_I64 or c == HPD_U32 or c == HPD_U8)
        ):
            var ctx = process_ctx[_HPDEV_SLOT]()
            var hi_w = UInt64(0)
            var lo_w = UInt64(0)
            with GILReleased(Python()):
                var t = device_isum(ctx, a, c, count)
                hi_w = t[0]
                lo_w = t[1]
            return Python.tuple(
                PythonObject(Int(bitcast[DType.int64](hi_w))),
                PythonObject(Int(lo_w >> 32)),
                PythonObject(Int(lo_w & UInt64(0xFFFFFFFF))),
            )
    comptime if IDN_HPDEV_REDUCE:
        var count = Int(py=n)
        var w = Int(py=what)
        var c = Int(py=code)
        var a = Int(py=addr)
        if count >= 1 and count <= HPD_MAX_N and a != 0 and c >= HPD_F32 and c <= HPD_U8:
            if w == _RS_INTEGRAL and (c == HPD_F32 or c == HPD_F64):
                var ctx = process_ctx[_HPDEV_SLOT]()
                var ok = False
                with GILReleased(Python()):
                    ok = device_all_integral(ctx, a, c, count)
                return PythonObject(1 if ok else 0)
            if (w == _RS_MIN or w == _RS_MAX or w == _RS_ARGMAX) and not _first_is_nan(a, c):
                var ctx = process_ctx[_HPDEV_SLOT]()
                var at = 0
                with GILReleased(Python()):
                    at = device_reduce_arg(ctx, a, c, count, w != _RS_MIN)
                if w == _RS_ARGMAX:
                    return PythonObject(at)
                if c == HPD_F32:
                    return _peek[DType.float32](a, at)
                if c == HPD_F64:
                    return _peek[DType.float64](a, at)
                if c == HPD_I32:
                    return _peek[DType.int32](a, at)
                if c == HPD_I64:
                    return _peek[DType.int64](a, at)
                if c == HPD_U32:
                    return _peek[DType.uint32](a, at)
                return _peek[DType.uint8](a, at)
    return host_reduce_stat_binding(addr, code, n, what)


# ---------------------------------------------------------------------------
# IDN_HPDEV_INIT
# ---------------------------------------------------------------------------


def uniform_init_f32_binding(
    dst_addr: PythonObject, n: PythonObject, low: PythonObject, high: PythonObject,
    seed_lo: PythonObject, seed_hi: PythonObject, offset: PythonObject,
) raises -> PythonObject:
    """`uniform_init_f32`, drawn on the device: the same counter-based
    splitmix64 stream, the arithmetic in binary64 words."""
    comptime if IDN_HPDEV_INIT:
        var count = Int(py=n)
        var d = Int(py=dst_addr)
        var lo = Float64(py=low)
        var hi = Float64(py=high)
        if count >= 1 and count <= HPD_MAX_N and d != 0 and isfinite(lo) and isfinite(hi) and isfinite(hi - lo):
            var seed = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
            var off = UInt64(Int(py=offset))
            var ctx = process_ctx[_HPDEV_SLOT]()
            with GILReleased(Python()):
                device_uniform_init_f32(ctx, d, count, lo, hi, seed, off)
            return PythonObject(0)
    return host_uniform_init_f32_binding(dst_addr, n, low, high, seed_lo, seed_hi, offset)


# ---------------------------------------------------------------------------
# IDN_HPDEV_CAST_F64 (candidate arm, default OFF)
# ---------------------------------------------------------------------------


def hpdev_try_cast_f64_to_f32(src_addr: Int, dst_addr: Int, n: Int) raises -> Bool:
    """The device `cast_f64_to_f32` for the base binding: True when the
    device wrote `dst`, False when the caller must run its host loop."""
    comptime if IDN_HPDEV_CAST_F64:
        if n < 1 or n > HPD_MAX_N or src_addr == 0 or dst_addr == 0:
            return False
        var ctx = process_ctx[_HPDEV_SLOT]()
        with GILReleased(Python()):
            device_cast_f64_to_f32(ctx, src_addr, dst_addr, n)
        return True
    return False
