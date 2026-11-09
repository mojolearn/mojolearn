# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The label helpers' HOST column (lane s1-shared-state-hardening-and-selector,
# S3): `unique_inverse` and the shared argument-and-label entry
# `prepare_targets`, moved word for word out of the core host binding root
# (bindings/_mojolearn_core_host.mojo, which imports them back under the same
# exported names). bindings/classical_labels.mojo calls them in-process on an
# install with no GPU base binding (labels_base_binding is None); on a GPU
# install it calls the base binding's device twins through the
# labels_native_api_v1 provider (bindings/labels_native_abi.mojo). Same
# statements, same bytes.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.host_helpers import all_finite_f32_binding, all_finite_f64_binding
from bindings.hotpath_helpers import (
    cast_elements_binding,
    encode_labels_f32_binding,
    encode_labels_f64_binding,
    encode_labels_i32_binding,
    encode_labels_i64_binding,
    encode_labels_u32_binding,
    encode_labels_u8_binding,
    reduce_stat_binding,
)
from core.hostptr import i32_ptr
from core.label_encode import host_unique_inverse


def unique_inverse_binding(
    src_addr: PythonObject, n: PythonObject, kind: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    """The GPU binding's `unique_inverse` (its contract and refusals) over
    the host twin `core/label_encode.mojo::host_unique_inverse`, the same
    statements as `core/label_encode_device.mojo` (cpu-gpu-cleanup
    w2-pyglue). Integers and bit moves: the same bytes."""
    var count = Int(py=n)
    var k_ind = Int(py=kind)
    if count < 1:
        raise Error("unique_inverse: n must be positive, got " + String(count))
    if k_ind != 0 and k_ind != 1:
        raise Error("unique_inverse: kind must be 0 (float64) or 1 (int64)")
    if Int(py=src_addr) == 0 or Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("unique_inverse: null buffer address")
    var sp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    var cp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(py=classes_addr))
    var dp = i32_ptr(Int(py=codes_addr))
    var k: Int
    with GILReleased(Python()):
        k = host_unique_inverse(sp, count, k_ind, cp, dp)
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


# ===========================================================================
# THE SHARED ARGUMENT-AND-LABEL ENTRY (public runtime migration, plan section
# 1 item 1, lane migrate-shared-labels). One native call takes a label
# vector and/or a sample-weight vector and returns the ORDER RULE's classes
# and codes and the weight verdict, so `_labels.encode_labels`,
# `_labels.unique_inverse` and `_sample_weight.check_sample_weight` each make
# this ONE call (they made up to three before, plus the Python glue between
# them). It composes the SAME helpers those paths called, in the same order,
# so no output byte moves: `encode_labels_<dt>` (capped encoder), then
# `cast_elements` + `unique_inverse` past the cap, `all_finite_<f>` and
# `reduce_stat` min/max for the weights. The core host binding carries the
# same body over its host twins under the same name.
#
#   labels: `y_addr` != 0 points at `n` values of dtype code `y_code`
#     (0 f32, 1 f64, 2 i32, 3 i64, 4 u32, 5 u8). The k sorted classes land
#     WIDENED at `classes_addr` (n slots of 64 bits: float64 for float
#     labels, int64 for integer labels), one int32 code per row at
#     `codes_addr`. A NaN label raises the ORDER RULE's message.
#   weights: `w_addr` != 0 points at `n` weights of code `w_code` (0 or 1).
#     `w_flags` bit 1 allows negative weights, bit 2 an all-zero vector.
#
# Returns k (0 without labels), or a negative status for refused weights:
# PT_W_NONFINITE, PT_W_NEGATIVE, PT_W_ALL_ZERO (the Python glue raises
# scikit-learn's message for each). Labels are checked before weights.
# ===========================================================================

comptime PT_ENCODE_CAP = 4096
comptime PT_W_NONFINITE = -3
comptime PT_W_NEGATIVE = -4
comptime PT_W_ALL_ZERO = -5
comptime PT_ALLOW_NEGATIVE = 1
comptime PT_ALLOW_ALL_ZERO = 2


def _pt_encode(
    y_addr: PythonObject, y_code: Int, n: PythonObject, classes_addr: Int,
    cap: Int, codes_addr: PythonObject,
) raises -> Int:
    """The capped encoder `encode_labels_<dt>` for dtype code `y_code`."""
    var ca = PythonObject(classes_addr)
    var cp = PythonObject(cap)
    if y_code == 0:
        return Int(py=encode_labels_f32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 1:
        return Int(py=encode_labels_f64_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 2:
        return Int(py=encode_labels_i32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 3:
        return Int(py=encode_labels_i64_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 4:
        return Int(py=encode_labels_u32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 5:
        return Int(py=encode_labels_u8_binding(y_addr, n, ca, cp, codes_addr))
    raise Error("prepare_targets: label dtype code must be 0..5, got " + String(y_code))


def prepare_targets_binding(
    y_addr: PythonObject, y_code: PythonObject, n: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject,
    w_addr: PythonObject, w_code: PythonObject, w_flags: PythonObject,
) raises -> PythonObject:
    """See the block comment above."""
    var count = Int(py=n)
    if count < 0:
        raise Error("prepare_targets: n must be non-negative, got " + String(count))
    var k = 0
    if Int(py=y_addr) != 0:
        if count < 1:
            raise Error("prepare_targets: labels need n >= 1")
        if Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
            raise Error("prepare_targets: null label output address")
        var yc = Int(py=y_code)
        var wide_kind = 0 if (yc == 0 or yc == 1) else 1
        var wide_code = 1 if wide_kind == 0 else 3
        var cap = count if count < PT_ENCODE_CAP else PT_ENCODE_CAP
        if yc == wide_code:
            # 64-bit labels: the encoder writes the classes in place
            k = _pt_encode(y_addr, yc, n, Int(py=classes_addr), cap, codes_addr)
        else:
            var tmp = List[UInt64](length=cap, fill=0)
            var tmp_addr = Int(tmp.unsafe_ptr())
            k = _pt_encode(y_addr, yc, n, tmp_addr, cap, codes_addr)
            if k > 0:
                # widen the k classes (exact: float32 -> float64, int -> int64)
                _ = cast_elements_binding(
                    PythonObject(tmp_addr), y_code, classes_addr,
                    PythonObject(wide_code), PythonObject(k),
                )
            _ = tmp^
        if k < 0:
            # more classes than the cap, or a NaN label: `unique_inverse` over
            # the widened labels (no cap; it raises the NaN refusal)
            if yc == wide_code:
                k = Int(py=unique_inverse_binding(
                    y_addr, n, PythonObject(wide_kind), classes_addr, codes_addr
                ))
            else:
                var wide = List[UInt64](length=count, fill=0)
                var wide_addr = Int(wide.unsafe_ptr())
                var st = Int(py=cast_elements_binding(
                    y_addr, y_code, PythonObject(wide_addr), PythonObject(wide_code), n
                ))
                if st != 0:
                    raise Error("prepare_targets: labels do not widen to 64 bits")
                k = Int(py=unique_inverse_binding(
                    PythonObject(wide_addr), n, PythonObject(wide_kind), classes_addr, codes_addr
                ))
                _ = wide^
    if Int(py=w_addr) != 0 and count > 0:
        var wc = Int(py=w_code)
        var flags = Int(py=w_flags)
        var fin: Int
        if wc == 1:
            fin = Int(py=all_finite_f64_binding(w_addr, n))
        elif wc == 0:
            fin = Int(py=all_finite_f32_binding(w_addr, n))
        else:
            raise Error("prepare_targets: weight dtype code must be 0 or 1, got " + String(wc))
        if fin == 0:
            return PythonObject(PT_W_NONFINITE)
        var lo = Float64(py=reduce_stat_binding(w_addr, w_code, n, PythonObject(0)))
        if (flags & PT_ALLOW_NEGATIVE) == 0 and lo < 0:
            return PythonObject(PT_W_NEGATIVE)
        if (flags & PT_ALLOW_ALL_ZERO) == 0:
            var hi = Float64(py=reduce_stat_binding(w_addr, w_code, n, PythonObject(1)))
            if not (hi > 0 or lo < 0):
                return PythonObject(PT_W_ALL_ZERO)
    return PythonObject(k)
