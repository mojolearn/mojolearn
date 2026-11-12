# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/hotpath_device.mojo` (wave 8 run split): the device
pair fold the GPU base image's parallel table calls, and the device input
fixes (float64 narrowing, transpose, strided copy) the base binding's doors
call with the GIL released. Each returns "not done" when the caller must run
its host loop. Typed Mojo only: no Python import."""
from core.hotpath_impl_rows import fold_pair_f32
from std.sys.compile import is_defined
from core.hotpath_device import (
    HPD_F32, HPD_F64, HPD_I32, HPD_I64, HPD_MAX_N, HPD_U32, HPD_U8, device_cast_f64_to_f32,
    device_encode_labels, device_fold_pair_f32, device_gather_u64,
)
from core.input_device import IND_MAX_N, device_strided_copy, device_transpose_to_f32
from core.neural_context import process_ctx
from core.label_device_steps import (
    LABELS_DEVICE_ELEM as IDN_HPDEV_ELEM, LABELS_DEVICE_INPUT as _HPDEV_INPUT,
    LABELS_DEVICE_CONTEXT as _HPDEV_SLOT, LABELS_DEVICE_BASE as _HPDEV_BASE, labels_device_try_finite,
)

comptime IDN_HPDEV_LABELS = _HPDEV_BASE and not is_defined["MOJOLEARN_IDN_HPDEV_LABELS_OFF"]()


def fold_pair_f32_device_run(d: Int, s: Int, count: Int) raises -> Int:
    """`fold_pair_f32`, on the device when the switch covers the call, else
    the host kernel (core/hotpath_impl_rows.mojo), which raises its own
    words. The GPU base image's parallel table entry
    (bindings/parallel_base_entry_device.mojo) calls it with the GIL
    released by its caller; the former Python door `fold_pair_f32` had no
    caller left once cross_vendor's fold read the table (lane
    calltable-parallel-archive, 2026-11-11)."""
    comptime if IDN_HPDEV_ELEM:
        if count >= 1 and count <= HPD_MAX_N and d != 0 and s != 0:
            var ctx = process_ctx[_HPDEV_SLOT]()
            device_fold_pair_f32(ctx, d, s, count)
            return 0
    return fold_pair_f32(d, s, count)


def hpdev_try_cast_f64_to_f32(src_addr: Int, dst_addr: Int, n: Int) raises -> Bool:
    """The device `cast_f64_to_f32` for the base binding: True when the
    device wrote `dst`, False when the caller must run its host loop (a
    sabotage build, or a size the twin does not cover)."""
    comptime if _HPDEV_INPUT:
        if n < 1 or n > HPD_MAX_N or src_addr == 0 or dst_addr == 0:
            return False
        var ctx = process_ctx[_HPDEV_SLOT]()
        device_cast_f64_to_f32(ctx, src_addr, dst_addr, n)
        return True
    return False


def hpdev_try_transpose_to_f32(
    src_addr: Int, dst_addr: Int, nr: Int, nc: Int, is_f64: Bool,
) raises -> Bool:
    """The device `transpose_f32` (float32 in) / `cast_colmajor_f64_to_f32`
    (float64 in, the `cast_f64_to_f32` narrowing per element): True when the
    device wrote `dst`, False when the caller must run its host loop."""
    comptime if _HPDEV_INPUT:
        if nr < 1 or nc < 1 or nr > IND_MAX_N // nc or src_addr == 0 or dst_addr == 0:
            return False
        var ctx = process_ctx[_HPDEV_SLOT]()
        device_transpose_to_f32(ctx, src_addr, dst_addr, nr, nc, is_f64)
        return True
    return False


def hpdev_strided_copy_run(src: Int, dst: Int, dims: Int, nd: Int, isz: Int) raises -> Int:
    """`strided_copy_bytes_binding`'s device copy: the element count when
    the device moved the bits, -1 when the caller must run the host helper
    (a refusal, an empty copy, or a span the twin does not cover)."""
    if nd >= 1 and nd <= 64 and dims != 0 and src != 0 and dst != 0 and (
        isz == 1 or isz == 2 or isz == 4 or isz == 8
    ):
        var dp = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=dims)
        var total = 1
        var valid = True
        for k in range(nd):  # small-loop(nd: at most 64 array dimensions): shape product, not data
            var e = Int(dp[k])
            if e < 1 or total > IND_MAX_N // e:
                valid = False
                break
            total *= e
        if valid:
            var ctx = process_ctx[_HPDEV_SLOT]()
            if device_strided_copy(ctx, src, dst, dims, nd, isz, total):
                return total
    return -1


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
        ok = device_gather_u64(ctx, table_addr, nt, codes_addr, n, dst_addr)
        return ok
    return False


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
        k = device_encode_labels(ctx, src_addr, code, n, classes_addr, max_classes, codes_addr)
        return k
    return -1


def hpdev_try_all_finite(addr: Int, n: Int, is_f64: Bool) raises -> Int:
    return labels_device_try_finite(addr, n, is_f64)
