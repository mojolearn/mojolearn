# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The linalg operations on training device arrays (lane vdf-linalg4, 2026-11-24).

`mojolearn.linalg` functions given `TrainDeviceArray` operands (python/mojolearn/_training_impl.py) read them IN
PLACE from the training device pool (core/train_dev_pool.mojo) and write their results into new pool handles the door
allocated: no download, no re-upload, nothing between device steps; the door returns the result as a new
TrainDeviceArray. Each entry runs the host-address entry's own kernel on the same words in the same order (the same
bits as the host-input path and, through the host column the host image runs, the CPU column), on the linalg
binding's own context, and synchronizes that context before returning, so the result is complete before any other
binding's context reads the handle.

IDENTICAL GPU builds only (the device arrays exist only there, training/dev_tensors.mojo `IDN_TRAIN_DEV_TENSORS`):
a fast set refuses in words (its GEMM routes differ from the identical chain), and the host image's table refuses
the operation codes (gemm/host/linalg_dispatch.mojo).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import process_ctx
from core.train_dev_pool import train_dev_view, train_dev_view_as
from gemm.checks.gemm_lowbit import (
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_into,
    identical_gemm_int8_into,
    quantize_rows_int8_device,
)
from gemm.linalg_native import _refuse_lowbit_shape
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import INT8_MAX_K, OP_NN, OP_NT, OP_TN

comptime _DEVCTX_SLOT = "MojolearnLinalgContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinalgContextFastV3"


def lin_dev_ready(who: String) raises:
    """The refusal of a device-array operand outside an identical GPU set."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("RuntimeError: mojolearn.linalg." + who + ": training device arrays are read in identical mode "
                    + "only; pass host arrays, or select identical")


def gemm_dev_native(c_h: Int, a_h: Int, b_h: Int, params: List[Int]) raises -> Int:
    """`matmul` on device arrays: handle c_h (m x n, written) = op(a_h) . op(b_h), params (m, n, k, op) word for word
    as `gemm_native`. The host-address entry's refusals and its one kernel call (gemm/host_entry.mojo
    `identical_gemm_host`: `identical_gemm_into` on the dispatcher's workspace); the uploads and the download are
    the only lines that differ. Returns m * n."""
    lin_dev_ready("matmul")
    if len(params) != 4:
        raise Error("gemm: params must contain 4 values (m, n, k, op), got " + String(len(params)))
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("identical_gemm_host: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n)
                    + " k=" + String(k))
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("identical_gemm_host: op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var a = train_dev_view(ctx, a_h, m * k, "a")
    var b = train_dev_view(ctx, b_h, n * k, "b")
    var c = train_dev_view(ctx, c_h, m * n, "the product")
    var ws = ctx.enqueue_create_buffer[DType.float32](max(1, identical_gemm_workspace_max_floats(m, n, k)))
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
    # One wait: the product is complete before the door hands the handle to another context, and the workspace's
    # free is drained (the host entry's DEVIATION 3010 wait).
    ctx.synchronize()
    _ = ws^
    _ = a^
    _ = b^
    _ = c^
    return m * n


# The low-bit products and conversions (lane vdf-linalg4): the host-address entries' kernels (gemm/linalg_native.mojo
# `*_native`) on typed pool views (core/train_dev_pool.mojo `train_dev_view_as`: int8 codes and int15 planes, int32
# row exponents, uint16 bf16 bits packed in the handle's words). The uploads and the downloads are the only lines that
# differ; one wait at the end of each.


def to_bf16_dev_native(dst_h: Int, src_h: Int, params: List[Int]) raises -> Int:
    """`to_bf16` on device arrays: handle dst_h (count uint16 bf16 bits, written) from handle src_h (count float32);
    `bf16_narrow` as `to_bf16_native`. Returns count."""
    lin_dev_ready("to_bf16")
    if len(params) != 1:
        raise Error("to_bf16: params must contain 1 value (count)")
    var count = params[0]
    if count <= 0:
        raise Error("to_bf16: count must be positive, got " + String(count))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dsrc = train_dev_view(ctx, src_h, count, "x")
    var ddst = train_dev_view_as[DType.uint16](ctx, dst_h, count, "the bf16 bits")
    bf16_narrow(ctx, ddst, dsrc, count)
    ctx.synchronize()
    _ = dsrc^
    _ = ddst^
    return count


def from_bf16_dev_native(dst_h: Int, src_h: Int, params: List[Int]) raises -> Int:
    """`from_bf16` on device arrays: handle dst_h (count float32, written) from handle src_h (count uint16 bf16 bits);
    `bf16_widen` as `from_bf16_native`. Returns count."""
    lin_dev_ready("from_bf16")
    if len(params) != 1:
        raise Error("from_bf16: params must contain 1 value (count)")
    var count = params[0]
    if count <= 0:
        raise Error("from_bf16: count must be positive, got " + String(count))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dsrc = train_dev_view_as[DType.uint16](ctx, src_h, count, "bits")
    var ddst = train_dev_view(ctx, dst_h, count, "the widened values")
    bf16_widen(ctx, ddst, dsrc, count)
    ctx.synchronize()
    _ = dsrc^
    _ = ddst^
    return count


def quantize_int8_dev_native(q_h: Int, e_h: Int, x_h: Int, params: List[Int]) raises -> Int:
    """`quantize_int8` on device arrays: codes q_h (rows x cols int8, written) and row exponents e_h (rows int32,
    written) of x_h (rows x cols float32); `quantize_rows_int8_device` as `quantize_int8_native`. Returns rows * cols."""
    lin_dev_ready("quantize_int8")
    if len(params) != 2:
        raise Error("quantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = train_dev_view(ctx, x_h, rows * cols, "x")
    var dq = train_dev_view_as[DType.int8](ctx, q_h, rows * cols, "the codes")
    var de = train_dev_view_as[DType.int32](ctx, e_h, rows, "the exponents")
    quantize_rows_int8_device(ctx, dq, de, dx, rows, cols)
    ctx.synchronize()
    _ = dx^
    _ = dq^
    _ = de^
    return rows * cols


def dequantize_int8_dev_native(y_h: Int, q_h: Int, e_h: Int, params: List[Int]) raises -> Int:
    """`dequantize_int8` on device arrays: y_h (rows x cols float32, written) from codes q_h and exponents e_h;
    `dequantize_rows_int8_device` as `dequantize_int8_native`. Returns rows * cols."""
    lin_dev_ready("dequantize_int8")
    if len(params) != 2:
        raise Error("dequantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dq = train_dev_view_as[DType.int8](ctx, q_h, rows * cols, "codes")
    var de = train_dev_view_as[DType.int32](ctx, e_h, rows, "exponents")
    var dy = train_dev_view(ctx, y_h, rows * cols, "the values")
    dequantize_rows_int8_device(ctx, dy, dq, de, rows, cols)
    ctx.synchronize()
    _ = dq^
    _ = de^
    _ = dy^
    return rows * cols


def gemm_int8_dev_native(c_h: Int, qa_h: Int, ea_h: Int, qb_h: Int, eb_h: Int, params: List[Int]) raises -> Int:
    """`matmul_int8` on device arrays: c_h (m x n float32, written) from the codes and exponents of a (m x k) and b
    (n x k, OP_NT); `identical_gemm_int8_into` as `gemm_int8_native`. Returns m * n."""
    lin_dev_ready("matmul_int8")
    if len(params) != 3:
        raise Error("gemm_int8: params must contain 3 values (m, n, k), got " + String(len(params)))
    var m = params[0]
    var n = params[1]
    var k = params[2]
    _refuse_lowbit_shape(m, n, k, OP_NT, String("gemm_int8"))
    if k > INT8_MAX_K:
        raise Error("gemm_int8: k must be at most " + String(INT8_MAX_K) + " (contract L-7), got " + String(k))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dqa = train_dev_view_as[DType.int8](ctx, qa_h, m * k, "a codes")
    var dqb = train_dev_view_as[DType.int8](ctx, qb_h, n * k, "b codes")
    var dea = train_dev_view_as[DType.int32](ctx, ea_h, m, "a exponents")
    var deb = train_dev_view_as[DType.int32](ctx, eb_h, n, "b exponents")
    var dc = train_dev_view(ctx, c_h, m * n, "the product")
    identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    _ = dqa^
    _ = dea^
    _ = dqb^
    _ = deb^
    _ = dc^
    return m * n


def gemm_bf16_dev_native(c_h: Int, a_h: Int, b_h: Int, params: List[Int]) raises -> Int:
    """`matmul_bf16` on device arrays: c_h (m x n float32, written) = op(a) . op(b), b uint16 bf16 bits, a float32 or
    (params[4] != 0) bf16 bits widened first; `bf16_widen` + `identical_gemm_bf16w_into` as `gemm_bf16_native` (the
    identical route; the Apple fast simdgroup arm is not in an identical set). Returns m * n."""
    lin_dev_ready("matmul_bf16")
    if len(params) != 5:
        raise Error("gemm_bf16: params must contain 5 values (m, n, k, op, a_bf16), got " + String(len(params)))
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    var a_bf16 = params[4] != 0
    _refuse_lowbit_shape(m, n, k, op, String("gemm_bf16"))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var db = train_dev_view_as[DType.uint16](ctx, b_h, n * k, "b")
    var dc = train_dev_view(ctx, c_h, m * n, "the product")
    var work = LowbitWorkspace(ctx)
    if a_bf16:
        var da_bits = train_dev_view_as[DType.uint16](ctx, a_h, m * k, "a")
        var da = ctx.enqueue_create_buffer[DType.float32](m * k)
        bf16_widen(ctx, da, da_bits, m * k)
        identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
        ctx.synchronize()
        _ = da_bits^
        _ = da^
    else:
        var da = train_dev_view(ctx, a_h, m * k, "a")
        identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
        ctx.synchronize()
        _ = da^
    _ = work^
    _ = db^
    _ = dc^
    return m * n
