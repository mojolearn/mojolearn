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
from core.train_dev_pool import train_dev_view
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NN, OP_NT, OP_TN

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
