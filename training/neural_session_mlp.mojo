# SPDX-License-Identifier: Apache-2.0
"""NN64 full stateless MLP session batching, with shared immutable call weights.

A session is one independent set of input rows. All use the same two-layer
Linear/ReLU/Linear model, FP32 profile and weights passed to this call. There
is no cache or RNG state to merge. A batches all rows through each projection;
B runs each session separately. Both include upload, admission and consumed
logits. This deliberately does not pretend to implement stateful decode.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from training.neural_ab_lifetime import NN64_SESSION_PACK
from training.mlp_ops import _mlp_kernel
from training.neural_session_shape import nn_mlp_session_shape
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NT
from core.device_scan import device_first_nonfinite

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]


def nn_mlp_sessions_device(ctx: DeviceContext, inputs: List[Int], rows: List[Int],
    w1: _FP, b1: _FP, w2: _FP, b2: _FP, output: _FP,
    in_width: Int, hidden: Int, out_width: Int,
) raises -> Int:
    var total = nn_mlp_session_shape(inputs, rows, in_width, hidden, out_width)
    var x = ctx.enqueue_create_buffer[DType.float32](max(1, total * in_width))
    var a = ctx.enqueue_create_buffer[DType.float32](max(1, total * hidden))
    var y = ctx.enqueue_create_buffer[DType.float32](max(1, total * out_width))
    var dw1 = ctx.enqueue_create_buffer[DType.float32](hidden * in_width)
    var db1 = ctx.enqueue_create_buffer[DType.float32](hidden)
    var dw2 = ctx.enqueue_create_buffer[DType.float32](out_width * hidden)
    var db2 = ctx.enqueue_create_buffer[DType.float32](out_width)
    ctx.enqueue_copy(dst_buf=dw1, src_ptr=w1)
    ctx.enqueue_copy(dst_buf=db1, src_ptr=b1)
    ctx.enqueue_copy(dst_buf=dw2, src_ptr=w2)
    ctx.enqueue_copy(dst_buf=db2, src_ptr=b2)
    var keep = List[DeviceBuffer[DType.float32]]()
    var base = 0
    for session in range(len(rows)):
        if rows[session] > 0:
            var part = x.create_sub_buffer[DType.float32](base * in_width, rows[session] * in_width)
            ctx.enqueue_copy(dst_buf=part, src_ptr=_FP(unsafe_from_address=inputs[session]))
            keep.append(part^)
        base += rows[session]
    ctx.synchronize()
    if (device_first_nonfinite(ctx, dw1, hidden * in_width) >= 0
        or device_first_nonfinite(ctx, db1, hidden) >= 0
        or device_first_nonfinite(ctx, dw2, out_width * hidden) >= 0
        or device_first_nonfinite(ctx, db2, out_width) >= 0):
        raise Error("neural MLP sessions: nonfinite weights")
    if total == 0:
        return 0
    if device_first_nonfinite(ctx, x, total * in_width) >= 0:
        raise Error("neural MLP sessions: nonfinite input")
    var workspace_cells = max(identical_gemm_workspace_max_floats(total, hidden, in_width),
        identical_gemm_workspace_max_floats(total, out_width, hidden))
    # Dispatch can choose different scratch plans on a session and the packed
    # batch, so size against every actual shape, not a monotonicity assumption.
    for i in range(len(rows)):  # small-loop(rows: per-session row counts): sizes the scratch over sessions, not rows
        if rows[i] > 0:
            workspace_cells = max(workspace_cells, max(
                identical_gemm_workspace_max_floats(rows[i], hidden, in_width),
                identical_gemm_workspace_max_floats(rows[i], out_width, hidden)))
    var ws = ctx.enqueue_create_buffer[DType.float32](max(1, workspace_cells))
    comptime if NN64_SESSION_PACK:
        nn_mlp_session_forward(ctx, x, dw1, db1, dw2, db2, a, y, ws,
            total, in_width, hidden, out_width)
    else:
        base = 0
        for session in range(len(rows)):
            var count = rows[session]
            if count > 0:
                var xv = x.create_sub_buffer[DType.float32](base * in_width, count * in_width)
                var av = a.create_sub_buffer[DType.float32](base * hidden, count * hidden)
                var yv = y.create_sub_buffer[DType.float32](base * out_width, count * out_width)
                nn_mlp_session_forward(ctx, xv, dw1, db1, dw2, db2, av, yv, ws,
                    count, in_width, hidden, out_width)
                keep.append(xv^)
                keep.append(av^)
                keep.append(yv^)
            base += count
    if device_first_nonfinite(ctx, y, total * out_width) >= 0:
        raise Error("neural MLP sessions: nonfinite logits")
    ctx.enqueue_copy(dst_ptr=output, src_buf=y)
    ctx.synchronize()
    _ = keep^
    _ = x^
    _ = a^
    _ = y^
    _ = dw1^
    _ = db1^
    _ = dw2^
    _ = db2^
    _ = ws^
    return total * out_width


def nn_mlp_session_forward(ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32], mut w1: DeviceBuffer[DType.float32],
    mut b1: DeviceBuffer[DType.float32], mut w2: DeviceBuffer[DType.float32],
    mut b2: DeviceBuffer[DType.float32], mut hidden_values: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32],
    rows: Int, in_width: Int, hidden: Int, out_width: Int,
) raises:
    identical_gemm_into(ctx, hidden_values, x, w1, ws, rows, hidden, in_width, OP_NT)
    # Bias and activation operate on independent cells; 128 is launch geometry,
    # never a fold or a model-dimension route. Original MLP seams are reused.
    # Each cell reads and writes itself; take one raw view for in-place use.
    var hidden_ptr = hidden_values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[_mlp_kernel](hidden_ptr, b1.unsafe_ptr(), hidden_ptr, Int32(rows), Int32(hidden), Int32(1), grid_dim=((rows * hidden + 127) // 128, 1, 1), block_dim=(128, 1, 1))
    identical_gemm_into(ctx, output, hidden_values, w2, ws, rows, out_width, hidden, OP_NT)
    var output_ptr = output.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[_mlp_kernel](output_ptr, b2.unsafe_ptr(), output_ptr, Int32(rows), Int32(out_width), Int32(0), grid_dim=((rows * out_width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
