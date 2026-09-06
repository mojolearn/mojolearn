# SPDX-License-Identifier: Apache-2.0
"""Bounded IDENTICAL FP32 GPU operations for the public small MLP.

C-row-major inputs; rows 1..256 and columns 1..64. Bias addition and
ascending row sums use checks.numerics' pinned FMA with multiplier one,
flushing operands/results at each seam. No atomics or vendor reductions.
ReLU returns positive values, otherwise +0; its derivative is zero at zero.
This is new surface arithmetic awaiting root-run numerical qualification.
Host pointers are borrowed for a synchronous call and never retained.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add


def mlp_validate_shape(rows: Int, cols: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small MLP operations require IDENTICAL numeric mode")
    if rows < 1 or rows > 256 or cols < 1 or cols > 64:
        raise Error("small MLP operations require rows 1..256 and cols 1..64")


def _finite(values: MutPointer[Float32, MutUntrackedOrigin], count: Int) raises:
    for i in range(count):
        if not isfinite(values.unsafe_load(i)):
            raise Error("small MLP operation has nonfinite input or output")


def _add(left: Float32, right: Float32) -> Float32:
    # Existing pinned FP32 arithmetic: fma(1, left, right) is a single
    # rounded addition. Both operands and the result use the IDENTICAL FTZ.
    return ftz(identical_mul_add(Float32(1), ftz(left), ftz(right)))


def _mlp_kernel(
    source: MutPointer[Float32, MutAnyOrigin],
    other: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32, operation_arg: Int32,
):
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var operation = Int(operation_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if operation == 3:
        if i >= cols:
            return
        var value = Float32(0)
        # One GPU lane per column. The fold order does not depend on GPU
        # width, warp size, launch partition or scheduling.
        for row in range(rows):
            value = _add(value, source.unsafe_load(row * cols + i))
        output.unsafe_store(i, value)
        return
    if i >= rows * cols:
        return
    if operation == 2:
        var activation = source.unsafe_load(i)
        var value = Float32(0)
        if activation > Float32(0):
            value = ftz(other.unsafe_load(i))
        output.unsafe_store(i, value)
        return
    var value = _add(source.unsafe_load(i), other.unsafe_load(i % cols))
    if operation == 1 and value <= Float32(0):
        value = Float32(0)
    output.unsafe_store(i, value)


def _mlp_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    other_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int, operation: Int,
) raises -> Int:
    mlp_validate_shape(rows, cols)
    if operation < 0 or operation > 3:
        raise Error("invalid small MLP operation")
    var count = rows * cols
    var out_count = cols if operation == 3 else count
    var other_count = count if operation == 2 else cols
    if operation == 3:
        other_count = 1  # unused kernel operand; borrowed input placeholder
    _finite(input_ptr, count)
    if operation != 3:
        _finite(other_ptr, other_count)
    var source = ctx.enqueue_create_buffer[DType.float32](count)
    var other = ctx.enqueue_create_buffer[DType.float32](other_count)
    var output = ctx.enqueue_create_buffer[DType.float32](out_count)
    ctx.enqueue_copy(dst_buf=source, src_ptr=input_ptr)
    if operation != 3:
        ctx.enqueue_copy(dst_buf=other, src_ptr=other_ptr)
    ctx.synchronize()
    ctx.enqueue_function[_mlp_kernel](
        source.unsafe_ptr(), other.unsafe_ptr(), output.unsafe_ptr(),
        Int32(rows), Int32(cols), Int32(operation),
        grid_dim=((out_count + 127) // 128, 1, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=output)
    ctx.synchronize()
    # Explicit ownership after the drain prevents last-use buffer teardown
    # while a queued kernel or copy still references borrowed memory.
    _ = output^
    _ = other^
    _ = source^
    _finite(out_ptr, out_count)
    return out_count


def mlp_bias_activation_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    bias_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int, relu_flag: Int,
) raises -> Int:
    mlp_validate_shape(rows, cols)
    if relu_flag != 0 and relu_flag != 1:
        raise Error("mlp_bias_activation relu_flag must be 0 or 1")
    return _mlp_host(ctx, input_ptr, bias_ptr, out_ptr, rows, cols, relu_flag)


def mlp_relu_backward_host(
    ctx: DeviceContext,
    activation_ptr: MutPointer[Float32, MutUntrackedOrigin],
    incoming_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int,
) raises -> Int:
    return _mlp_host(ctx, activation_ptr, incoming_ptr, out_ptr, rows, cols, 2)


def mlp_sum_rows_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int,
) raises -> Int:
    return _mlp_host(ctx, input_ptr, input_ptr, out_ptr, rows, cols, 3)
