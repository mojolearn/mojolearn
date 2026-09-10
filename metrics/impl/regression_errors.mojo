# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Finite Float32 regression errors; all metric arithmetic stays on GPU.

Reuse the R2 slab reduction schedule, without its unnecessary target-mean /
SST pass or host final fold. Residual, square, partials and result are Float32;
overflow yields +inf, including RMSE when the MSE overflows. IDENTICAL flushes
operands and arithmetic seams and uses portable division/sqrt. No float atomics.
"""
from std.gpu import thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, portable_sqrtf
from metrics.checks.pinned_sum import PINNED_SUM_W, virtual_block_sum, chunk_count, linear_block_id, physical_block_count
from metrics.checks.device_io import download_f32


def error_chunks_kernel[absolute: Bool, block_size: Int](
    y: MutPointer[Float32, MutAnyOrigin],
    prediction: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    comptime assert block_size > 0 and block_size <= PINNED_SUM_W and (block_size & (block_size - 1)) == 0, "block_size must be a positive power of two dividing 256"
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunk = linear_block_id()
    while chunk < chunk_count(Int(n)):
        var values = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                # Flush operands as well: opposite-sign subnormal inputs can
                # otherwise produce a normal residual on a non-FTZ device.
                var difference = ftz(ftz(y.unsafe_load(i)) - ftz(prediction.unsafe_load(i)))
                comptime if absolute:
                    values[r] = abs(difference)
                else:
                    values[r] = ftz(difference * difference)
        var total = virtual_block_sum[block_size](values)
        if tid == 0:
            partials.unsafe_store(chunk, ftz(total))
        chunk += physical_block_count()


def error_finalize_kernel[root: Bool](
    partials: MutPointer[Float32, MutAnyOrigin],
    chunks: Int32,
    n: Int32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    if Int(thread_idx.x) == 0:
        var total = Float32(0.0)
        # Same ascending partial fold as existing metrics, executed on GPU.
        for c in range(Int(chunks)):
            total = ftz(total + partials.unsafe_load(c))
        var value = ftz(identical_div(total, Float32(n)))
        comptime if root:
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                value = portable_sqrtf(value)
            else:
                value = sqrt(value)
        result.unsafe_store(0, ftz(value))


def regression_error[absolute: Bool = False, root: Bool = False, block_size: Int = 256](
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut prediction: DeviceBuffer[DType.float32],
    n: Int,
    gx: Int = 0,
    gy: Int = 1,
) raises -> Float32:
    comptime assert not (absolute and root), "root MAE is not a regression metric"
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(prediction):
        raise Error("regression_error: invalid input length")
    if gx < 0 or gy <= 0:
        raise Error("regression_error: invalid grid")
    var chunks = chunk_count(n)
    var partials = ctx.enqueue_create_buffer[DType.float32](chunks)
    var result = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[error_chunks_kernel[absolute, block_size]](
        y.unsafe_ptr(), prediction.unsafe_ptr(), Int32(n), partials.unsafe_ptr(),
        grid_dim=(chunks if gx == 0 else gx, gy, 1), block_dim=block_size,
    )
    ctx.enqueue_function[error_finalize_kernel[root]](
        partials.unsafe_ptr(), Int32(chunks), Int32(n), result.unsafe_ptr(),
        grid_dim=1, block_dim=32,
    )
    var value = download_f32(ctx, result, 1)[0]
    _ = partials^
    _ = result^
    return value
