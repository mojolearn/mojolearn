# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Float32 single-label log loss: clipped probabilities, fixed GPU reduction.

Inputs are prevalidated encoded labels and row-major probabilities. No
renormalization. Float32 epsilon clipping bounds every logarithm away from zero.
"""
from std.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_div, identical_log
from metrics.checks.pinned_sum import PINNED_SUM_W, virtual_block_sum, chunk_count, linear_block_id, physical_block_count
from metrics.checks.device_io import download_f32


def log_loss_chunks_kernel[block_size: Int](
    truth: MutPointer[Int32, MutAnyOrigin], probability: MutPointer[Float32, MutAnyOrigin],
    n: Int32, k: Int32, partials: MutPointer[Float32, MutAnyOrigin],
):
    comptime assert block_size > 0 and block_size <= PINNED_SUM_W and (block_size & (block_size-1)) == 0
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunk = linear_block_id()
    while chunk < chunk_count(Int(n)):
        var values = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                var p = probability.unsafe_load(i*Int(k)+Int(truth.unsafe_load(i)))
                p = min(max(p, Float32(0.00000011920928955078125)), Float32(0.99999988079071044921875))
                values[r] = ftz(-identical_log(p))
        var total = virtual_block_sum[block_size](values)
        if tid == 0:
            partials.unsafe_store(chunk, ftz(total))
        chunk += physical_block_count()


def log_loss_finalize_kernel(
    partials: MutPointer[Float32, MutAnyOrigin], chunks: Int32, n: Int32,
    normalize: Int32, result: MutPointer[Float32, MutAnyOrigin],
):
    if Int(thread_idx.x) == 0:
        var total = Float32(0)
        for c in range(Int(chunks)):
            total = ftz(total + partials.unsafe_load(c))
        if normalize != 0:
            total = ftz(identical_div(total, Float32(n)))
        result.unsafe_store(0, total)


def log_loss[block_size: Int = 256](
    ctx: DeviceContext, mut truth: DeviceBuffer[DType.int32],
    mut probability: DeviceBuffer[DType.float32], n: Int, k: Int, normalize: Int,
) raises -> Float32:
    if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
        raise Error("log_loss: invalid input dimensions")
    if n > len(truth) or n*k > len(probability) or normalize < 0 or normalize > 1:
        raise Error("log_loss: invalid input length or normalization")
    var chunks = chunk_count(n)
    var partials = ctx.enqueue_create_buffer[DType.float32](chunks)
    var result = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[log_loss_chunks_kernel[block_size]](
        truth.unsafe_ptr(), probability.unsafe_ptr(), Int32(n), Int32(k), partials.unsafe_ptr(),
        grid_dim=chunks, block_dim=block_size,
    )
    ctx.enqueue_function[log_loss_finalize_kernel](
        partials.unsafe_ptr(), Int32(chunks), Int32(n), Int32(normalize), result.unsafe_ptr(),
        grid_dim=1, block_dim=32,
    )
    var value = download_f32(ctx,result,1)[0]
    _ = partials^
    _ = result^
    return value
