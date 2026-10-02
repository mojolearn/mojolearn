# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Float32 single-label log loss: clipped probabilities, fixed GPU reduction.

Inputs are prevalidated encoded labels and row-major probabilities. No
renormalization. Float32 epsilon clipping bounds every logarithm away from zero.
The chunk partials fold on the device as the fixed slab tree of
`metrics/checks/pinned_sum.mojo` (`fold_partials_levels`); the last level and
the optional mean are one launch.
"""
from std.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_div, identical_log
from metrics.checks.pinned_sum import (
    PINNED_SUM_W, PINNED_SUM_TPB, virtual_block_sum, chunk_count, linear_block_id, physical_block_count,
    fold_partials_levels, fold_scratch_len, last_level_values,
)
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


def log_loss_finalize_kernel[block_size: Int](
    partials: MutPointer[Float32, MutAnyOrigin], m: Int32, count: Int32,
    normalize: Int32, result: MutPointer[Float32, MutAnyOrigin],
):
    """The last level of the partial tree (`m <= PINNED_SUM_W` partials),
    then the optional mean on thread 0."""
    var total = virtual_block_sum[block_size](last_level_values[block_size](partials, Int(m)))
    if Int(thread_idx.x) == 0:
        total = ftz(total)
        if normalize != 0:
            total = ftz(identical_div(total, Float32(count)))
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
    var s0 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var s1 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var result = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[log_loss_chunks_kernel[block_size]](
        truth.unsafe_ptr(), probability.unsafe_ptr(), Int32(n), Int32(k), partials.unsafe_ptr(),
        grid_dim=chunks, block_dim=block_size,
    )
    var lv = fold_partials_levels(
        ctx, rebind[MutPointer[Float32, MutAnyOrigin]](partials.unsafe_ptr()), chunks, s0, s1
    )
    ctx.enqueue_function[log_loss_finalize_kernel[PINNED_SUM_TPB]](
        lv[0], Int32(lv[1]), Int32(n), Int32(normalize), result.unsafe_ptr(),
        grid_dim=chunk_count(lv[1]), block_dim=PINNED_SUM_TPB,
    )
    var value = download_f32(ctx,result,1)[0]
    _ = partials^
    _ = s0^
    _ = s1^
    _ = result^
    return value
