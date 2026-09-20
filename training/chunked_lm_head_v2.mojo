# SPDX-License-Identifier: Apache-2.0
"""Opt-in device forward/loss for chunked LM-head v2; v1 is untouched."""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz, identical_div, identical_exp, identical_fmax, identical_log,
    identical_mul_add,
)
from training.checks.loss_oracle import CE_NEG_INF_BITS, neg_by_bits
from std.memory import bitcast

comptime LM_HEAD_V2_CHUNK = 256
comptime LM_HEAD_V2_TPB = 256


def _device_logit(
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    row: Int, token: Int, width: Int,
) -> Float32:
    var acc = Float32(0.0)
    for feature in range(width):
        acc = identical_mul_add(
            hidden.unsafe_load(row * width + feature),
            weight.unsafe_load(token * width + feature), acc,
        )
    return ftz(acc)


def chunked_lm_head_v2_rows_kernel(
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    row_loss: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32, vocab_in: Int32, width_in: Int32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_in)
    if row >= rows:
        return
    var vocab = Int(vocab_in)
    var width = Int(width_in)
    var maximum = bitcast[DType.float32](CE_NEG_INF_BITS)
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var chunk1 = min(chunk0 + LM_HEAD_V2_CHUNK, vocab)
        for token in range(chunk0, chunk1):
            maximum = identical_fmax(
                maximum, _device_logit(hidden, weight, row, token, width)
            )
    row_max.unsafe_store(row, maximum)
    var denom = Float32(0.0)
    var target_shift = Float32(0.0)
    var target = Int(targets.unsafe_load(row))
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var chunk1 = min(chunk0 + LM_HEAD_V2_CHUNK, vocab)
        for token in range(chunk0, chunk1):
            var shifted = ftz(
                ftz(_device_logit(hidden, weight, row, token, width)) - ftz(maximum)
            )
            denom = ftz(denom + ftz(identical_exp(shifted)))
            if token == target:
                target_shift = shifted
    row_denom.unsafe_store(row, denom)
    row_loss.unsafe_store(
        row, neg_by_bits(ftz(ftz(target_shift) - ftz(identical_log(ftz(denom)))))
    )


def chunked_lm_head_v2_total_kernel(
    loss: MutPointer[Float32, MutAnyOrigin],
    row_loss: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
):
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var rows = Int(rows_in)
    var total = Float32(0.0)
    for row in range(rows):
        total = ftz(total + row_loss.unsafe_load(row))
    loss.unsafe_store(0, ftz(identical_div(total, Float32(rows))))


def chunked_lm_head_v2_loss_host(
    ctx: DeviceContext,
    loss_out: MutPointer[Float32, MutUntrackedOrigin],
    max_out: MutPointer[Float32, MutUntrackedOrigin],
    denom_out: MutPointer[Float32, MutUntrackedOrigin],
    hidden_in: MutPointer[Float32, MutUntrackedOrigin],
    weight_in: MutPointer[Float32, MutUntrackedOrigin],
    targets_in: MutPointer[Int32, MutUntrackedOrigin],
    rows: Int, vocab: Int, width: Int,
) raises -> Int:
    """Validate, run v2 on device, and return the number of consumed rows."""
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("chunked lm head v2: rows/width must be positive and vocab >= 2")
    for i in range(rows * width):
        if not isfinite(hidden_in.unsafe_load(i)):
            raise Error("chunked lm head v2: non-finite hidden at flat index " + String(i))
    for i in range(vocab * width):
        if not isfinite(weight_in.unsafe_load(i)):
            raise Error("chunked lm head v2: non-finite weight at flat index " + String(i))
    for row in range(rows):
        var target = targets_in.unsafe_load(row)
        if target < 0 or Int(target) >= vocab:
            raise Error("chunked lm head v2: target outside [0, vocab) at row " + String(row))
    var hidden = ctx.enqueue_create_buffer[DType.float32](rows * width)
    var weight = ctx.enqueue_create_buffer[DType.float32](vocab * width)
    var targets = ctx.enqueue_create_buffer[DType.int32](rows)
    var maxima = ctx.enqueue_create_buffer[DType.float32](rows)
    var denom = ctx.enqueue_create_buffer[DType.float32](rows)
    var row_loss = ctx.enqueue_create_buffer[DType.float32](rows)
    var loss = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=hidden, src_ptr=hidden_in)
    ctx.enqueue_copy(dst_buf=weight, src_ptr=weight_in)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=targets_in)
    var grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    ctx.enqueue_function[chunked_lm_head_v2_rows_kernel](
        maxima.unsafe_ptr(), denom.unsafe_ptr(), row_loss.unsafe_ptr(),
        hidden.unsafe_ptr(), weight.unsafe_ptr(), targets.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width),
        grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_function[chunked_lm_head_v2_total_kernel](
        loss.unsafe_ptr(), row_loss.unsafe_ptr(), Int32(rows),
        grid_dim=1, block_dim=1,
    )
    ctx.enqueue_copy(dst_ptr=loss_out, src_buf=loss)
    ctx.enqueue_copy(dst_ptr=max_out, src_buf=maxima)
    ctx.enqueue_copy(dst_ptr=denom_out, src_buf=denom)
    ctx.synchronize()
    return rows
