# SPDX-License-Identifier: Apache-2.0
"""Opt-in device forward/loss/backward for chunked LM-head v2."""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz, identical_div, identical_exp, identical_fmax, identical_log,
    identical_mul_add,
)
from training.checks.loss_oracle import CE_NEG_INF_BITS, neg_by_bits
from std.memory import bitcast
from gemm.checks.gemm_identical import identical_gemm_into
from gemm.checks.gemm_oracle import OP_NT

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


def chunked_lm_head_v2_dhidden_kernel(
    d_hidden: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, vocab_in: Int32, width_in: Int32,
):
    """One row owner: every dHidden cell folds vocabulary globally ascending."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_in)
    if row >= rows:
        return
    var vocab = Int(vocab_in)
    var width = Int(width_in)
    for feature in range(width):
        d_hidden.unsafe_store(row * width + feature, Float32(0.0))
    var maximum = row_max.unsafe_load(row)
    var denominator = row_denom.unsafe_load(row)
    var target_id = Int(targets.unsafe_load(row))
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var chunk1 = min(chunk0 + LM_HEAD_V2_CHUNK, vocab)
        for token in range(chunk0, chunk1):
            var shifted = ftz(ftz(_device_logit(hidden, weight, row, token, width)) - ftz(maximum))
            var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denominator)))
            var target = Float32(1.0) if token == target_id else Float32(0.0)
            var dlogit = ftz(identical_div(ftz(probability - target), Float32(rows)))
            for feature in range(width):
                var cell = row * width + feature
                d_hidden.unsafe_store(cell, identical_mul_add(
                    dlogit, weight.unsafe_load(token * width + feature),
                    d_hidden.unsafe_load(cell),
                ))
    for feature in range(width):
        var cell = row * width + feature
        d_hidden.unsafe_store(cell, ftz(d_hidden.unsafe_load(cell)))


def chunked_lm_head_v2_dweight_kernel(
    d_weight: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, vocab_in: Int32, width_in: Int32,
):
    """One vocabulary owner: every dWeight cell folds rows ascending."""
    var token = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var vocab = Int(vocab_in)
    if token >= vocab:
        return
    var rows = Int(rows_in)
    var width = Int(width_in)
    for feature in range(width):
        d_weight.unsafe_store(token * width + feature, Float32(0.0))
    for row in range(rows):
        var shifted = ftz(ftz(_device_logit(hidden, weight, row, token, width)) - ftz(row_max.unsafe_load(row)))
        var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(row_denom.unsafe_load(row))))
        var target = Float32(1.0) if token == Int(targets.unsafe_load(row)) else Float32(0.0)
        var dlogit = ftz(identical_div(ftz(probability - target), Float32(rows)))
        for feature in range(width):
            var cell = token * width + feature
            d_weight.unsafe_store(cell, identical_mul_add(
                dlogit, hidden.unsafe_load(row * width + feature),
                d_weight.unsafe_load(cell),
            ))
    for feature in range(width):
        var cell = token * width + feature
        d_weight.unsafe_store(cell, ftz(d_weight.unsafe_load(cell)))


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


def chunked_lm_head_v2_train_host(
    ctx: DeviceContext,
    loss_out: MutPointer[Float32, MutUntrackedOrigin],
    max_out: MutPointer[Float32, MutUntrackedOrigin],
    denom_out: MutPointer[Float32, MutUntrackedOrigin],
    d_hidden_out: MutPointer[Float32, MutUntrackedOrigin],
    d_weight_out: MutPointer[Float32, MutUntrackedOrigin],
    hidden_in: MutPointer[Float32, MutUntrackedOrigin],
    weight_in: MutPointer[Float32, MutUntrackedOrigin],
    targets_in: MutPointer[Int32, MutUntrackedOrigin],
    rows: Int, vocab: Int, width: Int,
) raises -> Int:
    """Explicit opt-in training selector; v1 remains every trainer's default."""
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
    var d_hidden = ctx.enqueue_create_buffer[DType.float32](rows * width)
    var d_weight = ctx.enqueue_create_buffer[DType.float32](vocab * width)
    ctx.enqueue_copy(dst_buf=hidden, src_ptr=hidden_in)
    ctx.enqueue_copy(dst_buf=weight, src_ptr=weight_in)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=targets_in)
    var row_grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    var vocab_grid = (vocab + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    ctx.enqueue_function[chunked_lm_head_v2_rows_kernel](
        maxima.unsafe_ptr(), denom.unsafe_ptr(), row_loss.unsafe_ptr(),
        hidden.unsafe_ptr(), weight.unsafe_ptr(), targets.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=row_grid,
        block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_function[chunked_lm_head_v2_total_kernel](
        loss.unsafe_ptr(), row_loss.unsafe_ptr(), Int32(rows), grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[chunked_lm_head_v2_dhidden_kernel](
        d_hidden.unsafe_ptr(), hidden.unsafe_ptr(), weight.unsafe_ptr(),
        targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=row_grid,
        block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_function[chunked_lm_head_v2_dweight_kernel](
        d_weight.unsafe_ptr(), hidden.unsafe_ptr(), weight.unsafe_ptr(),
        targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=vocab_grid,
        block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_copy(dst_ptr=loss_out, src_buf=loss)
    ctx.enqueue_copy(dst_ptr=max_out, src_buf=maxima)
    ctx.enqueue_copy(dst_ptr=denom_out, src_buf=denom)
    ctx.enqueue_copy(dst_ptr=d_hidden_out, src_buf=d_hidden)
    ctx.enqueue_copy(dst_ptr=d_weight_out, src_buf=d_weight)
    ctx.synchronize()
    return rows


def chunked_lm_head_v2_forward_into(
    ctx: DeviceContext, mut loss: DeviceBuffer[DType.float32],
    mut maxima: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut row_loss: DeviceBuffer[DType.float32],
    mut hidden: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32], rows: Int, vocab: Int, width: Int,
) raises:
    """Device-resident V2 forward for an explicitly selected trainer."""
    var row_grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    ctx.enqueue_function[chunked_lm_head_v2_rows_kernel](
        maxima.unsafe_ptr(), denom.unsafe_ptr(), row_loss.unsafe_ptr(),
        hidden.unsafe_ptr(), weight.unsafe_ptr(), targets.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=row_grid,
        block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_function[chunked_lm_head_v2_total_kernel](
        loss.unsafe_ptr(), row_loss.unsafe_ptr(), Int32(rows), grid_dim=1, block_dim=1,
    )


def chunked_lm_head_v2_backward_into(
    ctx: DeviceContext, mut d_hidden: DeviceBuffer[DType.float32],
    mut d_weight: DeviceBuffer[DType.float32],
    mut hidden: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    mut maxima: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32], rows: Int, vocab: Int, width: Int,
) raises:
    """Device-resident V2 backward, consuming the matching forward statistics."""
    var row_grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    var vocab_grid = (vocab + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    ctx.enqueue_function[chunked_lm_head_v2_dhidden_kernel](
        d_hidden.unsafe_ptr(), hidden.unsafe_ptr(), weight.unsafe_ptr(),
        targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=row_grid,
        block_dim=LM_HEAD_V2_TPB,
    )
    ctx.enqueue_function[chunked_lm_head_v2_dweight_kernel](
        d_weight.unsafe_ptr(), hidden.unsafe_ptr(), weight.unsafe_ptr(),
        targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=vocab_grid,
        block_dim=LM_HEAD_V2_TPB,
    )


def _chunk_max_kernel(maxima: MutPointer[Float32, MutAnyOrigin], logits: MutPointer[Float32, MutAnyOrigin], rows: Int32, n: Int32, first: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows): return
    var acc = bitcast[DType.float32](CE_NEG_INF_BITS) if first != 0 else maxima.unsafe_load(row)
    for j in range(Int(n)):
        acc = identical_fmax(acc, logits.unsafe_load(row * Int(n) + j))
    maxima.unsafe_store(row, acc)


def _chunk_denom_kernel(denom: MutPointer[Float32, MutAnyOrigin], target_shift: MutPointer[Float32, MutAnyOrigin], logits: MutPointer[Float32, MutAnyOrigin], maxima: MutPointer[Float32, MutAnyOrigin], targets: MutPointer[Int32, MutAnyOrigin], rows: Int32, n: Int32, chunk0: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows): return
    var acc = Float32(0.0) if chunk0 == 0 else denom.unsafe_load(row)
    for j in range(Int(n)):
        var shifted = ftz(ftz(logits.unsafe_load(row * Int(n) + j)) - ftz(maxima.unsafe_load(row)))
        acc = ftz(acc + ftz(identical_exp(shifted)))
        if Int(chunk0) + j == Int(targets.unsafe_load(row)):
            target_shift.unsafe_store(row, shifted)
    denom.unsafe_store(row, acc)


def _chunk_loss_kernel(row_loss: MutPointer[Float32, MutAnyOrigin], denom: MutPointer[Float32, MutAnyOrigin], rows: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows): return
    row_loss.unsafe_store(row, neg_by_bits(ftz(ftz(row_loss.unsafe_load(row)) - ftz(identical_log(ftz(denom.unsafe_load(row)))))))


def _chunk_dhidden_kernel(d_hidden: MutPointer[Float32, MutAnyOrigin], logits: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin], targets: MutPointer[Int32, MutAnyOrigin], maxima: MutPointer[Float32, MutAnyOrigin], denom: MutPointer[Float32, MutAnyOrigin], rows: Int32, n: Int32, width: Int32, chunk0: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows): return
    if chunk0 == 0:
        for f in range(Int(width)): d_hidden.unsafe_store(row * Int(width) + f, Float32(0.0))
    for j in range(Int(n)):
        var shifted = ftz(ftz(logits.unsafe_load(row * Int(n) + j)) - ftz(maxima.unsafe_load(row)))
        var p = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denom.unsafe_load(row))))
        var target = Float32(1.0) if Int(chunk0) + j == Int(targets.unsafe_load(row)) else Float32(0.0)
        var dl = ftz(identical_div(ftz(p - target), Float32(rows)))
        for f in range(Int(width)):
            var cell = row * Int(width) + f
            d_hidden.unsafe_store(cell, identical_mul_add(dl, weight.unsafe_load((Int(chunk0) + j) * Int(width) + f), d_hidden.unsafe_load(cell)))


def _chunk_dweight_kernel(d_weight: MutPointer[Float32, MutAnyOrigin], logits: MutPointer[Float32, MutAnyOrigin], hidden: MutPointer[Float32, MutAnyOrigin], targets: MutPointer[Int32, MutAnyOrigin], maxima: MutPointer[Float32, MutAnyOrigin], denom: MutPointer[Float32, MutAnyOrigin], rows: Int32, n: Int32, width: Int32, chunk0: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= Int(n): return
    var token = Int(chunk0) + j
    # Every feature consumes the same dlogit for this (row, token). Compute
    # it once, while retaining each cell's exact row-ascending FMA chain.
    for f in range(Int(width)):
        d_weight.unsafe_store(token * Int(width) + f, Float32(0.0))
    for row in range(Int(rows)):
        var shifted = ftz(ftz(logits.unsafe_load(row * Int(n) + j)) - ftz(maxima.unsafe_load(row)))
        var p = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denom.unsafe_load(row))))
        var target = Float32(1.0) if token == Int(targets.unsafe_load(row)) else Float32(0.0)
        var dl = ftz(identical_div(ftz(p - target), Float32(rows)))
        for f in range(Int(width)):
            var cell = token * Int(width) + f
            d_weight.unsafe_store(cell, identical_mul_add(
                dl, hidden.unsafe_load(row * Int(width) + f),
                d_weight.unsafe_load(cell),
            ))
    for f in range(Int(width)):
        var cell = token * Int(width) + f
        d_weight.unsafe_store(cell, ftz(d_weight.unsafe_load(cell)))


def chunked_lm_head_v2_gemm_forward_into(ctx: DeviceContext, mut loss: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32], mut denom: DeviceBuffer[DType.float32], mut row_loss: DeviceBuffer[DType.float32], mut chunk: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], mut hidden: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32], mut targets: DeviceBuffer[DType.int32], rows: Int, vocab: Int, width: Int) raises:
    var grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_chunk_max_kernel](maxima.unsafe_ptr(), chunk.unsafe_ptr(), Int32(rows), Int32(n), Int32(1 if chunk0 == 0 else 0), grid_dim=grid, block_dim=LM_HEAD_V2_TPB)
        _ = wb
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_chunk_denom_kernel](denom.unsafe_ptr(), row_loss.unsafe_ptr(), chunk.unsafe_ptr(), maxima.unsafe_ptr(), targets.unsafe_ptr(), Int32(rows), Int32(n), Int32(chunk0), grid_dim=grid, block_dim=LM_HEAD_V2_TPB)
        _ = wb
    ctx.enqueue_function[_chunk_loss_kernel](row_loss.unsafe_ptr(), denom.unsafe_ptr(), Int32(rows), grid_dim=grid, block_dim=LM_HEAD_V2_TPB)
    ctx.enqueue_function[chunked_lm_head_v2_total_kernel](loss.unsafe_ptr(), row_loss.unsafe_ptr(), Int32(rows), grid_dim=1, block_dim=1)


def chunked_lm_head_v2_gemm_backward_into(ctx: DeviceContext, mut d_hidden: DeviceBuffer[DType.float32], mut d_weight: DeviceBuffer[DType.float32], mut chunk: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], mut hidden: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32], mut targets: DeviceBuffer[DType.int32], mut maxima: DeviceBuffer[DType.float32], mut denom: DeviceBuffer[DType.float32], rows: Int, vocab: Int, width: Int) raises:
    var row_grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_chunk_dhidden_kernel](d_hidden.unsafe_ptr(), chunk.unsafe_ptr(), weight.unsafe_ptr(), targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(), Int32(rows), Int32(n), Int32(width), Int32(chunk0), grid_dim=row_grid, block_dim=LM_HEAD_V2_TPB)
        ctx.enqueue_function[_chunk_dweight_kernel](d_weight.unsafe_ptr(), chunk.unsafe_ptr(), hidden.unsafe_ptr(), targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(), Int32(rows), Int32(n), Int32(width), Int32(chunk0), grid_dim=(n + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB, block_dim=LM_HEAD_V2_TPB)
        _ = wb
