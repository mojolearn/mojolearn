# SPDX-License-Identifier: Apache-2.0
"""Opt-in device forward/loss/backward for chunked LM-head v2."""
from max.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL,
    ftz, identical_div, identical_exp, identical_fmax, identical_log,
    identical_mul_add,
)
from training.checks.loss_contract import CE_NEG_INF_BITS, neg_by_bits
from std.memory import bitcast
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NN, OP_NT, OP_TN
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from training.chunked_lm_head_panel_math import (
    LMHEAD_FAST_GEMM_PANEL, LMHEAD_GP_LEAF,
    lmhead_gp_leaf_count, lmhead_gp_stack_slots,
    lmhead_gp_leaf_body, lmhead_gp_level_body, lmhead_gp_row_loss_body,
    lmhead_gp_dlogit_body, lmhead_gp_merge_body,
)

# Four passes visit every vocabulary chunk (max, denominator/loss, dHidden,
# dWeight). 1024 keeps the workspace bounded while quartering the launch
# count versus the original 256-cell slice; chunk boundaries do not alter
# any row/token fold order.
# NN53: independently selectable storage/launch experiments, default OFF.
# 512 halves panel storage; 2048 halves panel launches versus 1024. These
# powers-of-two arise from workspace/launch tradeoffs, not a vocabulary row.
# The canonical vocabulary loop and GEMM contraction are unchanged. Existing
# chunked_lm_head_v2 must also be selected; no ordinary head changes profile.
# All compilation/identity/quality/full-model A/B remain intentionally unrun.
from training.chunked_lm_head_profile import NN53_CHUNK512, NN53_CHUNK2048, LM_HEAD_V2_CHUNK
comptime LM_HEAD_V2_TPB = 256


def _device_logit(
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    row: Int, token: Int, width: Int,
) -> Float32:
    var acc = Float32(0.0)
    for feature in range(width):
        # Flush inputs and each fused fold, as in the host oracle. Flushing
        # only the final sum lets device denormal handling change both the
        # accumulated value and the sign of a zero gradient.
        acc = ftz(identical_mul_add(
            ftz(hidden.unsafe_load(row * width + feature)),
            ftz(weight.unsafe_load(token * width + feature)), acc,
        ))
    return ftz(acc)


@always_inline
def chunked_lm_head_v2_row_body(  # device-helper: one row of the rows kernel, also the host arm's per-owner body
    row: Int,
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    row_loss: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    vocab: Int, width: Int,
):
    """One row's maximum, denominator and loss (the rows kernel's body; the
    host arm of the executor op, autograd/lm_head_v2_op.mojo, runs it per
    row too, so both columns compute the same words)."""
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
    if row >= Int(rows_in):
        return
    chunked_lm_head_v2_row_body(row, row_max, row_denom, row_loss, hidden, weight, targets,
                                Int(vocab_in), Int(width_in))


#: The mean loss's row fold (lane n-train-mamba, cpu-gpu-cleanup 2026-10-02):
#: leaves of LM_HEAD_V2_LOSS_LEAF rows, each folded ascending from +0.0 with
#: every add flushed, then the fixed balanced tree over the leaf partials in
#: ascending leaf order (pairs `ftz(ftz(l) + ftz(r))`, an odd tail carried bit
#: for bit), then `ftz(total / rows)`. The CPU oracle
#: (`training/checks/chunked_lm_head_oracle.mojo::lm_head_v2_loss_fold`) folds
#: the same way. At rows <= LM_HEAD_V2_LOSS_LEAF it is the old serial chain.
comptime LM_HEAD_V2_LOSS_LEAF = 256


def chunked_lm_head_v2_loss_leaf_kernel(
    loss: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    row_loss: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
):
    """One thread per leaf: its rows ascending. A single leaf writes the mean."""
    var rows = Int(rows_in)
    var leaves = (rows + LM_HEAD_V2_LOSS_LEAF - 1) // LM_HEAD_V2_LOSS_LEAF
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= leaves:
        return
    var lo = j * LM_HEAD_V2_LOSS_LEAF
    var hi = min(lo + LM_HEAD_V2_LOSS_LEAF, rows)
    var acc = Float32(0.0)
    for row in range(lo, hi):
        acc = ftz(acc + row_loss.unsafe_load(row))
    partials.unsafe_store(j, acc)
    if leaves == 1:
        loss.unsafe_store(0, ftz(identical_div(acc, Float32(rows))))


def chunked_lm_head_v2_loss_level_kernel(
    loss: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    width_in: Int32,
    rows_in: Int32,
):
    """One tree level, one thread per output node. The level that leaves one
    node writes the mean."""
    var width = Int(width_in)
    var out_width = (width + 1) // 2
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= out_width:
        return
    var node: Float32
    if 2 * q + 1 < width:
        node = ftz(ftz(src.unsafe_load(2 * q)) + ftz(src.unsafe_load(2 * q + 1)))
    else:
        node = src.unsafe_load(2 * q)
    dst.unsafe_store(q, node)
    if out_width == 1:
        loss.unsafe_store(0, ftz(identical_div(ftz(node), Float32(Int(rows_in)))))


def _enqueue_loss_total(
    ctx: DeviceContext,
    mut loss: DeviceBuffer[DType.float32],
    mut row_loss: DeviceBuffer[DType.float32],
    rows: Int,
) raises:
    """`loss[0]` = the mean of `row_loss[0:rows]` in the fixed fold above:
    one grid-wide leaf launch, then one launch per tree level."""
    var leaves = (rows + LM_HEAD_V2_LOSS_LEAF - 1) // LM_HEAD_V2_LOSS_LEAF
    var a = ctx.enqueue_create_buffer[DType.float32](max(1, leaves))
    var b = ctx.enqueue_create_buffer[DType.float32](max(1, (leaves + 1) // 2))
    try:
        ctx.enqueue_function[chunked_lm_head_v2_loss_leaf_kernel](
            loss.unsafe_ptr(), a.unsafe_ptr(), row_loss.unsafe_ptr(), Int32(rows),
            grid_dim=(leaves + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB,
            block_dim=LM_HEAD_V2_TPB,
        )
        var width = leaves
        var from_a = True
        while width > 1:
            var out_width = (width + 1) // 2
            var grid = (out_width + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
            if from_a:
                ctx.enqueue_function[chunked_lm_head_v2_loss_level_kernel](
                    loss.unsafe_ptr(), b.unsafe_ptr(), a.unsafe_ptr(),
                    Int32(width), Int32(rows),
                    grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
                )
            else:
                ctx.enqueue_function[chunked_lm_head_v2_loss_level_kernel](
                    loss.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
                    Int32(width), Int32(rows),
                    grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
                )
            from_a = not from_a
            width = out_width
    except error:
        ctx.synchronize()
        raise error
    # Raw-pointer consumers do not retain these local scratch owners. Charge
    # their last-consumer wait to the complete operation before destruction.
    ctx.synchronize()
    _ = a^
    _ = b^


@always_inline
def chunked_lm_head_v2_dhidden_body(  # device-helper: one row of the dHidden kernel, also the host arm's per-owner body
    row: Int,
    d_hidden: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows: Int, vocab: Int, width: Int,
):
    """One row owner: every dHidden cell folds vocabulary globally ascending."""
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
                d_hidden.unsafe_store(cell, ftz(identical_mul_add(
                    dlogit, ftz(weight.unsafe_load(token * width + feature)),
                    d_hidden.unsafe_load(cell),
                )))
    for feature in range(width):
        var cell = row * width + feature
        d_hidden.unsafe_store(cell, ftz(d_hidden.unsafe_load(cell)))


def chunked_lm_head_v2_dhidden_kernel(
    d_hidden: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, vocab_in: Int32, width_in: Int32,
):
    """One row owner per thread (`chunked_lm_head_v2_dhidden_body`)."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_in)
    if row >= rows:
        return
    chunked_lm_head_v2_dhidden_body(row, d_hidden, hidden, weight, targets, row_max, row_denom,
                                    rows, Int(vocab_in), Int(width_in))


@always_inline
def chunked_lm_head_v2_dweight_body(  # device-helper: one vocabulary entry of the dWeight kernel, also the host arm's per-owner body
    token: Int,
    d_weight: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows: Int, width: Int,
):
    """One vocabulary owner: every dWeight cell folds rows ascending."""
    for feature in range(width):
        d_weight.unsafe_store(token * width + feature, Float32(0.0))
    for row in range(rows):
        var shifted = ftz(ftz(_device_logit(hidden, weight, row, token, width)) - ftz(row_max.unsafe_load(row)))
        var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(row_denom.unsafe_load(row))))
        var target = Float32(1.0) if token == Int(targets.unsafe_load(row)) else Float32(0.0)
        var dlogit = ftz(identical_div(ftz(probability - target), Float32(rows)))
        for feature in range(width):
            var cell = token * width + feature
            d_weight.unsafe_store(cell, ftz(identical_mul_add(
                dlogit, ftz(hidden.unsafe_load(row * width + feature)),
                d_weight.unsafe_load(cell),
            )))
    for feature in range(width):
        var cell = token * width + feature
        d_weight.unsafe_store(cell, ftz(d_weight.unsafe_load(cell)))


def chunked_lm_head_v2_dweight_kernel(
    d_weight: MutPointer[Float32, MutAnyOrigin],
    hidden: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    row_denom: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, vocab_in: Int32, width_in: Int32,
):
    """One vocabulary owner per thread (`chunked_lm_head_v2_dweight_body`)."""
    var token = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if token >= Int(vocab_in):
        return
    chunked_lm_head_v2_dweight_body(token, d_weight, hidden, weight, targets, row_max, row_denom,
                                    Int(rows_in), Int(width_in))


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
    comptime if LMHEAD_FAST_GEMM_PANEL:
        # MOJOLEARN_LMHEAD_FAST_GEMM_PANEL (default off, A/B owed): GEMM logits and tree folds; the oracle moves with it.
        return _gp_door(ctx, loss_out, max_out, denom_out, loss_out, loss_out,
                        hidden_in, weight_in, targets_in, rows, vocab, width, False)
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
    _enqueue_loss_total(ctx, loss, row_loss, rows)
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
    comptime if LMHEAD_FAST_GEMM_PANEL:
        # MOJOLEARN_LMHEAD_FAST_GEMM_PANEL (default off, A/B owed): GEMM logits and tree folds; the oracle moves with it.
        return _gp_door(ctx, loss_out, max_out, denom_out, d_hidden_out, d_weight_out,
                        hidden_in, weight_in, targets_in, rows, vocab, width, True)
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
    _enqueue_loss_total(ctx, loss, row_loss, rows)
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
    _enqueue_loss_total(ctx, loss, row_loss, rows)


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


def chunked_lm_head_v2_gemm_workspace_floats(rows: Int,vocab: Int,width: Int) -> Int:
    # Dispatcher scratch need is not monotone in the output width: a short
    # final panel may select a split plan while full panels select a tile.
    # Size both actual panel widths; do not assume the largest panel wins.
    var panel = min(vocab,LM_HEAD_V2_CHUNK)
    var required = identical_gemm_workspace_max_floats(rows,panel,width)
    var tail = vocab%LM_HEAD_V2_CHUNK
    if tail>0 and tail!=panel:
        required = max(required,identical_gemm_workspace_max_floats(rows,tail,width))
    return max(1,required)


def chunked_lm_head_v2_gemm_forward_into(ctx: DeviceContext, mut loss: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32], mut denom: DeviceBuffer[DType.float32], mut row_loss: DeviceBuffer[DType.float32], mut chunk: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], mut hidden: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32], mut targets: DeviceBuffer[DType.int32], rows: Int, vocab: Int, width: Int) raises:
    if len(ws)<chunked_lm_head_v2_gemm_workspace_floats(rows,vocab,width):
        raise Error("chunked GEMM head workspace does not cover every actual panel")
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
    _enqueue_loss_total(ctx, loss, row_loss, rows)


# ===========================================================================
# THE GEMM-PANEL ARM: -D MOJOLEARN_LMHEAD_FAST_GEMM_PANEL (default off)
# ===========================================================================
# The public door's loss and both gradients as GEMM panels plus parallel
# folds; the fold order is training/CHUNKED_LM_HEAD_V2.md, "GEMM-panel arm".
# Three logit products per training call (maximum, denominator, gradients),
# each a panel of at most LM_HEAD_V2_CHUNK vocabulary entries: no
# (rows, vocab) logits owner. Per panel the gradients are two more products
# of the same GEMM: dWeight's panel rows (OP_TN, final) and one dHidden
# partial (OP_NN), folded over panels as a balanced tree. Every kernel below
# runs one thread per output unit (the maximum: one per row and panel); none
# walks the vocabulary or the rows.
# The host column runs the same bodies (training/chunked_lm_head_panel_math.mojo)
# in training/chunked_lm_head_gemm_panel_host.mojo.
# Scratch beyond the operands and the returned gradients, in floats:
# rows * panel (logits) + rows * ceil(vocab / 64) * 1.5 (denominator leaves and
# their tree) + (floor(log2(panels)) + 1) * rows * width (dHidden partials)
# + the GEMM workspace.


def _gp_grid(units: Int) -> Int:
    return (units + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB


def _gp_flush_kernel(values: MutPointer[Float32, MutAnyOrigin], count: Int32):
    """One thread per operand element, in place: a subnormal becomes its
    signed zero, so every later stage reads flushed operands."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    values.unsafe_store(i, ftz(values.unsafe_load(i)))


def _gp_leaf_kernel(
    partial: MutPointer[Float32, MutAnyOrigin],
    target_shift: MutPointer[Float32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    maxima: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    rows: Int32, n: Int32, chunk0: Int32, leaves: Int32,
):
    """One thread per (row, leaf of this panel)."""
    var unit = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if unit >= Int(rows) * ((Int(n) + LMHEAD_GP_LEAF - 1) // LMHEAD_GP_LEAF):
        return
    lmhead_gp_leaf_body(unit, partial, target_shift, logits, maxima, targets,
                        Int(n), Int(chunk0), Int(leaves))


def _gp_level_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: Int32, width: Int32,
):
    """One thread per (row, output node) of one denominator tree level."""
    var unit = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if unit >= Int(rows) * ((Int(width) + 1) // 2):
        return
    lmhead_gp_level_body(unit, dst, src, Int(width))


def _gp_row_loss_kernel(
    denom: MutPointer[Float32, MutAnyOrigin],
    row_loss: MutPointer[Float32, MutAnyOrigin],
    root: MutPointer[Float32, MutAnyOrigin],
    rows: Int32,
):
    """One thread per row: the tree's root and the row's loss."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(rows):
        return
    lmhead_gp_row_loss_body(row, denom, row_loss, root)


def _gp_dlogit_kernel(
    logits: MutPointer[Float32, MutAnyOrigin],
    maxima: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    rows: Int32, n: Int32, chunk0: Int32,
):
    """One thread per panel cell, in place: logit to dlogit."""
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(rows) * Int(n):
        return
    lmhead_gp_dlogit_body(cell, logits, maxima, denom, targets, Int(rows), Int(n), Int(chunk0))


def _gp_merge_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    l: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    cells: Int32, has_r: Int32,
):
    """One thread per dHidden cell of one tree node over panel partials."""
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(cells):
        return
    lmhead_gp_merge_body(cell, dst, l, r, Int(has_r))


def _gp_panel_workspace(rows: Int, n: Int, width: Int) -> Int:
    """The three products of one panel of `n` entries: logits (OP_NT),
    the dHidden partial (OP_NN) and dWeight's rows (OP_TN)."""
    var required = identical_gemm_workspace_max_floats(rows, n, width)
    required = max(required, identical_gemm_workspace_max_floats(rows, width, n))
    return max(required, identical_gemm_workspace_max_floats(n, width, rows))


def chunked_lm_head_gp_workspace_floats(rows: Int, vocab: Int, width: Int) -> Int:
    """GEMM scratch covering every product of the arm at both actual panel
    widths (the dispatcher's need is not monotone in the panel width)."""
    var panel = min(vocab, LM_HEAD_V2_CHUNK)
    var required = _gp_panel_workspace(rows, panel, width)
    var tail = vocab % LM_HEAD_V2_CHUNK
    if tail > 0 and tail != panel:
        required = max(required, _gp_panel_workspace(rows, tail, width))
    return max(1, required)


def chunked_lm_head_gp_forward_into(
    ctx: DeviceContext, mut loss: DeviceBuffer[DType.float32],
    mut maxima: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut row_loss: DeviceBuffer[DType.float32],
    mut chunk: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut leaf_a: DeviceBuffer[DType.float32],
    mut leaf_b: DeviceBuffer[DType.float32],
    mut hidden: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32], rows: Int, vocab: Int, width: Int,
) raises:
    """The arm's forward over flushed device operands. `leaf_a` holds
    rows * leaves floats and `leaf_b` rows * ceil(leaves / 2). Waits once, in
    the mean-loss fold."""
    comptime assert LM_HEAD_V2_CHUNK % LMHEAD_GP_LEAF == 0, "a logit panel holds whole denominator leaves"
    var leaves = lmhead_gp_leaf_count(vocab)
    if len(ws) < chunked_lm_head_gp_workspace_floats(rows, vocab, width):
        raise Error("chunked GEMM-panel head workspace does not cover every actual panel")
    if len(leaf_a) < rows * leaves or len(leaf_b) < rows * ((leaves + 1) // 2):
        raise Error("chunked GEMM-panel head leaf scratch too short")
    var grid = _gp_grid(rows)
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        # The maximum is a total-order selection (identical_fmax), so the
        # order its operands arrive in cannot move a bit.
        ctx.enqueue_function[_chunk_max_kernel](
            maxima.unsafe_ptr(), chunk.unsafe_ptr(), Int32(rows), Int32(n),
            Int32(1 if chunk0 == 0 else 0), grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
        )
        _ = wb
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_gp_leaf_kernel](
            leaf_a.unsafe_ptr(), row_loss.unsafe_ptr(), chunk.unsafe_ptr(),
            maxima.unsafe_ptr(), targets.unsafe_ptr(),
            Int32(rows), Int32(n), Int32(chunk0), Int32(leaves),
            grid_dim=_gp_grid(rows * ((n + LMHEAD_GP_LEAF - 1) // LMHEAD_GP_LEAF)),
            block_dim=LM_HEAD_V2_TPB,
        )
        _ = wb
    var level_width = leaves
    var from_a = True
    while level_width > 1:  # driver-loop: over denominator tree levels
        var out_width = (level_width + 1) // 2
        if from_a:
            ctx.enqueue_function[_gp_level_kernel](
                leaf_b.unsafe_ptr(), leaf_a.unsafe_ptr(), Int32(rows), Int32(level_width),
                grid_dim=_gp_grid(rows * out_width), block_dim=LM_HEAD_V2_TPB,
            )
        else:
            ctx.enqueue_function[_gp_level_kernel](
                leaf_a.unsafe_ptr(), leaf_b.unsafe_ptr(), Int32(rows), Int32(level_width),
                grid_dim=_gp_grid(rows * out_width), block_dim=LM_HEAD_V2_TPB,
            )
        from_a = not from_a
        level_width = out_width
    if from_a:
        ctx.enqueue_function[_gp_row_loss_kernel](
            denom.unsafe_ptr(), row_loss.unsafe_ptr(), leaf_a.unsafe_ptr(), Int32(rows),
            grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
        )
    else:
        ctx.enqueue_function[_gp_row_loss_kernel](
            denom.unsafe_ptr(), row_loss.unsafe_ptr(), leaf_b.unsafe_ptr(), Int32(rows),
            grid_dim=grid, block_dim=LM_HEAD_V2_TPB,
        )
    _enqueue_loss_total(ctx, loss, row_loss, rows)


def _gp_enqueue_merge(
    ctx: DeviceContext, mut stack: DeviceBuffer[DType.float32], depth: Int, cells: Int,
) raises:
    """Fold the two newest dHidden partials of the stack into the older one."""
    var l = stack.create_sub_buffer[DType.float32]((depth - 2) * cells, cells)
    var r = stack.create_sub_buffer[DType.float32]((depth - 1) * cells, cells)
    ctx.enqueue_function[_gp_merge_kernel](
        l.unsafe_ptr(), l.unsafe_ptr(), r.unsafe_ptr(), Int32(cells), Int32(1),
        grid_dim=_gp_grid(cells), block_dim=LM_HEAD_V2_TPB,
    )
    _ = l
    _ = r


def chunked_lm_head_gp_backward_into(
    ctx: DeviceContext, mut d_hidden: DeviceBuffer[DType.float32],
    mut d_weight: DeviceBuffer[DType.float32],
    mut chunk: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut stack: DeviceBuffer[DType.float32],
    mut hidden: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    mut maxima: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32], rows: Int, vocab: Int, width: Int,
) raises:
    """The arm's backward over flushed device operands and the matching
    forward statistics. `stack` holds lmhead_gp_stack_slots(vocab) dHidden
    partials of rows * width floats. Asynchronous: the caller waits."""
    var cells = rows * width
    if len(ws) < chunked_lm_head_gp_workspace_floats(rows, vocab, width):
        raise Error("chunked GEMM-panel head workspace does not cover every actual panel")
    if len(stack) < lmhead_gp_stack_slots(vocab) * cells:
        raise Error("chunked GEMM-panel head dHidden stack too short")
    # The pairwise fold over panel partials as a binary counter: after panel
    # p the stack holds one node per set bit of p + 1, and the leftovers
    # fold newest first. That is the level-by-level balanced tree over
    # panels in ascending order with an odd tail carried.
    var depth = 0
    var panels = 0
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_gp_dlogit_kernel](
            chunk.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(), targets.unsafe_ptr(),
            Int32(rows), Int32(n), Int32(chunk0),
            grid_dim=_gp_grid(rows * n), block_dim=LM_HEAD_V2_TPB,
        )
        # dWeight[panel rows] = dlogits[rows, n]^T . hidden[rows, width]: final, no fold across panels.
        var dwb = d_weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, dwb, chunk, hidden, ws, n, width, rows, OP_TN)
        # One dHidden partial = dlogits[rows, n] . weight[panel rows, width].
        var top = stack.create_sub_buffer[DType.float32](depth * cells, cells)
        identical_gemm_into(ctx, top, chunk, wb, ws, rows, width, n, OP_NN)
        depth += 1
        panels += 1
        var carry = panels
        while carry % 2 == 0:  # driver-loop: over completed tree nodes
            _gp_enqueue_merge(ctx, stack, depth, cells)
            depth -= 1
            carry //= 2
        _ = wb
        _ = dwb
        _ = top
    while depth > 1:  # driver-loop: over leftover tree nodes
        _gp_enqueue_merge(ctx, stack, depth, cells)
        depth -= 1
    var root = stack.create_sub_buffer[DType.float32](0, cells)
    ctx.enqueue_function[_gp_merge_kernel](
        d_hidden.unsafe_ptr(), root.unsafe_ptr(), root.unsafe_ptr(), Int32(cells), Int32(0),
        grid_dim=_gp_grid(cells), block_dim=LM_HEAD_V2_TPB,
    )
    _ = root


def _gp_first_nonfinite(values: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Int:
    """The lowest flat index holding a NaN or an infinity, or -1: one scan
    split across the host's cores (each task stops at its own first)."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var first = List[Int](length=tasks, fill=-1)
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](first.unsafe_ptr())

    def _scan(k: Int) {imm values, imm fp, imm chunk, imm count}:
        var lo = k * chunk
        for i in range(lo, min(lo + chunk, count)):
            if not isfinite(values.unsafe_load(i)):
                fp[k] = i
                break

    host_parallelize(_scan, tasks)
    for k in range(tasks):  # small-loop(one slot per scan task)
        if first[k] >= 0:
            return first[k]
    return -1


def _gp_first_bad_target(targets: MutPointer[Int32, MutUntrackedOrigin], rows: Int, vocab: Int) -> Int:
    """The lowest row whose target is outside [0, vocab), or -1; split as above."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var first = List[Int](length=tasks, fill=-1)
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](first.unsafe_ptr())

    def _scan(k: Int) {imm targets, imm fp, imm chunk, imm rows, imm vocab}:
        var lo = k * chunk
        for row in range(lo, min(lo + chunk, rows)):
            var target = Int(targets.unsafe_load(row))
            if target < 0 or target >= vocab:
                fp[k] = row
                break

    host_parallelize(_scan, tasks)
    for k in range(tasks):  # small-loop(one slot per scan task)
        if first[k] >= 0:
            return first[k]
    return -1


def _gp_door(
    ctx: DeviceContext,
    loss_out: MutPointer[Float32, MutUntrackedOrigin],
    max_out: MutPointer[Float32, MutUntrackedOrigin],
    denom_out: MutPointer[Float32, MutUntrackedOrigin],
    d_hidden_out: MutPointer[Float32, MutUntrackedOrigin],
    d_weight_out: MutPointer[Float32, MutUntrackedOrigin],
    hidden_in: MutPointer[Float32, MutUntrackedOrigin],
    weight_in: MutPointer[Float32, MutUntrackedOrigin],
    targets_in: MutPointer[Int32, MutUntrackedOrigin],
    rows: Int, vocab: Int, width: Int, with_grad: Bool,
) raises -> Int:
    """The public loss (and, `with_grad`, both gradients) under the arm:
    the refusals of the default door in the same order and words, then one
    upload, the forward, the backward, one download and one final wait.
    Without `with_grad` the two gradient pointers are not written."""
    var leaves = lmhead_gp_leaf_count(vocab)
    var panel = min(vocab, LM_HEAD_V2_CHUNK)
    if (rows > 2147483647 or vocab > 2147483647 or width > 2147483647
            or rows * width > 2147483647 or vocab * width > 2147483647
            or rows * panel > 2147483647 or rows * leaves > 2147483647):
        raise Error("chunked lm head v2: operand span exceeds signed 32-bit kernel indexing")
    var bad = _gp_first_nonfinite(hidden_in, rows * width)
    if bad >= 0:
        raise Error("chunked lm head v2: non-finite hidden at flat index " + String(bad))
    bad = _gp_first_nonfinite(weight_in, vocab * width)
    if bad >= 0:
        raise Error("chunked lm head v2: non-finite weight at flat index " + String(bad))
    bad = _gp_first_bad_target(targets_in, rows, vocab)
    if bad >= 0:
        raise Error("chunked lm head v2: target outside [0, vocab) at row " + String(bad))
    var cells = rows * width
    var hidden = ctx.enqueue_create_buffer[DType.float32](cells)
    var weight = ctx.enqueue_create_buffer[DType.float32](vocab * width)
    var targets = ctx.enqueue_create_buffer[DType.int32](rows)
    var maxima = ctx.enqueue_create_buffer[DType.float32](rows)
    var denom = ctx.enqueue_create_buffer[DType.float32](rows)
    var row_loss = ctx.enqueue_create_buffer[DType.float32](rows)
    var loss = ctx.enqueue_create_buffer[DType.float32](1)
    var chunk = ctx.enqueue_create_buffer[DType.float32](rows * panel)
    var ws = ctx.enqueue_create_buffer[DType.float32](chunked_lm_head_gp_workspace_floats(rows, vocab, width))
    var leaf_a = ctx.enqueue_create_buffer[DType.float32](rows * leaves)
    var leaf_b = ctx.enqueue_create_buffer[DType.float32](rows * ((leaves + 1) // 2))
    # Without gradients the three gradient owners are one-cell placeholders.
    var d_hidden = ctx.enqueue_create_buffer[DType.float32](cells if with_grad else 1)
    var d_weight = ctx.enqueue_create_buffer[DType.float32](vocab * width if with_grad else 1)
    var stack = ctx.enqueue_create_buffer[DType.float32](
        lmhead_gp_stack_slots(vocab) * cells if with_grad else 1
    )
    try:
        ctx.enqueue_copy(dst_buf=hidden, src_ptr=hidden_in)
        ctx.enqueue_copy(dst_buf=weight, src_ptr=weight_in)
        ctx.enqueue_copy(dst_buf=targets, src_ptr=targets_in)
        ctx.enqueue_function[_gp_flush_kernel](
            hidden.unsafe_ptr(), Int32(cells), grid_dim=_gp_grid(cells), block_dim=LM_HEAD_V2_TPB,
        )
        ctx.enqueue_function[_gp_flush_kernel](
            weight.unsafe_ptr(), Int32(vocab * width),
            grid_dim=_gp_grid(vocab * width), block_dim=LM_HEAD_V2_TPB,
        )
        chunked_lm_head_gp_forward_into(
            ctx, loss, maxima, denom, row_loss, chunk, ws, leaf_a, leaf_b,
            hidden, weight, targets, rows, vocab, width,
        )
        ctx.enqueue_copy(dst_ptr=loss_out, src_buf=loss)
        ctx.enqueue_copy(dst_ptr=max_out, src_buf=maxima)
        ctx.enqueue_copy(dst_ptr=denom_out, src_buf=denom)
        if with_grad:
            chunked_lm_head_gp_backward_into(
                ctx, d_hidden, d_weight, chunk, ws, stack, hidden, weight, targets,
                maxima, denom, rows, vocab, width,
            )
            ctx.enqueue_copy(dst_ptr=d_hidden_out, src_buf=d_hidden)
            ctx.enqueue_copy(dst_ptr=d_weight_out, src_buf=d_weight)
    except error:
        ctx.synchronize()
        raise error
    ctx.synchronize()
    # Raw-pointer consumers do not retain these owners: name every one past
    # the wait that drains its last queued reader.
    _ = hidden^
    _ = weight^
    _ = targets^
    _ = maxima^
    _ = denom^
    _ = row_loss^
    _ = loss^
    _ = chunk^
    _ = ws^
    _ = leaf_a^
    _ = leaf_b^
    _ = d_hidden^
    _ = d_weight^
    _ = stack^
    return rows
