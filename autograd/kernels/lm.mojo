# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The LM backward the tape calls: the cross-entropy backward
(identical_ce_backward_into and its dlogits and weights kernels, moved
unchanged out of training/checks/loss.mojo, which keeps the forward and the
sabotage switches) and the chunked LM head's GEMM backward
(chunked_lm_head_v2_gemm_backward_into and its two chunk kernels, moved
unchanged out of training/chunked_lm_head_v2.mojo)."""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp
from max.gpu.host import DeviceBuffer, DeviceContext
from training.neural_identical_experiments import IDN_CE_GRAD_FUSED
from core.step_phase import step_count_launch
from checks.numerics import ftz, identical_div, identical_exp, identical_mul
from training.checks.loss_contract import CeConfig, REDUCTION_NONE, ce_divisor, ce_smoothing_targets
from training.checks.loss import (
    ANY_LOSS_SABOTAGE,
    CE_TPB,
    SAB_GRAD_DIVISOR_IS_N,
    SAB_GRAD_RECIPROCAL_MUL,
    SAB_GRAD_SIGN,
    SAB_GRAD_TARGET_OFF_BY_ONE,
    SAB_IGNORED_ROW_SKIPPED,
    SAB_W_VIA_EXP_LOGP,
    _grid_for,
    ce_divide_kernel,
    ce_row_nll_kernel,
    identical_ce_forward_into,
)
from checks.numerics import identical_mul_add
from gemm.neural_dispatch import identical_gemm_into
from gemm.contract import OP_NT
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK
from training.chunked_lm_head_v2 import LM_HEAD_V2_TPB, chunked_lm_head_v2_gemm_workspace_floats


def ce_weights_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    expo: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    logp: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
):
    """`w[i, v] = ftz(identical_div(e[i, v], denom[i]))`, seam L14.

    **TRANSFORMER DEVIATION 806 AT A SECOND SITE.** ONE division per weight,
    never `e * (1/denom)`. The transformer contract pinned this for
    `attn.weights` and this lane's brief required consistency with it; the
    two must agree, and `SAB_GRAD_RECIPROCAL_MUL` is the falsifier at seam
    L16 while this one is falsified by `SAB_W_VIA_EXP_LOGP`.

    `w = exp(logp)` is the refused alternative -- one more transcendental
    where a division suffices, and a different answer. `logp` is passed for
    that arm alone and is unused on the clean path; under `NUMERIC_FAST` on
    a build with no smoothing it may be an empty buffer, which is why the
    launcher refuses the arm unless `logp` was actually produced.
    """
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    comptime if SAB_W_VIA_EXP_LOGP:
        weights.unsafe_store(cell, ftz(identical_exp(logp.unsafe_load(cell))))
        return
    weights.unsafe_store(
        cell,
        ftz(
            identical_div(
                ftz(expo.unsafe_load(cell)), ftz(denom.unsafe_load(row))
            )
        ),
    )


def ce_weights_dlogits_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    dlogits: MutPointer[Float32, MutAnyOrigin],
    expo: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    t_target: Float32, t_other: Float32, divisor: Float32,
    n_rows_in: Int32, vocab_in: Int32, ignore_index_in: Int32,
):
    """NN52: the clean L14 and L16 cells, with a local probability value.

    Same-cell aliases used by the resident LM remain valid: load expo before
    either store, store the probability first and the final gradient last.
    Distinct buffers still receive every probability, including ignored rows.
    Sabotage builds retain the original kernels at the launcher below.
    """
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var vocab = Int(vocab_in)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    var v = cell - row * vocab
    var w = ftz(identical_div(ftz(expo.unsafe_load(cell)), ftz(denom.unsafe_load(row))))
    var y = Int(targets.unsafe_load(row))
    weights.unsafe_store(cell, w)
    if y == Int(ignore_index_in):
        dlogits.unsafe_store(cell, Float32(0.0))
        return
    var t = t_target if v == y else t_other
    var d = ftz(ftz(w) - ftz(t))
    dlogits.unsafe_store(cell, ftz(identical_div(d, divisor)))


def ce_dlogits_kernel(
    dlogits: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    t_target: Float32,
    t_other: Float32,
    divisor: Float32,
    n_rows_in: Int32,
    vocab_in: Int32,
    ignore_index_in: Int32,
):
    """`dl[i, v] = ftz(identical_div(ftz(ftz(w) - ftz(t)), divisor))`, L16.

    `t_target` and `t_other` are the two HOST constants of seam L15
    (`ce_smoothing_targets`), selected by an INTEGER compare. Integers do not
    flush anywhere, so there is no row-49 reasoning at this seam at all.

    **THE DIVISION LIVES PER CELL, NOT PER ROW.** `d / divisor` rounds once
    where `d * (1/divisor)` rounds twice, and the same paragraph that pins
    L13 pins this. At `divisor == +1.0` it is bitwise inert and it is spelled
    anyway, for `ce_divide_kernel`'s reason.

    **AT `eps == 0` THE TARGET CONSTANTS ARE EXACTLY `1.0` AND `+0.0`**, so
    the backward's smoothing arm is bitwise inert at eps zero and needs no
    branch, unlike the forward's. That asymmetry is
    `[[reached-but-inert]]` in the other direction and contract 6.3 names it:
    a sabotage that deletes a branch the gradient does not have moves
    nothing, and that is not evidence about the gradient.

    AN IGNORED ROW writes `V` copies of `+0.0`, STORED and not skipped, for
    `ce_row_nll_kernel`'s reason and because a `-0.0` gradient cell would
    flow into whatever the optimizer does next.

    **THIS IS THE STAGE THE EXACT-ANALYTIC GATE ASSERTS BY BITS.** Contract
    12 -- on a uniform row at a power-of-two `V` with a power-of-two divisor
    every value here is a dyadic rational a person can write down, so there
    is no epsilon in the comparison. Contract 12.1 also says what that
    family cannot do: being exact it separates NO spelling from any other,
    so run alone it would pass every sabotage in this file and gate nothing
    about the arithmetic.
    """
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    var v = cell - row * vocab
    var y = Int(targets.unsafe_load(row))
    if y == Int(ignore_index_in):
        comptime if SAB_IGNORED_ROW_SKIPPED:
            return
        dlogits.unsafe_store(cell, Float32(0.0))
        return
    comptime if SAB_GRAD_TARGET_OFF_BY_ONE:
        # SABOTAGE: the target vector applied at `y + 1`. INERT at `V == 1`;
        # at `V > 1` it moves exactly two cells per row, which is why the
        # gate must compare CELLS and not a row norm.
        y = (y + 1) % vocab
    var t = t_other
    if v == y:
        t = t_target
    var w = ftz(weights.unsafe_load(cell))
    var d = ftz(ftz(w) - ftz(t))
    comptime if SAB_GRAD_SIGN:
        # SABOTAGE: `t - w`. INERT only where `w == t` at every class.
        d = ftz(ftz(t) - ftz(w))
    comptime if SAB_GRAD_RECIPROCAL_MUL:
        var r = ftz(identical_div(Float32(1.0), divisor))
        dlogits.unsafe_store(cell, ftz(identical_mul(d, r)))
        return
    dlogits.unsafe_store(cell, ftz(identical_div(d, divisor)))


def identical_ce_backward_into(
    ctx: DeviceContext,
    mut weights: DeviceBuffer[DType.float32],
    mut dlogits: DeviceBuffer[DType.float32],
    mut expo: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut logp: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    count: Int,
    cfg: CeConfig,
) raises:
    """Seams L14 through L16, enqueued. Nothing waits.

    `expo` and `denom` must be the buffers `identical_ce_forward_into` wrote
    on the SAME inputs and the SAME config. **The backward recomputes
    nothing**, which is deliberate: a second spelling of the softmax is a
    second thing that can be wrong, and the gemm oracle's own header makes
    the same argument about its serial diagnostic.

    **REDUCTION_NONE HAS NO BACKWARD** and raises, contract section 11. A
    per-row upstream vector adds a second product per cell whose placement --
    before or after L16's division -- is a real decision with two different
    answers, and this lane has no caller for it. An unused wrapper is an
    ungated one.

    `dlogits` may alias nothing that the forward still needs; it is written
    once per cell and read by nobody here.

    THE DIVISOR COMES FROM `ce_divisor`, the same call the forward made, with
    the same three arguments. Sabotage `SAB_GRAD_DIVISOR_IS_N` substitutes
    `Float32(n_rows)` **and is bit-inert whenever no row is ignored under
    MEAN, and on every SUM** -- the gate must predict that and assert the
    inertness as a mask, or the arm will look broken on an ordinary fixture.
    """
    if n_rows < 1 or cfg.vocab < 1:
        return
    if cfg.reduction == REDUCTION_NONE:
        raise Error(
            String("ce: REDUCTION_NONE has no backward (contract section 11)")
        )
    var vocab = cfg.vocab
    var cells = n_rows * vocab
    var tv = ce_smoothing_targets(cfg.eps, vocab)
    var divisor = ce_divisor(cfg.reduction, count, cfg.num_items)
    comptime if SAB_GRAD_DIVISOR_IS_N:
        divisor = Float32(n_rows)

    comptime if IDN_CE_GRAD_FUSED and not ANY_LOSS_SABOTAGE:
        step_count_launch()
        ctx.enqueue_function[ce_weights_dlogits_kernel](
            weights.unsafe_ptr(), dlogits.unsafe_ptr(), expo.unsafe_ptr(),
            denom.unsafe_ptr(), targets.unsafe_ptr(), tv[0], tv[1], divisor,
            Int32(n_rows), Int32(vocab), Int32(cfg.ignore_index),
            grid_dim=(_grid_for(cells), 1, 1), block_dim=(CE_TPB, 1, 1),
        )
        return

    step_count_launch()
    ctx.enqueue_function[ce_weights_kernel](
        weights.unsafe_ptr(),
        expo.unsafe_ptr(),
        denom.unsafe_ptr(),
        logp.unsafe_ptr(),
        Int32(n_rows),
        Int32(vocab),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )
    step_count_launch()
    ctx.enqueue_function[ce_dlogits_kernel](
        dlogits.unsafe_ptr(),
        weights.unsafe_ptr(),
        targets.unsafe_ptr(),
        tv[0],
        tv[1],
        divisor,
        Int32(n_rows),
        Int32(vocab),
        Int32(cfg.ignore_index),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )


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


def chunked_lm_head_v2_gemm_backward_into(ctx: DeviceContext, mut d_hidden: DeviceBuffer[DType.float32], mut d_weight: DeviceBuffer[DType.float32], mut chunk: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], mut hidden: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32], mut targets: DeviceBuffer[DType.int32], mut maxima: DeviceBuffer[DType.float32], mut denom: DeviceBuffer[DType.float32], rows: Int, vocab: Int, width: Int) raises:
    if len(ws)<chunked_lm_head_v2_gemm_workspace_floats(rows,vocab,width):
        raise Error("chunked GEMM head workspace does not cover every actual panel")
    var row_grid = (rows + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        var wb = weight.create_sub_buffer[DType.float32](chunk0 * width, n * width)
        identical_gemm_into(ctx, chunk, hidden, wb, ws, rows, n, width, OP_NT)
        ctx.enqueue_function[_chunk_dhidden_kernel](d_hidden.unsafe_ptr(), chunk.unsafe_ptr(), weight.unsafe_ptr(), targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(), Int32(rows), Int32(n), Int32(width), Int32(chunk0), grid_dim=row_grid, block_dim=LM_HEAD_V2_TPB)
        ctx.enqueue_function[_chunk_dweight_kernel](d_weight.unsafe_ptr(), chunk.unsafe_ptr(), hidden.unsafe_ptr(), targets.unsafe_ptr(), maxima.unsafe_ptr(), denom.unsafe_ptr(), Int32(rows), Int32(n), Int32(width), Int32(chunk0), grid_dim=(n + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB, block_dim=LM_HEAD_V2_TPB)
        _ = wb
