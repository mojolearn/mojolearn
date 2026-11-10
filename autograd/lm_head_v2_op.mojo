# SPDX-License-Identifier: Apache-2.0
"""The chunked LM head (`LanguageModelConfig(chunked_lm_head_v2=True)`) as one
executor op with its backward rule (lane m2-neural-leftovers, 2026-11-09).

    loss = lm_head_v2(hidden (rows, width), weight (vocab, width), codes (rows, 1))
           OPS_LM_HEAD_V2

is the mean next-token cross-entropy of `hidden @ weight^T` against the class
codes, computed without the (rows, vocab) logits: per row, the maximum, the
denominator and the row loss over vocabulary panels
(training/chunked_lm_head_v2.mojo `chunked_lm_head_v2_row_body`), then the
fixed leaf-and-tree mean of the row losses. The recorded op is one tape
custom node (`LM_HEAD_V2_TAG`) over [hidden, weight]; its VJP
(`lm_head_v2_backward`, reached through autograd/rules_step_ops.mojo
StepOpRules) recomputes each logit from the saved row statistics: dHidden one
row owner per thread, dWeight one vocabulary owner per thread, both folds
ascending, each scaled by the upstream gradient. Memory: three row vectors and
the int32 codes, no logits.

One algorithm on every column: the device kernels here and the host arm
(autograd/lm_head_v2_host.mojo, reached through `Exec.lm_head_v2_host`, so no
GPU module imports it) call the same @always_inline bodies (`chunked_lm_head_v2_row_body`,
`chunked_lm_head_v2_dhidden_body`, `chunked_lm_head_v2_dweight_body`); the
host arm splits rows (and vocabulary entries) across cores, the fold inside
each owner is unchanged. The codes are class indices as float values (the
cross-entropy op's convention); a code that names no vocabulary entry (the
trainer's -100 targets) contributes `log(denominator)`, as the trainer's
chunked head always did.
"""
from std.gpu import block_dim, block_idx, thread_idx
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from checks.numerics import ftz, identical_div, identical_mul
from autograd.block_step_ops import block_buffer_view
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_TPB,
    chunked_lm_head_v2_rows_kernel, chunked_lm_head_v2_dhidden_kernel, chunked_lm_head_v2_dweight_kernel,
    _enqueue_loss_total,
)

#: The session op code (autograd/ops_api.mojo) and the tape custom tag.
comptime OPS_LM_HEAD_V2 = 87
# Integration (integ/pre-0867-w5): lane m2 took 109, which lane x56d gave
# M1_STEP_TAG (autograd/block_step_ops.mojo, 109-111) in the same rule set
# (rules_step_ops.StepRules): 115 is the first free custom tag (99-104
# graph, 108-111 block steps, 112-114 MoE, 128+ rank-N).
comptime LM_HEAD_V2_TAG = 115

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _fp(p: FP) -> _F:
    return _F(unsafe_from_address=Int(p))


@always_inline
def _ip(p: FP) -> _I:
    return _I(unsafe_from_address=Int(p))


def lm_head_v2_check(rows: Int, vocab: Int, width: Int) raises:
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("ValueError: ops: lm_head_v2 needs rows >= 1, vocab >= 2 and width >= 1")
    if rows > 2147483647 // width or vocab > 2147483647 // width:
        raise Error("ValueError: ops: lm_head_v2 exceeds signed 32-bit kernel indexing")


# ---------------------------------------------------------------- device
def _codes_kernel(targets: _I, codes: _F, rows_in: Int32):
    """targets[i] = the int32 class code of codes[i] (a float class value)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(rows_in):
        return
    targets.unsafe_store(i, codes.unsafe_load(i).cast[DType.int32]())


def _scale_kernel(dst: _F, upstream: _F, n_in: Int32):
    """dst[i] *= upstream[0] (the VJP's chain factor; 1 at a loss root)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(identical_mul(dst.unsafe_load(i), upstream.unsafe_load(0))))


def _grid(n: Int) -> Int:
    return (n + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB


def lm_head_v2_forward_device(
    ex: DeviceExec, hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """loss[0] = the mean row loss; the row statistics stay for the VJP."""
    var ctx = ex.ctx.copy()
    ctx.enqueue_function[_codes_kernel](_ip(targets), _fp(codes), Int32(rows),
                                        grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB)
    ctx.enqueue_function[chunked_lm_head_v2_rows_kernel](
        _fp(maxima), _fp(denom), _fp(row_loss), _fp(hidden), _fp(weight), _ip(targets),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB,
    )
    var lb = block_buffer_view(ex, loss, 1)
    var rb = block_buffer_view(ex, row_loss, rows)
    _enqueue_loss_total(ctx, lb, rb, rows)
    _ = lb^
    _ = rb^


def lm_head_v2_backward_device(
    ex: DeviceExec, d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """d_hidden (rows, width) and d_weight (vocab, width), each written whole
    (a null destination is skipped), scaled by upstream[0]."""
    var ctx = ex.ctx.copy()
    if Int(d_hidden) != 0:
        ctx.enqueue_function[chunked_lm_head_v2_dhidden_kernel](
            _fp(d_hidden), _fp(hidden), _fp(weight), _ip(targets), _fp(maxima), _fp(denom),
            Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB,
        )
        ctx.enqueue_function[_scale_kernel](_fp(d_hidden), _fp(upstream), Int32(rows * width),
                                            grid_dim=_grid(rows * width), block_dim=LM_HEAD_V2_TPB)
    if Int(d_weight) != 0:
        ctx.enqueue_function[chunked_lm_head_v2_dweight_kernel](
            _fp(d_weight), _fp(hidden), _fp(weight), _ip(targets), _fp(maxima), _fp(denom),
            Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(vocab), block_dim=LM_HEAD_V2_TPB,
        )
        ctx.enqueue_function[_scale_kernel](_fp(d_weight), _fp(upstream), Int32(vocab * width),
                                            grid_dim=_grid(vocab * width), block_dim=LM_HEAD_V2_TPB)
