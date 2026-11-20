# SPDX-License-Identifier: Apache-2.0
"""The chunked LM head (`LanguageModelConfig(chunked_lm_head_v2=True)`) as one
executor op with its backward rule (lane m2-neural-leftovers, 2026-11-09).

    loss = lm_head_v2(hidden (rows, width), weight (vocab, width), codes (rows, 1))
           OPS_LM_HEAD_V2

is the mean next-token cross-entropy of `hidden @ weight^T` against the class
codes, computed without the (rows, vocab) logits: per row, the maximum, the
denominator and the row loss over vocabulary panels
(training/chunked_lm_head_v2.mojo `chunked_lm_head_gp_forward_into`), then the
fixed leaf-and-tree mean of the row losses. The recorded op is one tape
custom node (`LM_HEAD_V2_TAG`) over [hidden, weight]; its VJP
(`lm_head_v2_backward`, reached through autograd/rules_step_ops.mojo
StepOpRules) recomputes each logit from the saved row statistics: dHidden one
row owner per thread, dWeight one vocabulary owner per thread, both folds
ascending, each scaled by the upstream gradient. Memory: three row vectors and
the int32 codes, no logits.

One algorithm on every column: the device kernels here and the host arm
(autograd/lm_head_v2_host.mojo, reached through `Exec.lm_head_v2_host`, so no
GPU module imports it) call the same @always_inline bodies
(training/chunked_lm_head_panel_math.mojo); the
host arm splits rows (and vocabulary entries) across cores, the fold inside
each owner is unchanged. The codes are class indices as float values (the
cross-entropy op's convention); a code that names no vocabulary entry (the
trainer's -100 targets) contributes `log(denominator)`, as the trainer's
chunked head always did.
"""
from max.gpu import block_dim, block_idx, thread_idx
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from checks.numerics import ftz, identical_div, identical_mul
from autograd.block_step_ops import block_buffer_view
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_TPB,
    _enqueue_loss_total,
    chunked_lm_head_gp_forward_into, chunked_lm_head_gp_backward_into, chunked_lm_head_gp_workspace_floats,
)
from training.chunked_lm_head_panel_math import lmhead_gp_leaf_count, lmhead_gp_stack_slots
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK

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
    # the arm's kernels also index a logit panel and the denominator leaves
    if rows > 2147483647 // min(vocab, LM_HEAD_V2_CHUNK) or rows > 2147483647 // lmhead_gp_leaf_count(vocab):
        raise Error("ValueError: ops: lm_head_v2 exceeds signed 32-bit kernel indexing")


# ---------------------------------------------------------------- device
def _scale_kernel(dst: _F, upstream: _F, n_in: Int32):
    """dst[i] *= upstream[0] (the VJP's chain factor; 1 at a loss root)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(identical_mul(dst.unsafe_load(i), upstream.unsafe_load(0))))


def _grid(n: Int) -> Int:
    return (n + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB


# ------------------------------------------------- device, GEMM-panel arm
# The GEMM-panel profile (the only profile since 2026-11-20): the op runs the
# public door's forward and backward (training/chunked_lm_head_v2.mojo
# `chunked_lm_head_gp_forward_into`, `chunked_lm_head_gp_backward_into`) over
# the executor's own storage, so the door, the op and the oracle share one
# fold order per build (training/CHUNKED_LM_HEAD_V2.md, "GEMM-panel arm").
# The op's own parts stay: the float class codes, a code that names no
# vocabulary entry (its row loss is log(denominator), its dlogit has no
# one-hot), and the upstream factor. The panel, GEMM workspace, leaf and
# dHidden scratch are allocated per call and released after a wait; holding
# them in the session is owed if the arm is promoted.
def _codes_zero_kernel(targets: _I, row_loss: _F, codes: _F, rows_in: Int32):
    """targets[i] = the int32 class code of codes[i], and row_loss[i] = +0.0:
    the shifted logit a row keeps when its code names no vocabulary entry."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(rows_in):
        return
    targets.unsafe_store(i, codes.unsafe_load(i).cast[DType.int32]())
    row_loss.unsafe_store(i, Float32(0.0))


def _forward_gp_device(
    ex: DeviceExec, hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    var ctx = ex.ctx.copy()
    var leaves = lmhead_gp_leaf_count(vocab)
    ctx.enqueue_function[_codes_zero_kernel](_ip(targets), _fp(row_loss), _fp(codes), Int32(rows),
                                             grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB)
    var hb = block_buffer_view(ex, hidden, rows * width)
    var wb = block_buffer_view(ex, weight, vocab * width)
    var mb = block_buffer_view(ex, maxima, rows)
    var db = block_buffer_view(ex, denom, rows)
    var rb = block_buffer_view(ex, row_loss, rows)
    var lb = block_buffer_view(ex, loss, 1)
    var chunk = ctx.enqueue_create_buffer[DType.float32](rows * min(vocab, LM_HEAD_V2_CHUNK))
    var ws = ctx.enqueue_create_buffer[DType.float32](chunked_lm_head_gp_workspace_floats(rows, vocab, width))
    var leaf_a = ctx.enqueue_create_buffer[DType.float32](rows * leaves)
    var leaf_b = ctx.enqueue_create_buffer[DType.float32](rows * ((leaves + 1) // 2))
    try:
        chunked_lm_head_gp_forward_into(ctx, lb, mb, db, rb, chunk, ws, leaf_a, leaf_b, hb, wb,
                                        _ip(targets), rows, vocab, width)
    except error:
        ctx.synchronize()
        raise error
    # The forward ends in the mean-loss fold's wait, so every queued reader
    # of the scratch has drained; name each owner past it.
    _ = hb^
    _ = wb^
    _ = mb^
    _ = db^
    _ = rb^
    _ = lb^
    _ = chunk^
    _ = ws^
    _ = leaf_a^
    _ = leaf_b^


def _backward_gp_device(
    ex: DeviceExec, d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    var want_dh = Int(d_hidden) != 0
    var want_dw = Int(d_weight) != 0
    if not want_dh and not want_dw:
        return
    var ctx = ex.ctx.copy()
    var cells = rows * width
    var hb = block_buffer_view(ex, hidden, cells)
    var wb = block_buffer_view(ex, weight, vocab * width)
    var mb = block_buffer_view(ex, maxima, rows)
    var db = block_buffer_view(ex, denom, rows)
    # A destination not asked for is a one-cell placeholder the backward never touches.
    var dhb = block_buffer_view(ex, d_hidden, cells) if want_dh else ctx.enqueue_create_buffer[DType.float32](1)
    var dwb = block_buffer_view(ex, d_weight, vocab * width) if want_dw else ctx.enqueue_create_buffer[DType.float32](1)
    var chunk = ctx.enqueue_create_buffer[DType.float32](rows * min(vocab, LM_HEAD_V2_CHUNK))
    var ws = ctx.enqueue_create_buffer[DType.float32](chunked_lm_head_gp_workspace_floats(rows, vocab, width))
    var stack = ctx.enqueue_create_buffer[DType.float32](lmhead_gp_stack_slots(vocab) * cells if want_dh else 1)
    try:
        chunked_lm_head_gp_backward_into(ctx, dhb, dwb, chunk, ws, stack, hb, wb, _ip(targets), mb, db,
                                         rows, vocab, width, want_dh, want_dw)
        if want_dh:
            ctx.enqueue_function[_scale_kernel](_fp(d_hidden), _fp(upstream), Int32(cells),
                                                grid_dim=_grid(cells), block_dim=LM_HEAD_V2_TPB)
        if want_dw:
            ctx.enqueue_function[_scale_kernel](_fp(d_weight), _fp(upstream), Int32(vocab * width),
                                                grid_dim=_grid(vocab * width), block_dim=LM_HEAD_V2_TPB)
    except error:
        ctx.synchronize()
        raise error
    # The per-call scratch is read through raw pointers by queued launches:
    # wait for them before it is released (the default VJP owns no scratch
    # and does not wait).
    ctx.synchronize()
    _ = hb^
    _ = wb^
    _ = mb^
    _ = db^
    _ = dhb^
    _ = dwb^
    _ = chunk^
    _ = ws^
    _ = stack^


def lm_head_v2_forward_device(
    ex: DeviceExec, hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """loss[0] = the mean row loss; the row statistics stay for the VJP."""
    _forward_gp_device(ex, hidden, weight, codes, targets, maxima, denom, row_loss, loss, rows, vocab, width)


def lm_head_v2_backward_device(
    ex: DeviceExec, d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """d_hidden (rows, width) and d_weight (vocab, width), each written whole
    (a null destination is skipped), scaled by upstream[0]."""
    _backward_gp_device(ex, d_hidden, d_weight, hidden, weight, targets, maxima, denom, upstream,
                        rows, vocab, width)
