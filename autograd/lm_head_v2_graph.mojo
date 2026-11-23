# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The chunked LM head and token cross-entropy as one tape operation over tape tensors (lane
neural-blocks-3-lm, 2026-11-22).

    loss = graph_lm_head_v2(hidden (rows, width), weight (vocab, width), codes (rows class values))

is OPS_LM_HEAD_V2 (autograd/lm_head_v2_op.mojo): the mean next-token cross-entropy of
`hidden @ weight^T` against the codes, without the logits. This is the op's ONE recording: the ops
session (`OpsSession._lm_head_v2`, autograd/ops_api.mojo) and a Sequential model's `lm_head` layer
under the `token_cross_entropy` loss (autograd/model.mojo) both call it, so both run the op's device
kernels or its host arm (the same @always_inline bodies, training/chunked_lm_head_panel_math.mojo)
and record the same custom node, `LM_HEAD_V2_TAG`, whose VJP is StepOpRules' (autograd/rules_step_ops.mojo).
"""
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import Args, FP
from autograd.tape import Tape, Tensor
from autograd.lm_head_v2_op import LM_HEAD_V2_TAG, lm_head_v2_check, lm_head_v2_forward_device


def graph_lm_head_v2[E: Exec](
    mut tape: Tape, mut ex: E, hidden: Tensor, weight: Tensor, codes: Tensor, rows: Int, vocab: Int, width: Int,
) raises -> Tensor:
    """The (1, 1) mean loss. hidden holds `rows` rows of `width` values (any rank), weight is (vocab, width),
    codes one class value per row (a code that names no vocabulary entry adds log(denominator)).
    Recorded (when the tape records): one custom node over [hidden, weight]; the row statistics are tape
    storage until the reset."""
    lm_head_v2_check(rows, vocab, width)
    var hp = tape.value_of(ex, hidden)
    var wp = tape.value_of(ex, weight)
    var cp = tape.value_of(ex, codes)
    var targets = tape.reserve(ex, rows)
    var maxima = tape.reserve(ex, rows)
    var denom = tape.reserve(ex, rows)
    var row_loss = tape.reserve(ex, rows)
    var loss = tape.reserve(ex, 1)
    comptime if E.IS_DEVICE:
        lm_head_v2_forward_device(rebind[DeviceExec](ex), hp, wp, cp, targets, maxima, denom, row_loss,
                                  loss, rows, vocab, width)
    else:
        # the host arm (autograd/lm_head_v2_host.mojo) through the executor
        var hb: List[FP] = [hp, wp, cp, targets, maxima, denom, row_loss, loss]
        ex.lm_head_v2_host(False, hb, rows, vocab, width)
    if not tape.is_recording():
        return tape.variable(ex, loss, 1, 1, False)
    var saved = Args()
    saved.p0 = targets
    saved.p1 = maxima
    saved.p2 = denom
    saved.i0 = rows
    saved.i1 = vocab
    saved.i2 = width
    var rx = List[Tensor]()
    rx.append(hidden)
    rx.append(weight)
    return tape.custom(ex, rx, loss, 1, 1, LM_HEAD_V2_TAG, saved)
