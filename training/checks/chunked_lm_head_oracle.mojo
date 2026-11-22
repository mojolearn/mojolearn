# SPDX-License-Identifier: Apache-2.0
"""Normative CPU oracle for ``mojolearn.identical.lm_head.chunked.fp32.v2``.

V1 remains the default production path.  V2 fixes vocabulary chunks at 256
columns and never materializes ``rows * vocab`` logits or dlogits.  Chunking
is storage only: for every row the global maximum and exponential denominator
visit vocabulary ids 0..V-1 serially; dHidden folds vocabulary ids 0..V-1
serially; each dWeight cell folds rows 0..M-1 serially.  Every add is flushed.
Changing chunk size, normalizing inside a chunk, or reducing chunks as a tree
is a different profile.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul_add,
)
from training.checks.loss_contract import CE_NEG_INF_BITS, neg_by_bits, refuse_nonfinite
from training.chunked_lm_head_panel_math import lmhead_gp_leaf_count
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK
from training.chunked_lm_head_gemm_panel_host import (
    lmhead_gp_host_forward, lmhead_gp_host_backward,
)


comptime LM_HEAD_V2_VOCAB_CHUNK = 256

#: THE NEGATIVE CONTROL for the training host binding, which is the only
#: caller built with it (`-D MOJOLEARN_HOST_SABOTAGE=1`, the training family's
#: define). It folds the row losses and every dWeight cell over rows
#: DESCENDING, a fold-order fault the contract forbids, while the arithmetic
#: of each term is unchanged. MEASURED on a RunPod x86 CPU pod (2026-09-23,
#: bench/results/runpod_cpu/2026-09-23_233919-exposure/): dWeight moved at
#: (5, 513, 8) and (16, 300, 12); the loss did NOT move at either shape (a
#: reversed sum of 5 or 16 row losses rounded to the same bits), so only the
#: dWeight part of a future lane is watched by this arm. The checks in
#: training/checks never define it.
comptime CHUNKED_LM_HEAD_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


#: The mean loss's row fold, shared with the device
#: (`training/chunked_lm_head_v2.mojo::_enqueue_loss_total`, cpu-gpu-cleanup
#: n-train-mamba 2026-10-02): leaves of this many rows, each folded ascending
#: from +0.0 with every add flushed; then the fixed balanced tree over the
#: leaf partials in ascending leaf order (pairs `ftz(ftz(l) + ftz(r))`, an
#: odd tail carried bit for bit); then `ftz(total / rows)`. At rows <= the
#: leaf it is the serial chain the profile had before.
comptime LM_HEAD_V2_LOSS_LEAF = 256


def lm_head_v2_loss_fold(row_loss: List[Float32], rows: Int) -> Float32:
    """The mean of `row_loss[0:rows]` in the fold above. The sabotage arm
    feeds the rows DESCENDING into the same structure."""
    var leaves = (rows + LM_HEAD_V2_LOSS_LEAF - 1) // LM_HEAD_V2_LOSS_LEAF
    var level = List[Float32](length=leaves, fill=Float32(0.0))
    for j in range(leaves):
        var acc = Float32(0.0)
        for r in range(j * LM_HEAD_V2_LOSS_LEAF, min((j + 1) * LM_HEAD_V2_LOSS_LEAF, rows)):
            var row = r
            comptime if CHUNKED_LM_HEAD_HOST_SABOTAGE:
                row = rows - 1 - r
            acc = ftz(acc + row_loss[row])
        level[j] = acc
    while len(level) > 1:
        var width = len(level)
        var nxt = List[Float32](length=(width + 1) // 2, fill=Float32(0.0))
        for q in range(len(nxt)):
            if 2 * q + 1 < width:
                nxt[q] = ftz(ftz(level[2 * q]) + ftz(level[2 * q + 1]))
            else:
                nxt[q] = level[2 * q]
        level = nxt^
    return ftz(identical_div(ftz(level[0]), Float32(rows)))


def chunked_lm_head_v2_peak_scratch_floats(rows: Int) -> Int:
    """Production upper bound: one logits chunk plus max and denominator."""
    return rows * (LM_HEAD_V2_VOCAB_CHUNK + 2)


@fieldwise_init
struct ChunkedLMHeadV2Result(Movable):
    var loss: Float32
    var row_max: List[Float32]
    var row_denom: List[Float32]
    var d_hidden: List[Float32]
    var d_weight: List[Float32]


def _gp_oracle(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int, with_grad: Bool,
) raises -> ChunkedLMHeadV2Result:
    """The oracle of the GEMM-panel profile (the default; `-D MOJOLEARN_LMHEAD_FAST_GEMM_PANEL_OFF` is the scalar one):
    the GEMM-panel arm's words (training/CHUNKED_LM_HEAD_V2.md, "GEMM-panel
    arm"), which the device door computes under the same define. The
    refusals are the default oracle's, in the same order and words, after
    the device door's span refusal. Without `with_grad` the gradient lists
    are empty and the loss is the same arithmetic in the same order."""
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("chunked lm head v2: rows/width must be positive and vocab >= 2")
    if len(hidden) != rows * width:
        raise Error("chunked lm head v2: hidden shape mismatch")
    if len(weight) != vocab * width:
        raise Error("chunked lm head v2: weight shape mismatch")
    if len(targets) != rows:
        raise Error("chunked lm head v2: target shape mismatch")
    if (rows > 2147483647 or vocab > 2147483647 or width > 2147483647
            or rows * width > 2147483647 or vocab * width > 2147483647
            or rows * min(vocab, LM_HEAD_V2_CHUNK) > 2147483647
            or rows * lmhead_gp_leaf_count(vocab) > 2147483647):
        raise Error("chunked lm head v2: operand span exceeds signed 32-bit kernel indexing")
    refuse_nonfinite("chunked lm head hidden", hidden)
    refuse_nonfinite("chunked lm head weight", weight)
    for row in range(rows):
        if targets[row] < 0 or Int(targets[row]) >= vocab:
            raise Error("chunked lm head v2: target outside [0, vocab) at row " + String(row))
    var maxima = List[Float32](length=rows, fill=Float32(0.0))
    var denom = List[Float32](length=rows, fill=Float32(0.0))
    var row_loss = List[Float32](length=rows, fill=Float32(0.0))
    lmhead_gp_host_forward(hidden, weight, targets, rows, vocab, width, maxima, denom, row_loss)
    var loss = lm_head_v2_loss_fold(row_loss, rows)
    if not with_grad:
        return ChunkedLMHeadV2Result(loss, maxima^, denom^, List[Float32](), List[Float32]())
    var d_hidden = List[Float32](length=rows * width, fill=Float32(0.0))
    var d_weight = List[Float32](length=vocab * width, fill=Float32(0.0))
    lmhead_gp_host_backward(hidden, weight, targets, rows, vocab, width, maxima, denom, d_hidden, d_weight)
    return ChunkedLMHeadV2Result(loss, maxima^, denom^, d_hidden^, d_weight^)


def chunked_lm_head_v2_oracle_forward(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
) raises -> ChunkedLMHeadV2Result:
    """Mean CE, row maxima and row denominators under the v2 serial chunk
    contract, with EMPTY gradient lists. The loss-only stage the host binding
    exports as `chunked_lm_head_v2_loss`; `chunked_lm_head_v2_oracle` runs
    exactly this and then the two gradient folds, so the loss a caller gets
    from either is the same arithmetic in the same order."""
    # The GEMM-panel profile (the only one since 2026-11-20): the oracle moves with the device door.
    return _gp_oracle(hidden, weight, targets, rows, vocab, width, False)


def chunked_lm_head_v2_oracle(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
) raises -> ChunkedLMHeadV2Result:
    """Mean CE and LM-head gradients under the v2 serial chunk contract."""
    # The GEMM-panel profile (the only one since 2026-11-20): the oracle moves with the device door.
    return _gp_oracle(hidden, weight, targets, rows, vocab, width, True)
