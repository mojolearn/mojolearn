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

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul_add,
)
from training.checks.loss_oracle import CE_NEG_INF_BITS, neg_by_bits, refuse_nonfinite


comptime LM_HEAD_V2_VOCAB_CHUNK = 256


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


def _logit(
    hidden: List[Float32], weight: List[Float32], row: Int, vocab_id: Int,
    width: Int,
) -> Float32:
    var acc = Float32(0.0)
    for feature in range(width):
        acc = identical_mul_add(
            hidden[row * width + feature],
            weight[vocab_id * width + feature], acc,
        )
    return ftz(acc)


def chunked_lm_head_v2_oracle(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
) raises -> ChunkedLMHeadV2Result:
    """Mean CE and LM-head gradients under the v2 serial chunk contract."""
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("chunked lm head v2: rows/width must be positive and vocab >= 2")
    if len(hidden) != rows * width:
        raise Error("chunked lm head v2: hidden shape mismatch")
    if len(weight) != vocab * width:
        raise Error("chunked lm head v2: weight shape mismatch")
    if len(targets) != rows:
        raise Error("chunked lm head v2: target shape mismatch")
    refuse_nonfinite("chunked lm head hidden", hidden)
    refuse_nonfinite("chunked lm head weight", weight)
    for row in range(rows):
        if targets[row] < 0 or Int(targets[row]) >= vocab:
            raise Error("chunked lm head v2: target outside [0, vocab) at row " + String(row))

    var maxima = List[Float32](length=rows, fill=Float32(0.0))
    var denom = List[Float32](length=rows, fill=Float32(0.0))
    var row_loss = List[Float32](length=rows, fill=Float32(0.0))

    # Pass one: chunk boundaries do not reset the global serial maximum.
    for row in range(rows):
        var maximum = bitcast[DType.float32](CE_NEG_INF_BITS)
        for chunk0 in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
            var chunk1 = min(chunk0 + LM_HEAD_V2_VOCAB_CHUNK, vocab)
            for token in range(chunk0, chunk1):
                maximum = identical_fmax(maximum, _logit(hidden, weight, row, token, width))
        maxima[row] = maximum

    # Pass two: one serial global denominator, never a sum of chunk sums.
    for row in range(rows):
        var total = Float32(0.0)
        var target_shift = Float32(0.0)
        for chunk0 in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
            var chunk1 = min(chunk0 + LM_HEAD_V2_VOCAB_CHUNK, vocab)
            for token in range(chunk0, chunk1):
                var shifted = ftz(ftz(_logit(hidden, weight, row, token, width)) - ftz(maxima[row]))
                total = ftz(total + ftz(identical_exp(shifted)))
                if token == Int(targets[row]):
                    target_shift = shifted
        denom[row] = total
        var logdenom = ftz(identical_log(ftz(total)))
        row_loss[row] = neg_by_bits(ftz(ftz(target_shift) - ftz(logdenom)))

    var loss_total = Float32(0.0)
    for row in range(rows):
        loss_total = ftz(loss_total + row_loss[row])
    var divisor = Float32(rows)
    var loss = ftz(identical_div(loss_total, divisor))

    # dHidden owns one cell and folds vocab ids globally in ascending order.
    var d_hidden = List[Float32](length=rows * width, fill=Float32(0.0))
    for row in range(rows):
        for feature in range(width):
            var acc = Float32(0.0)
            for chunk0 in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
                var chunk1 = min(chunk0 + LM_HEAD_V2_VOCAB_CHUNK, vocab)
                for token in range(chunk0, chunk1):
                    var shifted = ftz(ftz(_logit(hidden, weight, row, token, width)) - ftz(maxima[row]))
                    var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denom[row])))
                    var target = Float32(1.0) if token == Int(targets[row]) else Float32(0.0)
                    var dlogit = ftz(identical_div(ftz(probability - target), divisor))
                    acc = identical_mul_add(dlogit, weight[token * width + feature], acc)
            d_hidden[row * width + feature] = ftz(acc)

    # dWeight chunks are independent storage owners; every cell folds rows
    # ascending, so scheduling chunks cannot change a bit.
    var d_weight = List[Float32](length=vocab * width, fill=Float32(0.0))
    for chunk0 in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
        var chunk1 = min(chunk0 + LM_HEAD_V2_VOCAB_CHUNK, vocab)
        for token in range(chunk0, chunk1):
            for feature in range(width):
                var acc = Float32(0.0)
                for row in range(rows):
                    var shifted = ftz(ftz(_logit(hidden, weight, row, token, width)) - ftz(maxima[row]))
                    var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denom[row])))
                    var target = Float32(1.0) if token == Int(targets[row]) else Float32(0.0)
                    var dlogit = ftz(identical_div(ftz(probability - target), divisor))
                    acc = identical_mul_add(dlogit, hidden[row * width + feature], acc)
                d_weight[token * width + feature] = ftz(acc)
    return ChunkedLMHeadV2Result(loss, maxima^, denom^, d_hidden^, d_weight^)
