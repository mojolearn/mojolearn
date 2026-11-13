# SPDX-License-Identifier: Apache-2.0
"""NI34 native host counterpart of ByteTrainer's GEMM-chunked LM head.

This is the GEMM-logit variant, not the standalone serial-logit oracle.
Vocabulary and gradient folds mirror chunked_lm_head_v2_gemm_*_into exactly.
Only a [rows, <=256] logit chunk is materialized. Source-only, unverified.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_div, identical_exp, identical_fmax, identical_log, identical_mul_add
from gemm.contract import OP_NT
from gemm.host.gemm_host_rows import gemm_host_rows
from training.checks.chunked_lm_head_oracle import (
    ChunkedLMHeadV2Result, LM_HEAD_V2_VOCAB_CHUNK, lm_head_v2_loss_fold,
)
from training.checks.loss_contract import CE_NEG_INF_BITS, neg_by_bits, refuse_nonfinite


def _chunk_logits(hidden: List[Float32], weight: List[Float32], rows: Int,
                  width: Int, first: Int, columns: Int) raises -> List[Float32]:
    var part = List[Float32](length=columns * width, fill=Float32(0.0))
    for i in range(columns * width):
        part[i] = weight[first * width + i]
    return gemm_host_rows(hidden, part, OP_NT, rows, columns, width)


def _admit(hidden: List[Float32], weight: List[Float32], targets: List[Int32],
           rows: Int, vocab: Int, width: Int) raises:
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("chunked lm head v2: rows/width must be positive and vocab >= 2")
    if len(hidden) != rows * width or len(weight) != vocab * width or len(targets) != rows:
        raise Error("chunked lm head v2: shape mismatch")
    refuse_nonfinite("chunked lm head hidden", hidden)
    refuse_nonfinite("chunked lm head weight", weight)
    for row in range(rows):
        if targets[row] < 0 or Int(targets[row]) >= vocab:
            raise Error("chunked lm head v2: target outside [0, vocab) at row " + String(row))


def chunked_lm_head_gemm_host_forward(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
) raises -> ChunkedLMHeadV2Result:
    _admit(hidden, weight, targets, rows, vocab, width)
    var maxima = List[Float32](length=rows, fill=bitcast[DType.float32](CE_NEG_INF_BITS))
    var denom = List[Float32](length=rows, fill=Float32(0.0))
    var row_loss = List[Float32](length=rows, fill=Float32(0.0))
    for first in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
        var columns = min(LM_HEAD_V2_VOCAB_CHUNK, vocab - first)
        var logits = _chunk_logits(hidden, weight, rows, width, first, columns)
        for row in range(rows):
            var acc = maxima[row]
            for j in range(columns):
                acc = identical_fmax(acc, logits[row * columns + j])
            maxima[row] = acc
    for first in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
        var columns = min(LM_HEAD_V2_VOCAB_CHUNK, vocab - first)
        var logits = _chunk_logits(hidden, weight, rows, width, first, columns)
        for row in range(rows):
            var acc = denom[row]
            for j in range(columns):
                var shifted = ftz(ftz(logits[row * columns + j]) - ftz(maxima[row]))
                acc = ftz(acc + ftz(identical_exp(shifted)))
                if first + j == Int(targets[row]):
                    row_loss[row] = shifted
            denom[row] = acc
    for row in range(rows):
        row_loss[row] = neg_by_bits(ftz(ftz(row_loss[row]) - ftz(identical_log(ftz(denom[row])))))
    var loss = lm_head_v2_loss_fold(row_loss, rows)
    return ChunkedLMHeadV2Result(loss, maxima^, denom^, List[Float32](), List[Float32]())


def _dlogit(logit: Float32, maximum: Float32, denominator: Float32,
            target: Int32, token: Int, rows: Int) -> Float32:
    var shifted = ftz(ftz(logit) - ftz(maximum))
    var p = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denominator)))
    var t = Float32(1.0) if token == Int(target) else Float32(0.0)
    return ftz(identical_div(ftz(p - t), Float32(rows)))


def chunked_lm_head_gemm_host_train(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
) raises -> ChunkedLMHeadV2Result:
    var fwd = chunked_lm_head_gemm_host_forward(hidden, weight, targets, rows, vocab, width)
    var dh = List[Float32](length=rows * width, fill=Float32(0.0))
    var dw = List[Float32](length=vocab * width, fill=Float32(0.0))
    for first in range(0, vocab, LM_HEAD_V2_VOCAB_CHUNK):
        var columns = min(LM_HEAD_V2_VOCAB_CHUNK, vocab - first)
        var logits = _chunk_logits(hidden, weight, rows, width, first, columns)
        # Device _chunk_dhidden_kernel's row owner and ascending token fold.
        for row in range(rows):
            for j in range(columns):
                var dl = _dlogit(logits[row * columns + j], fwd.row_max[row], fwd.row_denom[row], targets[row], first + j, rows)
                for feature in range(width):
                    var cell = row * width + feature
                    dh[cell] = identical_mul_add(dl, weight[(first + j) * width + feature], dh[cell])
        # Device _chunk_dweight_kernel's token owner and ascending row fold.
        for j in range(columns):
            for row in range(rows):
                var dl = _dlogit(logits[row * columns + j], fwd.row_max[row], fwd.row_denom[row], targets[row], first + j, rows)
                for feature in range(width):
                    var cell = (first + j) * width + feature
                    dw[cell] = identical_mul_add(dl, hidden[row * width + feature], dw[cell])
            for feature in range(width):
                var cell = (first + j) * width + feature
                dw[cell] = ftz(dw[cell])
    return ChunkedLMHeadV2Result(fwd.loss, fwd.row_max.copy(), fwd.row_denom.copy(), dh^, dw^)
