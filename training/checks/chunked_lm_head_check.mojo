# SPDX-License-Identifier: Apache-2.0
"""Executable CPU checks for the chunked LM-head v2 foundation."""
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle
from checks.numerics import ftz, identical_div, identical_exp, identical_fmax, identical_log
from training.checks.chunked_lm_head_oracle import (
    chunked_lm_head_v2_oracle,
    chunked_lm_head_v2_peak_scratch_floats,
)
from training.checks.loss_oracle import CE_NEG_INF_BITS, CeConfig, ce_backward_oracle, ce_forward_oracle, neg_by_bits
from std.memory import bitcast


def _wrong_chunk_local_loss(logits: List[Float32], targets: List[Int32], rows: Int, vocab: Int) -> Float32:
    """Negative fixture: normalize only inside the target's 256-wide chunk."""
    var total = Float32(0.0)
    for row in range(rows):
        var target = Int(targets[row])
        var lo = target // 256 * 256
        var hi = min(lo + 256, vocab)
        var maximum = bitcast[DType.float32](CE_NEG_INF_BITS)
        for token in range(lo, hi):
            maximum = identical_fmax(maximum, logits[row * vocab + token])
        var denom = Float32(0.0)
        for token in range(lo, hi):
            var shifted = ftz(ftz(logits[row * vocab + token]) - ftz(maximum))
            denom = ftz(denom + ftz(identical_exp(shifted)))
        var target_shift = ftz(ftz(logits[row * vocab + target]) - ftz(maximum))
        var nll = neg_by_bits(ftz(target_shift - ftz(identical_log(denom))))
        total = ftz(total + nll)
    return ftz(identical_div(total, Float32(rows)))


def main() raises:
    comptime rows = 3
    comptime vocab = 513  # crosses two full chunks and a one-column tail
    comptime width = 5
    var hidden = List[Float32]()
    for i in range(rows * width):
        hidden.append(Float32((i * 17) % 29 - 14) / Float32(31.0))
    var weight = List[Float32]()
    for i in range(vocab * width):
        # Wide exponent inputs make a chunk-local normalization visibly wrong.
        weight.append(Float32((i * 37) % 101 - 50) / Float32(19.0))
    var targets = List[Int32](Int32(0), Int32(256), Int32(512), __list_literal__=None)

    var first = chunked_lm_head_v2_oracle(hidden, weight, targets, rows, vocab, width)
    var second = chunked_lm_head_v2_oracle(hidden, weight, targets, rows, vocab, width)
    if first.loss.to_bits() != second.loss.to_bits():
        raise Error("chunked lm head v2: repeated loss moved")
    for i in range(len(first.d_hidden)):
        if first.d_hidden[i].to_bits() != second.d_hidden[i].to_bits():
            raise Error("chunked lm head v2: repeated dHidden moved")
    for i in range(len(first.d_weight)):
        if first.d_weight[i].to_bits() != second.d_weight[i].to_bits():
            raise Error("chunked lm head v2: repeated dWeight moved")

    # Independent existing v1 composition is the quality reference. V2 has a
    # deliberately different serial denominator/row-loss fold, so tolerance,
    # not bit equality, is required between profiles.
    var logits = gemm_oracle(hidden, weight, OP_NT, rows, vocab, width)
    var v1 = ce_forward_oracle(logits, targets, CeConfig.causal_lm(vocab))
    ce_backward_oracle(v1, targets, CeConfig.causal_lm(vocab))
    var v1_dh = gemm_oracle(v1.dlogits, weight, OP_NN, rows, width, vocab)
    var v1_dw = gemm_oracle(v1.dlogits, hidden, OP_TN, vocab, width, rows)
    if abs(first.loss - v1.loss[0]) > Float32(2.0e-5):
        raise Error("chunked lm head v2: loss failed v1 quality bound")
    for i in range(len(first.d_hidden)):
        if abs(first.d_hidden[i] - v1_dh[i]) > Float32(2.0e-5):
            raise Error("chunked lm head v2: dHidden failed v1 quality bound")
    for i in range(len(first.d_weight)):
        if abs(first.d_weight[i] - v1_dw[i]) > Float32(2.0e-5):
            raise Error("chunked lm head v2: dWeight failed v1 quality bound")

    # The 513-way fixture separates the contract from the tempting but wrong
    # implementation that normalizes each target's chunk independently.
    var wrong = _wrong_chunk_local_loss(logits, targets, rows, vocab)
    if first.loss.to_bits() == wrong.to_bits():
        raise Error("chunked lm head v2: chunk-local negative fixture was inert")

    # B=8, L=2048, GPT-2 vocabulary: one full logits tensor is over 3 GiB;
    # the admitted v2 scratch bound is about 16 MiB, excluding persistent
    # model gradients that both paths must return.
    var realistic_rows = 8 * 2048
    var v1_logits_floats = realistic_rows * 50257
    var v2_scratch_floats = chunked_lm_head_v2_peak_scratch_floats(realistic_rows)
    if v2_scratch_floats * 100 >= v1_logits_floats:
        raise Error("chunked lm head v2: realistic scratch reduction below 100x")
    print("CHUNKED_LM_HEAD_V2_OK")
