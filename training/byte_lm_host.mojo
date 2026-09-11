# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU inference for the byte-level decoder language model (DEVIATION 2610).

HOST ONLY, AND NO NEW ARITHMETIC. Every number this file produces comes from
the host FP32 oracles that the device kernels are gated against bit for bit,
called in the order `training/byte_lm.mojo::_byte_forward_loss` launches the
device kernels:

    embedding   emb_forward_oracle             identical_embedding_forward_into
    each block  transformer_block_oracle       llama_decoder_layer_forward
                (fresh cache, prefill at 0)    (prefill_cache.s = 0)
    head        gemm_oracle(OP_NT)             identical_gemm_into(..., OP_NT)
    loss        ce_forward_oracle(causal_lm)   identical_ce_forward_into

There is no final norm between the last block and the head, because the
device forward has none. The parameter registry is
`training/byte_lm_config.mojo`'s, and block weights are sliced in the order
`training/byte_lm.mojo::_block_weights` hands them to the device.

What that composition promises is a PREDICTION until the gate runs.
`tools/byte_lm_host_gate.py` (DEVIATION 2613) compares the loss bytes this
file produces against the retained Metal, CUDA and HIP captures of the same
parameters and batches. Nothing here is qualified by construction.

DEVIATION 2611. `transformer/checks/transformer_oracle.mojo` imports
`core/identity_trace.mojo`, which imports `max.gpu.host` (that file's
DEVIATION 1003). This module never creates a context or a kernel, but the
import means a CPU-only build still needs the MAX package on the box. Whether
it compiles with no accelerator target is measured on the first CPU-only box,
not assumed.

DEVIATION 2612. `MOJOLEARN_BYTE_LM_HOST_SABOTAGE` replaces the head product
with the same k-term chain folded in REVERSE order through the same seam. At
the admitted profile k = 32, which `contract_leaf_size` makes one serial
leaf, so the reversal changes the fold and nothing else. It is the gate's
negative control: a build with it defined must fail the loss comparison, or
the comparison is not reaching the arithmetic.
"""

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_mul_add
from embedding.checks.embedding_oracle import EmbConfig, emb_forward_oracle
from gemm.checks.gemm_oracle import OP_NT, gemm_oracle
from training.byte_lm_config import ByteConfig
from training.checks.loss_oracle import CeConfig, ce_forward_oracle
from transformer.checks.transformer_fixture import (
    ScorePlant,
    TransformerDims,
    TransformerWeights,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    refuse_bad_weights,
    transformer_block_oracle,
)


comptime BYTE_HOST_SABOTAGE = is_defined["MOJOLEARN_BYTE_LM_HOST_SABOTAGE"]()


def byte_host_sabotage_compiled() -> Bool:
    """Whether this binary carries the DEVIATION 2612 negative control."""
    comptime if BYTE_HOST_SABOTAGE:
        return True
    return False


def _require_identical() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM host: requires -D MOJOLEARN_NUMERIC_IDENTICAL=1")


def _slice(values: List[Float32], offsets: List[Int], j: Int) -> List[Float32]:
    var out = List[Float32](capacity=offsets[j + 1] - offsets[j])
    for i in range(offsets[j], offsets[j + 1]):
        out.append(values[i])
    return out^


def byte_host_dims(config: ByteConfig) raises -> TransformerDims:
    """The block shape. RoPE is sized from the configured length, as the
    device table `LlamaRopeTable(ctx, dims, 10000, config.length)` is."""
    config.validate()
    return TransformerDims(config.d_model, config.n_heads, config.n_kv,
                           config.head_dim, config.intermediate, config.length)


def byte_host_block_weights(params: List[Float32], offsets: List[Int], block: Int,
                            dims: TransformerDims) raises -> TransformerWeights:
    """Registry order per block: norm1_w, w_q, w_k, w_v, w_o, norm2_w,
    w_gate, w_up, w_down (`training/byte_lm.mojo::byte_param_name`)."""
    var base = 1 + 9 * block
    var w = TransformerWeights(dims)
    w.norm1_w = _slice(params, offsets, base)
    w.w_q = _slice(params, offsets, base + 1)
    w.w_k = _slice(params, offsets, base + 2)
    w.w_v = _slice(params, offsets, base + 3)
    w.w_o = _slice(params, offsets, base + 4)
    w.norm2_w = _slice(params, offsets, base + 5)
    w.w_gate = _slice(params, offsets, base + 6)
    w.w_up = _slice(params, offsets, base + 7)
    w.w_down = _slice(params, offsets, base + 8)
    refuse_bad_weights(w)
    return w^


def _sabotaged_head(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int) -> List[Float32]:
    """DEVIATION 2612 only. The OP_NT chain with k walked DESCENDING."""
    var out = List[Float32](capacity=m * n)
    for i in range(m):
        for j in range(n):
            var acc = Float32(0.0)
            var kk = k - 1
            while kk >= 0:
                acc = identical_mul_add(a[i * k + kk], b[j * k + kk], acc)
                kk -= 1
            out.append(acc)
    return out^


def byte_host_logits(params: List[Float32], inputs: List[Int32], batch: Int,
                     length: Int, config: ByteConfig) raises -> List[Float32]:
    """Logits `[batch * length, vocab]`, row-major, for token ids
    `[batch, length]` starting at absolute position 0.

    `length` may be shorter than the configured length and `batch` may
    differ from it. The block and head contracts carry no fold across rows,
    so a row's logits do not depend on the batch around it; the gate
    measures that at the configured shape only."""
    _require_identical()
    config.validate()
    if batch <= 0 or length <= 0:
        raise Error("byte LM host: batch and length must be positive")
    if length > config.length:
        raise Error("byte LM host: length exceeds the configured length the RoPE table is sized for")
    var n = config.n_total()
    if len(params) != n:
        raise Error("byte LM host: expected " + String(n) + " parameters, got " + String(len(params)))
    if len(inputs) != batch * length:
        raise Error("byte LM host: ids must hold batch * length tokens")
    for t in range(len(inputs)):
        var v = Int(inputs[t])
        if v < 0 or v >= config.vocab_size:
            raise Error("byte LM host: token id outside [0, vocab) at " + String(t))
    var offsets = config.offsets()
    var dims = byte_host_dims(config)
    var rope = build_rope_table(dims)
    var m = batch * length
    var x = emb_forward_oracle(_slice(params, offsets, 0), inputs,
                               EmbConfig.llama(config.vocab_size, config.d_model))
    for layer in range(config.n_layers):
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var cache = TransformerKVCache(batch, dims, length, 0)
        var st = transformer_block_oracle(w, x, batch, length, cache, rope, ScorePlant.none())
        x = st.residual2_out.copy()
    var head = _slice(params, offsets, config.n_tensors() - 1)
    comptime if BYTE_HOST_SABOTAGE:
        return _sabotaged_head(x, head, m, config.vocab_size, config.d_model)
    return gemm_oracle(x, head, OP_NT, m, config.vocab_size, config.d_model)


def byte_host_loss(params: List[Float32], ids: List[Int32], config: ByteConfig) raises -> Float32:
    """Mean next-byte cross-entropy of ids `[batch, length + 1]` at the
    configured shape, split exactly as `_byte_forward_loss` splits them."""
    config.validate()
    var b = config.batch
    var l = config.length
    if len(ids) != b * (l + 1):
        raise Error("byte LM host: loss ids must be [batch, length + 1]")
    var inputs = List[Int32](capacity=b * l)
    var targets = List[Int32](capacity=b * l)
    for bi in range(b):
        for li in range(l):
            inputs.append(ids[bi * (l + 1) + li])
            targets.append(ids[bi * (l + 1) + li + 1])
    var logits = byte_host_logits(params, inputs, b, l, config)
    var st = ce_forward_oracle(logits, targets, CeConfig.causal_lm(config.vocab_size))
    return st.loss[0]
