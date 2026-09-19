# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The TransformerBlock surface on the HOST: the forward (stateless prefill,
carried-state prefill, decode step) and the zero-state prefill backward, in
the device binding's calling convention (`bindings/_mojolearn_transformer.mojo`),
over the lane's two host oracles (CPU training for the transformer lanes,
2026-09-15).

HOST ONLY. Nothing here creates a `DeviceContext` or launches a kernel. The
arithmetic is `transformer/checks/transformer_oracle.mojo::transformer_block_oracle`
and `transformer/checks/transformer_backward_oracle.mojo::transformer_block_backward_oracle`,
the same two functions the byte LM host step composes
(`training/byte_lm_host_backward.mojo`). This file adds no arithmetic of its
own: it only moves the caller's buffers into the oracles' `List`s and back,
and converts the KV cache between the two layouts:

  - the DEVICE layout, the one a `TransformerState` holds
    (`LlamaKVCache`, DEVIATION 795(ii)): full causal caches are PACKED at
    stride `cached_tokens` (the first `B * n_kv * cached_tokens * head_dim`
    floats are `[B, n_kv, cached_tokens, head_dim]`); a sliding window is a
    RING of `window` slots per (batch, kv head) at stride `window`, slot =
    position % window;
  - the ORACLE layout (`TransformerKVCache`): full causal at stride `cap`
    (= max_tokens), the ring exactly the device's.

The rotary table is built per call at `max_tokens` positions (the forward)
or `L` positions (the backward), the device binding's `LlamaRopeTable`
sizes; every entry is a pure function of its absolute position, so the size
decides only which positions exist.

After a full causal call the device writes its whole capacity buffer back,
whose tail past the packed region is not state (nothing reads it, and the
Python surface never exposes it); this file writes zeros there.

The sabotage arm is `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walked descending, which
moves the q, k, v, o and MLP projections forward and every weight and
activation gradient they reach.
"""

from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from transformer.checks.transformer_fixture import (
    ScorePlant,
    TransformerDims,
    TransformerWeights,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    build_rope_table_opts,
    refuse_bad_weights,
    transformer_block_oracle,
)
# lane/block-options (2026-09-17): the options record the bindings decode
# from their params tail and the eleven optional tensors from their addrs
# tail (`transformer/block_options.mojo` has both orders).
from transformer.block_options import BlockOptions
from transformer.checks.transformer_backward_oracle import (
    transformer_block_backward_oracle,
)

comptime TRANSFORMER_HOST_SABOTAGE = GEMM_ORACLE_HOST_SABOTAGE


def transformer_host_weights(
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, rope_positions: Int,
    norm1_w: List[Float32], norm2_w: List[Float32], w_q: List[Float32],
    w_k: List[Float32], w_v: List[Float32], w_o: List[Float32],
    w_gate: List[Float32], w_up: List[Float32], w_down: List[Float32],
) raises -> TransformerWeights:
    """The oracle's weight struct from the nine buffers in the device
    binding's address order (input_layernorm, post_attention_layernorm,
    q, k, v, o, gate, up, down). Refuses a non-finite weight by name, as
    the device path does at upload (DEVIATION 1875)."""
    var dims = TransformerDims(dm, nh, nkv, hd, it, rope_positions)
    dims.validate()
    var w = TransformerWeights(dims)
    w.norm1_w = norm1_w.copy()
    w.norm2_w = norm2_w.copy()
    w.w_q = w_q.copy()
    w.w_k = w_k.copy()
    w.w_v = w_v.copy()
    w.w_o = w_o.copy()
    w.w_gate = w_gate.copy()
    w.w_up = w_up.copy()
    w.w_down = w_down.copy()
    refuse_bad_weights(w)
    return w^


def transformer_host_weights_opts(
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, rope_positions: Int,
    opts: BlockOptions,
    norm1_w: List[Float32], norm2_w: List[Float32], w_q: List[Float32],
    w_k: List[Float32], w_v: List[Float32], w_o: List[Float32],
    w_gate: List[Float32], w_up: List[Float32], w_down: List[Float32],
    b_q: List[Float32], b_k: List[Float32], b_v: List[Float32],
    b_o: List[Float32], norm1_b: List[Float32], norm2_b: List[Float32],
    b_up: List[Float32], b_down: List[Float32], b_gate: List[Float32],
    qn_w: List[Float32], kn_w: List[Float32],
) raises -> TransformerWeights:
    """`transformer_host_weights` with the options record and the eleven
    optional tensors in the addrs-tail order (q_proj.bias, k_proj.bias,
    v_proj.bias, o_proj.bias, input_layernorm.bias,
    post_attention_layernorm.bias, up_proj.bias, down_proj.bias,
    gate_proj.bias, q_norm.weight, k_norm.weight); an absent tensor is an
    EMPTY list and `w_gate` is empty under an ungated MLP.
    `refuse_bad_weights` refuses every presence/flag mismatch by name."""
    var dims = TransformerDims(dm, nh, nkv, hd, it, rope_positions)
    dims.validate()
    var w = TransformerWeights(dims)
    w.opts = opts.copy()
    w.norm1_w = norm1_w.copy()
    w.norm2_w = norm2_w.copy()
    w.w_q = w_q.copy()
    w.w_k = w_k.copy()
    w.w_v = w_v.copy()
    w.w_o = w_o.copy()
    w.w_gate = w_gate.copy()
    w.w_up = w_up.copy()
    w.w_down = w_down.copy()
    w.b_q = b_q.copy()
    w.b_k = b_k.copy()
    w.b_v = b_v.copy()
    w.b_o = b_o.copy()
    w.norm1_b = norm1_b.copy()
    w.norm2_b = norm2_b.copy()
    w.b_up = b_up.copy()
    w.b_down = b_down.copy()
    w.b_gate = b_gate.copy()
    w.qn_w = qn_w.copy()
    w.kn_w = kn_w.copy()
    refuse_bad_weights(w)
    return w^


struct TransformerHostForward(Movable):
    """One forward call's outputs: the block output `y` (B*L*d_model), the
    post-call caches in the DEVICE layout, and the post-call cached_tokens."""

    var y: List[Float32]
    var k_cache: List[Float32]
    var v_cache: List[Float32]
    var cached_tokens: Int

    def __init__(out self):
        self.y = List[Float32]()
        self.k_cache = List[Float32]()
        self.v_cache = List[Float32]()
        self.cached_tokens = 0


def transformer_host_forward(
    w: TransformerWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    smax: Int,
    s0: Int,
    window: Int,
    k_in: List[Float32],
    v_in: List[Float32],
) raises -> TransformerHostForward:
    """`transformer_forward` (and, at L = 1, `transformer_decode_step`) on
    the host. `k_in`/`v_in` are the caller's caches in the device layout,
    `b * n_kv * cap * head_dim` floats (cap = window, or smax); an empty
    pair is a zero cache (the stateless prefill, `transformer_forward_fresh`,
    passes that with smax = L and s0 = 0)."""
    var dims = w.dims.copy()
    var nkv = dims.n_kv_heads
    var hd = dims.head_dim
    if s0 < 0 or s0 > smax:
        raise Error(
            String("transformer: cached_tokens must be in [0, ")
            + String(smax)
            + "] (the cache capacity, max_tokens), got "
            + String(s0)
            + "; the two sides of this boundary disagree about the state"
        )
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd
    var zero_cache = len(k_in) == 0 and len(v_in) == 0
    if not zero_cache and (len(k_in) != cache_n or len(v_in) != cache_n):
        raise Error("transformer host: the carried caches do not have B * n_kv * capacity * head_dim floats")
    var cache = TransformerKVCache(b, dims, smax, window)
    if not zero_cache:
        if window > 0:
            # The ring: the same slot map on both sides.
            for i in range(cache_n):
                cache.k[i] = k_in[i]
                cache.v[i] = v_in[i]
        else:
            # Packed at stride s0 -> the oracle's stride smax.
            for bb in range(b):
                for kv in range(nkv):
                    for j in range(s0):
                        for d in range(hd):
                            var src = ((bb * nkv + kv) * s0 + j) * hd + d
                            var dst = cache.slot(bb, kv, j, d)
                            cache.k[dst] = k_in[src]
                            cache.v[dst] = v_in[src]
    cache.used = s0
    # lane/block-options: the table from the record (theta, scaling,
    # rope_dim, max_positions); the default record is `build_rope_table`.
    var rope = build_rope_table_opts(dims, w.opts)
    var st = transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())
    var out = TransformerHostForward()
    out.y = st.residual2_out.copy()
    out.cached_tokens = cache.used
    if window > 0:
        out.k_cache = cache.k.copy()
        out.v_cache = cache.v.copy()
    else:
        var s1 = cache.used
        out.k_cache = List[Float32](length=cache_n, fill=Float32(0.0))
        out.v_cache = List[Float32](length=cache_n, fill=Float32(0.0))
        for bb in range(b):
            for kv in range(nkv):
                for j in range(s1):
                    for d in range(hd):
                        var dst = ((bb * nkv + kv) * s1 + j) * hd + d
                        var src = cache.slot(bb, kv, j, d)
                        out.k_cache[dst] = cache.k[src]
                        out.v_cache[dst] = cache.v[src]
    _ = st^
    _ = rope^
    return out^


def transformer_host_backward(
    w: TransformerWeights,
    x: List[Float32],
    d_out: List[Float32],
    b: Int,
    l: Int,
    window: Int,
) raises -> List[List[Float32]]:
    """`transformer_backward` on the host: the forward from a zero cache at
    positions [0, L) under `window`, then the backward oracle on its saved
    stages. Returns the ten gradients in the binding's order: d_x,
    input_layernorm, post_attention_layernorm, q, k, v, o, gate, up, down."""
    if window < 0:
        raise Error("transformer backward: window must be >= 0")
    # lane/block-options: the backward chains are the frozen profile's and
    # know none of the options; refused by name rather than run wrong.
    if not w.opts.is_default():
        raise Error(
            String("transformer backward: only the default block options are")
            + " supported (got "
            + w.opts.describe()
            + "); the backward oracle spells none of the option seams"
        )
    var dims = w.dims.copy()
    var rope = build_rope_table(dims)
    var cache = TransformerKVCache(b, dims, l, window)
    var fwd = transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())
    var bwd = transformer_block_backward_oracle(w, fwd, d_out, b, l, 0, rope, window)
    var out = List[List[Float32]]()
    out.append(bwd.d_x.copy())
    out.append(bwd.dw_norm1.copy())
    out.append(bwd.dw_norm2.copy())
    out.append(bwd.dw_q.copy())
    out.append(bwd.dw_k.copy())
    out.append(bwd.dw_v.copy())
    out.append(bwd.dw_o.copy())
    out.append(bwd.dw_gate.copy())
    out.append(bwd.dw_up.copy())
    out.append(bwd.dw_down.copy())
    _ = bwd^
    _ = fwd^
    _ = rope^
    return out^
