# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed side of the transformer host binding (`bindings/_mojolearn_transformer_host.mojo`).

Everything here takes typed values (Ints, addresses, lists, handles) and
imports no Python. The binding unpacks its PythonObject arguments, calls
one entry below, and packs the result. Moved word for word from the
binding (thin-shell lane, 2026-10-21): same operations, same order.
"""

from std.os import abort
from std.memory import memcpy

from core.hostptr import copy_f32, f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from transformer.checks.transformer_fixture import TransformerWeights
from transformer.host.transformer_block_host import (
    TRANSFORMER_HOST_SABOTAGE,
    transformer_host_backward,
    transformer_host_forward,
    transformer_host_weights,
    transformer_host_weights_opts,
)
# lane/block-options (2026-09-17): the block options record and the two
# tails, `transformer/block_options.mojo`'s and the GPU binding's word for
# word (the GPU binding's tail section is the mirror of this comment).
#
#   params tail, 17 ints, appended after the entry's own scalars:
#     +0  rope_theta_bits            Float32 bits of the RoPE base
#     +1  rope_scaling               0 none, 1 linear, 2 llama3
#     +2  rope_factor_bits           Float32 bits
#     +3  rope_low_freq_factor_bits  Float32 bits (llama3)
#     +4  rope_high_freq_factor_bits Float32 bits (llama3)
#     +5  rope_original_max_positions  int (llama3)
#     +6  rope_dim                   int, 0 = head_dim
#     +7  max_positions              int, the declared ceiling (8192 default)
#     +8  qkv_bias                   0 / 1
#     +9  o_bias                     0 / 1
#     +10 norm_kind                  0 rmsnorm, 1 layernorm, 2 rmsnorm_offset
#     +11 norm_eps_bits              Float32 bits
#     +12 norm_bias                  0 / 1
#     +13 mlp_kind                   0 swiglu, 1 gelu, 2 gelu_tanh, 3 geglu, 4 geglu_tanh
#     +14 mlp_bias                   0 / 1
#     +15 qk_norm                    0 / 1
#     +16 attn_softcap_bits          Float32 bits, 0 = none
#
#   addrs tail, 11 addresses (0 = absent), appended after the entry's own:
#     +0  q_proj.bias                +1 k_proj.bias        +2 v_proj.bias
#     +3  o_proj.bias                +4 input_layernorm.bias
#     +5  post_attention_layernorm.bias                    +6 up_proj.bias
#     +7  down_proj.bias             +8 gate_proj.bias
#     +9  q_norm.weight              +10 k_norm.weight
#
# Every forward entry accepts its old lists or the old lists plus the tails;
# with an ungated MLP the base gate_proj.weight slot carries 0. The backward
# keeps its old lists and the default record.
from transformer.block_options import (
    BLOCK_OPTION_ADDRS,
    BLOCK_OPTION_PARAMS,
    BlockOptions,
)
from transformer.experiments.profile import transformer_arithmetic_profile, transformer_arithmetic_profile_changed


def _write(addr: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), f32_ptr(addr), len(values))


def _opt_read(addr: Int, n: Int, on: Bool, name: String, what: String) raises -> List[Float32]:
    """One optional tensor: read at its length when its flag is on, empty
    otherwise; a presence/flag mismatch is refused by the tensor's name."""
    if on:
        if addr == 0:
            raise Error(what + ": " + name + " is required by its option and its address is null")
        return read_f32(addr, n)
    if addr != 0:
        raise Error(what + ": " + name + " was passed but its option is off; pass the option or drop the tensor")
    return List[Float32]()


def _host_weights_from(
    a: List[Int], dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    rope_positions: Int, opts: BlockOptions, what: String,
) raises -> TransformerWeights:
    """The oracle's weight struct from the 13 + 11 address layout under the
    record: the nine (the gate only when the MLP is gated) and the eleven
    optional tensors, each read at its length or left empty."""
    var qw = nh * hd
    var kw = nkv * hd
    var w_gate = List[Float32]()
    if opts.gated():
        if a[7] == 0:
            raise Error(what + ": gate_proj.weight address is null (a gated MLP needs it)")
        w_gate = read_f32(a[7], it * dm)
    elif a[7] != 0:
        raise Error(what + ": gate_proj.weight was passed but the MLP is ungated (mlp gelu / gelu_tanh)")
    return transformer_host_weights_opts(
        dm, nh, nkv, hd, it, rope_positions, opts,
        read_f32(a[1], dm), read_f32(a[2], dm), read_f32(a[3], qw * dm),
        read_f32(a[4], kw * dm), read_f32(a[5], kw * dm), read_f32(a[6], dm * qw),
        w_gate^, read_f32(a[8], it * dm), read_f32(a[9], dm * it),
        _opt_read(a[13], qw, opts.qkv_bias, "q_proj.bias", what),
        _opt_read(a[14], kw, opts.qkv_bias, "k_proj.bias", what),
        _opt_read(a[15], kw, opts.qkv_bias, "v_proj.bias", what),
        _opt_read(a[16], dm, opts.o_bias, "o_proj.bias", what),
        _opt_read(a[17], dm, opts.norm_bias, "input_layernorm.bias", what),
        _opt_read(a[18], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
        _opt_read(a[19], it, opts.mlp_bias, "up_proj.bias", what),
        _opt_read(a[20], dm, opts.mlp_bias, "down_proj.bias", what),
        _opt_read(a[21], it, opts.has_gate_bias(), "gate_proj.bias", what),
        _opt_read(a[22], hd, opts.qk_norm, "q_norm.weight", what),
        _opt_read(a[23], hd, opts.qk_norm, "k_norm.weight", what),
    )


def _run_decode_steps(
    a: List[Int], b: Int, n: Int, dm: Int, nh: Int, nkv: Int, hd: Int,
    it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`n` decode tokens per row on the host in one call (lane x4: replaces
    the decode door's per-token call of `transformer_decode_step`): `a` is
    `_run_forward`'s layout with x and y_out the caller's (B, n, d_model)
    slabs. Token t of every row is copied into a (B, d_model) buffer, the
    block runs at L = 1 on the carried caches, and the row results are
    copied back, so the bytes are n `transformer_decode_step` calls'.
    Serial by nature: each token reads the cache the previous one wrote."""
    if b <= 0 or n <= 0 or dm <= 0:
        raise Error("transformer decode: B, n_tokens and d_model must be positive")
    var row = b * dm
    var xt = List[Float32](length=row, fill=Float32(0.0))
    var yt = List[Float32](length=row, fill=Float32(0.0))
    var src = f32_ptr(a[0])
    var dst = f32_ptr(a[12])
    var at = a.copy()
    at[0] = Int(xt.unsafe_ptr())
    at[12] = Int(yt.unsafe_ptr())
    var cached = s0
    for t in range(n):  # driver-loop: over autoregressive decode tokens (each reads the cache the previous one wrote)
        for r in range(b):  # driver-loop: over the batch rows, token t's row copy in
            memcpy(dest=xt.unsafe_ptr() + r * dm, src=src + (r * n + t) * dm, count=dm)
        cached = _run_forward(at, b, 1, dm, nh, nkv, hd, it, smax, cached, window, True, opts)
        for r in range(b):  # driver-loop: over the batch rows, token t's row copy out
            memcpy(dest=dst + (r * n + t) * dm, src=yt.unsafe_ptr() + r * dm, count=dm)
    _ = xt^
    _ = yt^
    return cached


def _run_forward(
    a: List[Int], b: Int, l: Int, dm: Int, nh: Int, nkv: Int, hd: Int,
    it: Int, smax: Int, s0: Int, window: Int, carried: Bool,
    opts: BlockOptions,
) raises -> Int:
    """`a` is the device binding's thirteen: x, the nine weights, k_cache,
    v_cache, y_out, then the eleven optional addresses (zeros when absent).
    With `carried` False the caches are zero and not written back (the
    fresh prefill). `opts` is the record from the params tail (the default
    record when no tail was sent; `TransformerWeights` at the default
    record with every optional list empty is exactly what
    `transformer_host_weights` built before this lane)."""
    if b <= 0 or l <= 0:
        raise Error("transformer: B and L must be positive")
    if smax <= 0:
        raise Error("transformer: max_tokens must be positive")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    var qw = nh * hd
    var kw = nkv * hd
    var w = _host_weights_from(a, dm, nh, nkv, hd, it, smax, opts, String("transformer"))
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd
    var k_in = List[Float32]()
    var v_in = List[Float32]()
    if carried:
        k_in = read_f32(a[10], cache_n)
        v_in = read_f32(a[11], cache_n)
    var x = read_f32(a[0], b * l * dm)
    var out = transformer_host_forward(
        w, x, b, l, smax, s0, window, k_in, v_in, carried
    )
    _write(a[12], out.y)
    if carried:
        _write(a[10], out.k_cache)
        _write(a[11], out.v_cache)
    return out.cached_tokens


struct HostTransformerSession(Movable, Writable):
    # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
    # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
    # (the session's `tape` / `generation` and `invalidate_tape`.)
    var busy: Bool
    var closed: Bool

    def __init__(out self):
        self.busy = False
        self.closed = False

    def write_to(self,mut writer: Some[Writer]):
        writer.write("HostTransformerSession")

    def write_repr_to(self,mut writer: Some[Writer]):
        writer.write("HostTransformerSession")


def _backward_run(
    a: List[Int], b: Int, l: Int, dm: Int, nh: Int, nkv: Int, hd: Int,
    it: Int, window: Int,
) raises:
    """`transformer_backward`'s GIL-free half: read the weights, x and the
    output gradient, run the host VJP, write grad_x and the nine weight
    gradients (`a` is the binding's 21 addresses)."""
    var qw = nh * hd
    var kw = nkv * hd
    var w = transformer_host_weights(
        dm, nh, nkv, hd, it, l,
        read_f32(a[1], dm), read_f32(a[2], dm), read_f32(a[3], qw * dm),
        read_f32(a[4], kw * dm), read_f32(a[5], kw * dm), read_f32(a[6], dm * qw),
        read_f32(a[7], it * dm), read_f32(a[8], it * dm), read_f32(a[9], dm * it),
    )
    var x = read_f32(a[0], b * l * dm)
    var d_out = read_f32(a[10], b * l * dm)
    var grads = transformer_host_backward(w, x, d_out, b, l, window)
    for i in range(10):
        _write(a[11 + i], grads[i])


# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (`_session_forward_tape_run` / `_session_backward_tape_run` over HostAttentionTape.)
