# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_transformer` family: the TransformerBlock
forward (stateless prefill, carried-state prefill, decode step) and its
zero-state prefill backward (CPU training for the transformer lanes,
2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch. The arithmetic is the lane's
two host oracles, `transformer/checks/transformer_oracle.mojo` and
`transformer/checks/transformer_backward_oracle.mojo`, composed by
`transformer/host/transformer_block_host.mojo` (which also converts the KV
cache between the device layout a `TransformerState` holds and the oracle's).
So the block output, the decode step and the ten gradients are meant to be
the GPU columns' bytes, full causal and sliding window alike.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_transformer.mojo`,
mirrored in `python/mojolearn/_transformer_impl.py`), so `TransformerBlock`
runs unchanged on a CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_transformer": "_mojolearn_transformer_host"`):
`transformer_forward` (13 addresses, 10 params), `transformer_forward_fresh`
(11 addresses, 8 params), `transformer_decode_step` (13 addresses, 9 params),
`transformer_backward` (21 addresses, 8 params), `transformer_vendor`
answering "cpu" and `transformer_numeric_mode`.

The sabotage arm (`transformer_host_sabotage`) is
`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walked descending.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import copy_f32, f32_ptr, read_f32
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


def transformer_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "transformer host: IDENTICAL only; pass -D MOJOLEARN_NUMERIC_IDENTICAL=1"
        " (bindings/build_transformer_host.sh does)"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def transformer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def transformer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "transformer host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_transformer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def transformer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(TRANSFORMER_HOST_SABOTAGE)


def transformer_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def transformer_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if len(addrs) != n:
        raise Error(
            what + ": addrs must contain " + String(n) + " addresses, got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(n):
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    return a^


def _write(addr: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), f32_ptr(addr), len(values))


def _addrs_tail(addrs: PythonObject, n: Int, gate_slot: Int, what: String) raises -> List[Int]:
    """`n` addresses or `n + BLOCK_OPTION_ADDRS`; returned as the longer
    form with zeros where the tail was not sent. Base slots are null-checked
    except `gate_slot`, which is 0 under an ungated MLP."""
    var got = len(addrs)
    if got != n and got != n + BLOCK_OPTION_ADDRS:
        raise Error(
            what + ": addrs must contain " + String(n) + " addresses, or "
            + String(n) + " + " + String(BLOCK_OPTION_ADDRS)
            + " with the block options tail, got " + String(got)
        )
    var a = List[Int]()
    for i in range(got):
        var address = Int(py=addrs[i])
        if i < n and i != gate_slot and address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    while len(a) < n + BLOCK_OPTION_ADDRS:
        a.append(0)
    return a^


def _params_tail(params: PythonObject, n: Int, what: String) raises -> List[Int]:
    var got = len(params)
    if got != n and got != n + BLOCK_OPTION_PARAMS:
        raise Error(
            what + ": params must contain " + String(n) + " values, or "
            + String(n) + " + " + String(BLOCK_OPTION_PARAMS)
            + " with the block options tail, got " + String(got)
        )
    var p = List[Int]()
    for i in range(got):
        p.append(Int(py=params[i]))
    return p^


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
        w_gate, read_f32(a[8], it * dm), read_f32(a[9], dm * it),
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


def transformer_forward_fresh_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Stateless prefill from a zero cache of capacity L (or a window ring),
    only y written. Eleven pointers: x, nine weights, y. Eight scalars: B, L,
    d_model, n_heads, n_kv_heads, head_dim, intermediate, window."""
    var what = String("transformer_forward_fresh")
    var a0 = _addrs_tail(addrs, 11, 7, what)
    var p = _params_tail(params, 8, what)
    var opts = BlockOptions.from_params(p, 8)
    var a = List[Int]()
    for i in range(10):
        a.append(a0[i])
    a.append(0)
    a.append(0)
    a.append(a0[10])
    for i in range(BLOCK_OPTION_ADDRS):
        a.append(a0[11 + i])
    var b = p[0]
    var l = p[1]
    var dm = p[2]
    var nh = p[3]
    var nkv = p[4]
    var hd = p[5]
    var it = p[6]
    var window = p[7]
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, l, dm, nh, nkv, hd, it, l, 0, window, False, opts)
    return PythonObject(out_len)


def transformer_forward_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One block call from the caller's state. `addrs`: 0 x, 1-9 the nine
    weights (input_layernorm, post_attention_layernorm, q, k, v, o, gate, up,
    down), 10 k_cache, 11 v_cache, 12 y_out. `params`: 0 B, 1 L, 2 d_model,
    3 n_heads, 4 n_kv_heads, 5 head_dim, 6 intermediate, 7 max_tokens,
    8 cached_tokens, 9 window. Returns the post-call cached_tokens."""
    var what = String("transformer_forward")
    var a = _addrs_tail(addrs, 13, 7, what)
    if len(params) != 10 and len(params) != 10 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_forward: params must contain 10 values (B, L,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail, got "
            + String(len(params))
        )
    var p = _params_tail(params, 10, what)
    var opts = BlockOptions.from_params(p, 10)
    var b = p[0]
    var l = p[1]
    var dm = p[2]
    var nh = p[3]
    var nkv = p[4]
    var hd = p[5]
    var it = p[6]
    var smax = p[7]
    var s0 = p[8]
    var window = p[9]
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, l, dm, nh, nkv, hd, it, smax, s0, window, True, opts)
    return PythonObject(out_len)


def transformer_decode_step_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One decode token: `transformer_forward` at L = 1. `addrs`: the same
    thirteen. `params`: 0 B, 1 d_model, 2 n_heads, 3 n_kv_heads, 4 head_dim,
    5 intermediate, 6 max_tokens, 7 cached_tokens, 8 window."""
    var what = String("transformer_decode_step")
    var a = _addrs_tail(addrs, 13, 7, what)
    if len(params) != 9 and len(params) != 9 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_decode_step: params must contain 9 values (B,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail, got "
            + String(len(params))
        )
    var p = _params_tail(params, 9, what)
    var opts = BlockOptions.from_params(p, 9)
    var b = p[0]
    var dm = p[1]
    var nh = p[2]
    var nkv = p[3]
    var hd = p[4]
    var it = p[5]
    var smax = p[6]
    var s0 = p[7]
    var window = p[8]
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, 1, dm, nh, nkv, hd, it, smax, s0, window, True, opts)
    return PythonObject(out_len)


def transformer_backward_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The zero-state prefill VJP. `addrs`: 0 x, 1-9 the nine weights,
    10 grad_output, 11 grad_x, 12-20 the nine weight gradients. `params`:
    0 B, 1 L, 2 d_model, 3 n_heads, 4 n_kv_heads, 5 head_dim,
    6 intermediate, 7 window."""
    var a = _addrs(addrs, 21, String("transformer backward"))
    if len(params) != 8:
        raise Error(
            "transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " window)"
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var nh = Int(py=params[3])
    var nkv = Int(py=params[4])
    var hd = Int(py=params[5])
    var it = Int(py=params[6])
    var window = Int(py=params[7])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("transformer backward: B, L and d_model must be positive")
    with GILReleased(Python()):
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
    return PythonObject(0)


@export
def PyInit__mojolearn_transformer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_transformer_host")
        module.def_function[transformer_host_numeric_mode_binding]("transformer_host_numeric_mode")
        module.def_function[transformer_host_vendor_binding]("transformer_host_vendor")
        module.def_function[transformer_host_column_binding]("transformer_host_column")
        module.def_function[transformer_host_sabotage_binding]("transformer_host_sabotage")
        module.def_function[transformer_vendor_binding]("transformer_vendor")
        module.def_function[transformer_numeric_mode_binding]("transformer_numeric_mode")
        module.def_function[transformer_forward_binding]("transformer_forward")
        module.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        module.def_function[transformer_decode_step_binding]("transformer_decode_step")
        module.def_function[transformer_backward_binding]("transformer_backward")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_transformer_host: ", e))
