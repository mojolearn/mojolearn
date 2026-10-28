# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for public neural INFERENCE (lane/inference-tokenizer-neural,
2026-09-15): the small MLP's logits, and the TransformerBlock and Mamba
blocks' forward and DECODE STEP, FORWARD ONLY, from weights trained anywhere.

HOST ONLY, LOADED BY PATH (`python/mojolearn/neural_inference.py`), like
the byte LM's, the forest's and the tokenizer's bindings, and SHIPPED in the
wheels. It is the inference half of two training families that stay source
reference builds (`_mojolearn_training_host`, `_mojolearn_transformer_host`):
no optimizer, no loss and no backward is reachable from an export here, so
none of that code is compiled into this binary.

THE DECODE CACHE IS REACHABLE SINCE lane/stateful-cpu-decoding (2026-09-16),
and it is not a second implementation. Every carried entry below is the
SAME host function its fresh sibling calls, handed the caller's state
instead of a zero one: `transformer_host_forward` with a cache pair rather
than an empty one, and `mamba{1,2,3}_block_oracle` with a read state rather
than a constructed zero. That is what makes "a step-by-step decode equals
the one-shot prefill" a property of the arithmetic. Measured on this
column, B = 2, L = 16, d_model = 32: the transformer at window 0 and
window 8, and all three Mamba blocks, are BITWISE EQUAL at every position,
and the same comparison FIRES (first differing position and both values
printed) when one carried cell is moved by one ULP.

WHAT IT COMPUTES, and from which functions (no arithmetic of its own):

  `mlp_forward_logits(addrs, params)`
      `SmallMLPTrainer.predict_logits` (python/mojolearn/_mlp_impl.py
      `_forward`): `gemm_oracle(x, weight1, OP_NT)`, then
      `host_mlp_bias_activation(.., bias1, relu=1)`, then
      `gemm_oracle(hidden, weight2, OP_NT)`, then
      `host_mlp_bias_activation(.., bias2, relu=0)`; the same oracles the
      linalg and training host bindings call for those four steps.
      `addrs` = [x (rows*8), weight1 (16*8), bias1 (16), weight2 (3*16),
      bias2 (3), logits_out (rows*3)]; `params` = [rows], 1..256.
  `transformer_forward_fresh(addrs, params)`
      `_mojolearn_transformer_host`'s entry of the same name, word for word:
      eleven addresses (x, the nine weights, y) and eight scalars (B, L,
      d_model, n_heads, n_kv_heads, head_dim, intermediate, window), through
      `transformer/host/transformer_block_host.mojo::transformer_host_forward`
      from a zero cache.
  `transformer_forward(addrs, params)`, `transformer_decode_step(addrs, params)`
      `_mojolearn_transformer_host`'s entries of the same names: thirteen
      addresses (x, the nine weights, k_cache, v_cache, y) through the same
      `transformer_host_forward` with the caller's caches, which are read
      at entry and written back. Returns the post-call cached_tokens.
  `mamba3_forward_fresh`
      the block oracle from a constructed zero state, the final state
      discarded. `mamba3_forward_fresh` takes `_mojolearn_mamba`'s fifteen
      addresses (the four report buffers included), the arity
      `Mamba3Block._call_fresh` passes.
  `mamba1_forward`, `mamba1_decode_step` (14 addresses),
  `mamba2_forward`, `mamba2_decode_step` (16), `mamba3_forward`,
  `mamba3_decode_step` (25)
      `_mojolearn_mamba_host`'s entries of the same names and contracts: the
      same three oracles with the state pieces read at entry and written
      back, so `Mamba1Block`, `Mamba2Block` and `Mamba3Block` run their
      decode through this shipped binding.

THE SABOTAGE is `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walked descending, which
moves both MLP projections and every Transformer projection.
`neural_host_sabotage()` reads it back.
"""

from training.neural_arithmetic_profile import neural_training_profile

from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from bindings.py_balanced import py_call as _balanced_call
from bindings.sha256_door import sha256_hexdigest
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from core.neural_host_dispatch import NATIVE_HOST_PRIMITIVE_API, NATIVE_HOST_MAMBA_STATE_API, NATIVE_HOST_TRANSFORMER_API, NATIVE_HOST_MAMBA_BLOCK_API
from core.mojolearn_neural_host_impl import _finite, _opt_read, _transformer_weights_from, _write, _run_forward, _m1_weights, _mamba1_run, _m2_weights, _mamba2_run, _m3_weights, _mamba3_run
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import copy_f32, f32_ptr, read_f32, read_i32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE, OP_NT
from gemm.host.neural_gemm import gemm_oracle
from gemm.host.neural_gemm import gemm_host_rows
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights
# TOMBSTONE: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix: profile/state/branches in this file; LOSER avg 2.30x mamba1-forward) deleted 2026-10-11 by 291629888; code recoverable at 291629888^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN_1.patch
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.checks.mamba3_oracle import Mamba3State, mamba3_block_oracle
from training.host.mlp_oracle import host_mlp_bias_activation
from training.host.samba_ops_oracle import (
    host_samba_embedding_forward,
    host_samba_linear_forward,
    host_samba_rms_norm_forward,
)
from transformer.host.transformer_block_host import (
    transformer_host_forward,
    transformer_host_weights,
    transformer_host_weights_opts,
)
from transformer.checks.transformer_fixture import TransformerWeights
# lane/block-options (2026-09-17): the block options record and its two
# tails; `bindings/_mojolearn_transformer_host.mojo`'s comment and
# `transformer/block_options.mojo` carry the order word for word (17 params:
# rope_theta_bits, rope_scaling, rope_factor_bits, rope_low_freq_factor_bits,
# rope_high_freq_factor_bits, rope_original_max_positions, rope_dim,
# max_positions, qkv_bias, o_bias, norm_kind, norm_eps_bits, norm_bias,
# mlp_kind, mlp_bias, qk_norm, attn_softcap_bits; 11 addrs: q_proj.bias,
# k_proj.bias, v_proj.bias, o_proj.bias, input_layernorm.bias,
# post_attention_layernorm.bias, up_proj.bias, down_proj.bias,
# gate_proj.bias, q_norm.weight, k_norm.weight). Each transformer entry
# accepts its old lists or the old lists plus the tails; with an ungated
# MLP the base gate_proj.weight slot carries 0.
from transformer.block_options import (
    BLOCK_OPTION_ADDRS,
    BLOCK_OPTION_PARAMS,
    BlockOptions,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("neural host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if Int(py=len(addrs)) != n:
        raise Error(what + ": expected " + String(n) + " addresses, got " + String(Int(py=len(addrs))))
    var out = List[Int](capacity=n)
    for i in range(n):  # small-loop(address argument words, a fixed count per entry point)
        var a = _index(addrs[i])
        if a == 0:
            raise Error(what + ": address " + String(i) + " is null")
        out.append(a)
    return out^


def _addrs_tail(addrs: PythonObject, n: Int, gate_slot: Int, what: String) raises -> List[Int]:
    """`n` addresses or `n + BLOCK_OPTION_ADDRS`, returned as the longer
    form with zeros where the tail was not sent; base slots null-checked
    except `gate_slot` (0 under an ungated MLP)."""
    var got = Int(py=len(addrs))
    if got != n and got != n + BLOCK_OPTION_ADDRS:
        raise Error(
            what + ": expected " + String(n) + " addresses, or " + String(n)
            + " + " + String(BLOCK_OPTION_ADDRS)
            + " with the block options tail, got " + String(got)
        )
    var out = List[Int](capacity=n + BLOCK_OPTION_ADDRS)
    for i in range(got):  # small-loop(address argument words plus the fixed options tail)
        var a = _index(addrs[i])
        if i < n and i != gate_slot and a == 0:
            raise Error(what + ": address " + String(i) + " is null")
        out.append(a)
    while len(out) < n + BLOCK_OPTION_ADDRS:  # small-loop(pad the fixed options tail of address words)
        out.append(0)
    return out^


def _params_tail(params: PythonObject, n: Int, what: String) raises -> List[Int]:
    var got = Int(py=len(params))
    if got != n and got != n + BLOCK_OPTION_PARAMS:
        raise Error(
            what + ": expected " + String(n) + " scalars, or " + String(n)
            + " + " + String(BLOCK_OPTION_PARAMS)
            + " with the block options tail, got " + String(got)
        )
    var out = List[Int](capacity=got)
    for i in range(got):  # small-loop(scalar argument words plus the fixed options tail)
        out.append(_index(params[i]))
    return out^








def neural_primitives_api_binding() raises -> PythonObject:
    var api = NATIVE_HOST_PRIMITIVE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def mamba_state_api_binding() raises -> PythonObject:
    var api = NATIVE_HOST_MAMBA_STATE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def transformer_block_api_binding() raises -> PythonObject:
    var api = NATIVE_HOST_TRANSFORMER_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def mamba_block_api_binding() raises -> PythonObject:
    var api = NATIVE_HOST_MAMBA_BLOCK_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def neural_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "neural host: IDENTICAL only; pass -D MOJOLEARN_NUMERIC_IDENTICAL=1"
        " (bindings/build_neural_host.sh does)"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def neural_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def neural_host_column_binding() raises -> PythonObject:
    """THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "neural host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_neural_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def neural_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mlp_forward_logits_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The 8-16-3 MLP's logits. Returns rows * 3, the cells written."""
    var a = _addrs(addrs, 6, String("mlp_forward_logits"))
    if Int(py=len(params)) != 1:
        raise Error("mlp_forward_logits: params must be [rows]")
    var rows = _index(params[0])
    if rows < 1 or rows > 256:
        raise Error("mlp_forward_logits: rows must be in [1, 256], got " + String(rows))
    var wrote = 0
    with GILReleased(Python()):
        var x = read_f32(a[0], rows * 8)
        var w1 = read_f32(a[1], 16 * 8)
        var b1 = read_f32(a[2], 16)
        var w2 = read_f32(a[3], 3 * 16)
        var b2 = read_f32(a[4], 3)
        var p1 = gemm_host_rows(x, w1, OP_NT, rows, 16, 8)
        _finite(p1, String("mlp_forward_logits: the first projection"))
        var hidden = host_mlp_bias_activation(p1, b1, rows, 16, 1)
        var p2 = gemm_host_rows(hidden, w2, OP_NT, rows, 3, 16)
        _finite(p2, String("mlp_forward_logits: the second projection"))
        var logits = host_mlp_bias_activation(p2, b2, rows, 3, 0)
        copy_f32(logits.unsafe_ptr(), f32_ptr(a[5]), rows * 3)
        wrote = rows * 3
    return PythonObject(wrote)


def transformer_forward_fresh_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Stateless prefill from a zero cache of capacity L (or a window ring),
    only y written. Eleven pointers: x, nine weights, y. Eight scalars: B,
    L, d_model, n_heads, n_kv_heads, head_dim, intermediate, window."""
    var what = String("transformer_forward_fresh")
    var a0 = _addrs_tail(addrs, 11, 7, what)
    var p = _params_tail(params, 8, what)
    var opts = BlockOptions.from_params(p, 8)
    # Into `transformer_forward`'s 13 + 11 layout (slots 10 and 11 unused).
    var a = List[Int]()
    for i in range(10):  # small-loop(fixed address-word layout)
        a.append(a0[i])
    a.append(0)
    a.append(0)
    a.append(a0[10])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(fixed options-tail address words)
        a.append(a0[11 + i])
    var b = p[0]
    var l = p[1]
    var dm = p[2]
    var nh = p[3]
    var nkv = p[4]
    var hd = p[5]
    var it = p[6]
    var window = p[7]
    if b <= 0 or l <= 0:
        raise Error("transformer: B and L must be positive")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    if dm <= 0 or nh <= 0 or nkv <= 0 or hd <= 0 or it <= 0:
        raise Error("transformer: d_model, n_heads, n_kv_heads, head_dim and intermediate must be positive")
    var out_len = 0
    with GILReleased(Python()):
        var w = _transformer_weights_from(a, dm, nh, nkv, hd, it, l, opts, what)
        var x = read_f32(a[0], b * l * dm)
        var out = transformer_host_forward(
            w, x, b, l, l, 0, window, List[Float32](), List[Float32](), False
        )
        copy_f32(out.y.unsafe_ptr(), f32_ptr(a[12]), len(out.y))
        out_len = out.cached_tokens
    return PythonObject(out_len)


# ---------------------------------------------------------------------------
# THE CARRIED KV CACHE (lane/stateful-cpu-decoding, 2026-09-16).
# `_run_forward`, `transformer_forward` and `transformer_decode_step` are
# `bindings/_mojolearn_transformer_host.mojo`'s entries of the same names,
# word for word, over the SAME `transformer_host_forward` the fresh entry
# above already calls: the fresh prefill is that function with an empty
# cache pair, a carried call is that function with the caller's. ONE
# spelling, which is what makes "a step equals the prefill" a property of
# the arithmetic rather than of two implementations agreeing by luck.
# ---------------------------------------------------------------------------




def transformer_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """One block call from the caller's state. `addrs`: 0 x, 1-9 the nine
    weights (input_layernorm, post_attention_layernorm, q, k, v, o, gate, up,
    down), 10 k_cache, 11 v_cache, 12 y_out. `params`: 0 B, 1 L, 2 d_model,
    3 n_heads, 4 n_kv_heads, 5 head_dim, 6 intermediate, 7 max_tokens,
    8 cached_tokens, 9 window. Returns the post-call cached_tokens."""
    var what = String("transformer_forward")
    var a = _addrs_tail(addrs, 13, 7, what)
    if Int(py=len(params)) != 10 and Int(py=len(params)) != 10 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_forward: params must contain 10 values (B, L,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail"
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
        out_len = _run_forward(a, b, l, dm, nh, nkv, hd, it, smax, s0, window, opts)
    return PythonObject(out_len)


def transformer_forward_int15_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`transformer_forward` under numeric_profile="fixed15_v1"
    (lane/lowbit-blocks, 2026-09-29): the seven projections and the score
    product S11 on gemm.int15i64.v1's host oracle, both operands quantized
    row by row along the contracted extent (the weights by the same rule
    that packs them on a GPU, so the same codes), P.V on fp32. The same
    lists as `transformer_forward`, the float32 weights included, and the
    two option tails ALWAYS sent: 24 addresses, 27 params. Returns the
    post-call cached_tokens."""
    var what = String("transformer_forward_int15")
    if Int(py=len(addrs)) != 13 + BLOCK_OPTION_ADDRS or Int(py=len(params)) != 10 + BLOCK_OPTION_PARAMS:
        raise Error(what + ": addrs must hold 24 addresses and params 27 values (both tails always)")
    var a = _addrs_tail(addrs, 13, 7, what)
    var p = _params_tail(params, 10, what)
    var opts = BlockOptions.from_params(p, 10)
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts, True)
    return PythonObject(out_len)


def transformer_decode_step_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """One decode token: `transformer_forward` at L = 1, the SAME entry
    point. `addrs`: the same thirteen. `params`: 0 B, 1 d_model, 2 n_heads,
    3 n_kv_heads, 4 head_dim, 5 intermediate, 6 max_tokens, 7 cached_tokens,
    8 window."""
    var what = String("transformer_decode_step")
    var a = _addrs_tail(addrs, 13, 7, what)
    if Int(py=len(params)) != 9 and Int(py=len(params)) != 9 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_decode_step: params must contain 9 values (B,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail"
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
        out_len = _run_forward(a, b, 1, dm, nh, nkv, hd, it, smax, s0, window, opts)
    return PythonObject(out_len)


# ===========================================================================
# Mamba-1, Mamba-2 and Mamba-3: the zero-state prefill, only y written
# (lane/inference-neural-forward, 2026-09-15). The weights are read in the
# `_mojolearn_mamba_host` order (`_m1_weights`, `_m2_weights`, `_m3_weights`
# there), the state is the oracle's own zero construction, and the final
# state and the report stages are discarded.
# ===========================================================================


def _shape3(params: PythonObject, n: Int, what: String) raises -> List[Int]:
    if Int(py=len(params)) != n:
        raise Error(what + ": expected " + String(n) + " parameters, got " + String(Int(py=len(params))))
    var out = List[Int](capacity=3)
    for i in range(3):  # small-loop(the three shape argument words)
        out.append(_index(params[i]))
    if out[0] < 1 or out[1] < 1 or out[2] < 1:
        raise Error(what + ": B, L and d_model must be positive")
    return out^






# TOMBSTONE: mamba1_session_info, mamba1_forward_fresh, mamba2_forward_fresh (dead native entries: no live caller) deleted 2026-10-26 by 391118b63; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_mamba.patch


def mamba1_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`bindings/_mojolearn_mamba.mojo::mamba1_forward_binding`'s contract:
    14 addresses, params B, L, d_model (lane/stateful-cpu-decoding)."""
    var a = _addrs(addrs, 14, String("mamba1_forward"))
    var s = _shape3(params, 3, String("mamba1_forward"))
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, s[0], s[1], s[2], position)
    return PythonObject(0)


def mamba1_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1 with the state carried; params B, d_model."""
    var a = _addrs(addrs, 14, String("mamba1_decode_step"))
    if Int(py=len(params)) != 2:
        raise Error("mamba1_decode_step: params must contain 2 values (B, d_model)")
    var b = _index(params[0])
    var dm = _index(params[1])
    if b < 1 or dm < 1:
        raise Error("mamba1_decode_step: B and d_model must be positive")
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, b, 1, dm, position)
    return PythonObject(0)






def mamba2_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """16 addresses; params B, L, d_model, buf_len, dt_lo, dt_hi. Returns
    the post-call buf_len (lane/stateful-cpu-decoding)."""
    var a = _addrs(addrs, 16, String("mamba2_forward"))
    if Int(py=len(params)) != 6:
        raise Error(
            "mamba2_forward: params must contain 6 values (B, L, d_model,"
            " buf_len, dt_lo, dt_hi)"
        )
    var b = _index(params[0])
    var l = _index(params[1])
    var dm = _index(params[2])
    var q0 = _index(params[3])
    var dt_lo = Float32(Float64(py=params[4]))
    var dt_hi = Float32(Float64(py=params[5]))
    if dm < 1:
        raise Error("mamba2_forward: d_model must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, l, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)


def mamba2_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, dt_lo, dt_hi."""
    var a = _addrs(addrs, 16, String("mamba2_decode_step"))
    if Int(py=len(params)) != 5:
        raise Error(
            "mamba2_decode_step: params must contain 5 values (B, d_model,"
            " buf_len, dt_lo, dt_hi)"
        )
    var b = _index(params[0])
    var dm = _index(params[1])
    var q0 = _index(params[2])
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    if dm < 1:
        raise Error("mamba2_decode_step: d_model must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, 1, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)






def mamba3_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """15 addresses (x, the nine weights, y, the four reports); params B,
    L, d_model. A zero state whose resumption pieces are discarded --
    `bindings/_mojolearn_mamba.mojo`'s entry of the same name, WHICH IS THE
    ARITY `Mamba3Block._call_fresh` PASSES. Before
    lane/stateful-cpu-decoding this file carried an eleven-address variant
    of its own, reachable only through a wrapper that has since been
    deleted."""
    if Int(py=len(addrs)) != 15 or Int(py=len(params)) != 3:
        raise Error("mamba3_forward_fresh: expected 15 addresses and 3 parameters (B, L, d_model)")
    var a0 = _addrs(addrs, 15, String("mamba3_forward_fresh"))
    var a = List[Int]()
    for i in range(10):  # small-loop(fixed address-word layout)
        a.append(a0[i])
    for _ in range(10):  # small-loop(fixed address-word layout)
        a.append(0)
    for i in range(10, 15):  # small-loop(fixed address-word layout)
        a.append(a0[i])
    var b = _index(params[0])
    var l = _index(params[1])
    var dm = _index(params[2])
    if b < 1 or l < 1 or dm < 1:
        raise Error("mamba3_forward_fresh: B, L and d_model must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, 0, 0, True)
    return PythonObject(out_len)


def mamba3_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """25 addresses; params B, L, d_model, buf_len, pending
    (lane/stateful-cpu-decoding)."""
    var a = _addrs(addrs, 25, String("mamba3_forward"))
    if Int(py=len(params)) != 5:
        raise Error(
            "mamba3_forward: params must contain 5 values (B, L, d_model,"
            " buf_len, pending)"
        )
    var b = _index(params[0])
    var l = _index(params[1])
    var dm = _index(params[2])
    var q0 = _index(params[3])
    var pend = _index(params[4])
    if dm < 1:
        raise Error("mamba3_forward: d_model must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, q0, pend, False)
    return PythonObject(out_len)


def mamba3_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, pending."""
    var a = _addrs(addrs, 25, String("mamba3_decode_step"))
    if Int(py=len(params)) != 4:
        raise Error(
            "mamba3_decode_step: params must contain 4 values (B, d_model,"
            " buf_len, pending)"
        )
    var b = _index(params[0])
    var dm = _index(params[1])
    var q0 = _index(params[2])
    var pend = _index(params[3])
    if dm < 1:
        raise Error("mamba3_decode_step: d_model must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, 1, dm, q0, pend, False)
    return PythonObject(out_len)


# ===========================================================================
# The Samba stack's own steps around its blocks: the embedding gather, the
# final RMSNorm and the head, each `_mojolearn_training_host`'s entry of the
# same name and contract, over training/host/samba_ops_oracle.mojo.
# ===========================================================================


def embedding_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addresses [y (n*width), w (vocab*width), ids (n int32)]; params
    [n, vocab, width]. Returns n * width."""
    var a = _addrs(addrs, 3, String("embedding_forward"))
    if Int(py=len(params)) != 3:
        raise Error("embedding_forward: params must be [n_positions, vocab, width]")
    var n = _index(params[0])
    var vocab = _index(params[1])
    var width = _index(params[2])
    if n < 1 or vocab < 1 or width < 1:
        raise Error("embedding_forward: the embedding shape must be positive")
    with GILReleased(Python()):
        var y = host_samba_embedding_forward(read_f32(a[1], vocab * width), read_i32(a[2], n), n, vocab, width)
        copy_f32(y.unsafe_ptr(), f32_ptr(a[0]), n * width)
    return PythonObject(n * width)


def rms_norm_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addresses [y (m*dm), x (m*dm), w (dm)]; params [m, dm, eps]."""
    var a = _addrs(addrs, 3, String("rms_norm_forward"))
    if Int(py=len(params)) != 3:
        raise Error("rms_norm_forward: params must be [m, dm, eps]")
    var m = _index(params[0])
    var dm = _index(params[1])
    var eps = Float32(Float64(py=params[2]))
    if m < 1 or dm < 1:
        raise Error("rms_norm_forward: the shape must be positive")
    with GILReleased(Python()):
        var y = host_samba_rms_norm_forward(read_f32(a[1], m * dm), read_f32(a[2], dm), m, dm, eps)
        copy_f32(y.unsafe_ptr(), f32_ptr(a[0]), m * dm)
    return PythonObject(m * dm)


def linear_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`C[m, n] = A[m, k] . W[n, k]^T`; addresses [c, a, w]; params [m, n, k]."""
    var a = _addrs(addrs, 3, String("linear_forward"))
    if Int(py=len(params)) != 3:
        raise Error("linear_forward: params must be [m, n, k]")
    var m = _index(params[0])
    var n = _index(params[1])
    var k = _index(params[2])
    if m < 1 or n < 1 or k < 1:
        raise Error("linear_forward: the shape must be positive")
    with GILReleased(Python()):
        var c = host_samba_linear_forward(read_f32(a[1], m * k), read_f32(a[2], n * k), m, n, k)
        copy_f32(c.unsafe_ptr(), f32_ptr(a[0]), m * n)
    return PythonObject(m * n)


def sha256_hex_binding(value: PythonObject) raises -> PythonObject:
    """The SHA-256 of a bytes-like value, 64 lowercase hex digits (core/sha256.mojo)."""
    return sha256_hexdigest(value)


def neural_arithmetic_profile_binding() raises -> PythonObject:
    return PythonObject(neural_training_profile())


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_neural_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_neural_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_neural_host")
        module.def_function[neural_primitives_api_binding]("neural_primitives_api_v1")
        module.def_function[mamba_state_api_binding]("mamba_state_api_v1")
        module.def_function[transformer_block_api_binding]("transformer_block_api_v1")
        module.def_function[mamba_block_api_binding]("mamba_block_api_v1")
        module.def_function[neural_arithmetic_profile_binding]("neural_arithmetic_profile")
        module.def_function[sha256_hex_binding]("sha256_hex")
        module.def_function[neural_host_numeric_mode_binding]("neural_host_numeric_mode")
        module.def_function[neural_host_vendor_binding]("neural_host_vendor")
        module.def_function[neural_host_column_binding]("neural_host_column")
        module.def_function[neural_host_sabotage_binding]("neural_host_sabotage")
        module.def_function[mlp_forward_logits_binding]("mlp_forward_logits")
        module.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        module.def_function[transformer_forward_binding]("transformer_forward")
        module.def_function[transformer_forward_int15_binding]("transformer_forward_int15")
        module.def_function[transformer_decode_step_binding]("transformer_decode_step")
        # TOMBSTONE: mamba1_session_info, mamba1_forward_fresh, mamba2_forward_fresh (dead native entries: no live caller) deleted 2026-10-26 by 391118b63; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_mamba.patch
        module.def_function[mamba1_forward_binding]("mamba1_forward")
        module.def_function[mamba1_decode_step_binding]("mamba1_decode_step")
        # TOMBSTONE: mamba1_session_info, mamba1_forward_fresh, mamba2_forward_fresh (dead native entries: no live caller) deleted 2026-10-26 by 391118b63; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_mamba.patch
        module.def_function[mamba2_forward_binding]("mamba2_forward")
        module.def_function[mamba2_decode_step_binding]("mamba2_decode_step")
        module.def_function[mamba3_forward_fresh_binding]("mamba3_forward_fresh")
        module.def_function[mamba3_forward_binding]("mamba3_forward")
        module.def_function[mamba3_decode_step_binding]("mamba3_decode_step")
        module.def_function[embedding_forward_binding]("embedding_forward")
        module.def_function[rms_norm_forward_binding]("rms_norm_forward")
        module.def_function[linear_forward_binding]("linear_forward")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_neural_host: ", error))


def mamba1_profile_binding() -> PythonObject:
    return PythonObject(1)  # NN34 profile 2 (M1_SCAN=1) deleted 2026-10-11
