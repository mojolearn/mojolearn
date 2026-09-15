# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for public neural INFERENCE (lane/inference-tokenizer-neural,
2026-09-15): the small MLP's logits and the TransformerBlock stateless
prefill, FORWARD ONLY, from weights trained anywhere.

HOST ONLY, LOADED BY PATH (`python/mojolearn/neural_inference.py`), like
the byte LM's, the forest's and the tokenizer's bindings, and SHIPPED in the
wheels. It is the inference half of two training families that stay source
reference builds (`_mojolearn_training_host`, `_mojolearn_transformer_host`):
no optimizer, no loss, no backward and no decode cache is reachable from an
export here, so none of that code is compiled into this binary.

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

THE SABOTAGE is `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walked descending, which
moves both MLP projections and every Transformer projection.
`neural_host_sabotage()` reads it back.
"""
from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import copy_f32, f32_ptr, read_f32, read_i32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.host.gemm_oracle import GEMM_ORACLE_HOST_SABOTAGE, OP_NT, gemm_oracle
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import M2_D_CONV, Mamba2Dims, Mamba2Weights
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import M3_D_STATE, Mamba3Dims, Mamba3Weights
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
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("neural host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if Int(py=len(addrs)) != n:
        raise Error(what + ": expected " + String(n) + " addresses, got " + String(Int(py=len(addrs))))
    var out = List[Int](capacity=n)
    for i in range(n):
        var a = _index(addrs[i])
        if a == 0:
            raise Error(what + ": address " + String(i) + " is null")
        out.append(a)
    return out^


def _finite(values: List[Float32], what: String) raises:
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(what + " is not finite")


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
        var p1 = gemm_oracle(x, w1, OP_NT, rows, 16, 8)
        _finite(p1, String("mlp_forward_logits: the first projection"))
        var hidden = host_mlp_bias_activation(p1, b1, rows, 16, 1)
        var p2 = gemm_oracle(hidden, w2, OP_NT, rows, 3, 16)
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
    var a = _addrs(addrs, 11, String("transformer_forward_fresh"))
    if Int(py=len(params)) != 8:
        raise Error("transformer_forward_fresh: expected 11 addresses and 8 scalars")
    var b = _index(params[0])
    var l = _index(params[1])
    var dm = _index(params[2])
    var nh = _index(params[3])
    var nkv = _index(params[4])
    var hd = _index(params[5])
    var it = _index(params[6])
    var window = _index(params[7])
    if b <= 0 or l <= 0:
        raise Error("transformer: B and L must be positive")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    if dm <= 0 or nh <= 0 or nkv <= 0 or hd <= 0 or it <= 0:
        raise Error("transformer: d_model, n_heads, n_kv_heads, head_dim and intermediate must be positive")
    var out_len = 0
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
        var out = transformer_host_forward(
            w, x, b, l, l, 0, window, List[Float32](), List[Float32]()
        )
        copy_f32(out.y.unsafe_ptr(), f32_ptr(a[10]), len(out.y))
        out_len = out.cached_tokens
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
    for i in range(3):
        out.append(_index(params[i]))
    if out[0] < 1 or out[1] < 1 or out[2] < 1:
        raise Error(what + ": B, L and d_model must be positive")
    return out^


def mamba1_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """12 addresses (x, the ten `Mamba1Block` weights, y); params B, L,
    d_model. Returns B * L * d_model, the cells written."""
    var a = _addrs(addrs, 12, String("mamba1_forward_fresh"))
    var s = _shape3(params, 3, String("mamba1_forward_fresh"))
    var b = s[0]
    var l = s[1]
    var dm = s[2]
    with GILReleased(Python()):
        var dims = MambaDims.of(dm)
        var di = dims.d_inner
        var r = dims.dt_rank
        var xr = dims.x_proj_rows()
        var w = MambaWeights(dims)
        w.norm_w = read_f32(a[1], dm)
        w.w_in = read_f32(a[2], 2 * di * dm)
        w.conv_w = read_f32(a[3], di * D_CONV)
        w.conv_b = read_f32(a[4], di)
        w.w_x = read_f32(a[5], xr * di)
        w.w_dt = read_f32(a[6], di * r)
        w.b_dt = read_f32(a[7], di)
        w.a_log = read_f32(a[8], di * D_STATE)
        w.d_skip = read_f32(a[9], di)
        w.w_out = read_f32(a[10], dm * di)
        var state = MambaState(b, w.dims)
        var st = mamba_block_oracle(w, read_f32(a[0], b * l * dm), b, l, state)
        copy_f32(st.residual_out.unsafe_ptr(), f32_ptr(a[11]), b * l * dm)
    return PythonObject(b * l * dm)


def mamba2_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """11 addresses (x, the nine `Mamba2Block` weights, y); params B, L,
    d_model, dt_lo, dt_hi. Returns B * L * d_model."""
    var a = _addrs(addrs, 11, String("mamba2_forward_fresh"))
    var s = _shape3(params, 5, String("mamba2_forward_fresh"))
    var b = s[0]
    var l = s[1]
    var dm = s[2]
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    with GILReleased(Python()):
        var dims = Mamba2Dims.of(dm)
        var di = dims.d_inner
        var cd = dims.conv_dim()
        var nh = dims.nheads
        var dip = dims.d_in_proj()
        var w = Mamba2Weights(dims)
        w.norm_w = read_f32(a[1], dm)
        w.w_in = read_f32(a[2], dip * dm)
        w.conv_w = read_f32(a[3], cd * M2_D_CONV)
        w.conv_b = read_f32(a[4], cd)
        w.dt_bias = read_f32(a[5], nh)
        w.a_log = read_f32(a[6], nh)
        w.d_skip = read_f32(a[7], nh)
        w.gnorm_w = read_f32(a[8], di)
        w.w_out = read_f32(a[9], dm * di)
        var state = Mamba2State(b, w.dims)
        var st = mamba2_block_oracle(w, read_f32(a[0], b * l * dm), b, l, dt_lo, dt_hi, state)
        copy_f32(st.residual_out.unsafe_ptr(), f32_ptr(a[10]), b * l * dm)
    return PythonObject(b * l * dm)


def mamba3_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """11 addresses (x, the nine `Mamba3Block` weights, y); params B, L,
    d_model. Returns B * L * d_model."""
    var a = _addrs(addrs, 11, String("mamba3_forward_fresh"))
    var s = _shape3(params, 3, String("mamba3_forward_fresh"))
    var b = s[0]
    var l = s[1]
    var dm = s[2]
    with GILReleased(Python()):
        var dims = Mamba3Dims.of(dm)
        var di = dims.d_inner
        var nh = dims.nheads
        var dip = dims.d_in_proj()
        var w = Mamba3Weights(dims)
        w.norm_w = read_f32(a[1], dm)
        w.w_in = read_f32(a[2], dip * dm)
        w.dt_bias = read_f32(a[3], nh)
        w.bnorm_w = read_f32(a[4], M3_D_STATE)
        w.cnorm_w = read_f32(a[5], M3_D_STATE)
        w.b_bias = read_f32(a[6], nh * M3_D_STATE)
        w.c_bias = read_f32(a[7], nh * M3_D_STATE)
        w.d_skip = read_f32(a[8], nh)
        w.w_out = read_f32(a[9], dm * di)
        var state = Mamba3State(b, w.dims)
        var st = mamba3_block_oracle(w, read_f32(a[0], b * l * dm), b, l, state)
        copy_f32(st.residual_out.unsafe_ptr(), f32_ptr(a[10]), b * l * dm)
    return PythonObject(b * l * dm)


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


@export
def PyInit__mojolearn_neural_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_neural_host")
        module.def_function[neural_host_numeric_mode_binding]("neural_host_numeric_mode")
        module.def_function[neural_host_vendor_binding]("neural_host_vendor")
        module.def_function[neural_host_column_binding]("neural_host_column")
        module.def_function[neural_host_sabotage_binding]("neural_host_sabotage")
        module.def_function[mlp_forward_logits_binding]("mlp_forward_logits")
        module.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        module.def_function[mamba1_forward_fresh_binding]("mamba1_forward_fresh")
        module.def_function[mamba2_forward_fresh_binding]("mamba2_forward_fresh")
        module.def_function[mamba3_forward_fresh_binding]("mamba3_forward_fresh")
        module.def_function[embedding_forward_binding]("embedding_forward")
        module.def_function[rms_norm_forward_binding]("rms_norm_forward")
        module.def_function[linear_forward_binding]("linear_forward")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_neural_host: ", error))
