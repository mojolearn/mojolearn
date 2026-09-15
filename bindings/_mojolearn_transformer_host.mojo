# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_transformer` family: the TransformerBlock
forward (stateless prefill, carried-state prefill, decode step) and its
zero-state prefill backward (CPU training for the transformer lanes,
2026-09-15; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.2).

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
from transformer.host.transformer_block_host import (
    TRANSFORMER_HOST_SABOTAGE,
    transformer_host_backward,
    transformer_host_forward,
    transformer_host_weights,
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


def _run_forward(
    a: List[Int], b: Int, l: Int, dm: Int, nh: Int, nkv: Int, hd: Int,
    it: Int, smax: Int, s0: Int, window: Int, carried: Bool,
) raises -> Int:
    """`a` is the device binding's thirteen: x, the nine weights, k_cache,
    v_cache, y_out. With `carried` False the caches are zero and not written
    back (the fresh prefill)."""
    if b <= 0 or l <= 0:
        raise Error("transformer: B and L must be positive")
    if smax <= 0:
        raise Error("transformer: max_tokens must be positive")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    var qw = nh * hd
    var kw = nkv * hd
    var w = transformer_host_weights(
        dm, nh, nkv, hd, it, smax,
        read_f32(a[1], dm), read_f32(a[2], dm), read_f32(a[3], qw * dm),
        read_f32(a[4], kw * dm), read_f32(a[5], kw * dm), read_f32(a[6], dm * qw),
        read_f32(a[7], it * dm), read_f32(a[8], it * dm), read_f32(a[9], dm * it),
    )
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
    var out = transformer_host_forward(w, x, b, l, smax, s0, window, k_in, v_in)
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
    var a0 = _addrs(addrs, 11, String("transformer_forward_fresh"))
    if len(params) != 8:
        raise Error("transformer_forward_fresh: expected 11 addresses and 8 scalars")
    var a = List[Int]()
    for i in range(10):
        a.append(a0[i])
    a.append(0)
    a.append(0)
    a.append(a0[10])
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var nh = Int(py=params[3])
    var nkv = Int(py=params[4])
    var hd = Int(py=params[5])
    var it = Int(py=params[6])
    var window = Int(py=params[7])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, l, dm, nh, nkv, hd, it, l, 0, window, False)
    return PythonObject(out_len)


def transformer_forward_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One block call from the caller's state. `addrs`: 0 x, 1-9 the nine
    weights (input_layernorm, post_attention_layernorm, q, k, v, o, gate, up,
    down), 10 k_cache, 11 v_cache, 12 y_out. `params`: 0 B, 1 L, 2 d_model,
    3 n_heads, 4 n_kv_heads, 5 head_dim, 6 intermediate, 7 max_tokens,
    8 cached_tokens, 9 window. Returns the post-call cached_tokens."""
    var a = _addrs(addrs, 13, String("transformer_forward"))
    if len(params) != 10:
        raise Error(
            "transformer_forward: params must contain 10 values (B, L,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var nh = Int(py=params[3])
    var nkv = Int(py=params[4])
    var hd = Int(py=params[5])
    var it = Int(py=params[6])
    var smax = Int(py=params[7])
    var s0 = Int(py=params[8])
    var window = Int(py=params[9])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, l, dm, nh, nkv, hd, it, smax, s0, window, True)
    return PythonObject(out_len)


def transformer_decode_step_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One decode token: `transformer_forward` at L = 1. `addrs`: the same
    thirteen. `params`: 0 B, 1 d_model, 2 n_heads, 3 n_kv_heads, 4 head_dim,
    5 intermediate, 6 max_tokens, 7 cached_tokens, 8 window."""
    var a = _addrs(addrs, 13, String("transformer_decode_step"))
    if len(params) != 9:
        raise Error(
            "transformer_decode_step: params must contain 9 values (B,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var nh = Int(py=params[2])
    var nkv = Int(py=params[3])
    var hd = Int(py=params[4])
    var it = Int(py=params[5])
    var smax = Int(py=params[6])
    var s0 = Int(py=params[7])
    var window = Int(py=params[8])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _run_forward(a, b, 1, dm, nh, nkv, hd, it, smax, s0, window, True)
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
