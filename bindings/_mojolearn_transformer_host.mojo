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
from transformer.host.native_block_dispatch import NATIVE_TRANSFORMER_HOST_BLOCK_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

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

# Typed kernel side of this shell (thin-shell move, pure, no bit change).
from transformer.transformer_host_binding_impl import (
    _backward_run,
    _session_forward_tape_run,
    _session_backward_tape_run,
    _write,
    _opt_read,
    _host_weights_from,
    _run_forward,
    HostTransformerSession,
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


# TOMBSTONE: MOJOLEARN_IDN_ATTN_SOFTMAX=2 (slower) deleted 2026-10-10 by c1bf9d832; code recoverable at ca25d9321.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm2.patch


# TOMBSTONE: transformer_attention_profile, transformer_session_info, transformer_decode_session_info and the backward timing entries (dead native entries: no live caller) deleted 2026-10-26 by 79853ceba; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_transformer.patch


def transformer_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if len(addrs) != n:
        raise Error(
            what + ": addrs must contain " + String(n) + " addresses, got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(n):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    return a^


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
    for i in range(got):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if i < n and i != gate_slot and address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    while len(a) < n + BLOCK_OPTION_ADDRS:  # small-loop(BLOCK_OPTION_ADDRS: pads the absent option tail with zero addresses): argument words, not data
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
    for i in range(got):  # small-loop(params: fixed scalar list of this entry): argument words, not data
        p.append(Int(py=params[i]))
    return p^


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
    for i in range(10):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        a.append(a0[i])
    a.append(0)
    a.append(0)
    a.append(a0[10])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(BLOCK_OPTION_ADDRS: the 11 option-tail addresses): argument words, not data
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
        _backward_run(a, b, l, dm, nh, nkv, hd, it, window)
    return PythonObject(0)



from transformer.host.attention_tape import HostAttentionTape
from transformer.experiments.profile import transformer_arithmetic_profile, transformer_arithmetic_profile_changed


def transformer_arithmetic_profile_changed_binding() raises -> PythonObject:
    return PythonObject(Int(transformer_arithmetic_profile_changed()))


def transformer_arithmetic_profile_binding() raises -> PythonObject:
    return PythonObject(transformer_arithmetic_profile())


def transformer_session_create_binding() raises -> PythonObject:
    return PythonObject(alloc=HostTransformerSession())


def transformer_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    if owner[].busy:
        raise Error("transformer tape: session is busy")
    owner[].invalidate_tape()
    owner[].closed = True
    return PythonObject(0)


def transformer_session_discard_tape_binding(session: PythonObject,ticket: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    if owner[].busy:
        raise Error("transformer tape: session is busy")
    if len(ticket) != 2:
        raise Error("transformer tape: owner and epoch required")
    var cookie = Int(py=ticket[0])
    var epoch = Int(py=ticket[1])
    if cookie != Int(owner):
        raise Error("transformer tape: foreign session")
    # Closing an older ticket cannot discard a newer forward in this session.
    if not owner[].closed and epoch == owner[].generation:
        owner[].invalidate_tape()
    return PythonObject(0)


def transformer_session_invalidate_tape_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    owner[].invalidate_tape()
    return PythonObject(0)


def transformer_session_forward_tape_binding(session: PythonObject,addrs: PythonObject,
                                             params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or len(params) != 10:
        raise Error("transformer tape: IDENTICAL and ten scalars required")
    var a = _addrs(addrs,11,String("transformer tape"))
    var p = List[Int]()
    for i in range(10):  # small-loop(params: fixed scalar list of this entry): argument words, not data
        p.append(Int(py=params[i]))
    if p[0] <= 0 or p[1] <= 0 or p[7] < 0 or p[8] < 0 or p[9] < 0:
        raise Error("transformer tape: invalid shape/window/activation budget/cost")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _session_forward_tape_run(Int(owner), a, p)
    except error:
        owner[].invalidate_tape()
        owner[].busy = False
        raise error
    owner[].busy = False
    var ticket=Python.list()
    ticket.append(PythonObject(Int(owner)))
    ticket.append(PythonObject(owner[].generation))
    ticket.append(PythonObject(Int(owner[].tape.value().retained)))
    ticket.append(PythonObject(owner[].tape.value().retained_bytes))
    return ticket


def transformer_session_backward_tape_binding(session: PythonObject,addrs: PythonObject,
                                              ticket: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    if len(ticket) != 2:
        raise Error("transformer tape: owner and generation required")
    var cookie=Int(py=ticket[0])
    var generation=Int(py=ticket[1])
    if cookie != Int(owner) or generation != owner[].generation or not owner[].tape:
        raise Error("transformer tape: stale, foreign or already consumed ticket")
    var a = _addrs(addrs,11,String("transformer tape backward"))
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _session_backward_tape_run(Int(owner), a, generation)
    except error:
        owner[].invalidate_tape()
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_transformer_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).

def transformer_block_api_binding() raises -> PythonObject:
    var api = NATIVE_TRANSFORMER_HOST_BLOCK_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def transformer_native_session_handle_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[HostTransformerSession]()
    return PythonObject(Int(owner))


def build__mojolearn_transformer_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_transformer_host")
        module.def_function[transformer_block_api_binding]("transformer_block_api_v1")
        module.def_function[transformer_native_session_handle_binding]("transformer_native_session_handle_v1")
        _ = module.add_type[HostTransformerSession]("_TransformerSession")
        module.def_function[transformer_arithmetic_profile_changed_binding]("transformer_arithmetic_profile_changed")
        module.def_function[transformer_arithmetic_profile_binding]("transformer_arithmetic_profile")
        module.def_function[transformer_session_create_binding]("transformer_session_create")
        module.def_function[transformer_session_close_binding]("transformer_session_close")
        module.def_function[transformer_session_discard_tape_binding]("transformer_session_discard_tape")
        module.def_function[transformer_session_invalidate_tape_binding]("transformer_session_invalidate_tape")
        module.def_function[transformer_session_forward_tape_binding]("transformer_session_forward_tape")
        module.def_function[transformer_session_backward_tape_binding]("transformer_session_backward_tape")
        module.def_function[transformer_host_numeric_mode_binding]("transformer_host_numeric_mode")
        module.def_function[transformer_host_vendor_binding]("transformer_host_vendor")
        module.def_function[transformer_host_column_binding]("transformer_host_column")
        module.def_function[transformer_host_sabotage_binding]("transformer_host_sabotage")
        module.def_function[transformer_vendor_binding]("transformer_vendor")
        module.def_function[transformer_numeric_mode_binding]("transformer_numeric_mode")
        # TOMBSTONE: transformer_attention_profile, transformer_session_info, transformer_decode_session_info and the backward timing entries (dead native entries: no live caller) deleted 2026-10-26 by 79853ceba; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_transformer.patch
        module.def_function[transformer_forward_binding]("transformer_forward")
        module.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        module.def_function[transformer_decode_step_binding]("transformer_decode_step")
        module.def_function[transformer_backward_binding]("transformer_backward")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_transformer_host: ", e))
