# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the transformer (Llama-shaped decoder) block lane.

A FIFTEENTH extension module, and a separate one on purpose, for the
reason `bindings/_mojolearn_estimators.mojo`'s header gives: an
independently changing binding must not become a merge point. This file
gives the certified transformer block (profile
`mojolearn.identical.transformer.fp32.v1`, forward clause (a) and clause
(d) recorded on THREE columns 2026-08-28, the same card bytes on Apple,
NVIDIA and AMD -- `transformer/README.md`'s status block) its first
Python symbol: before it the lane exported nothing an outside consumer
could import.

THE BINDING CALLS THE CERTIFIED ENTRY POINT AND NOTHING ELSE. Every call
below goes through `llama_decoder_layer_forward`
(`transformer/impl/llama/modeling_llama.mojo`) --
the SAME function `transformer/checks/transformer_check.mojo` gates
(through its planted twin, with the plant OFF). NO arithmetic is
respelled in this file: a binding-side copy of any seam would be a
second spelling of a pinned rounding, which is the drift the whole lane
exists to forbid. The decode entry point is the contract's own
construction -- the block forward at `l = 1` with the KV cache carried
and `pos0 = cached_tokens` (contract section 7.2, ONE SPELLING FOR BOTH
PATHS) -- with no arithmetic of its own.

THE TWO-LIST ABI IS DEVIATION 791'S, ADOPTED, NOT RE-DECIDED. Each entry
point takes exactly TWO Python lists, `addrs` (every NumPy buffer
address, in an exact order written in each docstring and mirrored at the
`_transformer_impl.py` call site) and `params` (the scalars), because
`PythonModuleBuilder.def_function` stops elaborating above roughly nine
arguments (MEASURED 2026-09-01, `bindings/_mojolearn_gp.mojo`'s header)
and `transformer_forward` carries thirteen addresses. Every array whose
address goes into `addrs` MUST be bound to a Python local for the
duration of the call: an address inside a list keeps nothing alive
(`python/mojolearn/_arrays.py`).

DEVIATION 795 -- THE TRANSFORMER SURFACE'S OWN DEPARTURES, IN ONE BLOCK.

  (i) A POINTER-ABI SURFACE WHERE UPSTREAM HAS ONLY TORCH MODULES.
  Upstream's only Python face for this block is `LlamaDecoderLayer`, an
  `nn.Module` over torch tensors, autograd and a `DynamicCache` object.
  This surface is bare float32 buffer addresses plus packed scalars into
  the certified Mojo entry point, torch-free by design (the package's
  standing shape, mamba DEVIATIONS 791/792 applied to this lane): the
  consumer's job is cross-checking bits, and a torch dependency on the
  checking side would put the reference's own arithmetic inside the
  instrument.

  (ii) THE STATE IS THE KV CACHE, CROSSING AS TWO CALLER-OWNED
  FULL-CAPACITY BUFFERS PACKED AT THE USED STRIDE, PLUS ONE INTEGER.
  DEVIATION 792's caller-owned-buffers rule extends here with a twist
  none of the mamba states has: `LlamaKVCache` is PACKED at stride `s`
  (the USED length), not at stride `s_max` (DEVIATION 1022 -- the stage
  card records `[B, n_kv, S, head_dim]` and the packed used region IS
  that array), so the caller's `k_cache`/`v_cache` buffers are FLAT
  capacity buffers of `B * n_kv * max_tokens * head_dim` floats whose
  first `B * n_kv * cached_tokens * head_dim` elements are the cache at
  stride `cached_tokens`. Both are READ at entry and WRITTEN BACK whole
  before return, so the caller's bytes round-trip exactly; zeros in,
  with `cached_tokens = 0`, is a fresh sequence. `cached_tokens` is the
  one integer piece of state and crosses as a params scalar IN and as
  the return value OUT (mamba2 `buf_len`'s shape). `max_tokens` (the
  capacity, `s_max`) is a params scalar too, because the packing stride
  and every refusal about growth are functions of it.

  (iii) THE ROTARY TABLE AND THE ATTENTION SCALE USE FROZEN CONSTANTS,
  NEVER PARAMETERS. rms eps (1e-6, bits
  0x358637BD) and rope theta (10000.0, bits 0x461C4000) are contract
  section 3 FROZEN constants, imported from
  `transformer/checks/transformer_fixture.mojo` (the fixture is their
  bit authority); changing either is a v2 profile, not a knob. The
  rotary table is `LlamaRopeTable(ctx, dims, ROPE_THETA, max_tokens)`,
  computed on-device exactly as the lane's own check driver builds it.
  Legacy entry points rebuild it per call. The Python-owned session reuses
  it only for an exactly matching workspace configuration; S6-S8 are pure
  functions of (theta, head_dim, position). The ownership and refresh rules
  are under OWNERSHIP AND REFRESH below.

  (iv) WHAT IS REFUSED HERE, AND WHAT GOES DOWN UNJUDGED (DEVIATION
  793's split, applied). Refused HERE: a null address, an
  `addrs`/`params` list of the wrong length, and a `cached_tokens`
  outside `[0, max_tokens]` -- each means the two sides of THIS boundary
  disagree and no lane refusal exists for it. Everything else goes down
  UNJUDGED so the lane's own refusals stay reachable from Python: the
  five divisibility rules are `LlamaDims.validate`'s (refused by name in
  Mojo), capacity growth past `max_tokens` and the 8192 absolute-
  position ceiling (DEVIATION 812) are `llama_decoder_layer_forward`'s
  and `LlamaKVCache`'s, and non-finite inputs are refused BY NAME on the
  device path (`llama_refuse_bad_call` for x/rope/cache per call;
  weights once at upload, DEVIATION 1875). The dtype refusals (float32
  ONLY) live in `_transformer_impl.py`, because dtype is a NumPy
  property this side cannot see.

  (v) BLAST RADIUS. This file, `python/mojolearn/_transformer_impl.py`,
  `python/mojolearn/transformer.py` and
  `python/mojolearn/tests/test_transformer_surface.py`. The lane's
  kernels, oracle, gates and contract are untouched. Since 2026-09-09
  the BACKWARD (`autograd/kernels/transformer.mojo`) is on
  this surface too, as `transformer_backward`, IDENTICAL tier only, and
  `window` (sliding-window causal attention with a ring KV cache) is
  the tenth forward scalar.

THE TIER SEMANTICS ARE THE BUILD'S, NOT THIS FILE'S. The numeric
mode (fast / identical) is a compile-time define
(`checks/numerics.mojo`), one binary per tier under
`python/mojolearn/[<tier>/]_mojolearn_transformer.so`;
`transformer_numeric_mode` below is the read-back
`_transformer_impl.py` cross-checks so a wrong-arm measurement cannot be
correctly labelled by accident.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.

RUN LEDGER. First compiled 2026-09-02 on the Apple M4 (all three
tiers); the window and backward entry points first compiled and ran
on an NVIDIA L40S 2026-09-09 (identical tier), Apple RUN OWED. Build:
`bash bindings/build_transformer.sh` (per tier via
MOJOLEARN_NUMERIC_MODE); gate:
`cd python && python3 -m mojolearn.tests.test_transformer_surface`.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from core.hostptr import f32_ptr, read_f32, copy_f32, i8_ptr, i32_ptr
# lane/lowbit-blocks (2026-09-29): the block under numeric_profile="fixed15_v1".
from transformer.impl.llama.int15_block import (
    Int15Planes,
    LlamaInt15Stage,
    LlamaInt15Weights,
    llama_int15_proj,
)
from gemm.checks.gemm_int15 import (
    INT15_PIECE_SABOTAGE,
    INT15_QUANT_SABOTAGE,
    INT15_SABOTAGE,
)
from gemm.checks.gemm_int15_epilogue import INT15_EPILOGUE_SABOTAGE
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast, memcpy
from std.os import getenv
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN

from core.identity_trace import IdentityTrace
from core.device_scan import device_first_nonfinite
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.neural_identical_experiments import IDN_SAMBA_FORWARD_TAPE
from core.neural_context import neural_ctx
from checks.vendor import COMPILED_VENDOR
from gemm.checks.gemm_backward import ANY_BWD_SABOTAGE as GEMM_ANY_BWD_SABOTAGE
from gemm.checks.gemm_identical import ANY_SABOTAGE as GEMM_ANY_SABOTAGE

# The two FROZEN arithmetic constants, from their bit authority
# (`transformer_fixture.mojo` cross-checks both against their hex bits in
# `profile_constants_are_intact`). The binding importing from
# `transformer/checks/` is the mamba binding's own precedent (it imports
# `mamba_fixture`'s constants); it is the IMPL file that may not
# (`modeling_llama.mojo`'s "no fixture type crosses this boundary").
from transformer.checks.transformer_fixture import RMS_EPS, ROPE_THETA

# `_upload`/`_download` by their underscore names is DEVIATION 1112's
# settled pattern (`transformer_check.mojo` imports the same pair).
from transformer.impl.llama.fused_attention import (
    fused_forward_supported_head_dim,
)
from transformer.impl.llama.modeling_llama import (
    ATTN_PATH_EAGER,
    BLOCK_ANY_SABOTAGE,
    PLANT_AT_NONE,
    attention_path_choice,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _download,
    _upload,
    llama_decoder_layer_forward,
)
from autograd.kernels.transformer import (
    BWD_ANY_SABOTAGE,
    LlamaBackwardStages,
)
# The session and saved-tape backwards below run on the autograd tape.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_TRANSFORMER(_OFF) (PROMOTED, hand arm deleted) deleted 2026-10-19 by dd4194a23; code recoverable at dd4194a23^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch
from autograd.rules_transformer import llama_decoder_layer_backward, llama_decoder_layer_backward_autograd
# lane ag-tm-timing: the launch profile the backward timing entries read.
from sequence.launch_profile import launch_profile_level, launch_profile_reset, launch_profile_snapshot
# lane/block-options (2026-09-17): the block options record and the two
# tails. `transformer/block_options.mojo` is the order's authority; the
# docstrings below repeat it word for word because a binding docstring is
# what the Python side is written against.
from transformer.block_options import (
    BLOCK_OPTION_ADDRS,
    BLOCK_OPTION_PARAMS,
    BlockOptions,
)
from transformer.impl.llama.modeling_llama import _zeros as _llama_zeros
# lane/py-lm (2026-09-28): the resident CausalLM decode session runs the
# stack's three non-block steps on the kernels `training/samba_ops.mojo`
# calls for them (embedding gather, llama RMSNorm, the identical GEMM at
# OP_NT), so its bits are the per-call route's.
from embedding.checks.embedding_identical import identical_embedding_forward_into
from embedding.checks.embedding_oracle import EmbConfig
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.contract import OP_NT
from transformer.impl.llama.modeling_llama import llama_rms_norm
from std.math import isfinite
# lane afn-attn (2026-10-03): MOJOLEARN_AFN_ATTN_ARENA, FAST + Apple only
# (AFN_ATTN_ARENA is False on every other build, so these are never called).
from core.device_arena import arena_begin, arena_end, arena_release
from transformer.impl.llama.afn_apple_fast import AFN_ATTN_ARENA, afn_flash_call_count

from transformer.experiments.profile import transformer_arithmetic_profile, transformer_arithmetic_profile_changed

# Typed kernel side of this shell (thin-shell move, pure, no bit change).
from transformer.transformer_binding_impl import (
    _session_forward_tape_run,
    _session_backward_tape_run,
    _NEURAL_CTX,
    _optional_upload,
    _f32_ptr,
    _read_f32,
    _write_f32,
    _upload_addr,
    _download_addr,
    _btick,
    transformer_lean_stages,
    _load_transformer_weights,
    _tail_of,
    _check_base_addrs,
    _lean_for,
    _lean_for_w,
    _workspace_key,
    TransformerWorkspace,
    TransformerBackwardWorkspace,
    TransformerSession,
    _weights_key,
    _base_weight_lens,
    _weights_same_bits,
    _weights_host_copy,
    _weights_recopy,
    _stage_reset_on,
    _session_retain_cap_bytes,
    _session_weights,
    _transformer_run_session,
    TransformerForwardTape,
    _transformer_saved_backward,
    _transformer_run_session_backward,
    _tm_profile_launches,
    _transformer_session_backward_time,
    _transformer_saved_backward_timed,
    INT15_ENTRY_ADDRS,
    INT15_ENTRY_PARAMS,
    _upload_i8,
    _upload_i32,
    _upload_planes_one,
    _upload_planes,
    _transformer_run_session_int15,
    _transformer_run,
    TransformerDecodeSession,
    _require_session_open,
    _session_open_run,
    _session_open_run_int15,
    _session_step_run,
    _session_forward_run,
    _session_export_run,
    _session_load_run,
    CausalLMSession,
    _lm_refuse_nonfinite_device,
    _LM_ARGMAX_TPB,
    _LM_ARGMAX_NONE,
    _lm_order_key,
    _lm_last_argmax_kernel,
    _lm_open_run,
    _lm_open_run_int15,
    _lm_layers_run,
    _lm_run,
    _lm_forward_run,
    _transformer_backward_run,
)


# ===========================================================================
# THE TWO OPTION TAILS (lane/block-options, 2026-09-17).
#
# Every forward entry point below accepts its OLD address and params lists
# (every option at its default, the code path that existed before this
# lane) OR the old lists with these tails appended, and refuses any other
# length by name. The Python side sends the tails only for a non-default
# record, so a default block reaches the old lengths exactly.
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
# With an ungated MLP (mlp_kind 1 or 2) the base list's gate_proj.weight
# slot carries 0. A present tensor whose flag is off, an absent one whose
# flag is on, and every unsupported combination are refused BY NAME in
# `LlamaDeviceWeights`' options constructor and `BlockOptions.validate`.
# ===========================================================================


def _read_params(params: PythonObject, base: Int, what: String) raises -> List[Int]:
    """The entry's scalars, `base` of them or `base + BLOCK_OPTION_PARAMS`."""
    var n = len(params)
    if n != base and n != base + BLOCK_OPTION_PARAMS:
        raise Error(
            what
            + ": params must contain "
            + String(base)
            + " values, or "
            + String(base)
            + " + "
            + String(BLOCK_OPTION_PARAMS)
            + " with the block options tail, got "
            + String(n)
        )
    var p = List[Int]()
    for i in range(n):  # small-loop(n: block scalar params, base plus the 17 option tail): validated count of launch parameters, not data
        p.append(Int(py=params[i]))
    return p^


def _read_addrs_tail(
    addrs: PythonObject, base: Int, what: String
) raises -> List[Int]:
    """The entry's addresses, `base` or `base + BLOCK_OPTION_ADDRS` of them,
    returned as exactly `base + BLOCK_OPTION_ADDRS` (zeros where the tail
    was not sent). Null checks are the CALLER's, per slot, because the
    gate slot may legitimately be 0 under an ungated MLP."""
    var n = len(addrs)
    if n != base and n != base + BLOCK_OPTION_ADDRS:
        raise Error(
            what
            + ": addrs must contain "
            + String(base)
            + " addresses, or "
            + String(base)
            + " + "
            + String(BLOCK_OPTION_ADDRS)
            + " with the block options tail, got "
            + String(n)
        )
    var a = List[Int]()
    for i in range(n):  # small-loop(n: block buffer addresses, base plus the 11 option tail): validated count of tensor addresses, not data
        a.append(Int(py=addrs[i]))
    while len(a) < base + BLOCK_OPTION_ADDRS:  # small-loop(BLOCK_OPTION_ADDRS: pads the absent option tail with zero addresses): argument words, not data
        a.append(0)
    return a^


def transformer_flash_call_count_binding(grouped: PythonObject) raises -> PythonObject:
    return PythonObject(afn_flash_call_count(Bool(py=grouped)))


# TOMBSTONE: MOJOLEARN_IDN_ATTN_SOFTMAX=2 (slower) deleted 2026-10-10 by c1bf9d832; code recoverable at ca25d9321.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm2.patch


def transformer_attention_profile_binding() raises -> PythonObject:
    """Exact attention graph for this native binary's ordinary fp32 model path."""
    return PythonObject(String("attention-eager.fp32.v1"))


def transformer_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL. The same shape as `mamba_numeric_mode`
    and for the same reason: `_transformer_impl.py` reads it once and
    refuses to run if the binary it loaded disagrees with the mode the
    package asked for."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def transformer_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none' -- the accelerator API this
    binary was compiled for, folded in from `checks/vendor.mojo`. The
    answer comes from the binary that actually loaded, never from the
    directory it sat in."""
    return PythonObject(String(COMPILED_VENDOR))


def transformer_forward_tape_enabled_binding() -> PythonObject:
    return PythonObject(IDN_SAMBA_FORWARD_TAPE)


def transformer_forward_tape_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """[x, nine weights, y], [B,L,dm,nh,nkv,hd,it,window] -> owned tape."""
    comptime if not IDN_SAMBA_FORWARD_TAPE:
        raise Error("transformer forward tape: IDENTICAL opt-in is disabled")
    var incoming = _read_addrs_tail(addrs, 11, String("transformer_forward_tape"))
    # _read_addrs_tail always returns 11 + BLOCK_OPTION_ADDRS slots (zeros
    # where no options tail was sent), so the old `len(incoming) != 11` test
    # refused every call. Default options means the tail slots are all zero.
    for i in range(11, len(incoming)):  # small-loop(incoming: the block options tail slots): validated tensor addresses, not data
        if incoming[i] != 0:
            raise Error("transformer forward tape: default options only (no block options tail)")
    var p = _read_params(params, 8, String("transformer_forward_tape"))
    if len(p) != 8:
        raise Error("transformer forward tape: expected 8 shape scalars")
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        if incoming[i] == 0:
            raise Error("transformer forward tape: null address at slot " + String(i))
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        a.append(incoming[i])
    a.append(0)
    a.append(0)
    a.append(incoming[10])
    var tape = TransformerForwardTape(p)
    with GILReleased(Python()):
        try:
            _ = _transformer_run_session[True, True](tape.session, a,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], BlockOptions())
        except error:
            tape.close()
            raise error
    return PythonObject(alloc=tape^)


def transformer_backward_tape_binding(tape: PythonObject, addrs: PythonObject) raises -> PythonObject:
    comptime if not IDN_SAMBA_FORWARD_TAPE:
        raise Error("transformer backward tape: IDENTICAL opt-in is disabled")
    var owner = tape.downcast_value_ptr[TransformerForwardTape]()
    if owner[].busy or owner[].consumed:
        raise Error("transformer forward tape: busy, closed or already consumed")
    var a = _read_addrs_tail(addrs, 11, String("transformer_backward_tape"))
    # _read_addrs_tail pads to 11 + BLOCK_OPTION_ADDRS slots, so the old
    # `len(a) != 11` test refused every call (the forward tape entry had the
    # same fault). The backward takes no options: the tail must be zero.
    for i in range(11, len(a)):  # small-loop(a: the block options tail slots): validated tensor addresses, not data
        if a[i] != 0:
            raise Error("transformer backward tape: expected dy and ten outputs, no block options tail")
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        if a[i] == 0:
            raise Error("transformer backward tape: null address at slot " + String(i))
    owner[].busy = True
    owner[].consumed = True
    try:
        with GILReleased(Python()):
            try:
                _transformer_saved_backward(owner[], a)
            except error:
                owner[].close()
                raise error
            owner[].close()
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(0)


def transformer_close_tape_binding(tape: PythonObject) raises -> PythonObject:
    var owner = tape.downcast_value_ptr[TransformerForwardTape]()
    if owner[].busy:
        raise Error("transformer forward tape: busy")
    with GILReleased(Python()):
        owner[].close()
    return PythonObject(0)

def transformer_session_forward_fresh_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`transformer_session_forward` for a STATELESS prefill (lane/neural-net-
    experiment, 2026-09-30): the same 13 (+11) addresses with the cache slots
    10 and 11 ZERO, the same 10 (+17) params with `smax = L` and
    `cached_tokens = 0`. The cache is the session's retained buffers,
    zero-filled on the device; nothing of it crosses the host boundary. This
    is `transformer_forward_fresh` on a session: weights retained across
    calls (`_session_weights`), workspace retained under the budget."""
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer: session is busy or closed")
    var a = _read_addrs_tail(addrs, 13, String("transformer_session_forward_fresh"))
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9, 12], String("transformer_session_forward_fresh"))
    _ = a  # keep the Python object alive through the native call above
    var p = _read_params(params, 10, String("transformer_session_forward_fresh"))
    var opts = BlockOptions.from_params(p, 10)
    owner[].busy = True
    var result = 0
    try:
        with GILReleased(Python()):
            try:
                result = _transformer_run_session[True](owner[], a,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(result)


def transformer_session_backward_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`transformer_backward` on a session (lane/neural-net-experiment):
    the same twenty-one addresses and eight scalars, word for word, the
    weights, workspace and backward stages retained across calls."""
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer: session is busy or closed")
    if len(addrs) != 21 or len(params) != 8:
        raise Error(
            "transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " window)"
        )
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(
                "transformer backward: null buffer address at slot "
                + String(i)
            )
        a.append(address)
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var nh = Int(py=params[3])
    var nkv = Int(py=params[4])
    var hd = Int(py=params[5])
    var it = Int(py=params[6])
    var window = Int(py=params[7])
    owner[].busy = True
    try:
        with GILReleased(Python()):
            try:
                _transformer_run_session_backward(owner[], a, b, l, dm, nh, nkv, hd, it, window)
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(0)


def transformer_session_backward_time_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`transformer_session_backward`'s 21 addresses; its 8 scalars plus the
    number of timed calls N. Returns [autograd, launches, then forward and
    backward nanoseconds for N + 1 calls, the first a warm-up]."""
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer: session is busy or closed")
    if len(addrs) != 21 or len(params) != 9:
        raise Error("transformer backward time: expected 21 addresses and 9 scalars")
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("transformer backward time: null buffer address at slot " + String(i))
        a.append(address)
    var p = List[Int]()
    for i in range(9):  # small-loop(params: fixed scalar list of this entry): argument words, not data
        p.append(Int(py=params[i]))
    owner[].busy = True
    var stamps = List[Int]()
    try:
        with GILReleased(Python()):
            try:
                stamps = _transformer_session_backward_time(owner[], a,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    var out = Python.list()
    for i in range(len(stamps)):  # small-loop(stamps: one clock reading per timed call): measurement timestamps, not model data
        out.append(PythonObject(stamps[i]))
    return out


def transformer_backward_tape_time_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """22 addresses: `transformer_forward_tape`'s 11 (x, nine weights, y),
    then `transformer_backward_tape`'s 11 (dy, ten gradients); its 8 shape
    scalars plus N. Runs N + 1 saved forwards (untimed) each followed by a
    timed saved backward. Returns [autograd, launches, N + 1 backward
    nanoseconds, the first a warm-up]."""
    comptime if not IDN_SAMBA_FORWARD_TAPE:
        raise Error("transformer backward tape time: IDENTICAL opt-in is disabled")
    if len(addrs) != 22 or len(params) != 9:
        raise Error("transformer backward tape time: expected 22 addresses and 9 scalars")
    var fa = List[Int]()
    var ba = List[Int]()
    for i in range(22):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("transformer backward tape time: null address at slot " + String(i))
        if i < 10:
            fa.append(address)
        elif i == 10:
            fa.append(0)
            fa.append(0)
            fa.append(address)
        else:
            ba.append(address)
    var p = List[Int]()
    for i in range(8):  # small-loop(params: fixed scalar list of this entry): argument words, not data
        p.append(Int(py=params[i]))
    var calls = Int(py=params[8])
    if calls < 1:
        raise Error("transformer backward tape time: the call count must be >= 1")
    var out = Python.list()
    # Slot 0 is always 1 (tape); see _transformer_session_backward_time.
    out.append(PythonObject(1))
    var stamps = List[Int]()
    var launches = -1
    with GILReleased(Python()):
        var prof = launch_profile_level() > 0
        for k in range(calls + 1):  # driver-loop: over timed calls
            var tape = TransformerForwardTape(p)
            try:
                _ = _transformer_run_session[True, True](tape.session, fa,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], BlockOptions())
                if prof and k == calls:
                    launch_profile_reset()
                stamps.append(_transformer_saved_backward_timed(tape, ba))
                if prof and k == calls:
                    launches = _tm_profile_launches()
            except error:
                tape.close()
                raise error
            tape.close()
    out.append(PythonObject(launches))
    for i in range(len(stamps)):  # small-loop(stamps: one clock reading per timed call): measurement timestamps, not model data
        out.append(PythonObject(stamps[i]))
    return out


def transformer_session_create_binding() raises -> PythonObject:
    return PythonObject(alloc=TransformerSession())


def transformer_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy:
        raise Error("transformer: session is busy")
    owner[].closed = True
    owner[].clear()
    owner[].ctx = None
    return PythonObject(0)


def transformer_session_info_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy:
        raise Error("transformer: session is busy")
    var out = Python.list()
    out.append(PythonObject(owner[].contexts))
    out.append(PythonObject(owner[].workspaces))
    out.append(PythonObject(owner[].closed))
    var retained_bytes = 0
    if owner[].workspace:
        retained_bytes = owner[].workspace.value().retained_bytes()
    out.append(PythonObject(retained_bytes))
    return out


def transformer_session_forward_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer: session is busy or closed")
    # `transformer_forward`'s lists, with or without the two option tails
    # (the header of this section); the same contract word for word.
    var a = _read_addrs_tail(addrs, 13, String("transformer_forward"))
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], String("transformer_forward"))
    _ = a  # keep the Python object alive through the native call above
    var p = _read_params(params, 10, String("transformer_forward"))
    var opts = BlockOptions.from_params(p, 10)
    owner[].busy = True
    var result = 0
    try:
        with GILReleased(Python()):
            try:
                result = _transformer_run_session(owner[], a,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
            except error:
                # A failed call must not leave partially written scratch or
                # queued references for the next caller. Caller arrays are
                # still owned by its Python frame during this drain.
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(result)


def transformer_session_forward_int15_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer: session is busy or closed")
    var what = String("transformer_session_forward_int15")
    if len(addrs) != INT15_ENTRY_ADDRS:
        raise Error(what + ": addrs must contain " + String(INT15_ENTRY_ADDRS) + " addresses, got " + String(len(addrs)))
    if len(params) != INT15_ENTRY_PARAMS:
        raise Error(what + ": params must contain " + String(INT15_ENTRY_PARAMS) + " values, got " + String(len(params)))
    var a = List[Int]()
    for i in range(len(addrs)):  # small-loop(addrs: INT15_ENTRY_ADDRS validated tensor addresses): reads pointer list, not data
        a.append(Int(py=addrs[i]))
    _check_base_addrs(a, [0, 1, 2, 10, 11, 12], what)
    _ = a  # keep the Python object alive through the native call above
    var p = _read_params(params, 10, what)
    var opts = BlockOptions.from_params(p, 10)
    owner[].busy = True
    var result = 0
    try:
        with GILReleased(Python()):
            try:
                result = _transformer_run_session_int15(owner[], a,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(result)


def _transformer_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    """The thirteen base addresses, or the thirteen plus the eleven-address
    options tail (this file's tail section), returned as twenty-four with
    zeros where the tail was not sent."""
    if len(addrs) != 13 and len(addrs) != 13 + BLOCK_OPTION_ADDRS:
        raise Error(
            what
            + ": addrs must contain 13 addresses (x,"
            " input_layernorm.weight, post_attention_layernorm.weight,"
            " q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight,"
            " gate_proj.weight, up_proj.weight, down_proj.weight, k_cache,"
            " v_cache, y_out), optionally followed by the 11-address block"
            " options tail, got "
            + String(len(addrs))
        )
    return _read_addrs_tail(addrs, 13, what)


def transformer_forward_fresh_binding(
    addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Stateless prefill: original zero device cache, only y is returned.

    Eleven pointers: x, nine weights, y. Eight scalars: B, L, d_model,
    n_heads, n_kv_heads, head_dim, intermediate, window. Capacity remains
    L (or a window-sized ring), exactly as Python allocate_state(B, L).
    The zero sentinel cache pointers never reach a transfer in this arm.

    lane/block-options: or 11 + 11 pointers and 8 + 17 scalars, the two
    option tails appended (this file's tail section, word for word). The
    gate_proj.weight pointer (slot 7) is 0 under an ungated MLP.
    """
    var what = String("transformer_forward_fresh")
    var raw = _read_addrs_tail(addrs, 11, what)
    var p = _read_params(params, 8, what)
    var opts = BlockOptions.from_params(p, 8)
    # Into `transformer_forward`'s 13 + 11 layout: x, nine weights, the
    # two absent cache slots, y, then the tail.
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        a.append(raw[i])
    a.append(0)
    a.append(0)
    a.append(raw[10])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(BLOCK_OPTION_ADDRS: the 11 option-tail addresses): argument words, not data
        a.append(raw[11 + i])
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9], what)
    _ = a  # keep the Python object alive through the native call above
    if a[12] == 0:
        raise Error("transformer_forward_fresh: null output address")
    var b = p[0]
    var l = p[1]
    var dm = p[2]
    var nh = p[3]
    var nkv = p[4]
    var hd = p[5]
    var it = p[6]
    var window = p[7]
    if b <= 0 or l <= 0:
        raise Error("transformer_forward_fresh: B and L must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _transformer_run[True](a, b, l, dm, nh, nkv, hd, it, l, 0, window, opts)
    return PythonObject(out_len)


def transformer_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One transformer block call: prefill from the state the caller
    hands in -- fresh (zero caches, cached_tokens 0) or resumed (a
    carried cache: chunked-prefill continuation) -- any B and L that fit
    the capacity. Returns the post-call `cached_tokens`.

    `addrs` is the THIRTEEN buffer addresses, in this exact order
    (mirrored in `python/mojolearn/_transformer_impl.py`; QW =
    n_heads*head_dim = d_model, KW = n_kv_heads*head_dim, IT =
    intermediate_size, SMAX = max_tokens):

        0  x                                B * L * d_model float32, read
        1  input_layernorm.weight           d_model, read
        2  post_attention_layernorm.weight  d_model, read
        3  q_proj.weight                    QW * d_model, read (torch
                                             [out, in]; OP_NT reads it)
        4  k_proj.weight                    KW * d_model, read
        5  v_proj.weight                    KW * d_model, read
        6  o_proj.weight                    d_model * QW, read
        7  gate_proj.weight                 IT * d_model, read
        8  up_proj.weight                   IT * d_model, read
        9  down_proj.weight                 d_model * IT, read
       10  k_cache                          B * n_kv * SMAX * head_dim,
                                             READ AND WRITTEN (state;
                                             PACKED at stride
                                             cached_tokens, DEVIATION
                                             795(ii))
       11  v_cache                          B * n_kv * SMAX * head_dim,
                                             READ AND WRITTEN (state)
       12  y_out                            B * L * d_model, WRITTEN (the
                                             block output, both residual
                                             adds included -- contract
                                             section 2)

    `params` is, in this exact order:

        0  B
        1  L
        2  d_model        must equal n_heads*head_dim; refused otherwise
                           BY NAME in Mojo (LlamaDims.validate)
        3  n_heads
        4  n_kv_heads     n_heads % n_kv_heads must be 0 (GQA admitted,
                           contract DEVIATION 813)
        5  head_dim       must be even (RoPE pairs halves)
        6  intermediate   intermediate_size, > 0
        7  max_tokens     the cache capacity s_max; cached_tokens + L
                           past it, and any value over 8192 (DEVIATION
                           812's ceiling), are refused by name in Mojo
        8  cached_tokens  the carried cache's used length; 0 for a
                           fresh sequence
        9  window         0 for full causal attention; W > 0 for sliding-
                           window causal attention (query p sees keys
                           [max(0, p-W+1), p]) with k_cache/v_cache a
                           RING of B * n_kv * W * head_dim floats, slot
                           = position % W

    Profile constants (rms eps 1e-6, rope theta 10000.0, eager attention,
    the causal mask value, no biases, silu -- contract section 3) are NOT
    parameters: changing one is a v2 profile, not a knob. There is no
    bias address and no dropout: the profile REFUSES both by absence."""
    var a = _transformer_addrs(addrs, String("transformer_forward"))
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], String("transformer_forward"))
    _ = a  # keep the Python object alive through the native call above
    if len(params) != 10 and len(params) != 10 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_forward: params must contain 10 values (B, L,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail, got "
            + String(len(params))
        )
    var p = _read_params(params, 10, String("transformer_forward"))
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
        out_len = _transformer_run(
            a, b, l, dm, nh, nkv, hd, it, smax, s0, window, opts
        )
    return PythonObject(out_len)


def transformer_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: the profile's spelling -- the block forward at
    L = 1 with the KV cache carried and `pos0 = cached_tokens` (contract
    section 7.2: ONE SPELLING FOR BOTH PATHS is what makes clause (d),
    decode == prefill bitwise, a theorem the gate verifies rather than a
    coincidence it hopes for), with NO arithmetic of its own. There is
    deliberately no second entry path to drift.

    `addrs`: the same THIRTEEN as `transformer_forward` with L = 1
    shapes (x and y_out are B * d_model). `params`: 0 B, 1 d_model,
    2 n_heads, 3 n_kv_heads, 4 head_dim, 5 intermediate, 6 max_tokens,
    7 cached_tokens, 8 window. Returns the post-call cached_tokens."""
    var a = _transformer_addrs(addrs, String("transformer_decode_step"))
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], String("transformer_decode_step"))
    _ = a  # keep the Python object alive through the native call above
    if len(params) != 9 and len(params) != 9 + BLOCK_OPTION_PARAMS:
        raise Error(
            "transformer_decode_step: params must contain 9 values (B,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens, window), optionally followed by"
            " the 17-value block options tail, got "
            + String(len(params))
        )
    var p = _read_params(params, 9, String("transformer_decode_step"))
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
        out_len = _transformer_run(
            a, b, 1, dm, nh, nkv, hd, it, smax, s0, window, opts
        )
    return PythonObject(out_len)


def transformer_decode_session_create_binding() raises -> PythonObject:
    """A closed session. Creation performs no GPU operation."""
    return PythonObject(alloc=TransformerDecodeSession())


def transformer_decode_session_open_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Open a closed session on a block and a carried cache. `addrs` is
    ELEVEN addresses: the nine weights in `transformer_forward`'s order
    (slots 1 to 9 there), then k_cache and v_cache. `params` is NINE
    scalars: B, d_model, n_heads, n_kv_heads, head_dim, intermediate,
    max_tokens, cached_tokens, window. Everything is COPIED to the device;
    later edits to the caller's arrays are not observed. Returns
    cached_tokens. Refused on an open session."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    var what = String("transformer_decode_session_open")
    if (len(addrs) != 11 and len(addrs) != 11 + BLOCK_OPTION_ADDRS) or (
        len(params) != 9 and len(params) != 9 + BLOCK_OPTION_PARAMS
    ):
        raise Error(
            "transformer_decode_session_open: expected 11 addresses and 9"
            " scalars, each optionally followed by its block options tail"
        )
    if owner[].busy:
        raise Error("transformer decode session: busy")
    if owner[].is_open():
        raise Error("transformer decode session: already open; close it first")
    # lane/block-options: into `transformer_forward`'s 13 + 11 layout;
    # slot 0 (x) and slot 12 (y) are not part of an open.
    var raw = _read_addrs_tail(addrs, 11, what)
    var a = List[Int]()
    a.append(0)
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        a.append(raw[i])
    a.append(0)
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(BLOCK_OPTION_ADDRS: the 11 option-tail addresses): argument words, not data
        a.append(raw[11 + i])
    _check_base_addrs(a, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11], what)
    _ = a  # keep the Python object alive through the native call above
    var p = _read_params(params, 9, what)
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
    if b <= 0:
        raise Error("transformer_decode_session_open: B must be positive")
    owner[].busy = True
    owner[].usable = False
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_open_run(owner[], a, b, dm, nh, nkv, hd, it, smax, s0, window, opts)
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(out_len)


def transformer_decode_session_open_int15_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`transformer_decode_session_open` under numeric_profile="fixed15_v1"
    (lane/lowbit-default, 2026-09-29). `addrs` is EXACTLY 11 + the 11-address
    options tail + 21: the open's eleven with the seven projection slots
    ZERO (never read), the options tail (always sent), then the planes,
    three per projection in the order q, k, v, o, gate, up, down (`hi`,
    `lo`, `exponents`; zeros for gate under an ungated MLP). `params` is
    EXACTLY the open's nine scalars and the 17-int options tail. The planes
    are COPIED to the device once and stay there; every other entry of the
    session (step, forward, export, load, close, `causal_lm_session_run`)
    is the float32 session's own."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    var what = String("transformer_decode_session_open_int15")
    if len(addrs) != 11 + BLOCK_OPTION_ADDRS + 21 or len(params) != 9 + BLOCK_OPTION_PARAMS:
        raise Error(
            what + ": expected " + String(11 + BLOCK_OPTION_ADDRS + 21) + " addresses and "
            + String(9 + BLOCK_OPTION_PARAMS) + " scalars, got " + String(len(addrs)) + " and "
            + String(len(params))
        )
    if owner[].busy:
        raise Error("transformer decode session: busy")
    if owner[].is_open():
        raise Error("transformer decode session: already open; close it first")
    var raw = List[Int]()
    for i in range(len(addrs)):  # small-loop(addrs: validated decode tensor addresses, 11 plus 11 plus 21): reads pointer list, not data
        raw.append(Int(py=addrs[i]))
    # into `transformer_forward`'s 13 + 11 layout, then the 21 planes
    var a = List[Int]()
    a.append(0)
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        a.append(raw[i])
    a.append(0)
    for i in range(BLOCK_OPTION_ADDRS + 21):  # small-loop(BLOCK_OPTION_ADDRS: 11 option addresses plus 21 planes): reorders pointer list, not data
        a.append(raw[11 + i])
    _check_base_addrs(a, [1, 2, 10, 11], what)
    _ = a  # keep the Python object alive through the native call above
    var p = _read_params(params, 9, what)
    var opts = BlockOptions.from_params(p, 9)
    if p[0] <= 0:
        raise Error(what + ": B must be positive")
    owner[].busy = True
    owner[].usable = False
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_open_run_int15(
                owner[], a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], opts)
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(out_len)


def transformer_decode_session_step_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One decode token on an open session: `addrs` = [x, y_out], both
    B * d_model float32; `params` = [cached_tokens], the caller's belief of
    the resident position, refused on a mismatch so a step can never run
    against a state the caller does not think it holds. Returns the
    post-call cached_tokens. A failed call marks the session lost."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    if len(addrs) != 2 or len(params) != 1:
        raise Error("transformer_decode_session_step: expected 2 addresses and 1 scalar")
    var px = Int(py=addrs[0])
    var py = Int(py=addrs[1])
    if px == 0 or py == 0:
        raise Error("transformer_decode_session_step: null buffer address")
    var claimed = Int(py=params[0])
    _require_session_open(owner[])
    if owner[].kv.value().s != claimed:
        raise Error(
            String("transformer decode session: the caller believes cached_tokens = ") + String(claimed)
            + " but the resident cache holds " + String(owner[].kv.value().s))
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_step_run(owner[], px, py)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def transformer_decode_session_forward_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`L` tokens per row on an open session (a prefill or a chunked
    continuation of the resident cache): `addrs` = [x, y_out], both
    B * L * d_model float32; `params` = [L, cached_tokens]. Returns the
    post-call cached_tokens."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    if len(addrs) != 2 or len(params) != 2:
        raise Error("transformer_decode_session_forward: expected 2 addresses and 2 scalars")
    var px = Int(py=addrs[0])
    var py = Int(py=addrs[1])
    if px == 0 or py == 0:
        raise Error("transformer_decode_session_forward: null buffer address")
    var l = Int(py=params[0])
    var claimed = Int(py=params[1])
    if l <= 0:
        raise Error("transformer_decode_session_forward: L must be positive")
    _require_session_open(owner[])
    if owner[].kv.value().s != claimed:
        raise Error(
            String("transformer decode session: the caller believes cached_tokens = ") + String(claimed)
            + " but the resident cache holds " + String(owner[].kv.value().s))
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_forward_run(owner[], px, py, l)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def transformer_decode_session_export_state_binding(
    session: PythonObject, addrs: PythonObject,
) raises -> PythonObject:
    """Copy the resident cache into the caller's buffers: `addrs` =
    [k_cache, v_cache], each the full capacity buffer. Returns
    cached_tokens. The session stays open."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    if len(addrs) != 2:
        raise Error("transformer_decode_session_export_state: expected 2 addresses")
    var pk = Int(py=addrs[0])
    var pv = Int(py=addrs[1])
    if pk == 0 or pv == 0:
        raise Error("transformer_decode_session_export_state: null buffer address")
    _require_session_open(owner[])
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_export_run(owner[], pk, pv)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def transformer_decode_session_load_state_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Replace the resident cache with the caller's bytes: `addrs` =
    [k_cache, v_cache]; `params` = [cached_tokens]. This is the explicit
    refresh of the state half; the weight half is close and open."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    if len(addrs) != 2 or len(params) != 1:
        raise Error("transformer_decode_session_load_state: expected 2 addresses and 1 scalar")
    var pk = Int(py=addrs[0])
    var pv = Int(py=addrs[1])
    if pk == 0 or pv == 0:
        raise Error("transformer_decode_session_load_state: null buffer address")
    var s0 = Int(py=params[0])
    _require_session_open(owner[])
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _session_load_run(owner[], pk, pv, s0)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def transformer_decode_session_info_binding(session: PythonObject) raises -> PythonObject:
    """[open, usable, cached_tokens, B, max_tokens, window] as ints."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    var out = Python.list()
    var is_open = owner[].is_open()
    var held = 0
    if is_open:
        held = owner[].kv.value().s
    out.append(PythonObject(1 if is_open else 0))
    out.append(PythonObject(1 if owner[].usable else 0))
    out.append(PythonObject(held))
    out.append(PythonObject(owner[].b))
    out.append(PythonObject(owner[].smax))
    out.append(PythonObject(owner[].window))
    return out


def transformer_decode_session_close_binding(session: PythonObject) raises -> PythonObject:
    """Release every device buffer and the context. The caller's cache
    buffers are NOT written here; export_state first if the bytes matter."""
    var owner = session.downcast_value_ptr[TransformerDecodeSession]()
    owner[].close()
    return PythonObject(0)


def causal_lm_session_create_binding() raises -> PythonObject:
    """A closed session. Creation performs no GPU operation."""
    return PythonObject(alloc=CausalLMSession())


def causal_lm_session_open_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`addrs` = [embed (V * D f32), norm (D f32), head (V * D f32, or 0 for a
    head tied to the embedding)]; `params` = [B, V, D, eps (float)]. Every
    tensor is COPIED to the device; later edits are not observed."""
    var owner = session.downcast_value_ptr[CausalLMSession]()
    if len(addrs) != 3 or len(params) != 4:
        raise Error("causal_lm_session_open: expected 3 addresses and 4 scalars")
    if owner[].busy:
        raise Error("causal lm session: busy")
    if owner[].is_open():
        raise Error("causal lm session: already open; close it first")
    var pe = Int(py=addrs[0])
    var pn = Int(py=addrs[1])
    var ph = Int(py=addrs[2])
    if pe == 0 or pn == 0:
        raise Error("causal_lm_session_open: null buffer address")
    var b = Int(py=params[0])
    var v = Int(py=params[1])
    var d = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _lm_open_run(owner[], pe, pn, ph, b, v, d, eps)
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(0)


def causal_lm_session_open_int15_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`causal_lm_session_open` under numeric_profile="fixed15_v1"
    (lane/lowbit-default, 2026-09-29): `addrs` = [embed (V * D f32), norm
    (D f32), head hi (V * D int8), head lo (V * D int8), head exponents (V
    int32)], the head's planes whether or not it is tied; `params` = [B, V,
    D, eps (float)]. Every tensor is COPIED to the device."""
    var owner = session.downcast_value_ptr[CausalLMSession]()
    if len(addrs) != 5 or len(params) != 4:
        raise Error("causal_lm_session_open_int15: expected 5 addresses and 4 scalars")
    if owner[].busy:
        raise Error("causal lm session: busy")
    if owner[].is_open():
        raise Error("causal lm session: already open; close it first")
    var pe = Int(py=addrs[0])
    var pn = Int(py=addrs[1])
    var planes: List[Int] = [Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4])]
    if pe == 0 or pn == 0 or planes[0] == 0 or planes[1] == 0 or planes[2] == 0:
        raise Error("causal_lm_session_open_int15: null buffer address")
    var b = Int(py=params[0])
    var v = Int(py=params[1])
    var d = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _lm_open_run_int15(owner[], pe, pn, planes, b, v, d, eps)
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(0)


def causal_lm_session_run_binding(
    session: PythonObject, layers: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """One call for a prompt and a block of greedy tokens. `layers` is the
    stack's open `TransformerDecodeSession`s in order, all at the same
    cached length. `addrs` = [ids_in (B * L int32), ids_out (B * n_steps
    int32, row-major), logits_out (B * V f32 of the LAST pass's last
    position, or 0)]; `params` = [L, n_steps, cached_tokens]. The first
    pass runs the L prompt tokens; each later pass feeds back the previous
    pick. Returns the post-call cached_tokens. A failed call marks this
    session and every layer session lost."""
    var owner = session.downcast_value_ptr[CausalLMSession]()
    if len(addrs) != 3 or len(params) != 3:
        raise Error("causal_lm_session_run: expected 3 addresses and 3 scalars")
    if owner[].busy:
        raise Error("causal lm session: busy")
    if not owner[].is_open():
        raise Error("causal lm session: not open (causal_lm_session_open first)")
    if not owner[].usable:
        raise Error("causal lm session: lost after a failed call; close it and open a new one")
    var p_ids = Int(py=addrs[0])
    var p_out = Int(py=addrs[1])
    var p_logits = Int(py=addrs[2])
    if p_ids == 0 or p_out == 0:
        raise Error("causal_lm_session_run: null buffer address")
    var l = Int(py=params[0])
    var n_steps = Int(py=params[1])
    var claimed = Int(py=params[2])
    if l < 1 or n_steps < 1:
        raise Error("causal_lm_session_run: L and n_steps must be positive")
    var ptrs = List[Int]()
    var n_layers = Int(py=len(layers))
    if n_layers < 1:
        raise Error("causal_lm_session_run: no layers")
    for i in range(n_layers):  # small-loop(n_layers: decoder layers of the model): one session pointer per layer, not data
        var sp = layers[i].downcast_value_ptr[TransformerDecodeSession]()
        _require_session_open(sp[])
        if sp[].b != owner[].b or sp[].dm != owner[].d:
            raise Error("causal_lm_session_run: layer " + String(i) + " holds B = " + String(sp[].b)
                + ", d_model = " + String(sp[].dm) + "; the session has B = " + String(owner[].b)
                + ", d_model = " + String(owner[].d))
        if sp[].kv.value().s != claimed:
            raise Error(String("transformer decode session: the caller believes cached_tokens = ")
                + String(claimed) + " but layer " + String(i) + " holds " + String(sp[].kv.value().s))
        if sp[].kv.value().s + l + n_steps - 1 > sp[].smax:
            raise Error("causal_lm_session_run: " + String(l + n_steps - 1) + " more positions do not fit the "
                + String(sp[].smax) + "-token cache")
        ptrs.append(Int(sp))
    for i in range(n_layers):  # driver-loop: over decoder layers of the model
        var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
        sp[].busy = True
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _lm_run(owner[], ptrs, p_ids, p_out, p_logits, l, n_steps)
    except error:
        for i in range(n_layers):  # driver-loop: over decoder layers of the model
            var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
            sp[].busy = False
            sp[].usable = False
        owner[].busy = False
        owner[].usable = False
        raise error
    var result = 0
    for i in range(n_layers):  # driver-loop: over decoder layers of the model
        var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
        sp[].busy = False
        result = sp[].kv.value().s
    owner[].busy = False
    return PythonObject(result)


def causal_lm_session_forward_binding(
    session: PythonObject, layers: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`CausalLM.forward` on open sessions in one call: `layers` the stack's
    open `TransformerDecodeSession`s (all at the same cached length);
    `addrs` = [ids (B * L int32), logits_out (B * L * V f32)]; `params` =
    [L]. The layers' caches advance by L. Returns the post-call
    cached_tokens. A failed call marks this session and every layer session
    lost."""
    var owner = session.downcast_value_ptr[CausalLMSession]()
    if len(addrs) != 2 or len(params) != 1:
        raise Error("causal_lm_session_forward: expected 2 addresses and 1 scalar")
    if owner[].busy:
        raise Error("causal lm session: busy")
    if not owner[].is_open():
        raise Error("causal lm session: not open (causal_lm_session_open first)")
    if not owner[].usable:
        raise Error("causal lm session: lost after a failed call; close it and open a new one")
    var p_ids = Int(py=addrs[0])
    var p_logits = Int(py=addrs[1])
    if p_ids == 0 or p_logits == 0:
        raise Error("causal_lm_session_forward: null buffer address")
    var l = Int(py=params[0])
    if l < 1:
        raise Error("causal_lm_session_forward: L must be positive")
    var n_layers = Int(py=len(layers))
    if n_layers < 1:
        raise Error("causal_lm_session_forward: no layers")
    var ptrs = List[Int]()
    var held = -1
    for i in range(n_layers):  # small-loop(n_layers: decoder layers of the model): one session pointer per layer, not data
        var sp = layers[i].downcast_value_ptr[TransformerDecodeSession]()
        _require_session_open(sp[])
        if sp[].b != owner[].b or sp[].dm != owner[].d:
            raise Error("causal_lm_session_forward: layer " + String(i) + " holds B = " + String(sp[].b)
                + ", d_model = " + String(sp[].dm) + "; the session has B = " + String(owner[].b)
                + ", d_model = " + String(owner[].d))
        if held < 0:
            held = sp[].kv.value().s
        elif sp[].kv.value().s != held:
            raise Error("causal_lm_session_forward: the layers hold different cached lengths")
        if sp[].kv.value().s + l > sp[].smax:
            raise Error("causal_lm_session_forward: " + String(l) + " more positions do not fit the "
                + String(sp[].smax) + "-token cache")
        ptrs.append(Int(sp))
    for i in range(n_layers):  # driver-loop: over decoder layers of the model
        var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
        sp[].busy = True
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _lm_forward_run(owner[], ptrs, p_ids, p_logits, l)
    except error:
        for i in range(n_layers):  # driver-loop: over decoder layers of the model
            var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
            sp[].busy = False
            sp[].usable = False
        owner[].busy = False
        owner[].usable = False
        raise error
    var result = 0
    for i in range(n_layers):  # driver-loop: over decoder layers of the model
        var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=ptrs[i])
        sp[].busy = False
        result = sp[].kv.value().s
    owner[].busy = False
    return PythonObject(result)


def causal_lm_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[CausalLMSession]()
    if owner[].busy:
        raise Error("causal lm session: busy")
    owner[].usable = False
    owner[].release()
    return PythonObject(0)


def transformer_backward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The zero-state prefill VJP of one block call, IDENTICAL tier only.

    `addrs` is TWENTY-ONE addresses: 0 x (B*L*d_model), 1-9 the nine
    weights in `transformer_forward`'s order, 10 grad_output (B*L*d_model),
    11 grad_x (B*L*d_model, WRITTEN), 12-20 the nine weight gradients in
    the same order and shapes as the weights (WRITTEN). `params`: 0 B,
    1 L, 2 d_model, 3 n_heads, 4 n_kv_heads, 5 head_dim, 6 intermediate,
    7 window. The activation gradients are the lane's pinned chains; every
    weight gradient is a sum over this call's B*L tokens."""
    if len(addrs) != 21 or len(params) != 8:
        raise Error(
            "transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " window)"
        )
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(
                "transformer backward: null buffer address at slot "
                + String(i)
            )
        a.append(address)
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
        _transformer_backward_run(a, b, l, dm, nh, nkv, hd, it, window)
    return PythonObject(0)



def transformer_session_discard_tape_binding(session: PythonObject,ticket: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy:
        raise Error("transformer tape: session is busy")
    if len(ticket) != 2:
        raise Error("transformer tape: owner and epoch required")
    var cookie = Int(py=ticket[0])
    var epoch = Int(py=ticket[1])
    if cookie != Int(owner):
        raise Error("transformer tape: foreign session")
    # Closing an older ticket cannot discard a newer forward in this session.
    if not owner[].closed and epoch == owner[].tape_generation:
        owner[].invalidate_tape()
    return PythonObject(0)


def transformer_session_invalidate_tape_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    owner[].invalidate_tape()
    return PythonObject(0)


def transformer_arithmetic_profile_changed_binding() raises -> PythonObject:
    return PythonObject(Int(transformer_arithmetic_profile_changed()))


def transformer_arithmetic_profile_binding() raises -> PythonObject:
    return PythonObject(transformer_arithmetic_profile())


def transformer_session_forward_tape_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """11 buffers x/9 weights/y; ten scalars shape/window/budget/cost.

    Returns native session cookie, epoch, retained flag, counted stage bytes.
    Copies finish before return. The weight snapshot belongs to this epoch;
    any subsequent session operation invalidates the unconsumed ticket.
    """
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("transformer tape: IDENTICAL only")
    if len(addrs) != 11 or len(params) != 10:
        raise Error("transformer tape: expected 11 addresses and 10 scalars")
    var a = List[Int]()
    var p = List[Int]()
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("transformer tape: null buffer address")
        a.append(address)
    for i in range(10):  # small-loop(params: fixed scalar list of this entry): argument words, not data
        p.append(Int(py=params[i]))
    if p[0] <= 0 or p[1] <= 0 or p[7] < 0 or p[8] < 0 or p[9] < 0:
        raise Error("transformer tape: invalid shape/window/activation budget/cost")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            try:
                _session_forward_tape_run(Int(owner), a, p)
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    var ticket = Python.list()
    ticket.append(PythonObject(Int(owner)))
    ticket.append(PythonObject(owner[].tape_generation))
    ticket.append(PythonObject(Int(owner[].tape.value().retained)))
    ticket.append(PythonObject(owner[].tape.value().retained_bytes))
    return ticket


def transformer_session_backward_tape_binding(
    session: PythonObject, addrs: PythonObject, ticket: PythonObject,
) raises -> PythonObject:
    """Eleven addresses: dy, dx, norm1,norm2,Q,K,V,O,gate,up,down gradients."""
    var owner = session.downcast_value_ptr[TransformerSession]()
    if owner[].busy or owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    if len(addrs) != 11 or len(ticket) != 2:
        raise Error("transformer tape backward: expected 11 addresses and owner/epoch")
    var cookie = Int(py=ticket[0])
    var generation = Int(py=ticket[1])
    if cookie != Int(owner) or generation != owner[].tape_generation or not owner[].tape:
        raise Error("transformer tape: stale, foreign or already consumed ticket")
    var a = List[Int]()
    for i in range(11):  # small-loop(addrs: fixed buffer-address list of this entry): argument words, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("transformer tape backward: null buffer address")
        a.append(address)
    owner[].busy = True
    try:
        with GILReleased(Python()):
            try:
                _session_backward_tape_run(Int(owner), a, generation)
            except error:
                owner[].clear()
                raise error
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_transformer.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_transformer() -> PythonObject:
    # FAST AND IDENTICAL (lane neural, 2026-09-27). This lane was
    # IDENTICAL-only from 2026-09-10 because its fused kernels were once gated
    # on IDENTICAL and the lower tiers fell back to slower unfused arms. Those
    # fallbacks are gone: FAST runs the same kernels and the same launches with
    # the pins in checks/numerics.mojo compiled to the free schedule. FAST
    # promises quality, never bits.
    # DEVIATION 793's last clause, applied here: a sabotage arm exists to
    # be run by a gate and to FAIL; a Python surface that quietly served
    # one would be a wrong answer wearing a green label. Refuse to exist
    # instead.
    comptime if (
        BLOCK_ANY_SABOTAGE
        or BWD_ANY_SABOTAGE
        or GEMM_ANY_BWD_SABOTAGE
        or GEMM_ANY_SABOTAGE
        or INT15_SABOTAGE
        or INT15_PIECE_SABOTAGE
        or INT15_QUANT_SABOTAGE
        or INT15_EPILOGUE_SABOTAGE
    ):
        abort(
            String(
                "_mojolearn_transformer: refusing to initialize -- a"
                " sabotage arm is compiled into this binary. Sabotage"
                " defines are for the lane gates (transformer/checks/),"
                " never for a shipped binding; rebuild with"
                " bash bindings/build_transformer.sh and no"
                " MOJOLEARN_TRANSFORMER_SABOTAGE_*,"
                " MOJOLEARN_BATCHINV_SABOTAGE_* or GEMM sabotage define."
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_transformer")
        m.def_function[transformer_session_discard_tape_binding]("transformer_session_discard_tape")
        m.def_function[transformer_session_invalidate_tape_binding]("transformer_session_invalidate_tape")
        m.def_function[transformer_arithmetic_profile_changed_binding]("transformer_arithmetic_profile_changed")
        m.def_function[transformer_arithmetic_profile_binding]("transformer_arithmetic_profile")
        m.def_function[transformer_session_forward_tape_binding]("transformer_session_forward_tape")
        m.def_function[transformer_session_backward_tape_binding]("transformer_session_backward_tape")
        m.def_function[transformer_vendor_binding]("transformer_vendor")
        m.def_function[transformer_flash_call_count_binding]("transformer_flash_call_count")
        m.def_function[transformer_numeric_mode_binding](
            "transformer_numeric_mode"
        )
        m.def_function[transformer_attention_profile_binding]("transformer_attention_profile")
        m.def_function[transformer_forward_binding]("transformer_forward")
        _ = m.add_type[TransformerSession]("_TransformerSession")
        m.def_function[transformer_session_create_binding]("transformer_session_create")
        m.def_function[transformer_session_close_binding]("transformer_session_close")
        m.def_function[transformer_session_info_binding]("transformer_session_info")
        m.def_function[transformer_session_forward_binding]("transformer_session_forward")
        # lane/neural-net-experiment: the stateless prefill on a session.
        m.def_function[transformer_session_forward_fresh_binding]("transformer_session_forward_fresh")
        m.def_function[transformer_session_backward_binding]("transformer_session_backward")
        # lane/lowbit-blocks: numeric_profile="fixed15_v1".
        m.def_function[transformer_session_forward_int15_binding]("transformer_session_forward_int15")
        # NVIDIA's full public A/B gate admits discarded-cache prefill.
        # Apple retains its prior default until separately priced; the explicit
        # force flag remains available for its completed arithmetic gate.
        comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and (is_defined["MOJOLEARN_TRANSFORMER_FRESH_PREFILL"]() or (TARGET_COLUMN == COLUMN_NVIDIA and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL"]())):
            m.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        m.def_function[transformer_decode_step_binding](
            "transformer_decode_step"
        )
        m.def_function[transformer_backward_binding]("transformer_backward")
        _ = m.add_type[TransformerForwardTape]("_TransformerForwardTape")
        m.def_function[transformer_forward_tape_enabled_binding]("transformer_forward_tape_enabled")
        m.def_function[transformer_forward_tape_binding]("transformer_forward_tape")
        m.def_function[transformer_backward_tape_binding]("transformer_backward_tape")
        m.def_function[transformer_close_tape_binding]("transformer_close_tape")
        # lane ag-tm-timing: the backward timing entries (measurement only).
        m.def_function[transformer_session_backward_time_binding]("transformer_session_backward_time")
        m.def_function[transformer_backward_tape_time_binding]("transformer_backward_tape_time")
        # DEVIATION 2940: the resident decode session.
        _ = m.add_type[TransformerDecodeSession]("_TransformerDecodeSession")
        m.def_function[transformer_decode_session_create_binding]("transformer_decode_session_create")
        m.def_function[transformer_decode_session_open_binding]("transformer_decode_session_open")
        # lane/lowbit-default: the resident sessions under numeric_profile="fixed15_v1".
        m.def_function[transformer_decode_session_open_int15_binding]("transformer_decode_session_open_int15")
        m.def_function[transformer_decode_session_step_binding]("transformer_decode_session_step")
        m.def_function[transformer_decode_session_forward_binding]("transformer_decode_session_forward")
        m.def_function[transformer_decode_session_export_state_binding]("transformer_decode_session_export_state")
        m.def_function[transformer_decode_session_load_state_binding]("transformer_decode_session_load_state")
        m.def_function[transformer_decode_session_info_binding]("transformer_decode_session_info")
        m.def_function[transformer_decode_session_close_binding]("transformer_decode_session_close")
        # lane/py-lm: the resident CausalLM decode session.
        _ = m.add_type[CausalLMSession]("_CausalLMSession")
        m.def_function[causal_lm_session_create_binding]("causal_lm_session_create")
        m.def_function[causal_lm_session_open_binding]("causal_lm_session_open")
        m.def_function[causal_lm_session_open_int15_binding]("causal_lm_session_open_int15")
        m.def_function[causal_lm_session_run_binding]("causal_lm_session_run")
        m.def_function[causal_lm_session_forward_binding]("causal_lm_session_forward")
        m.def_function[causal_lm_session_close_binding]("causal_lm_session_close")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_transformer: ", e))
