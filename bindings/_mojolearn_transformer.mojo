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
  the BACKWARD (`transformer/checks/transformer_backward.mojo`) is on
  this surface too, as `transformer_backward`, IDENTICAL tier only, and
  `window` (sliding-window causal attention with a ring KV cache) is
  the tenth forward scalar.

THE THREE-TIER SEMANTICS ARE THE BUILD'S, NOT THIS FILE'S. The numeric
mode (fast / deterministic / identical) is a compile-time define
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
from bindings.hostptr import f32_ptr, read_f32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import memcpy
from std.os import getenv
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
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
from transformer.checks.transformer_backward import (
    BWD_ANY_SABOTAGE,
    LlamaBackwardStages,
    llama_decoder_layer_backward,
)
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
    for i in range(n):
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
    for i in range(n):
        a.append(Int(py=addrs[i]))
    while len(a) < base + BLOCK_OPTION_ADDRS:
        a.append(0)
    return a^


def _optional_upload(
    ctx: DeviceContext, addr: Int, n: Int, on: Bool, name: String, what: String,
) raises -> DeviceBuffer[DType.float32]:
    """One optional tensor: uploaded at its length when its flag is on,
    a one-element placeholder otherwise; a presence/flag mismatch is refused
    by the tensor's name HERE, before any device work."""
    if on:
        if addr == 0:
            raise Error(
                what + ": " + name + " is required by its option and its"
                " address is null"
            )
        return _upload_addr(ctx, addr, n)
    if addr != 0:
        raise Error(
            what + ": " + name + " was passed but its option is off; pass"
            " the option or drop the tensor"
        )
    return _llama_zeros[False](ctx, 1)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _read_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, n)


def _write_f32(addr: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), _f32_ptr(addr), len(values))


def _upload_addr(
    ctx: DeviceContext, addr: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy a live caller buffer, synchronizing before its borrow ends.

    IDENTICAL uses DeviceContext's ordinary host-pointer transfer directly.
    Other modes and the diagnostic legacy flag retain pinned staging.
    """
    var p = _f32_ptr(addr)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER"]():
        if n > 0:
            ctx.enqueue_copy(dst_buf=dev, src_ptr=p)
            ctx.synchronize()
            return dev^
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    if n > 0:
        memcpy(dest=host.unsafe_ptr(), src=p, count=n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _download_addr[wait: Bool = True](
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, addr: Int
) raises:
    """Copy the first `n` elements into a live caller buffer.

    IDENTICAL avoids pinned staging and its memcpy. With wait=False, the
    caller must retain both full-buffer owners until a final completion wait.
    Temporary views and legacy staging always finish before their owners die.
    """
    var p = _f32_ptr(addr)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER"]():
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=p, src_buf=buf)
            comptime if wait:
                ctx.synchronize()
        else:
            var direct_view = buf.create_sub_buffer[DType.float32](0, n)
            ctx.enqueue_copy(dst_ptr=p, src_buf=direct_view)
            ctx.synchronize()
            _ = direct_view^
        return
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    ctx.synchronize()
    memcpy(dest=p, src=host.unsafe_ptr(), count=n)
    _ = host^


def _btick(on: Bool, mut t: Int, name: String):
    if not on:
        return
    var now = Int(perf_counter_ns())
    print(
        "timing " + name + " " + String(Float64(now - t) / 1000000.0) + " ms"
    )
    t = now


def transformer_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `mamba_numeric_mode`
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


# ===========================================================================
# The block: profile mojolearn.identical.transformer.fp32.v1
# ===========================================================================


def transformer_lean_stages(hd: Int) -> Bool:
    """Whether this call may skip the `[B, n_heads, L, S]` attention stage
    allocations: the fused attention path will be attempted (IDENTICAL
    build, supported head_dim, not forced eager) and the trace is off here
    always. The eager fallback grows the buffers on demand."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return False
    if not fused_forward_supported_head_dim(hd):
        return False
    return attention_path_choice(PLANT_AT_NONE) != ATTN_PATH_EAGER


def _load_transformer_weights(
    ctx: DeviceContext, dims: LlamaDims, a: List[Int],
    opts: BlockOptions = BlockOptions(), tail: List[Int] = List[Int](),
) raises -> LlamaDeviceWeights:
    """One-call uploads; IDENTICAL skips intermediate host weight Lists.
    Mutable caller arrays are reread and refused on every call as before.

    lane/block-options: `opts` and `tail` (the eleven optional addresses,
    zeros where absent) select the options constructor. At the DEFAULT
    record with an all-zero tail this is the code that was here before,
    upload for upload, so a default block's weights take the old path.
    """
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var hd = dims.head_dim
    var extended = not opts.is_default()
    for i in range(len(tail)):
        if tail[i] != 0:
            extended = True
    if not extended:
        if a[7] == 0:
            raise Error(
                "transformer: gate_proj.weight address is null (the default"
                " options carry a gated MLP)"
            )
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY"]():
            return LlamaDeviceWeights(
                ctx, dims, RMS_EPS,
                _upload_addr(ctx, a[1], dm),
                _upload_addr(ctx, a[2], dm),
                _upload_addr(ctx, a[3], qw * dm),
                _upload_addr(ctx, a[4], kw * dm),
                _upload_addr(ctx, a[5], kw * dm),
                _upload_addr(ctx, a[6], dm * qw),
                _upload_addr(ctx, a[7], it * dm),
                _upload_addr(ctx, a[8], it * dm),
                _upload_addr(ctx, a[9], dm * it),
            )
        else:
            return LlamaDeviceWeights(
                ctx, dims, RMS_EPS,
                _read_f32(a[1], dm),
                _read_f32(a[2], dm),
                _read_f32(a[3], qw * dm),
                _read_f32(a[4], kw * dm),
                _read_f32(a[5], kw * dm),
                _read_f32(a[6], dm * qw),
                _read_f32(a[7], it * dm),
                _read_f32(a[8], it * dm),
                _read_f32(a[9], dm * it),
            )
    # The options constructor. `opts.validate` fires inside it by name.
    var what = String("transformer options")
    var t = tail.copy()
    while len(t) < BLOCK_OPTION_ADDRS:
        t.append(0)
    var w_gate: DeviceBuffer[DType.float32]
    if opts.gated():
        if a[7] == 0:
            raise Error(what + ": gate_proj.weight address is null (a gated MLP needs it)")
        w_gate = _upload_addr(ctx, a[7], it * dm)
    else:
        if a[7] != 0:
            raise Error(
                what + ": gate_proj.weight was passed but the MLP is ungated"
                " (mlp gelu / gelu_tanh); drop it or pick a gated form"
            )
        w_gate = _llama_zeros[False](ctx, 1)
    return LlamaDeviceWeights(
        ctx, dims, opts,
        _upload_addr(ctx, a[1], dm),
        _upload_addr(ctx, a[2], dm),
        _upload_addr(ctx, a[3], qw * dm),
        _upload_addr(ctx, a[4], kw * dm),
        _upload_addr(ctx, a[5], kw * dm),
        _upload_addr(ctx, a[6], dm * qw),
        w_gate^,
        _upload_addr(ctx, a[8], it * dm),
        _upload_addr(ctx, a[9], dm * it),
        _optional_upload(ctx, t[0], qw, opts.qkv_bias, "q_proj.bias", what),
        _optional_upload(ctx, t[1], kw, opts.qkv_bias, "k_proj.bias", what),
        _optional_upload(ctx, t[2], kw, opts.qkv_bias, "v_proj.bias", what),
        _optional_upload(ctx, t[3], dm, opts.o_bias, "o_proj.bias", what),
        _optional_upload(ctx, t[4], dm, opts.norm_bias, "input_layernorm.bias", what),
        _optional_upload(ctx, t[5], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
        _optional_upload(ctx, t[6], it, opts.mlp_bias, "up_proj.bias", what),
        _optional_upload(ctx, t[7], dm, opts.mlp_bias, "down_proj.bias", what),
        _optional_upload(ctx, t[8], it, opts.has_gate_bias(), "gate_proj.bias", what),
        _optional_upload(ctx, t[9], hd, opts.qk_norm, "q_norm.weight", what),
        _optional_upload(ctx, t[10], hd, opts.qk_norm, "k_norm.weight", what),
    )


def _tail_of(a: List[Int], base: Int) -> List[Int]:
    """The eleven optional addresses out of a base + tail list (zeros when
    the list is the base alone)."""
    var t = List[Int]()
    for i in range(BLOCK_OPTION_ADDRS):
        if base + i < len(a):
            t.append(a[base + i])
        else:
            t.append(0)
    return t^


def _check_base_addrs(a: List[Int], slots: List[Int], what: String) raises:
    """Null refusals for the base slots a caller must fill (the gate slot is
    checked by `_load_transformer_weights` against the MLP form)."""
    for i in range(len(slots)):
        if a[slots[i]] == 0:
            raise Error(what + ": null buffer address at slot " + String(slots[i]))


def _lean_for(hd: Int, opts: BlockOptions) -> Bool:
    """`transformer_lean_stages`, and never lean under a softcap, which
    forces the eager kernels (DEVIATION 2947); the eager fallback would
    grow a lean struct anyway, so this is a cost choice, not a correctness
    one."""
    if opts.has_softcap():
        return False
    return transformer_lean_stages(hd)


def _workspace_key(
    b: Int, l: Int, dims: LlamaDims, smax: Int, window: Int, lean: Bool,
    opts: BlockOptions,
) -> List[Int]:
    """The retained workspace's identity: the shape, and the whole options
    record (a different theta or rope_dim is a different rotary table)."""
    var key: List[Int] = [b, l, dims.d_model, dims.n_heads, dims.n_kv,
                          dims.head_dim, dims.intermediate, smax, window, Int(lean)]
    var tail = opts.to_params()
    for i in range(len(tail)):
        key.append(tail[i])
    return key^


struct TransformerWorkspace(Movable):
    var key: List[Int]
    var kv: LlamaKVCache
    var rope: LlamaRopeTable
    var stages: LlamaDeviceStages
    var x: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, dims: LlamaDims,
                 b: Int, l: Int, smax: Int, window: Int, lean: Bool,
                 opts: BlockOptions) raises:
        self.key = _workspace_key(b, l, dims, smax, window, lean, opts)
        self.kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
        # The table from the record: DEVIATIONS 2930-2933 and 2948; the
        # default record is `LlamaRopeTable(ctx, dims, ROPE_THETA, smax)`.
        self.rope = LlamaRopeTable(ctx, dims, opts, smax)
        self.stages = LlamaDeviceStages(ctx, b, l, smax, dims, window, lean=lean)
        self.x = ctx.enqueue_create_buffer[DType.float32](b * l * dims.d_model)

    def retained_bytes(self) -> Int:
        var cells = len(self.x) + len(self.kv.k) + len(self.kv.v)
        cells += len(self.rope.inv_freq) + len(self.rope.cos) + len(self.rope.sin)
        cells += len(self.stages.gemm_workspace.buffer)
        cells += len(self.stages.norm1_sumsq)
        cells += len(self.stages.norm1_out)
        cells += len(self.stages.q_proj)
        cells += len(self.stages.k_proj)
        cells += len(self.stages.v_proj)
        cells += len(self.stages.q_rope)
        cells += len(self.stages.k_rope)
        cells += len(self.stages.k_cache)
        cells += len(self.stages.v_cache)
        cells += len(self.stages.scores)
        cells += len(self.stages.masked)
        cells += len(self.stages.amax)
        cells += len(self.stages.aexp)
        cells += len(self.stages.denom)
        cells += len(self.stages.weights)
        cells += len(self.stages.ctxv)
        cells += len(self.stages.o_proj)
        cells += len(self.stages.residual1)
        cells += len(self.stages.norm2_sumsq)
        cells += len(self.stages.norm2_out)
        cells += len(self.stages.gate_proj)
        cells += len(self.stages.up_proj)
        cells += len(self.stages.silu_out)
        cells += len(self.stages.gated)
        cells += len(self.stages.down_proj)
        cells += len(self.stages.residual2)
        cells += len(self.stages.qbh)
        cells += len(self.stages.kbh)
        cells += len(self.stages.sbh)
        return cells * 4

    def matches(self, key: List[Int]) -> Bool:
        if len(key) != len(self.key):
            return False
        for i in range(len(key)):
            if key[i] != self.key[i]:
                return False
        return True


struct TransformerSession(Movable, Writable):
    """One Python-owned context/workspace; no retained host pointers or weights."""
    var ctx: Optional[DeviceContext]
    var workspace: Optional[TransformerWorkspace]
    var busy: Bool
    var closed: Bool
    var contexts: Int
    var workspaces: Int

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.workspace = Optional[TransformerWorkspace]()
        self.busy = False
        self.closed = False
        self.contexts = 0
        self.workspaces = 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("TransformerSession")

    def __deinit__(deinit self):
        # Same teardown order as ByteLMSession (DEVIATION 2520): buffer
        # destruction enqueues frees, which must drain before context death.
        _ = self.workspace^
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def clear(mut self) raises:
        if self.ctx:
            self.ctx.value().synchronize()
        self.workspace = None
        if self.ctx:
            self.ctx.value().synchronize()


def _transformer_run_session(
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(String("transformer: cached_tokens must be in [0, ")
            + String(smax) + "] (the cache capacity, max_tokens), got "
            + String(s0) + "; the two sides of this boundary disagree about the state")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    if not session.ctx:
        session.ctx = DeviceContext()
        session.contexts += 1
    ref ctx = session.ctx.value()
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var lean = _lean_for(hd, opts)
    var key = _workspace_key(b, l, dims, smax, window, lean, opts)
    var reused = False
    if session.workspace:
        reused = session.workspace.value().matches(key)
    if not reused:
        ctx.synchronize()
        session.workspace = None
        ctx.synchronize()
    # Always reread and revalidate weights; a Python address is not a version.
    var w = _load_transformer_weights(ctx, dims, a, opts, _tail_of(a, 13))
    _btick(ton, tk, "surface.weights_up")
    if not reused:
        session.workspace = TransformerWorkspace(ctx, dims, b, l, smax, window, lean, opts)
        session.workspaces += 1
    ref ws = session.workspace.value()
    if reused:
        ws.stages.reset(ctx)
    # Host cache state is authoritative on EVERY call, including resets and
    # same-address edits. Copy into retained allocations, never trust a pointer.
    ctx.enqueue_copy(dst_buf=ws.kv.k, src_ptr=_f32_ptr(a[10]))
    ctx.enqueue_copy(dst_buf=ws.kv.v, src_ptr=_f32_ptr(a[11]))
    ctx.enqueue_copy(dst_buf=ws.x, src_ptr=_f32_ptr(a[0]))
    ctx.synchronize()
    ws.kv.s = s0
    _btick(ton, tk, "surface.cache_stages_x_up")
    var trace = IdentityTrace.disabled()
    llama_decoder_layer_forward(ctx, ws.stages, ws.kv, ws.rope, w, ws.x,
                                b, l, s0, trace, String("py"))
    _btick(ton, tk, "surface.forward")
    _download_addr(ctx, ws.stages.residual2, b * l * dm, a[12])
    _download_addr(ctx, ws.kv.k, len(ws.kv.k), a[10])
    _download_addr(ctx, ws.kv.v, len(ws.kv.v), a[11])
    _btick(ton, tk, "surface.outputs_down")
    var result = ws.kv.s
    # Bound retained device-buffer storage per model. Large legal calls still
    # run, but release their workspace after completion rather than pinning it.
    var retain = ws.retained_bytes() <= 64 * 1024 * 1024
    _ = w^
    # Drain the per-call weight frees before returning to the next call.
    ctx.synchronize()
    if not retain:
        session.workspace = None
        ctx.synchronize()
    return result


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


def _transformer_run[discard_cache: Bool = False](
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    it: Int,
    smax: Int,
    s0: Int,
    window: Int,
    opts: BlockOptions,
) raises -> Int:
    """The GIL-free half of the two entry points: everything after the
    `PythonObject`s have been read. Builds the device weights, uploads
    the caller's cache state, runs THE certified entry point once, and
    writes the output and the post-call state back into the caller's
    buffers. Returns the post-call `cached_tokens` (DEVIATION 795(ii):
    the one integer piece of the state).

    `opts` (lane/block-options) is the record decoded from the params
    tail; `a` is the 13 base addresses plus the 11-address tail (zeros
    where absent).

    `[[mojo-buffer-freed-at-last-use]]`: every device buffer below is
    still alive when `llama_decoder_layer_forward` returns because that
    function synchronizes before it does, and the explicit transfers at
    the end hold them past the last download anyway."""
    # LlamaDims.validate REFUSES the five divisibility rules by name
    # (d_model == n_heads*head_dim, n_heads % n_kv == 0, head_dim even,
    # positivity) -- reached BEFORE any buffer size below is computed
    # from them, which is why nothing here pre-judges the shape.
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
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
    # The caller's cache buffers: `smax` slots for the linear cache, a ring
    # of `window` slots per (batch, kv head) under a sliding window.
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd

    var ctx = DeviceContext()
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    # Host weights THROUGH the lane's own struct (its length table is the
    # reference shape authority, and its constructor is where the weights
    # are refused non-finite ONCE, DEVIATION 1875); values arrive as
    # given bits, unjudged.
    var w = _load_transformer_weights(ctx, dims, a, opts, _tail_of(a, 13))
    # The caller's cache over the fresh zeros. LlamaKVCache's own
    # constructor refuses b <= 0, smax <= 0 and smax > max_positions (the
    # absolute-position ceiling, DEVIATION 812's 8192 at the default
    # record, DEVIATION 2933) BY NAME before the uploads below. Zeros in
    # with s0 == 0 IS a fresh sequence; anything else is a carried one,
    # packed at stride s0 (DEVIATION 795(ii)).
    _btick(ton, tk, "surface.weights_up")
    var kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
    comptime if not discard_cache:
        kv.k = _upload_addr(ctx, a[10], cache_n)
        kv.v = _upload_addr(ctx, a[11], cache_n)
    kv.s = s0
    # Per call, from the record (DEVIATION 795(iii) at the default record:
    # the FROZEN theta). p_max is the cache capacity: pos0 + l <= kv.s_max
    # <= p_max holds for every legal call, so the table always covers the
    # absolute positions used.
    var rope = LlamaRopeTable(ctx, dims, opts, smax)
    var stages = LlamaDeviceStages(
        ctx, b, l, smax, dims, window, lean=_lean_for(hd, opts)
    )
    var dx = _upload_addr(ctx, a[0], b * l * dm)
    _btick(ton, tk, "surface.cache_stages_x_up")

    var trace = IdentityTrace.disabled()
    # pos0 = s0: DEVIATION 1028 (cache slot j IS absolute position j)
    # refuses any other value, and this surface has no other value to
    # offer. Prefill, chunked continuation and decode are all this one
    # call; capacity growth past smax is refused by name downstream.
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, w, dx, b, l, s0, trace, String("py")
    )

    # The block output is residual2 (contract section 2's
    # `residual2 + mlp(...)`, stage `residual2.out`), and the cache goes
    # back to its owner whole -- the full capacity buffer, so the bytes
    # round-trip exactly whatever the used stride is.
    _btick(ton, tk, "surface.forward")
    _download_addr(ctx, stages.residual2, b * l * dm, a[12])
    comptime if not discard_cache:
        _download_addr(ctx, kv.k, cache_n, a[10])
        _download_addr(ctx, kv.v, cache_n, a[11])
    _btick(ton, tk, "surface.outputs_down")
    var out_len = kv.s
    _ = w^
    _ = kv^
    _ = rope^
    _ = stages^
    _ = dx^
    # Keep the context alive until every device allocation is destroyed
    # (DEVIATION 1946) AND drain the frees those destructions enqueued
    # before it goes (DEVIATION 2520, extended by DEVIATION 3010). Without
    # this drain the MAX runtime allocator's lock stays held for the whole
    # PROCESS and the next GPU allocation anywhere in it never returns:
    # measured as `verify --all` walking 47 lanes and then stopping dead at
    # `transformer-bf16w`, the first lane to allocate after this one's
    # context died.
    ctx.synchronize()
    _ = ctx^
    return out_len


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
    for i in range(10):
        a.append(raw[i])
    a.append(0)
    a.append(0)
    a.append(raw[10])
    for i in range(BLOCK_OPTION_ADDRS):
        a.append(raw[11 + i])
    _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9], what)
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


# ===========================================================================
# The resident decode session (DEVIATION 2940, lane/infer-speed-neural,
# 2026-09-17): device-owned weights, KV cache, rotary table and L = 1 stages
# that live across decode steps.
#
# WHY IT CANNOT MOVE A BIT. Every step calls `llama_decoder_layer_forward`,
# THE certified entry point, at `l = 1` and `pos0 = kv.s`, on the same
# structs the per-call entry builds: `LlamaDeviceWeights` from the same nine
# uploads, `LlamaKVCache` from the same cache bytes, `LlamaRopeTable` at the
# same `(theta, head_dim, s_max)`, `LlamaDeviceStages` at `(B, 1, s_max,
# window)`. What differs is LIFETIME: the per-call entry constructs all of
# that per token and tears it down; the session constructs it once. Stage
# reuse across calls is `training/byte_lm.mojo::ByteTrainer`'s settled
# pattern (its `forward` stages are reused on every training step) and every
# stage the block reads it writes first in the same call. No arithmetic is
# respelled here; the only new device operations are transfers.
#
# OWNERSHIP AND REFRESH. The
# session COPIES the weights and the cache at `open`; edits to the caller's
# arrays after that are NOT observed until `close` and a new `open`
# (weights) or `load_state` (cache). The caller's cache buffers are STALE
# while the session is open and become current at `export_state` or `close`.
# The Python owner (`_transformer_impl.py::TransformerDecodeSession`) marks
# the state object resident so the per-call `forward`/`step` refuse it until
# it is synced back.
#
# NOT `TransformerSession` above (`transformer_session_*`): that one is the
# per-call route's retained context and workspace, holding no weights and
# no authoritative cache (both reread every call). This one owns the weights
# and the cache for the life of the session. The entry points are
# `transformer_decode_session_*` so both are exported from one binding.
# ===========================================================================


struct TransformerDecodeSession(Movable, Writable):
    """Python-owned device lifetime for one block's decode. No global
    handles and no retained host pointers: every caller buffer is read or
    written inside the call that names it."""

    var ctx: Optional[DeviceContext]
    var w: Optional[LlamaDeviceWeights]
    var kv: Optional[LlamaKVCache]
    var rope: Optional[LlamaRopeTable]
    var stages: Optional[LlamaDeviceStages]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var b: Int
    var dm: Int
    var nh: Int
    var nkv: Int
    var hd: Int
    var it: Int
    var smax: Int
    var window: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.w = Optional[LlamaDeviceWeights]()
        self.kv = Optional[LlamaKVCache]()
        self.rope = Optional[LlamaRopeTable]()
        self.stages = Optional[LlamaDeviceStages]()
        self.dx = Optional[DeviceBuffer[DType.float32]]()
        self.b = 0
        self.dm = 0
        self.nh = 0
        self.nkv = 0
        self.hd = 0
        self.it = 0
        self.smax = 0
        self.window = 0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerDecodeSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("TransformerDecodeSession")

    def is_open(self) -> Bool:
        return Bool(self.ctx) and Bool(self.kv)

    def release(mut self):
        """Buffers die before their context, and the frees they enqueue
        drain before the context goes (DEVIATION 2520's ordering)."""
        self.stages = None
        self.dx = None
        self.rope = None
        self.kv = None
        self.w = None
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        self.ctx = None

    def __deinit__(deinit self):
        self.release()

    def close(mut self) raises:
        if self.busy:
            raise Error("transformer decode session: busy")
        self.usable = False
        self.release()


def _require_session_open(s: TransformerDecodeSession) raises:
    if s.busy:
        raise Error("transformer decode session: busy")
    if not s.is_open():
        raise Error("transformer decode session: not open (transformer_decode_session_open first)")
    if not s.usable:
        raise Error("transformer decode session: lost after a failed call; close it and open a new one")


def _session_open_run(
    mut s: TransformerDecodeSession, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int,
    hd: Int, it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """The GIL-free half of `transformer_decode_session_open`: one context, the nine
    weight uploads, the cache uploads over fresh zeros, the rotary table at
    `smax`, the L = 1 stages and the resident x buffer. `a` is the
    `transformer_forward` 13 + 11 layout with slots 0 and 12 unused."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(
            String("transformer decode session: cached_tokens must be in [0, ") + String(smax)
            + "], got " + String(s0))
    if window < 0:
        raise Error("transformer decode session: window must be >= 0 (0 = full causal)")
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd
    s.ctx = DeviceContext()
    s.w = _load_transformer_weights(s.ctx.value(), dims, a, opts, _tail_of(a, 13))
    var kv = LlamaKVCache(s.ctx.value(), b, dims, smax, window, opts.max_positions)
    kv.k = _upload_addr(s.ctx.value(), a[10], cache_n)
    kv.v = _upload_addr(s.ctx.value(), a[11], cache_n)
    kv.s = s0
    s.kv = kv^
    s.rope = LlamaRopeTable(s.ctx.value(), dims, opts, smax)
    s.stages = LlamaDeviceStages(s.ctx.value(), b, 1, smax, dims, window, lean=_lean_for(hd, opts))
    s.dx = s.ctx.value().enqueue_create_buffer[DType.float32](b * dm)
    s.ctx.value().synchronize()
    s.b = b
    s.dm = dm
    s.nh = nh
    s.nkv = nkv
    s.hd = hd
    s.it = it
    s.smax = smax
    s.window = window
    return s0


def _session_step_run(mut s: TransformerDecodeSession, px: Int, py: Int) raises -> Int:
    """One decode token on the resident structs: x in, the block output
    out, the cache advanced on the device. ONE completion wait, after the
    output copy is enqueued; the caller's x stays live for the call."""
    ref ctx = s.ctx.value()
    var n = s.b * s.dm
    ctx.enqueue_copy(dst_buf=s.dx.value(), src_ptr=_f32_ptr(px))
    var trace = IdentityTrace.disabled()
    var pos0 = s.kv.value().s
    llama_decoder_layer_forward(
        ctx, s.stages.value(), s.kv.value(), s.rope.value(), s.w.value(), s.dx.value(),
        s.b, 1, pos0, trace, String("py.session"))
    if n != len(s.stages.value().residual2):
        raise Error("transformer decode session: the L = 1 stages hold a different output length")
    ctx.enqueue_copy(dst_ptr=_f32_ptr(py), src_buf=s.stages.value().residual2)
    ctx.synchronize()
    return s.kv.value().s


def _session_forward_run(mut s: TransformerDecodeSession, px: Int, py: Int, l: Int) raises -> Int:
    """`l` tokens per row on the resident weights and cache (a prefill or a
    chunked continuation), with call-shaped stages built for this call
    alone. The same entry point at `l`, so the bytes are `transformer_forward`'s."""
    ref ctx = s.ctx.value()
    var dims = LlamaDims(s.dm, s.nh, s.nkv, s.hd, s.it)
    var n = s.b * l * s.dm
    var stages = LlamaDeviceStages(ctx, s.b, l, s.smax, dims, s.window, lean=_lean_for(s.hd, s.w.value().opts))
    var dx = _upload_addr(ctx, px, n)
    var trace = IdentityTrace.disabled()
    var pos0 = s.kv.value().s
    llama_decoder_layer_forward(
        ctx, stages, s.kv.value(), s.rope.value(), s.w.value(), dx, s.b, l, pos0, trace,
        String("py.session.forward"))
    _download_addr(ctx, stages.residual2, n, py)
    _ = stages^
    _ = dx^
    return s.kv.value().s


def _session_export_run(mut s: TransformerDecodeSession, pk: Int, pv: Int) raises -> Int:
    ref ctx = s.ctx.value()
    var cap = s.smax
    if s.window > 0:
        cap = s.window
    var cache_n = s.b * s.nkv * cap * s.hd
    _download_addr(ctx, s.kv.value().k, cache_n, pk)
    _download_addr(ctx, s.kv.value().v, cache_n, pv)
    return s.kv.value().s


def _session_load_run(mut s: TransformerDecodeSession, pk: Int, pv: Int, s0: Int) raises -> Int:
    if s0 < 0 or s0 > s.smax:
        raise Error(
            String("transformer decode session: cached_tokens must be in [0, ") + String(s.smax)
            + "], got " + String(s0))
    ref ctx = s.ctx.value()
    var cap = s.smax
    if s.window > 0:
        cap = s.window
    var cache_n = s.b * s.nkv * cap * s.hd
    s.kv.value().k = _upload_addr(ctx, pk, cache_n)
    s.kv.value().v = _upload_addr(ctx, pv, cache_n)
    s.kv.value().s = s0
    return s0


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
    for i in range(11):
        a.append(raw[i])
    a.append(0)
    for i in range(BLOCK_OPTION_ADDRS):
        a.append(raw[11 + i])
    _check_base_addrs(a, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11], what)
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
    [k_cache, v_cache]; `params` = [cached_tokens]. The explicit refresh
    for the state half; the weight half is close and open."""
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


# ===========================================================================
# The block backward: a zero-state prefill's VJP, IDENTICAL tier only.
# ===========================================================================


def _transformer_backward_run(
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    it: Int,
    window: Int,
) raises:
    """Forward from a zero cache at positions [0, L), then
    `llama_decoder_layer_backward` on the saved stages; the ten gradients
    are written into the caller's buffers. The forward is recomputed here
    rather than taken from a previous call because the lane's backward
    reads the forward's DEVICE stages (`LlamaDeviceStages`) and those are
    not part of the Python-visible state."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    if window < 0:
        raise Error("transformer backward: window must be >= 0")
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var m = b * l
    var ctx = DeviceContext()
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var w = _load_transformer_weights(ctx, dims, a)
    var kv = LlamaKVCache(ctx, b, dims, l, window)
    var rope = LlamaRopeTable(ctx, dims, ROPE_THETA, l)
    var lean = transformer_lean_stages(hd)
    var stages = LlamaDeviceStages(ctx, b, l, l, dims, window, lean=lean)
    var dx = _upload_addr(ctx, a[0], m * dm)
    _btick(ton, tk, "surface.inputs_up")
    var off = IdentityTrace.disabled()
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, w, dx, b, l, 0, off, String("pyf")
    )
    _btick(ton, tk, "surface.forward")
    var bst = LlamaBackwardStages(ctx, b, l, l, dims, lean=lean)
    var d_out = _read_f32(a[10], m * dm)
    var offb = IdentityTrace.disabled()
    llama_decoder_layer_backward(
        ctx, bst, stages, w, rope.cos, rope.sin, dx, d_out, b, l, 0,
        offb, String("pyb"),
    )
    _btick(ton, tk, "surface.backward")
    # All ten sources remain owned by bst, and destinations by the Python
    # frame. Queue their copies together and wait before either owner dies.
    _download_addr[False](ctx, bst.d_x, m * dm, a[11])
    _download_addr[False](ctx, bst.dw_norm1, dm, a[12])
    _download_addr[False](ctx, bst.dw_norm2, dm, a[13])
    _download_addr[False](ctx, bst.dw_q, qw * dm, a[14])
    _download_addr[False](ctx, bst.dw_k, kw * dm, a[15])
    _download_addr[False](ctx, bst.dw_v, kw * dm, a[16])
    _download_addr[False](ctx, bst.dw_o, dm * qw, a[17])
    _download_addr[False](ctx, bst.dw_gate, it * dm, a[18])
    _download_addr[False](ctx, bst.dw_up, it * dm, a[19])
    _download_addr[False](ctx, bst.dw_down, dm * it, a[20])
    ctx.synchronize()
    _btick(ton, tk, "surface.outputs_down")
    _ = bst^
    _ = stages^
    _ = w^
    _ = kv^
    _ = rope^
    _ = dx^
    # The `synchronize()` above drained the OUTPUT COPIES; these six
    # destructions enqueue the frees, and those must drain too before the
    # context is destroyed (DEVIATION 2520, extended by DEVIATION 3010).
    ctx.synchronize()
    _ = ctx^


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
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error(
            "transformer backward: only the IDENTICAL zero-state prefill"
            " backward is implemented"
        )
    if len(addrs) != 21 or len(params) != 8:
        raise Error(
            "transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " window)"
        )
    var a = List[Int]()
    for i in range(21):
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


@export
def PyInit__mojolearn_transformer() abi("C") -> PythonObject:
    # IDENTICAL-ONLY (2026-09-10). The FAST and DETERMINISTIC builds of this
    # lane were never a faster path: every fused kernel here is gated on
    # `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`, so the lower tiers fell back
    # to the unfused arms and ran SLOWER than the default. They are no longer
    # built (bindings/build_transformer.sh refuses) and the lane no longer carries
    # the fallbacks. Refuse to exist rather than answer under a tier label
    # whose arithmetic is gone.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        abort(
            String(
                "_mojolearn_transformer: refusing to initialize -- this lane supports only"
                " the IDENTICAL tier. Rebuild with"
                " MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh"
            )
        )
    # DEVIATION 793's last clause, applied here: a sabotage arm exists to
    # be run by a gate and to FAIL; a Python surface that quietly served
    # one would be a wrong answer wearing a green label. Refuse to exist
    # instead.
    comptime if (
        BLOCK_ANY_SABOTAGE
        or BWD_ANY_SABOTAGE
        or GEMM_ANY_BWD_SABOTAGE
        or GEMM_ANY_SABOTAGE
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
        m.def_function[transformer_vendor_binding]("transformer_vendor")
        m.def_function[transformer_numeric_mode_binding](
            "transformer_numeric_mode"
        )
        m.def_function[transformer_forward_binding]("transformer_forward")
        _ = m.add_type[TransformerSession]("_TransformerSession")
        m.def_function[transformer_session_create_binding]("transformer_session_create")
        m.def_function[transformer_session_close_binding]("transformer_session_close")
        m.def_function[transformer_session_info_binding]("transformer_session_info")
        m.def_function[transformer_session_forward_binding]("transformer_session_forward")
        # NVIDIA's full public A/B gate admits discarded-cache prefill.
        # Apple retains its prior default until separately priced; the explicit
        # force flag remains available for its completed arithmetic gate.
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and (is_defined["MOJOLEARN_TRANSFORMER_FRESH_PREFILL"]() or (TARGET_COLUMN == COLUMN_NVIDIA and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL"]())):
            m.def_function[transformer_forward_fresh_binding]("transformer_forward_fresh")
        m.def_function[transformer_decode_step_binding](
            "transformer_decode_step"
        )
        m.def_function[transformer_backward_binding]("transformer_backward")
        # DEVIATION 2940: the resident decode session.
        _ = m.add_type[TransformerDecodeSession]("_TransformerDecodeSession")
        m.def_function[transformer_decode_session_create_binding]("transformer_decode_session_create")
        m.def_function[transformer_decode_session_open_binding]("transformer_decode_session_open")
        m.def_function[transformer_decode_session_step_binding]("transformer_decode_session_step")
        m.def_function[transformer_decode_session_forward_binding]("transformer_decode_session_forward")
        m.def_function[transformer_decode_session_export_state_binding]("transformer_decode_session_export_state")
        m.def_function[transformer_decode_session_load_state_binding]("transformer_decode_session_load_state")
        m.def_function[transformer_decode_session_info_binding]("transformer_decode_session_info")
        m.def_function[transformer_decode_session_close_binding]("transformer_decode_session_close")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_transformer: ", e))
