# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed side of the transformer block binding (`bindings/_mojolearn_transformer.mojo`).

Everything here takes typed values (Ints, addresses, lists, handles) and
imports no Python. The binding unpacks its PythonObject arguments, calls
one entry below, and packs the result. Moved word for word from the
binding (thin-shell lane, 2026-10-21): same operations, same order.
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

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast, unsafe_memcpy
from std.os import getenv
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN

from core.identity_trace import IdentityTrace
from core.device_scan import device_first_nonfinite
from core.device_nonfinite_pair import DeviceNonfinitePair
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojolearnNeuralTransformerContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTransformerContextFastV3"
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
from transformer.impl.llama.fused_attention import fused_forward_supported_head_dim
# Lane x4: the public decode session runs the attention step op.
from transformer.attn_step_session import AttnStepArm, attn_step_ints, attn_step_same_layout
from core.weight_epoch import WeightKey
from autograd.block_step_ops import AttnPrefill
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
from transformer.impl.llama.afn_apple_fast import afn_flash_call_count


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


def _upload_addr[wait: Bool = True](
    ctx: DeviceContext, addr: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy a live caller buffer, synchronizing before its borrow ends.

    IDENTICAL uses DeviceContext's ordinary host-pointer transfer directly.
    Other modes and the diagnostic legacy flag retain pinned staging.
    `wait=False` (lane/neural-apple, 2026-09-28; the direct path only) leaves
    the copy in flight: the caller's array outlives the call, every reader
    is enqueued after it on the same in-order `ctx`, and the caller waits
    before it returns.
    """
    var p = _f32_ptr(addr)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER"]():
        if n > 0:
            ctx.enqueue_copy(dst_buf=dev, src_ptr=p)
            comptime if wait:
                ctx.synchronize()
            return dev^
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    if n > 0:
        unsafe_memcpy(dest=host.unsafe_ptr(), src=p, count=n)
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
    comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER"]():
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
    unsafe_memcpy(dest=p, src=host.unsafe_ptr(), count=n)
    _ = host^


def _btick(on: Bool, mut t: Int, name: String):
    if not on:
        return
    var now = Int(perf_counter_ns())
    print(
        "timing " + name + " " + String(Float64(now - t) / 1000000.0) + " ms"
    )
    t = now


# ===========================================================================
# The block: profile mojolearn.identical.transformer.fp32.v1
# ===========================================================================


# transformer_lean_stages and _lean_for live in transformer/lean_stages.mojo
# (lane x4: the attention step op imports them without a cycle).
from transformer.lean_stages import transformer_lean_stages, _lean_for
from core.shared_slot import SharedSlot


def _load_transformer_weights[wait: Bool = True](
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
    for i in range(len(tail)):  # small-loop(tail: the 11 optional block addresses): checks address presence, not data
        if tail[i] != 0:
            extended = True
    if not extended:
        if a[7] == 0:
            raise Error(
                "transformer: gate_proj.weight address is null (the default"
                " options carry a gated MLP)"
            )
        comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY"]():
            return LlamaDeviceWeights(
                ctx, dims, RMS_EPS,
                _upload_addr[wait](ctx, a[1], dm),
                _upload_addr[wait](ctx, a[2], dm),
                _upload_addr[wait](ctx, a[3], qw * dm),
                _upload_addr[wait](ctx, a[4], kw * dm),
                _upload_addr[wait](ctx, a[5], kw * dm),
                _upload_addr[wait](ctx, a[6], dm * qw),
                _upload_addr[wait](ctx, a[7], it * dm),
                _upload_addr[wait](ctx, a[8], it * dm),
                _upload_addr[wait](ctx, a[9], dm * it),
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
        w_gate = _upload_addr[wait](ctx, a[7], it * dm)
    else:
        if a[7] != 0:
            raise Error(
                what + ": gate_proj.weight was passed but the MLP is ungated"
                " (mlp gelu / gelu_tanh); drop it or pick a gated form"
            )
        w_gate = _llama_zeros[False](ctx, 1)
    return LlamaDeviceWeights(
        ctx, dims, opts,
        _upload_addr[wait](ctx, a[1], dm),
        _upload_addr[wait](ctx, a[2], dm),
        _upload_addr[wait](ctx, a[3], qw * dm),
        _upload_addr[wait](ctx, a[4], kw * dm),
        _upload_addr[wait](ctx, a[5], kw * dm),
        _upload_addr[wait](ctx, a[6], dm * qw),
        w_gate^,
        _upload_addr[wait](ctx, a[8], it * dm),
        _upload_addr[wait](ctx, a[9], dm * it),
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
    for i in range(len(slots)):  # small-loop(slots: a handful of base address slots): null checks on addresses, not data
        if a[slots[i]] == 0:
            raise Error(what + ": null buffer address at slot " + String(slots[i]))


def _lean_for_w(hd: Int, w: LlamaDeviceWeights) -> Bool:
    """`_lean_for` for weights that may carry the profile's planes: never
    lean under `fixed15_v1`, as `_transformer_run_session_int15` builds its
    stages (lane/lowbit-default: the resident sessions under the profile)."""
    if w.int15:
        return False
    return _lean_for(hd, w.opts)


# TOMBSTONE: MOJOLEARN_AFN_ATTN_ARENA (removed: dead; no caller built TransformerWorkspace after lanes x45c and x6) with TransformerWorkspace and _workspace_key deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_ATTN_ARENA.patch


from transformer.experiments.profile import transformer_arithmetic_profile, transformer_arithmetic_profile_changed


struct TransformerSession(Movable, Writable):
    """One Python-owned context/workspace; no retained host pointers or
    float32 weights. Under `numeric_profile="fixed15_v1"` it RETAINS the
    block's seven projection planes on the device (`planes`, keyed by
    their addresses and the shape): those arrays are private to the Python
    block, made once at its construction and never exposed or written, so
    unlike a caller's float32 array their address is their version."""
    var ctx: Optional[DeviceContext]
    var arm15: Optional[AttnStepArm]
    """Lane x45c: the fixed15_v1 block forward on the attention step op (an
    adopted entry over the profile's weights), keyed by `key15`: the shape,
    every weight and plane address and the weight epoch."""
    var key15: WeightKey
    # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
    # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
    # (the retained float32 `weights` with their byte-compare counters and
    # the borrowed session `tape` with its generation served only the
    # forward tape API.)
    var arm: Optional[AttnStepArm]
    """Lane x4b: the block's forward, step and backward on the attention
    step op (transformer/attn_step_session.mojo): its weights packed once
    into the op's weight buffer (keyed by the caller's addresses and the
    weight epoch, lane x45c), its cache in the op's state buffer, its L
    stages and the backward stages kept by the arm."""
    var busy: Bool
    var closed: Bool
    var contexts: Int
    var workspaces: Int

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.arm15 = Optional[AttnStepArm]()
        self.key15 = WeightKey()
        self.arm = Optional[AttnStepArm]()
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
        _ = self.arm^
        _ = self.arm15^
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def clear(mut self) raises:
        if self.ctx:
            self.ctx.value().synchronize()
        self.arm = None
        self.arm15 = None
        self.key15 = WeightKey()
        if self.ctx:
            self.ctx.value().synchronize()


struct _TransformerSessionEnv(Defaultable, Movable):
    """The session entries' environment switch, read once per process
    (lane ne3-c-blocks-embedding, row N159: every session forward built three
    or four `String(getenv(...))` values). Set before the process starts; a
    change after the first session call is not seen."""
    var retain_cap_bytes: Int

    def __init__(out self):
        var cap_mb = 512
        var s = String(getenv("MOJOLEARN_TRANSFORMER_RETAIN_MB"))
        if s != "":
            try:
                cap_mb = Int(atol(s))
            except:
                cap_mb = 512
        if cap_mb < 0:
            cap_mb = 0
        self.retain_cap_bytes = cap_mb * 1024 * 1024


# Process-wide (one environment per process); bump the name with the layout.
comptime _TRANSFORMER_SESSION_ENV = SharedSlot[_TransformerSessionEnv, "MojolearnTransformerSessionEnvV3"]


def _session_retain_cap_bytes() raises -> Int:
    """Retained workspace budget per session: `MOJOLEARN_TRANSFORMER_RETAIN_MB`
    (an integer, MiB), default 512. Was a fixed 64 MiB, which the board's
    block shape (B1 L2048 DM384 FF1024: about 87 MiB of lean stages) exceeded
    on every call, so the whole workspace was rebuilt per call."""
    return _TRANSFORMER_SESSION_ENV.get_or_create_ptr()[].retain_cap_bytes


# TOMBSTONE: MOJOLEARN_IDN_ATTN_DEFER_REGIME (loser: transformer-forward 1.376) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch
# TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (loser: transformer-forward 1.350) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
def _transformer_run_session[discard_cache: Bool = False](
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`discard_cache` (lane/neural-net-experiment): the stateless prefill on
    the session -- the caller passes no cache (slots 10 and 11 are 0), the
    op's cache is zero-filled on the device, nothing is uploaded or
    downloaded for it, and `s0` is 0. Every call runs on the attention step
    op (`_transformer_run_session_op`, lane x4b)."""
    # restored 2026-11-09 by lane r1-restore-measured-winners as the op's fresh prefill arm: the discard_cache body
    # this function ran in 0.8.66 (v0.8.66 / 616dbdf75f^) is autograd/block_step_ops.mojo attn_step_fresh_prefill,
    # which AttnStepArm.block_forward takes by default (-D MOJOLEARN_TRANSFORMER_GENERIC_PREFILL: the generic op run).
    # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
    # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
    # (the `retain_tape` parameter and its workspace body, the forward tape's
    # alone, went with the tape API.)
    # TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (loser: transformer-forward 1.350) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_DEFER_REGIME (loser: transformer-forward 1.376) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch
    return _transformer_run_session_op[discard_cache](session, a, b, l, dm, nh, nkv, hd, it, smax, s0,
                                                      window, opts)


def _stage_cells(stages: LlamaDeviceStages) -> Int:
    """The float cells of one block's stages (the retention budget's count;
    the attention step arm's pre stages are counted against the cap)."""
    var cells = len(stages.gemm_workspace.buffer)
    cells += len(stages.norm1_sumsq) + len(stages.norm1_out) + len(stages.q_proj) + len(stages.k_proj)
    cells += len(stages.v_proj) + len(stages.q_rope) + len(stages.k_rope) + len(stages.k_cache)
    cells += len(stages.v_cache) + len(stages.scores) + len(stages.masked) + len(stages.amax)
    cells += len(stages.aexp) + len(stages.denom) + len(stages.weights) + len(stages.ctxv)
    cells += len(stages.o_proj) + len(stages.residual1) + len(stages.norm2_sumsq) + len(stages.norm2_out)
    cells += len(stages.gate_proj) + len(stages.up_proj) + len(stages.silu_out) + len(stages.gated)
    cells += len(stages.down_proj) + len(stages.residual2) + len(stages.qbh) + len(stages.kbh) + len(stages.sbh)
    return cells


def _session_arm(
    mut session: TransformerSession, a: List[Int], ints: List[Int], b: Int, dm: Int, nh: Int, nkv: Int,
    hd: Int, it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises:
    """Leave `session.arm` open over this block's weights (lane x4b): the
    same arm when the weight layout is the same (its shape words and the
    caller's bytes are reconciled by the call itself), a new one otherwise."""
    if not session.ctx:
        session.ctx = neural_ctx[_NEURAL_CTX]()
        session.contexts += 1
    if session.arm and attn_step_same_layout(session.arm.value().ints, ints):
        return
    session.arm = None
    var words = List[Int]()
    if len(ints) > 8:
        for i in range(8, len(ints)):  # small-loop(ints: the 17 block option words): argument words, no data
            words.append(ints[i])
    session.arm = AttnStepArm(session.ctx.value(), a, b, dm, nh, nkv, hd, it, smax, s0, window, opts, words,
                              keyed=True)
    session.workspaces += 1


def _session_arm_budget(mut session: TransformerSession) raises:
    """The retained-storage budget of the block session
    (`MOJOLEARN_TRANSFORMER_RETAIN_MB`): kept L stages over it go after the
    call (the next call rebuilds them), as the workspace did."""
    if not session.arm:
        return
    ref arm = session.arm.value()
    if arm.pre.stages and _stage_cells(arm.pre.stages.value()) * 4 > _session_retain_cap_bytes():
        arm.pre.clear()
        arm.bst = None
        arm.dout = None
        arm.bkey = List[Int]()
        session.ctx.value().synchronize()


def _transformer_run_session_op[discard_cache: Bool](
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`_transformer_run_session` on the attention step op (lane x4b): the
    block session's arm (weights packed once, compared byte for byte with
    the caller's arrays every call) runs the call. `discard_cache`: the
    stateless prefill, zero caches at position 0 on the device, nothing of
    them crosses the host; otherwise the caller's caches up at s0 and back
    down after the run. One wait."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(String("transformer: cached_tokens must be in [0, ")
            + String(smax) + "] (the cache capacity, max_tokens), got "
            + String(s0) + "; the two sides of this boundary disagree about the state")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    comptime if discard_cache:
        if s0 != 0:
            raise Error("transformer: a discarded cache starts at cached_tokens 0")
    var ints = attn_step_ints(b, dm, nh, nkv, hd, it, smax, window, opts)
    _session_arm(session, a, ints, b, dm, nh, nkv, hd, it, smax, s0, window, opts)
    var result = session.arm.value().block_forward(a, ints, l, s0, discard_cache)
    _session_arm_budget(session)
    return result


# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (`TransformerForwardTape`, the standalone single-use forward owner, and
# `_transformer_saved_backward`, its backward over the saved workspace.)
# TOMBSTONE: MOJOLEARN_AUTOGRAD_TRANSFORMER(_OFF) (PROMOTED, hand arm deleted) deleted 2026-10-19 by dd4194a23; code recoverable at dd4194a23^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch


def _transformer_run_session_backward(
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, window: Int,
) raises:
    """`TransformerBlock.backward` on the block session (lane x4b): the
    zero-state prefill at positions [0, L) through the attention step op
    (the block's arm, `smax = L`, stages kept), then the transformer stage
    rule (`llama_decoder_layer_backward_autograd`) over the op's kept
    stages, weights, rotary table and input; the backward stages and the
    grad_output buffer kept by the arm. The ten gradients are written into
    the caller's buffers as before. `a` is the 21 backward addresses."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    if window < 0:
        raise Error("transformer backward: window must be >= 0")
    var opts = BlockOptions()
    var ints = attn_step_ints(b, dm, nh, nkv, hd, it, l, window, opts)
    # The arm opens on the forward layout: x and the nine weights, no caches.
    var fa = List[Int](length=13 + BLOCK_OPTION_ADDRS, fill=0)
    for i in range(10):  # small-loop(10: x and the nine base weight slots): argument words, no data
        fa[i] = a[i]
    _session_arm(session, fa, ints, b, dm, nh, nkv, hd, it, l, 0, window, opts)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_TRANSFORMER(_OFF) (PROMOTED, hand arm deleted) deleted 2026-10-19 by dd4194a23; code recoverable at dd4194a23^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch
    session.arm.value().block_backward(a, ints, l)
    _session_arm_budget(session)


# TOMBSTONE: transformer_attention_profile, transformer_session_info, transformer_decode_session_info and the backward timing entries (dead native entries: no live caller) deleted 2026-10-26 by 79853ceba; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_transformer.patch


# ===========================================================================
# THE PROFILE ENTRY (lane/lowbit-blocks, 2026-09-29): numeric_profile="fixed15_v1"
#
# `transformer_session_forward_int15(session, addrs, params)`. The float32
# entries above keep their lists word for word; this one is separate.
#   addrs, exactly 45: the 13 of `transformer_forward` with the seven
#     projection slots (3..9) ZERO (never read), then the 11-address options
#     tail (always sent), then 21 plane addresses, three per projection in
#     the order q, k, v, o, gate, up, down: `hi` (int8, rows x cols), `lo`
#     (int8, rows x cols), `exponents` (int32, rows). An ungated MLP sends
#     zeros for gate's three.
#   params, exactly 27: the 10 of `transformer_forward`, then the 17-int
#     options tail (always sent).
# The planes are uploaded on the session's first call and kept (the session
# struct's docstring says why that is sound here). Norms, biases, the cache
# and x are read per call as in the float32 entries.
# ===========================================================================

comptime INT15_ENTRY_ADDRS = 13 + BLOCK_OPTION_ADDRS + 21

comptime INT15_ENTRY_PARAMS = 10 + BLOCK_OPTION_PARAMS


def _upload_i8(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int8]:
    var dev = ctx.enqueue_create_buffer[DType.int8](n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=i8_ptr(addr))
    return dev^


def _upload_i32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var dev = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=i32_ptr(addr))
    return dev^


def _upload_planes_one(
    ctx: DeviceContext, a: List[Int], at: Int, rows: Int, cols: Int, name: String
) raises -> Int15Planes:
    if a[at] == 0 or a[at + 1] == 0 or a[at + 2] == 0:
        raise Error("transformer_session_forward_int15: " + name + " planes have a null address")
    return Int15Planes(
        _upload_i8(ctx, a[at], rows * cols), _upload_i8(ctx, a[at + 1], rows * cols),
        _upload_i32(ctx, a[at + 2], rows), rows, cols,
    )


def _upload_planes(
    ctx: DeviceContext, dims: LlamaDims, opts: BlockOptions, a: List[Int], base: Int
) raises -> LlamaInt15Weights:
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var gated = opts.gated()
    var gate: Int15Planes
    if gated:
        gate = _upload_planes_one(ctx, a, base + 12, it, dm, "gate_proj")
    else:
        if a[base + 12] != 0 or a[base + 13] != 0 or a[base + 14] != 0:
            raise Error("transformer_session_forward_int15: gate_proj planes were passed but the MLP is ungated")
        gate = Int15Planes(
            ctx.enqueue_create_buffer[DType.int8](1), ctx.enqueue_create_buffer[DType.int8](1),
            ctx.enqueue_create_buffer[DType.int32](1), 1, 1,
        )
    var out = LlamaInt15Weights(
        _upload_planes_one(ctx, a, base + 0, qw, dm, "q_proj"),
        _upload_planes_one(ctx, a, base + 3, kw, dm, "k_proj"),
        _upload_planes_one(ctx, a, base + 6, kw, dm, "v_proj"),
        _upload_planes_one(ctx, a, base + 9, dm, qw, "o_proj"),
        gate^,
        _upload_planes_one(ctx, a, base + 15, it, dm, "up_proj"),
        _upload_planes_one(ctx, a, base + 18, dm, it, "down_proj"),
        gated,
    )
    ctx.synchronize()
    out.check_shapes(dm, qw, kw, it)
    return out^


def _transformer_run_session_int15(
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`TransformerBlock.forward` under numeric_profile="fixed15_v1" on the
    attention step op (lane x45c; x4b moved the decode sessions): the
    profile's weights (norms, biases and the seven projection planes) built
    once per weight key into an adopted step-op entry, the caller's caches
    loaded at s0, the op, y and the caches back, one wait. The former body
    ran its own workspace and re-uploaded the norms and biases every call."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(String("transformer: cached_tokens must be in [0, ")
            + String(smax) + "] (the cache capacity, max_tokens), got "
            + String(s0) + "; the two sides of this boundary disagree about the state")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    for i in range(3, 10):  # small-loop(7: the float32 projection slots): address checks, no data
        if a[i] != 0:
            raise Error(
                "transformer_session_forward_int15: float32 projection slot "
                + String(i) + " must be 0 under the profile (the planes are the weights)"
            )
    if not session.ctx:
        session.ctx = neural_ctx[_NEURAL_CTX]()
        session.contexts += 1
    ref ctx = session.ctx.value()
    var pbase = 13 + BLOCK_OPTION_ADDRS
    var t = _tail_of(a, 13)
    var addrs: List[Int] = [b, smax, window, dm, nh, nkv, hd, it, Int(opts.gated()), a[1], a[2]]
    for i in range(len(t)):  # small-loop(t: the 11 optional weight addresses): key words, no data
        addrs.append(t[i])
    for i in range(21):  # small-loop(21: the seven plane triples): key words, no data
        addrs.append(a[pbase + i])
    if session.arm15 and session.key15.current(addrs):
        session.arm15.value().load(a[10], a[11], s0)
    else:
        session.arm15 = None
        session.key15 = WeightKey()
        var key = WeightKey.take(addrs)
        var what = String("transformer_session_forward_int15")
        var planes = _upload_planes(ctx, dims, opts, a, pbase)
        var w = LlamaDeviceWeights(
            ctx, dims, opts,
            _upload_addr(ctx, a[1], dm),
            _upload_addr(ctx, a[2], dm),
            planes^,
            _optional_upload(ctx, t[0], dims.q_width(), opts.qkv_bias, "q_proj.bias", what),
            _optional_upload(ctx, t[1], dims.kv_width(), opts.qkv_bias, "k_proj.bias", what),
            _optional_upload(ctx, t[2], dims.kv_width(), opts.qkv_bias, "v_proj.bias", what),
            _optional_upload(ctx, t[3], dm, opts.o_bias, "o_proj.bias", what),
            _optional_upload(ctx, t[4], dm, opts.norm_bias, "input_layernorm.bias", what),
            _optional_upload(ctx, t[5], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
            _optional_upload(ctx, t[6], it, opts.mlp_bias, "up_proj.bias", what),
            _optional_upload(ctx, t[7], dm, opts.mlp_bias, "down_proj.bias", what),
            _optional_upload(ctx, t[8], it, opts.has_gate_bias(), "gate_proj.bias", what),
            _optional_upload(ctx, t[9], hd, opts.qk_norm, "q_norm.weight", what),
            _optional_upload(ctx, t[10], hd, opts.qk_norm, "k_norm.weight", what),
        )
        var lean = _lean_for_w(hd, w)
        var ints = attn_step_ints(b, dm, nh, nkv, hd, it, smax, window, opts)
        var words = List[Int]()
        for i in range(8, len(ints)):  # small-loop(ints: the 17 block option words): argument words, no data
            words.append(ints[i])
        session.arm15 = AttnStepArm(ctx, w^, b, dm, nh, nkv, hd, it, smax, s0, window, words, a[10], a[11], lean)
        session.key15 = key^
    var cached = session.arm15.value().run(a[0], a[12], l)
    _ = session.arm15.value().export(a[10], a[11])
    return cached


# TOMBSTONE: the fixed15_v1 block forward's own workspace body (REFACTORED: onto the attention step op, an adopted entry) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/CAUSAL_GENERATE_RESIDENT_READS.patch; record in docs/TOMBSTONES.md.


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

    var ctx = neural_ctx[_NEURAL_CTX]()
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    # Host weights THROUGH the lane's own struct (its length table is the
    # reference shape authority, and its constructor is where the weights
    # are refused non-finite ONCE, DEVIATION 1875); values arrive as
    # given bits, unjudged.
    var w = _load_transformer_weights[False](ctx, dims, a, opts, _tail_of(a, 13))
    # The caller's cache over the fresh zeros. LlamaKVCache's own
    # constructor refuses b <= 0, smax <= 0 and smax > max_positions (the
    # absolute-position ceiling, DEVIATION 812's 8192 at the default
    # record, DEVIATION 2933) BY NAME before the uploads below. Zeros in
    # with s0 == 0 IS a fresh sequence; anything else is a carried one,
    # packed at stride s0 (DEVIATION 795(ii)).
    _btick(ton, tk, "surface.weights_up")
    var kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
    comptime if not discard_cache:
        kv.k = _upload_addr[False](ctx, a[10], cache_n)
        kv.v = _upload_addr[False](ctx, a[11], cache_n)
    kv.s = s0
    # Per call, from the record (DEVIATION 795(iii) at the default record:
    # the FROZEN theta). p_max is the cache capacity: pos0 + l <= kv.s_max
    # <= p_max holds for every legal call, so the table always covers the
    # absolute positions used.
    var rope = LlamaRopeTable(ctx, dims, opts, smax)
    var stages = LlamaDeviceStages(
        ctx, b, l, smax, dims, window, lean=_lean_for(hd, opts)
    )
    var dx = _upload_addr[False](ctx, a[0], b * l * dm)
    _btick(ton, tk, "surface.cache_stages_x_up")

    var trace = IdentityTrace.disabled()
    # pos0 = s0: DEVIATION 1028 (cache slot j IS absolute position j)
    # refuses any other value, and this surface has no other value to
    # offer. Prefill, chunked continuation and decode are all this one
    # call; capacity growth past smax is refused by name downstream.
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, w, dx, b, l, s0, trace, String("py"),
        forward_only=True,
    )

    # The block output is residual2 (contract section 2's
    # `residual2 + mlp(...)`, stage `residual2.out`), and the cache goes
    # back to its owner whole -- the full capacity buffer, so the bytes
    # round-trip exactly whatever the used stride is.
    _btick(ton, tk, "surface.forward")
    # One wait for the three downloads (lane/neural-apple, 2026-09-28): the
    # owners stay alive through it.
    _download_addr[False](ctx, stages.residual2, b * l * dm, a[12])
    comptime if not discard_cache:
        _download_addr[False](ctx, kv.k, cache_n, a[10])
        _download_addr[False](ctx, kv.v, cache_n, a[11])
    ctx.synchronize()
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
    written inside the call that names it.

    Lanes x4 / x4b: every session runs on the attention step op
    (transformer/attn_step_session.mojo `AttnStepArm`): the public decode
    session, the CausalLM layer sessions and the fixed15_v1 sessions (the
    profile's planes handed to the op as an adopted entry). The session's
    own block structs (weights, cache, rotary table, L = 1 stages, x) and
    the bodies over them are deleted."""

    var ctx: Optional[DeviceContext]
    # TOMBSTONE: MOJOLEARN_IDN_DECODE_RESIDENT_IO (session arm SUPERSEDED, never measured) deleted 2026-11-09 by lane x4-transformer-on-executor; code recoverable at f59d913b0b.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DECODE_RESIDENT_IO_TRANSFORMER.patch
    # TOMBSTONE: TRANSFORMER_DECODE_SESSION_STRUCTS (the session-owned block structs and their step / decode / forward bodies, SUPERSEDED by the attention step op) deleted 2026-11-09 by lane x4b-transformer-block-and-generate; code recoverable at c0b21c971c.
    # Restore: git apply experiments/removed/TRANSFORMER_DECODE_SESSION_STRUCTS.patch
    var op: Optional[AttnStepArm]
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
        self.op = Optional[AttnStepArm]()
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
        return Bool(self.ctx) and Bool(self.op)

    def cached(self) -> Int:
        """The position the session holds (the arm's mirror of the op's
        state word)."""
        if self.op:
            return self.op.value().cached
        return 0

    def release(mut self):
        """Buffers die before their context, and the frees they enqueue
        drain before the context goes (DEVIATION 2520's ordering)."""
        self.op = None
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


def _session_dims(
    mut s: TransformerDecodeSession, b: Int, dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, smax: Int,
    window: Int,
):
    s.b = b
    s.dm = dm
    s.nh = nh
    s.nkv = nkv
    s.hd = hd
    s.it = it
    s.smax = smax
    s.window = window


def _session_open_run(
    mut s: TransformerDecodeSession, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int,
    hd: Int, it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """The GIL-free half of `transformer_decode_session_open` (and of
    `transformer_decode_session_open_step_op`, the public door's name):
    the weights packed into the attention step op's one weight buffer, the
    caches into its one state buffer and the position into its state word
    (transformer/attn_step_session.mojo). `a` is the `transformer_forward`
    13 + 11 layout with slots 0 and 12 unused. Every later step, forward,
    decode, export and load runs on the op's operands."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(
            String("transformer decode session: cached_tokens must be in [0, ") + String(smax)
            + "], got " + String(s0))
    var ints = attn_step_ints(b, dm, nh, nkv, hd, it, smax, window, opts)
    var words = List[Int]()
    for i in range(8, len(ints)):  # small-loop(ints: the 17 block option words): argument words, no data
        words.append(ints[i])
    s.ctx = neural_ctx[_NEURAL_CTX]()
    s.op = AttnStepArm(s.ctx.value(), a, b, dm, nh, nkv, hd, it, smax, s0, window, opts, words)
    _session_dims(s, b, dm, nh, nkv, hd, it, smax, window)
    return s0


def _session_open_run_int15(
    mut s: TransformerDecodeSession, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int,
    hd: Int, it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`_session_open_run` under numeric_profile="fixed15_v1"
    (lane/lowbit-default): the weights are the profile constructor's, with
    the seven projections as the planes `a` carries after the options tail
    (`_upload_planes`, the layout of `transformer_session_forward_int15`),
    uploaded ONCE here. Lane x4b: they run on the attention step op as an
    adopted entry (`attn_step_adopt`), the caches in the op's state buffer,
    stages never lean over planes (`_lean_for_w`)."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(
            String("transformer decode session: cached_tokens must be in [0, ") + String(smax)
            + "], got " + String(s0))
    if window < 0:
        raise Error("transformer decode session: window must be >= 0 (0 = full causal)")
    for i in range(3, 10):
        if a[i] != 0:
            raise Error(
                "transformer_decode_session_open_int15: float32 projection slot "
                + String(i) + " must be 0 under the profile (the planes are the weights)"
            )
    s.ctx = neural_ctx[_NEURAL_CTX]()
    ref ctx = s.ctx.value()
    var what = String("transformer_decode_session_open_int15")
    var t = _tail_of(a, 13)
    var planes = _upload_planes(ctx, dims, opts, a, 13 + BLOCK_OPTION_ADDRS)
    var w = LlamaDeviceWeights(
        ctx, dims, opts,
        _upload_addr(ctx, a[1], dm),
        _upload_addr(ctx, a[2], dm),
        planes^,
        _optional_upload(ctx, t[0], dims.q_width(), opts.qkv_bias, "q_proj.bias", what),
        _optional_upload(ctx, t[1], dims.kv_width(), opts.qkv_bias, "k_proj.bias", what),
        _optional_upload(ctx, t[2], dims.kv_width(), opts.qkv_bias, "v_proj.bias", what),
        _optional_upload(ctx, t[3], dm, opts.o_bias, "o_proj.bias", what),
        _optional_upload(ctx, t[4], dm, opts.norm_bias, "input_layernorm.bias", what),
        _optional_upload(ctx, t[5], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
        _optional_upload(ctx, t[6], it, opts.mlp_bias, "up_proj.bias", what),
        _optional_upload(ctx, t[7], dm, opts.mlp_bias, "down_proj.bias", what),
        _optional_upload(ctx, t[8], it, opts.has_gate_bias(), "gate_proj.bias", what),
        _optional_upload(ctx, t[9], hd, opts.qk_norm, "q_norm.weight", what),
        _optional_upload(ctx, t[10], hd, opts.qk_norm, "k_norm.weight", what),
    )
    var lean = _lean_for_w(hd, w)
    var ints = attn_step_ints(b, dm, nh, nkv, hd, it, smax, window, opts)
    var words = List[Int]()
    for i in range(8, len(ints)):  # small-loop(ints: the 17 block option words): argument words, no data
        words.append(ints[i])
    s.op = AttnStepArm(ctx, w^, b, dm, nh, nkv, hd, it, smax, s0, window, words, a[10], a[11], lean)
    _session_dims(s, b, dm, nh, nkv, hd, it, smax, window)
    return s0


def _session_step_run(mut s: TransformerDecodeSession, px: Int, py: Int) raises -> Int:
    """One decode token on the step op. ONE completion wait; the caller's x
    stays live for the call."""
    return s.op.value().run(px, py, 1)


def _session_decode_run(mut s: TransformerDecodeSession, px: Int, py: Int, steps: Int) raises -> Int:
    """N decode tokens per row in one call (decode plan D2) on the step op
    (`AttnStepArm.decode`): x up once, the op at L = 1 per token with token
    t moved on the device, y down once behind one wait."""
    # TOMBSTONE: TRANSFORMER_DECODE_SESSION_STRUCTS (the fixed15_v1 sessions' own per-token decode loop over the session structs, SUPERSEDED: they decode on the op) deleted 2026-11-09 by lane x4b-transformer-block-and-generate; code recoverable at c0b21c971c.
    # Restore: git apply experiments/removed/TRANSFORMER_DECODE_SESSION_STRUCTS.patch
    return s.op.value().decode(px, py, steps)


def _session_forward_run(mut s: TransformerDecodeSession, px: Int, py: Int, l: Int) raises -> Int:
    """`l` tokens per row (a prefill or a chunked continuation) on the step
    op, the arm's kept L stages."""
    return s.op.value().run(px, py, l)


def _session_export_run(mut s: TransformerDecodeSession, pk: Int, pv: Int) raises -> Int:
    return s.op.value().export(pk, pv)


def _session_load_run(mut s: TransformerDecodeSession, pk: Int, pv: Int, s0: Int) raises -> Int:
    if s0 < 0 or s0 > s.smax:
        raise Error(
            String("transformer decode session: cached_tokens must be in [0, ") + String(s.smax)
            + "], got " + String(s0))
    return s.op.value().load(pk, pv, s0)


# ===========================================================================
# The block backward: a zero-state prefill's VJP, IDENTICAL tier only.
# ===========================================================================


# ===========================================================================
# THE RESIDENT CAUSAL LM DECODE SESSION (lane/py-lm, 2026-09-28)
#
# `CausalLM.generate` ran one binding call per layer per token, and every one
# of them uploaded that layer's weights and its whole KV cache and downloaded
# the cache again; the embedding table and the head crossed every token too
# (`_mojolearn_training`'s embedding_forward / linear_forward), and the full
# logits came down for a Python argmax. This session holds the embedding,
# the final norm weight and the head on the device, and one call runs a
# whole stack of open `TransformerDecodeSession`s (DEVIATION 2940: weights,
# KV cache, rotary table and L = 1 stages already resident) for a prompt and
# then a block of greedy tokens, returning the ids.
#
# NO NEW ARITHMETIC. Per token: `identical_embedding_forward_into` (what
# `samba_embedding_forward_host` runs), each layer's
# `llama_decoder_layer_forward` exactly as `_session_step_run` /
# `_session_forward_run` run it, `llama_rms_norm` over the same M rows
# (`samba_rms_norm_forward_host`), `identical_gemm_into` at OP_NT over the
# same M, N, K (`samba_linear_forward_host`; the plan is chosen from M, so M
# is kept: B * L at the prompt, B per decode token). The greedy pick is the
# host scan `models/causal_lm.py::_argmax_last` spells: over the last
# position's row, from index 0, a strict `>` (ties keep the lowest index, a
# NaN never replaces, a NaN at index 0 stays). The refusals of the per-call
# route that can fire mid-stack are restated on downloaded copies: a
# non-finite final block output (the norm's input scan) and a non-finite
# normed row (the head's input scan), in their words.
# ===========================================================================


struct CausalLMSession(Movable, Writable):
    var ctx: Optional[DeviceContext]
    var embed: Optional[DeviceBuffer[DType.float32]]
    var norm: Optional[DeviceBuffer[DType.float32]]
    var head: Optional[DeviceBuffer[DType.float32]]
    # lane/lowbit-default: the head under numeric_profile="fixed15_v1", its
    # planes resident, and the profile's per-call scratch
    var head15: Optional[Int15Planes]
    var st15: Optional[LlamaInt15Stage]
    var tied: Bool
    var b: Int
    var v: Int
    var d: Int
    var eps: Float32
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.embed = Optional[DeviceBuffer[DType.float32]]()
        self.norm = Optional[DeviceBuffer[DType.float32]]()
        self.head = Optional[DeviceBuffer[DType.float32]]()
        self.head15 = Optional[Int15Planes]()
        self.st15 = Optional[LlamaInt15Stage]()
        self.tied = False
        self.b = 0
        self.v = 0
        self.d = 0
        self.eps = Float32(0.0)
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CausalLMSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("CausalLMSession")

    def is_open(self) -> Bool:
        return Bool(self.ctx) and Bool(self.embed)

    def release(mut self):
        self.st15 = None
        self.head15 = None
        self.head = None
        self.norm = None
        self.embed = None
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        self.ctx = None

    def __deinit__(deinit self):
        self.release()


def _lm_refuse_nonfinite_device(
    ctx: DeviceContext, name: String, mut buf: DeviceBuffer[DType.float32], n: Int,
) raises:
    """The non-finite refusal on the device copy (cpu2-l11-neural): one
    `device_first_nonfinite` scan, one integer read back, the same message."""
    var bad = device_first_nonfinite(ctx, buf, n)
    if bad >= 0:
        raise Error("mojolearn samba ops: non-finite " + name + " at flat index " + String(bad))


comptime _LM_ARGMAX_TPB = 256
"""Threads per row of the greedy pick: one block per batch row."""

comptime _LM_ARGMAX_NONE: Int32 = 2147483647

comptime LM_RESIDENT_GENERATE = is_defined["MOJOLEARN_IDN_LM_RESIDENT_GENERATE"]()
"""Experiment, default off (lane lm-resident-generate-20261028): the resident
session also opens for a model without a final norm (`final_norm=False`,
norm address 0). Its head then reads the stack output itself, as the
per-layer route's head does (`hn` is a device copy of `x`): no float
operation changes. The B arm is today's behavior: such models are refused
here and take the per-layer route, one stage allocation, upload and download
per layer per token."""


@always_inline
def _lm_order_key(value: Float32) -> UInt32:
    """An unsigned key whose order is the float order (by bits, never a
    float compare: Metal flushes compare operands). -0.0 is keyed as +0.0,
    so equal floats get equal keys and the host loop's `>` is matched for
    every non-NaN value, subnormals included."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7FFFFFFF)) == UInt32(0):
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


def _lm_last_argmax_kernel(
    logits: MutPointer[Float32, MutAnyOrigin],
    next_ids: MutPointer[Int32, MutAnyOrigin],
    out_ids: MutPointer[Int32, MutAnyOrigin],
    cur_l_in: Int32,
    v_in: Int32,
    n_steps_in: Int32,
    step_in: Int32,
):
    """cpu2-l11-neural (2026-10-04): the greedy pick of `_lm_run` on the
    device. Block `r` reads row r's LAST position of `logits` ([b, cur_l,
    v]) and writes the first (lowest-index) maximum to `next_ids[r]` (the
    next step's input id) and `out_ids[r * n_steps + step]`. The host loop
    it replaces kept the lowest index of the maximum under `>`; an integer
    (key, index) minimum-index reduction picks the same index for every
    non-NaN row, on every vendor and at any block size."""
    var keys = stack_allocation[
        _LM_ARGMAX_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var idxs = stack_allocation[
        _LM_ARGMAX_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var v = Int(v_in)
    var cur_l = Int(cur_l_in)
    var base = (r * cur_l + cur_l - 1) * v
    var best_k = UInt32(0)
    var best_i = _LM_ARGMAX_NONE
    var c = tid
    while c < v:
        var k = _lm_order_key(logits.unsafe_load(base + c))
        if best_i == _LM_ARGMAX_NONE or k > best_k:
            best_k = k
            best_i = Int32(c)
        c += _LM_ARGMAX_TPB
    keys.unsafe_store(tid, best_k)
    idxs.unsafe_store(tid, best_i)
    barrier()
    var active = _LM_ARGMAX_TPB // 2
    while active > 0:
        if tid < active:
            var oi = idxs.unsafe_load(tid + active)
            var ok = keys.unsafe_load(tid + active)
            var mi = idxs.unsafe_load(tid)
            var mk = keys.unsafe_load(tid)
            if oi != _LM_ARGMAX_NONE and (mi == _LM_ARGMAX_NONE or ok > mk or (ok == mk and oi < mi)):
                keys.unsafe_store(tid, ok)
                idxs.unsafe_store(tid, oi)
        barrier()
        active = active // 2
    if tid == 0:
        var best = idxs.unsafe_load(0)
        next_ids.unsafe_store(r, best)
        out_ids.unsafe_store(r * Int(n_steps_in) + Int(step_in), best)


# TOMBSTONE: _lm_read_argmax_kernel / _lm_read_status_kernel / _lm_read_sample_kernel / _lm_read_progress / _LMSamplingCheckpoint / _lm_sampling_outputs (REFACTORED: the pick rule is the session op OPS_LM_PICK, autograd/generation_ops.mojo; reads run as the decode program) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/CAUSAL_GENERATE_RESIDENT_READS.patch; record in docs/TOMBSTONES.md.


def _lm_open_run(mut s: CausalLMSession, pe: Int, pn: Int, ph: Int, b: Int, v: Int, d: Int, eps: Float32) raises:
    if b < 1 or v < 1 or d < 1:
        raise Error("causal_lm_session_open: B, vocab and d_model must be positive")
    var has_norm = pn != 0
    comptime if not LM_RESIDENT_GENERATE:
        if not has_norm:
            raise Error("causal_lm_session_open: null buffer address")
    if has_norm and (not isfinite(eps) or eps < Float32(0.0)):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")
    s.ctx = neural_ctx[_NEURAL_CTX]()
    s.embed = _upload_addr(s.ctx.value(), pe, v * d)
    if has_norm:
        s.norm = _upload_addr(s.ctx.value(), pn, d)
    s.tied = ph == 0
    if not s.tied:
        s.head = _upload_addr(s.ctx.value(), ph, v * d)
    # cpu2-l11-neural (2026-10-04): the weight refusals scan the uploaded
    # copies on the device (same first flat index, same words); the entry
    # releases the session on the raise.
    _lm_refuse_nonfinite_device(s.ctx.value(), "embedding weight", s.embed.value(), v * d)
    if has_norm:
        _lm_refuse_nonfinite_device(s.ctx.value(), "rms_norm weight", s.norm.value(), d)
    if not s.tied:
        _lm_refuse_nonfinite_device(s.ctx.value(), "linear weight", s.head.value(), v * d)
    s.ctx.value().synchronize()
    s.b = b
    s.v = v
    s.d = d
    s.eps = eps


def _lm_open_run_int15(
    mut s: CausalLMSession, pe: Int, pn: Int, planes: List[Int], b: Int, v: Int, d: Int, eps: Float32,
) raises:
    """`_lm_open_run` under numeric_profile="fixed15_v1" (lane/lowbit-default):
    the embedding and the final norm as there, and the head as its planes
    (`planes` = [hi, lo, exponents] of the head, tied or not, as
    `CausalLM` made them with `mojolearn.linalg.quantize_int15`), resident.
    The head product is then `llama_int15_proj`: the parallel quantizer on
    the normed rows and `identical_gemm_int15_planes_into`, the calls
    `mojolearn.linalg.matmul_int15` makes on the per-layer route."""
    if b < 1 or v < 1 or d < 1:
        raise Error("causal_lm_session_open: B, vocab and d_model must be positive")
    var has_norm = pn != 0
    comptime if not LM_RESIDENT_GENERATE:
        if not has_norm:
            raise Error("causal_lm_session_open_int15: null buffer address")
    if has_norm and (not isfinite(eps) or eps < Float32(0.0)):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")
    s.ctx = neural_ctx[_NEURAL_CTX]()
    ref ctx = s.ctx.value()
    s.embed = _upload_addr(ctx, pe, v * d)
    if has_norm:
        s.norm = _upload_addr(ctx, pn, d)
    # cpu2-l11-neural: device refusals of the uploaded copies (see `_lm_open_run`).
    _lm_refuse_nonfinite_device(ctx, "embedding weight", s.embed.value(), v * d)
    if has_norm:
        _lm_refuse_nonfinite_device(ctx, "rms_norm weight", s.norm.value(), d)
    s.tied = False
    var a: List[Int] = [planes[0], planes[1], planes[2]]
    s.head15 = _upload_planes_one(ctx, a, 0, v, d, "head")
    s.st15 = LlamaInt15Stage(ctx)
    ctx.synchronize()
    s.b = b
    s.v = v
    s.d = d
    s.eps = eps


def _lm_scratch_buffer(ctx: DeviceContext, mut buffer: Optional[DeviceBuffer[DType.float32]], n: Int) raises:
    # Exact extents keep whole-buffer copies and checked kernel layouts honest.
    # Callers replace storage only after the preceding phase synchronized.
    if not buffer or len(buffer.value()) != n:
        buffer = ctx.enqueue_create_buffer[DType.float32](n)


struct _LMRunWorkspace(Movable):
    """Private scratch for one resident read; no output or input alias escapes.
    Each producer overwrites its full logical output, as with fresh allocation.
    Prompt-to-decode changes replace exact-sized buffers once; later tokens
    reuse them only after the existing end-of-step synchronization.
    """
    var x: Optional[DeviceBuffer[DType.float32]]
    var sumsq: Optional[DeviceBuffer[DType.float32]]
    var hn: Optional[DeviceBuffer[DType.float32]]
    var logits: Optional[DeviceBuffer[DType.float32]]
    var ws: Optional[DeviceBuffer[DType.float32]]
    var ws_rows: Int
    var ws_vocab: Int
    var ws_width: Int
    var ws_packed: Bool

    def __init__(out self):
        self.x = None
        self.sumsq = None
        self.hn = None
        self.logits = None
        self.ws = None
        self.ws_rows = -1
        self.ws_vocab = 0
        self.ws_width = 0
        self.ws_packed = False

    def prepare_gemm(mut self, ctx: DeviceContext, rows: Int, vocab: Int, width: Int, packed: Bool) raises:
        if not self.ws or self.ws_rows != rows or self.ws_vocab != vocab or self.ws_width != width or self.ws_packed != packed:
            var n = 1
            if not packed:
                n = identical_gemm_workspace_max_floats(rows, vocab, width)
            _lm_scratch_buffer(ctx, self.ws, n)
            self.ws_rows = rows
            self.ws_vocab = vocab
            self.ws_width = width
            self.ws_packed = packed


def _lm_layers_run(
    ctx: DeviceContext, layers: List[Int], mut x: DeviceBuffer[DType.float32], b: Int, l: Int,
) raises:
    """Every layer in order on `x` ([b * l * d], overwritten with the stack's
    output), each through the attention step op (lane x4b: the layer
    session's `AttnStepArm.run_device`): at l = 1 on the op's resident L = 1
    stages, otherwise on one operation-owned prefill workspace (`shared`)
    reused between compatible sequential layers, as before."""
    _ = b
    var shared = AttnPrefill()
    try:
        for i in range(len(layers)):  # driver-loop: over decoder layers of the model
            var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=layers[i])
            sp[].op.value().run_device(x, l, shared)
            if l > 1:
                ctx.synchronize()
    except original:
        # Drain while the shared stages stay live.
        try:
            ctx.synchronize()
        except:
            pass
        raise original^
    finally:
        _ = shared


def _lm_run(
    mut s: CausalLMSession, layers: List[Int], p_ids: Int, p_out: Int, p_logits: Int,
    l: Int, n_steps: Int,
) raises:
    """`n_steps` greedy tokens of the resident stack (`CausalLM._generate_resident`):
    the embedding, every layer, the norm, the head and the pick on the
    device, the ids fed back on the device, the [b, n_steps] ids (and, when
    asked, the last step's last-position logits) down once. Lane x45c: the
    stream reads (greedy and sampled, with EOS rows) moved to the decode
    program (transformer/causal_lm_program.mojo); this fixed-length loop
    stays for the private resident entry only."""
    ref ctx = s.ctx.value()
    var b = s.b
    var d = s.d
    var v = s.v
    var emb_cfg = EmbConfig.llama(v, d)
    var ids_i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=p_ids)
    var out_i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=p_out)
    var ids0 = ctx.enqueue_create_buffer[DType.int32](b * l)
    ctx.enqueue_copy(dst_buf=ids0, src_ptr=ids_i32)
    var next_ids = ctx.enqueue_create_buffer[DType.int32](b)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](b * n_steps)
    var logits_rows = List[DeviceBuffer[DType.float32]]()
    var workspace = _LMRunWorkspace()
    var finite_checks = Optional[DeviceNonfinitePair]()
    try:
        var cur_l = l
        for step in range(n_steps):
            var m = b * cur_l
            _lm_scratch_buffer(ctx, workspace.x, m * d)
            ref x = workspace.x.value()
            if step == 0:
                identical_embedding_forward_into(ctx, x, s.embed.value(), ids0, m, emb_cfg)
            else:
                identical_embedding_forward_into(ctx, x, s.embed.value(), next_ids, m, emb_cfg)
            _lm_layers_run(ctx, layers, x, b, cur_l)
            _lm_scratch_buffer(ctx, workspace.sumsq, m)
            _lm_scratch_buffer(ctx, workspace.hn, m * d)
            ref sumsq = workspace.sumsq.value()
            ref hn = workspace.hn.value()
            if s.norm:
                llama_rms_norm(ctx, sumsq, hn, x, s.norm.value(), m, d, s.eps)
            else:
                ctx.enqueue_copy(dst_buf=hn, src_buf=x)
            _lm_scratch_buffer(ctx, workspace.logits, m * v)
            workspace.prepare_gemm(ctx, m, v, d, Bool(s.head15))
            ref logits = workspace.logits.value()
            ref ws = workspace.ws.value()
            if s.head15:
                llama_int15_proj(ctx, s.st15.value(), logits, hn, s.head15.value(), m, v, d)
            elif s.tied:
                identical_gemm_into[ROLE=ROLE_HEAD](ctx, logits, hn, s.embed.value(), ws, m, v, d, OP_NT)
            else:
                identical_gemm_into[ROLE=ROLE_HEAD](ctx, logits, hn, s.head.value(), ws, m, v, d, OP_NT)
            if not finite_checks:
                finite_checks = DeviceNonfinitePair(ctx, max(b * l * d, b * d), Bool(s.norm))
            var bad = finite_checks.value().first_nonfinite(ctx, x, hn, m * d)
            if bad[0] >= 0:
                raise Error("mojolearn samba ops: non-finite rms_norm input at flat index " + String(bad[0]))
            if bad[1] >= 0:
                raise Error("mojolearn samba ops: non-finite linear input at flat index " + String(bad[1]))
            ctx.enqueue_function[_lm_last_argmax_kernel](
                logits.unsafe_ptr(), next_ids.unsafe_ptr(), out_dev.unsafe_ptr(),
                Int32(cur_l), Int32(v), Int32(n_steps), Int32(step),
                grid_dim=(b, 1, 1), block_dim=(_LM_ARGMAX_TPB, 1, 1),
            )
            if p_logits != 0 and step == n_steps - 1:
                var lp = _f32_ptr(p_logits)
                for r in range(b):
                    logits_rows.append(logits.create_sub_buffer[DType.float32]((r * cur_l + cur_l - 1) * v, v))
                    ctx.enqueue_copy(dst_ptr=lp + r * v, src_buf=logits_rows[r])
            ctx.synchronize()
            logits_rows.clear()
            cur_l = 1
        ctx.enqueue_copy(dst_ptr=out_i32, src_buf=out_dev)
        ctx.synchronize()
    except original:
        try:
            ctx.synchronize()
        except:
            pass
        raise original
    finally:
        _ = finite_checks
        _ = workspace
        _ = logits_rows
        _ = ids0^
        _ = next_ids^
        _ = out_dev^


def _lm_forward_run(
    mut s: CausalLMSession, layers: List[Int], p_ids: Int, p_logits: Int, l: Int,
) raises:
    """`CausalLM.forward` as ONE resident program (lane migrate-neural-2):
    the embedding, every layer on its session's cache, the norm and the head
    over all B * L positions on the device, then the (B, L, V) logits come
    down once. It is `_lm_run`'s first pass (the same kernels in the same
    order) with every position's logits kept instead of the greedy pick;
    before, each stage was its own call with host arrays between them."""
    ref ctx = s.ctx.value()
    var b = s.b
    var d = s.d
    var v = s.v
    var m = b * l
    var emb_cfg = EmbConfig.llama(v, d)
    var ids0 = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_copy(dst_buf=ids0, src_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=p_ids))
    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    identical_embedding_forward_into(ctx, x, s.embed.value(), ids0, m, emb_cfg)
    _lm_layers_run(ctx, layers, x, b, l)
    var sumsq = ctx.enqueue_create_buffer[DType.float32](m)
    var hn = ctx.enqueue_create_buffer[DType.float32](m * d)
    if s.norm:
        llama_rms_norm(ctx, sumsq, hn, x, s.norm.value(), m, d, s.eps)
    else:
        # No final norm (LM_RESIDENT_GENERATE): the head reads the stack
        # output, as the per-layer route's head does; a copy, no arithmetic.
        ctx.enqueue_copy(dst_buf=hn, src_buf=x)
    var logits = ctx.enqueue_create_buffer[DType.float32](m * v)
    var ws_n = 1
    if not s.head15:
        ws_n = identical_gemm_workspace_max_floats(m, v, d)
    var ws = ctx.enqueue_create_buffer[DType.float32](ws_n)
    if s.head15:
        llama_int15_proj(ctx, s.st15.value(), logits, hn, s.head15.value(), m, v, d)
    elif s.tied:
        identical_gemm_into[ROLE=ROLE_HEAD](ctx, logits, hn, s.embed.value(), ws, m, v, d, OP_NT)
    else:
        identical_gemm_into[ROLE=ROLE_HEAD](ctx, logits, hn, s.head.value(), ws, m, v, d, OP_NT)
    if s.norm:
        # Without a norm only the head's input scan applies (on `hn`, the
        # copy of `x`), as on the per-layer route.
        var bad_x = device_first_nonfinite(ctx, x, m * d)
        if bad_x >= 0:
            raise Error("mojolearn samba ops: non-finite rms_norm input at flat index " + String(bad_x))
    var bad_hn = device_first_nonfinite(ctx, hn, m * d)
    if bad_hn >= 0:
        raise Error("mojolearn samba ops: non-finite linear input at flat index " + String(bad_hn))
    ctx.enqueue_copy(dst_ptr=_f32_ptr(p_logits), src_buf=logits)
    ctx.synchronize()
    _ = ids0^
    _ = x^
    _ = sumsq^
    _ = hn^
    _ = logits^
    _ = ws^


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
    var ctx = neural_ctx[_NEURAL_CTX]()
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


# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (`_session_forward_tape_run` / `_session_backward_tape_run`, the budget-aware
# session tape over training/neural_attention_owner.mojo.)
