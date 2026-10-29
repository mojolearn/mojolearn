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
from std.memory import bitcast, memcpy
from std.os import getenv
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN

from core.identity_trace import IdentityTrace
from core.resident_generation_read import ResidentGreedyRead
from core.device_scan import device_first_nonfinite
from core.device_nonfinite_pair import DeviceNonfinitePair
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.neural_identical_experiments import IDN_SAMBA_FORWARD_TAPE
from core.neural_context import neural_ctx
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojoNeuralTransformerContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTransformerContextFast"
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
    IDN_DECODE_RESIDENT_IO,
    LlamaDeferredChecks,
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



# ===========================================================================
# The block: profile mojolearn.identical.transformer.fp32.v1
# ===========================================================================


def transformer_lean_stages(hd: Int) -> Bool:
    """Whether this call may skip the `[B, n_heads, L, S]` attention stage
    allocations: the fused attention path will be attempted (FAST or
    IDENTICAL build, supported head_dim, not forced eager) and the trace is off here
    always. The eager fallback grows the buffers on demand."""
    if not fused_forward_supported_head_dim(hd):
        return False
    return attention_path_choice(PLANT_AT_NONE) != ATTN_PATH_EAGER



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



def _lean_for(hd: Int, opts: BlockOptions) -> Bool:
    """`transformer_lean_stages`, and never lean under a softcap, which
    forces the eager kernels (DEVIATION 2947); the eager fallback would
    grow a lean struct anyway, so this is a cost choice, not a correctness
    one."""
    if opts.has_softcap():
        return False
    return transformer_lean_stages(hd)



def _lean_for_w(hd: Int, w: LlamaDeviceWeights) -> Bool:
    """`_lean_for` for weights that may carry the profile's planes: never
    lean under `fixed15_v1`, as `_transformer_run_session_int15` builds its
    stages (lane/lowbit-default: the resident sessions under the profile)."""
    if w.int15:
        return False
    return _lean_for(hd, w.opts)



def _workspace_key(
    b: Int, l: Int, dims: LlamaDims, smax: Int, window: Int, lean: Bool,
    opts: BlockOptions,
) -> List[Int]:
    """The retained workspace's identity: the shape, and the whole options
    record (a different theta or rope_dim is a different rotary table)."""
    var key: List[Int] = [b, l, dims.d_model, dims.n_heads, dims.n_kv,
                          dims.head_dim, dims.intermediate, smax, window, Int(lean)]
    var tail = opts.to_params()
    for i in range(len(tail)):  # small-loop(tail: the 17 block option params): builds a cache key, not data
        key.append(tail[i])
    return key^



struct TransformerWorkspace(Movable):
    var key: List[Int]
    var kv: LlamaKVCache
    var rope: LlamaRopeTable
    var stages: LlamaDeviceStages
    var x: DeviceBuffer[DType.float32]
    #: MOJOLEARN_AFN_ATTN_ARENA (FAST + Apple): the arena the workspace's
    #: buffers are views of; -1 (no arena) on every other build.
    var arena_id: Int

    def __init__(out self, ctx: DeviceContext, dims: LlamaDims,
                 b: Int, l: Int, smax: Int, window: Int, lean: Bool,
                 opts: BlockOptions) raises:
        self.key = _workspace_key(b, l, dims, smax, window, lean, opts)
        self.arena_id = -1
        comptime if AFN_ATTN_ARENA:
            # lane afn-attn: every buffer below (cache, rope, ~30 stages, x)
            # is a sub-buffer view of the arena's chunks (core/device_arena),
            # so a Metal launch pays for a few live allocations, not ~35.
            # `_zeros`/`_upload` take the views while the arena is active.
            self.arena_id = arena_begin()
        self.kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
        # The table from the record: DEVIATIONS 2930-2933 and 2948; the
        # default record is `LlamaRopeTable(ctx, dims, ROPE_THETA, smax)`.
        self.rope = LlamaRopeTable(ctx, dims, opts, smax)
        self.stages = LlamaDeviceStages(ctx, b, l, smax, dims, window, lean=lean)
        comptime if AFN_ATTN_ARENA:
            self.x = _llama_zeros(ctx, b * l * dims.d_model)
            arena_end(self.arena_id)
        else:
            self.x = ctx.enqueue_create_buffer[DType.float32](b * l * dims.d_model)

    def __deinit__(deinit self):
        """MOJOLEARN_AFN_ATTN_ARENA: the arena's chunks go back to the pool
        (not freed; the views in this struct's fields die right after this
        body). Nothing on every other build."""
        comptime if AFN_ATTN_ARENA:
            try:
                arena_release(self.arena_id)
            except:
                pass

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
        for i in range(len(key)):  # small-loop(key: shape and option entries of the workspace key): compares cache keys, not data
            if key[i] != self.key[i]:
                return False
        return True



struct TransformerBackwardWorkspace(Movable):
    """The retained half of `transformer_session_backward`: the backward
    stages for one shape and the device copy of grad_output. The forward
    half is the session's `TransformerWorkspace` (stages, cache, rope, x)
    at `smax = L`."""
    var key: List[Int]
    var bst: LlamaBackwardStages
    var d_out: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, dims: LlamaDims, b: Int, l: Int,
                 window: Int, lean: Bool) raises:
        self.key = [b, l, dims.d_model, dims.n_heads, dims.n_kv, dims.head_dim,
                    dims.intermediate, window, Int(lean)]
        self.bst = LlamaBackwardStages(ctx, b, l, l, dims, lean=lean)
        self.d_out = ctx.enqueue_create_buffer[DType.float32](b * l * dims.d_model)

    def matches(self, key: List[Int]) -> Bool:
        if len(key) != len(self.key):
            return False
        for i in range(len(key)):  # small-loop(key: nine shape entries of the backward key): compares cache keys, not data
            if key[i] != self.key[i]:
                return False
        return True


from training.neural_attention_owner import BorrowedAttentionTape
from transformer.experiments.profile import transformer_arithmetic_profile, transformer_arithmetic_profile_changed



struct TransformerSession(Movable, Writable):
    """One Python-owned context/workspace; no retained host pointers or
    float32 weights. Under `numeric_profile="fixed15_v1"` it RETAINS the
    block's seven projection planes on the device (`planes`, keyed by
    their addresses and the shape): those arrays are private to the Python
    block, made once at its construction and never exposed or written, so
    unlike a caller's float32 array their address is their version."""
    var ctx: Optional[DeviceContext]
    var workspace: Optional[TransformerWorkspace]
    var planes: Optional[LlamaInt15Weights]
    var planes_key: List[Int]
    var weights: Optional[LlamaDeviceWeights]
    """lane/neural-net-experiment (2026-09-30): the block's float32 weights
    RETAINED on the device across calls. A Python address is still not a
    version: `weights_host` keeps the exact bytes that were uploaded, and
    every call compares the caller's arrays against them bit for bit
    (`_weights_same_bits`) before reusing the device copy. Changed bytes are
    copied into the SAME device buffers (no reallocation) and revalidated by
    the constructor's own `_validate_finite`; a different shape or options
    record rebuilds. Default-record blocks only: an options block (`extended`)
    keeps the per-call upload."""
    var weights_key: List[Int]
    var weights_host: List[Float32]
    var weight_uploads: Int
    var weight_recopies: Int
    var weight_reuses: Int
    var backward: Optional[TransformerBackwardWorkspace]
    """lane/neural-net-experiment: the zero-state prefill backward's stages
    and its grad_output buffer, retained across `transformer_session_backward`
    calls the way the byte trainer retains `tr.backward[layer]` across steps
    (no reset exists or is needed: every cell the backward reads it wrote
    this call). Dropped with the forward workspace under the same budget."""
    var backward_workspaces: Int
    var tape: Optional[BorrowedAttentionTape]
    var tape_generation: Int
    var busy: Bool
    var closed: Bool
    var contexts: Int
    var workspaces: Int

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.workspace = Optional[TransformerWorkspace]()
        self.planes = Optional[LlamaInt15Weights]()
        self.planes_key = List[Int]()
        self.weights = Optional[LlamaDeviceWeights]()
        self.weights_key = List[Int]()
        self.weights_host = List[Float32]()
        self.weight_uploads = 0
        self.weight_recopies = 0
        self.weight_reuses = 0
        self.backward = Optional[TransformerBackwardWorkspace]()
        self.backward_workspaces = 0
        self.tape = None
        self.tape_generation = 1
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
        _ = self.tape^
        _ = self.backward^
        _ = self.workspace^
        _ = self.planes^
        _ = self.weights^
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def invalidate_tape(mut self) raises:
        # Ordinary calls with no explicit tape retain their previous launch
        # and synchronization schedule; only live tape owners need fences.
        if self.tape:
            if self.ctx:
                self.ctx.value().synchronize()
            self.tape = None
            if self.ctx:
                self.ctx.value().synchronize()
        self.tape_generation += 1

    def clear(mut self) raises:
        self.invalidate_tape()
        if self.ctx:
            self.ctx.value().synchronize()
        self.backward = None
        self.workspace = None
        self.planes = None
        self.planes_key = List[Int]()
        self.weights = None
        self.weights_key = List[Int]()
        self.weights_host = List[Float32]()
        if self.ctx:
            self.ctx.value().synchronize()



def _weights_key(dims: LlamaDims, opts: BlockOptions) -> List[Int]:
    """The retained weights' identity: the shape and the options record."""
    var key: List[Int] = [dims.d_model, dims.n_heads, dims.n_kv, dims.head_dim,
                          dims.intermediate]
    var tail = opts.to_params()
    for i in range(len(tail)):  # small-loop(tail: the 17 block option params): builds a cache key, not data
        key.append(tail[i])
    return key^



def _base_weight_lens(dims: LlamaDims) -> List[Int]:
    """Element counts of the nine base weights in slot order 1..9."""
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var lens: List[Int] = [dm, dm, qw * dm, kw * dm, kw * dm, dm * qw,
                           it * dm, it * dm, dm * it]
    return lens^



def _weights_same_bits(a: List[Int], host: List[Float32], lens: List[Int]) raises -> Bool:
    """Whether the caller's nine base arrays hold exactly the bytes in
    `host` (their concatenation in slot order): an integer compare of the
    bit patterns, eight lanes at a time (`-0.0` and `+0.0` differ, NaN
    payloads differ, as they must for a version check)."""
    comptime W = 8
    var off = 0
    for s in range(9):
        var n = lens[s]
        var p = _f32_ptr(a[s + 1])
        var hp = host.unsafe_ptr()
        var i = 0
        var body = n - n % W
        while i < body:
            var x = bitcast[DType.uint32, W](p.unsafe_load[width=W](i))
            var y = bitcast[DType.uint32, W](hp.unsafe_load[width=W](off + i))
            if (x ^ y).reduce_or() != UInt32(0):
                return False
            i += W
        while i < n:
            if bitcast[DType.uint32](p.unsafe_load(i)) != bitcast[DType.uint32](hp.unsafe_load(off + i)):
                return False
            i += 1
        off += n
    return True



def _weights_host_copy(a: List[Int], lens: List[Int]) raises -> List[Float32]:
    """The nine base arrays' bytes, concatenated in slot order."""
    var total = 0
    for s in range(9):
        total += lens[s]
    var out = List[Float32](length=total, fill=Float32(0))
    var off = 0
    for s in range(9):
        var n = lens[s]
        if n > 0:
            memcpy(dest=out.unsafe_ptr() + off, src=_f32_ptr(a[s + 1]), count=n)
        off += n
    return out^



def _weights_recopy(ctx: DeviceContext, mut w: LlamaDeviceWeights, a: List[Int], lens: List[Int]) raises:
    """Changed caller bytes into the RETAINED device buffers, slot order,
    one wait, then the constructor's own refusal of non-finite weights
    (DEVIATION 1875's order and names)."""
    ctx.enqueue_copy(dst_buf=w.norm1_w, src_ptr=_f32_ptr(a[1]))
    ctx.enqueue_copy(dst_buf=w.norm2_w, src_ptr=_f32_ptr(a[2]))
    ctx.enqueue_copy(dst_buf=w.w_q, src_ptr=_f32_ptr(a[3]))
    ctx.enqueue_copy(dst_buf=w.w_k, src_ptr=_f32_ptr(a[4]))
    ctx.enqueue_copy(dst_buf=w.w_v, src_ptr=_f32_ptr(a[5]))
    ctx.enqueue_copy(dst_buf=w.w_o, src_ptr=_f32_ptr(a[6]))
    ctx.enqueue_copy(dst_buf=w.w_gate, src_ptr=_f32_ptr(a[7]))
    ctx.enqueue_copy(dst_buf=w.w_up, src_ptr=_f32_ptr(a[8]))
    ctx.enqueue_copy(dst_buf=w.w_down, src_ptr=_f32_ptr(a[9]))
    ctx.synchronize()
    _ = lens
    w._validate_finite(ctx)



def _stage_reset_on() -> Bool:
    """`MOJOLEARN_TRANSFORMER_STAGE_RESET=0` skips the thirty zero-fills a
    reused workspace gets before each call. Bit-safe only if every cell a
    call reads it also wrote this call; the identity gate decides."""
    return String(getenv("MOJOLEARN_TRANSFORMER_STAGE_RESET")) != "0"



def _session_retain_cap_bytes() -> Int:
    """Retained workspace budget per session: `MOJOLEARN_TRANSFORMER_RETAIN_MB`
    (an integer, MiB), default 512. Was a fixed 64 MiB, which the board's
    block shape (B1 L2048 DM384 FF1024: about 87 MiB of lean stages) exceeded
    on every call, so the whole workspace was rebuilt per call."""
    var cap_mb = 512
    var s = String(getenv("MOJOLEARN_TRANSFORMER_RETAIN_MB"))
    if s != "":
        try:
            cap_mb = Int(atol(s))
        except:
            cap_mb = 512
    if cap_mb < 0:
        cap_mb = 0
    return cap_mb * 1024 * 1024



def _session_weights(
    mut session: TransformerSession, dims: LlamaDims,
    a: List[Int], opts: BlockOptions, tail: List[Int],
) raises:
    """Leave `session.weights` holding this call's weights: the retained
    device copy when the caller's bytes equal the ones uploaded, a recopy
    into the retained buffers when they differ, a fresh upload otherwise.
    Options blocks (a non-default record or any tail address) take the
    per-call upload as before; nothing is retained for them."""
    session.invalidate_tape()
    ref ctx = session.ctx.value()
    var extended = not opts.is_default()
    for i in range(len(tail)):  # small-loop(tail: the 11 optional block addresses): checks address presence, not data
        if tail[i] != 0:
            extended = True
    # MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS=0: the per-call upload for every
    # block (the A/B against the byte compare; measured 0.3 ms per call on
    # an L40S, where the upload it replaces was already cheap).
    if String(getenv("MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS")) == "0":
        extended = True
    if extended:
        session.weights = None
        session.weights_key = List[Int]()
        session.weights_host = List[Float32]()
        session.weights = _load_transformer_weights(ctx, dims, a, opts, tail)
        session.weight_uploads += 1
        return
    var key = _weights_key(dims, opts)
    var lens = _base_weight_lens(dims)
    var same_key = False
    if session.weights:
        same_key = len(session.weights_key) == len(key)
        if same_key:
            for i in range(len(key)):  # small-loop(key: shape and option entries of the weights key): compares cache keys, not data
                if key[i] != session.weights_key[i]:
                    same_key = False
    if same_key:
        if _weights_same_bits(a, session.weights_host, lens):
            session.weight_reuses += 1
            return
        # Bytes moved: copy them into the retained buffers. A refusal below
        # leaves the session without weights (the next call uploads anew).
        var fresh = _weights_host_copy(a, lens)
        try:
            ref retained = session.weights.value()
            _weights_recopy(ctx, retained, a, lens)
        except error:
            session.weights = None
            session.weights_key = List[Int]()
            session.weights_host = List[Float32]()
            raise error
        session.weights_host = fresh^
        session.weight_recopies += 1
        return
    session.weights = None
    session.weights_key = List[Int]()
    session.weights_host = List[Float32]()
    var host = _weights_host_copy(a, lens)
    session.weights = _load_transformer_weights(ctx, dims, a, opts, tail)
    session.weights_key = key^
    session.weights_host = host^
    session.weight_uploads += 1



# TOMBSTONE: MOJOLEARN_IDN_ATTN_DEFER_REGIME (loser: transformer-forward 1.376) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch
# TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (loser: transformer-forward 1.350) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
def _transformer_run_session[discard_cache: Bool = False, retain_tape: Bool = False](
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`discard_cache` (lane/neural-net-experiment): the stateless prefill on
    the session -- the caller passes no cache (slots 10 and 11 are 0), the
    retained cache buffers are zero-filled on the device (the bytes a fresh
    `LlamaKVCache` holds), nothing is uploaded or downloaded for it, and
    `s0` is 0. Same kernels, same launches, same order."""
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
        session.ctx = neural_ctx[_NEURAL_CTX]()
        session.contexts += 1
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var lean = _lean_for(hd, opts)
    var key = _workspace_key(b, l, dims, smax, window, lean, opts)
    var reused = False
    if session.workspace:
        reused = session.workspace.value().matches(key)
    if not reused:
        session.ctx.value().synchronize()
        session.workspace = None
        session.ctx.value().synchronize()
    # Reread the caller's weights on EVERY call; a Python address is not a
    # version. The device copy is reused only when the bytes are the same
    # (`_session_weights`), recopied in place when they moved.
    _session_weights(session, dims, a, opts, _tail_of(a, 13))
    ref ctx = session.ctx.value()
    ref w = session.weights.value()
    _btick(ton, tk, "surface.weights_up")
    if not reused:
        session.workspace = TransformerWorkspace(ctx, dims, b, l, smax, window, lean, opts)
        session.workspaces += 1
    ref ws = session.workspace.value()
    if reused and _stage_reset_on():
        ws.stages.reset(ctx)
    comptime if discard_cache:
        if s0 != 0:
            raise Error("transformer: a discarded cache starts at cached_tokens 0")
        # The bytes a fresh `LlamaKVCache` holds, on the device, no upload.
        # TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (loser: transformer-forward 1.350) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
        ws.kv.k.enqueue_fill(Float32(0.0))
        ws.kv.v.enqueue_fill(Float32(0.0))
    else:
        # Host cache state is authoritative on EVERY call, including resets and
        # same-address edits. Copy into retained allocations, never trust a pointer.
        ctx.enqueue_copy(dst_buf=ws.kv.k, src_ptr=_f32_ptr(a[10]))
        ctx.enqueue_copy(dst_buf=ws.kv.v, src_ptr=_f32_ptr(a[11]))
    ctx.enqueue_copy(dst_buf=ws.x, src_ptr=_f32_ptr(a[0]))
    comptime if not AFN_ATTN_ARENA:
        ctx.synchronize()
    # MOJOLEARN_AFN_ATTN_ARENA: no wait here. The uploads and the forward's
    # launches are in order on the one stream, and the caller's host arrays
    # outlive the call (the download wait below drains everything).
    ws.kv.s = s0
    _btick(ton, tk, "surface.cache_stages_x_up")
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_DEFER_REGIME (loser: transformer-forward 1.376) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch
    var trace = IdentityTrace.disabled()
    # forward_only: no backward reads this entry's stages.
    llama_decoder_layer_forward(ctx, ws.stages, ws.kv, ws.rope, w, ws.x,
                                b, l, s0, trace, String("py"), forward_only=not retain_tape)
    _btick(ton, tk, "surface.forward")
    # One wait for the downloads (the fresh entry's pattern): every owner
    # stays alive through it.
    _download_addr[False](ctx, ws.stages.residual2, b * l * dm, a[12])
    comptime if not discard_cache:
        _download_addr[False](ctx, ws.kv.k, len(ws.kv.k), a[10])
        _download_addr[False](ctx, ws.kv.v, len(ws.kv.v), a[11])
    ctx.synchronize()
    _btick(ton, tk, "surface.outputs_down")
    var result = ws.kv.s
    # Bound retained device-buffer storage per model. Large legal calls still
    # run, but release their workspace after completion rather than pinning it.
    var retain = retain_tape or ws.retained_bytes() <= _session_retain_cap_bytes()
    if not retain:
        session.workspace = None
        ctx.synchronize()
    return result




struct TransformerForwardTape(Movable, Writable):
    """NI36/48 single-use owner of input, weight snapshots and saved stages.

    The session is private to this tape: no future model call can overwrite
    it, so pointer equality is never used as a content-generation guarantee.
    It is retained through backward or close, including calls over the normal
    inference cache-retention budget. Peak tape memory belongs to the caller.
    """
    var session: TransformerSession
    var shape: List[Int]
    var consumed: Bool
    var busy: Bool

    def __init__(out self, shape: List[Int]):
        self.session = TransformerSession()
        self.shape = shape.copy()
        self.consumed = False
        self.busy = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerForwardTape")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("TransformerForwardTape")

    def close(mut self) raises:
        self.consumed = True
        self.session.clear()
        self.session.ctx = None



def _transformer_saved_backward(mut tape: TransformerForwardTape, a: List[Int]) raises:
    ref p = tape.shape
    var b = p[0]
    var l = p[1]
    var dm = p[2]
    var dims = LlamaDims(dm, p[3], p[4], p[5], p[6])
    ref session = tape.session
    if not session.ctx or not session.workspace or not session.weights:
        raise Error("transformer forward tape: saved owner is unavailable")
    ref ctx = session.ctx.value()
    ref ws = session.workspace.value()
    ref w = session.weights.value()
    var bw = TransformerBackwardWorkspace(ctx, dims, b, l, p[7], transformer_lean_stages(p[5]))
    ctx.enqueue_copy(dst_buf=bw.d_out, src_ptr=_f32_ptr(a[0]))
    ctx.synchronize()
    var off = IdentityTrace.disabled()
    # Reuse the real saved forward; no weight upload or forward recompute.
    # The stage order and fan-in sums are owned by the autograd tape
    # (autograd/rules_transformer.mojo).
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_TRANSFORMER(_OFF) (PROMOTED, hand arm deleted) deleted 2026-10-19 by dd4194a23; code recoverable at dd4194a23^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch
    llama_decoder_layer_backward_autograd(ctx, bw.bst, ws.stages, w,
        ws.rope.cos, ws.rope.sin, ws.x, bw.d_out, b, l, 0, off, String("saved"))
    _download_addr[False](ctx, bw.bst.d_x, b * l * dm, a[1])
    _download_addr[False](ctx, bw.bst.dw_norm1, dm, a[2])
    _download_addr[False](ctx, bw.bst.dw_norm2, dm, a[3])
    _download_addr[False](ctx, bw.bst.dw_q, dims.q_width() * dm, a[4])
    _download_addr[False](ctx, bw.bst.dw_k, dims.kv_width() * dm, a[5])
    _download_addr[False](ctx, bw.bst.dw_v, dims.kv_width() * dm, a[6])
    _download_addr[False](ctx, bw.bst.dw_o, dm * dims.q_width(), a[7])
    _download_addr[False](ctx, bw.bst.dw_gate, p[6] * dm, a[8])
    _download_addr[False](ctx, bw.bst.dw_up, p[6] * dm, a[9])
    _download_addr[False](ctx, bw.bst.dw_down, dm * p[6], a[10])
    ctx.synchronize()



def _transformer_run_session_backward(
    mut session: TransformerSession, a: List[Int], b: Int, l: Int,
    dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, window: Int,
) raises:
    """`_transformer_backward_run` on the session (lane/neural-net-experiment,
    2026-09-30): the same forward from a zero cache at positions [0, L) and
    the same backward on its stages, with the weights retained
    (`_session_weights`), the forward workspace retained at `smax = L`, the
    backward stages and the grad_output buffer retained, and the cache
    zero-filled on the device. Same kernels, same launches, same order;
    the ten gradients are written into the caller's buffers as before."""
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    if window < 0:
        raise Error("transformer backward: window must be >= 0")
    var opts = BlockOptions()
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var m = b * l
    if not session.ctx:
        session.ctx = neural_ctx[_NEURAL_CTX]()
        session.contexts += 1
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var lean = transformer_lean_stages(hd)
    # Weights: reused when the bytes are the same, recopied when they moved.
    _session_weights(session, dims, a, opts, List[Int]())
    ref ctx = session.ctx.value()
    ref w = session.weights.value()
    _btick(ton, tk, "surface.weights_up")
    # The forward half at smax = L (the fresh-prefill shape).
    var key = _workspace_key(b, l, dims, l, window, lean, opts)
    var reused = False
    if session.workspace:
        reused = session.workspace.value().matches(key)
    if not reused:
        ctx.synchronize()
        session.workspace = None
        ctx.synchronize()
        session.workspace = TransformerWorkspace(ctx, dims, b, l, l, window, lean, opts)
        session.workspaces += 1
    ref ws = session.workspace.value()
    if reused and _stage_reset_on():
        ws.stages.reset(ctx)
    # The backward half.
    var bkey: List[Int] = [b, l, dm, nh, nkv, hd, it, window, Int(lean)]
    var breused = False
    if session.backward:
        breused = session.backward.value().matches(bkey)
    if not breused:
        ctx.synchronize()
        session.backward = None
        ctx.synchronize()
        session.backward = TransformerBackwardWorkspace(ctx, dims, b, l, window, lean)
        session.backward_workspaces += 1
    ref bw = session.backward.value()
    # Inputs: x and grad_output into retained buffers; the cache is the
    # fresh zeros a `LlamaKVCache` holds.
    ws.kv.k.enqueue_fill(Float32(0.0))
    ws.kv.v.enqueue_fill(Float32(0.0))
    ws.kv.s = 0
    ctx.enqueue_copy(dst_buf=ws.x, src_ptr=_f32_ptr(a[0]))
    ctx.enqueue_copy(dst_buf=bw.d_out, src_ptr=_f32_ptr(a[10]))
    ctx.synchronize()
    _btick(ton, tk, "surface.inputs_up")
    var off = IdentityTrace.disabled()
    llama_decoder_layer_forward(
        ctx, ws.stages, ws.kv, ws.rope, w, ws.x, b, l, 0, off, String("pyf")
    )
    _btick(ton, tk, "surface.forward")
    var offb = IdentityTrace.disabled()
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_TRANSFORMER(_OFF) (PROMOTED, hand arm deleted) deleted 2026-10-19 by dd4194a23; code recoverable at dd4194a23^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch
    llama_decoder_layer_backward_autograd(
        ctx, bw.bst, ws.stages, w, ws.rope.cos, ws.rope.sin, ws.x, bw.d_out,
        b, l, 0, offb, String("pyb"),
    )
    _btick(ton, tk, "surface.backward")
    _download_addr[False](ctx, bw.bst.d_x, m * dm, a[11])
    _download_addr[False](ctx, bw.bst.dw_norm1, dm, a[12])
    _download_addr[False](ctx, bw.bst.dw_norm2, dm, a[13])
    _download_addr[False](ctx, bw.bst.dw_q, qw * dm, a[14])
    _download_addr[False](ctx, bw.bst.dw_k, kw * dm, a[15])
    _download_addr[False](ctx, bw.bst.dw_v, kw * dm, a[16])
    _download_addr[False](ctx, bw.bst.dw_o, dm * qw, a[17])
    _download_addr[False](ctx, bw.bst.dw_gate, it * dm, a[18])
    _download_addr[False](ctx, bw.bst.dw_up, it * dm, a[19])
    _download_addr[False](ctx, bw.bst.dw_down, dm * it, a[20])
    ctx.synchronize()
    _btick(ton, tk, "surface.outputs_down")
    # The same budget as the forward path, on the forward half's count
    # (the backward half is of the same order); both go together.
    if ws.retained_bytes() > _session_retain_cap_bytes():
        session.backward = None
        session.workspace = None
        ctx.synchronize()


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
    """`_transformer_run_session` under the profile: the same stages, cache
    handling and downloads, with the weights' seven projections as the
    session's retained planes."""
    session.invalidate_tape()
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    opts.validate(hd)
    if s0 < 0 or s0 > smax:
        raise Error(String("transformer: cached_tokens must be in [0, ")
            + String(smax) + "] (the cache capacity, max_tokens), got "
            + String(s0) + "; the two sides of this boundary disagree about the state")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    for i in range(3, 10):
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
    var pkey: List[Int] = [dm, nh, nkv, hd, it, Int(opts.gated())]
    for i in range(21):
        pkey.append(a[pbase + i])
    var same = Bool(session.planes) and len(session.planes_key) == len(pkey)
    if same:
        for i in range(len(pkey)):  # small-loop(pkey: six shapes plus 21 plane addresses): compares cache keys, not data
            if pkey[i] != session.planes_key[i]:
                same = False
    if not same:
        ctx.synchronize()
        session.planes = None
        session.planes = Optional[LlamaInt15Weights](_upload_planes(ctx, dims, opts, a, pbase))
        session.planes_key = pkey^
    var lean = False
    var key = _workspace_key(b, l, dims, smax, window, lean, opts)
    var reused = False
    if session.workspace:
        reused = session.workspace.value().matches(key)
    if not reused:
        ctx.synchronize()
        session.workspace = None
        ctx.synchronize()
    var what = String("transformer_session_forward_int15")
    var t = _tail_of(a, 13)
    var w = LlamaDeviceWeights(
        ctx, dims, opts,
        _upload_addr(ctx, a[1], dm),
        _upload_addr(ctx, a[2], dm),
        session.planes.value().copy(),
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
    if not reused:
        session.workspace = TransformerWorkspace(ctx, dims, b, l, smax, window, lean, opts)
        session.workspaces += 1
    ref ws = session.workspace.value()
    if reused:
        ws.stages.reset(ctx)
    ctx.enqueue_copy(dst_buf=ws.kv.k, src_ptr=_f32_ptr(a[10]))
    ctx.enqueue_copy(dst_buf=ws.kv.v, src_ptr=_f32_ptr(a[11]))
    ctx.enqueue_copy(dst_buf=ws.x, src_ptr=_f32_ptr(a[0]))
    ctx.synchronize()
    ws.kv.s = s0
    var trace = IdentityTrace.disabled()
    llama_decoder_layer_forward(ctx, ws.stages, ws.kv, ws.rope, w, ws.x,
                                b, l, s0, trace, String("py"))
    _download_addr(ctx, ws.stages.residual2, b * l * dm, a[12])
    _download_addr(ctx, ws.kv.k, len(ws.kv.k), a[10])
    _download_addr(ctx, ws.kv.v, len(ws.kv.v), a[11])
    var result = ws.kv.s
    var retain = ws.retained_bytes() <= 64 * 1024 * 1024
    _ = w^
    ctx.synchronize()
    if not retain:
        session.workspace = None
        ctx.synchronize()
    return result



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
    written inside the call that names it."""

    var ctx: Optional[DeviceContext]
    var w: Optional[LlamaDeviceWeights]
    var kv: Optional[LlamaKVCache]
    var rope: Optional[LlamaRopeTable]
    var stages: Optional[LlamaDeviceStages]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var defer: Optional[LlamaDeferredChecks]
    """IDN_DECODE_RESIDENT_IO: the step's deferred check window; None on a
    build without the define (nothing allocated)."""
    var snap_k: Optional[DeviceBuffer[DType.float32]]
    """IDN_DECODE_RESIDENT_IO: the key cache as it stood before the current
    step, restored when the step replays."""
    var snap_v: Optional[DeviceBuffer[DType.float32]]
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
        self.defer = Optional[LlamaDeferredChecks]()
        self.snap_k = Optional[DeviceBuffer[DType.float32]]()
        self.snap_v = Optional[DeviceBuffer[DType.float32]]()
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
        self.defer = None
        self.snap_k = None
        self.snap_v = None
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
    s.ctx = neural_ctx[_NEURAL_CTX]()
    s.w = _load_transformer_weights(s.ctx.value(), dims, a, opts, _tail_of(a, 13))
    var kv = LlamaKVCache(s.ctx.value(), b, dims, smax, window, opts.max_positions)
    kv.k = _upload_addr(s.ctx.value(), a[10], cache_n)
    kv.v = _upload_addr(s.ctx.value(), a[11], cache_n)
    kv.s = s0
    s.kv = kv^
    s.rope = LlamaRopeTable(s.ctx.value(), dims, opts, smax)
    s.stages = LlamaDeviceStages(s.ctx.value(), b, 1, smax, dims, window, lean=_lean_for_w(hd, s.w.value()))
    s.dx = s.ctx.value().enqueue_create_buffer[DType.float32](b * dm)
    comptime if IDN_DECODE_RESIDENT_IO:
        _session_resident_io_open(s, b, nh, smax)
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



def _session_open_run_int15(
    mut s: TransformerDecodeSession, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int,
    hd: Int, it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
) raises -> Int:
    """`_session_open_run` under numeric_profile="fixed15_v1"
    (lane/lowbit-default): the weights are the profile constructor's, with
    the seven projections as the planes `a` carries after the options tail
    (`_upload_planes`, the layout of `transformer_session_forward_int15`),
    uploaded ONCE here and resident for the session's life. Every later
    step and forward runs `llama_decoder_layer_forward` on them exactly as
    `_transformer_run_session_int15` does, so the bytes are the per-call
    profile entry's."""
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
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd
    s.ctx = neural_ctx[_NEURAL_CTX]()
    ref ctx = s.ctx.value()
    var what = String("transformer_decode_session_open_int15")
    var t = _tail_of(a, 13)
    var planes = _upload_planes(ctx, dims, opts, a, 13 + BLOCK_OPTION_ADDRS)
    s.w = LlamaDeviceWeights(
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
    var kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
    kv.k = _upload_addr(ctx, a[10], cache_n)
    kv.v = _upload_addr(ctx, a[11], cache_n)
    kv.s = s0
    s.kv = kv^
    s.rope = LlamaRopeTable(ctx, dims, opts, smax)
    s.stages = LlamaDeviceStages(ctx, b, 1, smax, dims, window, lean=False)
    s.dx = ctx.enqueue_create_buffer[DType.float32](b * dm)
    comptime if IDN_DECODE_RESIDENT_IO:
        _session_resident_io_open(s, b, nh, smax)
    ctx.synchronize()
    s.b = b
    s.dm = dm
    s.nh = nh
    s.nkv = nkv
    s.hd = hd
    s.it = it
    s.smax = smax
    s.window = window
    return s0



def _session_resident_io_open(mut s: TransformerDecodeSession, b: Int, nh: Int, smax: Int) raises:
    """IDN_DECODE_RESIDENT_IO, at session open (no wait here; the open's
    own wait follows): the deferred check window, the two cache snapshot
    buffers, and the fused forward's exp stash (`stages.aexp`) sized for
    the session's longest L = 1 call, `b * n_heads * smax` cells, so the
    `_estash` forward never grows it (a growth is an allocation and a wait
    per token, the stash growing by one key a step). Sizing is capacity
    only: the forward writes the same cells either way."""
    ref ctx = s.ctx.value()
    s.defer = LlamaDeferredChecks(ctx, True, True)
    s.snap_k = ctx.enqueue_create_buffer[DType.float32](len(s.kv.value().k))
    s.snap_v = ctx.enqueue_create_buffer[DType.float32](len(s.kv.value().v))
    var cells = b * nh * smax
    if len(s.stages.value().aexp) < cells:
        s.stages.value().aexp = ctx.enqueue_create_buffer[DType.float32](cells)
        s.stages.value().aexp.enqueue_fill(Float32(0.0))



def _session_step_run(mut s: TransformerDecodeSession, px: Int, py: Int) raises -> Int:
    """One decode token on the resident structs: x in, the block output
    out, the cache advanced on the device. ONE completion wait, after the
    output copy is enqueued; the caller's x stays live for the call."""
    comptime if IDN_DECODE_RESIDENT_IO:
        if s.defer and s.snap_k and s.snap_v:
            return _session_step_run_deferred(s, px, py)
    return _session_step_run_sync(s, px, py)



def _session_step_run_deferred(mut s: TransformerDecodeSession, px: Int, py: Int) raises -> Int:
    """IDN_DECODE_RESIDENT_IO (lane decode-resident, 2026-10-28): the step
    with ONE wait. Enqueued in order: x in, a device copy of both caches
    (the snapshot), the block with the session's deferred window armed (its
    call refusal scans and the fused forward's regime scan and corner flag
    go into the window instead of being read), y out, the window's words
    home; then the one wait. A clean window means the same kernels ran on
    the same inputs as `_session_step_run_sync`, so y and the cache hold its
    bytes. Otherwise (a refusal hit, a refused regime, a corner, or any
    error raised inside the armed call) the snapshot goes back, the position
    goes back, and the step runs again through `_session_step_run_sync`,
    which raises today's message or takes today's eager path. Arithmetic is
    untouched on every path: no bit moves."""
    ref ctx = s.ctx.value()
    var n = s.b * s.dm
    if (n != len(s.stages.value().residual2)
            or len(s.snap_k.value()) != len(s.kv.value().k)
            or len(s.snap_v.value()) != len(s.kv.value().v)):
        return _session_step_run_sync(s, px, py)
    var pos0 = s.kv.value().s
    ctx.enqueue_copy(dst_buf=s.dx.value(), src_ptr=_f32_ptr(px))
    ctx.enqueue_copy(dst_buf=s.snap_k.value(), src_buf=s.kv.value().k)
    ctx.enqueue_copy(dst_buf=s.snap_v.value(), src_buf=s.kv.value().v)
    s.defer.value().begin()
    var addr = Int(MutPointer(to=s.defer.value()).unsafe_origin_cast[MutAnyOrigin]())
    var clean = False
    try:
        s.stages.value().defer_addr = addr
        var trace = IdentityTrace.disabled()
        llama_decoder_layer_forward(
            ctx, s.stages.value(), s.kv.value(), s.rope.value(), s.w.value(), s.dx.value(),
            s.b, 1, pos0, trace, String("py.session"))
        s.stages.value().defer_addr = 0
        ctx.enqueue_copy(dst_ptr=_f32_ptr(py), src_buf=s.stages.value().residual2)
        s.defer.value().enqueue_readback(ctx)
        ctx.synchronize()
        clean = s.defer.value().clean_after_wait()
    except:
        # Drain what the armed call enqueued before the snapshot goes back.
        s.stages.value().defer_addr = 0
        ctx.synchronize()
        clean = False
    if clean:
        return s.kv.value().s
    ctx.enqueue_copy(dst_buf=s.kv.value().k, src_buf=s.snap_k.value())
    ctx.enqueue_copy(dst_buf=s.kv.value().v, src_buf=s.snap_v.value())
    s.kv.value().s = pos0
    return _session_step_run_sync(s, px, py)



def _session_step_run_sync(mut s: TransformerDecodeSession, px: Int, py: Int) raises -> Int:
    """`_session_step_run` as shipped: every check read where it runs."""
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
    var stages = LlamaDeviceStages(ctx, s.b, l, s.smax, dims, s.window, lean=_lean_for_w(s.hd, s.w.value()))
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




def _lm_read_argmax_kernel(
    logits: MutPointer[Float32, MutAnyOrigin], next_ids: MutPointer[Int32, MutAnyOrigin],
    out_ids: MutPointer[Int32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    batch_in: Int32, cur_l_in: Int32, vocab_in: Int32, capacity_in: Int32,
    step_in: Int32, pad: Int32,
):
    # One block per row: stable lowest-index maximum, first nonfinite refusal,
    # next input stays on device. Progress commits only after the next kernel
    # knows the first failing row, matching source-ordered failure publication.
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var batch = Int(batch_in)
    var cur_l = Int(cur_l_in)
    var vocab = Int(vocab_in)
    var capacity = Int(capacity_in)
    var step = Int(step_in)
    if status[r] != 0:
        if tid == 0:
            next_ids[r] = pad
            out_ids[r * capacity + step] = pad
            status[batch + r] = -1
        return
    var keys = stack_allocation[_LM_ARGMAX_TPB, UInt32, address_space=AddressSpace.SHARED]()
    var indices = stack_allocation[_LM_ARGMAX_TPB, Int32, address_space=AddressSpace.SHARED]()
    var bad = stack_allocation[_LM_ARGMAX_TPB, Int32, address_space=AddressSpace.SHARED]()
    var best_key = UInt32(0)
    var best_index = _LM_ARGMAX_NONE
    var first_bad = _LM_ARGMAX_NONE
    var base = (r * cur_l + cur_l - 1) * vocab
    var c = tid
    while c < vocab:
        var value = logits[base + c]
        if (bitcast[DType.uint32](value) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            first_bad = min(first_bad, Int32(c))
        var key = _lm_order_key(value)
        if best_index == _LM_ARGMAX_NONE or key > best_key:
            best_key = key
            best_index = Int32(c)
        c += _LM_ARGMAX_TPB
    keys[tid] = best_key
    indices[tid] = best_index
    bad[tid] = first_bad
    barrier()
    var active = _LM_ARGMAX_TPB // 2
    while active > 0:
        if tid < active:
            var oi = indices[tid + active]
            var ok = keys[tid + active]
            var mi = indices[tid]
            var mk = keys[tid]
            if oi != _LM_ARGMAX_NONE and (mi == _LM_ARGMAX_NONE or ok > mk or (ok == mk and oi < mi)):
                keys[tid] = ok
                indices[tid] = oi
            bad[tid] = min(bad[tid], bad[tid + active])
        barrier()
        active //= 2
    if tid == 0:
        status[batch + r] = -1 if bad[0] == _LM_ARGMAX_NONE else bad[0]
        if bad[0] != _LM_ARGMAX_NONE:
            return
        var pick = indices[0]
        next_ids[r] = pick
        out_ids[r * capacity + step] = pick




def _lm_read_status_kernel(status: MutPointer[Int32, MutAnyOrigin],
                           summary: MutPointer[Int32, MutAnyOrigin],
                           lengths: MutPointer[Int64, MutAnyOrigin],
                           next_ids: MutPointer[Int32, MutAnyOrigin],
                           eos: MutPointer[Int32, MutAnyOrigin], batch_in: Int32, n_eos_in: Int32):
    # First locate the first invalid active row. Then commit only earlier
    # rows (all rows when valid), exactly as the source-ordered picker did.
    # This uses the existing second launch and two-word control transfer.
    var batch = Int(batch_in)
    var tid = Int(thread_idx.x)
    var rows = stack_allocation[_LM_ARGMAX_TPB, Int32, address_space=AddressSpace.SHARED]()
    var pending = stack_allocation[_LM_ARGMAX_TPB, Int32, address_space=AddressSpace.SHARED]()
    var first_bad = _LM_ARGMAX_NONE
    var r = tid
    while r < batch:
        if status[batch + r] >= 0:
            first_bad = min(first_bad, Int32(r))
        r += _LM_ARGMAX_TPB
    rows[tid] = first_bad
    barrier()
    var width = _LM_ARGMAX_TPB // 2
    while width > 0:
        if tid < width:
            rows[tid] = min(rows[tid], rows[tid + width])
        barrier()
        width //= 2
    var stop = Int(rows[0])
    var active = Int32(0)
    r = tid
    while r < batch:
        if status[r] == 0 and r < stop:
            lengths[r] += 1
            var pick = next_ids[r]
            for i in range(Int(n_eos_in)):
                if pick == eos[i]:
                    status[r] = 1
        if status[r] == 0:
            active = 1
        r += _LM_ARGMAX_TPB
    pending[tid] = active
    barrier()
    width = _LM_ARGMAX_TPB // 2
    while width > 0:
        if tid < width:
            pending[tid] = max(pending[tid], pending[tid + width])
        barrier()
        width //= 2
    if tid == 0:
        summary[0] = pending[0]
        summary[1] = -1 if rows[0] == _LM_ARGMAX_NONE else status[batch + stop]



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
    output): at l = 1 on the session's own L = 1 stages and resident x buffer
    (`_session_step_run`), otherwise on call-shaped stages
    (`_session_forward_run`)."""
    var prefill_input = Optional[DeviceBuffer[DType.float32]]()
    var prefill_stages = Optional[LlamaDeviceStages]()
    try:
        for i in range(len(layers)):
            var sp = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=layers[i])
            var trace = IdentityTrace.disabled()
            var pos0 = sp[].kv.value().s
            if l == 1:
                ctx.enqueue_copy(dst_buf=sp[].dx.value(), src_buf=x)
                llama_decoder_layer_forward(
                    ctx, sp[].stages.value(), sp[].kv.value(), sp[].rope.value(), sp[].w.value(),
                    sp[].dx.value(), sp[].b, 1, pos0, trace, String("py.session"))
                ctx.enqueue_copy(dst_buf=x, src_buf=sp[].stages.value().residual2)
            else:
                var dims = LlamaDims(sp[].dm, sp[].nh, sp[].nkv, sp[].hd, sp[].it)
                prefill_stages = LlamaDeviceStages(ctx, sp[].b, l, sp[].smax, dims, sp[].window,
                                                  lean=_lean_for_w(sp[].hd, sp[].w.value()))
                ref stages = prefill_stages.value()
                _lm_scratch_buffer(ctx, prefill_input, b * l * sp[].dm)
                ref dx = prefill_input.value()
                ctx.enqueue_copy(dst_buf=dx, src_buf=x)
                llama_decoder_layer_forward(
                    ctx, stages, sp[].kv.value(), sp[].rope.value(), sp[].w.value(), dx, sp[].b, l,
                    pos0, trace, String("py.session.forward"))
                ctx.enqueue_copy(dst_buf=x, src_buf=stages.residual2)
                ctx.synchronize()
                prefill_stages = None
    except original:
        # Drain while the reused copy and any partially used stages stay live.
        try:
            ctx.synchronize()
        except:
            pass
        raise original^
    finally:
        _ = prefill_stages
        _ = prefill_input



def _lm_read_progress(ctx: DeviceContext, mut read: ResidentGreedyRead,
                      lengths: DeviceBuffer[DType.int64], finished: DeviceBuffer[DType.int32]) raises:
    # One normal or failure-only metadata publication. A failed copy must not
    # expose partially refreshed lists as a coherent progress snapshot.
    read.progress_ready = False
    ctx.enqueue_copy(dst_ptr=read.lengths.unsafe_ptr(), src_buf=lengths)
    ctx.enqueue_copy(dst_ptr=read.finished.unsafe_ptr(), src_buf=finished)
    ctx.synchronize()
    read.progress_ready = True


def _lm_run(
    mut s: CausalLMSession, layers: List[Int], p_ids: Int, p_out: Int, p_logits: Int,
    l: Int, n_steps: Int, read_address: Int = 0,
) raises:
    ref ctx = s.ctx.value()
    var b = s.b
    var d = s.d
    var v = s.v
    var emb_cfg = EmbConfig.llama(v, d)
    var ids_i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=p_ids)
    var out_i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=p_out)
    # cpu2-l11-neural (2026-10-04): the greedy pick runs on the device
    # (`_lm_last_argmax_kernel`), and its ids feed the next step from
    # `next_ids` without a host round trip; the [b, n_steps] ids and (when
    # asked) the last step's last-position logits come down once, at the end.
    var ids0 = ctx.enqueue_create_buffer[DType.int32](b * l)
    ctx.enqueue_copy(dst_buf=ids0, src_ptr=ids_i32)
    var next_ids = ctx.enqueue_create_buffer[DType.int32](b)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](b * n_steps)
    var read = MutPointer[ResidentGreedyRead, MutUntrackedOrigin](unsafe_from_address=read_address)
    var status = List[Int32]()
    var status_device = Optional[DeviceBuffer[DType.int32]]()
    var finished_device = Optional[DeviceBuffer[DType.int32]]()
    var summary_device = Optional[DeviceBuffer[DType.int32]]()
    var lengths_device = Optional[DeviceBuffer[DType.int64]]()
    var eos_device = Optional[DeviceBuffer[DType.int32]]()
    if read_address != 0:
        status = List[Int32](length=2, fill=0)
        status_device = ctx.enqueue_create_buffer[DType.int32](2 * b)
        finished_device = status_device.value().create_sub_buffer[DType.int32](0, b)
        summary_device = ctx.enqueue_create_buffer[DType.int32](2)
        lengths_device = ctx.enqueue_create_buffer[DType.int64](b)
        eos_device = ctx.enqueue_create_buffer[DType.int32](max(1, len(read[].eos)))
        ctx.enqueue_copy(dst_buf=finished_device.value(), src_ptr=read[].finished.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=lengths_device.value(), src_ptr=read[].lengths.unsafe_ptr())
        if len(read[].eos) > 0:
            ctx.enqueue_copy(dst_buf=eos_device.value(), src_ptr=read[].eos.unsafe_ptr())
    var output_rows = List[DeviceBuffer[DType.int32]]()
    var logits_rows = List[DeviceBuffer[DType.float32]]()
    var workspace = _LMRunWorkspace()
    var finite_checks = Optional[DeviceNonfinitePair]()
    var progress_synchronized = True
    try:
        var cur_l = l
        for step in range(n_steps):
            if read_address != 0:
                var consumed = l + step
                var position = read[].position + consumed
                if read[].max_positions >= 0 and position > read[].max_positions:
                    raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
                if position > read[].state_capacity:
                    raise Error("CausalLM: sequence exceeds allocated state capacity")
                if consumed > read[].cache_remaining:
                    raise Error("causal lm resident: additional positions do not fit the cache")
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
                # No final norm (LM_RESIDENT_GENERATE): the head reads the stack
                # output, as the per-layer route's head does; a copy, no arithmetic.
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
            # The same independent x/hn scans share one completion wait.
            # Lazily prepare at the original refusal point (not for zero steps)
            # and keep x error priority before progress or selection commits.
            if not finite_checks:
                finite_checks = DeviceNonfinitePair(ctx, max(b * l * d, b * d), Bool(s.norm))
            var bad = finite_checks.value().first_nonfinite(ctx, x, hn, m * d)
            if bad[0] >= 0:
                raise Error("mojolearn samba ops: non-finite rms_norm input at flat index " + String(bad[0]))
            if bad[1] >= 0:
                raise Error("mojolearn samba ops: non-finite linear input at flat index " + String(bad[1]))
            if read_address != 0:
                read[].consumed += cur_l
            if read_address == 0:
                ctx.enqueue_function[_lm_last_argmax_kernel](
                    logits.unsafe_ptr(), next_ids.unsafe_ptr(), out_dev.unsafe_ptr(),
                    Int32(cur_l), Int32(v), Int32(n_steps), Int32(step),
                    grid_dim=(b, 1, 1), block_dim=(_LM_ARGMAX_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[_lm_read_argmax_kernel](
                    logits.unsafe_ptr(), next_ids.unsafe_ptr(), out_dev.unsafe_ptr(),
                    status_device.value().unsafe_ptr(),
                    Int32(b), Int32(cur_l), Int32(v), Int32(n_steps), Int32(step), Int32(read[].pad),
                    grid_dim=(b, 1, 1), block_dim=(_LM_ARGMAX_TPB, 1, 1),
                )
                progress_synchronized = False
                ctx.enqueue_function[_lm_read_status_kernel](
                    status_device.value().unsafe_ptr(), summary_device.value().unsafe_ptr(),
                    lengths_device.value().unsafe_ptr(), next_ids.unsafe_ptr(), eos_device.value().unsafe_ptr(),
                    Int32(b), Int32(len(read[].eos)),
                    grid_dim=(1, 1, 1), block_dim=(_LM_ARGMAX_TPB, 1, 1),
                )
                ctx.enqueue_copy(dst_ptr=status.unsafe_ptr(), src_buf=summary_device.value())
            if p_logits != 0 and step == n_steps - 1:
                var lp = _f32_ptr(p_logits)
                for r in range(b):
                    logits_rows.append(logits.create_sub_buffer[DType.float32]((r * cur_l + cur_l - 1) * v, v))
                    ctx.enqueue_copy(dst_ptr=lp + r * v, src_buf=logits_rows[r])
            ctx.synchronize()
            progress_synchronized = True
            logits_rows.clear()
            cur_l = 1
            if read_address != 0:
                if status[1] >= 0:
                    raise Error("generation: non-finite logit at token " + String(status[1]))
                read[].wrote = step + 1
                if status[0] == 0:
                    break
        if read_address == 0:
            ctx.enqueue_copy(dst_ptr=out_i32, src_buf=out_dev)
        else:
            # A stopped read has unwritten trailing capacity. Copy only published
            # tokens, retaining row views until the final device synchronization.
            for r in range(b):  # driver-loop: completed output-row transfer spans, no token computation
                output_rows.append(out_dev.create_sub_buffer[DType.int32](r * n_steps, read[].wrote))
                ctx.enqueue_copy(dst_ptr=out_i32 + r * n_steps, src_buf=output_rows[r])
        if read_address != 0:
            _lm_read_progress(ctx, read[], lengths_device.value(), finished_device.value())
        else:
            ctx.synchronize()
    except original:
        if read_address != 0 and progress_synchronized:
            try:
                _lm_read_progress(ctx, read[], lengths_device.value(), finished_device.value())
            except recovery:
                # Device/readback failure cannot replace the original model or
                # policy error. progress_ready stays false after a partial copy.
                _ = recovery
        # Keep all read-owned scratch and transfer views live through failure
        # completion too; cleanup cannot replace the original operation error.
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
        _ = output_rows^
        _ = finished_device^
        _ = status_device^
        _ = summary_device^
        _ = lengths_device^
        _ = eos_device^
        _ = status^



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


def _session_forward_tape_run(owner_addr: Int, a: List[Int], p: List[Int]) raises:
    """`transformer_session_forward_tape`'s device half, GIL-free: build or
    reuse the session weights, run the forward that keeps the borrowed
    attention tape, copy y out and seal the tape. `owner_addr` is the session handle's address."""
    var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=owner_addr)
    var dims = LlamaDims(p[2],p[3],p[4],p[5],p[6])
    dims.validate()
    if not owner[].ctx:
        owner[].ctx = neural_ctx[_NEURAL_CTX]()
        owner[].contexts += 1
    var opts = BlockOptions()
    _session_weights(owner[],dims,a,opts,List[Int]())
    ref ctx = owner[].ctx.value()
    ref w = owner[].weights.value()
    var input = _upload_addr(ctx,a[0],p[0]*p[1]*p[2])
    owner[].tape = BorrowedAttentionTape(ctx,w,input^,p[0],p[1],p[7],owner[].tape_generation)
    ref tape = owner[].tape.value()
    _download_addr(ctx,tape.stages.value().residual2,p[0]*p[1]*p[2],a[10])
    tape.seal(ctx,p[8],p[9])


def _session_backward_tape_run(owner_addr: Int, a: List[Int], generation: Int) raises:
    """`transformer_session_backward_tape`'s device half, GIL-free: replay
    the sealed tape's backward and copy dx and the ten weight gradients out. `owner_addr` is the session handle's address."""
    var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=owner_addr)
    ref ctx = owner[].ctx.value()
    ref tape = owner[].tape.value()
    var dims = tape.dims.copy()
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var m = tape.b*tape.l
    var dy = _upload_addr(ctx,a[0],m*dm)
    var bst = LlamaBackwardStages(ctx,tape.b,tape.l,tape.l,dims,lean=True)
    tape.backward_into(ctx,owner[].weights.value(),bst,dy,generation,
        owner[].tape_generation,owner[].tape_generation,owner[].tape_generation)
    _download_addr[False](ctx,bst.d_x,m*dm,a[1])
    _download_addr[False](ctx,bst.dw_norm1,dm,a[2])
    _download_addr[False](ctx,bst.dw_norm2,dm,a[3])
    _download_addr[False](ctx,bst.dw_q,qw*dm,a[4])
    _download_addr[False](ctx,bst.dw_k,kw*dm,a[5])
    _download_addr[False](ctx,bst.dw_v,kw*dm,a[6])
    _download_addr[False](ctx,bst.dw_o,dm*qw,a[7])
    _download_addr[False](ctx,bst.dw_gate,it*dm,a[8])
    _download_addr[False](ctx,bst.dw_up,it*dm,a[9])
    _download_addr[False](ctx,bst.dw_down,dm*it,a[10])
    ctx.synchronize()
    owner[].tape = None
    owner[].tape_generation += 1
    ctx.synchronize()
