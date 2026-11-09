# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident Samba decode session (lane fix-d1-decode-sessions,
2026-11-08; IDENTICAL only, built where `IDN_SAMBA_RESIDENT_STEP` holds, as
the resident forward and train step are).

WHAT THE PER-CALL STEP PAYS. `SambaStack.step` (bindings/samba_inference_native.mojo
`samba_forward(step=True)`) drives the stack layer by layer per token: the
embedding into a host array, then per layer one public block door (a Mamba-3
layer through `_m3_state_call`: nine weight uploads, ten state uploads, stage
allocation, x upload, the block, fifteen downloads and a wait; an attention
layer through `neural_block_forward_binding`: weights by host address, the KV
cache up and down), each layer's output a new host array (owed row N094), then
the final norm and the head from host arrays. Measured on MI300X, 0.8.66 vs
0.8.55: samba-decode 1.57x (+4.2 ms per token); the device op list is the
0.8.55 one, the loss is the per-token host door
(docs/plans/DECODE_DOOR_AND_GLM_PLAN.md section 2.2).

WHAT THIS IS. One `SambaDecodeResident` per open `SambaDecodeSession`: the
stack's registry on the device (copied once at open, scanned once), per-layer
weight VIEWS into it (Mamba-3 `Mamba3DeviceWeights`, attention
`LlamaDeviceWeights`, exactly the views `SambaResidentSession` builds), per
Mamba-3 layer its carried `Mamba3DeviceState`, per attention layer its own
`LlamaKVCache` at the state's capacity plus L = 1 stages, one rotary table at
that capacity, and the B x d_model activations between layers. `step` uploads
B token ids, runs embedding -> layers -> final norm -> head with the hidden
state passed device to device, and downloads the B x vocab logits: one id
upload, one logits download, one wait per token.

NO NEW ARITHMETIC, SAME BITS. Every launch is the function the per-call step
reaches for the same stage, on the same operands, in the same order:
`identical_embedding_forward_into` (the `embedding_forward` entry's gather),
`mamba3_block_forward(..., L = 1)` on the carried state with stages built per
step at the carried `buf_len` (`_m3_session_step_run`, the Mamba-3 decode
session, held byte for byte to the per-call step by
python/mojolearn/tests/test_mamba23_decode_sessions.py),
`llama_decoder_layer_forward(..., l = 1, pos0 = kv.s, forward_only = True)`
on a cache, rotary table and stages at the state's capacity (`_transformer_run`
and `_session_step_run`), `llama_rms_norm` and the `OP_NT` head GEMM at the
default role (`_norm_head_forward(train = False)`, the resident forward's own
claim against `rms_norm_forward` / `linear_forward`). What changes is
residency: the weights, the states and the hidden state never leave the
device between those calls. The state goes back to the caller's `SambaState`
arrays at `export` (sync_state / close), byte for byte what the per-call step
would have left there.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.decode_slab import decode_slab_check, decode_slab_put, decode_slab_take_ids
from core.device_scan import device_first_nonfinite
from core.hostptr import f32_ptr
from core.identity_trace import IdentityTrace
from embedding.checks.embedding_identical import identical_embedding_forward_into
from embedding.checks.embedding_oracle import EmbConfig
from gemm.contract import OP_NT
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
)
from mamba.impl.modeling.modeling_mamba import mamba_copy_in
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    mamba3_block_forward,
)
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP
from training.samba_resident import (
    SAMBA_BLOCK_TENSORS,
    SAMBA_LAYER_ATTENTION,
    SAMBA_LAYER_MAMBA3,
    _d2d,
    _lean_stages,
    samba_registry_layout,
)
from transformer.block_options import BlockOptions
from transformer.checks.transformer_fixture import RMS_EPS
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    llama_decoder_layer_forward,
    llama_rms_norm,
)


#: Ten carried pieces per Mamba-3 layer, in `_m3_names()` order
#: (bindings/samba_inference_native.mojo): theta, h, buffer_qrot, buffer_krot,
#: buffer_v, buffer_dt, buffer_sig, buffer_adt, pending_k, pending_v.
comptime SAMBA_DECODE_M3_PIECES = 10
#: Two carried pieces per attention layer: k_cache, v_cache.
comptime SAMBA_DECODE_ATTN_PIECES = 2


def _m3_piece_sizes(b: Int, dims: Mamba3Dims) -> List[Int]:
    """The ten piece lengths in `_m3_names()` order (the shapes
    `_m3_shapes(b, nheads)` allocates)."""
    var nh = dims.nheads
    var qrow = b * M3_CHUNK_SIZE * nh
    return [
        b * nh * M3_NUM_ROPE_ANGLES,
        b * nh * M3_HEADDIM * M3_D_STATE,
        qrow * M3_D_STATE,
        qrow * M3_D_STATE,
        qrow * M3_HEADDIM,
        qrow,
        qrow,
        qrow,
        b * nh * M3_D_STATE,
        b * nh * M3_HEADDIM,
    ]


def _download_into(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int, addr: Int) raises:
    """`n` floats of `buf` to the caller's address (no wait; the caller
    waits once for every download it enqueued)."""
    if n <= 0:
        return
    ctx.enqueue_copy(dst_ptr=f32_ptr(addr), src_buf=buf.create_sub_buffer[DType.float32](0, n))


def _upload_new(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A fresh device buffer holding the caller's `n` floats (the
    `_upload_addr` shape the per-call transformer entry uses for the cache)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=f32_ptr(addr))
    return buf^


struct SambaDecodeResident(Movable, Writable):
    """One open Samba decode: the registry, the per-layer weight views and
    carried states, and the L = 1 workspaces, for a fixed batch `b` and cache
    capacity `smax`. Owned by the Python `SambaDecodeSession` through the
    training binding's `_SambaDecodeResident` type; dropped with it."""

    var ctx: DeviceContext
    var vocab: Int
    var dm: Int
    var tie: Bool
    var eps: Float32
    var has_attn: Bool
    var ldims: LlamaDims
    var lean: Bool
    var kinds: List[Int]
    var slot: List[Int]
    var offsets: List[Int]
    var n_total: Int
    var b: Int
    var smax: Int
    var param: DeviceBuffer[DType.float32]
    var m3_w: List[Mamba3DeviceWeights]
    var t_w: List[LlamaDeviceWeights]
    var m3_state: List[Mamba3DeviceState]
    var t_kv: List[LlamaKVCache]
    var t_stages: List[LlamaDeviceStages]
    var rope: Optional[LlamaRopeTable]
    var ids: DeviceBuffer[DType.int32]
    #: n_layers + 1 hidden states of B x d_model: acts[0] the embedding,
    #: acts[i + 1] layer i's output
    var acts: List[DeviceBuffer[DType.float32]]
    var sumsq: DeviceBuffer[DType.float32]
    var hn: DeviceBuffer[DType.float32]
    var logits: DeviceBuffer[DType.float32]
    var head_ws: DeviceBuffer[DType.float32]
    var busy: Bool
    var usable: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        kinds: List[Int],
        vocab: Int,
        dm: Int,
        nh: Int,
        nkv: Int,
        hd: Int,
        it: Int,
        tie: Bool,
        eps: Float32,
        n_total: Int,
        b: Int,
        smax: Int,
        param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ) raises:
        """The registry copied to the device and scanned once, the weight
        views, zero states and the L = 1 workspaces. The carried states are
        loaded by `samba_decode_load` right after (the open entry calls it)."""
        if vocab < 2 or dm < 1 or len(kinds) < 1:
            raise Error("mojolearn samba decode: vocab >= 2, d_model >= 1 and at least one layer are required")
        if b < 1 or smax < 1:
            raise Error("mojolearn samba decode: batch_size and max_tokens must be positive")
        self.ctx = ctx.copy()
        self.vocab = vocab
        self.dm = dm
        self.tie = tie
        self.eps = eps
        self.kinds = kinds.copy()
        self.b = b
        self.smax = smax
        self.has_attn = False
        for i in range(len(kinds)):  # small-loop(kinds: one code per layer): layer kinds, not data
            if kinds[i] == SAMBA_LAYER_ATTENTION:
                self.has_attn = True
            elif kinds[i] != SAMBA_LAYER_MAMBA3:
                raise Error("mojolearn samba decode: layer kind code " + String(kinds[i]) + " is neither 0 (mamba3) nor 1 (attention)")
        self.ldims = LlamaDims(dm, max(nh, 1), max(nkv, 1), max(hd, 1), max(it, 1))
        if self.has_attn:
            self.ldims = LlamaDims(dm, nh, nkv, hd, it)
            self.ldims.validate()
        self.lean = _lean_stages(hd) if self.has_attn else False
        self.offsets = List[Int]()
        self.slot = List[Int]()
        var running = samba_registry_layout(kinds, vocab, dm, self.ldims, it, tie, self.offsets, self.slot)
        self.n_total = running
        if n_total != running:
            raise Error(
                "mojolearn samba decode: the stack's registry holds " + String(n_total)
                + " floats, the configuration names " + String(running)
            )
        # The weights are copied once: edits to the stack's parameters after
        # the session opens are not seen (the session docstring says so).
        self.param = ctx.enqueue_create_buffer[DType.float32](running)
        ctx.enqueue_copy(dst_buf=self.param, src_ptr=param_ptr)
        # One scan of the whole registry (the resident forward's
        # `_admit_registry`), so the Mamba-3 views skip their per-call scans.
        var bad = device_first_nonfinite(ctx, self.param, running)
        if bad >= 0:
            raise Error("mojolearn samba decode: non-finite parameter at flat registry index " + String(bad))
        var opts = BlockOptions()
        self.rope = None
        if self.has_attn:
            self.rope = LlamaRopeTable(ctx, self.ldims, opts, smax)
        self.m3_w = List[Mamba3DeviceWeights]()
        self.t_w = List[LlamaDeviceWeights]()
        self.m3_state = List[Mamba3DeviceState]()
        self.t_kv = List[LlamaKVCache]()
        self.t_stages = List[LlamaDeviceStages]()
        self.ids = ctx.enqueue_create_buffer[DType.int32](b)
        self.acts = List[DeviceBuffer[DType.float32]]()
        for _ in range(len(kinds) + 1):  # driver-loop: one hidden buffer per layer boundary
            self.acts.append(ctx.enqueue_create_buffer[DType.float32](b * dm))
        self.sumsq = ctx.enqueue_create_buffer[DType.float32](b)
        self.hn = ctx.enqueue_create_buffer[DType.float32](b * dm)
        self.logits = ctx.enqueue_create_buffer[DType.float32](b * vocab)
        self.head_ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(b, vocab, dm))
        self.busy = False
        self.usable = True
        # Every field is set; the weight views below read `self.param`.
        for i in range(len(kinds)):  # driver-loop: over the stack's layers (weight views, state owners)
            var base = 1 + SAMBA_BLOCK_TENSORS * i
            if kinds[i] == SAMBA_LAYER_MAMBA3:
                var d3 = Mamba3Dims.of(dm)
                var w = Mamba3DeviceWeights(
                    d3,
                    self._view(base + 0), self._view(base + 1), self._view(base + 2),
                    self._view(base + 3), self._view(base + 4), self._view(base + 5),
                    self._view(base + 6), self._view(base + 7), self._view(base + 8),
                )
                w.weights_checked = True
                self.m3_w.append(w^)
                self.m3_state.append(Mamba3DeviceState(ctx, b, d3))
            else:
                # The default-options constructor at RMS_EPS, as the resident
                # forward builds a Samba attention layer; it scans its nine
                # tensors once here.
                self.t_w.append(
                    LlamaDeviceWeights(
                        ctx, self.ldims, RMS_EPS,
                        self._view(base + 0), self._view(base + 1), self._view(base + 2),
                        self._view(base + 3), self._view(base + 4), self._view(base + 5),
                        self._view(base + 6), self._view(base + 7), self._view(base + 8),
                    )
                )
                # Full causal attention (window 0) at the state's capacity:
                # the cache, rotary table and stages the per-call step builds.
                self.t_kv.append(LlamaKVCache(ctx, b, self.ldims, smax, 0, opts.max_positions))
                self.t_stages.append(LlamaDeviceStages(ctx, b, 1, smax, self.ldims, 0, lean=self.lean))
        ctx.synchronize()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SambaDecodeResident(rows=", self.b, ", max_tokens=", self.smax, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def _view(self, j: Int) raises -> DeviceBuffer[DType.float32]:
        """Registry tensor `j` as a view of `param`."""
        return self.param.create_sub_buffer[DType.float32](
            self.offsets[j], self.offsets[j + 1] - self.offsets[j]
        )

    def n_layers(self) -> Int:
        return len(self.kinds)

    def norm_f_index(self) -> Int:
        return 1 + SAMBA_BLOCK_TENSORS * len(self.kinds)

    def head_index(self) -> Int:
        if self.tie:
            return 0
        return self.norm_f_index() + 1

    def cache_count(self) -> Int:
        """Floats in one attention layer's k (or v) cache: B x n_kv x
        max_tokens x head_dim (window 0)."""
        return self.b * self.ldims.n_kv * self.smax * self.ldims.head_dim

    def piece_count(self) -> Int:
        """State addresses the load and export entries take, in stack order:
        ten per Mamba-3 layer, two per attention layer."""
        var n = 0
        for i in range(len(self.kinds)):  # small-loop(kinds: one code per layer): counts, not data
            n += SAMBA_DECODE_M3_PIECES if self.kinds[i] == SAMBA_LAYER_MAMBA3 else SAMBA_DECODE_ATTN_PIECES
        return n


def samba_decode_load(
    mut s: SambaDecodeResident, pieces: List[Int], cursors: List[Int]
) raises:
    """The caller's `SambaState` into the resident states. `pieces` is the
    state addresses in stack order (ten per Mamba-3 layer in `_m3_names()`
    order, then k_cache and v_cache per attention layer); `cursors` is per
    layer `buffered_tokens, pending` for a Mamba-3 layer and
    `cached_tokens, 0` for an attention layer. The Mamba-3 rules are
    `_m3_session_copy_in`'s (mamba/impl/mamba_shell_impl.mojo), word for word."""
    if len(pieces) != s.piece_count() or len(cursors) != 2 * s.n_layers():
        raise Error("samba decode: expected " + String(s.piece_count()) + " state addresses and "
                    + String(2 * s.n_layers()) + " cursor words")
    ref ctx = s.ctx
    var at = 0
    for i in range(s.n_layers()):  # driver-loop: over the stack's layers (state owners)
        var j = s.slot[i]
        var c0 = cursors[2 * i]
        var c1 = cursors[2 * i + 1]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            if c0 < 0 or c0 > M3_CHUNK_SIZE:
                raise Error("samba decode: layer " + String(i) + " buffered_tokens out of range")
            if c1 != 0 and c1 != 1:
                raise Error("samba decode: layer " + String(i) + " pending must be 0 or 1")
            if c1 == 1 and c0 != 0:
                raise Error("samba decode: layer " + String(i) + " Input_States pending requires a fresh state")
            ref st = s.m3_state[j]
            var sizes = _m3_piece_sizes(s.b, Mamba3Dims.of(s.dm))
            mamba_copy_in(ctx, st.theta, f32_ptr(pieces[at + 0]), sizes[0])
            mamba_copy_in(ctx, st.h, f32_ptr(pieces[at + 1]), sizes[1])
            mamba_copy_in(ctx, st.buf_qrot, f32_ptr(pieces[at + 2]), sizes[2])
            mamba_copy_in(ctx, st.buf_krot, f32_ptr(pieces[at + 3]), sizes[3])
            mamba_copy_in(ctx, st.buf_v, f32_ptr(pieces[at + 4]), sizes[4])
            mamba_copy_in(ctx, st.buf_dt, f32_ptr(pieces[at + 5]), sizes[5])
            mamba_copy_in(ctx, st.buf_sig, f32_ptr(pieces[at + 6]), sizes[6])
            mamba_copy_in(ctx, st.buf_adt, f32_ptr(pieces[at + 7]), sizes[7])
            mamba_copy_in(ctx, st.pend_k, f32_ptr(pieces[at + 8]), sizes[8])
            mamba_copy_in(ctx, st.pend_v, f32_ptr(pieces[at + 9]), sizes[9])
            st.buf_len = c0
            st.pending = c1 == 1
            at += SAMBA_DECODE_M3_PIECES
        else:
            if c0 < 0 or c0 > s.smax:
                raise Error("samba decode: layer " + String(i) + " cached_tokens must be in [0, "
                            + String(s.smax) + "], got " + String(c0))
            var n = s.cache_count()
            s.t_kv[j].k = _upload_new(ctx, pieces[at + 0], n)
            s.t_kv[j].v = _upload_new(ctx, pieces[at + 1], n)
            s.t_kv[j].s = c0
            at += SAMBA_DECODE_ATTN_PIECES
    # The caller's arrays are read inside this call only.
    ctx.synchronize()


def samba_decode_export(
    mut s: SambaDecodeResident, pieces: List[Int]
) raises -> List[Int]:
    """The resident states back to the caller's `SambaState` arrays (the
    addresses `samba_decode_load` takes); returns the cursor words in the
    same layout. One wait for every download."""
    if len(pieces) != s.piece_count():
        raise Error("samba decode: expected " + String(s.piece_count()) + " state addresses")
    ref ctx = s.ctx
    var cursors = List[Int]()
    var at = 0
    for i in range(s.n_layers()):  # driver-loop: over the stack's layers (state owners)
        var j = s.slot[i]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            ref st = s.m3_state[j]
            var sizes = _m3_piece_sizes(s.b, Mamba3Dims.of(s.dm))
            _download_into(ctx, st.theta, sizes[0], pieces[at + 0])
            _download_into(ctx, st.h, sizes[1], pieces[at + 1])
            _download_into(ctx, st.buf_qrot, sizes[2], pieces[at + 2])
            _download_into(ctx, st.buf_krot, sizes[3], pieces[at + 3])
            _download_into(ctx, st.buf_v, sizes[4], pieces[at + 4])
            _download_into(ctx, st.buf_dt, sizes[5], pieces[at + 5])
            _download_into(ctx, st.buf_sig, sizes[6], pieces[at + 6])
            _download_into(ctx, st.buf_adt, sizes[7], pieces[at + 7])
            _download_into(ctx, st.pend_k, sizes[8], pieces[at + 8])
            _download_into(ctx, st.pend_v, sizes[9], pieces[at + 9])
            cursors.append(st.buf_len)
            cursors.append(1 if st.pending else 0)
            at += SAMBA_DECODE_M3_PIECES
        else:
            var n = s.cache_count()
            _download_into(ctx, s.t_kv[j].k, n, pieces[at + 0])
            _download_into(ctx, s.t_kv[j].v, n, pieces[at + 1])
            cursors.append(s.t_kv[j].s)
            cursors.append(0)
            at += SAMBA_DECODE_ATTN_PIECES
    ctx.synchronize()
    return cursors^


def _samba_decode_token(mut s: SambaDecodeResident, mut m3_stages: List[Mamba3DeviceStages]) raises -> Int:
    """The stack on the B ids already in `s.ids`: embedding -> layers ->
    final norm -> head into `s.logits`, all enqueued, no wait. The Mamba-3
    stages it builds go into `m3_stages`, which the caller holds until its
    wait. Returns the first attention layer's cached_tokens after the token
    (0 for an all-Mamba-3 stack)."""
    ref ctx = s.ctx
    var b = s.b
    var dm = s.dm
    var emb = EmbConfig.llama(s.vocab, dm)
    var ew = s._view(0)
    identical_embedding_forward_into(ctx, s.acts[0], ew, s.ids, b, emb)
    var trace = IdentityTrace.disabled()
    # The Mamba-3 stages depend on the carried buf_len, so they are built per
    # step (as `_m3_session_step_run` builds them).
    var position = 0
    for i in range(s.n_layers()):  # driver-loop: over the stack's layers, in stack order
        var j = s.slot[i]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            var d3 = Mamba3Dims.of(dm)
            m3_stages.append(Mamba3DeviceStages(ctx, b, 1, s.m3_state[j].buf_len, d3))
            var k = len(m3_stages) - 1
            mamba3_block_forward(
                ctx, m3_stages[k], s.m3_state[j], s.m3_w[j], s.acts[i], b, 1,
                trace, String("samba.decode.m3"),
            )
            _d2d(ctx, s.acts[i + 1], 0, m3_stages[k].residual_out, b * dm)
        else:
            var pos0 = s.t_kv[j].s
            llama_decoder_layer_forward(
                ctx, s.t_stages[j], s.t_kv[j], s.rope.value(), s.t_w[j], s.acts[i],
                b, 1, pos0, trace, String("samba.decode.attn"), forward_only=True,
            )
            _d2d(ctx, s.acts[i + 1], 0, s.t_stages[j].residual2, b * dm)
            if position == 0:
                position = s.t_kv[j].s
    var nw = s._view(s.norm_f_index())
    var hw = s._view(s.head_index())
    llama_rms_norm(ctx, s.sumsq, s.hn, s.acts[s.n_layers()], nw, b, dm, s.eps)
    identical_gemm_into(ctx, s.logits, s.hn, hw, s.head_ws, b, s.vocab, dm, OP_NT)
    _ = ew^
    _ = nw^
    _ = hw^
    return position


def samba_decode_step(
    mut s: SambaDecodeResident,
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """One token per row: B ids (already range-checked by the door, and
    again on the device by the embedding's own refusal) -> B x vocab logits
    at `out_ptr`. Returns the first attention layer's cached_tokens after the
    step (0 for an all-Mamba-3 stack), which the door does not need but a
    probe can read. ONE wait, after the logits copy is enqueued."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("mojolearn samba decode: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
    ref ctx = s.ctx
    ctx.enqueue_copy(dst_buf=s.ids, src_ptr=ids_ptr)
    var m3_stages = List[Mamba3DeviceStages]()
    var position = _samba_decode_token(s, m3_stages)
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=s.logits)
    ctx.synchronize()
    _ = m3_stages^
    return position


def samba_decode_tokens(
    mut s: SambaDecodeResident,
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    steps: Int,
) raises -> Int:
    """N tokens per row in one call (decode plan D2): the (B, N) ids uploaded
    once, the body of `samba_decode_step` once per token with token t's ids
    taken on the device and its B x vocab logits put into a (B, N, vocab)
    device slab, one download, ONE wait. Same kernels, same order, same bytes
    as N `samba_decode_step` calls; the states end where they leave them."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("mojolearn samba decode: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
    decode_slab_check(s.b, steps, s.vocab, "samba decode")
    ref ctx = s.ctx
    var ids = ctx.enqueue_create_buffer[DType.int32](s.b * steps)
    var out = ctx.enqueue_create_buffer[DType.float32](s.b * steps * s.vocab)
    ctx.enqueue_copy(dst_buf=ids, src_ptr=ids_ptr)
    var position = 0
    var last = List[Mamba3DeviceStages]()
    for t in range(steps):  # driver-loop: over autoregressive decode steps (each reads the state the previous step left), the stack's one-token body each
        decode_slab_take_ids(ctx, s.ids, ids, s.b, steps, t)
        # The previous token's stages return to the stream-ordered pool.
        last = List[Mamba3DeviceStages]()
        position = _samba_decode_token(s, last)
        decode_slab_put(ctx, out, s.logits, s.b, steps, s.vocab, t)
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()
    _ = last^
    _ = ids^
    _ = out^
    return position
