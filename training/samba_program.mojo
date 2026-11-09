# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Samba stack as ONE executor program (lane x6-samba-on-executor,
2026-11-09; docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X6, the API
note docs/plans/EXECUTOR_X1_API.md, owed row N094).

`SambaStackProgram` is one inference `OpsSession` on the device holding

- the stack's flat registry as ONE uploaded tensor (`reg`), scanned once per
  load; the embedding, the final norm and the head read views of it;
- per layer, the block's weights as ONE packed parameter tensor
  (`upload_block`, the block struct's field order; the registry's nine
  tensors copied device to device into their slots), the carried float state
  as ONE declared state tensor and the carried integers as one int32-word
  state tensor (Mamba-3 [buf_len, pending], attention [cached tokens]);
- the hidden state between layers: every layer is ONE session op,
  `OPS_M3_STEP` / `OPS_ATTN_STEP` (autograd/block_step_ops.mojo), whose input
  is the previous op's result tensor. The hidden state never leaves the
  device.

A run is embedding -> the layers' step ops -> final RMSNorm -> head GEMM, then
the logits come down once. L = 1 is a decode step, L > 1 a prefill or a
stateful continuation, from a zero state (stateless `forward`) or from the
state the caller loaded. This one body serves `SambaStack.forward`,
`SambaStack.step` and `SambaDecodeSession` (bindings/samba_inference_native.mojo,
bindings/samba_decode_native.mojo). It replaced three device walks: the
resident session's forward (training/samba_resident.mojo `samba_resident_forward`),
the provider's per-layer walk over host arrays (providers/python/samba_execution_pyapi.mojo
`samba_provider_forward`) and the per-layer public block doors; and the
decode residency (training/samba_decode_resident.mojo). TOMBSTONES.md
SAMBA_THREE_TIER_FORWARD.

Kernels: the embedding gather (`identical_embedding_forward_into`, whose
device id refusal is the token-range check), `llama_rms_norm` and the `OP_NT`
head GEMM at the default role (`identical_gemm_into`) are the ones the
resident forward and the decode residency launched; the layers are the step
ops' bodies (the Mamba-3 and transformer decode sessions'). Bits: the
attention layers now run the step op's options-constructor weights and its
`forward_only=False` launch set (the decode residency ran `forward_only=True`);
same kernels on every vendor and in every column, so a version-level change
at most (identity lane `samba`).

Weights: the caller's flat registry is the truth (`SambaStack.parameters()`
are writable views of it). The program keeps a `WeightKey` (lane x45c,
core/weight_epoch.mojo): the registry address and the weight epoch at its
load. The per-call doors call `refresh_weights`, which reloads only when the
key is not current (integ-w4: formerly `load_weights` on every call, the
whole registry up per call and per step token). A reload is one
host-to-device copy of the registry, ONE device scan of it (the resident
forward's `_admit_registry` rule, naming the flat index), then the packed
copies device to device. Every Samba operation that writes the registry in
place (train step, apply, publish, checkpoint load, load_weights) bumps the
epoch (bindings/samba_public_native.mojo). The step ops' resident structs are views of
the packed tensors and hold no derived copies, so they stay valid; the
registry scan stands in for their per-block refusals (each block's own scan
ran once, at its first run). A decode session loads the weights once at open.

No arithmetic here; no host loop over rows or elements.
"""
from std.sys.info import has_amd_gpu_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext

from core.cast_in import CastIn
from core.decode_slab import decode_slab_check, decode_slab_put, decode_slab_take_ids
from core.device_scan import device_first_nonfinite
from embedding.checks.embedding_identical import identical_embedding_forward_into
from embedding.checks.embedding_oracle import EmbConfig
from gemm.contract import OP_NT
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from mamba.checks.mamba3_fixture import M3_CHUNK_SIZE
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from autograd.ops_api import OpsSession
from autograd.block_step_ops import (
    OPS_M3_STEP, OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS,
    block_weight_sizes, block_state_sizes, block_offsets, block_guard, block_buffer_view, block_step_words,
    block_part_floats,
)
from training.samba_resident import (
    SAMBA_BLOCK_TENSORS,
    SAMBA_LAYER_ATTENTION,
    SAMBA_LAYER_MAMBA3,
    samba_registry_layout,
)
from transformer.impl.llama.modeling_llama import LlamaDims, llama_rms_norm
from core.weight_epoch import WeightKey


#: The program runs block step ops, which are device only: a host build
#: registers no program entry and keeps the CPU per-layer route
#: (bindings/samba_inference_native.mojo).
comptime SAMBA_PROGRAM_ON = has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator() or has_apple_gpu_accelerator()

#: Ten carried pieces per Mamba-3 layer, in `_m3_names()` order
#: (bindings/samba_inference_native.mojo) = `_m3_state_sizes` order: theta, h,
#: buffer_qrot, buffer_krot, buffer_v, buffer_dt, buffer_sig, buffer_adt,
#: pending_k, pending_v.
comptime SAMBA_PROGRAM_M3_PIECES = 10
#: Two carried pieces per attention layer: k_cache, v_cache.
comptime SAMBA_PROGRAM_ATTN_PIECES = 2


struct SambaStackProgram(Movable, Writable):
    """One Samba stack on the executor for a fixed batch `b` and attention
    capacity `smax`: the registry, the layers' packed weights and states,
    and the L-shaped workspaces of the stack-level kernels."""

    var s: OpsSession[DeviceExec]
    var vocab: Int
    var dm: Int
    var tie: Bool
    var eps: Float32
    var kinds: List[Int]
    #: the registry offsets, J + 1 entries (`SambaConfig.registry` order)
    var offsets: List[Int]
    var n_total: Int
    var b: Int
    var smax: Int
    var reg: Int
    #: per layer: step op code, its integer arguments, the weights, state and
    #: words tensors
    var codes: List[Int]
    var lints: List[List[Int]]
    var w: List[Int]
    var st: List[Int]
    var q: List[Int]
    #: per layer two host words: Mamba-3 (buf_len, pending), attention
    #: (cached tokens, 0); what the last load or run left
    var cur: List[Int]
    # ---- the L-shaped workspaces (rebuilt when B * L changes)
    var m: Int
    var x0: Int
    var ids: DeviceBuffer[DType.int32]
    var sumsq: DeviceBuffer[DType.float32]
    var hn: DeviceBuffer[DType.float32]
    var logits: DeviceBuffer[DType.float32]
    var head_ws: DeviceBuffer[DType.float32]
    var busy: Bool
    var usable: Bool
    #: integ-w4 (lane x45c's recipe): the registry address and the weight
    #: epoch of the loaded copy; `refresh_weights` reloads only when stale.
    var wkey: WeightKey

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
        """The registry and every layer's packed weights from the caller's
        flat registry, zero states. `smax` is the attention capacity: the
        token count of a stateless forward, the state's `max_tokens` for a
        stateful call or a decode session."""
        if vocab < 2 or dm < 1 or len(kinds) < 1:
            raise Error("mojolearn samba program: vocab >= 2, d_model >= 1 and at least one layer are required")
        if b < 1 or smax < 1:
            raise Error("mojolearn samba program: batch_size and max_tokens must be positive")
        # The epoch is read BEFORE the registry upload below (a bump in
        # between only costs one reload).
        var wkey = WeightKey.take([Int(param_ptr)])
        var has_attn = False
        for i in range(len(kinds)):  # small-loop(kinds: one code per layer): layer kinds, not data
            if kinds[i] == SAMBA_LAYER_ATTENTION:
                has_attn = True
            elif kinds[i] != SAMBA_LAYER_MAMBA3:
                raise Error("mojolearn samba program: layer kind code " + String(kinds[i])
                            + " is neither 0 (mamba3) nor 1 (attention)")
        var ldims = LlamaDims(dm, max(nh, 1), max(nkv, 1), max(hd, 1), max(it, 1))
        if has_attn:
            ldims = LlamaDims(dm, nh, nkv, hd, it)
            ldims.validate()
        var offsets = List[Int]()
        var slot = List[Int]()
        var running = samba_registry_layout(kinds, vocab, dm, ldims, it, tie, offsets, slot)
        if n_total != running:
            raise Error("mojolearn samba program: the stack's registry holds " + String(n_total)
                        + " floats, the configuration names " + String(running))
        var ex = DeviceExec(ctx)
        var s = OpsSession[DeviceExec](ex^, False)
        var base = Int(param_ptr)
        var reg = s.upload(CastIn.f32_c(base, running, 1), False)
        var codes = List[Int]()
        var lints = List[List[Int]]()
        var ws = List[Int]()
        var sts = List[Int]()
        var qs = List[Int]()
        var cur = List[Int]()
        for i in range(len(kinds)):  # driver-loop: over the stack's layers (one block program each)
            var code = OPS_M3_STEP if kinds[i] == SAMBA_LAYER_MAMBA3 else OPS_ATTN_STEP
            var ints: List[Int] = [b, dm]
            if code == OPS_ATTN_STEP:
                # Full causal attention (window 0) at the capacity, the
                # default block options (the resident session's record).
                ints = [b, dm, nh, nkv, hd, it, smax, 0]
            var wsizes = block_weight_sizes(code, ints)
            var srcs = List[CastIn](capacity=len(wsizes))
            for k in range(len(wsizes)):  # driver-loop: over the block's weight pieces (model structure)
                if k < SAMBA_BLOCK_TENSORS:
                    var j = 1 + SAMBA_BLOCK_TENSORS * i + k
                    if offsets[j + 1] - offsets[j] != wsizes[k]:
                        raise Error("mojolearn samba program: layer " + String(i) + " piece " + String(k)
                                    + " holds " + String(offsets[j + 1] - offsets[j]) + " floats, the block "
                                    + String(wsizes[k]))
                    srcs.append(CastIn.f32_c(base + 4 * offsets[j], wsizes[k], 1))
                else:
                    if wsizes[k] != 0:
                        raise Error("mojolearn samba program: an attention layer has a non-default option piece")
                    srcs.append(CastIn.f32_c(0, 0, 1))
            ws.append(s.upload_block(code, ints, BLOCK_PART_WEIGHTS, srcs, List[Int]()))
            var ssizes = block_state_sizes(code, ints)
            var most = 1
            for k in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
                most = max(most, ssizes[k])
            var zero = List[Float32](length=most, fill=Float32(0))
            var zsrc = List[CastIn](capacity=len(ssizes))
            for k in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
                zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[k], 1))
            sts.append(s.upload_block(code, ints, BLOCK_PART_STATE, zsrc, List[Int]()))
            _ = len(zero)  # keep-alive past the staging copies
            var nw = 2 if code == OPS_M3_STEP else 1
            qs.append(s.upload_block(code, ints, BLOCK_PART_WORDS, List[CastIn](), List[Int](length=nw, fill=0)))
            codes.append(code)
            lints.append(ints^)
            cur.append(0)
            cur.append(0)
        var ctx2 = ctx.copy()
        self.ids = ctx2.enqueue_create_buffer[DType.int32](1)
        self.sumsq = ctx2.enqueue_create_buffer[DType.float32](1)
        self.hn = ctx2.enqueue_create_buffer[DType.float32](1)
        self.logits = ctx2.enqueue_create_buffer[DType.float32](1)
        self.head_ws = ctx2.enqueue_create_buffer[DType.float32](1)
        self.s = s^
        self.vocab = vocab
        self.dm = dm
        self.tie = tie
        self.eps = eps
        self.kinds = kinds.copy()
        self.offsets = offsets^
        self.n_total = running
        self.b = b
        self.smax = smax
        self.reg = reg
        self.codes = codes^
        self.lints = lints^
        self.w = ws^
        self.st = sts^
        self.q = qs^
        self.cur = cur^
        self.m = 0
        self.x0 = 0
        self.busy = False
        self.usable = True
        self.wkey = wkey^
        self._admit_registry()
        self.s.ex.sync()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SambaStackProgram(rows=", self.b, ", max_tokens=", self.smax, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def n_layers(self) -> Int:
        return len(self.kinds)

    def norm_f_index(self) -> Int:
        return 1 + SAMBA_BLOCK_TENSORS * len(self.kinds)

    def head_index(self) -> Int:
        if self.tie:
            return 0
        return self.norm_f_index() + 1

    def first_position(self) -> Int:
        """The first attention layer's cached tokens (0 for an all-Mamba-3
        stack)."""
        for i in range(len(self.kinds)):  # small-loop(kinds: one code per layer): cursor lookup, not data
            if self.kinds[i] == SAMBA_LAYER_ATTENTION:
                return self.cur[2 * i]
        return 0

    def piece_count(self) -> Int:
        """State addresses `load_state` / `export_state` take, in stack
        order: ten per Mamba-3 layer, two per attention layer."""
        var n = 0
        for i in range(len(self.kinds)):  # small-loop(kinds: one code per layer): counts, not data
            n += SAMBA_PROGRAM_M3_PIECES if self.kinds[i] == SAMBA_LAYER_MAMBA3 else SAMBA_PROGRAM_ATTN_PIECES
        return n

    # ------------------------------------------------------------ views
    def _reg_view(self, j: Int) raises -> DeviceBuffer[DType.float32]:
        """Registry tensor `j` as a view of the registry tensor (no copy)."""
        var whole = block_buffer_view(self.s.ex, self.s.buffer(self.reg), self.n_total)
        return whole.create_sub_buffer[DType.float32](self.offsets[j], self.offsets[j + 1] - self.offsets[j])

    def _part_view(self, i: Int, part: Int) raises -> DeviceBuffer[DType.float32]:
        var h = self.w[i] if part == BLOCK_PART_WEIGHTS else self.st[i]
        var n = block_part_floats(self.codes[i], self.lints[i], part)
        return block_buffer_view(self.s.ex, self.s.buffer(h), n)

    # ------------------------------------------------------------ weights
    def _admit_registry(mut self) raises:
        """One device scan of the whole registry (one word back); a hit names
        the flat index, as the resident forward's refusal does."""
        var whole = block_buffer_view(self.s.ex, self.s.buffer(self.reg), self.n_total)
        var bad = device_first_nonfinite(self.s.ex.ctx, whole, self.n_total)
        if bad >= 0:
            raise Error("mojolearn samba program: non-finite parameter at flat registry index " + String(bad))

    def refresh_weights(mut self, param_ptr: MutPointer[Float32, MutUntrackedOrigin]) raises:
        """`load_weights` only when the loaded copy is not the registry at
        `param_ptr` at the current weight epoch (no pass over the weights)."""
        var addrs: List[Int] = [Int(param_ptr)]
        if self.wkey.current(addrs):
            return
        var key = WeightKey.take(addrs)
        self.wkey = WeightKey()
        self.load_weights(param_ptr)
        self.wkey = key^

    def load_weights(mut self, param_ptr: MutPointer[Float32, MutUntrackedOrigin]) raises:
        """The caller's registry into the program: one copy up, one scan, the
        layers' packed weights rewritten device to device in place."""
        self.s.ex.upload(self.s.buffer(self.reg), FP(unsafe_from_address=Int(param_ptr)), self.n_total)
        self._admit_registry()
        var ctx = self.s.ex.ctx.copy()
        for i in range(self.n_layers()):  # driver-loop: over the stack's layers (packed weight copies)
            var sizes = block_weight_sizes(self.codes[i], self.lints[i])
            var offs = block_offsets(sizes, block_guard(self.codes[i]))
            var wv = self._part_view(i, BLOCK_PART_WEIGHTS)
            for k in range(SAMBA_BLOCK_TENSORS):  # driver-loop: over the block's nine registry tensors
                if sizes[k] == 0:
                    continue
                var src = self._reg_view(1 + SAMBA_BLOCK_TENSORS * i + k)
                ctx.enqueue_copy(dst_buf=wv.create_sub_buffer[DType.float32](offs[k], sizes[k]), src_buf=src)

    # ------------------------------------------------------------ state
    def zero_state(mut self) raises:
        """A fresh sequence on every layer: the state tensors zeroed on the
        device and the carried integers zero (no wait)."""
        for i in range(self.n_layers()):  # driver-loop: over the stack's layers (state owners)
            var view = self._part_view(i, BLOCK_PART_STATE)
            view.enqueue_fill(Float32(0))
            var nw = 2 if self.codes[i] == OPS_M3_STEP else 1
            self.s.load_block(self.q[i], self.codes[i], self.lints[i], BLOCK_PART_WORDS, List[CastIn](),
                              List[Int](length=nw, fill=0))
            self.cur[2 * i] = 0
            self.cur[2 * i + 1] = 0

    def load_state(mut self, pieces: List[Int], cursors: List[Int]) raises:
        """The caller's `SambaState` into the state tensors, in place.
        `pieces` = the state addresses in stack order (ten per Mamba-3 layer
        in `_m3_names()` order, then k_cache and v_cache per attention layer);
        `cursors` = per layer `buffered_tokens, pending` (Mamba-3) or
        `cached_tokens, 0` (attention). Enqueued; the caller's arrays must
        stay alive until the next wait."""
        if len(pieces) != self.piece_count() or len(cursors) != 2 * self.n_layers():
            raise Error("samba program: expected " + String(self.piece_count()) + " state addresses and "
                        + String(2 * self.n_layers()) + " cursor words")
        var at = 0
        for i in range(self.n_layers()):  # driver-loop: over the stack's layers (state owners)
            var c0 = cursors[2 * i]
            var c1 = cursors[2 * i + 1]
            var code = self.codes[i]
            var ssizes = block_state_sizes(code, self.lints[i])
            var srcs = List[CastIn](capacity=len(ssizes))
            for k in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
                if pieces[at + k] == 0:
                    raise Error("samba program: null state address at slot " + String(at + k))
                srcs.append(CastIn.f32_c(pieces[at + k], ssizes[k], 1))
            var words = List[Int]()
            if code == OPS_M3_STEP:
                if c0 < 0 or c0 > M3_CHUNK_SIZE:
                    raise Error("samba program: layer " + String(i) + " buffered_tokens out of range")
                if c1 != 0 and c1 != 1:
                    raise Error("samba program: layer " + String(i) + " pending must be 0 or 1")
                if c1 == 1 and c0 != 0:
                    raise Error("samba program: layer " + String(i) + " Input_States pending requires a fresh state")
                words = [c0, c1]
                at += SAMBA_PROGRAM_M3_PIECES
            else:
                if c0 < 0 or c0 > self.smax:
                    raise Error("samba program: layer " + String(i) + " cached_tokens must be in [0, "
                                + String(self.smax) + "], got " + String(c0))
                words = [c0]
                c1 = 0
                at += SAMBA_PROGRAM_ATTN_PIECES
            self.s.load_block(self.st[i], code, self.lints[i], BLOCK_PART_STATE, srcs, List[Int]())
            self.s.load_block(self.q[i], code, self.lints[i], BLOCK_PART_WORDS, List[CastIn](), words)
            self.cur[2 * i] = c0
            self.cur[2 * i + 1] = c1

    def enqueue_export(mut self, pieces: List[Int]) raises -> List[Int]:
        """The state tensors into the caller's arrays (`load_state`'s
        addresses), enqueued with no wait; returns the cursor words in the
        same layout."""
        if len(pieces) != self.piece_count():
            raise Error("samba program: expected " + String(self.piece_count()) + " state addresses")
        var at = 0
        for i in range(self.n_layers()):  # driver-loop: over the stack's layers (state owners)
            var code = self.codes[i]
            var ssizes = block_state_sizes(code, self.lints[i])
            var offs = block_offsets(ssizes, block_guard(code))
            var base = self.s.buffer(self.st[i])
            for k in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
                if pieces[at + k] == 0:
                    raise Error("samba program: null state address at slot " + String(at + k))
                self.s.ex.download_async(FP(unsafe_from_address=pieces[at + k]), base + offs[k], ssizes[k])
            at += len(ssizes)
        return self.cur.copy()

    def export_state(mut self, pieces: List[Int]) raises -> List[Int]:
        """`enqueue_export` behind one wait."""
        var cursors = self.enqueue_export(pieces)
        self.s.ex.sync()
        return cursors^

    # ------------------------------------------------------------ runs
    def _ensure_m(mut self, m: Int) raises:
        """The (B * L)-shaped workspaces: the hidden input tensor of the first
        layer, the ids, the final norm's and the head's buffers."""
        if self.m == m:
            return
        var ctx = self.s.ex.ctx.copy()
        self.s.ex.sync()
        if self.x0 != 0:
            self.s.free(self.x0)
            self.x0 = 0
        var zero = List[Float32](length=m * self.dm, fill=Float32(0))
        self.x0 = self.s.upload(CastIn.f32_c(Int(zero.unsafe_ptr()), m * self.dm, 1), False)
        self.s.ex.sync()
        _ = len(zero)  # keep-alive past the staging copy
        self.ids = ctx.enqueue_create_buffer[DType.int32](m)
        self.sumsq = ctx.enqueue_create_buffer[DType.float32](m)
        self.hn = ctx.enqueue_create_buffer[DType.float32](m * self.dm)
        self.logits = ctx.enqueue_create_buffer[DType.float32](m * self.vocab)
        self.head_ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, self.vocab, self.dm))
        self.m = m

    def _body(mut self, l: Int) raises:
        """The stack on the B * L ids in `self.ids`: embedding -> the layers'
        step ops -> final norm -> head into `self.logits`, enqueued. The
        layers' host words advance with the run."""
        var m = self.b * l
        var dm = self.dm
        var ctx = self.s.ex.ctx.copy()
        var emb = EmbConfig.llama(self.vocab, dm)
        var ew = self._reg_view(0)
        var x0v = block_buffer_view(self.s.ex, self.s.buffer(self.x0), m * dm)
        identical_embedding_forward_into(ctx, x0v, ew, self.ids, m, emb)
        var h = self.x0
        var none = List[Float32]()
        for i in range(self.n_layers()):  # driver-loop: over the stack's layers, one step op each
            var hs: List[Int] = [h, self.w[i], self.st[i], self.q[i]]
            h = self.s.op(self.codes[i], hs, self.lints[i], none)
            if self.codes[i] == OPS_M3_STEP:
                var words = block_step_words(self.s.steps, OPS_M3_STEP, Int(self.s.buffer(self.w[i])),
                                             Int(self.s.buffer(self.st[i])))
                self.cur[2 * i] = words[0]
                self.cur[2 * i + 1] = words[1]
            else:
                self.cur[2 * i] += l
        var hp = self.s.tape.value_of(self.s.ex, self.s.tensor(h))
        var hv = block_buffer_view(self.s.ex, hp, m * dm)
        var nw = self._reg_view(self.norm_f_index())
        var hw = self._reg_view(self.head_index())
        llama_rms_norm(ctx, self.sumsq, self.hn, hv, nw, m, dm, self.eps)
        identical_gemm_into(ctx, self.logits, self.hn, hw, self.head_ws, m, self.vocab, dm, OP_NT)
        _ = ew^
        _ = nw^
        _ = hw^

    def run(mut self, ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
            out_ptr: MutPointer[Float32, MutUntrackedOrigin], l: Int) raises:
        """L tokens per row from the state the program holds: (B, L) int32
        ids -> (B * L, vocab) logits at `out_ptr`. Enqueued; `finish` waits."""
        if l < 1:
            raise Error("samba program: L must be positive")
        self._ensure_m(self.b * l)
        self.s.ex.ctx.enqueue_copy(dst_buf=self.ids, src_ptr=ids_ptr)
        self._body(l)
        self.s.ex.ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=self.logits)

    def finish(mut self) raises:
        """Wait for everything enqueued, then drop the run's computed tensors
        (the uploads, the states and the resident structs stay)."""
        self.s.ex.sync()
        self.s.reset()

    def decode(mut self, ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
               out_ptr: MutPointer[Float32, MutUntrackedOrigin], steps: Int) raises:
        """N tokens per row in one call (decode plan D2): the (B, N) ids
        uploaded once, the one-token body once per token with token t's ids
        taken on the device and its B x vocab logits put into a (B, N, vocab)
        device slab, one download, ONE wait. Same ops, same order, same
        bytes as N one-token runs."""
        decode_slab_check(self.b, steps, self.vocab, "samba decode")
        self._ensure_m(self.b)
        var ctx = self.s.ex.ctx.copy()
        var ids = ctx.enqueue_create_buffer[DType.int32](self.b * steps)
        var out = ctx.enqueue_create_buffer[DType.float32](self.b * steps * self.vocab)
        ctx.enqueue_copy(dst_buf=ids, src_ptr=ids_ptr)
        for t in range(steps):  # driver-loop: over autoregressive decode steps (each reads the state the previous step left), the stack's one-token body each
            decode_slab_take_ids(ctx, self.ids, ids, self.b, steps, t)
            self._body(1)
            decode_slab_put(ctx, out, self.logits, self.b, steps, self.vocab, t)
        ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
        self.finish()
        _ = ids^
        _ = out^

    def close(mut self) raises:
        self.s.close()
