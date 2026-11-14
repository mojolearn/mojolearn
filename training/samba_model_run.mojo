# SPDX-License-Identifier: Apache-2.0
"""Typed whole-model Samba entry (lane py-lm-neural, 2026-11-14).

The Samba stack's whole-model forward, training gradients and decode session
on typed values: no Python object, no provider capsule, no door. A caller
(the public door's UNPACK, or a native check under checks/native_tests/)
names every buffer by address and every engine by its entry-table ref, and
this module drives the SAME engine the door drives
(`training/native_samba_execution.mojo` `NativeSambaModel` /
`NativeSambaLayer`), with the SAME operation codes and address layouts the
door admits (`providers/python/samba_execution_pyapi_door.mojo`
`AdmittedSambaExecution`): no new arithmetic, no respelled kernel.

Engines, all resolved once by the caller from the entry tables:

  primitives   `samba_ref[SAMBA_TABLE_TRAINING](key)` (or
               SAMBA_TABLE_NEURAL_HOST for the CPU inference model):
               embedding, RMSNorm, linear, losses, accumulate
               (PRIMITIVE_ABI_NAME, core/samba_entry_table.mojo).
  a layer      `neural_ref[NEURAL_TABLE_MAMBA_TRAIN](key)` for a Mamba-3
               layer (MAMBA_BLOCK_ABI_NAME: op 5 fresh prefill, op 4
               Mamba-3 backward), `neural_ref[NEURAL_TABLE_BLOCK_TRAIN](key)`
               for an attention layer (TRANSFORMER_BLOCK_ABI_NAME: op 4
               ephemeral forward, op 5 ephemeral backward, op 11 / 33
               carried-cache forward).
  a state      `Mamba3StateRoute.state_ref()` (core/mamba_state_entry_table.mojo,
               MAMBA_STATE_ABI_NAME: op 4 carried-state forward, op 5 one
               token) for a Mamba-3 layer of a decode session.

What the door keeps: Python admission (ids, weights, optimizer, schedule,
generator) and publication of the results; the training step's optimizer
update stays the optimizer family's own entry over the flat gradient this
entry writes (`SambaModelRun.accumulated_gradients`).

Scope (refused by name, as the door refuses them): attention layers with
non-default block options or packed int15 planes do not train (the door's
"TransformerBlock.backward requires default float32 block options"); a
decode session's attention layer runs the float32 cache forward (op 11 on a
device image, op 33 on the CPU inference image), never the packed op 35.

Parallel plan: every operation is one ordered pass over the model's layers
(a driver loop over model structure; each layer's work is the image's own
parallel kernel). Nothing here loops over rows or elements except the
decode session's per-token copy of token t into and out of the (B, d)
buffers, which is the door's host-arm loop, moved.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.native_call_output import NativeCallWorkspace
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_execution import NativeExecutionEndpoint
from core.hostptr import f32_ptr, i32_ptr
from training.native_samba_execution import NativeSambaModel, NativeSambaLayer

comptime SAMBA_LAYER_MAMBA3 = 0
"""A Mamba-3 layer (`Mamba3Block`)."""
comptime SAMBA_LAYER_ATTENTION = 1
"""An attention layer (`TransformerBlock`)."""

comptime _M3_WEIGHTS = 9
"""Mamba-3 weight slots in the block ABI (`Mamba3Block._W_NAMES` order)."""
comptime _M3_PIECES = 10
"""Mamba-3 carried-state pieces (`training/samba_inference_run.mojo` `_m3_names`)."""


def samba_entry_invoke(context: Int, operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int, error_context: Int,
                       sink: NativeDescriptorError) abi("C") -> Int:
    """`NativeExecutionInvoke` over one table entry. `context` is the address
    of an `EntryRef` its owner (`SambaEndpoints`) keeps at a stable heap
    address for the endpoint's life; the call is the entry's own span call
    (`entry_ref_call`, no heap). No session prefix, no lock and no GIL: a
    caller of this entry holds no Python state."""
    try:
        if context == 0 or result == 0:
            raise Error("samba model entry: invalid endpoint context or result")
        var entry = MutPointer[EntryRef, MutUntrackedOrigin](unsafe_from_address=context)[]
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = entry_ref_call(
            entry, operation, aa, na, pp, np, ff, nf)
        return 0
    except problem:
        native_descriptor_report(problem, error_context, sink)
        return 1


struct SambaEndpoints(Movable):
    """The stable storage the model's endpoints name: one heap `EntryRef` and
    one call workspace per endpoint (an inner List's storage does not move
    when the outer List grows or the owner moves). Built once per run."""

    var entries: List[List[EntryRef]]
    var workspaces: List[List[NativeCallWorkspace]]

    def __init__(out self):
        self.entries = List[List[EntryRef]]()
        self.workspaces = List[List[NativeCallWorkspace]]()

    def endpoint(mut self, entry: EntryRef) -> NativeExecutionEndpoint:
        var held = List[EntryRef]()
        held.append(entry)
        self.entries.append(held^)
        var workspace = List[NativeCallWorkspace]()
        workspace.append(NativeCallWorkspace())
        self.workspaces.append(workspace^)
        var last = len(self.entries) - 1
        return NativeExecutionEndpoint(samba_entry_invoke, Int(self.entries[last].unsafe_ptr()),
            Int(self.workspaces[last].unsafe_ptr()))


@fieldwise_init
struct SambaLayerRequest(Copyable, Movable):
    """One layer of the stack, typed.

    `kind`: SAMBA_LAYER_MAMBA3 or SAMBA_LAYER_ATTENTION. `entry`: the layer's
    block image (forward and backward run through it, as the door's provider
    does). `state_entry`: a Mamba-3 layer's carried-state image (decode
    sessions only; unused otherwise). `weights`: the block's weight addresses
    in its ABI order (Mamba-3: the nine `_W_NAMES` slots). `optional`: an
    attention layer's optional weight addresses (empty with default options).
    `words`: an attention layer's transformer word vector
    (`core/generation_metadata.mojo` `GenerationLayerWords.transformer`:
    [B, L, d, n_heads, n_kv, head_dim, ..., capacity, cached, window, tail]).
    `heads`: a Mamba-3 layer's `nheads`. `backward_words`: an attention
    layer's [n_heads, n_kv_heads, head_dim, intermediate, window].
    `gradient_sizes`: each weight's element count, in `weights` order."""

    var kind: Int
    var entry: EntryRef
    var state_entry: EntryRef
    var weights: List[Int]
    var optional: List[Int]
    var words: List[Int]
    var heads: Int
    var backward_words: List[Int]
    var gradient_sizes: List[Int]


@fieldwise_init
struct SambaModelRequest(Copyable, Movable):
    """The whole stack, typed: the primitive image, the embedding, final norm
    and head addresses ((V, d), (d,), (V, d); the head is the embedding when
    `tied`), `norm_eps`, the shape (`batch` rows of `length` tokens, `width`
    = d_model, `vocab`), the head-loss routes the image publishes
    (`fused_head_loss`: SAMBA_PRIM_HEAD_LOSS, device images; `chunked_head`:
    SAMBA_PRIM_CHUNKED_HEAD_TRAIN) and the layers in stack order.
    `cpu_inference`: the attention layers run on the CPU inference image
    (decode sessions: cache forward op 33 instead of 11)."""

    var primitives: EntryRef
    var embedding: Int
    var norm: Int
    var head: Int
    var norm_eps: Float64
    var batch: Int
    var length: Int
    var width: Int
    var vocab: Int
    var tied: Bool
    var fused_head_loss: Bool
    var chunked_head: Bool
    var cpu_inference: Bool
    var layers: List[SambaLayerRequest]


def _m3_report_sizes(b: Int, h: Int) -> List[Int]:
    """The four Mamba-3 report counts ((B, H, 64, 128), (B, H, 128),
    (B, H, 64), (B, H, 32)), as the door sizes them."""
    return [b * h * 64 * 128, b * h * 128, b * h * 64, b * h * 32]


def _m3_piece_counts(b: Int, h: Int) -> List[Int]:
    """Element counts of the ten carried-state pieces, in `_m3_names` order
    (shapes of `training/samba_inference_run.mojo` `_m3_shapes`)."""
    return [b * h * 32, b * h * 64 * 128, b * 64 * h * 128, b * 64 * h * 128, b * 64 * h * 64,
            b * 64 * h, b * 64 * h, b * 64 * h, b * h * 128, b * h * 64]


def _fresh_words(words: List[Int], b: Int, l: Int) -> List[Int]:
    """The ephemeral attention forward's words (op 4): the block's word vector
    at (B, L, capacity L, cached 0) without the capacity and cached slots, as
    the door spells them."""
    var p = words.copy()
    p[0] = b
    p[1] = l
    p[7] = l
    p[8] = 0
    var fresh = List[Int]()
    for j in range(7):  # small-loop(seven fresh transformer shape words)
        fresh.append(p[j])
    fresh.append(p[9])
    for j in range(10, len(p)):  # small-loop(the admitted block-option tail)
        fresh.append(p[j])
    return fresh^


def _layer_check(layer: SambaLayerRequest, training: Bool) raises:
    if layer.kind == SAMBA_LAYER_MAMBA3:
        if len(layer.weights) != _M3_WEIGHTS or layer.heads <= 0:
            raise Error("samba model entry: a Mamba-3 layer takes nine weight addresses and positive heads")
    elif layer.kind == SAMBA_LAYER_ATTENTION:
        if len(layer.words) < 10:
            raise Error("samba model entry: an attention layer needs its transformer word vector")
        if training and len(layer.optional) != 0:
            raise Error("NotImplementedError: TransformerBlock.backward requires default float32 block options (the default block options only)")
        if training and len(layer.backward_words) != 5:
            raise Error("samba model entry: attention backward takes five dimension words")
    else:
        raise Error("samba model entry: unknown layer kind " + String(layer.kind))
    if training and len(layer.gradient_sizes) != len(layer.weights):
        raise Error("samba model entry: one gradient size per weight")
    for address in layer.weights:  # driver-loop: over the layer's weight slots (admission)
        if address == 0:
            raise Error("samba model entry: null weight address")


def _stateless_layer(mut endpoints: SambaEndpoints, layer: SambaLayerRequest, b: Int, l: Int, width: Int,
                     training: Bool) raises -> NativeSambaLayer:
    """One layer of a stateless forward (and its backward when `training`):
    the door's non-session, non-owned, non-tape route."""
    _layer_check(layer, training)
    var endpoint = endpoints.endpoint(layer.entry)
    var a: List[Int] = [0]
    for address in layer.weights:  # driver-loop: over the layer's weight slots
        a.append(address)
    var output_slot = len(a)
    a.append(0)
    if layer.kind == SAMBA_LAYER_MAMBA3:
        for j in range(4):  # small-loop(four Mamba report pointer slots)
            a.append(0)
        var native = NativeSambaLayer(endpoint, 5, a^, [b, l, width], [], 0, output_slot, output_slot + 1,
            _m3_report_sizes(b, layer.heads))
        if training:
            native.backward_operation = 4
            native.backward_parameters = [b, l, width]
            native.backward_weights = layer.weights.copy()
            native.gradient_sizes = layer.gradient_sizes.copy()
            native.validate_gradient = True
        return native^
    for address in layer.optional:  # driver-loop: over the attention layer's optional weight slots
        a.append(address)
    var native = NativeSambaLayer(endpoint, 4, a^, _fresh_words(layer.words, b, l), [], 0, output_slot, -1, [])
    if training:
        native.backward_operation = 5
        var parameters: List[Int] = [b, l, width]
        for word in layer.backward_words:  # small-loop(five attention backward dimension words)
            parameters.append(word)
        native.backward_parameters = parameters^
        native.backward_weights = layer.weights.copy()
        native.gradient_sizes = layer.gradient_sizes.copy()
        native.validate_gradient = False
    return native^


struct SambaModelRun(Movable):
    """A stateless whole-model run: forward to logits, the loss and every
    parameter gradient, and the accumulated flat gradient of a training step.
    Built once per (model, shape); every call reuses its endpoints."""

    var endpoints: SambaEndpoints
    var model: NativeSambaModel
    var tied: Bool
    var fused: Bool
    var chunked: Bool

    def __init__(out self, request: SambaModelRequest, training: Bool) raises:
        if request.batch <= 0 or request.length <= 0 or request.width <= 0 or request.vocab <= 0:
            raise Error("samba model entry: batch, length, width and vocab must be positive")
        if request.embedding == 0 or request.norm == 0 or request.head == 0:
            raise Error("samba model entry: null embedding, norm or head address")
        var endpoints = SambaEndpoints()
        var primitive = endpoints.endpoint(request.primitives)
        var model = NativeSambaModel(primitive, request.embedding, request.norm,
            request.embedding if request.tied else request.head, request.norm_eps,
            request.batch * request.length, request.width, request.vocab)
        for i in range(len(request.layers)):  # driver-loop: over the model's layers in stack order
            model.layers.append(_stateless_layer(endpoints, request.layers[i], request.batch, request.length,
                request.width, training))
        self.endpoints = endpoints^
        self.model = model^
        self.tied = request.tied
        self.fused = request.fused_head_loss
        self.chunked = request.chunked_head

    def set_dropout(mut self, probability: Float64, seed: UInt64, stream: Int):
        """The embedding dropout of a training step (the door's words): rate
        `probability` (0 turns it off), the generator `seed` and the step's
        `stream` position."""
        if probability <= 0.0:
            self.model.dropout = False
            return
        var p = Float32(probability)
        self.model.dropout = True
        self.model.dropout_words = [self.model.rows * self.model.width, 0, Int(seed & 0xFFFFFFFF),
            Int((seed >> 32) & 0xFFFFFFFF), stream, 2]
        self.model.dropout_scalars = [Float64(p), Float64(Float32(1.0 / (1.0 - Float64(p))))]

    def forward(mut self, ids: Int, logits: Int) raises:
        """`ids` int32 (B*L,), `logits` float32 (B*L, V) written whole: the
        embedding, every layer in order, the final norm and the head."""
        if ids == 0 or logits == 0:
            raise Error("samba model entry: null ids or logits address")
        self.model.forward_hidden(ids, False)
        _ = self.model.primitive.call(4, [logits, Int(self.model.normalized.unsafe_ptr()), self.model.head],
            [self.model.rows, self.model.vocab, self.model.width], [])

    def gradients(mut self, ids: Int, targets: Int) raises -> Float64:
        """The mean token loss of (`ids`, `targets`) (int32 (B*L,), -100
        skipped) and every gradient, held by the run: `gradient_address(kind,
        layer, slot)` (0 embedding, 1 final norm, 2 head, 3 a layer's weight
        `slot`)."""
        if ids == 0 or targets == 0:
            raise Error("samba model entry: null ids or targets address")
        return self.model.gradients(ids, targets, 0, True, self.tied, self.fused, self.chunked)

    def gradient_address(self, kind: Int, layer: Int, slot: Int) -> Int:
        return self.model.gradient_address(kind, layer, slot)

    def accumulated_gradients(mut self, ids: Int, targets: Int, accumulation: Int, total_count: Int,
                              offsets: List[Int], kinds: List[Int], layers: List[Int], slots: List[Int],
                              output: Int) raises -> Float64:
        """The training step's gradient half: `accumulation` microbatches of
        this run's shape (ids and targets hold accumulation * B * L tokens),
        each gradient copied into the flat registry (`offsets` and the
        (kind, layer, slot) of each entry) and folded in order into `output`
        (float32, offsets[-1]); returns the folded loss. `total_count` is the
        whole step's counted targets."""
        if output == 0 or len(kinds) != len(layers) or len(kinds) != len(slots) or len(offsets) != len(kinds) + 1:
            raise Error("samba model entry: the gradient registry is incomplete")
        return self.model.accumulated_gradients(ids, targets, accumulation, total_count, self.tied, self.fused,
            self.chunked, offsets, kinds, layers, slots, output, String(""))


def _publish_counters(mut counters: List[Int], mut pending: List[Int], model: NativeSambaModel):
    """A run's completed layers publish their carried counts (attention
    cached, Mamba-3 buffered) back to the session; pending clears."""
    for i in range(len(model.layers)):  # driver-loop: over the layers' completed counters
        if model.layers[i].completed:
            counters[i] = model.state_counters[i]
            pending[i] = 0


struct SambaDecodeSessionRun(Movable):
    """A decode session over the whole stack on typed, session-owned state:
    a prefill of L tokens per row, then one token per row per `step`,
    carrying each layer's state (Mamba-3 pieces and buffered count; attention
    k/v caches and cached count). Equal, byte for byte, to the door's
    stateful forward of the same layers (`AdmittedSambaExecution` with a
    state): the same operations, words and slots."""

    var endpoints: SambaEndpoints
    var request: SambaModelRequest
    var capacity: Int
    var pieces: List[List[List[Float32]]]
    var counters: List[Int]
    var pending: List[Int]
    var step_model: NativeSambaModel
    var open: Bool

    def __init__(out self, request: SambaModelRequest, capacity: Int) raises:
        """`capacity`: the attention caches' token capacity per row (prefill
        plus every step); `request.length` is ignored (each call names its
        own)."""
        if request.batch <= 0 or request.width <= 0 or request.vocab <= 0 or capacity <= 0:
            raise Error("samba decode session: batch, width, vocab and capacity must be positive")
        var b = request.batch
        var pieces = List[List[List[Float32]]]()
        var counters = List[Int]()
        var pending = List[Int]()
        for i in range(len(request.layers)):  # driver-loop: over the model's layers, one state each
            ref layer = request.layers[i]
            _layer_check(layer, False)
            var held = List[List[Float32]]()
            if layer.kind == SAMBA_LAYER_MAMBA3:
                var counts = _m3_piece_counts(b, layer.heads)
                for j in range(_M3_PIECES):  # small-loop(ten carried-state pieces)
                    held.append(List[Float32](length=counts[j], fill=0))
            else:
                var window = layer.words[9]
                var slots = capacity if window == 0 else window
                var count = b * layer.words[4] * slots * layer.words[5]
                held.append(List[Float32](length=count, fill=0))
                held.append(List[Float32](length=count, fill=0))
            pieces.append(held^)
            counters.append(0)
            pending.append(0)
        var endpoints = SambaEndpoints()
        var primitive = endpoints.endpoint(request.primitives)
        self.step_model = NativeSambaModel(primitive, request.embedding, request.norm,
            request.embedding if request.tied else request.head, request.norm_eps, b, request.width, request.vocab)
        self.endpoints = endpoints^
        self.request = request.copy()
        self.capacity = capacity
        self.pieces = pieces^
        self.counters = counters^
        self.pending = pending^
        self.open = True
        var layers = self._layers(1, True)
        self.step_model.layers = layers^
        for i in range(len(self.request.layers)):  # driver-loop: over the layers' state indices
            self.step_model.state_counters.append(0)
            self.step_model.state_pending.append(0)

    def _layers(mut self, l: Int, step: Bool) raises -> List[NativeSambaLayer]:
        """The stateful layers for `l` tokens per row: Mamba-3 on its state
        image (op 5 one token, op 4 a sequence), attention on its block image
        (cache forward, op 11 device / 33 CPU inference)."""
        var b = self.request.batch
        var width = self.request.width
        var result = List[NativeSambaLayer]()
        for i in range(len(self.request.layers)):  # driver-loop: over the model's layers in stack order
            var layer = self.request.layers[i].copy()  # once per session build or prefill, never per token
            var a: List[Int] = [0]
            for address in layer.weights:  # driver-loop: over the layer's weight slots
                a.append(address)
            if layer.kind == SAMBA_LAYER_MAMBA3:
                var endpoint = self.endpoints.endpoint(layer.state_entry)
                for j in range(_M3_PIECES):  # small-loop(ten carried-state piece addresses)
                    a.append(Int(self.pieces[i][j].unsafe_ptr()))
                var output_slot = len(a)
                a.append(0)
                for j in range(4):  # small-loop(four Mamba report pointer slots)
                    a.append(0)
                var words: List[Int] = [b, width, 0, 0] if step else [b, l, width, 0, 0]
                var native = NativeSambaLayer(endpoint, 5 if step else 4, a^, words^, [], 0, output_slot,
                    output_slot + 1, _m3_report_sizes(b, layer.heads))
                native.counter_slot = 2 if step else 3
                native.pending_slot = native.counter_slot + 1
                native.state_index = i
                result.append(native^)
            else:
                var endpoint = self.endpoints.endpoint(layer.entry)
                a.append(Int(self.pieces[i][0].unsafe_ptr()))
                a.append(Int(self.pieces[i][1].unsafe_ptr()))
                var output_slot = len(a)
                a.append(0)
                for address in layer.optional:  # driver-loop: over the attention layer's optional weight slots
                    a.append(address)
                var words = layer.words.copy()
                words[0] = b
                words[1] = l
                words[7] = self.capacity
                words[8] = 0
                var native = NativeSambaLayer(endpoint, 33 if self.request.cpu_inference else 11, a^, words^, [], 0,
                    output_slot, -1, [])
                native.counter_slot = 8
                native.pending_slot = -1
                native.state_index = i
                result.append(native^)
        return result^

    def _check(self, l: Int, ids: Int, logits: Int) raises:
        if not self.open:
            raise Error("samba decode session: the session is closed")
        if ids == 0 or logits == 0 or l <= 0:
            raise Error("samba decode session: null ids or logits, or no tokens")
        for i in range(len(self.request.layers)):  # driver-loop: over the layers' cached counts
            if self.request.layers[i].kind == SAMBA_LAYER_ATTENTION and self.counters[i] + l > self.capacity:
                raise Error("neural block: sequence exceeds state capacity or max_positions")

    def prefill(mut self, ids: Int, length: Int, logits: Int) raises:
        """`ids` int32 (B*length,) row-major, `logits` float32 (B*length, V):
        the stack over `length` tokens per row, continuing the carried state."""
        self._check(length, ids, logits)
        var b = self.request.batch
        var model = NativeSambaModel(self.step_model.primitive, self.step_model.embedding, self.step_model.norm,
            self.step_model.head, self.step_model.norm_eps, b * length, self.request.width, self.request.vocab)
        model.layers = self._layers(length, False)
        model.state_counters = self.counters.copy()
        model.state_pending = self.pending.copy()
        try:
            model.forward_hidden(ids, False)
        except problem:
            _publish_counters(self.counters, self.pending, model)
            raise problem
        _publish_counters(self.counters, self.pending, model)
        _ = model.primitive.call(4, [logits, Int(model.normalized.unsafe_ptr()), model.head],
            [model.rows, model.vocab, model.width], [])
        _ = model^

    def step(mut self, ids: Int, logits: Int) raises:
        """One token per row: `ids` int32 (B,), `logits` float32 (B, V)."""
        self._check(1, ids, logits)
        for i in range(len(self.counters)):  # driver-loop: over the layers' carried counts (no allocation)
            self.step_model.state_counters[i] = self.counters[i]
            self.step_model.state_pending[i] = self.pending[i]
        var failure = String("")
        try:
            self.step_model.forward_hidden(ids, False)
        except problem:
            failure = String(problem)
        for i in range(len(self.counters)):  # driver-loop: over the layers' completed counters
            if self.step_model.layers[i].completed:
                self.counters[i] = self.step_model.state_counters[i]
                self.pending[i] = 0
        if failure.byte_length() != 0:
            raise Error(failure)
        _ = self.step_model.primitive.call(4, [logits, Int(self.step_model.normalized.unsafe_ptr()),
            self.step_model.head], [self.step_model.rows, self.step_model.vocab, self.step_model.width], [])

    def decode(mut self, ids: Int, tokens: Int, logits: Int) raises:
        """`tokens` steps: `ids` int32 (B, tokens) row-major, `logits` float32
        (B, tokens, V); equal to `tokens` calls of `step`."""
        var b = self.request.batch
        var v = self.request.vocab
        var xt = List[Int32](length=b, fill=0)
        var yt = List[Float32](length=b * v, fill=0)
        var src = i32_ptr(ids)
        var dst = f32_ptr(logits)
        for t in range(tokens):  # driver-loop: over decode tokens, one whole-model step each
            for r in range(b):  # driver-loop: over the session's rows, token t's id
                xt[r] = src[r * tokens + t]
            self.step(Int(xt.unsafe_ptr()), Int(yt.unsafe_ptr()))
            for r in range(b):  # driver-loop: over the session's rows, token t's logits
                unsafe_memcpy(dest=dst + (r * tokens + t) * v, src=yt.unsafe_ptr() + r * v, count=v)
        _ = xt^
        _ = yt^

    def positions(self, layer: Int) -> Int:
        """The tokens layer `layer` has consumed per row (attention: cached;
        Mamba-3: buffered, the carried count its image returns)."""
        return self.counters[layer]

    def state_address(self, layer: Int, piece: Int) -> Int:
        """The address of layer `layer`'s state piece `piece` (Mamba-3: the
        ten `_m3_names` pieces; attention: 0 k cache, 1 v cache)."""
        return Int(self.pieces[layer][piece].unsafe_ptr())

    def close(mut self):
        self.open = False
