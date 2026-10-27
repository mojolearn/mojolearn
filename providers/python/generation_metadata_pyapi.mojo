# SPDX-License-Identifier: Apache-2.0
# Operation-local Python ownership/admission for native generation metadata.
# No snapshot survives a public forward/generate/read call. A stream's next
# read recaptures current model fields and owners; this adds no thread-safety
# promise for concurrent model mutation during execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_type, py_call, py_contains
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_address
from core.generation_metadata import GenerationLayerWords


struct GenerationWeightSnapshot(Copyable, Movable):
    var owners: List[PythonObject]
    var addresses: List[Int]
    var refresh: List[Int]

    def __init__(out self, values: PythonObject, zero_from: Int = -1) raises:
        self.owners = List[PythonObject]()
        self.addresses = List[Int]()
        self.refresh = List[Int]()
        var cls = GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls
        for j in range(len(values)):  # driver-loop: admit each stable weight slot once
            var value = values[j]
            self.owners.append(value)
            if value is None or (zero_from >= 0 and j >= zero_from):
                self.addresses.append(0)
            elif py_type(value) is cls:
                # generation_address checks C order. The exact Array's owner
                # remains in self.owners after its native address is captured.
                var keep = List[PythonObject]()
                self.addresses.append(generation_address(value, keep))
            else:
                # The legacy foreign-buffer route copies bytes on each use.
                # Preserve that behavior (including aliases to mutable state),
                # rather than silently freezing a copied weight for this call.
                self.addresses.append(0)
                self.refresh.append(j)

    def resolve(self, mut keep: List[PythonObject]) raises -> List[Int]:
        var result = self.addresses.copy()
        for k in range(len(self.refresh)):  # driver-loop: retained foreign-buffer copy compatibility
            var index = self.refresh[k]
            result[index] = generation_address(self.owners[index], keep)
        return result^


struct GenerationLayerSnapshot(Copyable, Movable):
    var block: PythonObject
    var words: GenerationLayerWords
    var weights: GenerationWeightSnapshot
    var optional: GenerationWeightSnapshot
    var packed: GenerationWeightSnapshot
    var profile: PythonObject
    var profile_changed: Bool

    def __init__(out self, block: PythonObject, kind: String, planes: Bool) raises:
        self.block = block
        self.weights = GenerationWeightSnapshot(block["_w"], 2 if planes else -1)
        self.profile = PythonObject(None)
        self.profile_changed = False
        var parameters = List[Int]()
        var inner = 0
        var heads = 0
        var conv = 0
        var low = Float64(0)
        var high = Float64(0)
        var optional = Python.list()
        var packed = Python.list()
        if kind == "transformer":
            parameters = [0, 0, Int(py=block["d_model"]), Int(py=block["n_heads"]),
                          Int(py=block["n_kv_heads"]), Int(py=block["head_dim"]),
                          Int(py=block["intermediate"]), 0, 0, Int(py=block["window"])]
            var tail = block["_opts_tail"]
            for j in range(len(tail)):  # driver-loop: admitted immutable transformer option words
                parameters.append(Int(py=tail[j]))
            optional = block["_wopt"]
            if planes:
                var triples = block["_int15"]
                for j in range(len(triples)):  # driver-loop: packed weight plane registry
                    var trip = triples[j]
                    for k in range(3):  # small-loop(int15 plane triple)
                        _ = py_call(packed.append, PythonObject(None) if trip is None else trip[k])
            self.profile = py_get(block, "_arithmetic_profile", PythonObject(None))
            self.profile_changed = Bool(py=py_get(block, "_arithmetic_profile_changed", PythonObject(False)))
        elif kind == "mamba1":
            inner = Int(py=block["d_inner"])
        else:
            heads = Int(py=block["nheads"])
            conv = Int(py=block["conv_dim"])
            var limits = block["dt_limit"]
            low = Float64(py=limits[0])
            high = Float64(py=limits[1])
        self.words = GenerationLayerWords(parameters^, inner, heads, conv, low, high)
        self.optional = GenerationWeightSnapshot(optional)
        self.packed = GenerationWeightSnapshot(packed)


struct GenerationHeadSnapshot(Copyable, Movable):
    var entries: PythonObject
    var native: PythonObject
    var width: Int
    var vocab: Int
    var normalize: Bool
    var packed: Bool
    var norm_eps: Float64
    var weights: GenerationWeightSnapshot

    def __init__(out self, model: PythonObject, entries: PythonObject, width: Int, vocab: Int,
                 normalize: Bool, packed: Bool) raises:
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.width = width
        self.vocab = vocab
        var norm = model._norm if normalize else PythonObject(None)
        var planes = model._head_int15
        self.normalize = norm is not None
        self.packed = planes is not None and packed
        self.norm_eps = Float64(py=model.norm_eps) if self.normalize else Float64(0)
        var values = Python.list(norm, PythonObject(None) if self.packed else model._head)
        if self.packed:
            for k in range(3):  # small-loop(int15 head plane triple)
                _ = py_call(values.append, planes[k])
        self.weights = GenerationWeightSnapshot(values)


struct GenerationOperationSnapshot(Copyable, Movable):
    var model: PythonObject
    var entries: PythonObject
    var native: PythonObject
    var kind: String
    var width: Int
    var vocab: Int
    var planes: Bool
    var embedding: GenerationWeightSnapshot
    var head: GenerationHeadSnapshot
    var layers: List[GenerationLayerSnapshot]

    def __init__(out self, model: PythonObject, entries: PythonObject, blocks: List[PythonObject]) raises:
        self.model = model
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.kind = String(model.kind)
        self.width = Int(py=model.d_model)
        self.vocab = Int(py=model.vocab_size)
        self.embedding = GenerationWeightSnapshot(Python.list(model._embed))
        self.head = GenerationHeadSnapshot(model, entries, self.width, self.vocab, True, True)
        self.planes = self.head.packed and py_contains(entries, "forward15")
        self.layers = List[GenerationLayerSnapshot]()
        for i in range(len(blocks)):  # driver-loop: capture stable layer metadata once per operation
            self.layers.append(GenerationLayerSnapshot(blocks[i], self.kind, self.planes))


# This admission is an operation-state consumer of the common descriptor and
# buffer boundary, not a new public adapter. Only the native engine below its
# boundary owns token/layer execution.
from bindings.py_balanced import py_setitem, py_setattr
from bindings.generation_buffers import generation_mutable
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider
from providers.python.generation_session_pyapi import (
    GenerationNativeProvider, GenerationNativeSession, generation_connect_native_session,
    generation_native_session_matches,
)
from core.generation_metadata import generation_transformer_addresses
from transformer.native_generation_engine import NativeGenerationLayer, NativeGenerationModel
from transformer.native_session_call import generation_session_forward


struct GenerationEngineAdmission(Movable):
    var snapshot: GenerationOperationSnapshot
    var state: PythonObject
    var states: List[PythonObject]
    var links: List[PythonObject]
    var buffers: List[PythonObject]
    var native: NativeGenerationModel
    var reserved: List[Int]
    var published: Bool

    def __init__(out self, snapshot: GenerationOperationSnapshot, state: PythonObject, batch: Int, length: Int) raises:
        self.snapshot = snapshot
        self.state = state
        self.states = List[PythonObject]()
        self.links = List[PythonObject]()
        self.buffers = List[PythonObject]()
        self.reserved = List[Int]()
        self.published = False
        var entries = snapshot.entries
        var provider = py_get(entries, "native_session_provider")
        if snapshot.kind != "transformer" or snapshot.head.packed or snapshot.planes or provider is None or snapshot.native is None:
            raise Error("CausalLM: native generation engine requires admitted nonresident fp32 Transformer descriptors")
        if len(snapshot.embedding.refresh) != 0 or len(snapshot.head.weights.refresh) != 0:
            raise Error("CausalLM: native generation engine foreign-weight descriptor admission is pending")
        var primitive = snapshot.native.downcast_value_ptr[NeuralDescriptorProvider]()
        var model_limit = snapshot.model.max_positions
        var limited = model_limit is not None
        self.native = NativeGenerationModel(primitive[].entry, snapshot.embedding.addresses[0],
            snapshot.head.weights.addresses[0], snapshot.head.weights.addresses[1], snapshot.head.norm_eps,
            batch, snapshot.width, snapshot.vocab, Int(py=state.positions), Int(py=state.max_tokens),
            limited, Int(py=model_limit) if limited else 0)
        var layers = state.layers
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        var api = provider.downcast_value_ptr[GenerationNativeProvider]()
        for i in range(len(snapshot.layers)):  # driver-loop: admit layer/state owners once before native execution
            var layer = snapshot.layers[i]
            var st = layers[i]
            var native_layer = NativeGenerationLayer(api[].forward, 0, [], [], 0)
            try:
                if len(layer.weights.refresh) != 0 or len(layer.optional.refresh) != 0:
                    raise Error("CausalLM: native generation engine foreign-weight descriptor admission is pending")
                if py_call(g[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
                    raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
                var old_profile = py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None))
                var cached = Int(py=st.cached_tokens)
                native_layer.cached = cached
                if old_profile is None and cached != 0 and layer.profile_changed:
                    raise Error("CausalLM: cached state requires arithmetic profile metadata")
                if old_profile is not None and layer.profile is not None and not Bool(py=old_profile == layer.profile):
                    raise Error("CausalLM: state arithmetic profile differs")
                native_layer.profile_admitted = True
                var capacity = Int(py=st.max_tokens)
                var cache_count = layer.words.cache_count(batch, capacity)
                var key = st.k_cache
                var value = st.v_cache
                if py_type(key) is not g[].array_cls or py_type(value) is not g[].array_cls:
                    raise Error("CausalLM: native generation engine foreign mutable-state descriptor admission is pending")
                var copies = List[PythonObject]()
                var key_address = generation_mutable(key, [cache_count], self.buffers, copies, "k_cache")
                var value_address = generation_mutable(value, [cache_count], self.buffers, copies, "v_cache")
                var block = layer.block
                if block["_native_session"] is None:
                    py_setitem(block, "_native_session", py_call(entries["create"]))
                    py_setitem(block, "_session_binding", entries["block_binding"])
                var session = block["_native_session"]
                var linked = py_get(block, "_native_forward_owner")
                if linked is None or not generation_native_session_matches(linked, provider, session):
                    linked = generation_connect_native_session(provider, session)
                    py_setitem(block, "_native_forward_owner", linked)
                var held = linked.downcast_value_ptr[GenerationNativeSession]()
                var addresses = generation_transformer_addresses(0, layer.weights.addresses.copy(),
                    key_address, value_address, 0, layer.optional.addresses.copy(), [])
                native_layer.handle = held[].handle
                native_layer.addresses = addresses^
                native_layer.parameters = layer.words.transformer_params(batch, length, capacity, cached)
                self.links.append(linked)
            except problem:
                # Defer this refusal to its original execution position. Earlier
                # layers must retain their normal partial progress on failure.
                native_layer.failure = String(problem)
            self.states.append(st)
            self.native.layers.append(native_layer^)
        _ = provider
        _ = layers

    def reserve(mut self) raises:
        # The existing ABI specifies reservation under the GIL. Retain that
        # contract for every shared session while the native operation runs.
        for i in range(len(self.native.layers)):  # driver-loop: reserve admitted native session owners
            if len(self.native.layers[i].failure) != 0:
                break
            var already_reserved = False
            for index in self.reserved:  # driver-loop: tied blocks may share one admitted session
                if self.native.layers[index].handle == self.native.layers[i].handle:
                    already_reserved = True
                    break
            if already_reserved:
                continue
            try:
                _ = generation_session_forward(0, self.native.layers[i].forward,
                    self.native.layers[i].handle, [], [])
                self.reserved.append(i)
            except problem:
                # Preserve the legacy earlier-layer progress and invalidation
                # when this session is busy/closed. Never unlock another owner.
                self.native.layers[i].failure = String(problem)
                break

    def release(mut self) raises:
        while len(self.reserved) > 0:  # driver-loop: release only reservations owned by this operation
            var i = self.reserved.pop()
            _ = generation_session_forward(2, self.native.layers[i].forward,
                self.native.layers[i].handle, [], [])

    def publish(mut self) raises:
        if self.published:
            return
        for i in range(len(self.states)):  # driver-loop: publish completed/partial native state at operation boundary
            if self.native.layers[i].attempted:
                py_setattr(self.states[i], "cached_tokens", PythonObject(self.native.layers[i].cached))
            if self.native.layers[i].visited and self.native.layers[i].profile_admitted and self.snapshot.layers[i].profile is not None:
                py_setattr(self.states[i], "arithmetic_profile", self.snapshot.layers[i].profile)
        py_setattr(self.state, "positions", PythonObject(self.native.position))
        py_setattr(self.state, "_native_execution_route", PythonObject("transformer-nonresident-native-engine-v1"))
        py_setattr(self.state, "_native_engine_forwards", PythonObject(self.native.forwards))
        if self.native.failed:
            py_setattr(self.state, "_usable", PythonObject(False))
        var primitive = self.snapshot.native.downcast_value_ptr[NeuralDescriptorProvider]()
        for i in range(6):  # small-loop(common primitive diagnostic counters)
            primitive[].calls[i] += self.native.calls[i]
        self.published = True
        _ = self.buffers
        _ = self.links
        _ = self.snapshot
