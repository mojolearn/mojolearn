# SPDX-License-Identifier: Apache-2.0
# Operation-local Python ownership/admission for native generation metadata.
# No snapshot survives a public forward/generate/read call. A stream's next
# read recaptures current model fields and owners; this adds no thread-safety
# promise for concurrent model mutation during execution.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_get, py_type, py_call, py_contains
from std.memory import memcpy
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, generation_array_snapshot, generation_buffer_view
from bindings.native_loaded_module import native_loaded_module
from bindings.native_instance_item import native_instance_item
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.causal_model_config import model_dict, model_plain, model_string, model_integer, model_sequence
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from core.borrowed_buffer import BorrowedBufferLayout
from core.generation_metadata import GenerationLayerWords


def _generation_fields(value: PythonObject, names: PythonObject, allow_state: Bool = False) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is bi.dict:
        model_dict(value)
        return value
    if cls is native_loaded_module("types").SimpleNamespace:
        # Exact private C namespace: its __dict__ is a native member rather
        # than the public Python class getset expected by _fields. Read the
        # actual native dictionary without executing a descriptor lookup.
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr, PyObjectPtr())
        if not raw:
            raise cpy.unsafe_get_error()
        var fields = PythonObject(from_owned=raw)
        model_dict(fields)
        return fields
    if not allow_state:
        raise Error("TypeError: generation requires an admitted native metadata snapshot")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var admitted = cls is g[].state_cls
    model_dict(g[].layer_classes)
    var layer_classes = py_call(bi.list, g[].layer_classes.values())
    for i in range(len(layer_classes)):  # small-loop(connected internal state classes)
        admitted = admitted or cls is layer_classes[i]
    if not admitted:
        raise Error("TypeError: generation requires a connected native state record")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: inspect state descriptor ancestry before storage reads
        var row = _class_fields(ancestry[i])
        if py_contains(row, "__getattr__"):
            raise Error("TypeError: generation cannot invoke state attribute hooks")
        for hook in ["__getattribute__", "__setattr__"]:  # small-loop(state access hooks)
            if py_contains(row, hook) and row[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: generation cannot invoke state attribute hooks")
        for j in range(len(names)):  # small-loop(consumed state field names)
            if py_contains(row, names[j]):
                raise Error("TypeError: generation state fields cannot be replaced by descriptors")
    return _fields(value)


def _generation_number(value: PythonObject) raises -> Float64:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not bi.int and cls is not bi.float and cls is not bi.bool:
        raise Error("TypeError: generation requires captured native numeric metadata")
    return Float64(py=value)


def _generation_bool(value: PythonObject) raises -> Bool:
    if py_type(value) is not native_loaded_module("builtins").bool:
        raise Error("TypeError: generation requires captured native boolean metadata")
    return Bool(py=value)


struct GenerationWeightSnapshot(Copyable, Movable):
    var owners: List[PythonObject]
    var addresses: List[Int]
    var refresh: List[Int]
    var layouts: List[BorrowedBufferLayout]
    var formats: List[String]
    var failures: List[String]

    def __init__(out self, values: PythonObject, zero_from: Int = -1) raises:
        self.owners = List[PythonObject]()
        self.addresses = List[Int]()
        self.refresh = List[Int]()
        self.layouts = List[BorrowedBufferLayout]()
        self.formats = List[String]()
        self.failures = List[String]()
        model_sequence(values)
        var cls = GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls
        for j in range(len(values)):  # driver-loop: admit each stable weight slot once
            var value = values[j]
            if value is None or (zero_from >= 0 and j >= zero_from):
                self.owners.append(value)
                self.addresses.append(0)
            elif py_type(value) is cls:
                var captured = generation_array_snapshot(value)
                if model_string(captured.order) != "C":
                    raise Error("generation: internal model and state arrays must be C-contiguous")
                self.addresses.append(model_integer(captured._addr))
                # Keep the pinned captured storage, not only the original Array
                # whose _store may be replaced while execution releases GIL.
                self.owners.append(captured)
            else:
                var view = value
                try:
                    view = generation_buffer_view(value)
                    var pin = AdmittedBorrowedBuffer(view)
                    try:
                        var layout = pin.layout.value().copy()
                        if not layout.contiguous():
                            raise Error("generation: internal model and state buffers must be C-contiguous")
                        var format = pin.format
                        if format != "f" and format != "i" and format != "b":
                            raise Error("generation: unsupported native model buffer format " + format)
                        var width = 1 if format == "b" else 4
                        if layout.itemsize != width:
                            raise Error("generation: native model buffer format has an inconsistent byte width")
                        self.layouts.append(layout^)
                        self.formats.append(format)
                    except error:
                        pin.release()
                        raise error
                    pin.release()
                    self.failures.append(String(""))
                except error:
                    # Foreign weights were historically admitted at their
                    # execution position. Capture refusal now without running
                    # hooks, then raise there so earlier layer progress stays.
                    self.layouts.append(BorrowedBufferLayout(0, 1, True, [0], [1], [-1]))
                    self.formats.append(String(""))
                    self.failures.append(String(error))
                # The private memoryview retains its exporter while its native
                # descriptor is captured once. Only bytes refresh per use.
                self.owners.append(view)
                self.addresses.append(0)
                self.refresh.append(j)

    def resolve(self, mut keep: List[PythonObject]) raises -> List[Int]:
        var result = self.addresses.copy()
        for k in range(len(self.refresh)):  # driver-loop: retained foreign-buffer copy compatibility
            var index = self.refresh[k]
            if self.failures[k] != "":
                raise Error(self.failures[k])
            ref layout = self.layouts[k]
            var packed = generation_array([layout.count], self.formats[k])
            var address = Int(py=packed._addr)
            layout.copy_to(address)
            result[index] = address
            keep.append(packed)
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
        model_dict(block)
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
            parameters = [0, 0, model_integer(block["d_model"]), model_integer(block["n_heads"]),
                          model_integer(block["n_kv_heads"]), model_integer(block["head_dim"]),
                          model_integer(block["intermediate"]), 0, 0, model_integer(block["window"])]
            var tail = block["_opts_tail"]
            model_sequence(tail)
            for j in range(len(tail)):  # driver-loop: admitted immutable transformer option words
                parameters.append(model_integer(tail[j]))
            optional = block["_wopt"]
            if planes:
                var triples = block["_int15"]
                model_sequence(triples)
                for j in range(len(triples)):  # driver-loop: packed weight plane registry
                    var trip = triples[j]
                    if trip is not None:
                        model_sequence(trip)
                        if len(trip) != 3:
                            raise Error("TypeError: packed generation weights require native plane triples")
                    for k in range(3):  # small-loop(int15 plane triple)
                        _ = py_call(packed.append, PythonObject(None) if trip is None else trip[k])
            self.profile = model_plain(py_get(block, "_arithmetic_profile", PythonObject(None)))
            self.profile_changed = _generation_bool(py_get(block, "_arithmetic_profile_changed", PythonObject(False)))
        elif kind == "mamba1":
            inner = model_integer(block["d_inner"])
        else:
            heads = model_integer(block["nheads"])
            conv = model_integer(block["conv_dim"])
            var limits = block["dt_limit"]
            model_sequence(limits)
            if len(limits) != 2:
                raise Error("TypeError: Mamba generation requires native dt_limit bounds")
            low = _generation_number(limits[0])
            high = _generation_number(limits[1])
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
        model_dict(entries)
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.width = width
        self.vocab = vocab
        var fields = _generation_fields(model, Python.tuple("_norm", "_head_int15", "norm_eps", "_head"))
        var norm = fields["_norm"] if normalize else PythonObject(None)
        var planes = fields["_head_int15"]
        self.normalize = norm is not None
        self.packed = planes is not None and packed
        self.norm_eps = _generation_number(fields["norm_eps"]) if self.normalize else Float64(0)
        var values = Python.list(norm, PythonObject(None) if self.packed else fields["_head"])
        if self.packed:
            model_sequence(planes)
            if len(planes) != 3:
                raise Error("TypeError: packed generation head requires a native plane triple")
            for k in range(3):  # small-loop(int15 head plane triple)
                _ = py_call(values.append, planes[k])
        self.weights = GenerationWeightSnapshot(values)


struct GenerationOperationSnapshot(Copyable, Movable):
    var model: PythonObject
    var model_fields: PythonObject
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
        self.model_fields = _generation_fields(model, Python.tuple("kind", "d_model", "vocab_size", "_embed", "max_positions"))
        model_dict(entries)
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.kind = model_string(self.model_fields["kind"])
        self.width = model_integer(self.model_fields["d_model"])
        self.vocab = model_integer(self.model_fields["vocab_size"])
        self.embedding = GenerationWeightSnapshot(Python.list(self.model_fields["_embed"]))
        self.head = GenerationHeadSnapshot(model, entries, self.width, self.vocab, True, True)
        self.planes = self.head.packed and py_contains(entries, "forward15")
        self.layers = List[GenerationLayerSnapshot]()
        for i in range(len(blocks)):  # driver-loop: capture stable layer metadata once per operation
            self.layers.append(GenerationLayerSnapshot(blocks[i], self.kind, self.planes))


# This admission is an operation-state consumer of the common descriptor and
# buffer boundary, not a new public adapter. Only the native engine below its
# boundary owns token/layer execution.
from bindings.py_balanced import py_setitem
from bindings.generation_buffers import generation_mutable
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider
from providers.python.neural_descriptor_pyapi import neural_descriptor_handle, neural_descriptor_entry
from providers.python.native_execution_pyapi import AdmittedNativeExecution
from providers.python.operation_lock_pyapi import OperationLockLease
from core.generation_metadata import generation_transformer_addresses
from transformer.native_generation_engine import NativeGenerationLayer, NativeGenerationModel


def _generation_engine_weights(snapshot: GenerationWeightSnapshot,
    mut pins: List[AdmittedBorrowedBuffer], mut storage: List[List[Float32]],
    mut layouts: List[BorrowedBufferLayout], mut destinations: List[Int]) raises -> List[Int]:
    # The snapshot's private memoryviews already pin each foreign exporter.
    # Reuse its captured layout, preserving the old per-use byte refresh point.
    var addresses = snapshot.addresses.copy()
    for k in range(len(snapshot.refresh)):  # driver-loop: retained foreign weight descriptors, never recapture caller metadata
        var index = snapshot.refresh[k]
        if snapshot.failures[k] != "":
            raise Error(snapshot.failures[k])
        var layout = snapshot.layouts[k].copy()
        var bytes = layout.count * layout.itemsize
        storage.append(List[Float32](length=(bytes + 3) // 4, fill=0))
        var address = Int(storage[len(storage) - 1].unsafe_ptr())
        addresses[index] = address
        layouts.append(layout^)
        destinations.append(address)
    return addresses^


def _generation_engine_state(value: PythonObject, expected: List[Int], label: String,
    mut pins: List[AdmittedBorrowedBuffer], mut storage: List[List[Float32]],
    mut keep: List[PythonObject], mut layer: NativeGenerationLayer) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        var copies = List[PythonObject]()
        return generation_mutable(value, expected, keep, copies, label)
    var view = generation_buffer_view(value)
    var pin = AdmittedBorrowedBuffer(view, True)
    ref layout = pin.layout.value()
    if pin.format != "f" or layout.itemsize != 4 or not layout.contiguous():
        raise Error("CausalLM: state buffer " + label + " must be writable C-contiguous float32 (it is updated in place)")
    if len(layout.shape) != len(expected):
        raise Error("CausalLM: state buffer " + label + " rank differs from its required cache span")
    var count = 1
    for axis in range(len(expected)):  # small-loop(admitted state buffer shape)
        if layout.shape[axis] != expected[axis]:
            raise Error("CausalLM: state buffer " + label + " shape differs from its required cache span")
        count *= expected[axis]
    storage.append(List[Float32](length=count, fill=0))
    var address = Int(storage[len(storage) - 1].unsafe_ptr())
    layer.refresh_layouts.append(layout.copy())
    layer.refresh_addresses.append(address)
    layer.writeback_layouts.append(layout.copy())
    layer.writeback_addresses.append(address)
    pins.append(pin^)
    keep.append(view)
    return address


struct GenerationEngineAdmission(Movable):
    var snapshot: GenerationOperationSnapshot
    var state: PythonObject
    var states: List[PythonObject]
    var owners: List[AdmittedNativeExecution]
    var buffers: List[PythonObject]
    var pins: List[AdmittedBorrowedBuffer]
    var storage: List[List[Float32]]
    var state_keys: List[PythonObject]
    var native: NativeGenerationModel
    var published: Bool

    def __init__(out self, snapshot: GenerationOperationSnapshot, state: PythonObject, batch: Int, length: Int) raises:
        self.snapshot = snapshot.copy()
        self.state = _generation_fields(state, Python.tuple("positions", "max_tokens", "layers", "_usable", "_native_execution_route", "_native_engine_forwards"), True)
        self.states = List[PythonObject]()
        self.owners = List[AdmittedNativeExecution]()
        self.buffers = List[PythonObject]()
        self.pins = List[AdmittedBorrowedBuffer]()
        self.storage = List[List[Float32]]()
        self.state_keys = List[PythonObject]()
        self.published = False
        var entries = snapshot.entries
        var provider = py_get(entries, "native_block_operations" if snapshot.kind == "transformer" else "native_mamba")
        if snapshot.head.packed or snapshot.planes or provider is None or snapshot.native is None:
            raise Error("CausalLM: native generation engine requires admitted nonresident fp32 model descriptors")
        self.owners.append(AdmittedNativeExecution(snapshot.native, PythonObject(None), 0, -1, -1, PythonObject(None), 0))
        var model_limit = snapshot.model_fields["max_positions"]
        var limited = model_limit is not None
        self.native = NativeGenerationModel(self.owners[0].endpoint(), snapshot.embedding.addresses[0],
            snapshot.head.weights.addresses[0], snapshot.head.weights.addresses[1], snapshot.head.norm_eps,
            batch, snapshot.width, snapshot.vocab, model_integer(self.state["positions"]), model_integer(self.state["max_tokens"]),
            limited, model_integer(model_limit) if limited else 0)
        try:
            var embedding = _generation_engine_weights(snapshot.embedding, self.pins, self.storage,
                self.native.embedding_refresh, self.native.embedding_targets)
            self.native.embedding = embedding[0]
        except problem:
            self.native.embedding_failure = String(problem)
        try:
            var head = _generation_engine_weights(snapshot.head.weights, self.pins, self.storage,
                self.native.head_refresh, self.native.head_targets)
            self.native.norm = head[0]
            self.native.head = head[1]
        except problem:
            self.native.head_failure = String(problem)
        var layers = self.state["layers"]
        model_sequence(layers)
        if len(layers) != len(snapshot.layers):
            raise Error("CausalLM: state layer count differs from the captured model")
        var cpu = _generation_bool(entries["cpu"])
        for i in range(len(snapshot.layers)):  # driver-loop: admit layer/state owners once before native execution
            ref layer = snapshot.layers[i]
            var st = layers[i]
            var fields = Python.dict()
            var identity = st
            var native_layer = NativeGenerationLayer(self.owners[0].endpoint(), 33 if cpu else 11, [], [], 0)
            try:
                fields = _generation_fields(st, Python.tuple("_resident_session", "arithmetic_profile", "cached_tokens", "max_tokens", "k_cache", "v_cache", "conv_window", "h", "profile_id", "_nn34_usable", "absolute_position", "buffer_xbc", "buffer_dtraw", "buffered_tokens"), True)
                identity = py_get(fields, "_native_state_identity", py_get(fields, "_causal_original", st))
                if py_get(fields, "_resident_session") is not None:
                    raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
                var cached = 0
                if snapshot.kind == "transformer":
                    var old_profile = model_plain(py_get(fields, "arithmetic_profile"))
                    cached = model_integer(fields["cached_tokens"])
                    native_layer.cached = cached
                    if old_profile is None and cached != 0 and layer.profile_changed:
                        raise Error("CausalLM: cached state requires arithmetic profile metadata")
                    if old_profile is not None and layer.profile is not None and not Bool(py=old_profile == layer.profile):
                        raise Error("CausalLM: state arithmetic profile differs")
                    native_layer.profile_admitted = True
                    var capacity = model_integer(fields["max_tokens"])
                    var cache_count = layer.words.cache_count(batch, capacity)
                    var key = fields["k_cache"]
                    var value = fields["v_cache"]
                    var weights = _generation_engine_weights(layer.weights, self.pins, self.storage,
                        native_layer.refresh_layouts, native_layer.refresh_addresses)
                    var optional = _generation_engine_weights(layer.optional, self.pins, self.storage,
                        native_layer.refresh_layouts, native_layer.refresh_addresses)
                    var key_address = _generation_engine_state(key, [cache_count], "k_cache",
                        self.pins, self.storage, self.buffers, native_layer)
                    var value_address = _generation_engine_state(value, [cache_count], "v_cache",
                        self.pins, self.storage, self.buffers, native_layer)
                    var block = layer.block
                    var lock_owner = block["_runtime_lock"]
                    var lock = OperationLockLease(lock_owner)
                    if lock.handle == 0:
                        raise Error("CausalLM: native generation requires an admitted native operation lock")
                    var session = PythonObject(None)
                    var handle = 0
                    if not cpu:
                        if block["_native_session"] is None:
                            var create = entries["create"]
                            if py_type(create) is not py_type(native_loaded_module("builtins").len):
                                raise Error("TypeError: generation session creation requires its compiled native entry")
                            py_setitem(block, "_native_session", py_call(create))
                            py_setitem(block, "_session_binding", entries["block_binding"])
                        session = block["_native_session"]
                        handle = neural_descriptor_handle(provider,
                            neural_descriptor_entry(entries["block_binding"], "transformer_native_session_handle_v1"), session)
                    self.owners.append(AdmittedNativeExecution(provider, session, handle,
                        -1 if cpu else 0, -1 if cpu else 1, lock_owner, lock.handle))
                    native_layer.endpoint = self.owners[len(self.owners) - 1].endpoint()
                    var addresses = generation_transformer_addresses(0, weights,
                        key_address, value_address, 0, optional, [])
                    native_layer.addresses = addresses^
                    native_layer.parameters = layer.words.transformer_params(batch, length, capacity, cached)
                else:
                    var weights = _generation_engine_weights(layer.weights, self.pins, self.storage,
                        native_layer.refresh_layouts, native_layer.refresh_addresses)
                    var addresses: List[Int] = [0]
                    for address in weights:  # driver-loop: fixed Mamba weight descriptor vector
                        addresses.append(address)
                    self.owners.append(AdmittedNativeExecution(provider, PythonObject(None), 0,
                        -1, -1, PythonObject(None), 0))
                    native_layer.endpoint = self.owners[len(self.owners) - 1].endpoint()
                    var parameters: List[Int] = [batch, length, snapshot.width]
                    if snapshot.kind == "mamba1":
                        native_layer.kind = 1
                        native_layer.operation = 0
                        native_layer.output_slot = 13
                        var inner = layer.words.inner
                        addresses.append(_generation_engine_state(fields["conv_window"], [batch, inner, 4], "conv_window",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(_generation_engine_state(fields["h"], [batch, inner, 16], "h",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(0)
                        var profile = model_integer(entries["mamba1_profile"])
                        if model_integer(fields["profile_id"]) != profile or not _generation_bool(fields["_nn34_usable"]):
                            raise Error("CausalLM: incompatible or failed Mamba1 state profile")
                        if profile != 1:
                            raise Error("CausalLM: native generation requires the supported Mamba1 profile")
                        cached = model_integer(fields["absolute_position"])
                    else:
                        native_layer.kind = 2
                        native_layer.operation = 2
                        native_layer.output_slot = 14
                        var heads = layer.words.heads
                        var conv = layer.words.conv
                        addresses.append(_generation_engine_state(fields["conv_window"], [batch, conv, 4], "conv_window",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(_generation_engine_state(fields["h"], [batch, heads, 64, 128], "h",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(_generation_engine_state(fields["buffer_xbc"], [batch, 256, conv], "buffer_xbc",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(_generation_engine_state(fields["buffer_dtraw"], [batch, 256, heads], "buffer_dtraw",
                            self.pins, self.storage, self.buffers, native_layer))
                        addresses.append(0); addresses.append(0)
                        cached = model_integer(fields["buffered_tokens"])
                        parameters.append(cached)
                        native_layer.floats = [layer.words.dt_low, layer.words.dt_high]
                        native_layer.report_size = batch * heads * 64 * 128
                    native_layer.cached = cached
                    native_layer.addresses = addresses^
                    native_layer.parameters = parameters^
                var state_index = -1
                for index in range(len(self.state_keys)):  # driver-loop: exact alias identity of admitted state records
                    if self.state_keys[index] is identity:
                        state_index = index
                        break
                if state_index < 0:
                    state_index = len(self.state_keys)
                    self.state_keys.append(identity)
                    self.native.state_counters.append(cached)
                native_layer.state_index = state_index
            except problem:
                # Defer this refusal to its original execution position. Earlier
                # layers must retain their normal partial progress on failure.
                native_layer.failure = String(problem)
            self.states.append(fields)
            self.native.layers.append(native_layer^)
        _ = provider
        _ = layers

    def publish(mut self) raises:
        if self.published:
            return
        for i in range(len(self.states)):  # driver-loop: publish completed/partial native state at operation boundary
            if self.native.layers[i].attempted:
                var counter = "cached_tokens" if self.native.layers[i].kind == 0 else ("absolute_position" if self.native.layers[i].kind == 1 else "buffered_tokens")
                var counter_value = self.native.layers[i].cached
                var identity = self.native.layers[i].state_index
                if identity >= 0:
                    # A later alias can retain an earlier forward's cached
                    # field when a subsequent forward fails before visiting it.
                    # The per-identity counter is authoritative partial progress.
                    counter_value = self.native.state_counters[identity]
                py_setitem(self.states[i], counter, PythonObject(counter_value))
            if self.native.layers[i].kind == 2 and self.native.layers[i].completed:
                var heads = self.snapshot.layers[i].words.heads
                var report = generation_array([self.native.batch, heads, 64, 128])
                memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=report._addr)),
                    src=self.native.layers[i].report.unsafe_ptr(), count=self.native.layers[i].report_size)
                py_setitem(self.snapshot.layers[i].block, "h_last_", report)
            if self.native.layers[i].visited and self.native.layers[i].profile_admitted and self.snapshot.layers[i].profile is not None:
                py_setitem(self.states[i], "arithmetic_profile", self.snapshot.layers[i].profile)
        py_setitem(self.state, "positions", PythonObject(self.native.position))
        py_setitem(self.state, "_native_execution_route", PythonObject(self.snapshot.kind + "-nonresident-native-engine-v1"))
        py_setitem(self.state, "_native_engine_forwards", PythonObject(self.native.forwards))
        if self.native.failed:
            py_setitem(self.state, "_usable", PythonObject(False))
        self.published = True
        _ = self.buffers
        _ = self.owners
        _ = self.snapshot
