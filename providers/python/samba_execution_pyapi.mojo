# SPDX-License-Identifier: Apache-2.0
# Whole-operation Samba admission/publication over the common native provider.
# The executor receives only native endpoints, pointers and stable metadata.
from std.python import Python, PythonObject
from std.os import getenv
from std.memory import memcpy
from bindings.py_balanced import py_call, py_get, py_setitem, py_setattr, py_type
from bindings.generation_buffers import generation_address, generation_array
from bindings.causal_lm_public import _native_function, _mutable
from bindings.neural_block_public import neural_training_entries
from providers.python.neural_descriptor_pyapi import neural_descriptor_handle, neural_descriptor_matches, neural_descriptor_reuse, neural_descriptor_construct
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME, MAMBA_STATE_ABI_NAME
from providers.python.native_execution_pyapi import AdmittedNativeExecution
from providers.python.operation_lock_pyapi import OperationLockLease
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from providers.python.generation_metadata_pyapi import GenerationLayerSnapshot
from bindings.native_buffer_address import native_buffer_address
from training.native_samba_execution import NativeSambaLayer, NativeSambaModel


struct SambaLayerPublication(Movable):
    var fields: PythonObject
    var state: PythonObject
    var profile: PythonObject
    var report_shapes: List[List[Int]]
    var copies: List[PythonObject]
    var session: PythonObject
    var binding: PythonObject
    var provider: PythonObject
    var entries: PythonObject
    var weight_owners: List[PythonObject]
    var weight_addresses: List[Int]
    var gradient_shapes: List[List[Int]]
    var gradient_names: List[String]
    var publish_reports: Bool
    var transformer: Bool
    var restore_owned: Bool

    def __init__(out self, fields: PythonObject, state: PythonObject, transformer: Bool):
        self.fields = fields
        self.state = state
        self.transformer = transformer
        self.profile = PythonObject(None)
        self.report_shapes = List[List[Int]]()
        self.copies = List[PythonObject]()
        self.session = PythonObject(None)
        self.binding = PythonObject(None)
        self.provider = PythonObject(None)
        self.entries = PythonObject(None)
        self.weight_owners = List[PythonObject]()
        self.weight_addresses = List[Int]()
        self.gradient_shapes = List[List[Int]]()
        self.gradient_names = List[String]()
        self.publish_reports = True
        self.restore_owned = False


def _samba_native_fields(value: PythonObject) raises -> PythonObject:
    return py_call(Python.import_module("builtins").object.__getattribute__, value, "__dict__")


def _samba_native_copy(values: List[Float32], shape: List[Int]) raises -> PythonObject:
    var output = generation_array(shape)
    if len(values) != 0:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr)),
            src=values.unsafe_ptr(), count=len(values))
    return output


struct AdmittedSambaExecution(Movable):
    var native: NativeSambaModel
    var owners: List[AdmittedNativeExecution]
    var keep: List[PythonObject]
    var pins: List[AdmittedBorrowedBuffer]
    var publications: List[SambaLayerPublication]
    var state_keys: List[PythonObject]
    var batch: Int
    var length: Int
    var width: Int
    var vocab: Int

    def __init__(out self, fields: PythonObject, config: PythonObject, state: PythonObject,
                 primitive: PythonObject, b: Int, l: Int, step: Bool, context: PythonObject) raises:
        var bi = Python.import_module("builtins")
        self.owners = List[AdmittedNativeExecution]()
        self.keep = List[PythonObject]()
        self.pins = List[AdmittedBorrowedBuffer]()
        self.publications = List[SambaLayerPublication]()
        self.state_keys = List[PythonObject]()
        self.batch = b
        self.length = l
        self.width = Int(py=config["d_model"])
        self.vocab = Int(py=config["vocab"])
        self.owners.append(AdmittedNativeExecution(primitive, PythonObject(None), 0, -1, -1, PythonObject(None), 0))
        var arrays = fields["arrays"]
        var head = arrays["embed.weight"] if Bool(py=config["tie_embeddings"]) else arrays["lm_head.weight"]
        self.native = NativeSambaModel(self.owners[0].endpoint(),
            generation_address(arrays["embed.weight"], self.keep), generation_address(arrays["norm_f.weight"], self.keep),
            generation_address(head, self.keep), Float64(py=config["norm_eps"]), b * l, self.width, self.vocab)
        var blocks = fields["_blocks"]
        var kinds = config["layers"]
        var states = PythonObject(None)
        if state is not None:
            var slots = _samba_native_fields(state)
            if Int(py=slots["batch_size"]) != b or len(slots["layers"]) != len(kinds):
                raise Error("Samba: decode state batch or layer count differs")
            states = slots["layers"]
        if len(blocks) != len(kinds):
            raise Error("Samba: block count differs from the configured layer count")
        for i in range(len(kinds)):  # driver-loop: one metadata/lifetime admission per model layer
            var block = blocks[i]
            var layer_fields = _samba_native_fields(block)
            var transformer = String(kinds[i]) == "attention"
            var layer_state = PythonObject(None) if states is None else states[i]
            var publication = SambaLayerPublication(layer_fields, layer_state, transformer)
            var layer = NativeSambaLayer(self.owners[0].endpoint(), 0, [], [], [], 0, 0, -1, [])
            try:
                var entries = neural_training_entries(layer_fields, context)
                publication.entries = entries
                var source_weights = layer_fields["_w"]
                for j in range(len(source_weights)):  # driver-loop: retain immutable weight owners at admission
                    publication.weight_owners.append(source_weights[j])
                var names = block._W_NAMES
                for j in range(len(names)):  # driver-loop: admit fixed named gradient metadata
                    publication.gradient_names.append(String(names[j]))
                var canonical = entries.canonical
                if transformer:
                    var lock = OperationLockLease(layer_fields["_runtime_lock"])
                    if lock.handle == 0:
                        raise Error("native Samba: layer needs its admitted native operation lock")
                    var snapshot = GenerationLayerSnapshot(layer_fields, "transformer", False)
                    var weights = snapshot.weights.resolve(self.keep)
                    publication.weight_addresses = weights.copy()
                    var optional = snapshot.optional.resolve(self.keep)
                    var provider = layer_fields["_native_block_provider_cache"]
                    publication.provider = provider
                    var use_session = layer_state is not None or Bool(py=py_call(bi.hasattr, entries, "transformer_session_forward_fresh"))
                    var session = PythonObject(None)
                    if use_session:
                        session = layer_fields["_native_session"]
                        if session is None or layer_fields["_session_binding"] is not canonical:
                            # Empty owner admission performs no numerical work. New
                            # candidate blocks already have this shared typed owner.
                            session = _native_function(entries, "transformer_session_create")()
                        publication.session = session
                        publication.binding = canonical
                        var handle = neural_descriptor_handle(provider,
                            _native_function(entries, "transformer_native_session_handle_v1"), session)
                        self.owners.append(AdmittedNativeExecution(provider, session, handle, 0, 1, lock.owner, lock.handle))
                    else:
                        self.owners.append(AdmittedNativeExecution(provider, PythonObject(None), 0, -1, -1, lock.owner, lock.handle))
                    var a: List[Int] = [0]
                    for address in weights:  # driver-loop: native transformer weight-address metadata
                        a.append(address)
                    var p = snapshot.words.transformer_params(b, l, l, 0)
                    var operation = 2
                    if layer_state is not None:
                        var slots = _samba_native_fields(layer_state)
                        if py_get(slots, "_resident_session") is not None:
                            raise Error("neural block: state is owned by a resident decode session")
                        var cached = Int(py=slots["cached_tokens"])
                        var capacity = Int(py=slots["max_tokens"])
                        if Int(py=slots["batch_size"]) != b or Int(py=slots["n_kv_heads"]) != p[4] or Int(py=slots["head_dim"]) != p[5] or Int(py=slots["window"]) != p[9]:
                            raise Error("neural block: transformer state dimensions differ")
                        if cached < 0 or cached + l > capacity or cached + l > Int(py=layer_fields["max_positions"]):
                            raise Error("neural block: sequence exceeds state capacity or max_positions")
                        var old_profile = py_get(slots, "arithmetic_profile")
                        if old_profile is None and cached != 0 and snapshot.profile_changed:
                            raise Error("CausalLM: cached state requires arithmetic profile metadata")
                        if old_profile is not None and snapshot.profile is not None and not Bool(py=old_profile == snapshot.profile):
                            raise Error("CausalLM: state arithmetic profile differs")
                        publication.profile = snapshot.profile
                        var count = snapshot.words.cache_count(b, capacity)
                        a.append(_mutable(slots["k_cache"], [count], self.keep, publication.copies, "k_cache"))
                        a.append(_mutable(slots["v_cache"], [count], self.keep, publication.copies, "v_cache"))
                        p = snapshot.words.transformer_params(b, l, capacity, cached)
                        operation = 11
                    elif use_session:
                        a.append(0)
                        a.append(0)
                    else:
                        var fresh = List[Int]()
                        for j in range(7):  # small-loop(seven fresh transformer shape words)
                            fresh.append(p[j])
                        fresh.append(p[9])
                        for j in range(10, len(p)):  # small-loop(the admitted block-option tail)
                            fresh.append(p[j])
                        p = fresh^
                        operation = 4
                    a.append(0)
                    for address in optional:  # driver-loop: native optional transformer weight metadata
                        a.append(address)
                    layer = NativeSambaLayer(self.owners[len(self.owners) - 1].endpoint(), operation, a^, p^, [], 0, 12 if use_session else 10, -1, [])
                else:
                    var h = Int(py=layer_fields["nheads"])
                    var weights = layer_fields["_w"]
                    var a: List[Int] = [0]
                    var owned = py_get(layer_fields, "_owned_generation")
                    var restore = Bool(py=py_get(layer_fields, "_owned_restore_pending", PythonObject(False)))
                    var use_owned = owned is not None or restore
                    var weight_addresses = List[Int]()
                    for j in range(len(weights)):  # driver-loop: one admission of the Mamba3 weight slots
                        weight_addresses.append(generation_address(weights[j], self.keep))
                        if not use_owned:
                            a.append(weight_addresses[j])
                    publication.weight_addresses = weight_addresses.copy()
                    var provider = layer_fields["_native_mamba_block_provider_cache"]
                    var session = py_get(layer_fields, "_prefill_session")
                    var use_session = use_owned or (layer_state is None and Bool(py=py_call(bi.hasattr, entries, "mamba3_prefill_session_forward")))
                    var handle = 0
                    if use_session:
                        if session is None or py_get(layer_fields, "_prefill_binding") is not canonical:
                            if owned is not None and not restore:
                                raise Error("RuntimeError: Mamba3 owned snapshot requires its installed binding/setup")
                            session = _native_function(entries, "mamba3_prefill_session_create")()
                        publication.session = session
                        publication.binding = canonical
                        handle = neural_descriptor_handle(provider, _native_function(entries, "mamba_prefill_native_handle_v1"), session)
                        self.owners.append(AdmittedNativeExecution(provider, session, handle, 0, 1, PythonObject(None), 0))
                    elif layer_state is None:
                        self.owners.append(AdmittedNativeExecution(provider, PythonObject(None), 0, -1, -1, PythonObject(None), 0))
                    else:
                        provider = neural_descriptor_reuse(py_get(layer_fields, "_native_mamba_state"),
                            _native_function(entries, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME))
                        py_setitem(layer_fields, "_native_mamba_state", provider)
                        self.owners.append(AdmittedNativeExecution(provider, PythonObject(None), 0, -1, -1, PythonObject(None), 0))
                    publication.provider = provider
                    var p: List[Int] = [b, l, self.width]
                    var operation = 8 if use_owned else (6 if use_session else 5)
                    var output_slot = 1 if use_owned else 10
                    if layer_state is not None:
                        var slots = _samba_native_fields(layer_state)
                        if py_get(slots, "_resident_session") is not None:
                            raise Error("Mamba3: state belongs to an open resident decode session; synchronize or close it first")
                        if use_owned:
                            for j in range(9):  # small-loop(unread owned weight pointer slots)
                                a.append(0)
                        var names = [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"), String("buffer_v"),
                            String("buffer_dt"), String("buffer_sig"), String("buffer_adt"), String("pending_k"), String("pending_v")]
                        var shapes = List[List[Int]]()
                        shapes.append([b, h, 32]); shapes.append([b, h, 64, 128])
                        shapes.append([b, 64, h, 128]); shapes.append([b, 64, h, 128]); shapes.append([b, 64, h, 64])
                        shapes.append([b, 64, h]); shapes.append([b, 64, h]); shapes.append([b, 64, h])
                        shapes.append([b, h, 128]); shapes.append([b, h, 64])
                        for j in range(len(names)):  # driver-loop: one admission of each carried-state buffer
                            a.append(_mutable(slots[names[j]], shapes[j], self.keep, publication.copies, names[j]))
                        var q = Int(py=slots["buffered_tokens"])
                        var pending = 1 if Bool(py=slots["pending"]) else 0
                        if use_owned:
                            p = [b, l, self.width, q, pending, Int(py=owned) if owned is not None else 0, 1 if step else 0]
                            operation = 16
                        else:
                            p = [b, self.width, q, pending] if step else [b, l, self.width, q, pending]
                            operation = 5 if step else 4
                        output_slot = 20
                    elif use_owned:
                        p.append(Int(py=owned) if owned is not None else 0)
                    var sizes = List[Int]()
                    publication.report_shapes.append([b, h, 64, 128]); publication.report_shapes.append([b, h, 128])
                    publication.report_shapes.append([b, h, 64]); publication.report_shapes.append([b, h, 32])
                    var skip = False
                    if layer_state is None and use_session and Bool(py=py_get(layer_fields, "_discard_reports", PythonObject(False))) and Bool(py=py_call(bi.hasattr, entries, "mamba3_prefill_session_reports_optional")):
                        skip = Bool(py=_native_function(entries, "mamba3_prefill_session_reports_optional")())
                    a.append(0)
                    for j in range(4):  # small-loop(four Mamba report pointer/size slots)
                        a.append(0)
                        var count = 1
                        for axis in range(len(publication.report_shapes[j])):  # small-loop(native report shape rank)
                            count *= publication.report_shapes[j][axis]
                        sizes.append(0 if skip else count)
                    layer = NativeSambaLayer(self.owners[len(self.owners) - 1].endpoint(), operation, a^, p^, [], 0, output_slot, output_slot + 1, sizes^)
                    if restore:
                        if not Bool(py=_native_function(entries, "mamba3_owned_weights_enabled")()):
                            raise Error("RuntimeError: Mamba3 owned weights require the compiled owned-weight entry")
                        publication.restore_owned = True
                        layer.prepare_operation = 15
                        layer.prepare_addresses = weight_addresses^
                        layer.prepare_parameters = [self.width]
                        layer.generation_slot = 3 if layer_state is None else 5
                if layer_state is not None:
                    layer.counter_slot = 8 if transformer else (2 if step and layer.operation == 5 else 3)
                    layer.pending_slot = -1 if transformer else layer.counter_slot + 1
                    var state_index = -1
                    for j in range(len(self.state_keys)):  # driver-loop: coalesce aliases of the same mutable continuation owner
                        if self.state_keys[j] is layer_state:
                            state_index = j
                            break
                    if state_index < 0:
                        state_index = len(self.state_keys)
                        self.state_keys.append(layer_state)
                        self.native.state_counters.append(layer.parameters[layer.counter_slot])
                        self.native.state_pending.append(0 if transformer else layer.parameters[layer.pending_slot])
                    layer.state_index = state_index
                for j in range(len(publication.copies)):  # driver-loop: pin deferred foreign-buffer refresh/writeback once
                    var pin = AdmittedBorrowedBuffer(publication.copies[j][0], True)
                    layer.refresh_layouts.append(pin.layout.value().copy())
                    layer.refresh_addresses.append(native_buffer_address(publication.copies[j][1]))
                    self.pins.append(pin^)
            except problem:
                # Admission errors occur at their former layer position. Prior
                # completed layers keep their public state/report progress.
                layer.failure = String(problem)
            self.publications.append(publication^)
            self.native.layers.append(layer^)

    def configure_training(mut self) raises:
        var bi = Python.import_module("builtins")
        for i in range(len(self.native.layers)):  # driver-loop: admit backward/tape owners before whole native execution
            ref layer = self.native.layers[i]
            ref publication = self.publications[i]
            if layer.failure.byte_length() != 0:
                continue
            var forward_setup = False
            try:
                var entries = publication.entries
                var fields = publication.fields
                var kind = "transformer" if publication.transformer else "mamba3"
                var tape = Bool(py=py_call(bi.hasattr, entries, kind + "_forward_tape_enabled"))
                if tape:
                    tape = Bool(py=_native_function(entries, kind + "_forward_tape_enabled")())
                if publication.transformer:
                    tape = tape and fields["_int15"] is None and not Bool(py=fields["_extended"])
                    if fields["_int15"] is not None or Bool(py=fields["_extended"]):
                        raise Error("NotImplementedError: TransformerBlock.backward requires default float32 block options (the default block options only)")
                if not tape and not Bool(py=py_call(bi.hasattr, entries, kind + "_backward")):
                    raise Error("RuntimeError: loaded native training binding lacks " + kind + "_backward")
                layer.validate_gradient = not publication.transformer and not tape
                for j in range(len(publication.weight_owners)):  # driver-loop: snapshot each parameter's gradient shape once
                    var shape = publication.weight_owners[j].shape
                    var dimensions = List[Int]()
                    var count = 1
                    for axis in range(len(shape)):  # small-loop(parameter shape rank)
                        var dimension = Int(py=shape[axis])
                        dimensions.append(dimension)
                        count *= dimension
                    publication.gradient_shapes.append(dimensions^)
                    layer.gradient_sizes.append(count)
                var parameters: List[Int] = [self.batch, self.length, self.width]
                if publication.transformer:
                    for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("window")]:  # small-loop(five Transformer backward dimension words)
                        parameters.append(Int(py=fields[name]))
                layer.backward_parameters = parameters.copy()
                layer.backward_weights = publication.weight_addresses.copy()
                if tape:
                    forward_setup = True
                    var owner = neural_descriptor_construct(publication.provider,
                        _native_function(entries, kind + "_tape_create_native_v1"))
                    var handle = neural_descriptor_handle(publication.provider,
                        _native_function(entries, kind + "_tape_handle_native_v1"), owner)
                    var lock_owner = PythonObject(None)
                    var lock_handle = 0
                    if publication.transformer:
                        var lock = OperationLockLease(fields["_runtime_lock"])
                        lock_owner = lock.owner
                        lock_handle = lock.handle
                    var base = 6 if publication.transformer else 10
                    self.owners.append(AdmittedNativeExecution(publication.provider, owner, handle, base, base + 1, lock_owner, lock_handle))
                    layer.endpoint = self.owners[len(self.owners) - 1].endpoint()
                    layer.backward_endpoint = layer.endpoint
                    self.owners.append(AdmittedNativeExecution(publication.provider, owner, handle, base, base + 1, PythonObject(None), 0))
                    layer.close_endpoint = self.owners[len(self.owners) - 1].endpoint()
                    layer.operation = base + 2
                    layer.backward_operation = base + 3
                    layer.close_operation = base + 4
                    var a: List[Int] = [0]
                    for address in publication.weight_addresses:  # driver-loop: immutable forward snapshot address slots
                        a.append(address)
                    a.append(0)
                    layer.addresses = a^
                    layer.parameters = parameters^
                    layer.output_slot = 10
                    layer.report_slot = -1
                    layer.report_sizes.clear()
                    layer.backward_parameters.clear()
                    layer.prepare_operation = -1
                    layer.tape = True
                    publication.publish_reports = False
                    publication.restore_owned = False
                else:
                    layer.backward_endpoint = layer.endpoint
                    if publication.transformer:
                        layer.backward_operation = 3 if publication.session is not None else 5
                    elif layer.operation == 8:
                        layer.backward_operation = 9
                        layer.owned_backward = True
                        layer.backward_parameters.append(layer.parameters[3])
                    else:
                        layer.backward_operation = 7 if publication.session is not None else 4
            except problem:
                if forward_setup:
                    layer.failure = String(problem)
                else:
                    layer.backward_failure = String(problem)

    def gradients(mut self, ids: PythonObject, targets: PythonObject, requested_items: Int,
                  automatic_items: Bool, tied: Bool, fused: Bool, chunked: Bool,
                  names: PythonObject) raises -> PythonObject:
        var loss = Float64(0)
        try:
            loss = self.native.gradients(generation_address(ids, self.keep), generation_address(targets, self.keep),
                requested_items, automatic_items, tied, fused, chunked)
        except problem:
            self.publish()
            raise problem
        self.publish()
        var result = Python.dict()
        py_setitem(result, "norm_f.weight", _samba_native_copy(self.native.norm_gradient, [self.width]))
        py_setitem(result, "embed.weight", _samba_native_copy(self.native.embedding_gradient, [self.vocab, self.width]))
        if not tied:
            py_setitem(result, "lm_head.weight", _samba_native_copy(self.native.head_gradient, [self.vocab, self.width]))
        for i in range(len(self.native.layers)):  # driver-loop: completed native gradient result publication
            ref publication = self.publications[i]
            for j in range(len(publication.gradient_names)):  # driver-loop: completed parameter gradient buffers
                py_setitem(result, "layers." + String(i) + "." + publication.gradient_names[j],
                    _samba_native_copy(self.native.layers[i].gradients[j], publication.gradient_shapes[j]))
        var ordered = Python.list()
        for i in range(len(names)):  # driver-loop: publish the model's admitted gradient registry order
            _ = py_call(ordered.append, result[names[i]])
        return Python.tuple(PythonObject(loss), ordered)

    def publish(mut self) raises:
        var report_names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
        for i in range(len(self.native.layers)):  # driver-loop: completed native operation publication, never execution
            ref layer = self.native.layers[i]
            ref publication = self.publications[i]
            if not layer.visited:
                continue
            if publication.session is not None:
                py_setitem(publication.fields, "_native_session" if publication.transformer else "_prefill_session", publication.session)
                py_setitem(publication.fields, "_session_binding" if publication.transformer else "_prefill_binding", publication.binding)
            if layer.prepared:
                py_setitem(publication.fields, "_owned_generation", PythonObject(layer.prepared_result))
                py_setitem(publication.fields, "_owned_restore_pending", False)
                py_setitem(publication.fields, "_owned_native_provider", publication.provider)
            if publication.transformer and publication.state is not None and publication.profile is not None:
                py_setattr(publication.state, "arithmetic_profile", publication.profile)
            if not layer.completed:
                continue
            if publication.state is not None:
                if publication.transformer:
                    py_setattr(publication.state, "cached_tokens", PythonObject(layer.result))
                else:
                    py_setattr(publication.state, "buffered_tokens", PythonObject(layer.result))
                    py_setattr(publication.state, "pending", False)
            if not publication.transformer and publication.publish_reports:
                for j in range(4):  # small-loop(four completed native reports)
                    var report = PythonObject(None)
                    if layer.report_sizes[j] != 0:
                        report = _samba_native_copy(layer.reports[j], publication.report_shapes[j])
                    py_setitem(publication.fields, report_names[j], report)

    def forward(mut self, ids: PythonObject) raises -> PythonObject:
        var logits = List[Float32]()
        try:
            logits = self.native.forward(generation_address(ids, self.keep))
        except problem:
            self.publish()
            raise problem
        self.publish()
        return _samba_native_copy(logits, [self.batch, self.length, self.vocab])


def samba_provider_eligible(fields: PythonObject) raises -> Bool:
    var blocks = fields["_blocks"]
    var bi = Python.import_module("builtins")
    for i in range(len(blocks)):  # driver-loop: route eligibility at the single operation boundary
        var block = _samba_native_fields(blocks[i])
        if String(block["_native_route"]) != "gpu" or String(block["numeric_mode"]) != "identical":
            return False
        if String(block["_native_kind"]) == "transformer":
            if block["_int15"] is not None or getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP", "0") == "1" or getenv("MOJOLEARN_TRANSFORMER_SESSION_FRESH", "1") == "0":
                return False
            var lock = OperationLockLease(block["_runtime_lock"])
            if lock.handle == 0:
                return False
        elif getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") == "1":
            return False
    return True


def samba_provider_forward(fields: PythonObject, config: PythonObject, entries: PythonObject, ids: PythonObject,
                           state: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    var primitive = py_get(fields, "_native_primitives")
    if primitive is None:
        return PythonObject(None)
    primitive = neural_descriptor_reuse(primitive, _native_function(entries, "neural_primitives_api_v1"), String(PRIMITIVE_ABI_NAME), 25)
    py_setitem(fields, "_native_primitives", primitive)
    if not samba_provider_eligible(fields):
        return PythonObject(None)
    var shape = ids.shape
    var admitted = AdmittedSambaExecution(fields, config, state, primitive, Int(py=shape[0]), Int(py=shape[1]), step, context)
    return admitted.forward(ids)


def samba_provider_gradients(fields: PythonObject, config: PythonObject, entries: PythonObject,
                             ids: PythonObject, targets: PythonObject, num_items: PythonObject,
                             stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    var primitive = py_get(fields, "_native_primitives")
    if primitive is None or not samba_provider_eligible(fields):
        return PythonObject(None)
    primitive = neural_descriptor_reuse(primitive, _native_function(entries, "neural_primitives_api_v1"), String(PRIMITIVE_ABI_NAME), 25)
    py_setitem(fields, "_native_primitives", primitive)
    var shape = ids.shape
    var admitted = AdmittedSambaExecution(fields, config, PythonObject(None), primitive,
        Int(py=shape[0]), Int(py=shape[1]), False, context)
    admitted.configure_training()
    var bi = Python.import_module("builtins")
    var fused = Bool(py=py_call(bi.hasattr, entries, "samba_head_loss"))
    var chunked = False
    if Bool(py=py_call(bi.hasattr, entries, "training_chunked_lm_head_enabled")):
        chunked = Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")())
    var probability = Float64(py=config["dropout"])
    if probability > 0.0 and stream is not None:
        var p = Float32(probability)
        var scale = Float32(1.0 / (1.0 - Float64(p)))
        var rng = _samba_native_fields(fields["generator"])
        var seed = rng["seed"]
        admitted.native.dropout = True
        admitted.native.dropout_words = [admitted.batch * admitted.length * admitted.width, token_offset * admitted.width,
            Int(py=seed & PythonObject(0xffffffff)), Int(py=(seed >> PythonObject(32)) & PythonObject(0xffffffff)), Int(py=stream), 2]
        admitted.native.dropout_scalars = [Float64(p), Float64(scale)]
    return admitted.gradients(ids, targets, 0 if num_items is None else Int(py=num_items), num_items is None,
        Bool(py=config["tie_embeddings"]), fused, chunked, fields["names"])
