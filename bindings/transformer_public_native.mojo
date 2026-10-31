# SPDX-License-Identifier: Apache-2.0
# Public Transformer state and training-ticket ownership. These functions own
# admission, buffer retention, native execution, and finished result creation.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_error_class import error_without_class
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from bindings.neural_block_public import neural_block_fields, neural_block_snapshot, neural_block_f32, neural_training_entries, _neural_input
from bindings.neural_block_admission import neural_native_scalar, neural_array_slots, neural_record_fields, neural_string_keys
from bindings.causal_lm_construct import _model_shape
from providers.python.operation_lock_pyapi import OperationLockLease
from providers.python.neural_descriptor_pyapi import neural_descriptor_connect, neural_descriptor_construct, neural_descriptor_handle
from providers.python.transformer_session_tape_pyapi import transformer_session_tape_forward, transformer_session_tape_backward, transformer_session_tape_discard
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME


def _transformer_fields(mut mods: NativeModules, value: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    # _fields checks the dictionary descriptor before the native storage read.
    # Also reject field descriptors before bypassing them: replacing a data
    # field with a property must never silently change descriptor precedence.
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(value))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: passive subclasses retain native owner storage
        admitted = admitted or ancestry[i] is cls
    if not admitted:
        raise Error("TypeError: Transformer operation requires its native-connected owner class")
    for i in range(len(ancestry)):  # driver-loop: owner layout metadata only
        var namespace = _class_fields(ancestry[i])
        for j in range(len(names)):  # driver-loop: fixed owner storage schema
            if py_contains(namespace, names[j]):
                raise Error("TypeError: Transformer storage fields cannot be replaced by class descriptors")
    return _fields(value)


def _transformer_fields(value: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _transformer_fields(mods, value, cls, names)


def _transformer_int(mut mods: NativeModules, value: PythonObject, name: String, neural: PythonObject) raises -> Int:
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if cls is not bi.str and not neural_native_scalar(mods, value, neural):
        raise Error("TypeError: Transformer " + name + " requires an ordinary scalar; Python conversion hooks are unsupported")
    return Int(py=py_call(bi.int, value))


def _transformer_stored_int(mut mods: NativeModules, fields: PythonObject, name: String) raises -> Int:
    var value = fields[name]
    if py_type(value) is not mods.module("builtins").int:
        raise Error("TypeError: Transformer " + name + " must be an ordinary integer")
    return Int(py=value)


def _transformer_array_slots(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    _ = neural_array_slots(context)
    var namespace = _class_fields(context["array_class"])
    var slots = context["array_slots"]
    var names = py_call(mods.module("builtins").list, slots.keys())
    for i in range(len(names)):  # driver-loop: pinned native Array storage layout
        if py_get(namespace, names[i]) is not slots[names[i]]:
            raise Error("TypeError: Transformer output requires unchanged native Array storage slots")
    return slots


struct TransformerBuffer(Movable):
    var owner: PythonObject
    var store: PythonObject
    var pin: AdmittedBorrowedBuffer
    var address: Int
    var count: Int
    var readonly: Bool

    def __init__(out self, mut mods: NativeModules, value: PythonObject, context: PythonObject) raises:
        self.owner = value
        var bi = mods.module("builtins")
        var store = value
        var is_array = py_type(value) is context["array_class"]
        var saved_address = 0
        var saved_count = 1
        var saved_readonly = False
        if is_array:
            var slots = _transformer_array_slots(mods, context)
            var cls = context["array_class"]
            var dtype = py_call(slots["dtype"].__get__, value, cls)
            var order = py_call(slots["order"].__get__, value, cls)
            var shape = py_call(slots["shape"].__get__, value, cls)
            var address = py_call(slots["_addr"].__get__, value, cls)
            var readonly = py_call(slots["_readonly"].__get__, value, cls)
            var strides = py_call(slots["strides"].__get__, value, cls)
            var size = py_call(slots["size"].__get__, value, cls)
            var nbytes = py_call(slots["nbytes"].__get__, value, cls)
            var itemsize = py_call(slots["itemsize"].__get__, value, cls)
            if (py_type(dtype) is not bi.str or py_type(order) is not bi.str
                    or py_type(shape) is not bi.tuple or py_type(address) is not bi.int
                    or py_type(readonly) is not bi.bool or py_type(strides) is not bi.tuple
                    or py_type(size) is not bi.int or py_type(nbytes) is not bi.int
                    or py_type(itemsize) is not bi.int):
                raise Error("TypeError: Transformer cache requires ordinary Array metadata")
            if String(dtype) != "<f4" or String(order) != "C":
                raise Error("TypeError: Transformer cache requires contiguous float32 storage")
            for i in range(len(shape)):  # driver-loop: capture shape metadata once
                if py_type(shape[i]) is not bi.int:
                    raise Error("TypeError: Transformer cache dimensions must be ordinary integers")
                var dimension = Int(py=shape[i])
                if dimension < 0 or (dimension != 0 and saved_count > 2147483647 // dimension):
                    raise Error("Transformer cache dimensions exceed native indexing")
                saved_count *= dimension
            if len(shape) != len(strides) or Int(py=size) != saved_count or Int(py=nbytes) != saved_count * 4 or Int(py=itemsize) != 4:
                raise Error("Transformer cache shape and storage metadata differ")
            var step = 4
            for reverse in range(len(shape)):  # driver-loop: validate the captured contiguous strides
                var axis = len(shape) - reverse - 1
                if py_type(strides[axis]) is not bi.int or (saved_count > 0 and Int(py=shape[axis]) > 1 and Int(py=strides[axis]) != step):
                    raise Error("Transformer cache strides differ from its contiguous shape")
                step *= Int(py=shape[axis])
            saved_address = Int(py=address)
            saved_readonly = Bool(py=readonly)
            store = py_call(slots["_store"].__get__, value, cls)
            if store is None:
                store = py_call(slots["_mv"].__get__, value, cls)
                if py_type(store) is not bi.memoryview or store.obj is None:
                    raise Error("TypeError: Transformer cache Array must retain its native exporter")
        admit_native_buffer_exporter(store)
        self.store = store
        self.pin = AdmittedBorrowedBuffer(store)
        var layout = self.pin.layout.value().copy()
        if not layout.contiguous():
            raise Error("TypeError: Transformer cache requires C-contiguous native storage")
        if is_array:
            var offset = saved_address - layout.address
            if saved_count != 0 and (offset < 0 or offset > layout.count * layout.itemsize
                    or saved_count > (layout.count * layout.itemsize - offset) // 4):
                raise Error("Transformer cache view lies outside its retained storage")
            self.address = saved_address
            self.count = saved_count
            self.readonly = saved_readonly or layout.readonly
        else:
            if (self.pin.format != "f" and self.pin.format != "<f"
                    and self.pin.format != "=f" and self.pin.format != "@f") or layout.itemsize != 4:
                raise Error("TypeError: Transformer cache requires float32 storage")
            self.address = layout.address
            self.count = layout.count
            self.readonly = layout.readonly


def _transformer_state_words(mut mods: NativeModules, fields: PythonObject) raises -> List[Int]:
    var result = List[Int]()
    for name in [String("batch_size"), String("n_kv_heads"), String("head_dim"), String("max_tokens"), String("cached_tokens"), String("window")]:  # driver-loop: fixed cache scalar schema
        result.append(_transformer_stored_int(mods, fields, name))
    if result[0] < 1 or result[1] < 1 or result[2] < 1 or result[3] < 1 or result[4] < 0 or result[4] > result[3] or result[5] < 0:
        raise Error("TransformerState dimensions/cached_tokens/window are invalid")
    var capacity = result[5] if result[5] > 0 else result[3]
    var count = 1
    for dimension in [result[0], result[1], capacity, result[2]]:  # driver-loop: cache extent overflow admission
        if count > 2147483647 // dimension:
            raise Error("TransformerState cache exceeds signed 32-bit indexing")
        count *= dimension
    result.append(count)
    return result^


def _transformer_state_words(fields: PythonObject) raises -> List[Int]:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _transformer_state_words(mods, fields)


def transformer_state_init(mut mods: NativeModules, state: PythonObject, values: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    var fields = _transformer_fields(mods, state, context["state_class"], context["state_fields"])
    var fresh = Python.dict()
    var names = context["state_fields"]
    for i in range(6):  # driver-loop: scalar constructor arguments, no conversion protocols
        py_setitem(fresh, names[i], PythonObject(_transformer_int(mods, values[i], String(names[i]), neural)))
    var words = _transformer_state_words(mods, fresh)
    var keys = TransformerBuffer(mods, values[6], context)
    var vals = TransformerBuffer(mods, values[7], context)
    if keys.count < words[6] or vals.count < words[6]:
        raise Error("TransformerState cache buffers are smaller than their declared capacity")
    py_setitem(fresh, "k_cache", values[6])
    py_setitem(fresh, "v_cache", values[7])
    _ = py_call(fields.update, fresh)
    return PythonObject(None)


def _transformer_cache_view(mut mods: NativeModules, buffer: TransformerBuffer, dims: List[Int], context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var slots = _transformer_array_slots(mods, context)
    var cls = context["array_class"]
    var result = py_call(bi.object.__new__, cls)
    var count = 1
    for dimension in dims:  # driver-loop: result shape metadata
        count *= dimension
    var view = py_call(bi.memoryview, buffer.store)
    var bytes = py_call(view.cast, "B")
    var offset = 0 if count == 0 else buffer.address - buffer.pin.layout.value().address
    var selected = bytes[py_call(bi.slice, offset, offset + count * 4)]
    var flat = py_call(selected.cast, "f")
    if buffer.readonly and not Bool(py=flat.readonly):
        flat = py_call(flat.toreadonly)
    var fields = Python.dict()
    py_setitem(fields, "_store", buffer.store)
    py_setitem(fields, "_base", buffer.owner)
    py_setitem(fields, "_pin", view)
    py_setitem(fields, "_mv", flat)
    py_setitem(fields, "_addr", PythonObject(buffer.address))
    py_setitem(fields, "_readonly", PythonObject(buffer.readonly))
    py_setitem(fields, "shape", py_call(bi.tuple, py_ints(dims)))
    py_setitem(fields, "dtype", PythonObject("<f4"))
    py_setitem(fields, "ndim", PythonObject(len(dims)))
    py_setitem(fields, "size", PythonObject(count))
    py_setitem(fields, "nbytes", PythonObject(count * 4))
    py_setitem(fields, "itemsize", PythonObject(4))
    py_setitem(fields, "order", PythonObject("C"))
    var strides = List[Int](length=len(dims), fill=0)
    var stride = 4
    for reverse in range(len(dims)):  # driver-loop: output stride metadata
        var axis = len(dims) - reverse - 1
        strides[axis] = stride
        stride *= dims[axis]
    py_setitem(fields, "strides", py_call(bi.tuple, py_ints(strides)))
    var names = py_call(bi.list, fields.keys())
    for i in range(len(names)):  # driver-loop: exact native slot writes, never setattr hooks
        _ = py_call(slots[names[i]].__set__, result, fields[names[i]])
    return result


def transformer_state_read(mut mods: NativeModules, state: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    var fields = _transformer_fields(mods, state, context["state_class"], context["state_fields"])
    var words = _transformer_state_words(mods, fields)
    var b = words[0]; var kv = words[1]; var hd = words[2]
    var cached = words[4]; var window = words[5]
    if operation == "state_capacity":
        return PythonObject(window if window > 0 else words[3])
    var buffer = TransformerBuffer(mods, fields["k_cache" if operation == "state_keys" else "v_cache"], context)
    if buffer.count < words[6]:
        raise Error("TransformerState cache buffers are smaller than their declared capacity")
    var held = min(cached, window) if window > 0 else cached
    var dims: List[Int] = [b, kv, held, hd]
    if window == 0:
        return _transformer_cache_view(mods, buffer, dims, context)
    _ = _transformer_array_slots(mods, context)
    var output = generation_array(dims)
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=buffer.address)
    var start = (cached - held) % window
    var first = min(held, window - start)
    for head in range(b * kv):  # cpu-route: byte-preserving chronological ring-cache copy
        if first != 0:
            memcpy(dest=dst + head * held * hd, src=src + (head * window + start) * hd, count=first * hd)
        if held > first:
            memcpy(dest=dst + (head * held + first) * hd, src=src + head * window * hd, count=(held - first) * hd)
    _ = buffer
    return output



struct TransformerPublicTapeOwner(Movable, Writable):
    var block: PythonObject
    var session: PythonObject
    var binding: PythonObject
    var provider: PythonObject
    var handle: Int
    var token: List[Int]
    var shape: List[Int]
    var configuration: List[Int]
    var profile: String
    var numeric_profile: String
    var weight_shapes: List[List[Int]]
    var consumed: Bool

    def __init__(out self, block: PythonObject, session: PythonObject, binding: PythonObject,
                 provider: PythonObject, handle: Int, var token: List[Int],
                 var shape: List[Int], var configuration: List[Int], var profile: String,
                 var numeric_profile: String, var weight_shapes: List[List[Int]]):
        self.block = block
        self.session = session
        self.binding = binding
        self.provider = provider
        self.handle = handle
        self.token = token^
        self.shape = shape^
        self.configuration = configuration^
        self.profile = profile^
        self.numeric_profile = numeric_profile^
        self.weight_shapes = weight_shapes^
        self.consumed = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerPublicTapeOwner(consumed=", self.consumed, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _transformer_configuration(mut mods: NativeModules, fields: PythonObject) raises -> List[Int]:
    var result = List[Int]()
    for key in [String("d_model"), String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("window")]:  # driver-loop: fixed block configuration snapshot
        result.append(_transformer_stored_int(mods, fields, key))
    if (result[0] < 1 or result[1] < 1 or result[2] < 1 or result[3] < 1
            or result[4] < 1 or result[5] < 0 or result[1] > 2147483647 // result[3]
            or result[0] != result[1] * result[3] or result[1] % result[2] != 0 or result[3] % 2 != 0):
        raise Error("Transformer configuration has invalid attention/MLP dimensions")
    var tail = fields["_opts_tail"]
    var bi = mods.module("builtins")
    if py_type(tail) is not bi.tuple and py_type(tail) is not bi.list:
        raise Error("TypeError: Transformer options require an ordinary integer vector")
    for i in range(len(tail)):  # driver-loop: checked fixed native block option words
        if py_type(tail[i]) is not bi.int:
            raise Error("TypeError: Transformer option words must be ordinary integers")
        result.append(Int(py=tail[i]))
    return result^


def _transformer_configuration(fields: PythonObject) raises -> List[Int]:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _transformer_configuration(mods, fields)


def _transformer_same_words(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # driver-loop: native shape/configuration metadata comparison
        if left[i] != right[i]:
            return False
    return True



def _transformer_block_snapshot(mut mods: NativeModules, block: PythonObject, neural: PythonObject) raises -> PythonObject:
    _ = neural_record_fields(mods, block, Python.tuple("_native_session", "_session_binding", "_runtime_lock"))
    return neural_block_snapshot(block, neural)


def _transformer_entry_fields(mut mods: NativeModules, entries: PythonObject) raises -> PythonObject:
    var types = mods.module("types")
    if py_type(entries) is not types.SimpleNamespace:
        raise Error("TypeError: Transformer tape requires its connected native entry record")
    # Exact SimpleNamespace has native member storage, unlike ordinary model
    # __dict__ getset descriptors. Read that descriptor directly after type
    # admission; never use a module/instance attribute lookup as an entry probe.
    var fields = py_call(types.SimpleNamespace.__dict__["__dict__"].__get__, entries, types.SimpleNamespace)
    neural_string_keys(mods, fields)
    return fields


def _transformer_native_entry(mut mods: NativeModules, entries: PythonObject, name: String) raises -> PythonObject:
    var fields = _transformer_entry_fields(mods, entries)
    var entry = py_get(fields, name)
    if py_type(entry) is not py_type(mods.module("builtins").len):
        raise Error("NotImplementedError: Transformer tape lacks its connected compiled " + name + " entry")
    return entry


def _transformer_tape_provider(mut mods: NativeModules, entries: PythonObject) raises -> PythonObject:
    # The ticket retains this provider through backward/discard. Keep it out
    # of model state: ordinary getstate/setstate remains unchanged, and the
    # existing 36-operation block caches need not be resized or invalidated.
    var windowed_result19 = neural_descriptor_connect(_transformer_native_entry(mods, entries, "transformer_block_api_v1"),
        String(TRANSFORMER_BLOCK_ABI_NAME), 41)
    mods.released()
    return windowed_result19^


def _transformer_tape_handle(mut mods: NativeModules, provider: PythonObject, entries: PythonObject, session: PythonObject) raises -> Int:
    var windowed_result18 = neural_descriptor_handle(provider,
        _transformer_native_entry(mods, entries, "transformer_native_session_handle_v1"), session)
    mods.released()
    return windowed_result18^


def _transformer_old_binding(mut mods: NativeModules, binding: PythonObject, neural: PythonObject) raises -> PythonObject:
    # Session cache fields are mutable. A replaced binding must not reach
    # _native_function's attribute lookup before proving it is connected.
    var registered = neural["training_bindings"]
    var keys = py_call(mods.module("builtins").list, registered.keys())
    for i in range(len(keys)):  # driver-loop: import-time compiled binding identities
        var entries = registered[keys[i]]["transformer"]
        if py_type(entries) is not mods.module("builtins").str:
            if binding is _transformer_entry_fields(mods, entries)["canonical"]:
                return entries
    raise Error("TypeError: Transformer session binding is not an admitted native module")


def _transformer_tape_entries(mut mods: NativeModules, block: PythonObject, fields: PythonObject, neural: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(block))
    for i in range(len(ancestry)):  # driver-loop: preserve the existing inference-only refusal
        if ancestry[i] is neural["block_classes"]["cpu"]["transformer"]:
            raise Error("NotImplementedError: TransformerBlockInference loaded binding lacks the native tape API")
    if py_get(fields, "_int15") is not None or Bool(py=fields["_extended"]):
        raise Error("NotImplementedError: Transformer training tape requires default FP32 block options")
    var entries = neural_training_entries(mods, fields, neural)
    if Int(py=_transformer_native_entry(mods, entries, "transformer_numeric_mode")()) != 1:
        mods.released()
        raise Error("NotImplementedError: Transformer training tape requires identical mode")
    mods.released()
    var live = fields["_neural_live_fields"]
    for key in [String("_arithmetic_profile"), String("_arithmetic_profile_mode"), String("_arithmetic_profile_changed")]:  # driver-loop: binding profile bookkeeping on the live owner
        py_setitem(live, key, fields[key])
    return entries


def _transformer_tape_publish(mut mods: NativeModules, tape: PythonObject, block: PythonObject, fields: PythonObject,
                               entries: PythonObject, provider: PythonObject, handle: Int,
                               token_value: PythonObject, var shape: List[Int],
                               var weight_shapes: List[List[Int]], context: PythonObject) raises:
    var slots = _transformer_fields(mods, tape, context["tape_class"], context["tape_fields"])
    var bi = mods.module("builtins")
    if (py_type(token_value) is not bi.tuple and py_type(token_value) is not bi.list) or len(token_value) != 4:
        raise Error("TypeError: Transformer tape token requires four native integers")
    var token = List[Int]()
    for i in range(4):  # driver-loop: original native session ticket words
        if py_type(token_value[i]) is not bi.int:
            raise Error("TypeError: Transformer tape token requires native integer words")
        token.append(Int(py=token_value[i]))
    var profile = py_get(fields, "_arithmetic_profile")
    if py_type(profile) is not bi.str:
        raise Error("TypeError: Transformer tape requires a native arithmetic profile")
    var owner = TransformerPublicTapeOwner(block, fields["_native_session"], entries.canonical,
        provider, handle, token.copy(), shape.copy(),
        _transformer_configuration(mods, fields), String(profile), String(fields["numeric_profile"]), weight_shapes^)
    py_setitem(slots, "_native_owner", PythonObject(alloc=owner^))
    py_setitem(slots, "_block", block)
    py_setitem(slots, "_session", fields["_native_session"])
    py_setitem(slots, "_binding", entries.canonical)
    py_setitem(slots, "_token", Python.tuple(PythonObject(token[0]), PythonObject(token[1])))
    py_setitem(slots, "retained", PythonObject(token[2] != 0))
    py_setitem(slots, "stage_bytes", PythonObject(token[3]))
    py_setitem(slots, "shape", py_call(bi.tuple, py_ints(shape)))
    py_setitem(slots, "profile", profile)
    py_setitem(slots, "_consumed", PythonObject(False))


def _transformer_weight_shapes(weights: PythonObject, configuration: List[Int], neural: PythonObject,
                                mut owners: List[PythonObject], mut addresses: List[Int]) raises -> List[List[Int]]:
    var shapes = List[List[Int]]()
    var d = configuration[0]
    var kw = configuration[2] * configuration[3]
    var it = configuration[4]
    var expected = List[List[Int]]()
    expected.append([d]); expected.append([d])
    expected.append([d, d]); expected.append([kw, d]); expected.append([kw, d])
    expected.append([d, d]); expected.append([it, d]); expected.append([it, d]); expected.append([d, it])
    if len(weights) != len(expected):
        raise Error("Transformer training tape requires nine base weights")
    for i in range(len(weights)):  # driver-loop: snapshot each retained weight buffer once
        var weight = weights[i]  # captured/prepared once by neural_block_snapshot
        if weight is None:
            raise Error("Transformer training tape requires all nine base weights")
        _model_shape(weight, expected[i], "Transformer tape weight")
        var dims = List[Int]()
        for j in range(len(weight.shape)):  # driver-loop: fresh admitted native Array shape
            dims.append(Int(py=weight.shape[j]))
        shapes.append(dims^)
        addresses.append(Int(py=weight._addr))
        owners.append(weight)
    return shapes^


def transformer_forward_with_tape(mut mods: NativeModules, block: PythonObject, input: PythonObject, budget_value: PythonObject,
                                  cost_value: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    var fields = _transformer_block_snapshot(mods, block, neural)
    var entries = _transformer_tape_entries(mods, block, fields, neural)
    var live = fields["_neural_live_fields"]
    var provider = _transformer_tape_provider(mods, entries)
    var budget = _transformer_int(mods, budget_value, "activation_budget_bytes", neural)
    var cost = _transformer_int(mods, cost_value, "minimum_replay_ops_per_byte", neural)
    if budget < 0 or cost < 0:
        raise Error("Transformer activation budget and replay cost must be nonnegative")
    _ = _transformer_array_slots(mods, context)
    var configuration = _transformer_configuration(mods, fields)
    var x = _neural_input(mods, input, configuration[0], False, neural)
    var shape: List[Int] = [Int(py=x.shape[0]), Int(py=x.shape[1]), configuration[0]]
    var y = generation_array(shape)
    var keep = List[PythonObject]()
    var addresses: List[Int] = [Int(py=x._addr)]
    var weight_shapes = _transformer_weight_shapes(fields["_w"], configuration, neural, keep, addresses)
    if len(weight_shapes) != 9:
        raise Error("Transformer training tape requires nine base weights")
    addresses.append(Int(py=y._addr))
    var params: List[Int] = [shape[0], shape[1], configuration[0], configuration[1], configuration[2],
        configuration[3], configuration[4], configuration[5], budget, cost]
    var lease = OperationLockLease(fields["_runtime_lock"])
    lease.acquire()
    mods.released()
    var tape = PythonObject(None)
    try:
        var session = live["_native_session"]
        var binding = live["_session_binding"]
        if session is not None and binding is not entries.canonical:
            var previous = _transformer_old_binding(mods, binding, neural)
            # Lifetime cleanup remains this admitted native entry: its close is
            # idempotent, while the existing GPU handle-admission entry rejects
            # closed owners. No numerical work uses this compiled close door.
            _ = py_call(_transformer_native_entry(mods, previous, "transformer_session_close"), session)
            mods.released()
            session = PythonObject(None)
            py_setitem(live, "_native_session", session)
            py_setitem(live, "_session_binding", PythonObject(None))
        if session is None:
            session = neural_descriptor_construct(provider, _transformer_native_entry(mods, entries, "transformer_session_create"))
            mods.released()
            py_setitem(live, "_native_session", session)
            py_setitem(live, "_session_binding", entries.canonical)
        py_setitem(fields, "_native_session", session)
        var handle = _transformer_tape_handle(mods, provider, entries, session)
        var report = transformer_session_tape_forward(provider, session, handle, addresses, params)
        mods.released()
        tape = py_call(mods.module("builtins").object.__new__, context["tape_class"])
        _transformer_tape_publish(mods, tape, block, fields, entries, provider, handle,
            py_ints(report), shape.copy(), weight_shapes^, context)
        lease.release()
    except problem:
        mods.released()
        lease.release()
        raise problem
    _ = keep^
    _ = x^
    return Python.tuple(y, tape)


def transformer_tape_init(mut mods: NativeModules, tape: PythonObject, values: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    var block = values[0]
    var fields = _transformer_block_snapshot(mods, block, neural)
    var entries = _transformer_tape_entries(mods, block, fields, neural)
    if values[1] is not fields["_native_session"] or values[2] is not entries.canonical:
        raise Error("Transformer tape requires the block's current native session and binding")
    var bi = mods.module("builtins")
    var shape_value = values[4]
    if py_type(shape_value) is not bi.tuple and py_type(shape_value) is not bi.list:
        raise Error("TypeError: Transformer tape shape requires an ordinary integer vector")
    var shape = List[Int]()
    for i in range(len(shape_value)):  # driver-loop: constructor result shape
        if py_type(shape_value[i]) is not bi.int or Int(py=shape_value[i]) < 1:
            raise Error("TypeError: Transformer tape shape requires positive native integers")
        shape.append(Int(py=shape_value[i]))
    if len(shape) != 3 or shape[2] != _transformer_stored_int(mods, fields, "d_model"):
        raise Error("Transformer tape shape must be (B, L, d_model)")
    if py_type(values[6]) is not bi.str or String(values[6]) != String(fields["_arithmetic_profile"]):
        raise Error("Transformer tape arithmetic profile differs from the block")
    var configuration = values[5]
    if py_type(configuration) is not bi.tuple or len(configuration) != 8:
        raise Error("TypeError: Transformer tape configuration requires its native tuple schema")
    var expected = _transformer_configuration(mods, fields)
    for i in range(6):  # driver-loop: scalar configuration supplied to the public ticket constructor
        if py_type(configuration[i]) is not bi.int or Int(py=configuration[i]) != expected[i]:
            raise Error("Transformer tape configuration differs from its block")
    var tail = configuration[6]
    if py_type(tail) is not bi.tuple or len(tail) != len(expected) - 6:
        raise Error("TypeError: Transformer tape options require their native integer tuple")
    for i in range(len(tail)):  # driver-loop: option configuration words
        if py_type(tail[i]) is not bi.int or Int(py=tail[i]) != expected[i + 6]:
            raise Error("Transformer tape options differ from its block")
    if py_type(configuration[7]) is not bi.str or String(configuration[7]) != String(fields["numeric_profile"]):
        raise Error("Transformer tape numeric profile differs from its block")
    var keep = List[PythonObject]()
    var addresses = List[Int]()
    var weight_shapes = _transformer_weight_shapes(fields["_w"], expected, neural, keep, addresses)
    var provider = _transformer_tape_provider(mods, entries)
    var handle = _transformer_tape_handle(mods, provider, entries, fields["_native_session"])
    _transformer_tape_publish(mods, tape, block, fields, entries, provider, handle,
        values[3], shape^, weight_shapes^, context)
    _ = keep^
    return PythonObject(None)


def transformer_backward_from_tape(mut mods: NativeModules, block: PythonObject, tape: PythonObject, input: PythonObject,
                                   context: PythonObject, neural: PythonObject) raises -> PythonObject:
    # The documented contract: anything that is not this block's tape (None,
    # another object, another block's tape) is refused as a foreign tape with
    # ValueError, before owner storage is read.
    var ancestry = py_call(mods.module("builtins").type.__dict__["__mro__"].__get__, py_type(tape))
    var is_tape = False
    for i in range(len(ancestry)):  # driver-loop: tape class ancestry metadata
        is_tape = is_tape or ancestry[i] is context["tape_class"]
    if not is_tape:
        raise Error("Transformer tape belongs to another block")
    var slots = _transformer_fields(mods, tape, context["tape_class"], context["tape_fields"])
    var hobj = slots["_native_owner"]
    var owner = hobj.downcast_value_ptr[TransformerPublicTapeOwner]()
    if owner[].block is not block:
        raise Error("Transformer tape belongs to another block")
    var fields = _transformer_block_snapshot(mods, block, neural)
    var entries = _transformer_tape_entries(mods, block, fields, neural)
    _ = _transformer_array_slots(mods, context)
    var dy = neural_block_f32(input, "Transformer tape grad_output", neural)
    _model_shape(dy, owner[].shape, "Transformer tape grad_output")
    var lease = OperationLockLease(fields["_runtime_lock"])
    lease.acquire()
    mods.released()
    var result = Python.dict()
    try:
        if (owner[].consumed or owner[].session is not fields["_native_session"]
                or owner[].binding is not entries.canonical
                or owner[].profile != String(fields["_arithmetic_profile"])
                or owner[].numeric_profile != String(fields["numeric_profile"])
                or not _transformer_same_words(owner[].configuration, _transformer_configuration(mods, fields))):
            raise Error("Transformer stale, consumed or incompatible tape")
        var keep = List[PythonObject]()
        var addresses: List[Int] = [Int(py=dy._addr)]
        var dx = generation_array(owner[].shape)
        addresses.append(Int(py=dx._addr))
        py_setitem(result, "x", dx)
        keep.append(dx)
        var names = context["weight_names"]
        for i in range(len(owner[].weight_shapes)):  # driver-loop: one native output buffer per weight
            var gradient = generation_array(owner[].weight_shapes[i])
            addresses.append(Int(py=gradient._addr))
            py_setitem(result, names[i], gradient)
            keep.append(gradient)
        owner[].consumed = True
        py_setitem(slots, "_consumed", PythonObject(True))
        transformer_session_tape_backward(owner[].provider, owner[].session, owner[].handle,
            addresses, owner[].token[0], owner[].token[1])
        mods.released()
        _ = keep^
        lease.release()
    except problem:
        mods.released()
        lease.release()
        raise problem
    _ = dy^
    _ = hobj
    return result


def transformer_tape_close(mut mods: NativeModules, tape: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    var slots = _transformer_fields(mods, tape, context["tape_class"], context["tape_fields"])
    var hobj = slots["_native_owner"]
    var owner = hobj.downcast_value_ptr[TransformerPublicTapeOwner]()
    var fields = neural_block_fields(owner[].block, neural)
    _ = neural_record_fields(mods, owner[].block, Python.tuple("_native_session", "_runtime_lock"))
    var lease = OperationLockLease(fields["_runtime_lock"])
    lease.acquire()
    mods.released()
    try:
        if not owner[].consumed and owner[].session is fields["_native_session"]:
            transformer_session_tape_discard(owner[].provider, owner[].session, owner[].handle,
                owner[].token[0], owner[].token[1])
            mods.released()
        owner[].consumed = True
        py_setitem(slots, "_consumed", PythonObject(True))
        lease.release()
    except problem:
        mods.released()
        lease.release()
        raise problem
    _ = hobj
    return PythonObject(None)


def transformer_public_call(mut mods: NativeModules, operation: String, values: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(neural["generation_classes"])
    if operation == "state_init":
        return transformer_state_init(mods, values[0], values[1], context, neural)
    if operation == "state_capacity" or operation == "state_keys" or operation == "state_values":
        return transformer_state_read(mods, values[0], operation, context)
    if operation == "forward_tape":
        return transformer_forward_with_tape(mods, values[0], values[1], values[2], values[3], context, neural)
    if operation == "backward_tape":
        return transformer_backward_from_tape(mods, values[0], values[1], values[2], context, neural)
    if operation == "tape_init":
        return transformer_tape_init(mods, values[0], values[1], context, neural)
    if operation == "tape_close":
        return transformer_tape_close(mods, values[0], context, neural)
    if operation == "tape_reduce":
        raise Error("TypeError: TransformerForwardTape is a live native owner and cannot be serialized")
    raise Error("Transformer public operation is not connected: " + operation)


def transformer_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 4 or py_type(args[0]) is not mods.module("builtins").str:
            raise Error("TypeError: Transformer public operation requires its connected arguments")
        var result = transformer_public_call(mods, String(args[0]), args[1], args[2], args[3])
        return result.steal_data()
    except problem:
        if String(problem).startswith("TypeError:"):
            return raise_python_exception(error_without_class(problem, "TypeError"), ExceptionType.TypeError)
        if String(problem).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(problem, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(problem, "ValueError"), ExceptionType.ValueError)
