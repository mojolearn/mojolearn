# SPDX-License-Identifier: Apache-2.0
# Public Transformer state ownership. These functions own
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
from bindings.neural_block_admission import neural_native_scalar, neural_array_slots
from std.ffi import _Global
from bindings.native_prepared_owner import prepared_key_add, prepared_key_same
from bindings.native_class_admission import class_record, class_admit, class_watch_ready


struct _GenerationStamp(Defaultable, Movable):
    # Identity key of the last generation class connection admitted by
    # generation_initialize_binding from this entry: the builtins and array
    # module handles, the class tuple, its items and every layer registry
    # key and class. Any replacement, insertion or removal changes the key.
    var key: PythonObject

    def __init__(out self):
        self.key = PythonObject(None)


comptime _GENERATION_STAMP = _Global[
    StorageType=_GenerationStamp, name="MojolearnTransformerGenerationStampV1",
    init_fn=_GenerationStamp.__init__,
]


def _generation_connected(mut mods: NativeModules, classes: PythonObject) raises:
    # generation_initialize_binding re-admits the whole class registry on
    # every call. Its result depends only on the identities in the key below,
    # so it runs again only when one of them changed.
    var key = mods.module("builtins").list()
    _ = py_call(key.append, mods.module("builtins"))
    _ = py_call(key.append, mods.module("array"))
    prepared_key_add(mods, key, classes)
    var g = _GENERATION_STAMP.get_or_create_ptr()
    if g[].key is not None and prepared_key_same(g[].key, key):
        return
    g[].key = PythonObject(None)
    _ = generation_initialize_binding(classes)
    g[].key = key


def _transformer_fields(mut mods: NativeModules, value: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    # _fields checks the dictionary descriptor before the native storage read.
    # Also reject field descriptors before bypassing them: replacing a data
    # field with a property must never silently change descriptor precedence.
    # The ancestry and descriptor checks read only the value's class rows
    # and the connected (cls, names) pair: once per class state
    # (native_class_admission). The record holds the connected owner class
    # and its field names, never the value's own class unless it is that
    # import-connected class.
    var value_class = py_type(value)
    var record = class_record(mods, _FIELDS_CLASS_SLOT, value_class)
    if record is None or record[0] is not cls or not _same_names(record[1], names):
        var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, value_class)
        var admitted = False
        for i in range(len(ancestry)):  # driver-loop: passive subclasses retain native owner storage
            admitted = admitted or ancestry[i] is cls
        if not admitted:
            raise Error("TypeError: Transformer operation requires its native-connected owner class")
        for i in range(len(ancestry)):  # driver-loop: owner layout metadata only
            var namespace = _class_fields(mods, ancestry[i])
            for j in range(len(names)):  # driver-loop: fixed owner storage schema
                if py_contains(namespace, names[j]):
                    raise Error("TypeError: Transformer storage fields cannot be replaced by class descriptors")
        # Not built where no record can be kept (Python 3.10/3.11).
        if class_watch_ready():
            class_admit(mods, _FIELDS_CLASS_SLOT, value_class, Python.tuple(cls, names))
    return _fields(mods, value)


def _same_names(kept: PythonObject, names: PythonObject) raises -> Bool:
    """The record's field names equal `names`: the same tuple, or the same
    exact strings in order. Callers build their names tuple per call (the
    decode step, lane p3-neural-blocks), so an identity test never hit and
    every step re-ran the ancestry walk and rewrote the record. Both tuples
    are native-built tuples of exact str: no Python comparison hook runs."""
    if kept is names:
        return True
    if len(kept) != len(names):
        return False
    for i in range(len(names)):  # driver-loop: fixed owner storage schema names
        if String(py=kept[i]) != String(py=names[i]):
            return False
    return True


comptime _FIELDS_CLASS_SLOT = "transformer_owner_fields_class_v1"
comptime _ARRAY_SLOTS_CLASS_SLOT = "transformer_array_slots_class_v1"


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
    var slots = context["array_slots"]
    var array_class = context["array_class"]
    var names = py_call(mods.module("builtins").list, slots.keys())
    # The slot identities depend only on the Array class rows and the
    # connected slot table: compared once per class state.
    var record = class_record(mods, _ARRAY_SLOTS_CLASS_SLOT, array_class)
    if record is not None and record[0] is slots and Int(py=record[1]) == len(names):
        var same = True
        for i in range(len(names)):  # driver-loop: pinned native Array storage layout
            if record[2][i] is not slots[names[i]]:
                same = False
                break
        if same:
            return slots
    var namespace = _class_fields(mods, array_class)
    # The record payload is built only where a record can be kept (not on
    # Python 3.10/3.11, which have no type watchers).
    var ready = class_watch_ready()
    var descriptors = mods.module("builtins").list()
    for i in range(len(names)):  # driver-loop: pinned native Array storage layout
        if py_get(namespace, names[i]) is not slots[names[i]]:
            raise Error("TypeError: Transformer output requires unchanged native Array storage slots")
        if ready:
            _ = py_call(descriptors.append, slots[names[i]])
    if ready:
        class_admit(mods, _ARRAY_SLOTS_CLASS_SLOT, array_class,
                    Python.tuple(slots, PythonObject(len(names)), py_call(mods.module("builtins").tuple, descriptors)))
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



# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by DELETION_SHA; code recoverable at DELETION_SHA^.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (`TransformerPublicTapeOwner`, the public `TransformerForwardTape` ticket owner.)


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



# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by DELETION_SHA; code recoverable at DELETION_SHA^.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (`forward_with_tape`, `backward_from_tape` and the ticket's init/close: their
# entry-record admission, compiled-mode cache, provider, handle and publish helpers.)


def transformer_public_call(mut mods: NativeModules, operation: String, values: PythonObject, context: PythonObject, neural: PythonObject) raises -> PythonObject:
    _generation_connected(mods, neural["generation_classes"])
    if operation == "state_init":
        return transformer_state_init(mods, values[0], values[1], context, neural)
    if operation == "state_capacity" or operation == "state_keys" or operation == "state_values":
        return transformer_state_read(mods, values[0], operation, context)
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
