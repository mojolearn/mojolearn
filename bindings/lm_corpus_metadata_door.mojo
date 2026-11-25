# SPDX-License-Identifier: Apache-2.0
"""Native corpus records, passive metadata capture and int32 array ownership."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.neural_block_admission_door import neural_array_slots, neural_array_snapshot, neural_native_scalar
from bindings.native_array_storage_door import native_array_zeros
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.lm_corpus_files_door import corpus_files_initialize
from bindings.lm_corpus_scalars_unpack import corpus_numeric_base
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter, connect_native_mmap_exporter
from core.shared_slot import SharedSlot
from core.native_json_value import (JsonDoc, json_add_array, json_add_bool, json_add_null, json_add_number,
    json_add_object, json_add_string, json_push, json_put)
from bindings.lm_keys_unpack import lm_keys
from bindings.py_balanced import py_item
from core.entry_table import EntryRef, entry_ref_unresolved
from core.tokenizer_gpu_entry_table import tokenizer_gpu_resolve, tokenizer_gpu_vendor_word


struct CorpusConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime CORPUS_CONNECTION = SharedSlot[CorpusConnection, "MojolearnCorpusPublicConnectionV1"]


struct CorpusGpuConnection(Defaultable, Movable):
    """What `lm_corpus.prepare(device=...)` needs, read ONCE at the import-time connection (lane
    vdf2-tok): whether this installation is CPU only (the backend's `_CPU_ONLY` record), the GPU
    tokenizer image's entry resolved from its table (core/tokenizer_gpu_entry_table.mojo), or the
    words device='gpu' is refused with when that image is not connected. No Python object."""

    var cpu_only: Bool
    var refusal: String
    var entry: EntryRef

    def __init__(out self):
        self.cpu_only = True
        self.refusal = String("RuntimeError: corpus operations require import-time native connection")
        self.entry = entry_ref_unresolved()


comptime CORPUS_GPU_CONNECTION = SharedSlot[CorpusGpuConnection, "MojolearnCorpusGpuConnectionV1"]


def corpus_gpu_connect(backend_fields: PythonObject, gpu: PythonObject) raises:
    """UNPACK at import: the backend's `_CPU_ONLY` record and the GPU tokenizer binding's connection
    tuple (`mojolearn.tokenizer._gpu_binding`: the module first, or one refusal string). The entry
    is resolved here once; a call never looks it up."""
    var bi = native_loaded_module("builtins")
    if py_type(backend_fields) is not bi.dict:
        raise Error("TypeError: corpus connection requires the backend's fields")
    if py_type(gpu) is not bi.tuple or len(gpu) < 1:
        raise Error("TypeError: corpus connection requires the GPU tokenizer connection tuple")
    ref slot = CORPUS_GPU_CONNECTION.get_or_create_ptr()[]
    slot.cpu_only = py_get(backend_fields, PythonObject("_CPU_ONLY")) is not None
    if py_type(gpu[0]) is bi.str:
        slot.refusal = String(py=gpu[0])
        return
    try:
        var module = gpu[0]
        var mode = Int(py=py_call(module.x_tokenizer_numeric_mode))
        var vendor = tokenizer_gpu_vendor_word(String(py=py_call(module.x_tokenizer_vendor)))
        slot.entry = tokenizer_gpu_resolve(mode, vendor)
        slot.refusal = String("")
    except error:
        slot.refusal = String("ImportError: device='gpu' needs the GPU tokenizer binding, which is not connected: ") + String(error)


def corpus_context() raises -> PythonObject:
    var context = CORPUS_CONNECTION.get_or_create_ptr()[].context
    if context is None:
        raise Error("RuntimeError: corpus operations require import-time native connection")
    return context


def corpus_connect(args: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    if py_type(args) is not bi.tuple or len(args) != 8:
        raise Error("TypeError: corpus connection requires its native classes")
    var prior = CORPUS_CONNECTION.get_or_create_ptr()[].context
    if prior is not None:
        var prior_names = Python.tuple("corpus_class", "batches_class", "iterator_class", "array_class", "mmap_class")
        for i in range(5):  # small-loop(import connection cannot replace retained owner identities)
            if args[i] is not prior[prior_names[i]]:
                raise Error("TypeError: corpus import connection cannot be replaced")
        if args[5] is not py_item(prior, nk[].k_scalar_connection):
            raise Error("TypeError: corpus scalar admission connection cannot be replaced")
        return PythonObject(None)
    var context = Python.dict()
    var names = Python.tuple("corpus_class", "batches_class", "iterator_class", "array_class", "mmap_class")
    for i in range(5):  # small-loop(five import-connected native/public record classes)
        if py_type(args[i]) is not bi.type:
            raise Error("TypeError: corpus connection classes must have ordinary native type metadata")
        py_setitem(context, names[i], args[i])
    corpus_dict(args[5])
    var admission = py_call(bi.dict, args[5])
    py_setitem(admission, nk[].k_native_module_type, py_type(bi))
    py_setitem(context, nk[].k_scalar_connection, args[5])
    py_setitem(context, nk[].k_admission_context, admission)
    py_setitem(context, nk[].k_array_slots, neural_array_slots(context))
    py_setitem(context, nk[].k_batch_ids_method, py_item(_class_fields(py_item(context, nk[].k_batches_class)), nk[].k_ids))
    py_setitem(context, nk[].k_mmap_hooks, corpus_mmap_hooks(py_item(context, nk[].k_mmap_class)))
    connect_native_mmap_exporter(py_item(context, nk[].k_mmap_class))
    corpus_files_initialize()
    # args[6], args[7]: the backend's fields and the GPU tokenizer connection (lane vdf2-tok).
    corpus_gpu_connect(args[6], args[7])
    CORPUS_CONNECTION.get_or_create_ptr()[].context = context
    return PythonObject(None)


def corpus_int(value: PythonObject, name: String, strict: Bool = False) raises -> Int:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is bi.int:
        # An exact int converts natively; int() would only return it.
        return Int(py=value)
    if strict or (cls is not bi.str and cls is not bi.bytes and not neural_native_scalar(value, corpus_context())):
        raise Error("TypeError: corpus " + name + " requires a native integer; Python conversion hooks are unsupported")
    return Int(py=py_call(bi.int, value))


def corpus_batch_base(step: PythonObject, batch: Int, length: Int) raises -> Int:
    return corpus_numeric_base(step, batch, length, corpus_context())


def corpus_dict(value: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.dict:
        raise Error("TypeError: corpus metadata requires an ordinary dictionary")
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit all names before keyed metadata access
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: corpus metadata requires exact string keys")


def corpus_plain(value: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 1000:
        raise Error("ValueError: corpus metadata nesting limit exceeded")
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.bytes or cls is bi.int or cls is bi.float or cls is bi.bool:
        return value
    if cls is bi.list or cls is bi.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: capture passive range/configuration elements once
            py_call(result.append, corpus_plain(value[i], depth + 1))
        return py_call(bi.tuple, result) if cls is bi.tuple else result
    if cls is bi.dict:
        corpus_dict(value)
        var result = Python.dict()
        var keys = py_call(bi.list, value.keys())
        for i in range(len(keys)):  # driver-loop: capture passive manifest/checkpoint metadata once
            py_setitem(result, keys[i], corpus_plain(value[keys[i]], depth + 1))
        return result
    raise Error("TypeError: corpus metadata cannot invoke Python scalar/container protocols")


def corpus_json_node(mut doc: JsonDoc, value: PythonObject, depth: Int = 0) raises -> Int:
    """UNPACK: a passive metadata value (None, exact str, bool, int, float,
    list, tuple, dict with exact str keys) as a node of `doc`, captured once.
    A number keeps the interpreter's own spelling; nothing of the caller's is
    called."""
    if depth > 128:
        raise Error("ValueError: corpus metadata exceeds supported JSON nesting")
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None:
        return json_add_null(doc)
    if cls is bi.str:
        return json_add_string(doc, String(py=value))
    if cls is bi.bool:
        return json_add_bool(doc, Bool(py=value))
    if cls is bi.int:
        return json_add_number(doc, String(value))
    if cls is bi.float:
        var number = Float64(py=value)
        if number != number:
            return json_add_number(doc, String("NaN"))
        if number - number != 0.0:
            return json_add_number(doc, String("Infinity") if number > 0.0 else String("-Infinity"))
        return json_add_number(doc, String(value))
    if cls is bi.list or cls is bi.tuple:
        var node = json_add_array(doc)
        for i in range(len(value)):  # driver-loop: capture passive range/configuration elements once
            var element = corpus_json_node(doc, value[i], depth + 1)
            json_push(doc, node, element)
        return node
    if cls is not bi.dict:
        raise Error("TypeError: corpus metadata requires native JSON containers and scalar values")
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit all dictionary keys before native JSON item lookup
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: corpus JSON object keys must be exact strings")
    var node = json_add_object(doc)
    for i in range(len(keys)):  # driver-loop: capture passive manifest/checkpoint metadata once
        var member = corpus_json_node(doc, value[keys[i]], depth + 1)
        json_put(doc, node, String(py=keys[i]), member)
    return node


def corpus_fields(owner: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, py_type(owner))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: passive subclass ancestry without isinstance hooks
        admitted = admitted or ancestry[i] is cls
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, nk[].k___getattr__) or py_contains(namespace, nk[].k___del__):
            raise Error("TypeError: corpus records cannot invoke Python attribute hooks")
        for hook in [String("__getattribute__"), String("__setattr__")]:  # small-loop(native record access hooks)
            if py_contains(namespace, hook) and namespace[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: corpus records cannot invoke Python attribute hooks")
        for j in range(len(names)):  # small-loop(fixed corpus record field names)
            if py_contains(namespace, names[j]):
                raise Error("TypeError: corpus fields cannot be replaced by class descriptors")
    if not admitted:
        raise Error("TypeError: corpus operation requires its native-connected owner class")
    return _fields(owner)


def corpus_new(cls: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, cls)
    for i in range(len(ancestry)):  # small-loop(public record allocation finalizer admission)
        if py_contains(_class_fields(ancestry[i]), nk[].k___del__):
            raise Error("TypeError: native corpus records cannot invoke Python finalizers")
    return py_call(bi.object.__new__, cls)


def corpus_array(store: PythonObject, shape: PythonObject, context: PythonObject,
                 readonly: Bool = False) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    var slots = neural_array_slots(context)
    admit_native_buffer_exporter(store)
    var view = py_call(bi.memoryview, store)
    var count = 1
    var strides = Python.list()
    var stride = 4
    for i in range(len(shape)):  # small-loop(int32 output shape and stride words)
        var size = corpus_int(shape[len(shape) - i - 1], "shape", True)
        if size < 0 or (size != 0 and count > 2147483647 // size):
            raise Error("ValueError: corpus array exceeds native int32 indexing")
        count *= size
        py_call(strides.insert, 0, stride)
        stride *= size
    if Int(py=view.nbytes) != count * 4:
        raise Error("ValueError: corpus int32 storage differs from its shape")
    var result = corpus_new(py_item(context, nk[].k_array_class))
    var values = Python.dict(_store=store, _base=PythonObject(None), _pin=PythonObject(None),
        _mv=py_call(py_call(view.cast, "B").cast, "i"), _addr=PythonObject(native_buffer_address(store)),
        _readonly=PythonObject(readonly), shape=py_call(bi.tuple, shape), dtype=PythonObject("<i4"),
        ndim=PythonObject(len(shape)), size=PythonObject(count), nbytes=PythonObject(count * 4),
        itemsize=PythonObject(4), order=PythonObject("C"), strides=py_call(bi.tuple, strides))
    var names = py_call(bi.list, values.keys())
    for i in range(len(names)):  # small-loop(native Array slots; no Python constructor or setter)
        py_call(slots[names[i]].__set__, result, values[names[i]])
    return result


def corpus_empty(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var count = 1
    for i in range(len(shape)):  # small-loop(int32 output allocation dimensions)
        var n = corpus_int(shape[i], "shape", True)
        if n < 0 or (n != 0 and count > 2147483647 // n):
            raise Error("ValueError: corpus array exceeds native int32 indexing")
        count *= n
    return corpus_array(native_array_zeros("i", count), shape, context)


def corpus_mmap_hooks(cls: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, cls)
    var hooks = Python.dict()
    for name in ["__new__", "__init__", "__del__", "__buffer__", "__release_buffer__"]:  # small-loop(mmap allocation and lifetime slots)
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546); PACK stores None for an absent slot.
        var member = Optional[PythonObject]()
        for i in range(len(ancestry)):  # small-loop(native extension ancestry)
            var namespace = _class_fields(ancestry[i])
            if py_contains(namespace, name):
                member = namespace[name]
                break
        if member:
            var kind = py_type(member.value())
            if kind is not py_type(bi.object.__repr__) and kind is not py_type(bi.bytes.decode) and kind is not py_type(bi.object.__new__):
                raise Error("TypeError: mmap allocation and lifetime slots require native descriptors")
            py_setitem(hooks, name, member.value())
        else:
            py_setitem(hooks, name, PythonObject(None))
    return hooks


def corpus_mapped(path: PythonObject, size: Int, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    if size == 0:
        return corpus_empty(Python.tuple(0), context)
    var file = py_call(native_loaded_module("builtins").open, path, "rb")
    var mapping = PythonObject(None)
    try:
        var cls = py_item(context, nk[].k_mmap_class)
        var hooks = corpus_mmap_hooks(cls)
        var expected = py_item(context, nk[].k_mmap_hooks)
        var keys = py_call(native_loaded_module("builtins").list, hooks.keys())
        for i in range(len(keys)):  # small-loop(import-pinned native mmap lifecycle slots)
            if hooks[keys[i]] is not expected[keys[i]]:
                raise Error("TypeError: corpus mmap native lifecycle slots changed")
        mapping = py_call(cls, py_call(file.fileno), 0, access=1)
    finally:
        py_call(file.close)
    return corpus_array(mapping, Python.tuple(size // 4), context, True)


def corpus_ids_snapshot(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var captured = neural_array_snapshot(value, context)
    if String(captured.dtype) != "<i4" or String(captured.order) != "C" or Int(py=captured.ndim) != 1:
        raise Error("TypeError: corpus ids_all requires a native contiguous int32 vector")
    return captured
