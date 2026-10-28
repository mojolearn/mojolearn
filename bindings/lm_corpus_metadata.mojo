# SPDX-License-Identifier: Apache-2.0
"""Native corpus records, passive metadata capture and int32 array ownership."""
from std.python import Python, PythonObject
from std.ffi import _Global
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.neural_block_admission import neural_array_slots, neural_array_snapshot, neural_native_scalar
from bindings.native_array_storage import native_array_zeros
from bindings.native_buffer_address import native_buffer_address
from bindings.lm_corpus_files import corpus_files_initialize
from bindings.lm_corpus_scalars import corpus_numeric_base
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, connect_native_mmap_exporter


struct CorpusConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime CORPUS_CONNECTION = _Global[StorageType=CorpusConnection,
    name="MojolearnCorpusPublicConnection", init_fn=CorpusConnection.__init__]


def corpus_context() raises -> PythonObject:
    var context = CORPUS_CONNECTION.get_or_create_ptr()[].context
    if context is None:
        raise Error("RuntimeError: corpus operations require import-time native connection")
    return context


def corpus_connect(args: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(args) is not bi.tuple or len(args) != 6:
        raise Error("TypeError: corpus connection requires its native classes")
    var prior = CORPUS_CONNECTION.get_or_create_ptr()[].context
    if prior is not None:
        var prior_names = Python.tuple("corpus_class", "batches_class", "iterator_class", "array_class", "mmap_class")
        for i in range(5):  # small-loop(import connection cannot replace retained owner identities)
            if args[i] is not prior[prior_names[i]]:
                raise Error("TypeError: corpus import connection cannot be replaced")
        if args[5] is not prior["scalar_connection"]:
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
    py_setitem(admission, "native_module_type", py_type(bi))
    py_setitem(context, "scalar_connection", args[5])
    py_setitem(context, "admission_context", admission)
    py_setitem(context, "array_slots", neural_array_slots(context))
    py_setitem(context, "batch_ids_method", _class_fields(context["batches_class"])["ids"])
    py_setitem(context, "mmap_hooks", corpus_mmap_hooks(context["mmap_class"]))
    connect_native_mmap_exporter(context["mmap_class"])
    corpus_files_initialize()
    CORPUS_CONNECTION.get_or_create_ptr()[].context = context
    return PythonObject(None)


def corpus_int(value: PythonObject, name: String, strict: Bool = False) raises -> Int:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not bi.int:
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


def corpus_fields(owner: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(owner))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: passive subclass ancestry without isinstance hooks
        admitted = admitted or ancestry[i] is cls
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "__getattr__") or py_contains(namespace, "__del__"):
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
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # small-loop(public record allocation finalizer admission)
        if py_contains(_class_fields(ancestry[i]), "__del__"):
            raise Error("TypeError: native corpus records cannot invoke Python finalizers")
    return py_call(native_loaded_module("builtins").object.__new__, cls)


def corpus_array(store: PythonObject, shape: PythonObject, context: PythonObject,
                 readonly: Bool = False) raises -> PythonObject:
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
    var result = corpus_new(context["array_class"])
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
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    var hooks = Python.dict()
    for name in ["__new__", "__init__", "__del__", "__buffer__", "__release_buffer__"]:  # small-loop(mmap allocation and lifetime slots)
        var member = PythonObject(None)
        for i in range(len(ancestry)):  # small-loop(native extension ancestry)
            var namespace = _class_fields(ancestry[i])
            if py_contains(namespace, name):
                member = namespace[name]
                break
        if member is not None:
            var kind = py_type(member)
            if kind is not py_type(bi.object.__repr__) and kind is not py_type(bi.bytes.decode) and kind is not py_type(bi.object.__new__):
                raise Error("TypeError: mmap allocation and lifetime slots require native descriptors")
        py_setitem(hooks, name, member)
    return hooks


def corpus_mapped(path: PythonObject, size: Int, context: PythonObject) raises -> PythonObject:
    if size == 0:
        return corpus_empty(Python.tuple(0), context)
    var file = py_call(native_loaded_module("builtins").open, path, "rb")
    var mapping = PythonObject(None)
    try:
        var cls = context["mmap_class"]
        var hooks = corpus_mmap_hooks(cls)
        var expected = context["mmap_hooks"]
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
