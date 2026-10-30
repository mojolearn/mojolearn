# SPDX-License-Identifier: Apache-2.0
"""Native safetensors header admission, mapped reads and owned tensor buffers."""
from std.python import Python, PythonObject
from core.causal_checkpoint_impl import checkpoint_widen_halves
from std.memory import bitcast
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from bindings.native_json import native_json_parse, native_json_read, native_path, native_path_exists
from gemm.f16_widen import f16_bits_to_f32
from std.ffi import _Global
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_instance_item import native_instance_item
from bindings.native_array_storage import native_array_from_bytes
from bindings.native_buffer_address import native_buffer_address
from bindings.native_numpy_metadata import native_numpy_type
from bindings.neural_block_admission import neural_array_slots, neural_array_snapshot
from bindings.mode_metadata_native import _class_fields
from bindings.causal_model_config import model_record_fields, model_dict, model_plain, model_string, model_integer, model_sequence
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from bindings.linalg_ops_lowbit import _lin_from_f16, lin_connected_binding


struct CheckpointConnection(Defaultable, Movable):
    var context: PythonObject
    var entries: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)
        self.entries = PythonObject(None)


comptime CHECKPOINT_CONNECTION = _Global[StorageType=CheckpointConnection,
    name="MojolearnCheckpointConnection", init_fn=CheckpointConnection.__init__]


def _checkpoint_context() raises -> PythonObject:
    var context = CHECKPOINT_CONNECTION.get_or_create_ptr()[].context
    if context is None:
        raise Error("RuntimeError: checkpoints require import-time native connection")
    return context


def _checkpoint_entries() raises -> PythonObject:
    _ = _checkpoint_context()
    return CHECKPOINT_CONNECTION.get_or_create_ptr()[].entries


def checkpoint_connect(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    # Import-only connection. Product operations use these retained native
    # filesystem entries and never load modules or call Python file wrappers.
    model_dict(mods, context)
    var bi = mods.module("builtins")
    var connected = Python.dict()
    for name in ["array_class", "bf16_class", "info_class", "file_class", "checkpoint_class"]:  # small-loop(checkpoint owner classes)
        var cls = context[name]
        if py_type(cls) is not bi.type:
            raise Error("TypeError: checkpoint connection requires ordinary classes")
        py_setitem(connected, name, cls)
    py_setitem(connected, "linalg_context", context["linalg_context"])
    var entries = Python.dict()
    var posix = Python.import_module("posix")
    for name in ["open", "read", "close", "fstat", "listdir"]:  # small-loop(import-only native checkpoint filesystem entries)
        var entry = py_call(bi.getattr, posix, name)
        if py_type(entry) is not py_type(bi.len):
            raise Error("TypeError: checkpoint filesystem entry must be native")
        py_setitem(entries, name, entry)
    py_setitem(entries, "O_RDONLY", PythonObject(model_integer(mods, posix.O_RDONLY)))
    var mmap = Python.import_module("mmap")
    var cls = mmap.mmap
    if py_type(cls) is not bi.type:
        raise Error("TypeError: checkpoint mmap connection requires its native type")
    var namespace = _class_fields(cls)
    var close = namespace["close"]
    if py_type(close) is not mods.module("types").MethodDescriptorType or close.__objclass__ is not cls:
        raise Error("TypeError: checkpoint mmap close must be a native method")
    py_setitem(entries, "mmap_class", cls)
    py_setitem(entries, "mmap_close", close)
    py_setitem(entries, "ACCESS_READ", PythonObject(model_integer(mods, mmap.ACCESS_READ)))
    _ = neural_array_slots(connected)
    var state = CHECKPOINT_CONNECTION.get_or_create_ptr()
    if state[].context is not None:
        raise Error("TypeError: checkpoints are already connected")
    state[].context = connected
    state[].entries = entries
    return PythonObject(None)


def _file_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, value, _checkpoint_context()["file_class"],
        Python.tuple("path", "_fh", "_mm", "_infos", "data_start", "data_size", "metadata"))


def _checkpoint_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, value, _checkpoint_context()["checkpoint_class"],
        Python.tuple("root", "_paths", "_weight_map", "_open", "_infos"))


def _info_slots(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    var cls = context["info_class"]
    var bi = mods.module("builtins")
    if py_type(value) is not cls:
        raise Error("TypeError: tensor metadata requires its connected TensorInfo class")
    var namespace = _class_fields(cls)
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not bi.object:
        raise Error("TypeError: TensorInfo class ancestry changed")
    for name in ["__getattribute__", "__getattr__", "__setattr__"]:  # small-loop(TensorInfo access hook names)
        if py_contains(namespace, name):
            raise Error("TypeError: TensorInfo cannot invoke Python attribute hooks")
    var slots = Python.dict()
    var member = mods.module("types").MemberDescriptorType
    for name in ["name", "dtype", "shape", "begin", "end", "file"]:  # small-loop(TensorInfo native member schema)
        var slot = py_get(namespace, name)
        if py_type(slot) is not member or slot.__objclass__ is not cls or model_string(mods, slot.__name__) != name:
            raise Error("TypeError: TensorInfo storage descriptors changed")
        py_setitem(slots, name, slot)
    return slots


def _info_validate(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    model_sequence(mods, fields)
    if len(fields) != 6:
        raise Error("TypeError: TensorInfo requires six native metadata fields")
    var name = model_string(mods, fields[0])
    var dtype = model_string(mods, fields[1])
    var shape = Python.list()
    var count = 1
    var input_shape = fields[2]
    model_sequence(mods, input_shape)
    for j in range(len(input_shape)):  # small-loop(tensor shape metadata)
        var size = model_integer(mods, input_shape[j])
        if size < 0 or (size != 0 and count > 9223372036854775807 // size):
            raise Error("TensorInfo shape must hold non-negative dimensions")
        count *= size
        _ = py_call(shape.append, size)
    var begin = model_integer(mods, fields[3])
    var end = model_integer(mods, fields[4])
    if begin < 0 or end < begin:
        raise Error("TensorInfo offsets must satisfy 0 <= begin <= end")
    var itemsize = 8 if dtype == "I64" else (4 if dtype == "F32" or dtype == "I32" else (2 if dtype == "BF16" or dtype == "F16" else 0))
    if itemsize != 0 and (count > 9223372036854775807 // itemsize or count * itemsize != end - begin):
        raise Error("TensorInfo bytes differ from dtype/shape")
    var file = PythonObject(None) if fields[5] is None else native_path(fields[5])
    return Python.dict(name=PythonObject(name), dtype=PythonObject(dtype), shape=py_call(bi.tuple, shape),
        begin=PythonObject(begin), end=PythonObject(end), file=file, count=PythonObject(count))


def _info_snapshot(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var slots = _info_slots(mods, value)
    var fields = Python.list()
    for name in ["name", "dtype", "shape", "begin", "end", "file"]:  # small-loop(capture each TensorInfo slot once)
        _ = py_call(fields.append, py_call(slots[name].__get__, value, _checkpoint_context()["info_class"]))
    return _info_validate(mods, fields)


def _info_map(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    model_dict(mods, value)
    var names = py_call(mods.module("builtins").list, value.keys())
    for i in range(len(names)):  # driver-loop: reject foreign mutable tensor-info records before lookup
        _ = _info_slots(mods, native_instance_item(value, names[i]))
    return value


def _checkpoint_pinned_bytes(pin: AdmittedBorrowedBuffer) raises -> PythonObject:
    var count = pin.layout.value().count * pin.layout.value().itemsize
    var bytes = List[UInt8](length=count, fill=0)
    pin.layout.value().copy_to(Int(bytes.unsafe_ptr()))
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(bytes.unsafe_ptr(), count)
    _ = bytes  # The C API copies from this native owner before it can be released.
    _ = pin
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _checkpoint_bytes(value: PythonObject) raises -> PythonObject:
    admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(value)
    var raw = PythonObject(None)
    try:
        raw = _checkpoint_pinned_bytes(pin)
    except error:
        pin.release()
        raise error
    pin.release()
    return raw


def _checkpoint_read_bytes(mut mods: NativeModules, fd: PythonObject, count: Int) raises -> PythonObject:
    var pieces = Python.list()
    var total = 0
    var read = _checkpoint_entries()["read"]
    while total < count:  # driver-loop: consume a bounded native file header including short reads
        var part = py_call(read, fd, count - total)
        mods.released()
        if len(part) == 0:
            break
        total += len(part)
        _ = py_call(pieces.append, part)
    var empty = py_call(mods.module("builtins").bytes)
    return py_call(empty.join, pieces)


def checkpoint_array(mut mods: NativeModules, raw: PythonObject, shape: PythonObject, code: String, array_class: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    if array_class is not context["array_class"]:
        raise Error("TypeError: checkpoint output requires its connected Array class")
    var slots = neural_array_slots(context)
    var bi = mods.module("builtins")
    model_sequence(mods, shape)
    var captured_shape = model_plain(mods, shape)
    var bytes = _checkpoint_bytes(raw)
    var store = native_array_from_bytes(code, bytes)
    var result = py_call(bi.object.__new__, array_class)
    var itemsize = Int(py=store.itemsize)
    var count = 1
    var strides = Python.list()
    var stride = itemsize
    for i in range(len(captured_shape)):  # small-loop(tensor shape and row-major stride metadata)
        var size = model_integer(mods, captured_shape[len(captured_shape) - i - 1])
        if size < 0 or (size != 0 and stride > 9223372036854775807 // size):
            raise Error("safetensors: tensor shape exceeds native indexing")
        count *= size
        _ = py_call(strides.insert, 0, stride)
        stride *= size
    if count != len(store):
        raise Error("safetensors: tensor shape does not match stored bytes")
    var fields = Python.dict(_store=store, _mv=py_call(bi.memoryview, store), _base=PythonObject(None), _pin=PythonObject(None),
        _addr=PythonObject(native_buffer_address(store)), _readonly=PythonObject(False), shape=py_call(bi.tuple, captured_shape),
        dtype=PythonObject("<f4" if code == "f" else ("<u2" if code == "H" else ("<i8" if code == "q" else ("<i1" if code == "b" else "<i4")))),
        ndim=PythonObject(len(captured_shape)), size=PythonObject(count), nbytes=PythonObject(count * itemsize),
        itemsize=PythonObject(itemsize), order=PythonObject("C"), strides=py_call(bi.tuple, strides))
    mods.released()
    var names = py_call(bi.list, fields.keys())
    for i in range(len(names)):  # small-loop(publish through admitted native Array member descriptors)
        _ = py_call(slots[names[i]].__set__, result, fields[names[i]])
    return result


def checkpoint_array(raw: PythonObject, shape: PythonObject, code: String, array_class: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return checkpoint_array(mods, raw, shape, code, array_class)


def tensor_info_init_binding(mut mods: NativeModules, value: PythonObject, fields: PythonObject) raises -> PythonObject:
    var slots = _info_slots(mods, value)
    var captured = _info_validate(mods, fields)
    for name in ["name", "dtype", "shape", "begin", "end", "file"]:  # small-loop(publish admitted native TensorInfo fields)
        _ = py_call(slots[name].__set__, value, captured[name])
    return PythonObject(None)


def tensor_info_size_binding(mut mods: NativeModules, value: PythonObject, bytes: PythonObject) raises -> PythonObject:
    if py_type(bytes) is not mods.module("builtins").bool:
        raise Error("TypeError: tensor size operation requires an exact boolean")
    var fields = _info_snapshot(mods, value)
    return fields["end"] - fields["begin"] if Bool(py=bytes) else fields["count"]


def safetensors_init_binding(mut mods: NativeModules, value: PythonObject, input_path: PythonObject, supplied_context: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    var fields = _file_fields(mods, value)
    var entries = _checkpoint_entries()
    var bi = mods.module("builtins")
    var path = native_path(input_path)
    if not native_path_exists(path):
        mods.released()
        raise Error("FileNotFoundError: safetensors file does not exist")
    mods.released()
    var fd = py_call(entries["open"], path, entries["O_RDONLY"])
    mods.released()
    var raw = PythonObject(None)
    var total = 0
    var n = 0
    try:
        var head = _checkpoint_read_bytes(mods, fd, 8)
        if len(head) != 8:
            raise Error("safetensors: file is shorter than its 8-byte header length")
        var length = UInt64(0)
        for i in range(8):  # small-loop(the little-endian 8-byte header length)
            length |= UInt64(Int(py=head[i])) << UInt64(8 * i)
        if length > 100 * 1024 * 1024:
            raise Error("safetensors: header length exceeds 104857600")
        n = Int(length)
        raw = _checkpoint_read_bytes(mods, fd, n)
        if len(raw) != n:
            raise Error("safetensors: header is truncated")
        total = Int(py=py_call(entries["fstat"], fd).st_size)
        mods.released()
    except error:
        mods.released()
        _ = py_call(entries["close"], fd)
        mods.released()
        raise error
    _ = py_call(entries["close"], fd)
    mods.released()
    var header = native_json_parse(raw)
    model_dict(mods, header)
    var metadata = py_call(header.pop, "__metadata__", PythonObject(None))
    var infos = Python.dict()
    var names = py_call(bi.list, header.keys())
    for i in range(len(names)):  # driver-loop: validate complete native checkpoint header before publishing it
        var name = names[i]
        var entry = header[name]
        model_dict(mods, entry)
        if not py_contains(entry, "dtype") or not py_contains(entry, "shape") or not py_contains(entry, "data_offsets"):
            raise Error("safetensors: tensor " + String(name) + " lacks dtype/shape/data_offsets")
        var offsets = entry["data_offsets"]
        model_sequence(mods, offsets)
        if len(offsets) != 2:
            raise Error("safetensors: data_offsets requires a pair")
        var begin = model_integer(mods, offsets[0])
        var end = model_integer(mods, offsets[1])
        if begin < 0 or end < begin or end > total - 8 - n:
            raise Error("safetensors: tensor " + String(name) + " offsets fall outside the data section")
        var info = py_call(bi.object.__new__, context["info_class"])
        _ = tensor_info_init_binding(mods, info, Python.tuple(name, entry["dtype"], entry["shape"], begin, end, path))
        py_setitem(infos, name, info)
    py_setitem(fields, "path", path)
    py_setitem(fields, "_fh", PythonObject(None))
    py_setitem(fields, "_mm", PythonObject(None))
    py_setitem(fields, "_infos", infos)
    py_setitem(fields, "data_start", PythonObject(8 + n))
    py_setitem(fields, "data_size", PythonObject(total - 8 - n))
    py_setitem(fields, "metadata", metadata)
    return PythonObject(None)


def _safetensors_map(fields: PythonObject) raises -> PythonObject:
    var entries = _checkpoint_entries()
    var current = native_instance_item(fields, PythonObject("_mm"))
    if current is not None:
        if py_type(current) is not entries["mmap_class"]:
            raise Error("TypeError: safetensors mapping requires its retained native mmap")
        return current
    var path = native_path(native_instance_item(fields, PythonObject("path")))
    var fd = py_call(entries["open"], path, entries["O_RDONLY"])
    var mapped = PythonObject(None)
    try:
        mapped = py_call(entries["mmap_class"], fd, 0, access=entries["ACCESS_READ"])
    except error:
        _ = py_call(entries["close"], fd)
        raise error
    # mmap owns the mapping after creation; no Python file owner remains.
    _ = py_call(entries["close"], fd)
    py_setitem(fields, "_mm", mapped)
    return mapped


def safetensors_map_binding(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var windowed_result2 = _safetensors_map(_file_fields(mods, value))
    mods.released()
    return windowed_result2^


def safetensors_close_binding(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var fields = _file_fields(mods, value)
    var current = native_instance_item(fields, PythonObject("_mm"))
    if current is not None:
        var entries = _checkpoint_entries()
        if py_type(current) is not entries["mmap_class"]:
            raise Error("TypeError: safetensors close requires its retained native mmap")
        _ = py_call(entries["mmap_close"], current)
        mods.released()
        py_setitem(fields, "_mm", PythonObject(None))
    if native_instance_item(fields, PythonObject("_fh")) is not None:
        raise Error("TypeError: safetensors does not accept Python file owners")
    return PythonObject(None)


def _safetensors_info(mut mods: NativeModules, fields: PythonObject, name: PythonObject) raises -> PythonObject:
    _ = model_string(mods, name)
    var infos = _info_map(mods, native_instance_item(fields, PythonObject("_infos")))
    if not py_contains(infos, name):
        raise Error("KeyError: safetensors holds no tensor " + String(name))
    return native_instance_item(infos, name)


def safetensors_info_binding(mut mods: NativeModules, value: PythonObject, name: PythonObject) raises -> PythonObject:
    var info = _safetensors_info(mods, _file_fields(mods, value), name)
    _ = _info_snapshot(mods, info)
    return info


def _safetensors_read(mut mods: NativeModules, fields: PythonObject, info: PythonObject, name: PythonObject, bf16: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    var format = model_string(mods, bf16)
    if format != "widen" and format != "bits":
        raise Error("safetensors: bf16 must be widen or bits")
    var dtype = String(info["dtype"])
    if dtype != "F32" and dtype != "F16" and dtype != "BF16" and dtype != "I32" and dtype != "I64":
        raise Error("TypeError: tensor " + String(name) + " dtype " + dtype + " refuses by name (the reader admits F32, F16, BF16, I32 and I64)")
    var begin = model_integer(mods, info["begin"])
    var end = model_integer(mods, info["end"])
    var data_start = model_integer(mods, native_instance_item(fields, PythonObject("data_start")))
    var data_size = model_integer(mods, native_instance_item(fields, PythonObject("data_size")))
    if data_start < 0 or data_size < 0 or end > data_size or data_start > 9223372036854775807 - end:
        raise Error("ValueError: safetensors data offsets exceed the retained file")
    var mapped = _safetensors_map(fields)
    mods.released()
    if data_start + end > len(mapped):
        raise Error("ValueError: safetensors file changed after header admission")
    var bi = mods.module("builtins")
    var view = py_call(bi.memoryview, mapped)
    var raw = view[py_call(bi.slice, data_start + begin, data_start + end)]
    var code = "f" if dtype == "F32" else ("q" if dtype == "I64" else ("i" if dtype == "I32" else "H"))
    var owned = checkpoint_array(mods, raw, info["shape"], code, context["array_class"])
    _ = py_call(raw.release)
    _ = py_call(view.release)
    if dtype == "F16" or (dtype == "BF16" and format == "widen"):
        var count = Int(py=info["count"])
        var out = checkpoint_array(mods, py_call(bi.bytes, 4 * count), info["shape"], "f", context["array_class"])
        if count != 0:
            var source = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(py=owned._addr))
            var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
            checkpoint_widen_halves(source, dest, count, dtype)
            _ = owned^
        return out
    if dtype == "BF16" and format == "bits" and len(info["shape"]) == 2:
        var packed = py_call(bi.object.__new__, context["bf16_class"])
        var packed_fields = model_record_fields(mods, packed, context["bf16_class"], Python.tuple("bits", "shape"))
        py_setitem(packed_fields, "bits", owned)
        py_setitem(packed_fields, "shape", info["shape"])
        return packed
    return owned


def safetensors_read_binding(mut mods: NativeModules, value: PythonObject, name: PythonObject, bf16: PythonObject, supplied_context: PythonObject) raises -> PythonObject:
    var fields = _file_fields(mods, value)
    var info = _info_snapshot(mods, _safetensors_info(mods, fields, name))
    return _safetensors_read(mods, fields, info, name, bf16)


def checkpoint_init_binding(mut mods: NativeModules, value: PythonObject, files: PythonObject, weight_map: PythonObject, root: PythonObject) raises -> PythonObject:
    var fields = _checkpoint_fields(mods, value)
    model_sequence(mods, files)
    var paths = Python.list()
    for i in range(len(files)):  # driver-loop: capture native shard paths without path protocols
        _ = py_call(paths.append, native_path(files[i]))
    model_dict(mods, weight_map)
    var captured_map = Python.dict()
    var names = py_call(mods.module("builtins").list, weight_map.keys())
    for i in range(len(names)):  # driver-loop: capture native checkpoint weight-to-shard index
        py_setitem(captured_map, names[i], native_path(native_instance_item(weight_map, names[i])))
    py_setitem(fields, "root", native_path(root))
    py_setitem(fields, "_paths", paths)
    py_setitem(fields, "_weight_map", captured_map)
    py_setitem(fields, "_open", Python.dict())
    py_setitem(fields, "_infos", PythonObject(None))
    return PythonObject(None)


def checkpoint_open_binding(mut mods: NativeModules, cls: PythonObject, input_path: PythonObject, supplied_context: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    if cls is not context["checkpoint_class"]:
        raise Error("TypeError: checkpoint factory requires its connected class")
    var bi = mods.module("builtins")
    var path = native_path(input_path)
    var files = Python.list()
    var weight_map = Python.dict()
    var root = path
    if native_path_exists(path, True):
        mods.released()
        var index = py_call(path.rstrip, "/") + PythonObject("/model.safetensors.index.json")
        var single = py_call(path.rstrip, "/") + PythonObject("/model.safetensors")
        if native_path_exists(index):
            mods.released()
            var data = native_json_read(index)
            mods.released()
            model_dict(mods, data)
            var wm = py_get(data, "weight_map")
            model_dict(mods, wm)
            if len(wm) == 0:
                raise Error("safetensors: index has no weight_map")
            var names = py_call(bi.list, wm.keys())
            for i in range(len(names)):  # driver-loop: resolve each admitted checkpoint index filename
                var filename = wm[names[i]]
                _ = model_string(mods, filename)
                if not Bool(py=py_call(filename.startswith, "/")):
                    filename = py_call(path.rstrip, "/") + PythonObject("/") + filename
                if not native_path_exists(filename):
                    mods.released()
                    raise Error("FileNotFoundError: checkpoint index names shard that does not exist: " + String(filename))
                mods.released()
                py_setitem(weight_map, names[i], filename)
                if not py_contains(files, filename):
                    _ = py_call(files.append, filename)
            _ = py_call(files.sort)
        elif native_path_exists(single):
            mods.released()
            mods.released()
            _ = py_call(files.append, single)
        else:
            mods.released()
            mods.released()
            var entries = py_call(_checkpoint_entries()["listdir"], path)
            mods.released()
            for i in range(len(entries)):  # driver-loop: find an unambiguous native checkpoint file
                if Bool(py=py_call_typed(entries[i].endswith, ".safetensors")):
                    _ = py_call(files.append, py_call(path.rstrip, "/") + PythonObject("/") + entries[i])
            if len(files) != 1:
                raise Error("FileNotFoundError: checkpoint holds neither model.safetensors nor model.safetensors.index.json (shard files alone are not an index)")
        mods.released()
        mods.released()
    elif native_path_exists(path):
        mods.released()
        mods.released()
        _ = py_call(files.append, path)
        var pieces = py_call(path.rsplit, "/", 1)
        root = pieces[0] if len(pieces) == 2 else PythonObject(".")
        if len(root) == 0:
            root = PythonObject("/")
    else:
        mods.released()
        mods.released()
        raise Error("FileNotFoundError: checkpoint path does not exist")
    mods.released()
    mods.released()
    var value = py_call(bi.object.__new__, cls)
    _ = checkpoint_init_binding(mods, value, files, weight_map, root)
    return value


def _checkpoint_file(mut mods: NativeModules, fields: PythonObject, path: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    var filename = native_path(path)
    var opened = native_instance_item(fields, PythonObject("_open"))
    model_dict(mods, opened)
    var file = py_get(opened, filename)
    if file is None:
        file = py_call(mods.module("builtins").object.__new__, context["file_class"])
        _ = safetensors_init_binding(mods, file, filename, context)
        py_setitem(opened, filename, file)
    else:
        _ = _file_fields(mods, file)
    return file


def checkpoint_file_binding(mut mods: NativeModules, value: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    return _checkpoint_file(mods, _checkpoint_fields(mods, value), path)


def _checkpoint_index(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    var infos = native_instance_item(fields, PythonObject("_infos"))
    if infos is not None:
        return _info_map(mods, infos)
    var bi = mods.module("builtins")
    infos = Python.dict()
    var paths = model_plain(mods, native_instance_item(fields, PythonObject("_paths")))
    model_sequence(mods, paths)
    var weight_map = model_plain(mods, native_instance_item(fields, PythonObject("_weight_map")))
    model_dict(mods, weight_map)
    for i in range(len(paths)):  # driver-loop: load and index each captured checkpoint shard once
        var file = _checkpoint_file(mods, fields, paths[i])
        var file_fields = _file_fields(mods, file)
        var file_infos = _info_map(mods, native_instance_item(file_fields, PythonObject("_infos")))
        var names = py_call(bi.list, file_infos.keys())
        for j in range(len(names)):  # driver-loop: merge admitted native tensor-info records
            var name = names[j]
            if py_contains(infos, name):
                raise Error("safetensors: tensor " + String(name) + " appears in both shards")
            py_setitem(infos, name, file_infos[name])
    var names = py_call(bi.list, weight_map.keys())
    for i in range(len(names)):  # driver-loop: validate each captured weight-to-shard index reference
        var name = names[i]
        var path = native_path(weight_map[name])
        if not py_contains(infos, name):
            raise Error("safetensors: index maps " + String(name) + " to a shard that does not hold it")
        var info = _info_snapshot(mods, infos[name])
        if not Bool(py=info["file"] == path):
            raise Error("safetensors: index maps " + String(name) + " to a shard that does not hold it")
    py_setitem(fields, "_infos", infos)
    return infos


def checkpoint_index_binding(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    return _checkpoint_index(mods, _checkpoint_fields(mods, value))


def checkpoint_access_binding(mut mods: NativeModules, value: PythonObject, operation: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    var infos = _checkpoint_index(mods, _checkpoint_fields(mods, value))
    var op = model_integer(mods, operation)
    if op == 0:
        return py_call(mods.module("builtins").list, infos.keys())
    _ = model_string(mods, name)
    if op == 1:
        if not py_contains(infos, name):
            raise Error("KeyError: checkpoint holds no tensor " + String(name))
        var info = infos[name]
        _ = _info_snapshot(mods, info)
        return info
    if op != 2:
        raise Error("ValueError: unknown checkpoint access operation")
    return py_contains(infos, name)


def checkpoint_read_binding(mut mods: NativeModules, value: PythonObject, name: PythonObject, bf16: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _checkpoint_fields(mods, value)
    var infos = _checkpoint_index(mods, fields)
    _ = model_string(mods, name)
    if not py_contains(infos, name):
        raise Error("KeyError: checkpoint holds no tensor " + String(name))
    var retained_info = infos[name]
    var info = _info_snapshot(mods, retained_info)
    var file = _checkpoint_file(mods, fields, info["file"])
    var file_fields = _file_fields(mods, file)
    if _safetensors_info(mods, file_fields, name) is not retained_info:
        raise Error("ValueError: checkpoint tensor metadata differs from its retained shard index")
    return _safetensors_read(mods, file_fields, info, name, bf16)


def checkpoint_close_binding(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var fields = _checkpoint_fields(mods, value)
    var opened = native_instance_item(fields, PythonObject("_open"))
    model_dict(mods, opened)
    var files = py_call(mods.module("builtins").list, opened.values())
    # Admit all owners before any resource release.
    for i in range(len(files)):  # driver-loop: validate captured open checkpoint shards
        _ = _file_fields(mods, files[i])
    for i in range(len(files)):  # driver-loop: release retained native checkpoint mappings
        _ = safetensors_close_binding(mods, files[i])
    py_setitem(fields, "_open", Python.dict())
    return PythonObject(None)


def checkpoint_repr(mut mods: NativeModules, value: PythonObject, kind: PythonObject) raises -> PythonObject:
    var tag = model_integer(mods, kind)
    var bi = mods.module("builtins")
    if tag == 0:
        var info = _info_snapshot(mods, value)
        return PythonObject("TensorInfo(" + String(py_call(bi.repr, info["name"])) + ", " + String(info["dtype"]) + ", shape=" + String(py_call(bi.repr, info["shape"])) + ")")
    if tag == 1:
        var fields = _file_fields(mods, value)
        var path = native_path(native_instance_item(fields, PythonObject("path")))
        var infos = _info_map(mods, native_instance_item(fields, PythonObject("_infos")))
        return PythonObject("SafetensorsFile(" + String(py_call(bi.repr, path)) + ", tensors=" + String(len(infos)) + ")")
    var fields = _checkpoint_fields(mods, value)
    var root = native_path(native_instance_item(fields, PythonObject("root")))
    var paths = model_plain(mods, native_instance_item(fields, PythonObject("_paths")))
    model_sequence(mods, paths)
    return PythonObject("Checkpoint(" + String(py_call(bi.repr, root)) + ", shards=" + String(len(paths)) + ")")


def checkpoint_widen_f16(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var context = _checkpoint_context()
    var cls = context["array_class"]
    var slots = neural_array_slots(context)
    var captured = PythonObject(None)
    if py_type(value) is cls:
        var names = py_call(mods.module("builtins").list, slots.keys())
        captured = py_call(mods.module("builtins").object.__new__, cls)
        for i in range(len(names)):  # small-loop(capture caller Array slots once before dtype normalization)
            _ = py_call(slots[names[i]].__set__, captured, py_call(slots[names[i]].__get__, value, cls))
    else:
        # The pre-migration public operation also accepted ordinary NumPy
        # float16/uint16 arrays. Capture their native buffer once and copy raw
        # bits in C order; no ndarray conversion, Python hook, or arithmetic
        # may execute while preparing the existing widening kernel's input.
        if py_type(value) is not native_numpy_type("ndarray", context["linalg_context"]):
            raise Error("TypeError: widen_f16 requires a native Array or exact ndarray of float16 or uint16 bits")
        admit_native_buffer_exporter(value)
        var pin = AdmittedBorrowedBuffer(value)
        var code = pin.format
        if (pin.layout.value().itemsize != 2 or
                (code != "e" and code != "H" and code != "<e" and code != "<H"
                 and code != "=e" and code != "=H" and code != "@e" and code != "@H")):
            raise Error("TypeError: widen_f16 requires little-endian float16 or uint16 bit storage")
        var shape = Python.list()
        for axis in range(len(pin.layout.value().shape)):  # small-loop(captured native buffer shape)
            _ = py_call(shape.append, PythonObject(pin.layout.value().shape[axis]))
        var raw = _checkpoint_pinned_bytes(pin)
        captured = checkpoint_array(mods, raw, shape, "H", cls)
        pin.release()
    var dtype = model_string(mods, captured.dtype)
    if dtype != "<f2" and dtype != "<u2":
        raise Error("TypeError: widen_f16 requires float16 or uint16 bit storage")
    # Float16 and uint16 have the same two-byte storage. Only this private
    # captured descriptor changes; shared admission validates every remaining
    # layout word and pins its native exporter before the original provider.
    _ = py_call(slots["dtype"].__set__, captured, PythonObject("<u2"))
    var source = neural_array_snapshot(captured, context)
    if Int(py=source.size) == 0:
        return checkpoint_array(mods, py_call(mods.module("builtins").bytes), source.shape, "f", cls)
    var linalg = context["linalg_context"]
    var binding = lin_connected_binding(linalg)
    mods.released()
    var result = _lin_from_f16(Python.tuple(binding, source), linalg)
    mods.released()
    _ = source
    _ = captured
    _ = value
    return result


def checkpoint_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = tensor_info_init_binding(mods, args[0], args[1])
        elif operation == 1:
            result = tensor_info_size_binding(mods, args[0], args[1])
        elif operation == 2:
            result = safetensors_init_binding(mods, args[0], args[1], args[2])
        elif operation == 3:
            result = safetensors_info_binding(mods, args[0], args[1])
        elif operation == 4:
            result = safetensors_map_binding(mods, args[0])
        elif operation == 5:
            result = safetensors_read_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 6:
            result = safetensors_close_binding(mods, args[0])
        elif operation == 7:
            result = checkpoint_init_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 8:
            result = checkpoint_open_binding(mods, args[0], args[1], args[2])
        elif operation == 9:
            result = checkpoint_file_binding(mods, args[0], args[1], args[2])
        elif operation == 10:
            result = checkpoint_index_binding(mods, args[0], args[1])
        elif operation == 11:
            result = checkpoint_access_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 12:
            result = checkpoint_read_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 13:
            result = checkpoint_close_binding(mods, args[0])
        elif operation == 14:
            if py_type(args[0]) is _checkpoint_context()["file_class"]:
                _ = _file_fields(mods, args[0])
            else:
                _ = _checkpoint_fields(mods, args[0])
            result = args[0]
        elif operation == 15:
            var fields = _file_fields(mods, args[0])
            var infos = _info_map(mods, native_instance_item(fields, PythonObject("_infos")))
            result = py_call(mods.module("builtins").list, infos.keys())
        elif operation == 16:
            var fields = _file_fields(mods, args[0])
            var infos = _info_map(mods, native_instance_item(fields, PythonObject("_infos")))
            _ = model_string(mods, args[1])
            result = py_contains(infos, args[1])
        elif operation == 17:
            result = checkpoint_connect(mods, args[0])
        elif operation == 18:
            result = checkpoint_repr(mods, args[0], args[1])
        elif operation == 19:
            result = checkpoint_widen_f16(mods, args[0])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
