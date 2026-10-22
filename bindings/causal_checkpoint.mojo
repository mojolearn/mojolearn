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


def checkpoint_array(raw: PythonObject, shape: PythonObject, code: String, array_class: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var store = py_call(Python.import_module("array").array, code)
    _ = py_call(store.frombytes, raw)
    var result = py_call(bi.object.__new__, array_class)
    var itemsize = Int(py=store.itemsize)
    var count = 1
    var strides = bi.list()
    var stride = itemsize
    for i in range(len(shape)):  # small-loop(tensor shape dimension words)
        var size = Int(py=shape[len(shape) - i - 1])
        if size < 0:
            raise Error("safetensors: negative shape dimension")
        count *= size
        _ = py_call(strides.insert, 0, stride)
        stride *= size
    if count != len(store):
        raise Error("safetensors: tensor shape does not match stored bytes")
    py_setattr(result, "_store", store)
    py_setattr(result, "_mv", py_call(bi.memoryview, store))
    py_setattr(result, "_base", PythonObject(None))
    py_setattr(result, "_pin", PythonObject(None))
    py_setattr(result, "_addr", store.buffer_info()[0])
    py_setattr(result, "_readonly", PythonObject(False))
    py_setattr(result, "shape", py_call(bi.tuple, shape))
    py_setattr(result, "dtype", PythonObject("<f4" if code == "f" else ("<u2" if code == "H" else ("<i8" if code == "q" else ("<i1" if code == "b" else "<i4")))))
    py_setattr(result, "ndim", PythonObject(len(shape)))
    py_setattr(result, "size", PythonObject(count))
    py_setattr(result, "nbytes", PythonObject(count * itemsize))
    py_setattr(result, "itemsize", PythonObject(itemsize))
    py_setattr(result, "order", PythonObject("C"))
    py_setattr(result, "strides", py_call(bi.tuple, strides))
    return result


def tensor_info_init_binding(value: PythonObject, fields: PythonObject) raises -> PythonObject:
    # A TensorInfo built by hand admits exactly what the header parser
    # (safetensors_init_binding) admits: a str name and dtype, non-negative
    # dimensions, ordered offsets and, for a dtype the reader knows, a byte
    # span equal to dtype x shape.
    var bi = Python.import_module("builtins")
    if py_type(fields[0]) is not bi.str or py_type(fields[1]) is not bi.str:
        raise Error("TypeError: TensorInfo name and dtype must be str")
    var dtype = String(fields[1])
    var shape = bi.list()
    var count = 1
    var input_shape = fields[2]
    for j in range(len(input_shape)):  # small-loop(tensor shape dimension words)
        var size = Int(py=py_call(bi.int, input_shape[j]))
        if size < 0 or (size != 0 and count > 9223372036854775807 // size):
            raise Error("TensorInfo shape must hold non-negative dimensions")
        count *= size
        _ = py_call(shape.append, size)
    var begin = Int(py=py_call(bi.int, fields[3]))
    var end = Int(py=py_call(bi.int, fields[4]))
    if begin < 0 or end < begin:
        raise Error("TensorInfo offsets must satisfy 0 <= begin <= end")
    var itemsize = 8 if dtype == "I64" else (4 if dtype == "F32" or dtype == "I32" else (2 if dtype == "BF16" or dtype == "F16" else 0))
    if itemsize != 0 and (count > 9223372036854775807 // itemsize or count * itemsize != end - begin):
        raise Error("TensorInfo bytes differ from dtype/shape")
    py_setattr(value, "name", fields[0])
    py_setattr(value, "dtype", fields[1])
    py_setattr(value, "shape", py_call(bi.tuple, shape))
    py_setattr(value, "begin", PythonObject(begin))
    py_setattr(value, "end", PythonObject(end))
    py_setattr(value, "file", fields[5])
    return PythonObject(None)


def tensor_info_size_binding(value: PythonObject, bytes: PythonObject) raises -> PythonObject:
    if Bool(py=bytes):
        return PythonObject(Int(py=value.end) - Int(py=value.begin))
    var count = 1
    for i in range(len(value.shape)):  # small-loop(tensor shape dimension words)
        count *= Int(py=value.shape[i])
    return PythonObject(count)


def safetensors_init_binding(value: PythonObject, input_path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var path = native_path(input_path)
    py_setattr(value, "path", path)
    py_setattr(value, "_fh", PythonObject(None))
    py_setattr(value, "_mm", PythonObject(None))
    py_setattr(value, "_infos", bi.dict())
    if not native_path_exists(path):
        raise Error("FileNotFoundError: safetensors file does not exist")
    var file = py_call(bi.open, path, "rb")
    var raw = PythonObject(None)
    var total = 0
    var n = 0
    try:
        var head = py_call(file.read, 8)
        if len(head) != 8:
            raise Error("safetensors: file is shorter than its 8-byte header length")
        var length = UInt64(0)
        for i in range(8):  # small-loop(the 8-byte header length word)
            length |= UInt64(Int(py=head[i])) << UInt64(8 * i)
        if length > 100 * 1024 * 1024:
            raise Error("safetensors: header length exceeds 104857600")
        n = Int(length)
        raw = py_call(file.read, n)
        if len(raw) != n:
            raise Error("safetensors: header is truncated")
        _ = py_call(file.seek, 0, 2)
        total = Int(py=file.tell())
    except error:
        file.close()
        raise error
    file.close()
    py_setattr(value, "data_start", PythonObject(8 + n))
    py_setattr(value, "data_size", PythonObject(total - 8 - n))
    var header = native_json_parse(raw)
    if py_type(header) is not bi.dict:
        raise Error("safetensors: header is not a JSON object")
    py_setattr(value, "metadata", py_call(header.pop, "__metadata__", PythonObject(None)))
    var names = py_call(bi.list, header)
    for i in range(len(names)):  # driver-loop: over the checkpoint header tensor entries
        var name = names[i]
        var entry = header[name]
        if py_type(entry) is not bi.dict or not py_contains(entry, "dtype") or not py_contains(entry, "shape") or not py_contains(entry, "data_offsets"):
            raise Error("safetensors: tensor " + String(name) + " lacks dtype/shape/data_offsets")
        var dtype = String(entry["dtype"])
        var shape = bi.list()
        var count = 1
        var input_shape = entry["shape"]
        for j in range(len(input_shape)):  # small-loop(tensor shape dimension words)
            var size = Int(py=py_call(bi.int, input_shape[j]))
            if size < 0 or (size != 0 and count > 9223372036854775807 // size):
                raise Error("safetensors: invalid tensor shape")
            count *= size
            _ = py_call(shape.append, size)
        var offsets = entry["data_offsets"]
        if len(offsets) != 2:
            raise Error("safetensors: data_offsets requires a pair")
        var begin = Int(py=py_call(bi.int, offsets[0]))
        var end = Int(py=py_call(bi.int, offsets[1]))
        if begin < 0 or end < begin or end > total - 8 - n:
            raise Error("safetensors: tensor " + String(name) + " offsets fall outside the data section")
        var itemsize = 8 if dtype == "I64" else (4 if dtype == "F32" or dtype == "I32" else (2 if dtype == "BF16" or dtype == "F16" else 0))
        if itemsize != 0 and (count > 9223372036854775807 // itemsize or count * itemsize != end - begin):
            raise Error("safetensors: tensor " + String(name) + " bytes differ from dtype/shape")
        var info = py_call(bi.object.__new__, context["info_class"])
        py_setattr(info, "name", name)
        py_setattr(info, "dtype", entry["dtype"])
        py_setattr(info, "shape", py_call(bi.tuple, shape))
        py_setattr(info, "begin", PythonObject(begin))
        py_setattr(info, "end", PythonObject(end))
        py_setattr(info, "file", path)
        py_setitem(value._infos, name, info)
    return PythonObject(None)


def safetensors_map_binding(value: PythonObject) raises -> PythonObject:
    if value._mm is None:
        var bi = Python.import_module("builtins")
        var mmap = Python.import_module("mmap")
        py_setattr(value, "_fh", py_call(bi.open, value.path, "rb"))
        py_setattr(value, "_mm", py_call(mmap.mmap, value._fh.fileno(), 0, access=mmap.ACCESS_READ))
    return value._mm


def safetensors_close_binding(value: PythonObject) raises -> PythonObject:
    if value._mm is not None:
        value._mm.close()
        py_setattr(value, "_mm", PythonObject(None))
    if value._fh is not None:
        value._fh.close()
        py_setattr(value, "_fh", PythonObject(None))
    return PythonObject(None)


def safetensors_info_binding(value: PythonObject, name: PythonObject) raises -> PythonObject:
    if not py_contains(value._infos, name):
        raise Error("KeyError: safetensors holds no tensor " + String(name))
    return value._infos[name]


def safetensors_read_binding(value: PythonObject, name: PythonObject, bf16: PythonObject, context: PythonObject) raises -> PythonObject:
    var format = String(bf16)
    if format != "widen" and format != "bits":
        raise Error("safetensors: bf16 must be widen or bits")
    var info = safetensors_info_binding(value, name)
    var dtype = String(info.dtype)
    if dtype != "F32" and dtype != "F16" and dtype != "BF16" and dtype != "I32" and dtype != "I64":
        raise Error("TypeError: tensor " + String(name) + " dtype " + dtype + " refuses by name (the reader admits F32, F16, BF16, I32 and I64)")
    var bi = Python.import_module("builtins")
    var mmap = safetensors_map_binding(value)
    var begin = Int(py=value.data_start) + Int(py=info.begin)
    var end = Int(py=value.data_start) + Int(py=info.end)
    var raw = py_call(bi.memoryview, mmap)[py_call(bi.slice, begin, end)]
    var code = "f" if dtype == "F32" else ("q" if dtype == "I64" else ("i" if dtype == "I32" else "H"))
    var owned = checkpoint_array(raw, info.shape, code, context["array_class"])
    raw.release()
    if dtype == "F16" or (dtype == "BF16" and format == "widen"):
        var count = Int(py=owned.size)
        var out = checkpoint_array(py_call(bi.bytes, 4 * count), info.shape, "f", context["array_class"])
        if count != 0:
            var source = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(py=owned._addr))
            var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
            checkpoint_widen_halves(source, dest, count, dtype)
            _ = owned^  # read by address above
        return out
    if dtype == "BF16" and format == "bits" and len(info.shape) == 2:
        var packed = py_call(bi.object.__new__, context["bf16_class"])
        py_setattr(packed, "bits", owned)
        py_setattr(packed, "shape", info.shape)
        return packed
    return owned


def checkpoint_init_binding(value: PythonObject, files: PythonObject, weight_map: PythonObject, root: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    py_setattr(value, "root", root)
    py_setattr(value, "_paths", py_call(bi.list, files))
    py_setattr(value, "_weight_map", py_call(bi.dict, weight_map))
    py_setattr(value, "_open", bi.dict())
    py_setattr(value, "_infos", PythonObject(None))
    return PythonObject(None)


def checkpoint_open_binding(cls: PythonObject, input_path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var path = native_path(input_path)
    var files = bi.list()
    var weight_map = bi.dict()
    var root = path
    if native_path_exists(path, True):
        var index = py_call(path.rstrip, "/") + PythonObject("/model.safetensors.index.json")
        var single = py_call(path.rstrip, "/") + PythonObject("/model.safetensors")
        if native_path_exists(index):
            var data = native_json_read(index)
            if py_type(data) is not bi.dict:
                raise Error("safetensors: index has no weight_map")
            var wm = py_get(data, "weight_map")
            if py_type(wm) is not bi.dict or len(wm) == 0:
                raise Error("safetensors: index has no weight_map")
            var names = py_call(bi.list, wm)
            for i in range(len(names)):  # driver-loop: over the checkpoint index weight-map entries
                var filename = wm[names[i]]
                if not Bool(py=py_call(filename.startswith, "/")):
                    filename = py_call(path.rstrip, "/") + PythonObject("/") + filename
                if not native_path_exists(filename):
                    raise Error("FileNotFoundError: checkpoint index names shard that does not exist: " + String(filename))
                py_setitem(weight_map, names[i], filename)
                if not py_contains(files, filename):
                    _ = py_call(files.append, filename)
            files.sort()
        elif native_path_exists(single):
            _ = py_call(files.append, single)
        else:
            var entries = py_call(Python.import_module("os").listdir, path)
            for i in range(len(entries)):  # driver-loop: over checkpoint directory entries
                if Bool(py=py_call_typed(entries[i].endswith, ".safetensors")):
                    _ = py_call(files.append, py_call(path.rstrip, "/") + PythonObject("/") + entries[i])
            if len(files) != 1:
                raise Error("FileNotFoundError: checkpoint holds neither model.safetensors nor model.safetensors.index.json (shard files alone are not an index)")
    elif native_path_exists(path):
        _ = py_call(files.append, path)
        var pieces = py_call(path.rsplit, "/", 1)
        root = pieces[0] if len(pieces) == 2 else PythonObject(".")
        if len(root) == 0:
            root = PythonObject("/")
    else:
        raise Error("FileNotFoundError: checkpoint path does not exist")
    var value = py_call(bi.object.__new__, cls)
    _ = checkpoint_init_binding(value, files, weight_map, root)
    return value


def checkpoint_file_binding(value: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var file = py_get(value._open, path)
    if file is None:
        file = py_call(Python.import_module("builtins").object.__new__, context["file_class"])
        _ = safetensors_init_binding(file, path, context)
        py_setitem(value._open, path, file)
    return file


def checkpoint_index_binding(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value._infos is None:
        var bi = Python.import_module("builtins")
        var infos = bi.dict()
        for i in range(len(value._paths)):  # driver-loop: over checkpoint shard files
            var path = value._paths[i]
            var file = checkpoint_file_binding(value, path, context)
            var names = py_call(bi.list, file._infos)
            for j in range(len(names)):  # driver-loop: over a shard's tensor entries
                var name = names[j]
                if py_contains(infos, name):
                    raise Error("safetensors: tensor " + String(name) + " appears in both shards")
                py_setitem(infos, name, file._infos[name])
        var names = py_call(bi.list, value._weight_map)
        for i in range(len(names)):  # driver-loop: over the checkpoint weight-map entries
            var name = names[i]
            var path = value._weight_map[name]
            if not py_contains(infos, name) or not Bool(py=infos[name].file == path):
                raise Error("safetensors: index maps " + String(name) + " to a shard that does not hold it")
        py_setattr(value, "_infos", infos)
    return value._infos


def checkpoint_access_binding(value: PythonObject, operation: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    var infos = checkpoint_index_binding(value, context)
    var op = Int(py=operation)
    if op == 0:
        return py_call(Python.import_module("builtins").list, infos)
    if op == 1:
        if not py_contains(infos, name):
            raise Error("KeyError: checkpoint holds no tensor " + String(name))
        return infos[name]
    return py_contains(infos, name)


def checkpoint_read_binding(value: PythonObject, name: PythonObject, bf16: PythonObject, context: PythonObject) raises -> PythonObject:
    var info = checkpoint_access_binding(value, PythonObject(1), name, context)
    return safetensors_read_binding(checkpoint_file_binding(value, info.file, context), name, bf16, context)


def checkpoint_close_binding(value: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var files = py_call(bi.list, value._open.values())
    for i in range(len(files)):  # driver-loop: over open shard files
        _ = safetensors_close_binding(files[i])
    py_setattr(value, "_open", bi.dict())
    return PythonObject(None)


def checkpoint_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = tensor_info_init_binding(args[0], args[1])
        elif operation == 1:
            result = tensor_info_size_binding(args[0], args[1])
        elif operation == 2:
            result = safetensors_init_binding(args[0], args[1], args[2])
        elif operation == 3:
            result = safetensors_info_binding(args[0], args[1])
        elif operation == 4:
            result = safetensors_map_binding(args[0])
        elif operation == 5:
            result = safetensors_read_binding(args[0], args[1], args[2], args[3])
        elif operation == 6:
            result = safetensors_close_binding(args[0])
        elif operation == 7:
            result = checkpoint_init_binding(args[0], args[1], args[2], args[3])
        elif operation == 8:
            result = checkpoint_open_binding(args[0], args[1], args[2])
        elif operation == 9:
            result = checkpoint_file_binding(args[0], args[1], args[2])
        elif operation == 10:
            result = checkpoint_index_binding(args[0], args[1])
        elif operation == 11:
            result = checkpoint_access_binding(args[0], args[1], args[2], args[3])
        elif operation == 12:
            result = checkpoint_read_binding(args[0], args[1], args[2], args[3])
        elif operation == 13:
            result = checkpoint_close_binding(args[0])
        elif operation == 14:
            result = args[0]
        elif operation == 15:
            result = py_call(Python.import_module("builtins").list, args[0]._infos)
        elif operation == 16:
            result = args[0]._infos.__contains__(args[1])
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
