# SPDX-License-Identifier: Apache-2.0
"""Native safetensors header admission, mapped reads and owned tensor buffers."""
from std.python import Python, PythonObject
from std.memory import bitcast
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_json import native_json_parse, native_json_read, native_path, native_path_exists
from gemm.f16_widen import f16_bits_to_f32


def checkpoint_array(raw: PythonObject, shape: PythonObject, code: String, array_class: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var store = Python.import_module("array").array(code)
    store.frombytes(raw)
    var result = bi.object.__new__(array_class)
    var itemsize = Int(py=store.itemsize)
    var count = 1
    var strides = bi.list()
    var stride = itemsize
    for i in range(len(shape)):
        var size = Int(py=shape[len(shape) - i - 1])
        if size < 0:
            raise Error("safetensors: negative shape dimension")
        count *= size
        strides.insert(0, stride)
        stride *= size
    if count != len(store):
        raise Error("safetensors: tensor shape does not match stored bytes")
    result._store = store
    result._mv = bi.memoryview(store)
    result._base = PythonObject(None)
    result._pin = PythonObject(None)
    result._addr = store.buffer_info()[0]
    result._readonly = PythonObject(False)
    result.shape = bi.tuple(shape)
    result.dtype = PythonObject("<f4" if code == "f" else ("<u2" if code == "H" else ("<i8" if code == "q" else "<i4")))
    result.ndim = PythonObject(len(shape))
    result.size = PythonObject(count)
    result.nbytes = PythonObject(count * itemsize)
    result.itemsize = PythonObject(itemsize)
    result.order = PythonObject("C")
    result.strides = bi.tuple(strides)
    return result


def tensor_info_init_binding(value: PythonObject, fields: PythonObject) raises -> PythonObject:
    value.name = fields[0]
    value.dtype = fields[1]
    value.shape = fields[2]
    value.begin = fields[3]
    value.end = fields[4]
    value.file = fields[5]
    return PythonObject(None)


def tensor_info_size_binding(value: PythonObject, bytes: PythonObject) raises -> PythonObject:
    if Bool(py=bytes):
        return PythonObject(Int(py=value.end) - Int(py=value.begin))
    var count = 1
    for i in range(len(value.shape)):
        count *= Int(py=value.shape[i])
    return PythonObject(count)


def safetensors_init_binding(value: PythonObject, input_path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var path = native_path(input_path)
    value.path = path
    value._fh = PythonObject(None)
    value._mm = PythonObject(None)
    value._infos = bi.dict()
    if not native_path_exists(path):
        raise Error("FileNotFoundError: safetensors file does not exist")
    var file = bi.open(path, "rb")
    var raw = PythonObject(None)
    var total = 0
    var n = 0
    try:
        var head = file.read(8)
        if len(head) != 8:
            raise Error("safetensors: file is shorter than its 8-byte header length")
        var length = UInt64(0)
        for i in range(8):
            length |= UInt64(Int(py=head[i])) << UInt64(8 * i)
        if length > 100 * 1024 * 1024:
            raise Error("safetensors: header length exceeds 104857600")
        n = Int(length)
        raw = file.read(n)
        if len(raw) != n:
            raise Error("safetensors: header is truncated")
        file.seek(0, 2)
        total = Int(py=file.tell())
    except error:
        file.close()
        raise error
    file.close()
    value.data_start = PythonObject(8 + n)
    value.data_size = PythonObject(total - 8 - n)
    var header = native_json_parse(raw)
    if bi.type(header) is not bi.dict:
        raise Error("safetensors: header is not a JSON object")
    value.metadata = header.pop("__metadata__", PythonObject(None))
    var names = bi.list(header)
    for i in range(len(names)):
        var name = names[i]
        var entry = header[name]
        if bi.type(entry) is not bi.dict or not Bool(py=entry.__contains__("dtype")) or not Bool(py=entry.__contains__("shape")) or not Bool(py=entry.__contains__("data_offsets")):
            raise Error("safetensors: tensor " + String(name) + " lacks dtype/shape/data_offsets")
        var dtype = String(entry["dtype"])
        var shape = bi.list()
        var count = 1
        var input_shape = entry["shape"]
        for j in range(len(input_shape)):
            var size = Int(py=bi.int(input_shape[j]))
            if size < 0 or (size != 0 and count > 9223372036854775807 // size):
                raise Error("safetensors: invalid tensor shape")
            count *= size
            shape.append(size)
        var offsets = entry["data_offsets"]
        if len(offsets) != 2:
            raise Error("safetensors: data_offsets requires a pair")
        var begin = Int(py=bi.int(offsets[0]))
        var end = Int(py=bi.int(offsets[1]))
        if begin < 0 or end < begin or end > total - 8 - n:
            raise Error("safetensors: tensor " + String(name) + " offsets fall outside the data section")
        var itemsize = 8 if dtype == "I64" else (4 if dtype == "F32" or dtype == "I32" else (2 if dtype == "BF16" or dtype == "F16" else 0))
        if itemsize != 0 and (count > 9223372036854775807 // itemsize or count * itemsize != end - begin):
            raise Error("safetensors: tensor " + String(name) + " bytes differ from dtype/shape")
        var info = bi.object.__new__(context["info_class"])
        info.name = name
        info.dtype = entry["dtype"]
        info.shape = bi.tuple(shape)
        info.begin = PythonObject(begin)
        info.end = PythonObject(end)
        info.file = path
        value._infos[name] = info
    return PythonObject(None)


def safetensors_map_binding(value: PythonObject) raises -> PythonObject:
    if value._mm is None:
        var bi = Python.import_module("builtins")
        var mmap = Python.import_module("mmap")
        value._fh = bi.open(value.path, "rb")
        value._mm = mmap.mmap(value._fh.fileno(), 0, access=mmap.ACCESS_READ)
    return value._mm


def safetensors_close_binding(value: PythonObject) raises -> PythonObject:
    if value._mm is not None:
        value._mm.close()
        value._mm = PythonObject(None)
    if value._fh is not None:
        value._fh.close()
        value._fh = PythonObject(None)
    return PythonObject(None)


def safetensors_info_binding(value: PythonObject, name: PythonObject) raises -> PythonObject:
    if not Bool(py=value._infos.__contains__(name)):
        raise Error("KeyError: safetensors holds no tensor " + String(name))
    return value._infos[name]


def safetensors_read_binding(value: PythonObject, name: PythonObject, bf16: PythonObject, context: PythonObject) raises -> PythonObject:
    var format = String(bf16)
    if format != "widen" and format != "bits":
        raise Error("safetensors: bf16 must be widen or bits")
    var info = safetensors_info_binding(value, name)
    var dtype = String(info.dtype)
    if dtype != "F32" and dtype != "F16" and dtype != "BF16" and dtype != "I32" and dtype != "I64":
        raise Error("TypeError: tensor " + String(name) + " dtype " + dtype + " is refused by name")
    var bi = Python.import_module("builtins")
    var mmap = safetensors_map_binding(value)
    var begin = Int(py=value.data_start) + Int(py=info.begin)
    var end = Int(py=value.data_start) + Int(py=info.end)
    var raw = bi.memoryview(mmap)[bi.slice(begin, end)]
    var code = "f" if dtype == "F32" else ("q" if dtype == "I64" else ("i" if dtype == "I32" else "H"))
    var owned = checkpoint_array(raw, info.shape, code, context["array_class"])
    raw.release()
    if dtype == "F16" or (dtype == "BF16" and format == "widen"):
        var count = Int(py=owned.size)
        var out = checkpoint_array(bi.bytes(4 * count), info.shape, "f", context["array_class"])
        if count != 0:
            var source = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(py=owned._addr))
            var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
            for i in range(count):
                dest[i] = f16_bits_to_f32(source[i]) if dtype == "F16" else bitcast[DType.float32](UInt32(source[i]) << 16)
        return out
    if dtype == "BF16" and format == "bits" and len(info.shape) == 2:
        var packed = bi.object.__new__(context["bf16_class"])
        packed.bits = owned
        packed.shape = info.shape
        return packed
    return owned


def checkpoint_init_binding(value: PythonObject, files: PythonObject, weight_map: PythonObject, root: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    value.root = root
    value._paths = bi.list(files)
    value._weight_map = bi.dict(weight_map)
    value._open = bi.dict()
    value._infos = PythonObject(None)
    return PythonObject(None)


def checkpoint_open_binding(cls: PythonObject, input_path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var path = native_path(input_path)
    var files = bi.list()
    var weight_map = bi.dict()
    var root = path
    if native_path_exists(path, True):
        var index = path.rstrip("/") + PythonObject("/model.safetensors.index.json")
        var single = path.rstrip("/") + PythonObject("/model.safetensors")
        if native_path_exists(index):
            var data = native_json_read(index)
            if bi.type(data) is not bi.dict:
                raise Error("safetensors: index has no weight_map")
            var wm = data.get("weight_map")
            if bi.type(wm) is not bi.dict or len(wm) == 0:
                raise Error("safetensors: index has no weight_map")
            var names = bi.list(wm)
            for i in range(len(names)):
                var filename = wm[names[i]]
                if not Bool(py=filename.startswith("/")):
                    filename = path.rstrip("/") + PythonObject("/") + filename
                if not native_path_exists(filename):
                    raise Error("FileNotFoundError: checkpoint index names shard that does not exist: " + String(filename))
                weight_map[names[i]] = filename
                if not Bool(py=files.__contains__(filename)):
                    files.append(filename)
            files.sort()
        elif native_path_exists(single):
            files.append(single)
        else:
            var entries = Python.import_module("os").listdir(path)
            for i in range(len(entries)):
                if Bool(py=entries[i].endswith(".safetensors")):
                    files.append(path.rstrip("/") + PythonObject("/") + entries[i])
            if len(files) != 1:
                raise Error("FileNotFoundError: checkpoint holds neither model.safetensors nor model.safetensors.index.json")
    elif native_path_exists(path):
        files.append(path)
        var pieces = path.rsplit("/", 1)
        root = pieces[0] if len(pieces) == 2 else PythonObject(".")
        if len(root) == 0:
            root = PythonObject("/")
    else:
        raise Error("FileNotFoundError: checkpoint path does not exist")
    var value = bi.object.__new__(cls)
    _ = checkpoint_init_binding(value, files, weight_map, root)
    return value


def checkpoint_file_binding(value: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var file = value._open.get(path)
    if file is None:
        file = Python.import_module("builtins").object.__new__(context["file_class"])
        _ = safetensors_init_binding(file, path, context)
        value._open[path] = file
    return file


def checkpoint_index_binding(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value._infos is None:
        var bi = Python.import_module("builtins")
        var infos = bi.dict()
        for i in range(len(value._paths)):
            var path = value._paths[i]
            var file = checkpoint_file_binding(value, path, context)
            var names = bi.list(file._infos)
            for j in range(len(names)):
                var name = names[j]
                if Bool(py=infos.__contains__(name)):
                    raise Error("safetensors: tensor " + String(name) + " appears in both shards")
                infos[name] = file._infos[name]
        var names = bi.list(value._weight_map)
        for i in range(len(names)):
            var name = names[i]
            var path = value._weight_map[name]
            if not Bool(py=infos.__contains__(name)) or not Bool(py=infos[name].file == path):
                raise Error("safetensors: index maps " + String(name) + " to a shard that does not hold it")
        value._infos = infos
    return value._infos


def checkpoint_access_binding(value: PythonObject, operation: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    var infos = checkpoint_index_binding(value, context)
    var op = Int(py=operation)
    if op == 0:
        return Python.import_module("builtins").list(infos)
    if op == 1:
        if not Bool(py=infos.__contains__(name)):
            raise Error("KeyError: checkpoint holds no tensor " + String(name))
        return infos[name]
    return infos.__contains__(name)


def checkpoint_read_binding(value: PythonObject, name: PythonObject, bf16: PythonObject, context: PythonObject) raises -> PythonObject:
    var info = checkpoint_access_binding(value, PythonObject(1), name, context)
    return safetensors_read_binding(checkpoint_file_binding(value, info.file, context), name, bf16, context)


def checkpoint_close_binding(value: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var files = bi.list(value._open.values())
    for i in range(len(files)):
        _ = safetensors_close_binding(files[i])
    value._open = bi.dict()
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
            result = Python.import_module("builtins").list(args[0]._infos)
        elif operation == 16:
            result = args[0]._infos.__contains__(args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error, ExceptionType("PyExc_KeyError"))
        return raise_python_exception(error, ExceptionType.ValueError)
