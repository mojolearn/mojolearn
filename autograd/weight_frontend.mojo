# SPDX-License-Identifier: Apache-2.0
# Existing Session.load_weights inputs, parsed and validated natively.
# Checkpoint files use native mmap ownership; no Python safetensors reader,
# NumPy conversion, tensor conversion or package callback executes here.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.program_arrays import mapped_byte_array
from autograd.program_pyapi import ProgramInput, _integer
from autograd.pyapi import ops_load_weights_py
from bindings.native_json import native_json_parse, native_json_read, native_path, native_path_exists


def _dtype(dtype: String) raises -> Tuple[Int, Int]:
    if dtype == "F16" or dtype == "torch.float16":
        return (23, 2)
    if dtype == "F32" or dtype == "torch.float32":
        return (11, 4)
    if dtype == "F64" or dtype == "torch.float64":
        return (12, 8)
    if dtype == "BF16" or dtype == "torch.bfloat16":
        return (4096, 2)
    raise Error("TypeError: a weight must be float16, bfloat16, float32 or float64; got " + dtype)


def _file_header(path: PythonObject) raises -> PythonObject:
    if not native_path_exists(path):
        raise Error("FileNotFoundError: safetensors file does not exist: " + String(path))
    var view = mapped_byte_array(path)
    var raw = view.base
    if len(raw) < 8:
        raise Error("ValueError: safetensors file has no header")
    var count = UInt64(0)
    for k in range(8):  # small-loop(8: safetensors header-length bytes): file metadata
        count |= UInt64(py=raw[k]) << (8 * k)
    if count > UInt64(len(raw) - 8):
        raise Error("ValueError: safetensors header exceeds the file")
    var b = Python.import_module("builtins")
    var header = native_json_parse(raw[b.slice(8, 8 + Int(count))])
    if b.type(header) is not b.dict:
        raise Error("ValueError: safetensors header must be an object")
    var out = Python.list()
    out.append(view)
    out.append(header)
    out.append(8 + Int(count))
    return out


def _checkpoint(path: PythonObject, mut owners: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var file = native_path(path)
    var files = Python.list()
    var weight_map = b.dict()
    if native_path_exists(file, True):
        var base = String(file)
        var index = PythonObject(base + "/model.safetensors.index.json")
        if native_path_exists(index):
            var metadata = native_json_read(index)
            weight_map = metadata["weight_map"]
            if b.type(weight_map) is not b.dict or len(weight_map) == 0:
                raise Error("ValueError: safetensors index has no weight_map")
            var shards = b.list(weight_map.values())
            var seen = b.set()
            for k in range(len(shards)):  # driver-loop: over named checkpoint shards
                var full = PythonObject(base + "/" + String(shards[k]))
                if not Bool(py=seen.__contains__(full)):
                    seen.add(full)
                    files.append(full)
        elif native_path_exists(PythonObject(base + "/model.safetensors")):
            files.append(base + "/model.safetensors")
        else:
            var names = Python.import_module("os").listdir(file)
            for k in range(len(names)):  # driver-loop: over checkpoint directory entries
                if String(names[k]).endswith(".safetensors"):
                    files.append(base + "/" + String(names[k]))
            if len(files) != 1:
                raise Error("FileNotFoundError: checkpoint directory needs one safetensors file or model.safetensors.index.json")
    else:
        files.append(file)
    var tensors = b.dict()
    for k in range(len(files)):  # driver-loop: over checkpoint shard headers
        var entry = _file_header(files[k])
        owners.append(entry)
        var header = entry[1]
        var keys = b.list(header.keys())
        for j in range(len(keys)):  # driver-loop: over tensor metadata in the shard
            var key = keys[j]
            if String(key) == "__metadata__":
                continue
            if Bool(py=tensors.__contains__(key)):
                raise Error("ValueError: tensor appears in two checkpoint shards: " + String(key))
            var located = Python.list()
            located.append(entry[0])
            located.append(header[key])
            located.append(entry[2])
            tensors[key] = located
    var expected = b.list(weight_map.keys())
    for k in range(len(expected)):  # driver-loop: over index declarations
        if not Bool(py=tensors.__contains__(expected[k])):
            raise Error("ValueError: safetensors index names a tensor absent from its shards: " + String(expected[k]))
    return tensors


def _record(name: PythonObject, target: Int, layout: Int, value: PythonObject, checkpoint: Bool,
            mut scalar_owners: List[ProgramInput]) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var addr = 0
    var offset = 0
    var dtype = 0
    var size = 0
    var dims = PythonObject(None)
    var strides = PythonObject(None)
    var unit = 0
    if checkpoint:
        var view = value[0]
        var info = value[1]
        var dt = _dtype(String(info["dtype"]))
        dtype = dt[0]
        size = dt[1]
        dims = info["shape"]
        addr = Int(py=view.__array_interface__["data"][0])
        var offsets = info["data_offsets"]
        var begin = _integer(offsets[0], "tensor begin", 0)
        var end = _integer(offsets[1], "tensor end", begin)
        var n = 1
        for k in range(len(dims)):  # small-loop(dims: weight rank): byte extent validation
            n *= _integer(dims[k], "dimension", 1)
        offset = Int(py=value[2]) + begin
        if end - begin != n * size or Int(py=value[2]) + end > len(view):
            raise Error("ValueError: safetensors tensor extent disagrees with shape or file length: " + String(name))
    elif String(b.type(value).__module__).startswith("torch"):
        var dt = _dtype(String(value.dtype))
        dtype = dt[0]
        size = dt[1]
        if String(value.device.type) != "cpu":
            raise Error("ValueError: load_weights requires CPU source tensors")
        addr = Int(py=value.data_ptr())
        dims = value.shape
        strides = value.stride()
        unit = 2
    elif Bool(py=b.hasattr(value, "dtype")) and Bool(py=b.hasattr(value, "strides")):
        if not Bool(py=value.dtype.isnative):
            raise Error("TypeError: a weight must use native byte order")
        dtype = 4096 if String(value.dtype.name) == "bfloat16" else Int(py=value.dtype.num)
        size = Int(py=value.itemsize)
        addr = Int(py=value.__array_interface__["data"][0])
        dims = value.shape
        strides = value.strides
        unit = 1
    else:
        scalar_owners.append(ProgramInput(value, False))
        var at = len(scalar_owners) - 1
        addr = scalar_owners[at].src.addr
        dtype = 12
        size = 8
        dims = Python.list()
        for k in range(scalar_owners[at].shape.ndim):  # small-loop(rank: list weight shape): metadata only
            dims.append(scalar_owners[at].shape.d[k])
    var nd = len(dims)
    if nd < 1 or nd > 4:
        raise Error("ValueError: a weight must have one to four dimensions")
    var row = Python.list()
    row.append(target)
    row.append(0)
    row.append(layout)
    row.append(addr)
    row.append(offset)
    row.append(dtype)
    row.append(size)
    row.append(nd)
    for k in range(4):  # small-loop(4: fixed weight descriptor dimensions): metadata only
        row.append(_integer(dims[k], "dimension", 1) if k < nd else 0)
    row.append(unit)
    for k in range(4):  # small-loop(4: fixed weight descriptor strides): metadata only
        row.append(Int(py=strides[k]) if unit != 0 and k < nd else 0)
    return row


def ops_weights_native_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, obj: PythonObject, names: PythonObject, source: PythonObject,
) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var path = Bool(py=b.isinstance(obj, b.str)) or Bool(py=b.isinstance(obj, b.bytes)) or Bool(py=b.hasattr(obj, "__fspath__"))
    var kind = ("safetensors" if path else "torch") if source is PythonObject(None) else String(source)
    if kind != "torch" and kind != "safetensors":
        raise Error("ValueError: source must be 'torch' or 'safetensors'")
    if (kind == "safetensors") != path:
        raise Error("TypeError: safetensors takes a path; torch takes a mapping of names to tensors")
    if not path and (not Bool(py=b.hasattr(obj, "keys")) or not Bool(py=b.hasattr(obj, "__getitem__"))):
        raise Error("TypeError: source='torch' takes a mapping of names to tensors")
    var owners = Python.list()
    var tensor_owners = Python.list()
    var values = _checkpoint(obj, owners) if path else obj
    var known = b.set(values.keys())
    var requested = b.list(names)
    if len(requested) == 0:
        raise Error("ValueError: names must hold at least one (handle, name) pair")
    var records = Python.list()
    var keys = Python.list()
    var handles = Python.list()
    var scalar_owners = List[ProgramInput]()
    for k in range(len(requested)):  # driver-loop: over requested named tensors
        var item = requested[k]
        if (b.type(item) is not b.list and b.type(item) is not b.tuple) or (len(item) != 2 and len(item) != 3):
            raise Error("ValueError: names takes (handle, name) or (handle, name, 'transpose') pairs")
        var handle = _integer(item[0], "handle", 1)
        if not Bool(py=b.isinstance(item[1], b.str)):
            raise Error("TypeError: a tensor name must be a string")
        var layout = 2
        if len(item) == 3:
            if String(item[2]) != "transpose":
                raise Error("ValueError: the third item of a names entry must be 'transpose'")
            layout = 3
        if not Bool(py=known.__contains__(item[1])):
            raise Error("ValueError: load_weights source has no tensor '" + String(item[1]) + "'")
        # A caller mapping may create a fresh tensor on lookup. Keep the
        # returned object, as well as the mapping, through native transfer.
        var value = values[item[1]]
        tensor_owners.append(value)
        records.extend(_record(item[1], handle, layout, value, path, scalar_owners))
        keys.append(item[1])
        handles.append(handle)
    _ = ops_load_weights_py[E, KIND](sid, records, keys)
    _ = owners^
    _ = values^
    _ = tensor_owners^
    _ = scalar_owners^
    return handles
