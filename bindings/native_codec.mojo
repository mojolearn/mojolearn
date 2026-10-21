# SPDX-License-Identifier: Apache-2.0
"""Native structural state codec for process workers; never executes pickle.

Descriptors contain only exact builtins. Array payloads retain their dtype,
shape, physical order and bytes. Bounds are checked before allocation; tensor
ranges must cover the payload exactly in traversal order. Object identity and
arbitrary Python classes are deliberately outside this private wire format.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.math import isfinite


struct CodecBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var bf16_cls: PythonObject
    var int8_cls: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.bf16_cls = PythonObject(None)
        self.int8_cls = PythonObject(None)


comptime CODEC_BOUNDARY = _Global[
    StorageType=CodecBoundary, name="MojoNativeStateCodec",
    init_fn=CodecBoundary.__init__,
]


def codec_initialize(array_cls: PythonObject, bf16_cls: PythonObject, int8_cls: PythonObject) raises:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].arrays = Python.import_module("array").array
    c[].array_cls = array_cls
    if bf16_cls is not None:
        c[].bf16_cls = bf16_cls
    if int8_cls is not None:
        c[].int8_cls = int8_cls


def _codec_code(dtype: String) raises -> String:
    if dtype == "<f4":
        return "f"
    if dtype == "<f8":
        return "d"
    if dtype == "<i4":
        return "i"
    if dtype == "<i8":
        return "q"
    if dtype == "<u4":
        return "I"
    if dtype == "<u1":
        return "B"
    if dtype == "<f2" or dtype == "<u2":
        return "H"
    if dtype == "<i1":
        return "b"
    raise Error("codec: unsupported tensor dtype " + dtype)


def _codec_dtype(format: String, itemsize: Int) raises -> String:
    var code = format
    if code.startswith("=") or code.startswith("@") or code.startswith("<"):
        code = String(PythonObject(format).lstrip("=@<"))
    if code == "f" and itemsize == 4:
        return "<f4"
    if code == "d" and itemsize == 8:
        return "<f8"
    if code == "e" and itemsize == 2:
        return "<f2"
    if (code == "i" or code == "l") and itemsize == 4:
        return "<i4"
    if (code == "q" or code == "l") and itemsize == 8:
        return "<i8"
    if (code == "I" or code == "L") and itemsize == 4:
        return "<u4"
    if code == "H" and itemsize == 2:
        return "<u2"
    if code == "b" and itemsize == 1:
        return "<i1"
    if code == "B" and itemsize == 1:
        return "<u1"
    raise Error("codec: unsupported or nonnative buffer format " + format)


def _codec_integer(value: PythonObject, maximum: Int) raises -> Int:
    var bi = CODEC_BOUNDARY.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.int:
        raise Error("codec: expected an exact integer extent")
    var n = Int(py=value)
    if n < 0 or n > maximum:
        raise Error("codec: extent outside the admitted range")
    return n


def _codec_node(depth: Int, mut budget: Int) raises:
    budget -= 1
    if budget < 0 or depth > 64:
        raise Error("codec: structure exceeds its depth or node budget")


def _codec_buffer(raw: PythonObject, payload: PythonObject, kind: String, maximum: Int) raises -> PythonObject:
    var bi = CODEC_BOUNDARY.get_or_create_ptr()[].bi
    var begin = len(payload)
    var count = len(raw)
    if count > maximum - begin:
        raise Error("codec: payload exceeds its byte limit")
    var item = bi.dict()
    item["kind"] = PythonObject(kind)
    item["begin"] = PythonObject(begin)
    item["size"] = PythonObject(count)
    payload.extend(raw)
    return item


def _codec_encode(value: PythonObject, payload: PythonObject, maximum: Int,
                  depth: Int, mut budget: Int) raises -> PythonObject:
    _codec_node(depth, budget)
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var cls = bi.type(value)
    if value is None or cls is bi.bool or cls is bi.int or cls is bi.str:
        return value
    if cls is bi.float:
        if not isfinite(Float64(py=value)):
            var tagged = bi.dict()
            tagged["kind"] = PythonObject("float")
            tagged["value"] = bi.str(value)
            return tagged
        return value
    if cls is bi.bytes or cls is bi.bytearray:
        return _codec_buffer(value, payload, "bytes" if cls is bi.bytes else "bytearray", maximum)
    var result = bi.dict()
    if cls is bi.list or cls is bi.tuple or cls is bi.dict:
        result["kind"] = PythonObject("dict" if cls is bi.dict else ("tuple" if cls is bi.tuple else "list"))
        var items = bi.list()
        if cls is bi.dict:
            var keys = bi.list(value)
            for i in range(len(keys)):  # driver-loop: over structural state dictionary fields
                var pair = bi.list()
                var key = keys[i]
                var key_cls = bi.type(key)
                if key_cls is not bi.str and key_cls is not bi.int and key_cls is not bi.bool:
                    raise Error("codec: dictionary keys require string or integer metadata")
                pair.append(key)
                pair.append(_codec_encode(value[key], payload, maximum, depth + 1, budget))
                items.append(pair)
        else:
            for i in range(len(value)):  # driver-loop: over structural state sequence entries
                items.append(_codec_encode(value[i], payload, maximum, depth + 1, budget))
        result["items"] = items
        return result
    if c[].bf16_cls is not None and cls is c[].bf16_cls:
        result["kind"] = PythonObject("bf16")
        result["bits"] = _codec_encode(value.bits, payload, maximum, depth + 1, budget)
        return result
    if c[].int8_cls is not None and cls is c[].int8_cls:
        result["kind"] = PythonObject("int8")
        result["codes"] = _codec_encode(value.codes, payload, maximum, depth + 1, budget)
        result["exponents"] = _codec_encode(value.exponents, payload, maximum, depth + 1, budget)
        return result
    var native = cls is c[].array_cls
    var view = value._mv if native else bi.memoryview(value)
    var dtype = String(value.dtype) if native else _codec_dtype(String(view.format), Int(py=view.itemsize))
    _ = _codec_code(dtype)
    var shape = value.shape if native else view.shape
    var order = String(value.order) if native else String("C")
    if order != "C" and order != "F":
        raise Error("codec: tensor order must be C or F")
    var raw = view.tobytes("C")
    result = _codec_buffer(raw, payload, "array", maximum)
    result["dtype"] = PythonObject(dtype)
    result["shape"] = bi.list(shape)
    result["order"] = PythonObject(order)
    result["readonly"] = value._readonly if native else view.readonly
    return result


def _codec_slice(item: PythonObject, payload: PythonObject, maximum: Int,
                 mut cursor: Int) raises -> PythonObject:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var begin = _codec_integer(item["begin"], maximum)
    var size = _codec_integer(item["size"], maximum)
    if begin != cursor or size > len(payload) - begin:
        raise Error("codec: tensor ranges must be contiguous and inside payload")
    cursor += size
    return payload[c[].bi.slice(begin, cursor)]


def _codec_array(item: PythonObject, raw: PythonObject, maximum: Int) raises -> PythonObject:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var dtype = String(item["dtype"])
    var code = _codec_code(dtype)
    var store = c[].arrays(code)
    var itemsize = Int(py=store.itemsize)
    var shape = item["shape"]
    if bi.type(shape) is not bi.list or len(shape) > 64:
        raise Error("codec: tensor shape must be a bounded list")
    var count = 1
    for i in range(len(shape)):  # driver-loop: over tensor dimension metadata
        var n = _codec_integer(shape[i], maximum)
        if n != 0 and count > maximum // n:
            raise Error("codec: tensor element count overflows payload bound")
        count *= n
    if count > maximum // itemsize or count * itemsize != len(raw):
        raise Error("codec: tensor shape and dtype differ from stored byte count")
    var order = String(item["order"])
    if order != "C" and order != "F" or bi.type(item["readonly"]) is not bi.bool:
        raise Error("codec: invalid tensor layout metadata")
    store.frombytes(raw)
    var result = bi.object.__new__(c[].array_cls)
    result._store = store
    result._base = PythonObject(None)
    result._pin = PythonObject(None)
    result._mv = bi.memoryview(store)
    result._addr = store.buffer_info()[0]
    result._readonly = item["readonly"]
    result.shape = bi.tuple(shape)
    result.dtype = PythonObject(dtype)
    result.ndim = PythonObject(len(shape))
    result.size = PythonObject(count)
    result.nbytes = PythonObject(len(raw))
    result.itemsize = PythonObject(itemsize)
    result.order = PythonObject(order)
    var strides = bi.list()
    var stride = itemsize
    for i in range(len(shape)):  # driver-loop: over tensor layout strides
        if order == "C":
            strides.insert(0, stride)
            var size = Int(py=shape[len(shape) - i - 1])
            if size != 0 and stride > 9223372036854775807 // size:
                raise Error("codec: tensor stride overflows native indexing")
            stride *= size
        else:
            strides.append(stride)
            var size = Int(py=shape[i])
            if size != 0 and stride > 9223372036854775807 // size:
                raise Error("codec: tensor stride overflows native indexing")
            stride *= size
    result.strides = bi.tuple(strides)
    return result


def _codec_decode(item: PythonObject, payload: PythonObject, maximum: Int,
                  depth: Int, mut budget: Int, mut cursor: Int) raises -> PythonObject:
    _codec_node(depth, budget)
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var cls = bi.type(item)
    if item is None or cls is bi.bool or cls is bi.int or cls is bi.str:
        return item
    if cls is bi.float:
        if not isfinite(Float64(py=item)):
            raise Error("codec: scalar metadata must be finite")
        return item
    if cls is not bi.dict or bi.type(item.get("kind")) is not bi.str:
        raise Error("codec: invalid structural descriptor")
    var kind = String(item["kind"])
    if kind == "float":
        var text = String(item["value"])
        if text != "inf" and text != "-inf" and text != "nan":
            raise Error("codec: invalid nonfinite scalar tag")
        return bi.float(item["value"])
    if kind == "bytes" or kind == "bytearray" or kind == "array":
        var raw = _codec_slice(item, payload, maximum, cursor)
        if kind == "array":
            return _codec_array(item, raw, maximum)
        return bi.bytearray(raw) if kind == "bytearray" else bi.bytes(raw)
    if kind == "list" or kind == "tuple" or kind == "dict":
        var items = item["items"]
        if bi.type(items) is not bi.list:
            raise Error("codec: container entries must be a list")
        var result = bi.dict() if kind == "dict" else bi.list()
        for i in range(len(items)):  # driver-loop: over structural state container entries
            if kind == "dict":
                var pair = items[i]
                if bi.type(pair) is not bi.list or len(pair) != 2:
                    raise Error("codec: dictionary entries require key/value pairs")
                var key = pair[0]
                var key_cls = bi.type(key)
                if key_cls is not bi.str and key_cls is not bi.int and key_cls is not bi.bool:
                    raise Error("codec: invalid dictionary key")
                if Bool(py=result.__contains__(key)):
                    raise Error("codec: duplicate dictionary key")
                result[key] = _codec_decode(pair[1], payload, maximum, depth + 1, budget, cursor)
            else:
                result.append(_codec_decode(items[i], payload, maximum, depth + 1, budget, cursor))
        return bi.tuple(result) if kind == "tuple" else result
    if kind == "bf16":
        if c[].bf16_cls is None:
            raise Error("codec: bfloat16 weight class is not connected")
        var bits = _codec_decode(item["bits"], payload, maximum, depth + 1, budget, cursor)
        if bi.type(bits) is not c[].array_cls or String(bits.dtype) != "<u2" or len(bits.shape) != 2:
            raise Error("codec: bfloat16 weights require a two-dimensional uint16 tensor")
        var result = bi.object.__new__(c[].bf16_cls)
        result.bits = bits
        result.shape = bits.shape
        return result
    if kind == "int8":
        if c[].int8_cls is None:
            raise Error("codec: int8 weight class is not connected")
        var codes = _codec_decode(item["codes"], payload, maximum, depth + 1, budget, cursor)
        var exponents = _codec_decode(item["exponents"], payload, maximum, depth + 1, budget, cursor)
        if bi.type(codes) is not c[].array_cls or String(codes.dtype) != "<i1" or len(codes.shape) != 2:
            raise Error("codec: int8 weights require a two-dimensional int8 tensor")
        if bi.type(exponents) is not c[].array_cls or String(exponents.dtype) != "<i4" or len(exponents.shape) != 1 or Int(py=exponents.shape[0]) != Int(py=codes.shape[0]):
            raise Error("codec: int8 exponents must be int32 with one element per row")
        var result = bi.object.__new__(c[].int8_cls)
        result.codes = codes
        result.exponents = exponents
        result.shape = codes.shape
        return result
    raise Error("codec: unsupported descriptor kind " + kind)


def codec_pack(value: PythonObject, max_bytes: Int) raises -> PythonObject:
    if max_bytes < 0:
        raise Error("codec: maximum bytes must be nonnegative")
    var bi = CODEC_BOUNDARY.get_or_create_ptr()[].bi
    var payload = bi.bytearray()
    var budget = 1000000
    var descriptor = _codec_encode(value, payload, max_bytes, 0, budget)
    var result = bi.list()
    result.append(descriptor)
    result.append(bi.bytes(payload))
    return bi.tuple(result)


def codec_unpack(descriptor: PythonObject, payload: PythonObject, max_bytes: Int) raises -> PythonObject:
    var bi = CODEC_BOUNDARY.get_or_create_ptr()[].bi
    if max_bytes < 0 or len(payload) > max_bytes or (bi.type(payload) is not bi.bytes and bi.type(payload) is not bi.bytearray):
        raise Error("codec: invalid or oversized binary payload")
    var budget = 1000000
    var cursor = 0
    var result = _codec_decode(descriptor, payload, max_bytes, 0, budget, cursor)
    if cursor != len(payload):
        raise Error("codec: unclaimed bytes at the end of the payload")
    return result


def codec_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var operation = String(args[0])
        var result = PythonObject(None)
        if operation == "initialize":
            codec_initialize(args[1][0], args[1][1], args[1][2])
        elif operation == "pack":
            result = codec_pack(args[1][0], Int(py=args[1][1]))
        elif operation == "unpack":
            result = codec_unpack(args[1][0], args[1][1], Int(py=args[1][2]))
        else:
            raise Error("codec: unknown operation " + operation)
        return result.steal_data()
    except error:
        return raise_python_exception(error, ExceptionType.ValueError)
