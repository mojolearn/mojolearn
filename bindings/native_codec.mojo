# SPDX-License-Identifier: Apache-2.0
"""Native structural state codec for process workers; never executes pickle.

Descriptors contain only exact builtins. Array payloads retain their dtype,
shape, physical order and bytes. Bounds are checked before allocation; tensor
ranges must cover the payload exactly in traversal order. Object identity and
arbitrary Python classes are deliberately outside this private wire format.
"""
from bindings.native_array_storage import native_array_empty
from bindings.native_loaded_module import native_loaded_module
from bindings.native_array_snapshot import native_array_snapshot, native_array_slots
from bindings.native_buffer_address import native_buffer_address
from bindings.native_instance_item import native_instance_item
from bindings.mode_metadata_native import _fields, _class_fields
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
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
    var bi = native_loaded_module("builtins")
    _ = native_array_slots(Python.dict(array_class=array_cls))
    if c[].array_cls is not None and c[].array_cls is not array_cls:
        raise Error("codec: native Array class connection cannot be replaced")
    if bf16_cls is not None:
        _codec_packed_class(bf16_cls, Python.tuple("bits", "shape"))
        if c[].bf16_cls is not None and c[].bf16_cls is not bf16_cls:
            raise Error("codec: native bfloat16 class connection cannot be replaced")
    if int8_cls is not None:
        _codec_packed_class(int8_cls, Python.tuple("codes", "exponents", "shape"))
        if c[].int8_cls is not None and c[].int8_cls is not int8_cls:
            raise Error("codec: native int8 class connection cannot be replaced")
    c[].bi = bi
    c[].arrays = native_loaded_module("array").array
    c[].array_cls = array_cls
    if bf16_cls is not None:
        c[].bf16_cls = bf16_cls
    if int8_cls is not None:
        c[].int8_cls = int8_cls


def _codec_text(value: PythonObject) raises -> String:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("codec: metadata tags require exact strings")
    return String(value)


def _codec_keys(value: PythonObject, strings_only: Bool = False) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.dict:
        raise Error("codec: metadata requires an exact dictionary")
    var keys = py_call(bi.list, value.keys())
    # Check the complete key set before any lookup. Otherwise a later foreign
    # colliding key could run __eq__ when an earlier ordinary key is read.
    for i in range(len(keys)):  # driver-loop: admit all dictionary keys before reading values
        var cls = py_type(keys[i])
        if cls is not bi.str and (strings_only or (cls is not bi.int and cls is not bi.bool)):
            raise Error("codec: dictionary keys require exact string or integer metadata")
    return keys


def _codec_packed_class(cls: PythonObject, names: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    if py_type(cls) is not bi.type:
        raise Error("codec: packed weights require ordinary connected class metadata")
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not bi.object:
        raise Error("codec: packed weight class ancestry changed")
    var row = _class_fields(cls)
    for hook in ["__getattribute__", "__getattr__", "__setattr__", "__del__"]:  # small-loop(packed weight attribute and lifetime hooks)
        if py_contains(row, hook):
            raise Error("codec: packed weights cannot invoke Python attribute or lifetime hooks")
    for i in range(len(names)):  # small-loop(packed weight consumed state field names)
        if py_contains(row, names[i]):
            raise Error("codec: packed weight fields cannot be replaced by class descriptors")
    # Refuse replacement __dict__ descriptors before allocation too: otherwise
    # cleanup of a rejected newly allocated owner could reach a Python hook.
    if py_contains(row, "__dict__"):
        var member = row["__dict__"]
        if (py_type(member) is not py_type(bi.type.__dict__["__dict__"])
                or member.__objclass__ is not cls or _codec_text(member.__name__) != "__dict__"):
            raise Error("codec: packed weights require native dictionary storage")


def _codec_packed_fields(value: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    if py_type(value) is not cls:
        raise Error("codec: packed weight requires its exact connected class")
    _codec_packed_class(cls, names)
    var fields = _fields(value)
    _ = _codec_keys(fields, True)
    return fields


def _codec_captured_bytes(pin: AdmittedBorrowedBuffer, maximum: Int) raises -> PythonObject:
    var layout = pin.layout.value().copy()
    if layout.count * layout.itemsize > maximum:
        raise Error("codec: payload exceeds its byte limit")
    var bytes = List[UInt8](length=layout.count * layout.itemsize, fill=0)
    # Same logical C-order byte sequence as memoryview.tobytes("C"). For an
    # Array the admitted _mv describes its physical store, retaining F order.
    layout.copy_to(Int(bytes.unsafe_ptr()))
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(bytes.unsafe_ptr(), len(bytes))
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


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
        code = String(py_call(PythonObject(format).lstrip, "=@<"))
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
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.int:
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
    py_setitem(item, "kind", PythonObject(kind))
    py_setitem(item, "begin", PythonObject(begin))
    py_setitem(item, "size", PythonObject(count))
    _ = py_call(payload.extend, raw)
    return item


def _codec_encode(value: PythonObject, payload: PythonObject, maximum: Int,
                  depth: Int, mut budget: Int) raises -> PythonObject:
    _codec_node(depth, budget)
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var cls = py_type(value)
    if value is None or cls is bi.bool or cls is bi.int or cls is bi.str:
        return value
    if cls is bi.float:
        if not isfinite(Float64(py=value)):
            var tagged = bi.dict()
            py_setitem(tagged, "kind", PythonObject("float"))
            py_setitem(tagged, "value", py_call(bi.str, value))
            return tagged
        return value
    if cls is bi.bytes or cls is bi.bytearray:
        return _codec_buffer(value, payload, "bytes" if cls is bi.bytes else "bytearray", maximum)
    var result = bi.dict()
    if cls is bi.list or cls is bi.tuple or cls is bi.dict:
        py_setitem(result, "kind", PythonObject("dict" if cls is bi.dict else ("tuple" if cls is bi.tuple else "list")))
        var items = bi.list()
        if cls is bi.dict:
            var keys = _codec_keys(value)
            for i in range(len(keys)):  # driver-loop: encode only after the complete key set was admitted
                var pair = bi.list()
                var key = keys[i]
                _ = py_call(pair.append, key)
                _ = py_call(pair.append, _codec_encode(value[key], payload, maximum, depth + 1, budget))
                _ = py_call(items.append, pair)
        else:
            for i in range(len(value)):  # driver-loop: over structural state sequence entries
                _ = py_call(items.append, _codec_encode(value[i], payload, maximum, depth + 1, budget))
        py_setitem(result, "items", items)
        return result
    if c[].bf16_cls is not None and cls is c[].bf16_cls:
        var fields = _codec_packed_fields(value, c[].bf16_cls, Python.tuple("bits", "shape"))
        var bits = native_instance_item(fields, PythonObject("bits"))
        py_setitem(result, "kind", PythonObject("bf16"))
        py_setitem(result, "bits", _codec_encode(bits, payload, maximum, depth + 1, budget))
        return result
    if c[].int8_cls is not None and cls is c[].int8_cls:
        var fields = _codec_packed_fields(value, c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        var codes = native_instance_item(fields, PythonObject("codes"))
        var exponents = native_instance_item(fields, PythonObject("exponents"))
        py_setitem(result, "kind", PythonObject("int8"))
        py_setitem(result, "codes", _codec_encode(codes, payload, maximum, depth + 1, budget))
        py_setitem(result, "exponents", _codec_encode(exponents, payload, maximum, depth + 1, budget))
        return result
    var native = cls is c[].array_cls
    var captured = PythonObject(None)
    var exporter = value
    if native:
        captured = native_array_snapshot(value, Python.dict(array_class=c[].array_cls), True)
        exporter = captured._mv
    else:
        # Admission precedes any buffer protocol, including creation of a
        # memoryview. The native Py_buffer keeps this exporter alive until the
        # complete tensor descriptor and payload bytes have been published.
        admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(exporter)
    try:
        var dtype = _codec_text(captured.dtype) if native else _codec_dtype(pin.format, pin.layout.value().itemsize)
        _ = _codec_code(dtype)
        var shape = Python.list()
        if native:
            shape = py_call(bi.list, captured.shape)
        else:
            for i in range(len(pin.layout.value().shape)):  # small-loop(captured native buffer dimensions)
                _ = py_call(shape.append, pin.layout.value().shape[i])
        var order = _codec_text(captured.order) if native else String("C")
        var readonly = captured._readonly if native else PythonObject(pin.layout.value().readonly)
        var raw = _codec_captured_bytes(pin, maximum - len(payload))
        result = _codec_buffer(raw, payload, "array", maximum)
        py_setitem(result, "dtype", PythonObject(dtype))
        py_setitem(result, "shape", shape)
        py_setitem(result, "order", PythonObject(order))
        py_setitem(result, "readonly", readonly)
    except error:
        pin.release()
        raise error
    pin.release()
    _ = captured
    _ = exporter
    _ = value
    return result


def _codec_slice(item: PythonObject, payload: PythonObject, maximum: Int,
                 mut cursor: Int) raises -> PythonObject:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var begin = _codec_integer(item["begin"], maximum)
    var size = _codec_integer(item["size"], maximum)
    if begin != cursor or size > len(payload) - begin:
        raise Error("codec: tensor ranges must be contiguous and inside payload")
    cursor += size
    return payload[py_call(c[].bi.slice, begin, cursor)]


def _codec_array(item: PythonObject, raw: PythonObject, maximum: Int) raises -> PythonObject:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var dtype = _codec_text(item["dtype"])
    var code = _codec_code(dtype)
    var store = native_array_empty(String(code))
    var itemsize = Int(py=store.itemsize)
    var shape = item["shape"]
    if py_type(shape) is not bi.list or len(shape) > 64:
        raise Error("codec: tensor shape must be a bounded list")
    var count = 1
    for i in range(len(shape)):  # driver-loop: over tensor dimension metadata
        var n = _codec_integer(shape[i], maximum)
        if n != 0 and count > maximum // n:
            raise Error("codec: tensor element count overflows payload bound")
        count *= n
    if count > maximum // itemsize or count * itemsize != len(raw):
        raise Error("codec: tensor shape and dtype differ from stored byte count")
    var order = _codec_text(item["order"])
    if order != "C" and order != "F" or py_type(item["readonly"]) is not bi.bool:
        raise Error("codec: invalid tensor layout metadata")
    _ = py_call(store.frombytes, raw)
    var slots = native_array_slots(Python.dict(array_class=c[].array_cls))
    var result = py_call(bi.object.__new__, c[].array_cls)
    var fields = Python.dict(_store=store, _base=PythonObject(None), _pin=PythonObject(None),
        _mv=py_call(bi.memoryview, store), _addr=PythonObject(native_buffer_address(store)),
        _readonly=item["readonly"], shape=py_call(bi.tuple, shape), dtype=PythonObject(dtype),
        ndim=PythonObject(len(shape)), size=PythonObject(count), nbytes=PythonObject(len(raw)),
        itemsize=PythonObject(itemsize), order=PythonObject(order))
    var strides = bi.list()
    var stride = itemsize
    for i in range(len(shape)):  # driver-loop: over tensor layout strides
        if order == "C":
            _ = py_call(strides.insert, 0, stride)
            var size = Int(py=shape[len(shape) - i - 1])
            if size != 0 and stride > 9223372036854775807 // size:
                raise Error("codec: tensor stride overflows native indexing")
            stride *= size
        else:
            _ = py_call(strides.append, stride)
            var size = Int(py=shape[i])
            if size != 0 and stride > 9223372036854775807 // size:
                raise Error("codec: tensor stride overflows native indexing")
            stride *= size
    py_setitem(fields, "strides", py_call(bi.tuple, strides))
    var names = py_call(bi.list, fields.keys())
    for i in range(len(names)):  # small-loop(publish admitted Array metadata through native member setters)
        _ = py_call(slots[names[i]].__set__, result, fields[names[i]])
    return result


def _codec_decode(item: PythonObject, payload: PythonObject, maximum: Int,
                  depth: Int, mut budget: Int, mut cursor: Int) raises -> PythonObject:
    _codec_node(depth, budget)
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var cls = py_type(item)
    if item is None or cls is bi.bool or cls is bi.int or cls is bi.str:
        return item
    if cls is bi.float:
        if not isfinite(Float64(py=item)):
            raise Error("codec: scalar metadata must be finite")
        return item
    if cls is not bi.dict:
        raise Error("codec: invalid structural descriptor")
    _ = _codec_keys(item, True)
    var kind = _codec_text(py_get(item, "kind"))
    if kind == "float":
        var text = _codec_text(item["value"])
        if text != "inf" and text != "-inf" and text != "nan":
            raise Error("codec: invalid nonfinite scalar tag")
        return py_call(bi.float, item["value"])
    if kind == "bytes" or kind == "bytearray" or kind == "array":
        var raw = _codec_slice(item, payload, maximum, cursor)
        if kind == "array":
            return _codec_array(item, raw, maximum)
        return py_call(bi.bytearray, raw) if kind == "bytearray" else py_call(bi.bytes, raw)
    if kind == "list" or kind == "tuple" or kind == "dict":
        var items = item["items"]
        if py_type(items) is not bi.list:
            raise Error("codec: container entries must be a list")
        var result = bi.dict() if kind == "dict" else bi.list()
        for i in range(len(items)):  # driver-loop: over structural state container entries
            if kind == "dict":
                var pair = items[i]
                if py_type(pair) is not bi.list or len(pair) != 2:
                    raise Error("codec: dictionary entries require key/value pairs")
                var key = pair[0]
                var key_cls = py_type(key)
                if key_cls is not bi.str and key_cls is not bi.int and key_cls is not bi.bool:
                    raise Error("codec: invalid dictionary key")
                if py_contains(result, key):
                    raise Error("codec: duplicate dictionary key")
                py_setitem(result, key, _codec_decode(pair[1], payload, maximum, depth + 1, budget, cursor))
            else:
                _ = py_call(result.append, _codec_decode(items[i], payload, maximum, depth + 1, budget, cursor))
        return py_call(bi.tuple, result) if kind == "tuple" else result
    if kind == "bf16":
        if c[].bf16_cls is None:
            raise Error("codec: bfloat16 weight class is not connected")
        var bits = _codec_decode(item["bits"], payload, maximum, depth + 1, budget, cursor)
        if py_type(bits) is not c[].array_cls or String(bits.dtype) != "<u2" or len(bits.shape) != 2:
            raise Error("codec: bfloat16 weights require a two-dimensional uint16 tensor")
        _codec_packed_class(c[].bf16_cls, Python.tuple("bits", "shape"))
        var result = py_call(bi.object.__new__, c[].bf16_cls)
        var fields = _codec_packed_fields(result, c[].bf16_cls, Python.tuple("bits", "shape"))
        py_setitem(fields, "bits", bits)
        py_setitem(fields, "shape", bits.shape)
        return result
    if kind == "int8":
        if c[].int8_cls is None:
            raise Error("codec: int8 weight class is not connected")
        var codes = _codec_decode(item["codes"], payload, maximum, depth + 1, budget, cursor)
        var exponents = _codec_decode(item["exponents"], payload, maximum, depth + 1, budget, cursor)
        if py_type(codes) is not c[].array_cls or String(codes.dtype) != "<i1" or len(codes.shape) != 2:
            raise Error("codec: int8 weights require a two-dimensional int8 tensor")
        if py_type(exponents) is not c[].array_cls or String(exponents.dtype) != "<i4" or len(exponents.shape) != 1 or Int(py=exponents.shape[0]) != Int(py=codes.shape[0]):
            raise Error("codec: int8 exponents must be int32 with one element per row")
        _codec_packed_class(c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        var result = py_call(bi.object.__new__, c[].int8_cls)
        var fields = _codec_packed_fields(result, c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        py_setitem(fields, "codes", codes)
        py_setitem(fields, "exponents", exponents)
        py_setitem(fields, "shape", codes.shape)
        return result
    raise Error("codec: unsupported descriptor kind " + kind)


def codec_pack(value: PythonObject, max_bytes: Int) raises -> PythonObject:
    if max_bytes < 0:
        raise Error("codec: maximum bytes must be nonnegative")
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    if c[].array_cls is None:
        raise Error("codec: native storage classes are not connected")
    var bi = c[].bi
    var payload = bi.bytearray()
    var budget = 1000000
    var descriptor = _codec_encode(value, payload, max_bytes, 0, budget)
    var result = bi.list()
    _ = py_call(result.append, descriptor)
    _ = py_call(result.append, py_call(bi.bytes, payload))
    return py_call(bi.tuple, result)


def codec_unpack(descriptor: PythonObject, payload: PythonObject, max_bytes: Int) raises -> PythonObject:
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    if c[].array_cls is None:
        raise Error("codec: native storage classes are not connected")
    var bi = c[].bi
    if max_bytes < 0 or (py_type(payload) is not bi.bytes and py_type(payload) is not bi.bytearray):
        raise Error("codec: invalid or oversized binary payload")
    if len(payload) > max_bytes:
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
        var bi = native_loaded_module("builtins")
        if py_type(args) is not bi.tuple or len(args) != 2:
            raise Error("codec: operation requires a name and native argument sequence")
        var operation = _codec_text(args[0])
        var values = args[1]
        if py_type(values) is not bi.tuple and py_type(values) is not bi.list:
            raise Error("codec: arguments require an exact tuple or list")
        var result = PythonObject(None)
        if operation == "initialize":
            if len(values) != 3:
                raise Error("codec: initialize requires three storage classes")
            codec_initialize(values[0], values[1], values[2])
        elif operation == "pack":
            if len(values) != 2:
                raise Error("codec: pack requires state and a byte limit")
            result = codec_pack(values[0], _codec_integer(values[1], 9223372036854775807))
        elif operation == "unpack":
            if len(values) != 3:
                raise Error("codec: unpack requires metadata, payload and a byte limit")
            result = codec_unpack(values[0], values[1], _codec_integer(values[2], 9223372036854775807))
        else:
            raise Error("codec: unknown operation " + operation)
        return result.steal_data()
    except error:
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
