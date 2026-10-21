# SPDX-License-Identifier: Apache-2.0
# Native array-like admission preserving ndarray, standard buffer, and
# rectangular scalar/array sequence inputs. Container traversal and scalar
# serialization are host ABI work; existing kernels still own model math.
from std.memory import bitcast
from std.python import Python, PythonObject
from gemm.f16_widen import f16_bits_to_f32


def _promote(mut kind: String, mut size: Int, incoming: String, width: Int) raises:
    if (incoming != "b" and incoming != "i" and incoming != "u" and incoming != "f") or (
        incoming == "f" and width != 2 and width != 4 and width != 8
    ) or (incoming != "f" and width != 1 and width != 2 and width != 4 and width != 8):
        raise Error("TypeError: inputs must contain supported numeric scalars")
    if size == 0 or kind == "b":
        kind = incoming
        size = width
    elif incoming == "b":
        return
    elif kind == incoming:
        size = max(size, width)
    elif kind == "f" or incoming == "f":
        var integer_width = width if kind == "f" else size
        var float_width = size if kind == "f" else width
        # Numeric dtype promotion: half holds 8-bit integers, single holds
        # 16-bit integers; 32/64-bit integer mixtures use double precision.
        size = max(float_width, 2 if integer_width <= 1 else (4 if integer_width <= 2 else 8))
        kind = "f"
    else:
        var signed_width = size if kind == "i" else width
        var unsigned_width = size if kind == "u" else width
        if signed_width > unsigned_width:
            kind = "i"
            size = signed_width
        elif unsigned_width < 8:
            kind = "i"
            size = unsigned_width * 2
        else:
            kind = "f"
            size = 8


def _buffer(value: PythonObject, context: PythonObject, samples: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var view = b.memoryview(value)
    if samples and Int(py=view.ndim) > 2 and not Bool(py=view.c_contiguous):
        raise Error("X with more than two dimensions must be C-contiguous")
    var format = view.format
    var prefix = ""
    if len(format) > 1:
        var first = String(format[0])
        if first == "@" or first == "=" or first == "<" or first == ">" or first == "!":
            if first == "<" or first == ">" or first == "!":
                prefix = ">" if first == "!" else first
            format = format[b.slice(1, PythonObject(None))]
    var code = String(format)
    var kind = String()
    if code == "?":
        kind = "b"
    elif code == "e" or code == "f" or code == "d":
        kind = "f"
    elif code == "b" or code == "h" or code == "i" or code == "l" or code == "q" or code == "n":
        kind = "i"
    elif code == "B" or code == "H" or code == "I" or code == "L" or code == "Q" or code == "N":
        kind = "u"
    else:
        raise Error("TypeError: inputs must expose a numeric buffer")
    var size = Int(py=view.itemsize)
    var dummy_kind = String()
    var dummy_size = 0
    _promote(dummy_kind, dummy_size, kind, size)
    var dtype = context["numeric_dtypes"][Python.tuple(kind, size)]
    if prefix.byte_length() > 0 and kind != "b":
        dtype = PythonObject(prefix + kind + String(size))
    if Bool(py=view.contiguous):
        # Both C and Fortran contiguous buffers are borrowed without a copy.
        # ndarray's C type owns the memoryview and preserves read-only status.
        return context["ndarray"](view.shape, dtype=dtype, buffer=view, strides=view.strides)
    # CPython's native buffer implementation packs signed/zero strides in
    # logical C order, including read-only views. No NumPy conversion runs.
    var packed = view.tobytes("C")
    return context["ndarray"](view.shape, dtype=dtype, buffer=packed)


def _leaf_shape(shape: PythonObject, depth: Int, mut dims: List[Int], mut leaf_depth: Int) raises:
    var rank = len(shape)
    if depth + rank > 8:
        raise Error("input sequences must have rank at most eight")
    if leaf_depth >= 0 and leaf_depth != depth + rank:
        raise Error("input sequences must be rectangular")
    leaf_depth = depth + rank
    for j in range(rank):  # driver-loop: over nested native-array rank metadata
        var count = Int(py=shape[j])
        if count == 0:
            raise Error("input sequences must be nonempty")
        if depth + j == len(dims):
            dims.append(count)
        elif dims[depth + j] != count:
            raise Error("input sequences must be rectangular")
    if len(dims) != leaf_depth:
        raise Error("input sequences must be rectangular")


def _array_values(value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                  mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var view = b.memoryview(value)
    _leaf_shape(view.shape, depth, dims, leaf_depth)
    var dtype = value.dtype
    var incoming = String(dtype.kind)
    var width = Int(py=dtype.itemsize)
    _promote(kind, size, incoming, width)
    var packed = context["array_type"](context["buffer_codes"][Python.tuple(incoming, width)])
    packed.frombytes(view.tobytes("C"))
    if not Bool(py=dtype.isnative) and width > 1:
        packed.byteswap()
    for k in range(len(packed)):  # native input marshaling: packed buffer scalars become sequence storage
        if incoming == "f" and width == 2:
            values.append(PythonObject(f16_bits_to_f32(UInt16(Int(py=packed[k])))))
        else:
            values.append(packed[k])


def _sequence_values(value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                     mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(value, context["ndarray"])) or Bool(py=b.isinstance(value, context["scalar_type"])):
        _array_values(value, depth, dims, values, kind, size, leaf_depth, context)
    elif b.type(value) is context["array_class"]:
        var view = context["array_fields"]["_mv"](value, context["array_class"])
        var shape = context["array_fields"]["shape"](value, context["array_class"])
        var dtype = context["array_fields"]["dtype"](value, context["array_class"])
        var order = context["array_fields"]["order"](value, context["array_class"])
        var array = context["ndarray"](shape, dtype=dtype, buffer=view, order=order)
        _array_values(array, depth, dims, values, kind, size, leaf_depth, context)
    elif b.type(value) is b.memoryview or b.type(value) is context["array_type"] or b.type(value) is b.bytearray:
        var array = _buffer(value, context, False)
        _array_values(array, depth, dims, values, kind, size, leaf_depth, context)
    elif Bool(py=b.isinstance(value, b.list)) or Bool(py=b.isinstance(value, b.tuple)) or b.type(value) is b.range:
        var n = len(value)
        if depth >= 8 or n == 0:
            raise Error("input sequences must be nonempty and have rank at most eight")
        if leaf_depth >= 0 and depth >= leaf_depth:
            raise Error("input sequences must be rectangular")
        if depth == len(dims):
            dims.append(n)
        elif dims[depth] != n:
            raise Error("input sequences must be rectangular")
        for k in range(n):  # native input marshaling: nested scalar/container values become native storage
            _sequence_values(value[k], depth + 1, dims, values, kind, size, leaf_depth, context)
    else:
        _leaf_shape(Python.tuple(), depth, dims, leaf_depth)
        if Bool(py=b.isinstance(value, b.bool)):
            _promote(kind, size, "b", 1)
        elif Bool(py=b.isinstance(value, b.float)):
            _promote(kind, size, "f", 8)
        elif Bool(py=b.isinstance(value, b.int)):
            if Bool(py=value < context["int64_min"]) or Bool(py=value > context["uint64_max"]):
                raise Error("TypeError: input integers exceed the supported numeric array range")
            _promote(kind, size, "u" if Bool(py=value > context["int64_max"]) else "i", 8)
        else:
            raise Error("TypeError: inputs must contain numeric scalars")
        values.append(value)


def sequential_input(value: PythonObject, context: PythonObject, samples: Bool = False) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(value, context["ndarray"])):
        return value
    if b.type(value) is context["array_class"]:
        var view = context["array_fields"]["_mv"](value, context["array_class"])
        var shape = context["array_fields"]["shape"](value, context["array_class"])
        var dtype = context["array_fields"]["dtype"](value, context["array_class"])
        var order = context["array_fields"]["order"](value, context["array_class"])
        return context["ndarray"](shape, dtype=dtype, buffer=view, order=order)
    if b.type(value) is b.memoryview or b.type(value) is context["array_type"] or b.type(value) is b.bytearray:
        return _buffer(value, context, samples)
    var dims = List[Int]()
    var values = Python.list()
    var kind = String()
    var size = 0
    var leaf_depth = -1
    _sequence_values(value, 0, dims, values, kind, size, leaf_depth, context)
    var shape = Python.list()
    for k in range(len(dims)):  # driver-loop: over sequence result shape metadata
        shape.append(dims[k])
    var dtype = context["numeric_dtypes"][Python.tuple(kind, size)]
    var code = context["buffer_codes"][Python.tuple(kind, size)]
    var storage = PythonObject(None)
    if kind == "f" and size == 2:
        storage = context["array_type"]("H")
        for k in range(len(values)):  # native input marshaling: exact half scalars are serialized as half words
            storage.append(PythonObject(bitcast[DType.uint16](Float16(Float64(py=values[k])))))
    else:
        storage = context["array_type"](code, values)
    return context["ndarray"](shape, dtype=dtype, buffer=storage)
