# SPDX-License-Identifier: Apache-2.0
# Native array-like admission preserving ndarray, standard buffer, and
# rectangular scalar/array sequence inputs. Container traversal and scalar
# serialization are host ABI work; existing kernels still own model math.
from bindings.native_array_storage import native_array_from_values, native_array_zeros
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.native_buffer_owner import native_buffer_view
from bindings.arena_buffer_copy import buffer_copy_to_array
from bindings.native_array_view import native_array_ndarray_view
from bindings.py_balanced import py_call, py_type, py_get
from bindings.p3_labels_metrics_native import p3_type_word, p3_item
from bindings.cpython_table import cpython_table
from core.sequential_inputs_impl import sequence_widen_halves, sequence_narrow_halves
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


def _buffer(mut mods: NativeModules, value: PythonObject, context: PythonObject, samples: Bool) raises -> PythonObject:
    var b = mods.module("builtins")
    var view = native_buffer_view(value)
    if samples and Int(py=view.ndim) > 2 and not Bool(py=view.c_contiguous):
        raise Error("X with more than two dimensions must be C-contiguous")
    var format = view.format
    var prefix = ""
    if len(format) > 1:
        var first = String(format[0])
        if first == "@" or first == "=" or first == "<" or first == ">" or first == "!":
            if first == "<" or first == ">" or first == "!":
                prefix = ">" if first == "!" else first
            format = format[py_call(b.slice, 1, PythonObject(None))]
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
        var windowed_result17 = py_call(context["ndarray"], view.shape, dtype=dtype, buffer=view, strides=view.strides)
        mods.released()
        return windowed_result17^
    # CPython's native buffer implementation packs signed/zero strides in
    # logical C order, including read-only views. No NumPy conversion runs.
    var packed = py_call(view.tobytes, "C")
    var windowed_result16 = py_call(context["ndarray"], view.shape, dtype=dtype, buffer=packed)
    mods.released()
    return windowed_result16^


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


def _array_values(mut mods: NativeModules, value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                  mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject) raises:
    var b = mods.module("builtins")
    var view = native_buffer_view(value)
    _leaf_shape(view.shape, depth, dims, leaf_depth)
    var dtype = value.dtype
    var incoming = String(dtype.kind)
    var width = Int(py=dtype.itemsize)
    _promote(kind, size, incoming, width)
    # One native C-order copy into the owned store (was tobytes then frombytes).
    var packed = buffer_copy_to_array(String(context["buffer_codes"][Python.tuple(incoming, width)]), view)
    if not Bool(py=dtype.isnative) and width > 1:
        packed.byteswap()
    if incoming == "f" and width == 2:
        var count = len(packed)
        var wide = native_array_zeros(String("f"), count)
        sequence_widen_halves(Int(py=packed.buffer_info()[0]), Int(py=wide.buffer_info()[0]), count)
        py_call(values.extend, wide)
        _ = wide^
    else:
        py_call(values.extend, packed)
    _ = packed^


def _sequence_values(mut mods: NativeModules, value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                     mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject,
                     float_target: Bool = False) raises:
    var b = mods.module("builtins")
    # Exact builtin containers and numbers (the common nested-list input)
    # take their branch by type identity: none of them is an ndarray, NumPy
    # scalar, Array or buffer, so the isinstance chain below would reach the
    # same branch after up to ten Python calls per element.
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple or cls is b.range:
        _nested_values(mods, value, depth, dims, values, kind, size, leaf_depth, context, float_target)
        return
    if cls is b.float or cls is b.int or cls is b.bool:
        _scalar_values(mods, value, depth, dims, values, kind, size, leaf_depth, context, float_target)
        return
    # A NumPy-free context connects no ndarray: its sequences hold Python
    # numbers only, and an array leaf is refused below as a non-number.
    var numpy = py_get(context, "ndarray") is not None
    if numpy and (Bool(py=py_call(b.isinstance, value, context["ndarray"])) or Bool(py=py_call(b.isinstance, value, context["scalar_type"]))):
        _array_values(mods, value, depth, dims, values, kind, size, leaf_depth, context)
    elif numpy and py_type(value) is context["array_class"]:
        var view = py_call(context["array_fields"]["_mv"], value, context["array_class"])
        mods.released()
        var shape = py_call(context["array_fields"]["shape"], value, context["array_class"])
        mods.released()
        var dtype = py_call(context["array_fields"]["dtype"], value, context["array_class"])
        mods.released()
        var order = py_call(context["array_fields"]["order"], value, context["array_class"])
        mods.released()
        var array = py_call(context["ndarray"], shape, dtype=dtype, buffer=view, order=order)
        mods.released()
        _array_values(mods, array, depth, dims, values, kind, size, leaf_depth, context)
    elif numpy and (py_type(value) is b.memoryview or py_type(value) is context["array_type"] or py_type(value) is b.bytearray):
        var array = _buffer(mods, value, context, False)
        _array_values(mods, array, depth, dims, values, kind, size, leaf_depth, context)
    elif Bool(py=py_call(b.isinstance, value, b.list)) or Bool(py=py_call(b.isinstance, value, b.tuple)) or py_type(value) is b.range:
        _nested_values(mods, value, depth, dims, values, kind, size, leaf_depth, context, float_target)
    else:
        _scalar_values(mods, value, depth, dims, values, kind, size, leaf_depth, context, float_target)


def _nested_values(mut mods: NativeModules, value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                   mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject,
                   float_target: Bool) raises:
    var n = len(value)
    if depth >= 8 or n == 0:
        raise Error("input sequences must be nonempty and have rank at most eight")
    if leaf_depth >= 0 and depth >= leaf_depth:
        raise Error("input sequences must be rectangular")
    if depth == len(dims):
        dims.append(n)
    elif dims[depth] != n:
        raise Error("input sequences must be rectangular")
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is not b.list and cls is not b.tuple:
        # A range or a list / tuple subclass: the element protocol as before.
        for k in range(n):  # driver-loop: per ELEMENT of a range or sequence subclass input (data-sized)
            _sequence_values(mods, value[k], depth + 1, dims, values, kind, size, leaf_depth, context, float_target)
        return
    # Lane p2-labels-lists-arrays: an exact list or tuple is read borrowed
    # from its C storage. An exact float or bool leaf (and an exact int when
    # the target is float) is admitted by one class-pointer compare and
    # appended by PyList_Append: no boxed index, subscript, recursion or
    # Python append call per element. Every other element takes the general
    # admission below, unchanged.
    var is_tuple = cls is b.tuple
    var float_word = Int(b.float._obj_ptr)
    var bool_word = Int(b.bool._obj_ptr)
    var int_word = Int(b.int._obj_ptr) if float_target else 0
    ref cpy = Python().cpython()
    var t = cpython_table()
    for k in range(n):  # driver-loop: per ELEMENT of a list input (data-sized), one borrowed read each from the C storage
        var raw = p3_item(value, k, is_tuple)
        var word = p3_type_word(raw)
        if word == float_word or word == bool_word or word == int_word:
            if depth + 1 > 8:
                raise Error("input sequences must have rank at most eight")
            if leaf_depth >= 0 and leaf_depth != depth + 1:
                raise Error("input sequences must be rectangular")
            leaf_depth = depth + 1
            if word == bool_word:
                _promote(kind, size, "b", 1)
            else:
                _promote(kind, size, "f", 8)
            if Int(t[].PyList_Append(values._obj_ptr, raw)) != 0:
                raise cpy.unsafe_get_error()
        else:
            _sequence_values(mods, PythonObject(from_borrowed=raw), depth + 1, dims, values, kind, size, leaf_depth,
                             context, float_target)
    _ = value  # keep the container alive through the borrowed reads above


def _fits_int64(value: PythonObject) -> Bool:
    # Exact int only (the caller checked the type): the C conversion either
    # returns the word or raises OverflowError, which is cleared here.
    try:
        _ = Int(py=value)
        return True
    except:
        return False


def _scalar_values(mut mods: NativeModules, value: PythonObject, depth: Int, mut dims: List[Int], mut values: PythonObject,
                   mut kind: String, mut size: Int, mut leaf_depth: Int, context: PythonObject,
                   float_target: Bool) raises:
    var b = mods.module("builtins")
    _leaf_shape(Python.tuple(), depth, dims, leaf_depth)
    var cls = py_type(value)
    # Exact builtin types answer by identity; subclasses keep isinstance.
    if cls is b.bool or (cls is not b.float and cls is not b.int and Bool(py=py_call(b.isinstance, value, b.bool))):
        _promote(kind, size, "b", 1)
    elif cls is b.float or (cls is not b.int and Bool(py=py_call(b.isinstance, value, b.float))):
        _promote(kind, size, "f", 8)
    elif cls is b.int and not float_target and _fits_int64(value):
        # An exact int that converts to a signed 64-bit word is inside
        # [int64_min, int64_max]: the "i" kind, with one native conversion
        # instead of three rich comparisons against the context bounds.
        _promote(kind, size, "i", 8)
    elif cls is b.int or Bool(py=py_call(b.isinstance, value, b.int)):
        if not float_target and (Bool(py=value < context["int64_min"]) or Bool(py=value > context["uint64_max"])):
            raise Error("TypeError: input integers exceed the supported numeric array range")
        _promote(kind, size, "f" if float_target else ("u" if Bool(py=value > context["int64_max"]) else "i"), 8)
    else:
        raise Error("TypeError: inputs must contain numeric scalars")
    py_call(values.append, value)


def sequential_input(mut mods: NativeModules, value: PythonObject, context: PythonObject, samples: Bool = False) raises -> PythonObject:
    var b = mods.module("builtins")
    if Bool(py=py_call(b.isinstance, value, context["ndarray"])):
        return value
    if py_type(value) is context["array_class"]:
        # Views returned from input preparation may escape into fitted state
        # or public outputs. Array._mv can be ownerless; retain its exporter.
        return native_array_ndarray_view(value, context)
    if py_type(value) is b.memoryview or py_type(value) is context["array_type"] or py_type(value) is b.bytearray:
        return _buffer(mods, value, context, samples)
    var packed = sequential_sequence(mods, value, context)
    var dtype = context["numeric_dtypes"][Python.tuple(packed[1], packed[2])]
    var windowed_result15 = py_call(context["ndarray"], packed[0], dtype=dtype, buffer=packed[3])
    mods.released()
    return windowed_result15^


def sequential_input(value: PythonObject, context: PythonObject, samples: Bool = False) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return sequential_input(mods, value, context, samples)


def sequential_sequence(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A rectangular nest of numbers as (shape, kind, size, storage): the
    shape list, the promoted element kind letter and width, and a C-order
    `array.array` holding the elements in that type."""
    var b = mods.module("builtins")
    var dims = List[Int]()
    var values = Python.list()
    var kind = String()
    var size = 0
    var leaf_depth = -1
    _sequence_values(mods, value, 0, dims, values, kind, size, leaf_depth, context)
    var shape = Python.list()
    for k in range(len(dims)):  # driver-loop: over sequence result shape metadata
        py_call(shape.append, dims[k])
    var code = context["buffer_codes"][Python.tuple(kind, size)]
    var storage = PythonObject(None)
    if kind == "f" and size == 2:
        var count = len(values)
        var wide = native_array_from_values(String("d"), values)
        storage = native_array_zeros(String("H"), count)
        sequence_narrow_halves(Int(py=wide.buffer_info()[0]), Int(py=storage.buffer_info()[0]), count)
        _ = wide^
    else:
        storage = native_array_from_values(String(code), values)
    return Python.tuple(shape, kind, size, storage)


def sequential_sequence(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return sequential_sequence(mods, value, context)


def sequential_sequence_f32(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Array.from_list(..., '<f4') constructs array.array('f', flat) directly.
    # Preserve its Python-number -> C double -> float conversion, including
    # large integer inputs, without an int64 intermediate or a different cast.
    var dims = List[Int]()
    var values = Python.list()
    var kind = String()
    var size = 0
    var leaf_depth = -1
    _sequence_values(mods, value, 0, dims, values, kind, size, leaf_depth, context, True)
    var shape = Python.list()
    for k in range(len(dims)):  # driver-loop: over admitted sequence shape
        py_call(shape.append, dims[k])
    var storage = native_array_from_values(String("f"), values)
    return Python.tuple(shape, storage)


def sequential_sequence_f32(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return sequential_sequence_f32(mods, value, context)
