# SPDX-License-Identifier: Apache-2.0
# Operation-local scalar decoding for manifold settings. Original settings
# have already been admitted and copied by ManifoldSettings; this cache owns
# one descriptor/storage capture of each private native scalar/array value.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_scalar_decode import NativeScalarBuffer, native_scalar_buffer_dtype
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.metrics_native_support import metric_warn
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.native_array_storage import native_array_from_bytes
from bindings.optimizer_public_words import stash_python_error
from bindings.umap_parameters_native import umap_is_string
from bindings.manifold_temporal_native import temporal_object_kind
from core.native_scalar_decode import NativeRealBits


struct ManifoldScalarSnapshot(Movable, Writable):
    var owner: PythonObject
    var dtype: PythonObject
    var shape: PythonObject
    var kind: String
    var char: String
    var count: Int
    var array: Bool
    var numeric: Bool
    var number: NativeScalarBuffer
    var item: PythonObject
    var raw: PythonObject
    var layout: String
    var little: Bool

    def __init__(out self, owner: PythonObject, dtype: PythonObject,
                 shape: PythonObject, kind: String, char: String, count: Int,
                 array: Bool, numeric: Bool, number: NativeScalarBuffer,
                 item: PythonObject, raw: PythonObject, layout: String, little: Bool):
        self.owner = owner
        self.dtype = dtype
        self.shape = shape
        self.kind = kind
        self.char = char
        self.count = count
        self.array = array
        self.numeric = numeric
        self.number = number
        self.item = item
        self.raw = raw
        self.layout = layout
        self.little = little

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ManifoldScalarSnapshot(count=", self.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def manifold_buffer_bytes(value: PythonObject) raises -> PythonObject:
    # Raw transport deliberately omits PyBUF_FORMAT: NumPy temporal/native
    # unicode storage need not have a PEP3118 spelling to copy its bytes.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var words = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, words, Int32(0x118)) != 0:
        var error = stash_python_error()
        raise error^
    var result = PythonObject(None)
    try:
        var bytes = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(Int(0), view[2])
        if not bytes:
            var error = stash_python_error()
            raise error^
        result = PythonObject(from_owned=bytes)
        var address = cpy.lib.get_function[Int]("PyBytes_AsString")(result._obj_ptr)
        if not address or cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(address, words, view[2], Int8(67)) != 0:
            var error = stash_python_error()
            raise error^
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(words)
    return result


def manifold_scalar_type(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.complex:
        return True
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.list or cls is b.tuple or cls is b.dict or value is None:
        return False
    if py_get(context["modules"], "numpy") is None:
        return False
    var classes = sequence_dtype_classes(context)
    for i in range(len(classes)):  # small-loop(dtype: fixed immutable native scalar identities)
        if cls is classes[i][1]:
            return True
    for code in ["q", "Q"]:  # small-loop(dtype: platform long-long scalar aliases)
        if cls is sequence_dtype_create(PythonObject(code), context).type:
            return True
    return False


def _snapshot(value: PythonObject, context: PythonObject,
              captured_dtype: PythonObject = PythonObject(None)) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.frozenset or value is None:
        return PythonObject(None)
    var cache = context["family_mani_scalar_cache"]
    var key = PythonObject(Int(value._obj_ptr))
    var previous = py_get(cache, key)
    if previous is not None:
        return previous
    if cls is b.bytearray or cls is b.memoryview or cls is context["array_type"]:
        var shape = value.shape if cls is b.memoryview else Python.tuple(len(value))
        var count = Int(py=shape[0]) if len(shape) > 0 else -1
        var contiguous = Bool(py=value.c_contiguous) if cls is b.memoryview else True
        var raw = manifold_buffer_bytes(value)
        var flavor = String("memoryview") if cls is b.memoryview else String("bytearray") if cls is b.bytearray else String("array")
        var item = PythonObject(None)
        ref cpy = Python().cpython()
        if cls is b.memoryview:
            # memoryview's native repr identifies the original wrapper and
            # never formats its payload. Freeze it before kernel execution;
            # invalid float text must not show a replacement bytes/view.
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(value._obj_ptr)
            if not pointer:
                var error = stash_python_error()
                raise error^
            item = PythonObject(from_owned=pointer)
        elif cls is b.bytearray:
            var address = cpy.lib.get_function[Int]("PyBytes_AsString")(raw._obj_ptr)
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyByteArray_FromStringAndSize")(address, len(raw))
            if not pointer:
                var error = stash_python_error()
                raise error^
            item = PythonObject(from_owned=pointer)
        else:
            item = native_array_from_bytes(String(value.typecode), raw)
        var number = NativeScalarBuffer(NativeRealBits(), NativeRealBits(), False, False)
        var result = PythonObject(alloc=ManifoldScalarSnapshot(value, PythonObject(None), shape,
            "buffer", flavor, count, False, False, number, item, raw,
            String("C") if contiguous else String("strided"), True))
        py_setitem(cache, key, result)
        return result
    var array = cls is py_get(context, "ndarray")
    if py_get(context["modules"], "numpy") is None:
        return PythonObject(None)
    var classes = sequence_dtype_classes(context)
    var admitted = array
    for i in range(len(classes)):  # small-loop(dtype: fixed native scalar type identities)
        admitted = admitted or cls is classes[i][1]
    for code in ["q", "Q"]:  # small-loop(dtype: platform long-long scalar aliases)
        admitted = admitted or cls is sequence_dtype_create(PythonObject(code), context).type
    if not admitted:
        return PythonObject(None)
    var dtype = value.dtype if captured_dtype is None else captured_dtype
    var native_dtype = sequence_dtype_instance(dtype, classes)
    for code in ["q", "Q"]:  # small-loop(dtype: platform long-long descriptor aliases)
        native_dtype = native_dtype or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
    if not native_dtype:
        raise Error("TypeError: manifold scalar arrays require a built-in native dtype")
    var shape = value.shape if array else Python.tuple()
    var count = 1
    for i in range(len(shape)):  # small-loop(rank: private scalar-array shape captured once)
        count *= Int(py=shape[i])
    var kind = String(dtype.kind)
    var char = String("")
    var numeric = kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c"
    var number = NativeScalarBuffer(NativeRealBits(), NativeRealBits(), False, False)
    var item = PythonObject(None)
    var raw = PythonObject(None)
    var layout = String("")
    var little = True
    var retained = value
    if count == 1:
        if numeric:
            number = native_scalar_buffer_dtype(value, dtype)
            # The decoder already captured dtype.char. Only extended complex
            # needs a scalar-conversion distinction; derive it from decoded
            # metadata instead of rereading the original descriptor.
            if kind == "c" and number.longdouble:
                char = "G"
        elif kind == "M" or kind == "m" or kind == "V":
            if kind == "V" and Bool(py=dtype.hasobject):
                raise Error("TypeError: manifold configuration refuses structured object arrays")
            raw = manifold_buffer_bytes(value)
            layout = String(dtype.str)
            little = String(dtype.byteorder) != ">"
            item = dtype.names if kind == "V" else PythonObject(None)
        elif kind == "S" and not array:
            raw = manifold_buffer_bytes(value)
        elif array:
            var indexes = Python.list()
            for i in range(len(shape)):  # small-loop(rank: one native singleton index per captured axis)
                _ = py_call(indexes.append, 0)
            var index = py_call(b.tuple, indexes)
            ref cpy = Python().cpython()
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, index._obj_ptr)
            if not pointer:
                var error = stash_python_error()
                raise error^
            item = PythonObject(from_owned=pointer)
    var result = PythonObject(alloc=ManifoldScalarSnapshot(retained, dtype, shape, kind, char,
                                                          count, array, numeric, number, item,
                                                          raw, layout, little))
    py_setitem(cache, key, result)
    return result


def _builtin_number(value: PythonObject, integer: Bool) raises -> PythonObject:
    # CPython is transport/parser for exact builtin numbers and native text,
    # never a NumPy numeric slot or an application conversion protocol.
    # Remaining shared-parser debt: CPython3.13 longobject.c can invoke
    # _pylong.int_from_string for >6000 decimal digits when the interpreter's
    # configured digit limit permits them. Preserve that existing input
    # contract here; this increment does not claim the text parser is fully
    # native. The retained source is cpython-3.13-longobject.c in lane evidence.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.complex and cls is not b.str and cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview and cls is not b.list and cls is not b.tuple and cls is not b.dict and value is not None:
        raise Error("TypeError: manifold scalar settings require native numbers or text; Python conversion protocols cannot execute")
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long" if integer else "PyNumber_Float")(value._obj_ptr)
    if not pointer:
        var error = stash_python_error()
        raise error^
    return PythonObject(from_owned=pointer)


def _buffer_number(state: ManifoldScalarSnapshot, integer: Bool) raises -> PythonObject:
    # CPython's int parser normalizes all three buffer wrappers to bytes for
    # invalid-literal errors (Objects/abstract.c PyNumber_Long). Its float
    # parser retains the wrapper in its error, so bytearray/array use a private
    # instance of the original exact native type. memoryview error repr is the
    # captured identity-only word, independent of mutable buffer contents.
    if integer:
        return _builtin_number(state.raw, True)
    ref cpy = Python().cpython()
    var input = state.raw if state.char == "memoryview" else state.item
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Float")(input._obj_ptr)
    if pointer:
        return PythonObject(from_owned=pointer)
    if state.char == "memoryview":
        var b = native_loaded_module("builtins")
        if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(b.ValueError._obj_ptr) != 0:
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            var message = String("could not convert string to float: ") + String(state.item)
            cpy.PyErr_SetString(b.ValueError._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var error = stash_python_error()
    raise error^


def _temporal_word(state: ManifoldScalarSnapshot) raises -> Int64:
    if len(state.raw) != 8:
        raise Error("TypeError: manifold temporal scalar storage must contain one 64-bit word")
    ref cpy = Python().cpython()
    var address = cpy.lib.get_function[Int]("PyBytes_AsString")(state.raw._obj_ptr)
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var result = UInt64(0)
    for i in range(8):  # small-loop(itemsize: one admitted temporal storage word)
        result |= UInt64(pointer[i if state.little else 7-i]) << UInt64(8*i)
    return Int64(result)


def manifold_number(value: PythonObject, integer: Bool, context: PythonObject,
                    depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: manifold scalar settings exceed native nesting bounds")
    var captured = _snapshot(value, context)
    if captured is None:
        return _builtin_number(value, integer)
    ref state = captured.downcast_value_ptr[ManifoldScalarSnapshot]()[]
    if state.kind == "buffer":
        if state.layout == "C":
            return _buffer_number(state, integer)
        # CPython's scalar text parsers refuse a strided memoryview when
        # acquiring a simple contiguous buffer, before reading its payload.
        # Retain that native error rather than enabling a packed input.
        return _builtin_number(state.owner, integer)
    if state.array:
        if len(state.shape) > 0 and Int(py=context["family_mani_numpy_release"]) >= 204:
            raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
        if state.count != 1:
            raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
        if len(state.shape) > 0:
            metric_warn("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)", native_loaded_module("builtins").DeprecationWarning, context)
    if state.numeric:
        if state.kind == "c":
            # F/D ndarray item conversion boxes builtin complex and refuses
            # both parsers. G retains the extended native scalar and follows
            # NumPy's scalar warning/real-part conversion for int and float.
            if state.array and state.char != "G":
                var name = String("int") if integer else String("float")
                var bytes = String(", a bytes-like object") if integer else String("")
                raise Error("TypeError: " + name + "() argument must be a string" + bytes + " or a real number, not 'complex'")
            var exceptions = py_get(context["modules"], "numpy.exceptions")
            if exceptions is None or py_type(exceptions) is not context["module_type"]:
                raise Error("TypeError: complex scalar settings require native NumPy exception metadata")
            var category = py_get(native_numpy_module_fields(exceptions), "ComplexWarning")
            metric_warn("Casting complex values to real discards the imaginary part", category, context)
        return state.number.to_python_number(integer)
    if state.kind == "M" or state.kind == "m":
        var word = _temporal_word(state)
        var kind = temporal_object_kind(word, state.layout, state.kind == "m")
        if kind == 0:
            return _builtin_number(PythonObject(word), integer)
        var name = String("NoneType") if kind == 1 else String("datetime.date") if kind == 2 else String("datetime.datetime") if kind == 3 else String("datetime.timedelta")
        var operation = String("int") if integer else String("float")
        var bytes = String(", a bytes-like object") if integer else String("")
        raise Error("TypeError: " + operation + "() argument must be a string" + bytes + " or a real number, not '" + name + "'")
    if state.kind == "V":
        if state.item is not None:
            return _builtin_number(Python.tuple(), integer)
        return _builtin_number(state.raw, integer)
    if state.array:
        return manifold_number(state.item, integer, context, depth + 1)
    # Invoke the fixed CPython text parser directly, bypassing NumPy number
    # slots. Keep the admitted immutable scalar as the diagnostic owner:
    # float's invalid-literal error formats that object, and replacing it by
    # a builtin str/bytes would change NumPy2's scalar repr in the message.
    ref cpy = Python().cpython()
    if umap_is_string(value):
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyLong_FromUnicodeObject")(value._obj_ptr, Int32(10)) if integer else cpy.lib.get_function[PyObjectPtr]("PyFloat_FromString")(value._obj_ptr)
        if not pointer:
            var error = stash_python_error()
            raise error^
        return PythonObject(from_owned=pointer)
    if state.kind == "S":
        if integer:
            # PyNumber_Long normalizes bytes subclasses to builtin bytes in
            # invalid-literal messages, unlike PyFloat_FromString.
            return _builtin_number(state.raw, True)
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyFloat_FromString")(value._obj_ptr)
        if not pointer:
            var error = stash_python_error()
            raise error^
        return PythonObject(from_owned=pointer)
    raise Error("TypeError: manifold scalar settings require native numeric or text storage")

