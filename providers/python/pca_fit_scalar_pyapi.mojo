# SPDX-License-Identifier: Apache-2.0
# Mutable scalar-array settings are captured before input preparation can
# release the GIL. Conversion, warnings and failures stay at their historical
# validation point. These native records never escape through the Python ABI.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_get, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.sequential_admission import sequential_admit_value
from bindings.native_scalar_decode import NativeScalarBuffer, native_scalar_buffer_dtype
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native import sequence_array_release
from bindings.training_scalar_native import training_scalar_warning
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, stash_python_error, _restore_python_error
from bindings.manifold_temporal_native import temporal_object_kind
from core.native_scalar_decode import NativeRealBits


def _pca_scalar_bytes(value: PythonObject) raises -> PythonObject:
    # Public native buffer transport without requesting PEP3118 format:
    # datetime/timedelta and Unicode need not expose a format spelling.
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


def _pca_builtin_number(value: PythonObject, integer: Bool, context: PythonObject) raises -> PythonObject:
    # Only already-admitted native builtins/buffer wrappers reach this
    # parser. NumPy temporal/numeric conversion slots never do.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is not None and cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.complex and cls is not b.str and cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview and cls is not b.list and cls is not b.tuple and cls is not b.dict and cls is not b.set and cls is not b.frozenset and cls is not context["array_type"]:
        raise Error("TypeError: PCA scalar conversion requires native numbers or text without Python conversion hooks")
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long" if integer else "PyNumber_Float")(value._obj_ptr)
    if not pointer:
        var error = stash_python_error()
        raise error^
    return PythonObject(from_owned=pointer)


def pca_temporal_word(value: PythonObject, dtype: PythonObject) raises -> Int64:
    if Int(py=dtype.itemsize) != 8:
        raise Error("TypeError: PCA temporal scalar storage must contain one 64-bit word")
    var raw = _pca_scalar_bytes(value)
    if len(raw) != 8:
        raise Error("TypeError: PCA temporal scalar storage must contain one 64-bit word")
    var little = String(dtype.byteorder) != ">"
    ref cpy = Python().cpython()
    var address = cpy.lib.get_function[Int]("PyBytes_AsString")(raw._obj_ptr)
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var word = UInt64(0)
    for index in range(8):  # small-loop(one admitted temporal storage word)
        word |= UInt64(pointer[index if little else 7-index]) << UInt64(index * 8)
    _ = raw^
    return Int64(word)


struct PCAFitScalarError(Copyable, Movable):
    var pending: Bool
    var has_value: Bool
    var has_traceback: Bool
    var kind: PythonObject
    var value: PythonObject
    var traceback: PythonObject

    def __init__(out self):
        self.pending = False
        self.has_value = False
        self.has_traceback = False
        self.kind = PythonObject(None)
        self.value = PythonObject(None)
        self.traceback = PythonObject(None)

    def capture(mut self):
        # Same public CPython ownership pattern as ByteCheckpointIOError:
        # Fetch transfers references; native PythonObject fields retain them
        # while input preparation runs and on every discarded-error path.
        ref cpy = Python().cpython()
        var triple = cpy.PyErr_FetchTriple()
        self.pending = True
        self.has_value = Bool(triple[1])
        self.has_traceback = Bool(triple[2])
        self.kind = PythonObject(from_owned=triple[0]) if triple[0] else PythonObject(None)
        self.value = PythonObject(from_owned=triple[1]) if triple[1] else PythonObject(None)
        self.traceback = PythonObject(from_owned=triple[2]) if triple[2] else PythonObject(None)

    def raise_pending(self) raises:
        var kind = self.kind
        var value = self.value
        var traceback = self.traceback
        var value_ptr = value.steal_data() if self.has_value else PyObjectPtr()
        var traceback_ptr = traceback.steal_data() if self.has_traceback else PyObjectPtr()
        ref cpy = Python().cpython()
        cpy.PyErr_Restore(kind.steal_data(), value_ptr, traceback_ptr)
        var error = stash_python_error()
        raise error^


def pca_fit_scalar_admit(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    var scalar = context["pca_fit_scalar_context"]
    var classes = sequence_dtype_classes(scalar)
    for index in range(len(classes)):  # driver-loop: all exact native scalar classes supported by the dedicated parser
        if cls is classes[index][1]:
            return
    for code in ["q", "Q"]:  # small-loop(two native long-long scalar aliases)
        if cls is sequence_dtype_create(PythonObject(code), scalar).type:
            return
    sequential_admit_value(value, context)


struct PCAFitScalarArray(Copyable, Movable):
    var ndim: Int
    var count: Int
    var kind: String
    var char: String

    def __init__(out self, ndim: Int, count: Int, kind: String, char: String):
        self.ndim = ndim
        self.count = count
        self.kind = kind
        self.char = char


struct PCAFitScalar(Copyable, Movable):
    var source: PythonObject
    var owners: PythonObject
    var value: PythonObject
    var arrays: List[PCAFitScalarArray]
    var number: NativeScalarBuffer
    var numeric: Bool
    var refusal: String
    var release: Int
    var temporal: Bool
    var temporal_value: Int64
    var temporal_layout: String
    var temporal_delta: Bool
    var pending_error: PCAFitScalarError
    var scalar_complex: Bool
    var truth_ready: Bool
    var truth_value: Bool

    def __init__(out self, source: PythonObject, context: PythonObject, truth_mode: Bool = False,
                 operation: String = "", depth: Int = 0) raises:
        self.source = source
        self.owners = Python.list(source)
        self.value = source
        self.arrays = List[PCAFitScalarArray]()
        self.number = NativeScalarBuffer(NativeRealBits(), NativeRealBits(), False, False)
        self.numeric = False
        self.refusal = String("")
        self.release = 0
        self.temporal = False
        self.temporal_value = 0
        self.temporal_layout = String("")
        self.temporal_delta = False
        self.pending_error = PCAFitScalarError()
        self.scalar_complex = False
        self.truth_ready = False
        self.truth_value = False
        var scalar_context = context["pca_fit_scalar_context"]
        var b = native_loaded_module("builtins")
        var seen_arrays = Python.list()
        for level in range(66 - depth):  # driver-loop: admitted nested object scalar arrays, same 64-level bound as scalar parsing
            var value = self.value
            var cls = py_type(value)
            if cls is b.int or cls is b.float or cls is b.bool or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.frozenset or value is None:
                # Truth of exact containers depends only on their length;
                # their elements do not participate in conversion protocols.
                if truth_mode:
                    self.truth_ready = True
                    if value is None:
                        self.truth_value = False
                    elif cls is b.int or cls is b.float or cls is b.bool:
                        self.truth_value = Bool(py=value)
                    elif cls is b.complex:
                        ref cpy = Python().cpython()
                        self.truth_value = cpy.lib.get_function[Float64]("PyComplex_RealAsDouble")(value._obj_ptr) != 0 or cpy.lib.get_function[Float64]("PyComplex_ImagAsDouble")(value._obj_ptr) != 0
                    else:
                        self.truth_value = len(value) != 0
                break
            if truth_mode and (cls is b.bytearray or cls is b.memoryview or cls is context["array_type"]):
                sequential_admit_value(value, context)
                self.truth_ready = True
                self.truth_value = len(value) != 0
                break
            if cls is b.bytearray:
                # Preserve the builtin parser's bytearray error spelling too.
                self.value = py_call(b.bytearray, value)
                break
            if cls is b.memoryview:
                sequential_admit_value(value, context)
                if Bool(py=value.c_contiguous):
                    self.value = py_call(b.memoryview, _pca_scalar_bytes(value))
                else:
                    # int/float require a contiguous text buffer. Preserve a
                    # strided native view for their existing refusal without
                    # retaining mutable source bytes for later conversion.
                    var storage = py_call(b.bytes, 4)
                    var view = py_call(b.memoryview, storage)
                    self.value = py_call(view.__getitem__, py_call(b.slice, 0, 4, 2))
                break
            if cls is context["array_type"]:
                # Retain the exact native buffer wrapper/typecode, including
                # its error repr, instead of exposing the private bytes copy.
                var code = value.typecode
                var raw = _pca_scalar_bytes(value)
                self.value = py_call(context["array_type"], code, raw)
                break
            if cls is not context["ndarray"]:
                var classes = sequence_dtype_classes(scalar_context)
                var native_scalar = False
                for index in range(len(classes)):  # driver-loop: exact native scalar identities before any dtype access
                    native_scalar = native_scalar or cls is classes[index][1]
                for code in ["q", "Q"]:  # small-loop(two native long-long scalar aliases)
                    native_scalar = native_scalar or cls is sequence_dtype_create(PythonObject(code), scalar_context).type
                if not native_scalar or cls is b.object:
                    sequential_admit_value(value, context)
                    break
                var dtype = value.dtype
                var kind = String(dtype.kind)
                _ = py_call(self.owners.append, dtype)
                if kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c":
                    self.number = native_scalar_buffer_dtype(value, dtype)
                    self.numeric = True
                    self.scalar_complex = kind == "c"
                elif kind == "M" or kind == "m":
                    self.temporal = True
                    self.temporal_delta = kind == "m"
                    self.temporal_layout = String(dtype.str)
                    self.temporal_value = pca_temporal_word(value, dtype)
                elif kind == "V":
                    if dtype.fields is not None:
                        if truth_mode:
                            self.truth_ready = True
                            self.truth_value = _pca_structured_truth(value, dtype, operation, context, depth + level + 1)
                        self.value = Python.tuple()
                    else:
                        self.value = _pca_scalar_bytes(value)
                        if truth_mode:
                            self.truth_ready = True
                            self.truth_value = _pca_bytes_truth(self.value)
                elif kind == "U":
                    ref cpy = Python().cpython()
                    var text = cpy.lib.get_function[PyObjectPtr]("PyUnicode_FromObject")(value._obj_ptr)
                    if not text:
                        self.pending_error.capture()
                    else:
                        self.value = PythonObject(from_owned=text)
                        self.truth_ready = True
                        self.truth_value = len(self.value) != 0
                elif kind == "S":
                    self.value = _pca_scalar_bytes(value)
                    self.truth_ready = True
                    self.truth_value = len(value) != 0
                else:
                    self.refusal = "TypeError: PCA scalar settings require native numeric or text storage"
                break
            var repeated = False
            for index in range(len(seen_arrays)):  # driver-loop: avoid reacquiring metadata for cyclic object scalar arrays
                repeated = repeated or seen_arrays[index] is value
            if repeated or level + depth == 65:
                self.refusal = "TypeError: scalar configuration exceeds native nesting; supply a native number"
                break
            _ = py_call(seen_arrays.append, value)
            if self.release == 0:
                self.release = sequence_array_release(scalar_context)
            var classes = sequence_dtype_classes(scalar_context)
            var dtype = value.dtype
            var native_dtype = sequence_dtype_instance(dtype, classes)
            for code in ["q", "Q"]:  # small-loop(two native long-long descriptor aliases)
                native_dtype = native_dtype or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), scalar_context))
            if not native_dtype:
                self.refusal = "TypeError: scalar arrays require a built-in native dtype"
                break
            var shape = value.shape
            var count = 1
            for axis in range(len(shape)):  # driver-loop: scalar-array shape captured once, no later ndim/size reread
                count *= Int(py=shape[axis])
            var kind = String(dtype.kind)
            var char = String(dtype.char)
            self.arrays.append(PCAFitScalarArray(len(shape), count, kind, char))
            _ = py_call(self.owners.append, Python.tuple(value, dtype, shape))
            if count != 1:
                break
            try:
                if kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c":
                    self.number = native_scalar_buffer_dtype(value, dtype)
                    self.numeric = True
                    break
                if kind == "M" or kind == "m":
                    self.temporal = True
                    self.temporal_delta = kind == "m"
                    self.temporal_layout = String(dtype.str)
                    self.temporal_value = pca_temporal_word(value, dtype)
                    break
                if kind == "V":
                    if dtype.fields is not None:
                        if truth_mode:
                            # A native void scalar view supplies the field
                            # dictionary; recursive captures freeze its values
                            # before public input preparation starts.
                            var indexes = Python.list()
                            for axis in range(len(shape)):  # driver-loop: one native singleton index per captured axis
                                _ = py_call(indexes.append, 0)
                            var index = py_call(b.tuple, indexes)
                            ref cpy = Python().cpython()
                            var item = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, index._obj_ptr)
                            if not item:
                                self.pending_error.capture()
                            else:
                                var scalar = PythonObject(from_owned=item)
                                self.truth_ready = True
                                self.truth_value = _pca_structured_truth(scalar, dtype, operation, context, depth + level + 1)
                        self.value = Python.tuple()
                    else:
                        self.value = _pca_scalar_bytes(value)
                        if truth_mode:
                            self.truth_ready = True
                            self.truth_value = _pca_bytes_truth(self.value)
                    break
                # Exact ndarray and native dtype: this C API boxes one
                # element without invoking its numeric conversion slots. Hold
                # the result separately from a mutable object's original cell.
                var indexes = Python.list()
                for axis in range(len(shape)):  # driver-loop: one zero index per captured singleton dimension
                    _ = py_call(indexes.append, 0)
                var index = py_call(b.tuple, indexes)
                ref cpy = Python().cpython()
                var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, index._obj_ptr)
                if not pointer:
                    self.pending_error.capture()
                    break
                self.value = PythonObject(from_owned=pointer)
                _ = py_call(self.owners.append, self.value)
            except error:
                # Take a pending native exception out of the ABI transport
                # immediately; the private record owns its exact category,
                # value and traceback until its original conversion stage.
                if String(error) == PYTHON_ERROR_PENDING:
                    _ = _restore_python_error()
                    self.pending_error.capture()
                else:
                    self.refusal = String(error)
                break

    def convert(self, operation: String, context: PythonObject, integer: Bool) raises -> PythonObject:
        var scalar = context["pca_fit_scalar_context"]
        var b = native_loaded_module("builtins")
        for index in range(len(self.arrays)):  # driver-loop: replay outer-to-inner scalar-array validation without input metadata reads
            ref array = self.arrays[index]
            if array.ndim > 0 and self.release >= 204:
                raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
            if array.count != 1:
                raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
            if array.ndim > 0:
                training_scalar_warning("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)",
                                        b.DeprecationWarning, operation, scalar)
            if array.kind == "c":
                if integer or array.char != "G":
                    var name = String("int") if integer else String("float")
                    var bytes = String(", a bytes-like object") if integer else String("")
                    raise Error("TypeError: " + name + "() argument must be a string" + bytes + " or a real number, not 'complex'")
                var exceptions = py_get(scalar["modules"], "numpy.exceptions")
                if exceptions is None or py_type(exceptions) is not scalar["module_type"]:
                    raise Error("TypeError: complex scalar warnings require NumPy's native exception metadata")
                var category = py_get(native_numpy_module_fields(exceptions), "ComplexWarning")
                training_scalar_warning("Casting complex values to real discards the imaginary part", category, operation, scalar)
        if self.pending_error.pending:
            self.pending_error.raise_pending()
        if self.refusal.byte_length() > 0:
            raise Error(self.refusal)
        if self.numeric:
            if self.scalar_complex:
                var exceptions = py_get(scalar["modules"], "numpy.exceptions")
                if exceptions is None or py_type(exceptions) is not scalar["module_type"]:
                    raise Error("TypeError: complex scalar warnings require NumPy's native exception metadata")
                training_scalar_warning("Casting complex values to real discards the imaginary part", py_get(native_numpy_module_fields(exceptions), "ComplexWarning"), operation, scalar)
            return self.number.to_python_number(integer)
        if self.temporal:
            var kind = temporal_object_kind(self.temporal_value, self.temporal_layout, self.temporal_delta)
            if kind == 0:
                return _pca_builtin_number(PythonObject(self.temporal_value), integer, context)
            var name = String("NoneType") if kind == 1 else String("datetime.date") if kind == 2 else String("datetime.datetime") if kind == 3 else String("datetime.timedelta")
            var function = String("int") if integer else String("float")
            var bytes = String(", a bytes-like object") if integer else String("")
            raise Error("TypeError: " + function + "() argument must be a string" + bytes + " or a real number, not '" + name + "'")
        return _pca_builtin_number(self.value, integer, context)


    def truth(self, operation: String, context: PythonObject) raises -> Bool:
        var b = native_loaded_module("builtins")
        for index in range(len(self.arrays)):  # driver-loop: admitted nested scalar-array truth rules, independent of rank
            var count = self.arrays[index].count
            if count > 1:
                raise Error("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            if count == 0:
                if self.release >= 202:
                    raise Error("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
                training_scalar_warning("The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.", b.DeprecationWarning, operation, context["pca_fit_scalar_context"])
                return False
        if self.pending_error.pending:
            self.pending_error.raise_pending()
        if self.refusal.byte_length() > 0:
            raise Error(self.refusal)
        if self.numeric:
            return self.number.truth()
        if self.temporal:
            return self.temporal_value != 0
        if self.truth_ready:
            return self.truth_value
        raise Error("TypeError: PCA scalar truth requires an admitted native snapshot")


def _pca_bytes_truth(raw: PythonObject) raises -> Bool:
    ref cpy = Python().cpython()
    var address = cpy.lib.get_function[Int]("PyBytes_AsString")(raw._obj_ptr)
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for index in range(len(raw)):  # small-loop(raw void scalar storage truth)
        if pointer[index] != 0:
            return True
    return False


def _pca_structured_truth(value: PythonObject, dtype: PythonObject, operation: String,
                           context: PythonObject, depth: Int) raises -> Bool:
    if depth > 64:
        raise Error("TypeError: scalar configuration exceeds native nesting; supply a native number")
    var names = dtype.names
    ref cpy = Python().cpython()
    for index in range(len(names)):  # driver-loop: admitted structured scalar fields in their native order
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, names[index]._obj_ptr)
        if not pointer:
            var error = stash_python_error()
            raise error^
        var field = PythonObject(from_owned=pointer)
        var captured = PCAFitScalar(field, context, True, operation, depth)
        if captured.truth(operation, context):
            return True
    return False


def pca_fit_scalar_capture(value: PythonObject, mut cache: List[PCAFitScalar], context: PythonObject) raises -> Int:
    for index in range(len(cache)):  # small-loop(three original scalar setting identities)
        if cache[index].source is value:
            return index
    var index = len(cache)
    cache.append(PCAFitScalar(value, context))
    return index
