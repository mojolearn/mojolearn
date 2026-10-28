# SPDX-License-Identifier: Apache-2.0
# Mutable scalar-array settings are captured before input preparation can
# release the GIL. Conversion, warnings and failures stay at their historical
# validation point. These native records never escape through the Python ABI.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_get, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.native_scalar_decode import NativeScalarBuffer, native_scalar_buffer_dtype
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native import sequence_array_release
from bindings.training_scalar_native import training_scalar_number, training_scalar_warning, training_void_bytes
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, stash_python_error
from core.native_scalar_decode import NativeRealBits


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

    def __init__(out self, source: PythonObject, context: PythonObject) raises:
        self.source = source
        self.owners = Python.list(source)
        self.value = source
        self.arrays = List[PCAFitScalarArray]()
        self.number = NativeScalarBuffer(NativeRealBits(), NativeRealBits(), False, False)
        self.numeric = False
        self.refusal = String("")
        self.release = 0
        var scalar_context = context["pca_fit_scalar_context"]
        var b = native_loaded_module("builtins")
        for depth in range(66):  # driver-loop: admitted nested object scalar arrays, same 64-level bound as scalar parsing
            var value = self.value
            var cls = py_type(value)
            if cls is b.bytearray:
                # Preserve the builtin parser's bytearray error spelling too.
                self.value = py_call(b.bytearray, value)
                break
            if cls is b.memoryview:
                if Bool(py=value.c_contiguous):
                    self.value = py_call(b.memoryview, training_void_bytes(value))
                else:
                    # int/float require a contiguous text buffer. Preserve a
                    # strided native view for their existing refusal without
                    # retaining mutable source bytes for later conversion.
                    var storage = py_call(b.bytes, 4)
                    var view = py_call(b.memoryview, storage)
                    self.value = py_call(view.__getitem__, py_call(b.slice, 0, 4, 2))
                break
            if cls is context["array_type"]:
                # The numeric parser reads an array.array as raw text bytes.
                self.value = training_void_bytes(value)
                break
            if cls is not context["ndarray"]:
                # Primitive NumPy scalars and builtin numbers/text are
                # immutable. Unsupported protocol objects are merely retained;
                # the shared scalar admission refuses them before conversion.
                break
            if depth == 65:
                self.refusal = "TypeError: scalar configuration exceeds native nesting; supply a native number"
                break
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
                if kind == "V":
                    if dtype.fields is not None:
                        self.refusal = "TypeError: float() argument must be a string or a real number, not 'tuple'"
                    else:
                        self.value = training_void_bytes(value)
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
                    var failure = stash_python_error()
                    raise failure^
                self.value = PythonObject(from_owned=pointer)
                _ = py_call(self.owners.append, self.value)
            except error:
                # Numeric storage errors are replayed at conversion; a
                # pending native CPython exception must cross the door while
                # its exact category/traceback is still retained by the ABI.
                if String(error) == PYTHON_ERROR_PENDING:
                    raise error^
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
                if array.char != "G":
                    raise Error("TypeError: float() argument must be a string or a real number, not 'complex'")
                var exceptions = py_get(scalar["modules"], "numpy.exceptions")
                if exceptions is None or py_type(exceptions) is not scalar["module_type"]:
                    raise Error("TypeError: complex scalar warnings require NumPy's native exception metadata")
                var category = py_get(native_numpy_module_fields(exceptions), "ComplexWarning")
                training_scalar_warning("Casting complex values to real discards the imaginary part", category, operation, scalar)
        if self.refusal.byte_length() > 0:
            raise Error(self.refusal)
        if self.numeric:
            return self.number.to_python_number(integer)
        return training_scalar_number(self.value, operation, scalar, integer)


def pca_fit_scalar_capture(value: PythonObject, mut cache: List[PCAFitScalar], context: PythonObject) raises -> Int:
    for index in range(len(cache)):  # small-loop(three original scalar setting identities)
        if cache[index].source is value:
            return index
    var index = len(cache)
    cache.append(PCAFitScalar(value, context))
    return index
