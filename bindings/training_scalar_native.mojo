# SPDX-License-Identifier: Apache-2.0
# Native scalar configuration unboxing; never ndarray computation or casts.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_fields, py_get
from bindings.optimizer_admission import opt_exact, opt_value
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_warnings import classical_warning
from bindings.native_numpy_metadata import native_numpy_type
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native import sequence_array_release


def training_void_bytes(value: PythonObject) raises -> PythonObject:
    # Exact builtin raw-void dtype admission precedes this documented buffer
    # API. A mutable builtins.memoryview name cannot redirect the operation.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromObject")(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var view = PythonObject(from_owned=pointer)
    return py_call(view.tobytes)


def training_scalar_warning(message: String, category: PythonObject, operation: String,
                            context: PythonObject) raises:
    var warning = context["training_scalar_warnings"]
    try:
        classical_warning(message, category, warning, context["training_scalar_locations"][PythonObject(operation)], "TrainingScalarWarning")
    except error:
        if not String(error).startswith("TrainingScalarWarning: "):
            raise error^
        # classical_warning admitted every exception slot before selecting
        # the error action. Preserve that exact category across the Mojo door.
        var instance = py_call(category, message)
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr, instance._obj_ptr)
        var failure = stash_python_error()
        raise failure^


def training_scalar_number(value: PythonObject, operation: String, context: PythonObject,
                           integer: Bool, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: scalar configuration exceeds native nesting; supply a native number")
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var admitted = opt_exact(cls, context["scalar_types"]) or cls is b.bytearray or cls is b.list or cls is b.tuple or cls is b.dict
    if cls is b.memoryview:
        opt_value(value, context)
        admitted = True
    if not admitted:
        var numpy = py_get(context["modules"], "numpy")
        if numpy is None:
            raise Error("TypeError: scalar configuration requires native values; Python conversion hooks cannot execute")
        var classes = sequence_dtype_classes(context)
        var input = context["input_context"]
        classical_connect_numpy(input)
        if cls is native_numpy_type("ndarray", context):
            var dtype = value.dtype
            var native_dtype = sequence_dtype_instance(dtype, classes)
            for code in ["q", "Q"]:  # driver-loop: native long-long descriptor aliases
                native_dtype = native_dtype or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
            if not native_dtype:
                raise Error("TypeError: scalar arrays require a built-in native dtype")
            var ndim = Int(py=value.ndim)
            var release = sequence_array_release(context)
            if ndim > 0 and release >= 204:
                # NumPy2.4 expired the1.25 scalar-array deprecation.
                # https://numpy.org/doc/2.4/release/2.4.0-notes.html#raise-typeerror-on-attempt-to-convert-array-with-ndim-0-to-scalar
                raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
            if Int(py=value.size) != 1:
                raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
            if ndim > 0:
                training_scalar_warning("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)",
                                        b.DeprecationWarning, operation, context)
            if String(dtype.kind) == "c":
                # ndarray's scalar slot boxes a Python complex; unlike a
                # NumPy complex scalar, float() rejects it without a warning.
                raise Error("TypeError: float() argument must be a string or a real number, not 'complex'")
            if String(dtype.kind) == "V":
                if dtype.fields is not None:
                    # A record unboxes to a tuple, which float rejects. Do
                    # not read arbitrary field values just to reject it.
                    raise Error("TypeError: float() argument must be a string or a real number, not 'tuple'")
                # Raw void's native storage is bytes. NumPy's float slot
                # parses those bytes (e.g. V4(b'1e-5')); keep both successful
                # text parsing and ValueError for invalid raw bytes.
                var raw = training_void_bytes(value)
                return training_scalar_number(raw, operation, context, integer, depth + 1)
            var indexes = Python.list()
            for i in range(ndim):  # driver-loop: native scalar index, one zero per singleton axis
                py_call(indexes.append, 0)
            var index = py_call(b.tuple, indexes)
            ref cpy = Python().cpython()
            # Exact ndarray + admitted builtin dtype: this documented native
            # item access only unboxes one scalar. Object/StringDType missing
            # values are admitted recursively before float can see a hook.
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, index._obj_ptr)
            _ = index^
            if not pointer:
                var failure = stash_python_error()
                raise failure^
            var scalar = PythonObject(from_owned=pointer)
            var result = training_scalar_number(scalar, operation, context, integer, depth + 1)
            _ = scalar^
            return result
        for i in range(len(classes)):  # driver-loop: exact native scalar class identities
            admitted = admitted or cls is classes[i][1]
        for code in ["q", "Q"]:  # driver-loop: distinct long-long scalar aliases on supported hosts
            admitted = admitted or cls is sequence_dtype_create(PythonObject(code), context).type
        if not admitted:
            raise Error("TypeError: scalar configuration requires native values; Python conversion hooks cannot execute")
        var dtype = value.dtype
        var kind = String(dtype.kind)
        if kind == "V":
            if dtype.fields is not None:
                raise Error("TypeError: float() argument must be a string or a real number, not 'tuple'")
            var raw = training_void_bytes(value)
            return training_scalar_number(raw, operation, context, integer, depth + 1)
        if kind == "c":
            var exceptions = py_get(context["modules"], "numpy.exceptions")
            if exceptions is None or py_type(exceptions) is not context["module_type"]:
                raise Error("TypeError: complex scalar warnings require NumPy's native exception metadata")
            var category = py_get(py_fields(exceptions), "ComplexWarning")
            training_scalar_warning("Casting complex values to real discards the imaginary part", category, operation, context)
            return training_scalar_number(value.real, operation, context, integer, depth + 1)
    # Only verified native scalar slots reach CPython's public unboxing API.
    # Native long-double slots preserve the platform's supported conversion
    # to Python double without reading private scalar storage or changing ABI.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long" if integer else "PyNumber_Float")(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def training_scalar_float(value: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    return training_scalar_number(value, operation, context, False)
