# SPDX-License-Identifier: Apache-2.0
# Native truth and reduction configuration for the public training loss.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.optimizer_admission import opt_exact, opt_value
from bindings.optimizer_public_door import TensorView, tensor_view
from bindings.optimizer_public_words import stash_python_error
from bindings.native_numpy_metadata import native_numpy_type
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native import sequence_array_release
from bindings.training_scalar_native import training_scalar_warning


def loss_native_item(value: PythonObject, key: PythonObject) raises -> PythonObject:
    # Callers first admit the exact builtin ndarray/void dtype and key.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, key._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def loss_truth(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> Bool:
    if depth > 64:
        raise Error("TypeError: loss options require native scalar configuration")
    var cls = py_type(value)
    var b = Python.import_module("builtins")
    var admitted = opt_exact(cls, context["scalar_types"]) or opt_exact(cls, context["buffer_types"])
    admitted = admitted or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.frozenset
    if cls is b.memoryview:
        opt_value(value, context)
        admitted = True
    if cls is context["array_class"]:
        var view = TensorView()
        _ = tensor_view(value, view)
        if len(view.shape) == 0:
            raise Error("TypeError: len() of unsized object")
        return view.shape[0] != 0
    if not admitted:
        var numpy = py_get(context["modules"], "numpy")
        if numpy is None:
            raise Error("TypeError: return_grad requires native configuration without Python truth hooks")
        var classes = sequence_dtype_classes(context)
        var ndarray = native_numpy_type("ndarray", context)
        var is_array = cls is ndarray
        for i in range(len(classes)):  # driver-loop: fixed native scalar class identities
            admitted = admitted or cls is classes[i][1]
        for code in ["q", "Q"]:  # driver-loop: native long-long scalar aliases
            admitted = admitted or cls is sequence_dtype_create(PythonObject(code), context).type
        if not admitted and not is_array:
            raise Error("TypeError: return_grad requires native configuration without Python truth hooks")
        var dtype = value.dtype
        var builtin = sequence_dtype_instance(dtype, classes)
        for code in ["q", "Q"]:  # driver-loop: native long-long descriptor aliases
            builtin = builtin or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
        if not builtin:
            raise Error("TypeError: return_grad arrays require a builtin native dtype")
        if is_array:
            var size = Int(py=value.size)
            if size > 1:
                raise Error("ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            if size == 0:
                # NumPy2.2 expired the empty-array truth deprecation.
                # https://numpy.org/doc/2.2/release/2.2.0-notes.html#expired-deprecations
                if sequence_array_release(context) >= 202:
                    raise Error("ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
                training_scalar_warning("The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.",
                                        b.DeprecationWarning, "cross_entropy", context)
                return False
            if Bool(py=dtype.hasobject) or String(dtype.kind) == "T":
                var indexes = Python.list()
                for i in range(Int(py=value.ndim)):  # driver-loop: one native singleton coordinate per rank
                    py_call(indexes.append, 0)
                var key = py_call(b.tuple, indexes)
                var scalar = loss_native_item(value, key)
                var result = loss_truth(scalar, context, depth + 1)
                _ = scalar^
                return result
        elif Bool(py=dtype.hasobject):
            # A structured scalar's truth is the first nonzero field. Walk
            # native fields here so object fields cannot invoke Python hooks.
            var names = dtype.names
            if names is None:
                raise Error("TypeError: object truth requires an admitted native scalar value")
            for i in range(len(names)):  # driver-loop: native structured scalar truth with short circuit
                var scalar = loss_native_item(value, names[i])
                if loss_truth(scalar, context, depth + 1):
                    return True
            return False
    # This is scalar configuration unboxing through the documented public
    # CPython slot API. Exact native types and builtin dtype metadata are
    # admitted above; object-bearing array/record values took the guarded walk.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[Int32]("PyObject_IsTrue")(value._obj_ptr)
    if result < 0:
        var failure = stash_python_error()
        raise failure^
    return result != 0


def loss_reduction(value: PythonObject, context: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var text = cls is context["scalar_types"][5]
    if not text and py_get(context["modules"], "numpy") is not None:
        var classes = sequence_dtype_classes(context)
        for i in range(len(classes)):  # driver-loop: recognize the native NumPy unicode scalar
            if cls is classes[i][1]:
                text = String(value.dtype.kind) == "U"
    if text:
        var name = String(value)
        if name == "none":
            return 0
        if name == "sum":
            return 1
        if name == "mean":
            return 2
        raise Error("ValueError: cross_entropy reduction must be 'none', 'sum' or 'mean'")
    # In particular, opt_value refuses np.void and recursively refuses it
    # inside tuples. A readonly structured scalar can hash object fields;
    # it must never reach the generic hash slot below.
    opt_value(value, context)
    # Match dictionary membership's native unhashable errors without letting
    # an unadmitted Python __hash__/__eq__ enter the operation.
    ref cpy = Python().cpython()
    var hash_value = cpy.lib.get_function[Int]("PyObject_Hash")(value._obj_ptr)
    if hash_value == -1:
        var failure = stash_python_error()
        raise failure^
    raise Error("ValueError: cross_entropy reduction must be 'none', 'sum' or 'mean'")
