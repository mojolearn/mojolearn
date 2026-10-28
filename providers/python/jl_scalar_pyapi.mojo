# SPDX-License-Identifier: Apache-2.0
# Scalar admission and decoding for the Johnson-Lindenstrauss public bound.
# Only exact native scalar owners can reach a conversion or buffer descriptor.
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_get, py_type
from bindings.py_typed_call import typed_python_error
from bindings.sequential_admission import _one_of
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.native_numpy_metadata import native_numpy_dtype


def jl_scalar(value: PythonObject, epsilon: Bool, context: PythonObject) raises -> Float64:
    var cls = py_type(value)
    var float_type = py_type(PythonObject(0.0))
    var int_type = py_type(PythonObject(0))
    var bool_type = py_type(PythonObject(True))
    var positive_error = String("The JL bound is defined for n_samples greater than zero")
    var epsilon_error = String("The JL bound is defined for eps in ]0, 1[")
    if cls is int_type or cls is bool_type:
        # No integer lies strictly between zero and one. The direct CPython
        # integer predicate handles arbitrary-size n without narrowing first;
        # exact type admission excludes every user comparison/conversion slot.
        if epsilon:
            raise Error(epsilon_error)
        ref cpy = Python().cpython()
        var zero = PythonObject(0)
        var nonpositive = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(
            value._obj_ptr, zero._obj_ptr, Int32(1))
        _ = zero^
        if nonpositive < 0:
            raise typed_python_error()
        if nonpositive != 0:
            raise Error(positive_error)
        var number = cpy.lib.get_function[Float64]("PyLong_AsDouble")(value._obj_ptr)
        if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
            raise typed_python_error()
        return number
    if cls is float_type:
        var number = Float64(py=value)
        if epsilon:
            if not (number > 0.0 and number < 1.0):
                raise Error(epsilon_error)
        elif number <= 0.0:
            raise Error(positive_error)
        return number
    var array = cls is py_get(context, "ndarray")
    var scalar = _one_of(cls, context["scalar_types"])
    # The generic shared list includes f16/f32/f64. Preserve long-double
    # scalar admission too; construct its exact native type without invoking
    # a mutable NumPy module export or a user scalar subclass.
    if not scalar and not array and py_get(context, "ndarray") is not None:
        scalar = cls is py_call(native_numpy_dtype(context), "g").type
    if not scalar and not array:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires native real scalars without Python hooks")
    if array and Int(py=value.ndim) != 0:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires scalar or zero-dimensional numeric inputs")
    # One capture of dtype/buffer metadata and bytes, retaining the original
    # owner throughout. Object dtype is rejected by the decoder before any
    # contained object's comparison, truth or numeric conversion can execute.
    var decoded = native_scalar_buffer(value)
    if decoded.complex:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires native real scalars")
    var real = decoded.real
    if epsilon:
        # Check the original precision before the historical binary64
        # conversion. A long-double value just below one may round to one.
        if real.special != 0 or real.negative or not real.truth() or real.highest() + real.scale >= 0:
            raise Error(epsilon_error)
    elif real.special != 2 and (real.negative or not real.truth()):
        raise Error(positive_error)
    return bitcast[DType.float64](real.float64_bits())
