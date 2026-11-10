# SPDX-License-Identifier: Apache-2.0
"""Exact scalar UNPACK for optimizers; no mutable builtin calls or hooks."""
from std.python import PythonObject
from bindings.cpython_table import cpython_table
from bindings.optimizer_admission import opt_scalar
from bindings.optimizer_public_words import stash_python_error
from bindings.py_balanced import py_type


def optimizer_float(value: PythonObject, context: PythonObject) raises -> Float64:
    if py_type(value) is context["scalar_types"][3]:
        return Float64(py=value)
    opt_scalar(value, context)
    # Admission permits only fixed native builtin/NumPy scalar identities and
    # recursively admitted containers. Their numeric slots cannot call a user
    # conversion hook. PyNumber_Float retains builtin text/number semantics.
    var pointer = cpython_table()[].PyNumber_Float(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var number = PythonObject(from_owned=pointer)
    return Float64(py=number)


def optimizer_integer(value: PythonObject, context: PythonObject) raises -> Int:
    opt_scalar(value, context)
    if py_type(value) is context["scalar_types"][2]:
        return Int(py=value)
    var pointer = cpython_table()[].PyNumber_Long(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var number = PythonObject(from_owned=pointer)
    return Int(py=number)


def optimizer_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    opt_scalar(value, context)
    var result = cpython_table()[].PyObject_IsTrue(value._obj_ptr)
    if result < 0:
        var failure = stash_python_error()
        raise failure^
    return result != 0
