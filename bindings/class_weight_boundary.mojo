# SPDX-License-Identifier: Apache-2.0
"""Exact dictionary/scalar UNPACK for native class-weight plans."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_item
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.label_admission_native import labels_admit_primitive
from bindings.native_scalar_decode import native_scalar_buffer_dtype
from core.class_weight_plan import ClassWeightInputs, class_weight_kind, class_weight_value


def class_weight_admit_dictionary(value: PythonObject, context: PythonObject) raises:
    # PyDict_Next reads an exact dictionary. Admit every key/value before any
    # class hashing or numeric conversion, retaining the original refusal order.
    var cursor = InlineArray[Int, 1](fill=0)
    var items = InlineArray[PyObjectPtr, 2](fill=PyObjectPtr())
    var api = cpython_table()
    while api[].PyDict_Next(value._obj_ptr, Int(cursor.unsafe_ptr()),
                           Int(items.unsafe_ptr()), Int(items.unsafe_ptr() + 1)):
        _ = labels_admit_primitive(PythonObject(from_borrowed=items[0]), context)
        _ = labels_admit_primitive(PythonObject(from_borrowed=items[1]), context)
    _ = cursor^
    _ = items^
    _ = value


def class_weight_dictionary_values(value: PythonObject, classes: PythonObject,
                                   context: PythonObject) raises -> ClassWeightInputs:
    # This is exact input-dictionary admission, before RUN. Keep the existing
    # set-difference semantics for every supported native numeric key, including
    # NumPy extended/weak comparisons and arbitrary-size integers. No private
    # Python container carries numerical weight state into RUN.
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, value)
    var unknown = py_call(py_call(b.set, keys).difference, classes)
    if len(unknown) > 0:
        raise Error("class_weight contains labels absent from y")
    var result = ClassWeightInputs()
    var is_tuple = py_type(classes) is b.tuple
    var api = cpython_table()
    ref cpy = Python().cpython()
    var overflow = InlineArray[Int32, 1](fill=0)
    for i in range(len(classes)):  # exact scalar UNPACK requires the GIL
        var key = borrowed_item(classes, i, is_tuple)
        var raw = api[].PyDict_GetItemWithError(value._obj_ptr, key)
        if not raw:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            result.append(1.0, True)
            continue
        var item = PythonObject(from_borrowed=raw)
        var kind = labels_admit_primitive(item, context)
        class_weight_kind(kind)
        var cls = py_type(item)
        var number = Float64(0)
        var exact_one = False
        if cls is b.float or cls is b.int:
            number = api[].PyFloat_AsDouble(raw)
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            class_weight_value(number)
            if cls is b.float:
                exact_one = number == 1
            else:
                overflow[0] = 0
                var word = api[].PyLong_AsLongLongAndOverflow(raw, Int(overflow.unsafe_ptr()))
                if cpy.PyErr_Occurred():
                    raise cpy.unsafe_get_error()
                exact_one = overflow[0] == 0 and word == 1
        else:
            # Decode the exact NumPy scalar's buffer once. Native conversion
            # preserves binary64 rounding and exact source equality without
            # entering a NumPy numeric slot or a Python warning callback.
            var scalar = native_scalar_buffer_dtype(item, item.dtype)
            number = bitcast[DType.float64](scalar.real.float64_bits())
            class_weight_value(number)
            exact_one = scalar.equals_integer(1)
        result.append_validated(number, exact_one)
    _ = overflow^
    _ = value
    _ = classes
    return result^
