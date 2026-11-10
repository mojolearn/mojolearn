# SPDX-License-Identifier: Apache-2.0
"""Exact dictionary/scalar UNPACK for native class-weight plans."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_item
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.label_admission_native import labels_admit_primitive, labels_primitive_kind
from bindings.sequential_admission import _one_of
from bindings.native_scalar_decode import native_scalar_buffer_dtype
from core.class_weight_plan import ClassWeightInputs, class_weight_kind, class_weight_value

comptime CLASS_WEIGHT_CLASSICAL = 0
comptime CLASS_WEIGHT_SVC = 1
comptime CLASS_WEIGHT_TREES = 2


def class_weight_admit_dictionary[Values: Bool = True](value: PythonObject, context: PythonObject) raises:
    # PyDict_Next reads the admitted native dict storage. Admit keys/values before any
    # class hashing or numeric conversion, retaining the original refusal order.
    var cursor = InlineArray[Int, 1](fill=0)
    var items = InlineArray[PyObjectPtr, 2](fill=PyObjectPtr())
    var api = cpython_table()
    while api[].PyDict_Next(value._obj_ptr, Int(cursor.unsafe_ptr()),
                           Int(items.unsafe_ptr()), Int(items.unsafe_ptr() + 1)):
        _ = labels_admit_primitive(PythonObject(from_borrowed=items[0]), context)
        @parameter
        if Values:
            _ = labels_admit_primitive(PythonObject(from_borrowed=items[1]), context)
    _ = cursor^
    _ = items^
    _ = value


def class_weight_dictionary_values[Profile: Int = CLASS_WEIGHT_CLASSICAL](value: PythonObject, classes: PythonObject,
                                   context: PythonObject) raises -> ClassWeightInputs:
    # This is exact input-dictionary admission, before RUN. Keep the existing
    # set-difference semantics for every supported native numeric key, including
    # NumPy extended/weak comparisons and arbitrary-size integers. No private
    # Python container carries numerical weight state into RUN.
    var b = native_loaded_module("builtins")
    @parameter
    if Profile == CLASS_WEIGHT_TREES:
        # Tree constructor admission also accepts exact pathlib objects for
        # unrelated options. Those keys still have Python hash/equality hooks;
        # only native label primitives may enter the key-set operation below.
        if not _one_of(py_type(value), context["mapping_types"]):
            raise Error("TypeError: class_weight requires an exact native dictionary")
        # Tree admission includes exact OrderedDict/defaultdict as well as
        # dict. Their native dictionary storage needs no user protocol, and
        # GetItemWithError keeps dict.get's no-default-factory semantics.
        class_weight_admit_dictionary[False](value, context)
    var keys = value
    @parameter
    if Profile != CLASS_WEIGHT_TREES:
        keys = py_call(b.list, value)
    @parameter
    if Profile == CLASS_WEIGHT_SVC:
        for i in range(len(keys)):  # exact dictionary key UNPACK, before hashing
            if labels_primitive_kind(py_type(keys[i]), context) == 0:
                raise Error("TypeError: class_weight keys must be native numbers or strings")
    var unknown = py_call(py_call(b.set, keys).difference, classes)
    if len(unknown) > 0:
        raise Error("class_weight contains labels absent from y")
    var result = ClassWeightInputs()
    var is_tuple = py_type(classes) is b.tuple
    var api = cpython_table()
    ref cpy = Python().cpython()
    for i in range(len(classes)):  # exact scalar UNPACK requires the GIL
        var key = borrowed_item(classes, i, is_tuple)
        var raw = api[].PyDict_GetItemWithError(value._obj_ptr, key)
        if not raw:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            result.append(1.0, True)
            continue
        var item = PythonObject(from_borrowed=raw)
        var cls = py_type(item)
        @parameter
        if Profile == CLASS_WEIGHT_CLASSICAL:
            class_weight_kind(labels_admit_primitive(item, context))
        elif Profile == CLASS_WEIGHT_TREES:
            # Keep the tree family's existing exact scalar membership rather
            # than silently widening it to every label scalar alias.
            var bool_types = py_get(context, "bool_types")
            var is_bool = cls is b.bool or (bool_types is not None and _one_of(cls, bool_types))
            if is_bool or not (cls is b.int or cls is b.float or _one_of(cls, context["scalar_types"])):
                raise Error("class weights must be finite nonnegative real numbers")
        else:
            class_weight_kind(labels_primitive_kind(cls, context))
        var number = Float64(0)
        var exact_one = False
        if cls is b.float or cls is b.int:
            number = api[].PyFloat_AsDouble(raw)
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            class_weight_value(number)
            # No other integer rounds to binary64 one: neighboring integers
            # differ by at least one, far beyond its rounding interval. Reuse
            # the required float conversion instead of another CPython call.
            exact_one = number == 1
        else:
            # Decode the exact NumPy scalar's buffer once. Native conversion
            # preserves binary64 rounding and exact source equality without
            # entering a NumPy numeric slot or a Python warning callback.
            var scalar = native_scalar_buffer_dtype(item, item.dtype)
            number = bitcast[DType.float64](scalar.real.float64_bits())
            class_weight_value(number)
            exact_one = scalar.equals_integer(1)
        result.append_validated(number, exact_one)
    _ = value
    _ = classes
    return result^
