# SPDX-License-Identifier: Apache-2.0
# Native dtype-schema admission and result ownership for sequence __array__.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from core.sequence_array_cast import sequence_cast_leaf, sequence_wide_word
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report
from bindings.native_numpy_metadata import native_numpy_dtype


def sequence_dtype_classes(context: PythonObject) raises -> PythonObject:
    var dtype = native_numpy_dtype(context)
    var stored = py_get(context, "sequence_dtype_classes")
    if stored is not None:
        return stored
    var result = Python.list()
    # Only NumPy's fixed built-in scalar dtype schemas are admitted. A user
    # DType with discovery/casting hooks requires its own native interface.
    for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: over fixed native dtype metadata
        var item = py_call(dtype, code)
        py_call(result.append, Python.tuple(py_type(item), item.type))
    py_setitem(context, "sequence_dtype_classes", result)
    return result


def sequence_dtype_instance(value: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over connected native dtype layouts
        if py_type(value) is classes[i][0]:
            return True
    return False


def sequence_dtype_spec(value: PythonObject, classes: PythonObject, context: PythonObject, depth: Int = 0) raises:
    var b = Python.import_module("builtins")
    if depth > 64:
        raise Error("TypeError: dtype schema nesting exceeds the native limit")
    var kind = py_type(value)
    if value is None or kind is b.int or kind is b.bool:
        return
    if kind is b.str or kind is b.bytes:
        var text = value if kind is b.str else py_call(b.bytes.decode, value, "ascii", "strict")
        var aliases = py_fields(context["modules"]["numpy"])["sctypeDict"]
        if py_type(aliases) is not b.dict:
            raise Error("TypeError: dtype aliases require native dictionary metadata")
        var keys = py_call(b.list, aliases)
        for i in range(len(keys)):  # driver-loop: validate dtype alias keys before keyed lookup
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: dtype aliases require native string keys")
        var mapped = py_get(aliases, text)
        if mapped is not None:
            sequence_dtype_spec(mapped, classes, context, depth + 1)
        elif py_call(context["sequence_dtype_fullmatch"], context["sequence_dtype_pattern"], text) is None:
            raise Error("TypeError: dtype text requires an explicit native dtype schema")
        return
    if value is b.float or value is b.int or value is b.bool or value is b.complex or value is b.str or value is b.bytes or value is b.object:
        return
    if sequence_dtype_instance(value, classes):
        return
    for i in range(len(classes)):  # driver-loop: over fixed native scalar classes accepted as dtypes
        if value is classes[i][1]:
            return
    if kind is b.tuple or kind is b.list:
        for i in range(len(value)):  # driver-loop: over nested native dtype schema metadata
            sequence_dtype_spec(value[i], classes, context, depth + 1)
        return
    if kind is b.dict:
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: over native dtype schema fields
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: dtype schema keys must be native strings")
            sequence_dtype_spec(value[keys[i]], classes, context, depth + 1)
        return
    raise Error("TypeError: dtype requires a built-in native dtype/schema; Python dtype protocols cannot execute")


def sequence_dtype_leaves(dtype: PythonObject, offset: Int, classes: PythonObject,
                          leaves: PythonObject, depth: Int = 0) raises:
    var b = Python.import_module("builtins")
    if depth > 64 or not sequence_dtype_instance(dtype, classes):
        raise Error("TypeError: output dtype requires a built-in native layout")
    var fields = dtype.fields
    if fields is not None:
        var names = dtype.names
        if py_type(names) is not b.tuple:
            raise Error("TypeError: structured dtype requires native field names")
        var keys = py_call(b.list, fields)
        for i in range(len(keys)):  # driver-loop: validate structured-field keys before lookup
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: dtype field titles require native strings")
        for i in range(len(names)):  # driver-loop: over structured dtype field schemas
            var item = fields[names[i]]
            if py_type(item) is not b.tuple or py_type(item[1]) is not b.int:
                raise Error("TypeError: dtype fields require native layout metadata")
            sequence_dtype_leaves(item[0], offset + Int(py=item[1]), classes, leaves, depth + 1)
        return
    var subarray = dtype.subdtype
    if subarray is not None:
        var shape = subarray[1]
        var count = 1
        for axis in range(len(shape)):  # driver-loop: over dtype subarray rank metadata
            count *= Int(py=shape[axis])
        var size = Int(py=subarray[0].itemsize)
        for i in range(count):  # driver-loop: over fixed dtype subarray elements
            sequence_dtype_leaves(subarray[0], offset + i * size, classes, leaves, depth + 1)
        return
    var kind = String(dtype.kind)
    var size = Int(py=dtype.itemsize)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "c" and kind != "O" and kind != "S" and kind != "U" and kind != "V" and kind != "m" and kind != "M":
        raise Error("TypeError: dtype requires an explicit native output layout")
    py_call(leaves.append, Python.tuple(kind, size, String(dtype.byteorder) != ">", offset))


def sequence_object_leaf(source: Int, destination: Int, count: Int, stride: Int, offset: Int) raises:
    ref cpy = Python().cpython()
    var input = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    for index in range(count):  # driver-loop: box one completed scalar into each native object-array output slot
        var pointer = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=destination + index * stride + offset)
        # ndarray allocation initializes object fields to None. Only that
        # native singleton is replaced; no user object's finalizer is called.
        var previous = pointer[0]
        var boxed = PythonObject(bitcast[DType.float64](sequence_wide_word(input[index])))
        pointer[0] = boxed.steal_data()
        if previous:
            cpy.Py_DecRef(previous)


def sequence_array_cast(source: PythonObject, requested: PythonObject, context: PythonObject) raises -> PythonObject:
    # No public route is installed until the complete prior dtype contract is
    # qualified; this helper is not an unsupported-dtype Python fallback.
    if requested is None:
        return source
    var dtype_constructor = native_numpy_dtype(context)
    var input = context["input_context"]
    classical_connect_numpy(input)
    if py_get(input, "ndarray") is None:
        raise Error("TypeError: the ndarray protocol requires NumPy's native array type to be connected")
    var classes = sequence_dtype_classes(context)
    sequence_dtype_spec(requested, classes, context)
    var dtype = py_call(dtype_constructor, requested)
    if (String(dtype.kind) == "S" or String(dtype.kind) == "U") and Int(py=dtype.itemsize) == 0:
        dtype = py_call(dtype_constructor, String(dtype.kind) + "32")
    var leaves = Python.list()
    sequence_dtype_leaves(dtype, 0, classes, leaves)
    if dtype == source.dtype:
        # astype(copy=False) returns the downloaded view for an equal dtype,
        # preserving its raw NaN words, backing owner and writeability flags.
        return source
    var settings = sequence_array_errstate(context)
    # Native ndarray allocation owns and initializes object slots. No buffer
    # protocol, scalar conversion or user allocation subclass is invoked.
    var result = py_call(input["ndarray"], source.shape, dtype=dtype)
    var count = Int(py=source.size)
    var stride = Int(py=dtype.itemsize)
    var start = Int(py=classical_address(source))
    var destination = Int(py=classical_address(result))
    var status = 0
    for i in range(len(leaves)):  # driver-loop: over admitted native output field layouts
        var leaf = leaves[i]
        if String(leaf[0]) == "O":
            sequence_object_leaf(start, destination, count, stride, Int(py=leaf[3]))
        else:
            status |= sequence_cast_leaf(start, destination, count, stride, Int(py=leaf[3]), String(leaf[0]), Int(py=leaf[1]), Bool(py=leaf[2]))
    sequence_array_report(status, settings, context)
    return result
