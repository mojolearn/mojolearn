# SPDX-License-Identifier: Apache-2.0
# Native dtype-schema admission and result ownership for sequence __array__.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_attributes import classical_connect_numpy
from core.sequence_array_cast import sequence_cast_leaf, sequence_wide_word
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_normalize
from bindings.sequence_string_storage import sequence_string_admit, sequence_string_leaf


def sequence_dtype_leaves(dtype: PythonObject, offset: Int, classes: PythonObject,
                          leaves: PythonObject, context: PythonObject, depth: Int = 0) raises:
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
            sequence_dtype_leaves(item[0], offset + Int(py=item[1]), classes, leaves, context, depth + 1)
        return
    var subarray = dtype.subdtype
    if subarray is not None:
        var shape = subarray[1]
        var count = 1
        for axis in range(len(shape)):  # driver-loop: over dtype subarray rank metadata
            count *= Int(py=shape[axis])
        var size = Int(py=subarray[0].itemsize)
        for i in range(count):  # driver-loop: over fixed dtype subarray elements
            sequence_dtype_leaves(subarray[0], offset + i * size, classes, leaves, context, depth + 1)
        return
    var kind = String(dtype.kind)
    var size = Int(py=dtype.itemsize)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "c" and kind != "O" and kind != "S" and kind != "U" and kind != "V" and kind != "m" and kind != "M" and kind != "T":
        raise Error("TypeError: dtype requires an explicit native output layout")
    if kind == "T":
        sequence_string_admit(dtype, classes, context)
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


def sequence_array_address(value: PythonObject) raises -> Int:
    # Both inputs here are exact native ndarrays retained by the caller.
    # __array_interface__ requests a structured dtype's `descr`, which enters
    # NumPy's Python _array_descr helper. A simple buffer export needs only
    # the contiguous data address and invokes no dtype-format conversion.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0))
    if status != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return address


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
    var dtype = sequence_dtype_normalize(requested, classes, context)
    if (String(dtype.kind) == "S" or String(dtype.kind) == "U") and Int(py=dtype.itemsize) == 0:
        dtype = py_call(dtype_constructor, String(dtype.kind) + "32")
    if String(dtype.kind) == "V" and Int(py=dtype.itemsize) == 0 and dtype.fields is None:
        # astype adapts an unsized void target (including zero-element
        # subarrays) to the source itemsize. Empty structured layouts retain
        # their fields and zero width instead of becoming raw void storage.
        dtype = py_call(dtype_constructor, "V" + String(Int(py=source.itemsize)))
    var leaves = Python.list()
    sequence_dtype_leaves(dtype, 0, classes, leaves, context)
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
    # Raw integer addresses do not carry a Mojo lifetime. Retain the source
    # owner explicitly until every conversion loop has consumed its words.
    var source_owner = source
    var start = sequence_array_address(source_owner)
    var destination = sequence_array_address(result)
    var status = 0
    for i in range(len(leaves)):  # driver-loop: over admitted native output field layouts
        var leaf = leaves[i]
        if String(leaf[0]) == "O":
            sequence_object_leaf(start, destination, count, stride, Int(py=leaf[3]))
        elif String(leaf[0]) == "T":
            status |= sequence_string_leaf(start, destination, count, stride, Int(py=leaf[3]), result.dtype, classes, settings, context)
        else:
            status |= sequence_cast_leaf(start, destination, count, stride, Int(py=leaf[3]), String(leaf[0]), Int(py=leaf[1]), Bool(py=leaf[2]))
    _ = source_owner^
    sequence_array_report(status, settings, context)
    return result
