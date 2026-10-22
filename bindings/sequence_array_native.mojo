# SPDX-License-Identifier: Apache-2.0
# Native dtype-schema admission and result ownership for sequence __array__.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast
from std.sys.info import CompilationTarget
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_attributes import classical_connect_numpy
from core.sequence_array_cast import sequence_cast_leaf, sequence_wide_word, sequence_signaling, SEQ_CAST_INVALID
from bindings.sequence_print_options import sequence_print_mode
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_normalize, sequence_dtype_allocation_shape
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
    py_call(leaves.append, Python.tuple(kind, size, String(dtype.byteorder) != ">", offset, dtype))


def sequence_object_leaf(source: Int, destination: Int, count: Int, stride: Int, offset: Int) raises -> Int:
    ref cpy = Python().cpython()
    var input = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    var status = 0
    for index in range(count):  # driver-loop: box one completed scalar into each native object-array output slot
        var pointer = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=destination + index * stride + offset)
        # ndarray allocation initializes object fields to None. Only that
        # native singleton is replaced; no user object's finalizer is called.
        var previous = pointer[0]
        var bits = input[index]
        if sequence_signaling(bits):
            # NumPy widens float32 to the boxed native double; quieting a
            # signaling NaN preserves its payload and sets invalid status.
            status |= SEQ_CAST_INVALID
        var boxed = PythonObject(bitcast[DType.float64](sequence_wide_word(bits)))
        pointer[0] = boxed.steal_data()
        if previous:
            cpy.Py_DecRef(previous)
    return status


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


def sequence_array_release(context: PythonObject) raises -> Int:
    # NumPy 2.3 added NPY_METH_NO_FLOATINGPOINT_ERRORS to any_to_object
    # (numpy/_core/src/multiarray/dtype_transfer.c). Its 2.0 ABI version
    # cannot distinguish this change from 2.0--2.2, which still report flags.
    # Read only exact native module/string metadata; no version helper runs.
    var module = py_get(context["modules"], "numpy")
    if module is None or py_type(module) is not context["module_type"]:
        raise Error("TypeError: array cast compatibility requires NumPy's native module")
    var version = py_get(py_fields(module), "__version__")
    if py_type(version) is not Python.import_module("builtins").str:
        raise Error("TypeError: array cast compatibility requires native NumPy version text")
    var text = String(version)
    var parts = text.split(".")
    if len(parts) < 2:
        raise Error("TypeError: array cast compatibility requires NumPy major.minor metadata")
    var major = Int(String(parts[0]))
    var minor = Int(String(parts[1]))
    return major * 100 + minor


def sequence_array_object_reports(context: PythonObject) raises -> Bool:
    return sequence_array_release(context) < 203


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
    sequence_dtype_allocation_shape(source.shape, dtype, context)
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
    var managed = False
    for i in range(len(leaves)):  # driver-loop: identify layouts requiring descriptor-owned storage
        managed = managed or String(leaves[i][0]) == "T"
    if managed:
        # NumPy may clone a top-level StringDType, or expand outer subarrays
        # into result.shape. Derive each allocator descriptor from the actual
        # allocated base layout, including nested structured String fields.
        # Requested descriptors and a structured parent are not substitutes
        # for the descriptor owning the destination's string storage.
        var allocated = result.dtype
        var itemsize = Int(py=allocated.itemsize)
        if itemsize <= 0 or stride % itemsize != 0:
            raise Error("TypeError: allocated managed dtype has an incompatible native layout")
        leaves = Python.list()
        for i in range(stride // itemsize):  # driver-loop: repeat a base layout expanded into outer array dimensions
            sequence_dtype_leaves(allocated, i * itemsize, classes, leaves, context)
    # Raw integer addresses do not carry a Mojo lifetime. Retain the source
    # owner explicitly until every conversion loop has consumed its words.
    var source_owner = source
    var start = sequence_array_address(source_owner)
    var destination = sequence_array_address(result)
    var status = 0
    var reports_status = False
    var print_mode = -1
    for i in range(len(leaves)):  # driver-loop: over admitted native output field layouts
        var leaf = leaves[i]
        var kind = String(leaf[0])
        var width = Int(py=leaf[1])
        if print_mode == -1 and (kind == "S" or kind == "U" or kind == "T"):
            print_mode = sequence_print_mode(context, sequence_array_release(context))
        # The combined structured transfer reports flags when ANY leaf can
        # report them. Object boxing still sets invalid status for a later
        # numeric/String leaf, even when an object-only cast suppresses it.
        # Numeric copies/byteswaps and legacy S/U formatting also carry
        # NO_FLOATINGPOINT_ERRORS; other float32 casts do not.
        if kind == "O":
            reports_status = reports_status or sequence_array_object_reports(context)
        elif kind != "S" and kind != "U" and not (kind == "f" and width == 4):
            reports_status = True
        if kind == "O":
            status |= sequence_object_leaf(start, destination, count, stride, Int(py=leaf[3]))
        elif kind == "T":
            status = sequence_string_leaf(start, destination, count, stride, Int(py=leaf[3]), leaf[4], classes, settings, context, status, print_mode)
        else:
            var software_half = False
            comptime if CompilationTarget.has_neon():
                # NumPy1.26--2.3 half.hpp explicitly disables ARM __fp16 and
                # uses FromFloatBits. NumPy2.4 enables ARM half conversion.
                # The software path preserves signaling NaNs without invalid
                # status; this is a dtype/version contract, not a shape rule.
                if kind == "f" and width == 2:
                    software_half = sequence_array_release(context) < 204
            status |= sequence_cast_leaf(start, destination, count, stride, Int(py=leaf[3]), kind, width, Bool(py=leaf[2]), software_half, print_mode)
    _ = source_owner^
    if reports_status:
        sequence_array_report(status, settings, context)
    return result
