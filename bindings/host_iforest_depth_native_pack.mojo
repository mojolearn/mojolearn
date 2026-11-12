# SPDX-License-Identifier: Apache-2.0
# Fit guard state is product execution: native admission, NumPy-versioned
# promotion, arithmetic, warnings and publication all stay in this owner.
# The PACK half of the former `bindings/host_iforest_depth_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/host_iforest_depth_native_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_numpy_metadata_pack import native_numpy_type, native_numpy_module_fields
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.sequence_array_warnings_unpack import sequence_array_report
from bindings.sequence_array_warnings_pack import sequence_array_errstate
from core.sequence_array_cast import SEQ_CAST_INVALID
from core.host_iforest_depth_run import host_iforest_depth_step_run


def _depth_dtype(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return PythonObject(None)
    if py_get(context,"ndarray") is None:
        return PythonObject(None)
    var array = cls is context["ndarray"]
    if array and Int(py=value.size) != 1:
        raise Error("TypeError: scalar model state requires a scalar or zero-dimensional numeric array")
    var held = _depth_dtypes(context)
    var dtypes = held[1]
    var types = held[2]
    var value_dtype_type = py_type(value.dtype) if array else PythonObject(None)
    for i in range(len(dtypes)):  # driver-loop: exact primitive native scalar identities, including distinct aliases
        var dtype = dtypes[i]
        if array and value_dtype_type is py_type(dtype):
            return value.dtype
        if cls is types[i]:
            return dtype
    return PythonObject(None)


def _depth_dtypes(context: PythonObject) raises -> PythonObject:
    """`(dtype constructor, the 18 primitive dtypes, their scalar types)`,
    built once per connected NumPy dtype constructor and kept in the context
    (p3-trees: up to 18 np.dtype() calls ran 4-5 times per fit). NumPy
    dtype objects of these primitive codes are immutable singletons."""
    var constructor = native_numpy_dtype(context)
    var held = py_get(context, "_host_depth_dtypes")
    if held is not None and held[0] is constructor:
        return held
    var dtypes = Python.list()
    var types = Python.list()
    for code in ["?","b","B","h","H","i","I","l","L","q","Q","e","f","d","g","F","D","G"]:  # driver-loop: exact primitive native scalar identities, including distinct aliases
        var dtype = py_call(constructor, code)
        _ = py_call(dtypes.append, dtype)
        _ = py_call(types.append, dtype.type)
    var record = Python.tuple(constructor, dtypes, types)
    py_setitem(context, "_host_depth_dtypes", record)
    return record


def _depth_box(data: MutPointer[UInt8, MutUntrackedOrigin], dtype: PythonObject,
               shape: PythonObject, scalar: Bool, context: PythonObject) raises -> PythonObject:
    # Allocate native numeric storage, then copy already-computed bytes.
    # Neither a NumPy cast nor a scalar numeric constructor participates.
    var array = py_call(native_numpy_type("ndarray",context),shape,dtype=dtype)
    var width = Int(py=dtype.itemsize)
    var view = List[Int](length=10,fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(array._obj_ptr,pointer,Int32(0x119))
    if status != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        if view[2] != width or width < 1 or width > 32:
            raise Error("TypeError: fit depth publication requires one native primitive element")
        var destination = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view[0])
        for i in range(width):  # driver-loop: publish at most one complex long-double element
            destination[i] = data[i]
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
        # Keep the Py_buffer storage alive past the release on every path
        # (ASAP destruction could free it before PyBuffer_Release reads it).
        _ = len(view)
    _ = view^
    if not scalar:
        return array
    var index = Python.tuple()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(array._obj_ptr,index._obj_ptr)
    _ = index^
    _ = array^
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)
