# SPDX-License-Identifier: Apache-2.0
# Shared host output allocation. Keep this leaf separate from saved-model
# reconstruction: sparse conversion and inference need storage, not archive
# schemas, tree model readers or training implementations.
from bindings.native_array_storage import native_array_zeros
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.classical_arrays import classical_wrap_array, _store_code


def host_archive_typed(shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    # Archive output is always Array storage; obtain the width from its fixed
    # native typecode, including int32 (the linear allocator only supports f4/f8/i8).
    var count = 1
    for i in range(len(shape)):  # driver-loop: native output shape metadata
        var dimension = Int(py=shape[i])
        if dimension < 0:
            raise Error("saved model array dimensions must be nonnegative")
        count *= dimension
    # p3-array-arena: one zeroed allocation (was an empty store, a zero bytes
    # object of the output size and a frombytes copy). Same zero words.
    var store = native_array_zeros(String(_store_code(dtype)), count)
    return classical_wrap_array(store, shape, dtype, "C", context)
