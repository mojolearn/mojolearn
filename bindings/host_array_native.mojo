# SPDX-License-Identifier: Apache-2.0
# Shared host output allocation. Keep this leaf separate from saved-model
# reconstruction: sparse conversion and inference need storage, not archive
# schemas, tree model readers or training implementations.
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
    var store = py_call(context["array_type"], _store_code(dtype))
    var raw = py_call(Python.import_module("builtins").bytes, count * Int(py=store.itemsize))
    _ = py_call(store.frombytes, raw)
    return classical_wrap_array(store, shape, dtype, "C", context)
