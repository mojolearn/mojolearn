# SPDX-License-Identifier: Apache-2.0
# Compile-only root for bindings/public_lifecycle_owner.mojo (shared lifecycle helper).
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.public_lifecycle_owner import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result

@export
def lifecycle_owner_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=packed)
        var depth = lifecycle_begin(call[0], "transform", call[1], True, False, True, call[2])
        lifecycle_end(call[0], "transform", depth)
        lifecycle_complete(call[0], "fit", call[1], True, call[2])
        return lifecycle_result(call[0], "transform", call[1], "", call[2]).steal_data()
    except:
        return PyObjectPtr()
