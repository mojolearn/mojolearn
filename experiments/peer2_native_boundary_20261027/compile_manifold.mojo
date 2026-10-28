# SPDX-License-Identifier: Apache-2.0
# Compile the existing manifold entry without rebuilding unrelated families.
from std.python._cpython import PyObjectPtr
from bindings.runtime_manifold_entry import runtime_manifold_c


@export
def peer2_manifold(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return runtime_manifold_c(py_self, args_ptr)
