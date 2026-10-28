# SPDX-License-Identifier: Apache-2.0
# Compile the actual native caller of the assigned timedelta range repair.
# This is compile coverage only; no public operation is executed.
from std.python._cpython import PyObjectPtr
from bindings.arena_rows_native import arena_rows_c


@export
def peer2_arena_rows(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return arena_rows_c(py_self, args_ptr)
