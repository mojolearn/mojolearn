# SPDX-License-Identifier: Apache-2.0
# Compile the actual x_linear_native_call entry (expansion-linear public lifecycle
# and every linear/CV/isotonic/SGD/coordinate-descent route it dispatches).
from std.python._cpython import PyObjectPtr
from bindings.x_linear_native import x_linear_c

@export
def expansion_linear_public_lifecycle_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return x_linear_c(py_self, packed)
