# SPDX-License-Identifier: Apache-2.0
# Compile the actual sequential_c entry (Sequential's native public operations,
# including load_state_dict's exact-type admission in bindings/sequential_weights.mojo).
from std.python._cpython import PyObjectPtr
from bindings.sequential_native import sequential_c

@export
def sequential_weights_admission_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return sequential_c(py_self, packed)
