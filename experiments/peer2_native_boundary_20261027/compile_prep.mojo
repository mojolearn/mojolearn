# SPDX-License-Identifier: Apache-2.0
# Focused compilation of the real preprocessing entry and all its branches.
# This exported forwarding root is compile evidence only; no runtime is claimed.
from std.python._cpython import PyObjectPtr
from bindings.prep_stateless_native import prep_stateless_c


@export
def peer2_prep_entry(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return prep_stateless_c(py_self, args_ptr)
