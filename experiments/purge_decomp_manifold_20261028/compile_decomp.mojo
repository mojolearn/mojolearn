# SPDX-License-Identifier: Apache-2.0
# Compile-only root: the whole decomposition owner door (decomp_c), including
# the RandomProjection components_ getter/setter and __getstate__ operations.
# Compilation is not runtime qualification.
from std.python._cpython import PyObjectPtr
from bindings.decomp_native import decomp_c


@export
def purge_decomp_compile_root(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return decomp_c(py_self, args_ptr)
