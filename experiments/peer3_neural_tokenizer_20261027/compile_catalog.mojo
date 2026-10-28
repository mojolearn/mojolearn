# SPDX-License-Identifier: Apache-2.0
# Compile-only catalog metadata and exported-graph import/execution closure.
from std.python._cpython import PyObjectPtr
from bindings.model_catalog_native import catalog_public_c


@export
def peer3_catalog_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return catalog_public_c(py_self, args)
