# SPDX-License-Identifier: Apache-2.0
# Focused compilation of the real public PCA entry (fit, fit_transform, query,
# params, archive branches) and all its transitive native callees.
# This exported forwarding root is compile evidence only; no runtime is claimed.
from std.python._cpython import PyObjectPtr
from bindings.pca_native import pca_c


@export
def peer2_pca_public_entry(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return pca_c(py_self, args_ptr)
