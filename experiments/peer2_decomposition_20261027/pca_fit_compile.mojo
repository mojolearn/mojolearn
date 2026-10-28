# SPDX-License-Identifier: Apache-2.0
# Coordinator-owned compilation only: both fit results and solver branches.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.classical_attributes import classical_connect_numpy
from bindings.pca_fit_native import pca_fit_operation


@export
def peer2_pca_fit_compile_root(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args)
        var context = call[3]
        classical_connect_numpy(context)
        var result = pca_fit_operation(call[0], call[1], Bool(py=call[2]), context)
        return result.steal_data()
    except error:
        return raise_python_exception(error)
