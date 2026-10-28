# SPDX-License-Identifier: Apache-2.0
# Coordinator-owned compilation only. This forwarder instantiates the complete
# PCA query operation without unrelated fit/archive/estimator-family dispatch.
# Do not execute or count compilation as public runtime verification.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.classical_attributes import classical_connect_numpy
from bindings.pca_query_native import pca_query_operation


@export
def peer2_pca_query_compile_root(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args)
        var context = call[3]
        classical_connect_numpy(context)
        var result = pca_query_operation(call[0], call[1], Bool(py=call[2]), context)
        return result.steal_data()
    except error:
        return raise_python_exception(error)
