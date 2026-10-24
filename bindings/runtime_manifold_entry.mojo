# SPDX-License-Identifier: Apache-2.0
# Manifold orchestration and its helpers compile in one native owner. Existing
# matrix/device owners are passed by retained Python references; no MKit value
# or exposed Mojo type crosses this C-entry boundary.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.sys.compile import is_defined
from bindings.classical_error_boundary import classical_python_error
from bindings.manifold_native import manifold_operation

# This owner reports the existing whole-host-set negative-control flag. Its
# connected kernel owners provide the arithmetic controls; the extraction
# adds no numerical branch or standalone owner control.
comptime RUNTIME_MANIFOLD_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_manifold_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(call[1])
        if not operation.startswith("family_mani_"):
            raise Error("RuntimeError: unknown family operation " + operation)
        var result = manifold_operation(call[0], operation, call[2], call[3])
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
