# SPDX-License-Identifier: Apache-2.0
# Compile the actual selection, restoration and C error translation paths.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_exceptions import classical_not_fitted_class, classical_error_restore, classical_error_reduce
from bindings.classical_error_boundary import classical_python_error


@export
def classical_exception_admission_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_not_fitted_class(args[3])
        if args[0] is not None:
            return classical_error_reduce(args[0], args[3]).steal_data()
        return classical_error_restore(args[2], args[3]).steal_data()
    except error:
        return classical_python_error(error^, packed)
