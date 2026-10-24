# SPDX-License-Identifier: Apache-2.0
# The model-selection owner has its own ordinary CPython entry. Its helpers
# and pending-error storage compile together; no exposed Mojo type crosses
# this owner boundary. Development builds can compile it independently.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.sys.compile import is_defined
from bindings.classical_error_boundary import classical_python_error
from bindings.model_selection_native import msel_operation

comptime RUNTIME_MODEL_SELECTION_SABOTAGE = is_defined["MOJOLEARN_RUNTIME_MODEL_SELECTION_SABOTAGE"]()


def runtime_model_selection_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(call[1])
        var result = msel_operation(call[0], operation, call[2], call[3])
        comptime if RUNTIME_MODEL_SELECTION_SABOTAGE:
            # Negative controls for public splitter counts and the actual
            # train/test outputs hashed by x-metrics-splitters. The result
            # of train_test_split is our own exact list, never caller code.
            # The ordinary host module loader refuses this configuration.
            if operation == "family_msel_n_splits":
                result = PythonObject(Int(py=result) + 1)
            if operation == "family_msel_train_test_split":
                var first = result[0]
                result[0] = result[1]
                result[1] = first
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
