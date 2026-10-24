# SPDX-License-Identifier: Apache-2.0
# The legacy module retains each C entry name before import-time owner
# connection. A missing owner refuses natively; this executes no model work
# and never calls a Python fallback. Complete installations replace these
# entries with the original native function objects from the one owner.
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception


def runtime_classical_unavailable_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var error = Error("mojolearn: this operation requires the compiled _mojolearn_runtime_classical_host binding; build the complete host layout and start a fresh process")
    return raise_python_exception(error^, ExceptionType("PyExc_ModuleNotFoundError"))
