# SPDX-License-Identifier: Apache-2.0
# The legacy module retains each C entry name before import-time owner
# connection. A missing owner refuses natively; this executes no model work
# and never calls a Python fallback. Complete installations replace these
# entries with the original native function objects from their one owner
# (since the 2026-11-07 host split, one of several _mojolearn_runtime_*_host
# bindings; python/mojolearn/host_surface.py HOST_IMPORT_ALIASES).
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception


def runtime_classical_unavailable_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var error = Error("mojolearn: this operation requires its compiled host owner binding (a _mojolearn_runtime_*_host file); build the complete host layout and start a fresh process")
    return raise_python_exception(error^, ExceptionType("PyExc_ModuleNotFoundError"))
