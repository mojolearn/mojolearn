# SPDX-License-Identifier: Apache-2.0
# Preserve Python's established argument-error classes at the native ABI.
# These are documented std.python.bindings APIs (Mojo 1.0), not CPython
# internals or a Python catch-and-rethrow callback.
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception


def graph_python_error(error: Error) -> PyObjectPtr:
    var message = String(error)
    if message.startswith("TypeError: "):
        return raise_python_exception(Error(message[byte=11:]), ExceptionType.TypeError)
    if message.startswith("ValueError: "):
        return raise_python_exception(Error(message[byte=12:]), ExceptionType.ValueError)
    return raise_python_exception(error)
