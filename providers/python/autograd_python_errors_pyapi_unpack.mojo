# SPDX-License-Identifier: Apache-2.0
# Preserve Python's established argument-error classes at the native ABI.
# These are documented std.python.bindings APIs (Mojo 1.0), not CPython
# internals or a Python catch-and-rethrow callback.
from std.python._cpython import PyObjectPtr
from std.python import Python, PythonObject
from bindings.py_balanced import py_type
from std.python.bindings import ExceptionType, raise_python_exception


def graph_python_error(error: Error) -> PyObjectPtr:
    var message = String(error)
    if message.startswith("TypeError: "):
        return raise_python_exception(Error(message[byte=11:]), ExceptionType.TypeError)
    if message.startswith("ValueError: "):
        return raise_python_exception(Error(message[byte=12:]), ExceptionType.ValueError)
    if message.startswith("OverflowError: "):
        return raise_python_exception(Error(message[byte=15:]), ExceptionType("PyExc_OverflowError"))
    if message.startswith("FileNotFoundError: "):
        return raise_python_exception(Error(message[byte=19:]), ExceptionType("PyExc_FileNotFoundError"))
    if message.startswith("ImportError: "):
        return raise_python_exception(Error(message[byte=13:]), ExceptionType("PyExc_ImportError"))
    if message.startswith("RuntimeError: "):
        # The graph owners' RuntimeError refusals (an ONNX reader status
        # outside 1-4, truncated reader metadata, a missing reader entry)
        # keep their class instead of a bare Exception.
        return raise_python_exception(Error(message[byte=14:]), ExceptionType("PyExc_RuntimeError"))
    return raise_python_exception(error)
