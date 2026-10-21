# SPDX-License-Identifier: Apache-2.0
# Preserve Python's established argument-error classes at the native ABI.
# These are documented std.python.bindings APIs (Mojo 1.0), not CPython
# internals or a Python catch-and-rethrow callback.
from std.python._cpython import PyObjectPtr
from std.python import Python, PythonObject
from std.python.bindings import ExceptionType, raise_python_exception


def graph_int(var value: PythonObject) raises -> PythonObject:
    # PyNumber_Long is CPython's public int() conversion. Its result is a
    # new reference; the source stays owned by `owner` for the whole call.
    # PythonObject.__call__ converts a Python exception to its text only.
    # Read the original exception through the supported CPython API here,
    # so the existing native exception boundary can preserve its class.
    ref cpy = Python().cpython()
    var source = value.steal_data()
    var owner = PythonObject(from_owned=source)
    var result = cpy.PyNumber_Long(source)
    if result:
        _ = owner^
        return PythonObject(from_owned=result)
    var raised = cpy.PyErr_Fetch() if cpy.version.minor < 12 else cpy.PyErr_GetRaisedException()
    var exception = PythonObject(from_owned=raised)
    var kind = String(Python.import_module("builtins").type(exception).__name__)
    var message = String(exception)
    _ = owner^
    raise Error(kind + ": " + message)


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
    return raise_python_exception(error)
