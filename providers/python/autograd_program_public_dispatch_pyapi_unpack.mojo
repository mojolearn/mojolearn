# SPDX-License-Identifier: Apache-2.0
# The owning Session image completes the whole public Program operation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setitem
from providers.python.autograd_program_arrays_pyapi_door import program_numeric_scalar, program_admit_value, uint64_input


def _option(value: PythonObject) raises:
    if not program_numeric_scalar(value):
        raise Error("TypeError: Program flags require native boolean or numeric scalars")


def _arguments(args: PythonObject, count: Int) raises:
    if py_type(args) is not py_type(Python.tuple()) or len(args) != count:
        raise Error("TypeError: invalid native Program operation arguments")
