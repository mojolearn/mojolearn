# SPDX-License-Identifier: Apache-2.0
# One native public call; route to the exact Session provider image retaining
# the captured Program dictionary and original owner for the complete call.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_setitem
from bindings.native_loaded_module import native_loaded_module
from providers.python.autograd_program_owner_pyapi import program_owner_fields
from providers.python.autograd_session_public_request_pyapi import ops_session_public_call_py


def ops_program_public_call_py(owner: PythonObject, operation: PythonObject,
                              args: PythonObject) raises -> PythonObject:
    if py_type(operation) is not py_type(PythonObject("")) or py_type(args) is not py_type(Python.tuple()):
        raise Error("TypeError: Program requires an exact operation string and argument tuple")
    var fields = program_owner_fields(owner)
    var pid = fields["_id"]
    if pid is not None and py_type(pid) is not py_type(PythonObject(0)):
        raise Error("TypeError: Program handle requires an exact native integer")
    # Do not copy unrelated user fields: destroying a snapshot after execution
    # must not release application finalizers that another thread removed.
    var snapshot = Python.dict()
    py_setitem(snapshot, "_s", fields["_s"])
    py_setitem(snapshot, "_id", pid)
    return ops_session_public_call_py(snapshot["_s"], PythonObject("program_public"),
                                      Python.tuple(owner, snapshot, fields, operation, args))
