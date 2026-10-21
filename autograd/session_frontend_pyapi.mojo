# SPDX-License-Identifier: Apache-2.0
# Native Session setup over connections established by the package importer.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setattr, py_type
from autograd.program_arrays_pyapi import initialize_program_arrays


def ops_session_initialize_py(session: PythonObject, record: PythonObject, mode: PythonObject,
                              connections: PythonObject, backend: PythonObject, program_class: PythonObject) raises -> PythonObject:
    if session is PythonObject(None):
        initialize_program_arrays()
        return PythonObject(None)
    var b = Python.import_module("builtins")
    var requested = mode
    if requested is PythonObject(None):
        requested = backend._DEFAULT_MODE
        if requested is PythonObject(None):
            requested = backend._SELECTED
    if not Bool(py=py_call(b.isinstance, requested, b.str)):
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var normalized = requested.strip().lower()
    var key = String(normalized)
    if key != "identical" and key != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entry = connections[normalized]
    if py_type(entry) is b.str:
        raise Error(String(entry))
    var expected = 1 if key == "identical" else 0
    if Int(py=entry.numeric_mode()) != expected:
        raise Error("ValueError: selected session binding does not implement the requested numeric mode")
    # Set the finalizer's ownership fields before the first allocation. A
    # failure leaves an empty shell that can be safely destroyed.
    var recording = Bool(py=record)
    var target = session
    py_setattr(target, "_id", PythonObject(None))
    py_setattr(target, "_session_native", entry.call)
    py_setattr(target, "record", PythonObject(recording))
    py_setattr(target, "numeric_mode", mode)
    py_setattr(target, "_b", entry.binding)
    py_setattr(target, "_program_execute", entry.execute)
    py_setattr(target, "_program_inspect", entry.inspect)
    py_setattr(target, "_program_control", entry.control)
    py_setattr(target, "_program_fit", entry.fit)
    py_setattr(target, "_program_class", program_class)
    py_setattr(target, "_distributed_binding", entry.distributed)
    py_setattr(target, "_opt_kind", b.dict())
    var created = py_call(entry.open, Int(recording))
    py_setattr(target, "_lifetime", created[1])
    py_setattr(target, "_id", created[0])
    return PythonObject(None)
