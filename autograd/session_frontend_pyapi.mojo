# SPDX-License-Identifier: Apache-2.0
# Native Session setup over connections established by the package importer.
from std.python import Python, PythonObject
from autograd.program_arrays import initialize_program_arrays


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
    if not Bool(py=b.isinstance(requested, b.str)):
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var normalized = requested.strip().lower()
    var key = String(normalized)
    if key != "identical" and key != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entry = connections[normalized]
    if b.type(entry) is b.str:
        raise Error(String(entry))
    var expected = 1 if key == "identical" else 0
    if Int(py=entry.numeric_mode()) != expected:
        raise Error("ValueError: selected session binding does not implement the requested numeric mode")
    # Set the finalizer's ownership fields before the first allocation. A
    # failure leaves an empty shell that can be safely destroyed.
    var recording = Bool(py=record)
    var target = session
    target._id = PythonObject(None)
    target._session_native = entry.call
    target.record = PythonObject(recording)
    target.numeric_mode = mode
    target._b = entry.binding
    target._program_execute = entry.execute
    target._program_inspect = entry.inspect
    target._program_control = entry.control
    target._program_fit = entry.fit
    target._program_class = program_class
    target._distributed_binding = entry.distributed
    target._opt_kind = b.dict()
    var created = entry.open(Int(recording))
    target._lifetime = created[1]
    target._id = created[0]
    return PythonObject(None)
