# SPDX-License-Identifier: Apache-2.0
# Admission before replacing retained forecast state. All updates are staged;
# an identical retained reference requires no release admission.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_lifetime import native_release_admit
from bindings.classical_attributes import classical_connect_numpy
from bindings.py_balanced import py_call, py_type, py_get, py_setitem


def forecast_commit(fields: PythonObject, updates: PythonObject, ownership: PythonObject) raises:
    classical_connect_numpy(ownership)
    var b = native_loaded_module("builtins")
    var names = py_call(b.list, updates)
    var retained = Python.list()
    for i in range(len(names)):  # driver-loop: over staged fields before any owner is replaced
        var old = py_get(fields, names[i])
        var new = updates[names[i]]
        _ = py_call(retained.append, old)
        if old is not new:
            native_release_admit(old, ownership)
    for i in range(len(names)):  # driver-loop: over fully admitted constructor state updates
        py_setitem(fields, names[i], updates[names[i]])
    _ = retained
