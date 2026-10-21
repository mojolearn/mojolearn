# SPDX-License-Identifier: Apache-2.0
"""Native finalization of existing optimizer resident allocations."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem, py_type


def optimizer_finalize_binding(owner: PythonObject) raises -> PythonObject:
    var bi: PythonObject
    try:
        bi = Python.import_module("builtins")
    except:
        # interpreter shutdown (no import machinery left): the process is
        # releasing everything; there is nothing to close through
        return PythonObject(None)
    # Optimizers inherit Python attribute policy. These are stored ownership
    # records, so use the base C getter and never invoke that policy at cleanup.
    var fields = py_call(bi.object.__getattribute__, owner, "__dict__")
    var handle = py_get(fields, "_res")
    var binding = py_get(fields, "_res_binding")
    if handle is None or binding is None:
        return PythonObject(None)
    # Resident creation retains the exact owning extension. Resolve only its
    # stored C entry, never a backend proxy's Python __getattr__ or a callback.
    var entries = py_call(bi.object.__getattribute__, binding, "__dict__")
    var close = py_get(entries, "optimizer_resident_close")
    if py_type(close) is not py_type(bi.len):
        raise Error("optimizer finalization: resident owner lacks a compiled close entry")
    try:
        _ = py_call(close, handle)
    except:
        # Preserve the established best-effort finalizer for an already lost
        # native resource. Keep its handle available for an explicit retry.
        return PythonObject(None)
    py_setitem(fields, "_res", PythonObject(None))
    py_setitem(fields, "_res_binding", PythonObject(None))
    return PythonObject(None)
