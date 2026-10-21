# SPDX-License-Identifier: Apache-2.0
"""Native finalization of existing optimizer resident allocations."""
from std.python import Python, PythonObject


def optimizer_finalize_binding(owner: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    # Optimizers inherit Python attribute policy. These are stored ownership
    # records, so use the base C getter and never invoke that policy at cleanup.
    var fields = bi.object.__getattribute__(owner, "__dict__")
    var handle = fields.get("_res")
    var binding = fields.get("_res_binding")
    if handle is None or binding is None:
        return PythonObject(None)
    # Resident creation retains the exact owning extension. Resolve only its
    # stored C entry, never a backend proxy's Python __getattr__ or a callback.
    var entries = bi.object.__getattribute__(binding, "__dict__")
    var close = entries.get("optimizer_resident_close")
    if bi.type(close) is not bi.type(bi.len):
        raise Error("optimizer finalization: resident owner lacks a compiled close entry")
    try:
        _ = close(handle)
    except:
        # Preserve the established best-effort finalizer for an already lost
        # native resource. Keep its handle available for an explicit retry.
        return PythonObject(None)
    fields["_res"] = PythonObject(None)
    fields["_res_binding"] = PythonObject(None)
    return PythonObject(None)
