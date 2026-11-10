# SPDX-License-Identifier: Apache-2.0
"""Optimizer finalizer UNPACK/PACK; resident release uses its native owner."""
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_get, py_setitem
from bindings.optimizer_admission import opt_scalar
from bindings.optimizer_public_door import OPT_DOOR
from bindings.optimizer_descriptor_boundary import optimizer_close_slot


def optimizer_finalize_descriptor_binding(context: PythonObject, owner: PythonObject) raises -> PythonObject:
    return optimizer_finalize_binding(owner)


def optimizer_finalize_binding(owner: PythonObject) raises -> PythonObject:
    # Direct native dictionary access bypasses Python attribute policy. The
    # descriptor connection admits the exact extension owner and pins it.
    var fields = py_fields(owner)
    var handle = py_get(fields, "_res")
    var binding = py_get(fields, "_res_binding")
    if handle is None or binding is None:
        return PythonObject(None)
    try:
        opt_scalar(handle, OPT_DOOR.get_or_create_ptr()[].context)
        optimizer_close_slot(binding, Int(py=handle))
    except:
        # Preserve best-effort cleanup during shutdown or after resource loss;
        # retain the ownership fields so explicit retry remains possible.
        return PythonObject(None)
    py_setitem(fields, "_res", PythonObject(None))
    py_setitem(fields, "_res_binding", PythonObject(None))
    return PythonObject(None)
