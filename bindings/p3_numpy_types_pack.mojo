# SPDX-License-Identifier: Apache-2.0
"""Admitted NumPy type identities, resolved once per NumPy module object.

Optimizer and Samba admission accepted a value's type as a NumPy buffer or
scalar type by reading `sys.modules["numpy"]`, calling `type.__flags__` on
the type (a Python-defined subclass is a heap type and is refused), reading
the module dictionary and comparing against each admitted name, for every
value that was not an exact builtin. The admitted set only depends on the
NumPy module object: resolve it once (keeping only static extension types,
so a heap type can never be in it) and compare identities after that. A
reloaded NumPy (a different module object in `sys.modules`) re-resolves. A
module attribute replaced after resolution is not followed: the original
static NumPy type stays admitted, and the replacement is never admitted,
which is the stricter of the two readings.

Cache key: the connection's `numpy_names` tuple (connection contexts live for
the process, so its address is never recycled while the entry is used).
"""
# The PACK half of the former `bindings/p3_numpy_types.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/p3_numpy_types_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem
from core.shared_slot import SharedSlot


def _resolve(module: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var fields = py_fields(module)
    var names = context["numpy_names"]
    var types = Python.list()
    for i in range(len(names)):  # driver-loop: over fixed admitted NumPy type names, once per NumPy module
        var candidate = py_get(fields, names[i])
        if candidate is None or not Bool(py=py_call(bi.isinstance, candidate, bi.type)):
            continue
        if (Int(py=py_call(context["type_flags"], candidate)) & 512) != 0:
            continue
        _ = py_call(types.append, candidate)
    return Python.tuple(module, py_call(bi.tuple, types))
