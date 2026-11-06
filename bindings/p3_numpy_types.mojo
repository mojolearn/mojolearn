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
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem


struct P3NumpyTypes(Defaultable, Movable):
    var table: PythonObject

    def __init__(out self):
        self.table = PythonObject(None)


# Named globals may be shared by every Mojo extension in the process: bump the
# name's version whenever the record layout changes.
comptime P3_NUMPY_TYPES = _Global[StorageType=P3NumpyTypes,
    name="MojolearnP3NumpyTypesV1", init_fn=P3NumpyTypes.__init__]


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


def numpy_admitted_type(cls: PythonObject, context: PythonObject) raises -> Bool:
    """True when `cls` is one of the static NumPy types named by
    `context["numpy_names"]` in the live `sys.modules["numpy"]`."""
    var module = py_get(context["modules"], "numpy")
    if module is None or py_type(module) is not context["module_type"]:
        return False
    var g = P3_NUMPY_TYPES.get_or_create_ptr()
    if g[].table is None:
        g[].table = Python.dict()
    var key = PythonObject(Int(context["numpy_names"]._obj_ptr))
    var entry = py_get(g[].table, key)
    if entry is None or entry[0] is not module:
        entry = _resolve(module, context)
        py_setitem(g[].table, key, entry)
    var types = entry[1]
    for i in range(len(types)):  # driver-loop: over the resolved static NumPy type identities
        if cls is types[i]:
            return True
    return False

