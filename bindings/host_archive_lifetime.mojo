# SPDX-License-Identifier: Apache-2.0
# CPU saved-model finalization. Only IVF's CPU inference binding owns a
# resident registry entry; KDE, embedding and neighbors host bindings keep
# ordinary buffers. Their inherited Python device finalizers are replaced
# by this native descriptor, without changing GPU model classes.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem


def host_archive_finalize(model: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var plan = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: connected host cleanup ownership
        plan = py_get(context["host_archive_lifetime_layouts"], mro[i])
        if plan is not None:
            break
    if plan is None or plan[0] is None:
        return
    # The connected built-in dictionary descriptor bypasses replacement
    # __getattribute__/__dict__ hooks even during interpreter teardown.
    var fields = py_call(plan[1].__get__, model, cls)
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: native dictionary admission
        if py_type(keys[i]) is not b.str:
            return
    var cached = py_get(fields, "_resident")
    if cached is None or py_type(cached) is not b.tuple or len(cached) != 3:
        return
    var token = cached[1]
    if py_type(token) is not b.int:
        return
    py_setitem(fields, "_resident", PythonObject(None))
    # Exact native entry recorded from _mojolearn_ivf_search_host. Never
    # invoke the cached module or a Python release callback.
    _ = py_call(plan[0], token)


def host_archive_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        host_archive_finalize(args[1], args[0])
    except:
        # Preserve the previous quiet teardown semantics (released handles,
        # partially constructed objects, or a shutting-down interpreter).
        pass
    return PythonObject(None).steal_data()
