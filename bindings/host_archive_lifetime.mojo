# SPDX-License-Identifier: Apache-2.0
# Saved-model finalization. Only IVF's CPU inference binding owns a
# resident registry entry; KDE, embedding and neighbors host bindings keep
# ordinary buffers. Their inherited device finalizers are replaced by this
# native descriptor. The GPU index classes (NearestNeighbors and its
# subclasses, KernelDensity, IVFIndex and the x_ann indexes) take it too
# ("entry" plans): their device handle is released through the compiled
# release entry of the binding that minted it.
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
    if plan is None:
        return
    # A device class connected by `_native_archive_bridge.lifetime`:
    # (None, dictionary descriptor, "entry", holder key or None, release
    # entry name). Its resident tuple carries the binding that minted the
    # handle, `(key, handle, binding)`, under `_resident` or under the
    # holder dictionary's "index".
    var entry_plan = len(plan) > 2 and String(plan[2]) == "entry"
    if not entry_plan and plan[0] is None:
        return
    # The connected built-in dictionary descriptor bypasses replacement
    # __getattribute__/__dict__ hooks even during interpreter teardown.
    var fields = py_call(plan[1].__get__, model, cls)
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: native dictionary admission
        if py_type(keys[i]) is not b.str:
            return
    if entry_plan:
        var owner = fields
        var key = PythonObject("_resident")
        if plan[3] is not None:
            owner = py_get(fields, plan[3])
            key = PythonObject("index")
            if owner is None or py_type(owner) is not b.dict:
                return
        var held = py_get(owner, key)
        if held is None or py_type(held) is not b.tuple or len(held) != 3 or py_type(held[1]) is not b.int:
            return
        py_setitem(owner, key, PythonObject(None))
        # The binding's own compiled release entry; never a Python callback.
        var release = py_call(b.getattr, held[2], plan[4], PythonObject(None))
        if release is None or py_type(release) is not py_type(b.len):
            return
        _ = py_call(release, held[1])
        return
    var groves = len(plan) > 2 and String(plan[2]) == "groves"
    var cached = py_get(fields, "_groves" if groves else "_resident")
    if groves:
        if cached is None or py_type(cached) is not b.tuple or len(cached) != 2 or py_type(cached[0]) is not b.int or cached[1] is not None:
            return
        var family = py_get(fields, "family")
        if py_type(family) is not b.str or (String(family) != "rf" and String(family) != "et"):
            return
        py_setitem(fields, "_groves", PythonObject(None))
        _ = py_call(plan[0], cached[0], 0 if String(family) == "rf" else 1)
        return
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
