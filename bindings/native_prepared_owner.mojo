# SPDX-License-Identifier: Apache-2.0
"""Prepared native owners: full setup once, then a cheap identity check per call.

A public runtime call used to redo its whole setup on every call: binding
lookup and vendor admission, owner and snapshot rebuilds, weight metadata
capture and module connection scans. A prepared owner keeps the result of that
setup in a native registry keyed by the Python object it belongs to, and the
next call reuses it after one identity check of everything the setup read.

Rules (shared by every site that uses it):
- The first use runs the complete original setup, so a refusal keeps its type,
  message and position.
- The registry entry lives outside the object's own dictionary (no pickling,
  copy or `__getstate__` change) and holds a weak reference to the object; the
  entry is dropped when the object dies, so it never extends a lifetime.
- The validity key is a flat list of the exact objects the setup read
  (container objects and their items, recursively, bounded depth). Any
  reassignment, in-place container edit, insertion or removal changes the key
  and the next call runs the full setup again. Values are compared by identity
  only, so no user `__eq__`/`__hash__` hook runs.
- Each call runs this check once at entry, after its module connection check;
  GIL release windows inside the call are handled by the caller's existing
  `NativeModules.released()` re-check, exactly as before.
- Nothing here computes on data or changes an arithmetic or fold order.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_setitem


struct PreparedOwners(Defaultable, Movable):
    var tables: PythonObject

    def __init__(out self):
        self.tables = PythonObject(None)


comptime PREPARED_OWNERS = _Global[
    StorageType=PreparedOwners, name="MojolearnPreparedOwnersV1",
    init_fn=PreparedOwners.__init__,
]


def _prepared_table(mut mods: NativeModules, slot: String) raises -> PythonObject:
    var g = PREPARED_OWNERS.get_or_create_ptr()
    var bi = mods.module("builtins")
    if g[].tables is None:
        g[].tables = bi.dict()
    var name = PythonObject(slot)
    var table = py_get(g[].tables, name)
    if table is None:
        table = bi.dict()
        py_setitem(g[].tables, name, table)
    return table


def _prepared_id(value: PythonObject) -> PythonObject:
    return PythonObject(Int(value._obj_ptr))


def prepared_get(mut mods: NativeModules, slot: String, owner: PythonObject) raises -> PythonObject:
    """The record stored for `owner` under `slot`, or None."""
    var table = _prepared_table(mods, slot)
    var entry = py_get(table, _prepared_id(owner))
    if entry is None:
        return PythonObject(None)
    # The weak reference is a native type: calling it never runs Python code.
    if py_call(entry[0]) is not owner:
        return PythonObject(None)
    return entry[1]


def prepared_put(mut mods: NativeModules, slot: String, owner: PythonObject, record: PythonObject) raises:
    """Store `record` for `owner`; silently skip objects without weak references."""
    var table = _prepared_table(mods, slot)
    var key = _prepared_id(owner)
    ref cpy = Python().cpython()
    # On the owner's death CPython calls table.pop(key, ref): a native method
    # bound with MethodType, so the entry goes away without Python code.
    var callback = py_call(mods.module("types").MethodType, table.pop, key)
    var raw = cpy.lib.get_function[PyObjectPtr]("PyWeakref_NewRef")(owner._obj_ptr, callback._obj_ptr)
    if not raw:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return
    var weak = PythonObject(from_owned=raw)
    py_setitem(table, key, Python.list(weak, record))


def prepared_drop(mut mods: NativeModules, slot: String, owner: PythonObject) raises:
    var table = _prepared_table(mods, slot)
    _ = py_call(table.pop, _prepared_id(owner), PythonObject(None))


def prepared_key_add(mut mods: NativeModules, key: PythonObject, value: PythonObject, depth: Int = 0) raises:
    """Append `value` and, for exact list/tuple/dict containers, their items."""
    _ = py_call(key.append, value)
    if depth >= 4:
        return
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if cls is bi.list or cls is bi.tuple:
        for i in range(len(value)):  # small-loop(value: retained container items): identity only, no hooks
            prepared_key_add(mods, key, value[i], depth + 1)
    elif cls is bi.dict:
        # items() of an exact dict never hashes or compares keys.
        var items = py_call(bi.list, py_call(value.items))
        for i in range(len(items)):  # small-loop(items: retained mapping items): identity only, no hooks
            _ = py_call(key.append, items[i][0])
            prepared_key_add(mods, key, items[i][1], depth + 1)


def prepared_key_fields(mut mods: NativeModules, key: PythonObject, fields: PythonObject, skip: List[String]) raises:
    """Append an exact field dictionary's items except the runtime-owned `skip` names."""
    var bi = mods.module("builtins")
    var items = py_call(bi.list, py_call(fields.items))
    for i in range(len(items)):  # small-loop(items: stored owner fields): identity only, no hooks
        var name = items[i][0]
        if py_type(name) is bi.str:
            var text = String(name)
            var skipped = False
            for j in range(len(skip)):  # small-loop(skip: runtime-owned field names): fixed name list only
                if skip[j] == text:
                    skipped = True
                    break
            if skipped:
                continue
        _ = py_call(key.append, name)
        prepared_key_add(mods, key, items[i][1], 1)


def prepared_key_same(a: PythonObject, b: PythonObject) raises -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(a: prepared key objects): pointer identity compare only
        if a[i] is not b[i]:
            return False
    return True
