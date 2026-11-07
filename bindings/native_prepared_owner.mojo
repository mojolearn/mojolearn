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
  entry is dropped promptly when the object dies. Callers must not store a
  record that strongly retains its own owner; a weak key cannot break that cycle.
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
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import NativeModules, native_namespace_string_keys
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_model_records import records_get, records_put, records_drop, records_key_append, RECORDS_PREPARED


# The registry itself lives in bindings/native_model_records.mojo (lane
# ne1-e-records, 2026-11-06): one weak entry per object for every family, the
# prepared records in its RECORDS_PREPARED payloads. The weakref callback,
# token and recycled-address check moved there unchanged.


def prepared_get(mut mods: NativeModules, slot: String, owner: PythonObject) raises -> PythonObject:
    # Native slot lookup: no Python dict, boxed identity integer, temporary
    # record list or Python-object method dispatch on a cache hit. The entry's
    # weak referent is confirmed to be `owner` (a recycled address never
    # matches a dead entry).
    return records_get(RECORDS_PREPARED, owner, slot)


def prepared_put(mut mods: NativeModules, slot: String, owner: PythonObject, record: PythonObject) raises:
    # Nothing is stored for an owner that takes no weak reference.
    _ = records_put(RECORDS_PREPARED, owner, slot, record)


def prepared_drop(mut mods: NativeModules, slot: String, owner: PythonObject) raises:
    records_drop(RECORDS_PREPARED, owner, slot)


def prepared_key_add(mut mods: NativeModules, key: PythonObject, value: PythonObject, depth: Int = 0, limit: Int = 4) raises:
    """Append `value` and, for exact list/tuple/dict containers, their items
    (to `limit` levels; deeper containers are compared by identity only)."""
    records_key_append(key, value)
    if depth >= limit:
        return
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if cls is bi.list or cls is bi.tuple:
        for i in range(len(value)):  # small-loop(value: retained container items): identity only, no hooks
            prepared_key_add(mods, key, value[i], depth + 1, limit)
    elif cls is bi.dict:
        # items() of an exact dict never hashes or compares keys.
        var items = py_call(bi.list, py_call(value.items))
        for i in range(len(items)):  # small-loop(items: retained mapping items): identity only, no hooks
            records_key_append(key, items[i][0])
            prepared_key_add(mods, key, items[i][1], depth + 1, limit)


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
        records_key_append(key, name)
        prepared_key_add(mods, key, items[i][1], 1)


def prepared_key_same(a: PythonObject, b: PythonObject) raises -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(a: prepared key objects): pointer identity compare only
        if a[i] is not b[i]:
            return False
    return True


def prepared_class_stamp(mut mods: NativeModules, cls: PythonObject, extras: PythonObject) raises -> PythonObject:
    """Identity stamp of a class admission: `extras` (the admission's own
    inputs), the ancestry tuple, and every key and value of each mutable
    ancestry row (immutable builtin rows cannot change). A namespace edit,
    insertion or removal on any mutable row changes it."""
    var bi = mods.module("builtins")
    ref cpy = Python().cpython()
    var stamp = py_call(bi.list, extras)
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    records_key_append(stamp, ancestry)
    var dict_get = bi.type.__dict__["__dict__"].__get__
    for i in range(len(ancestry)):  # small-loop(ancestry: class rows): identity stamp of mutable rows
        var row = ancestry[i]
        records_key_append(stamp, row)
        # Py_TPFLAGS_IMMUTABLETYPE (1 << 8): attributes cannot be set.
        var flags = cpython_table()[].PyType_GetFlags(row._obj_ptr)
        if (flags & UInt64(1 << 8)) != 0:
            continue
        var items = py_call(bi.list, py_call(py_call(dict_get, row).items))
        for k in range(len(items)):  # small-loop(items: one class namespace): identity only, no hooks
            records_key_append(stamp, items[k][0])
            records_key_append(stamp, items[k][1])
    return stamp


def prepared_namespace_admitted(mut mods: NativeModules, module: PythonObject, fields: PythonObject) raises -> Bool:
    # A dictionary identity and size cannot establish unchanged keys: deleting
    # a string and inserting a non-string preserves both. Walk live keys using
    # native iteration; no Python key list, identity list or weakref cache.
    return native_namespace_string_keys(fields._obj_ptr, mods.module("builtins").str)


def prepared_namespace_admit(mut mods: NativeModules, module: PythonObject, fields: PythonObject) raises:
    # Compatibility entry for existing callers; live native admission above
    # needs no retained Python record and cannot retain the binding module.
    pass


def _class_stamp_identity(stamp: PythonObject, value: PythonObject, mut offset: Int) raises -> Bool:
    var table = cpython_table()
    if offset >= table[].PyList_Size(stamp._obj_ptr):
        return False
    var current = table[].PyList_GetItem(stamp._obj_ptr, offset)
    offset += 1
    return current == value._obj_ptr


def prepared_class_matches(mut mods: NativeModules, cls: PythonObject, extras: List[PythonObject], stamp: PythonObject) raises -> Bool:
    # Compare the retained admission directly. No new identity list, items
    # list or per-field tuple is built on the reuse path. Python >=3.10 uses
    # its supported mappingproxy iterator; PyType_GetDict requires >=3.12,
    # so this boundary still creates a proxy and iterator per mutable row.
    var offset = 0
    for i in range(len(extras)):  # small-loop(extras: fixed admission inputs): compares only connected metadata identities
        if not _class_stamp_identity(stamp, extras[i], offset):
            return False
    var bi = mods.module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    if not _class_stamp_identity(stamp, ancestry, offset):
        return False
    var dict_get = bi.type.__dict__["__dict__"].__get__
    ref cpy = Python().cpython()
    for i in range(len(ancestry)):  # driver-loop: over class ancestry metadata
        var row = ancestry[i]
        if not _class_stamp_identity(stamp, row, offset):
            return False
        var flags = cpython_table()[].PyType_GetFlags(row._obj_ptr)
        if (flags & UInt64(1 << 8)) != 0:
            continue
        var fields = py_call(dict_get, row)
        var raw_iterator = cpy.PyObject_GetIter(fields._obj_ptr)
        if not raw_iterator:
            raise cpy.unsafe_get_error()
        var iterator = PythonObject(from_owned=raw_iterator)
        var field_count = len(fields)
        var row_offset = offset
        var string_type = bi.str
        # Admit ALL live keys before any keyed value lookup. A later foreign
        # key can collide with an earlier ordinary key, so interleaving the
        # two checks could invoke its __eq__ before reaching its own position.
        for field_index in range(field_count):  # small-loop(field_count: class fields): admit every namespace key first
            var raw_key = cpy.PyIter_Next(iterator._obj_ptr)
            if not raw_key:
                if cpy.PyErr_Occurred():
                    raise cpy.unsafe_get_error()
                return False
            var key = PythonObject(from_owned=raw_key)
            if py_type(key) is not string_type or not _class_stamp_identity(stamp, key, offset):
                return False
            offset += 1  # skip the retained value in the alternating stamp
        # No user callback or GIL release occurs between passes. All live
        # keys now match exact retained strings. Read keys directly from the
        # retained stamp to avoid another iterator or temporary Python list.
        offset = row_offset
        for field_index in range(field_count):  # small-loop(field_count: class fields): compare admitted namespace values second
            var raw_key = cpython_table()[].PyList_GetItem(stamp._obj_ptr, offset)
            offset += 1
            var raw_value = cpython_table()[].PyObject_GetItem(fields._obj_ptr, raw_key)
            if not raw_value:
                raise cpy.unsafe_get_error()
            var value = PythonObject(from_owned=raw_value)
            if not _class_stamp_identity(stamp, value, offset):
                return False
    return offset == cpython_table()[].PyList_Size(stamp._obj_ptr)
