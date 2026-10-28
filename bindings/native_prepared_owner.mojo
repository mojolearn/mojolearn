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
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, ExceptionType, raise_python_exception
from bindings.native_loaded_module import NativeModules, native_namespace_string_keys
from bindings.py_balanced import py_call, py_type, py_get, py_setitem


struct PreparedEntry(Movable):
    var slot: String
    var owner_address: Int
    var weak: PythonObject
    var record: PythonObject

    def __init__(out self, var slot: String, owner_address: Int, weak: PythonObject, record: PythonObject):
        self.slot = slot^
        self.owner_address = owner_address
        self.weak = weak
        self.record = record


struct PreparedOwners(Defaultable, Movable):
    var slots: Dict[String, Dict[Int, Int]]
    var entries: Dict[Int, PreparedEntry]
    var next_id: Int
    var cleanup_module: PythonObject
    var cleanup_function: PythonObject
    var weakref_get_ref: Bool

    def __init__(out self):
        self.slots = Dict[String, Dict[Int, Int]]()
        self.entries = Dict[Int, PreparedEntry]()
        self.next_id = 1
        self.cleanup_module = PythonObject(None)
        self.cleanup_function = PythonObject(None)
        self.weakref_get_ref = False


comptime PREPARED_OWNERS = _Global[
    StorageType=PreparedOwners, name="MojolearnPreparedOwnersV2",
    init_fn=PreparedOwners.__init__,
]


def _prepared_release(identifier: Int) raises:
    var g = PREPARED_OWNERS.get_or_create_ptr()
    if identifier not in g[].entries:
        return
    # Move the entry OUT before releasing any Python references. Its record
    # may release buffers or trigger another owner's weakref callback; that
    # reentrant cleanup must see fully updated maps and no live Dict borrow.
    var released = g[].entries.pop(identifier)
    if released.slot in g[].slots:
        if released.owner_address in g[].slots[released.slot]:
            if g[].slots[released.slot][released.owner_address] == identifier:
                ref slot_entries = g[].slots[released.slot]
                _ = slot_entries.pop(released.owner_address)
    # released is destroyed only after both maps have been updated. A stale
    # callback cannot erase a replacement: every record gets a unique ID.


def _prepared_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # A native C entry constructed by the supported Mojo module builder.
    # PyMethod_New binds an opaque integer ID; weakref supplies its own object
    # as argument two. No Python function implementation is invoked.
    try:
        ref cpy = Python().cpython()
        if cpy.lib.get_function[Int]("PyTuple_Size")(args_ptr) != 2:
            raise Error("prepared owner cleanup requires its native ID and weakref")
        var raw_id = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(args_ptr, 0)
        var identifier = cpy.lib.get_function[Int64]("PyLong_AsLongLong")(raw_id)
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        _prepared_release(Int(identifier))
        return PythonObject(None).steal_data()
    except error:
        return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))


def _prepared_cleanup_function() raises -> PythonObject:
    var g = PREPARED_OWNERS.get_or_create_ptr()
    if g[].cleanup_function is None:
        # The standard builder owns the method definition and creates the
        # native C callable. Keep its finalized module alive in this native
        # global; no sys.modules write, import, Python function, custom ABI
        # struct layout or borrowed temporary PyMethodDef is involved.
        var module = PythonModuleBuilder("_mojolearn_prepared_lifetime")
        module.def_py_c_function(_prepared_finalize_c, "release")
        var native_module = module.finalize()
        ref cpy = Python().cpython()
        var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(native_module._obj_ptr)
        var name = String("release")
        var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemString")(fields, name.as_c_string_slice().unsafe_ptr())
        if not raw:
            raise Error("prepared owner native cleanup entry is absent")
        g[].cleanup_module = native_module
        g[].cleanup_function = PythonObject(from_owned=cpy.Py_NewRef(raw))
        var version_ptr = cpy.lib.get_function[Pointer[Int8, MutUntrackedOrigin]]("Py_GetVersion")()
        var version = String(unsafe_from_utf8_ptr=version_ptr)
        # GetRef entered the Stable ABI in 3.13. GetObject is available at
        # the 3.10 floor but removed in 3.15; select once during cold setup.
        g[].weakref_get_ref = not (version.startswith("3.10.") or version.startswith("3.11.") or version.startswith("3.12."))
    return g[].cleanup_function


def prepared_get(mut mods: NativeModules, slot: String, owner: PythonObject) raises -> PythonObject:
    # Native slot/address lookup: no Python dict, boxed identity integer,
    # temporary record list or Python-object method dispatch on a cache hit.
    var g = PREPARED_OWNERS.get_or_create_ptr()
    var address = Int(owner._obj_ptr)
    if slot not in g[].slots or address not in g[].slots[slot]:
        return PythonObject(None)
    var identifier = g[].slots[slot][address]
    ref cpy = Python().cpython()
    # Weakref admission uses public native APIs, never Python-object call
    # dispatch. Keep this check even with prompt cleanup: an exceptional
    # failure invoking a destruction callback must not admit a recycled address.
    var weak = g[].entries[identifier].weak._obj_ptr
    if g[].weakref_get_ref:
        var raw = PyObjectPtr()
        var status = cpy.lib.get_function[Int32]("PyWeakref_GetRef")(weak, Pointer(to=raw))
        if status < 0:
            raise cpy.unsafe_get_error()
        if status == 0:
            return PythonObject(None)
        var held = PythonObject(from_owned=raw)
        if held is not owner:
            return PythonObject(None)
    else:
        # <=3.12 only. Borrowed referent is safe while GIL is held and the
        # public call strongly owns the supplied owner argument.
        var raw = cpy.lib.get_function[PyObjectPtr]("PyWeakref_GetObject")(weak)
        if not raw:
            raise cpy.unsafe_get_error()
        if raw != owner._obj_ptr:
            return PythonObject(None)
    return g[].entries[identifier].record


def prepared_put(mut mods: NativeModules, slot: String, owner: PythonObject, record: PythonObject) raises:
    var g = PREPARED_OWNERS.get_or_create_ptr()
    if g[].next_id == 9223372036854775807:
        raise Error("prepared owner registry exhausted native record IDs")
    var identifier = g[].next_id
    g[].next_id += 1
    ref cpy = Python().cpython()
    var function = _prepared_cleanup_function()
    var token = PythonObject(identifier)
    # Documented public APIs: PyMethod_New binds an object without executing
    # it; PyWeakref_NewRef invokes the native callback promptly on owner death.
    # https://docs.python.org/3/c-api/method.html#c.PyMethod_New
    # https://docs.python.org/3/c-api/weakref.html#c.PyWeakref_NewRef
    var raw_callback = cpy.lib.get_function[PyObjectPtr]("PyMethod_New")(function._obj_ptr, token._obj_ptr)
    if not raw_callback:
        raise cpy.unsafe_get_error()
    var callback = PythonObject(from_owned=raw_callback)
    var raw_weak = cpy.lib.get_function[PyObjectPtr]("PyWeakref_NewRef")(owner._obj_ptr, callback._obj_ptr)
    if not raw_weak:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return
    var weak = PythonObject(from_owned=raw_weak)
    var address = Int(owner._obj_ptr)
    var previous = 0
    if slot not in g[].slots:
        g[].slots[slot] = Dict[Int, Int]()
    elif address in g[].slots[slot]:
        previous = g[].slots[slot][address]
    g[].entries[identifier] = PreparedEntry(slot.copy(), address, weak, record)
    g[].slots[slot][address] = identifier
    # Publish replacement before destroying the old record, matching mapping
    # replacement behavior even if that destruction reenters this registry.
    if previous != 0:
        _prepared_release(previous)


def prepared_drop(mut mods: NativeModules, slot: String, owner: PythonObject) raises:
    var g = PREPARED_OWNERS.get_or_create_ptr()
    var address = Int(owner._obj_ptr)
    if slot in g[].slots and address in g[].slots[slot]:
        _prepared_release(g[].slots[slot][address])


def prepared_key_add(mut mods: NativeModules, key: PythonObject, value: PythonObject, depth: Int = 0, limit: Int = 4) raises:
    """Append `value` and, for exact list/tuple/dict containers, their items
    (to `limit` levels; deeper containers are compared by identity only)."""
    _ = py_call(key.append, value)
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
            _ = py_call(key.append, items[i][0])
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
        _ = py_call(key.append, name)
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
    _ = py_call(stamp.append, ancestry)
    var dict_get = bi.type.__dict__["__dict__"].__get__
    for i in range(len(ancestry)):  # small-loop(ancestry: class rows): identity stamp of mutable rows
        var row = ancestry[i]
        _ = py_call(stamp.append, row)
        # Py_TPFLAGS_IMMUTABLETYPE (1 << 8): attributes cannot be set.
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(row._obj_ptr)
        if (flags & UInt64(1 << 8)) != 0:
            continue
        var items = py_call(bi.list, py_call(py_call(dict_get, row).items))
        for k in range(len(items)):  # small-loop(items: one class namespace): identity only, no hooks
            _ = py_call(stamp.append, items[k][0])
            _ = py_call(stamp.append, items[k][1])
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
    ref cpy = Python().cpython()
    if offset >= cpy.lib.get_function[Int]("PyList_Size")(stamp._obj_ptr):
        return False
    var current = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")(stamp._obj_ptr, offset)
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
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(row._obj_ptr)
        if (flags & UInt64(1 << 8)) != 0:
            continue
        var fields = py_call(dict_get, row)
        var raw_iterator = cpy.lib.get_function[PyObjectPtr]("PyObject_GetIter")(fields._obj_ptr)
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
            var raw_key = cpy.lib.get_function[PyObjectPtr]("PyIter_Next")(iterator._obj_ptr)
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
            var raw_key = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")(stamp._obj_ptr, offset)
            offset += 1
            var raw_value = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(fields._obj_ptr, raw_key)
            if not raw_value:
                raise cpy.unsafe_get_error()
            var value = PythonObject(from_owned=raw_value)
            if not _class_stamp_identity(stamp, value, offset):
                return False
    return offset == cpy.lib.get_function[Int]("PyList_Size")(stamp._obj_ptr)
