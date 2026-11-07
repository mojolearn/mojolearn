# SPDX-License-Identifier: Apache-2.0
"""Per-class admission records, re-checked only when the class changes.

Native metadata entries (public attributes, get_params/set_params, fitted-state
and mode read-backs) used to repeat a full class admission on every call: an
MRO walk with a `type.__dict__` key scan of every ancestry row, then the
connected-plan and hook checks. Those results depend only on the class rows
and the import-connected context, so they are kept per class here.

Validity: CPython's type watchers (Python >= 3.12, `PyType_AddWatcher` /
`PyType_Watch`). `PyType_Modified` runs on every class attribute assignment or
deletion, `__bases__` and `__class__` assignment, and propagates to every
subclass that holds a valid version tag; a watched class is notified for its own
and every ancestor's modification. A record is admitted only after
`PyUnstable_Type_AssignVersionTag` confirms the class holds a valid tag, so the
next modification of the class or any ancestor notifies the watcher, which bumps
the class epoch and invalidates every record of that class. The callback runs no
Python and touches only a native map.

Python 3.10 and 3.11 have no type watcher API. There (lane ne2-i-j,
2026-11-07) a record carries the class's own `tp_version_tag` instead of the
epoch, and is valid while the class holds `Py_TPFLAGS_VALID_VERSION_TAG` with
that tag: `PyType_Modified` clears the flag on the class and every subclass
holding one, and a new tag never repeats (bindings/py_version_tags.mojo, whose
layout is proven once per process; unproven, no record is admitted and every
call runs the full admission, as before).

Records live in the prepared-owner registry keyed by a weak reference to the
class, so a dead class's records are dropped and a recycled address never
matches. A record must not strongly retain the class it belongs to.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.perf2_shared_cpython_table import perf2_cpython_table
from bindings.native_loaded_module import NativeModules, native_loaded_module, shared_dict_watcher, shared_dict_epoch, shared_dict_watch
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.py_balanced import py_call


struct ClassWatch(Defaultable, Movable):
    # watcher: -2 not set up yet, -1 unavailable (Python < 3.12 or no free
    # watcher slot), otherwise the interpreter's type watcher ID.
    var watcher: Int
    var epochs: Dict[Int, Int]

    def __init__(out self):
        self.watcher = -2
        self.epochs = Dict[Int, Int]()


comptime CLASS_WATCH = _Global[
    StorageType=ClassWatch, name="MojolearnClassWatchV1",
    init_fn=ClassWatch.__init__,
]


def _class_modified_c(type_ptr: PyObjectPtr) abi("C") -> Int32:
    # Type watcher callback: CPython calls it with the GIL held when a
    # watched class (or one of its ancestors) is modified. Native map update
    # only; never raises into CPython.
    try:
        var g = CLASS_WATCH.get_or_create_ptr()
        var address = Int(type_ptr)
        if address in g[].epochs:
            g[].epochs[address] = g[].epochs[address] + 1
        else:
            g[].epochs[address] = 1
    except:
        pass
    return 0


def _class_watcher() raises -> Int:
    var g = CLASS_WATCH.get_or_create_ptr()
    if g[].watcher != -2:
        return g[].watcher
    ref cpy = Python().cpython()
    var version_ptr = cpy.lib.get_function[Pointer[Int8, MutUntrackedOrigin]]("Py_GetVersion")()
    var version = String(unsafe_from_utf8_ptr=version_ptr)
    if version.startswith("3.10.") or version.startswith("3.11."):
        g[].watcher = -1
        return -1
    var identifier = cpy.lib.get_function[Int32]("PyType_AddWatcher")(_class_modified_c)
    if identifier < 0:
        # All interpreter watcher slots are taken: keep the full per-call
        # admission instead of failing the operation.
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        g[].watcher = -1
        return -1
    g[].watcher = Int(identifier)
    return g[].watcher


def class_watch_ready() raises -> Bool:
    """True where class records can be kept (Python >= 3.12 with a type
    watcher, or CPython 3.10 / 3.11 with proven version tags). Where False,
    `class_admit` is a no-op and `class_record` always misses, so callers
    skip building record payloads."""
    if _class_watcher() >= 0:
        return True
    return type_version_tags()


def _class_epoch(cls: PythonObject) raises -> Int:
    var g = CLASS_WATCH.get_or_create_ptr()
    var address = Int(cls._obj_ptr)
    if address in g[].epochs:
        return g[].epochs[address]
    return 0


def _class_arm(cls: PythonObject) raises -> Bool:
    # Watch the class and confirm it holds a valid version tag; without a
    # tag CPython skips the notification, so no record may be admitted.
    var watcher = _class_watcher()
    if watcher < 0:
        return False
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyType_Watch")(Int32(watcher), cls._obj_ptr) < 0:
        # Watch refused (e.g. a sub-interpreter that does not know this
        # watcher ID): admit no record, so the caller keeps the full per-call
        # admission instead of failing the user's call.
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    return cpy.lib.get_function[Int32]("PyUnstable_Type_AssignVersionTag")(cls._obj_ptr) != 0


def class_record(mut mods: NativeModules, slot: String, cls: PythonObject) raises -> PythonObject:
    """The payload admitted for `cls` under `slot`, or None when there is no
    record or the class (or an ancestor) changed since it was admitted."""
    var g = CLASS_WATCH.get_or_create_ptr()
    if g[].watcher < 0:
        if g[].watcher == -2 or not type_version_tags():
            return PythonObject(None)
        # CPython 3.10 / 3.11: the record holds the class's version tag
        var tagged = prepared_get(mods, slot, cls)
        if tagged is None:
            return PythonObject(None)
        var tag = type_version_tag(cls)
        if tag == 0 or Int(py=tagged[0]) != tag:
            return PythonObject(None)
        return tagged[1]
    var record = prepared_get(mods, slot, cls)
    if record is None:
        return PythonObject(None)
    # Pointers resolved once per process (perf2-shared table), not three
    # dlsym lookups per record read: doors read records per MRO row.
    var t = perf2_cpython_table()
    var raw_epoch = t[].PyTuple_GetItem(record._obj_ptr, 0)
    var epoch = t[].PyLong_AsLongLong(raw_epoch)
    if Int(epoch) != _class_epoch(cls):
        return PythonObject(None)
    return PythonObject(from_borrowed=t[].PyTuple_GetItem(record._obj_ptr, 1))


def class_admit(mut mods: NativeModules, slot: String, cls: PythonObject, payload: PythonObject) raises:
    """Keep `payload` for `cls` under `slot` after a full admission passed.
    No-op where type watchers are unavailable. `payload` must not strongly
    retain `cls`."""
    if _class_watcher() < 0:
        # CPython 3.10 / 3.11: keyed on the class's version tag (assigned now
        # when it has none); no tag, no record
        if not type_version_tags():
            return
        var tag = type_version_tag_assign(cls)
        if tag == 0:
            return
        prepared_put(mods, slot, cls, Python.tuple(PythonObject(tag), payload))
        return
    if not _class_arm(cls):
        return
    prepared_put(mods, slot, cls, Python.tuple(PythonObject(_class_epoch(cls)), payload))


# ---------------------------------------------------------------- dictionaries
# Lane ne2-i-j (2026-11-07): the CPython 3.10 / 3.11 stamps.
from bindings.native_loaded_module import shared_dict_stamped
from bindings.py_version_tags import type_version_tags, type_version_tag, type_version_tag_assign
# The same idea for exact dictionaries whose key admission (every key an
# exact str, so a keyed lookup cannot run a foreign __eq__) used to be
# repeated per call: module namespaces read by native metadata. Python >=
# 3.12 dictionary watchers (`PyDict_AddWatcher` / `PyDict_Watch`) report
# every insertion, replacement, deletion, clear, clone and deallocation of a
# watched dictionary; each event bumps the dictionary's epoch, so a recycled
# address or any edit forces the full scan again. Python 3.10 and 3.11 keep
# the full scan on every call.


struct DictWatch(Defaultable, Movable):
    # address -> watch epoch at the last full key admission. The watcher and
    # the epochs are the process-wide shared dictionary watcher in
    # native_loaded_module (one CPython watcher slot for both uses).
    var admitted: Dict[Int, Int]

    def __init__(out self):
        self.admitted = Dict[Int, Int]()


comptime DICT_WATCH = _Global[
    StorageType=DictWatch, name="MojolearnDictWatchV2",
    init_fn=DictWatch.__init__,
]


def dict_keys_admitted(fields: PythonObject) raises -> Bool:
    """True when `fields` passed a full key admission and has not changed
    since (watched, or on CPython 3.10 / 3.11 its version tag unchanged).
    Always False where neither is available."""
    if not shared_dict_stamped():
        return False
    var g = DICT_WATCH.get_or_create_ptr()
    var address = Int(fields._obj_ptr)
    var seen = g[].admitted.find(address)
    if not seen:
        return False
    return seen.value() == shared_dict_epoch(address)


def dict_keys_admit(fields: PythonObject) raises:
    """Record that `fields` (an exact dict) just passed a full key admission."""
    if not shared_dict_watch(fields._obj_ptr):
        # Unavailable or refused: record nothing; the full key scan stays per call.
        return
    var address = Int(fields._obj_ptr)
    var g = DICT_WATCH.get_or_create_ptr()
    g[].admitted[address] = shared_dict_epoch(address)


# ---------------------------------------------------------------- static types
# CPython's static builtin types (builtin_function_or_method, AttributeError,
# ...) are process constants: their identity can never change. Resolve each
# once from the admitted module connection instead of re-admitting that
# module on every call that only compares a type identity.


struct StaticTypes(Defaultable, Movable):
    var types: Dict[String, PythonObject]

    def __init__(out self):
        self.types = Dict[String, PythonObject]()


comptime STATIC_TYPES = _Global[
    StorageType=StaticTypes, name="MojolearnStaticTypesV1",
    init_fn=StaticTypes.__init__,
]


def static_builtin_type(module: String, name: String) raises -> PythonObject:
    """`module.name` for a static (non-heap) builtin type, resolved once."""
    var g = STATIC_TYPES.get_or_create_ptr()
    var key = module + "." + name
    if key in g[].types:
        return g[].types[key]
    var value = py_call(native_loaded_module("builtins").getattr, native_loaded_module(module), PythonObject(name))
    ref cpy = Python().cpython()
    # Py_TPFLAGS_HEAPTYPE (1 << 9): only immortal static types are cached.
    if cpython_table()[].PyType_GetFlags(value._obj_ptr) & UInt64(1 << 9) != 0:
        return value
    g[].types[key] = value
    return value
