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

Python 3.10 and 3.11 have no type watcher API: every call there runs the full
admission exactly as before (no record is ever admitted).

Records live in the prepared-owner registry keyed by a weak reference to the
class, so a dead class's records are dropped and a recycled address never
matches. A record must not strongly retain the class it belongs to.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import NativeModules
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
        raise cpy.unsafe_get_error()
    return cpy.lib.get_function[Int32]("PyUnstable_Type_AssignVersionTag")(cls._obj_ptr) != 0


def class_record(mut mods: NativeModules, slot: String, cls: PythonObject) raises -> PythonObject:
    """The payload admitted for `cls` under `slot`, or None when there is no
    record or the class (or an ancestor) changed since it was admitted."""
    var g = CLASS_WATCH.get_or_create_ptr()
    if g[].watcher < 0:
        return PythonObject(None)
    var record = prepared_get(mods, slot, cls)
    if record is None:
        return PythonObject(None)
    ref cpy = Python().cpython()
    var raw_epoch = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(record._obj_ptr, 0)
    var epoch = cpy.lib.get_function[Int64]("PyLong_AsLongLong")(raw_epoch)
    if Int(epoch) != _class_epoch(cls):
        return PythonObject(None)
    return PythonObject(from_borrowed=cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(record._obj_ptr, 1))


def class_admit(mut mods: NativeModules, slot: String, cls: PythonObject, payload: PythonObject) raises:
    """Keep `payload` for `cls` under `slot` after a full admission passed.
    No-op where type watchers are unavailable. `payload` must not strongly
    retain `cls`."""
    if not _class_arm(cls):
        return
    prepared_put(mods, slot, cls, Python.tuple(PythonObject(_class_epoch(cls)), payload))
