# SPDX-License-Identifier: Apache-2.0
# Internal fitted-field reads that skip the public attribute hook (lane
# perf2-shared, 2026-11-06).
#
# Native owners read their own model's fitted fields and parameters
# (`cluster_centers_`, `n_features_in_`, `labels_`, `metric`, ...). Reading
# them with `getattr(model, name)` runs the class's public attribute hook on
# every read: the native class plan admission, the instance-storage
# admission, the member resolution with descriptor-kind checks and, for a
# trailing-underscore Array, the caller-frame walk. For a name that no class
# in the model's ancestry defines, Python's own lookup order (data
# descriptor of the class, then the instance dictionary, then the class)
# gives the instance dictionary's value, and every package hook returns that
# same object for such a name. So this reader answers from the instance
# dictionary when:
#   - no ancestry row of the class defines the name (computed once per class
#     state under the type watchers of native_class_admission; recomputed on
#     any class or ancestor change),
#   - every ancestry key is an exact string (no foreign __eq__ can run), and
#     the resolved __getattribute__ is not a Python function (a Python-level
#     hook is never skipped),
#   - the instance dictionary is an exact dict with exact string keys and
#     holds the name.
# Where no type watcher is available (Python 3.10/3.11, or no free watcher
# slot) the names could not be kept, so the reader goes straight to `getattr`
# (the cost before the perf2 lanes) instead of rebuilding them per read.
# Anything else falls back to `getattr`, exactly as before: absence, a class
# member of that name, a Python hook, or a model without an instance dict.
# What it does not repeat for internal reads: the public hook's class-layout
# refusals (they still run on every public attribute read and in the doors'
# own admission), and the classical hook's ndarray view of a fitted Array
# (internal readers take an Array or an ndarray over the same storage).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import NativeModules, native_loaded_module, native_namespace_string_keys
from bindings.native_class_admission import class_record, class_admit, class_watch_ready
from bindings.native_instance_item import native_instance_find_admitted
from bindings.py_balanced import py_call, py_type, py_setitem, py_contains


comptime _CLASS_NAMES_SLOT = "perf2_fitted_field_class_names_v1"


def _class_member_names(mut mods: NativeModules, cls: PythonObject) raises -> PythonObject:
    """An exact dict of every name an ancestry row of `cls` defines, or False
    when the reader must not answer for this class (a non-string class key,
    or a Python-function __getattribute__). The payload holds strings only,
    never the class."""
    var record = class_record(mods, _CLASS_NAMES_SLOT, cls)
    if record is not None:
        return record
    var b = mods.module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var get_namespace = b.type.__dict__["__dict__"].__get__
    var function_type = native_loaded_module("types").FunctionType
    var names = Python.dict()
    var usable = True
    var hook_seen = False
    for i in range(len(mro)):  # driver-loop: over class ancestry metadata
        var namespace = py_call(get_namespace, mro[i])
        var keys = py_call(b.list, py_call(namespace.keys))
        for j in range(len(keys)):  # small-loop(keys: one class namespace): exact string names only
            if py_type(keys[j]) is not b.str:
                usable = False
                break
        if not usable:
            break
        for j in range(len(keys)):  # small-loop(keys: one class namespace): admitted exact string names
            py_setitem(names, keys[j], PythonObject(True))
        if not hook_seen and py_contains(names, PythonObject("__getattribute__")):
            # First row defining __getattribute__ is the one Python resolves.
            hook_seen = True
            if py_type(namespace["__getattribute__"]) is function_type:
                usable = False
                break
    var payload = names if usable else PythonObject(False)
    class_admit(mods, _CLASS_NAMES_SLOT, cls, payload)
    return payload


def _instance_value(mut mods: NativeModules, model: PythonObject, key: PythonObject) raises -> Tuple[Bool, PythonObject]:
    if not class_watch_ready():
        # No class record can be kept: rebuilding the ancestry names on every
        # read costs more than the public hook it skips.
        return (False, PythonObject(None))
    var b = mods.module("builtins")
    var names = _class_member_names(mods, py_type(model))
    if py_type(names) is not b.dict or py_contains(names, key):
        return (False, PythonObject(None))
    var raw = cpython_table()[].PyObject_GenericGetDict(model._obj_ptr, PyObjectPtr())
    if not raw:
        # No instance dictionary: getattr decides, with its own error.
        ref cpy = Python().cpython()
        cpython_table()[].PyErr_Clear()
        return (False, PythonObject(None))
    var fields = PythonObject(from_owned=raw)
    if py_type(fields) is not b.dict or not native_namespace_string_keys(fields._obj_ptr, b.str):
        return (False, PythonObject(None))
    return native_instance_find_admitted(fields, key)


def native_fitted_get(mut mods: NativeModules, model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)` for a package-internal read."""
    var key = PythonObject(name)
    var found = _instance_value(mods, model, key)
    if found[0]:
        return found[1]
    return py_call(mods.module("builtins").getattr, model, key)


def native_fitted_get(mut mods: NativeModules, model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)` for a package-internal read."""
    var key = PythonObject(name)
    var found = _instance_value(mods, model, key)
    if found[0]:
        return found[1]
    return py_call(mods.module("builtins").getattr, model, key, default)
