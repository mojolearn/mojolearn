# SPDX-License-Identifier: Apache-2.0
# Shared estimator storage admission before any family owner can read fields.
# This admits connected class/storage identities, not the arithmetic or every
# legacy transitive call. Existing owner-specific scalar/buffer checks remain.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_key_same
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.classical_attributes import _type_mro
from bindings.mode_metadata_native import _class_fields


def _family_descriptor(value: PythonObject) raises -> Bool:
    var ancestry = _type_mro(py_type(value))
    for i in range(len(ancestry)):  # driver-loop: descriptor slots through admitted native class namespaces
        if py_contains(_class_fields(ancestry[i]), "__get__"):
            return True
    return False


def _family_native_member(value: PythonObject, owner: PythonObject) raises -> Bool:
    var types = native_loaded_module("types")
    var kind = py_type(value)
    return (kind is types.GetSetDescriptorType or kind is types.MemberDescriptorType) and value.__objclass__ is owner


def classical_family_admit(model: PythonObject) raises -> PythonObject:
    # Functions such as metrics/resampling have no estimator storage.
    if model is None:
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native family operations refuse Python metaclass hooks")
    var module = native_loaded_module("mojolearn._native_family_layouts")
    var layouts = py_fields(module)["LAYOUTS"]
    if py_type(layouts) is not b.dict:
        raise Error("TypeError: native family layouts require their connected dictionary")
    var ancestry = _type_mro(cls)
    var plan = PythonObject(None)
    var start = -1
    for i in range(len(ancestry)):  # driver-loop: locate a connected owner without invoking subclass protocols
        if py_type(ancestry[i]) is not b.type:
            raise Error("TypeError: native family operations refuse Python metaclass hooks")
        if plan is None:
            plan = py_get(layouts, ancestry[i])
            if plan is not None:
                start = i
    if plan is None:
        raise Error("TypeError: native family operation requires a connected estimator class")
    # Prepared class admission: the ancestry, storage-hook and descriptor
    # checks below depend only on the class rows; they run on first use and
    # again whenever any mutable row's namespace changed (identity stamp of
    # every key and value, no descriptor resolution). Instance storage is
    # admitted on every call.
    var mods = NativeModules()
    var stamp = _family_class_stamp(b, layouts, plan, ancestry)
    var admitted = prepared_get(mods, _FAMILY_CLASS_SLOT, cls)
    if admitted is not None and prepared_key_same(admitted, stamp):
        return _family_instance_fields(b, model, cls, plan)
    var connected = plan["mro"]
    if len(ancestry) - start != len(connected):
        raise Error("TypeError: native family estimator ancestry changed after connection")
    for i in range(len(connected)):  # driver-loop: preserve the exact connected ancestry tail
        if ancestry[start + i] is not connected[i]:
            raise Error("TypeError: native family estimator ancestry changed after connection")
    var names = plan["names"]
    var missing = PythonObject(None)
    for hook in ["__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__", "__class__", "__dict__"]:  # small-loop(fixed storage/lookup hooks)
        var current = missing
        var found = False
        for i in range(len(ancestry)):  # driver-loop: resolve hook identities from native class namespaces
            var row = _class_fields(ancestry[i])
            if not found and py_contains(row, hook):
                current = row[hook]
                found = True
        if current is not py_get(names, hook):
            raise Error("TypeError: native family operation refuses changed Python storage hooks")
    var function_type = native_loaded_module("types").FunctionType
    var snapshots = plan["descriptors"]
    for i in range(len(ancestry)):  # driver-loop: reject descriptor replacement before any instance field read
        var owner = ancestry[i]
        var row = _class_fields(owner)
        if i >= start:
            var saved = snapshots[i - start][1]
            var saved_keys = py_call(b.list, saved)
            for k in range(len(saved_keys)):  # driver-loop: pinned connected descriptor identities
                if py_get(row, saved_keys[k]) is not saved[saved_keys[k]]:
                    raise Error("TypeError: native family estimator descriptor changed after connection")
        var keys = py_call(b.list, row)
        for k in range(len(keys)):  # driver-loop: detect new properties before legacy owners use ordinary attribute reads
            var name = keys[k]
            var member = row[name]
            if not _family_descriptor(member):
                continue
            if i >= start:
                if not py_contains(snapshots[i - start][1], name):
                    raise Error("TypeError: native family operation refuses new Python descriptors")
            elif _family_native_member(member, owner):
                # Native slot/dict descriptors on a passive subclass remain
                # storage, not Python callbacks. The connected getter below
                # reads the inherited dictionary without resolving __dict__.
                continue
            elif py_type(member) is function_type and (String(name) == "__init__" or not py_contains(names, name)):
                # Application constructors and unrelated methods may exist;
                # no such Python function is invoked or rebound by this guard.
                continue
            elif py_get(names, name) is not member:
                raise Error("TypeError: native family operation refuses overridden Python descriptors")
    prepared_put(mods, _FAMILY_CLASS_SLOT, cls, stamp)
    return _family_instance_fields(b, model, cls, plan)


def _family_instance_fields(b: PythonObject, model: PythonObject, cls: PythonObject, plan: PythonObject) raises -> PythonObject:
    var fields = py_call(plan["fields_get"], model, cls)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native family estimator requires an ordinary instance dictionary")
    var keys = py_call(b.list, fields)
    for k in range(len(keys)):  # driver-loop: exact keys before any owner dictionary lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native family estimator state requires exact string keys")
    return fields


comptime _FAMILY_CLASS_SLOT = "classical_family_class_v1"


def _family_class_stamp(b: PythonObject, layouts: PythonObject, plan: PythonObject, ancestry: PythonObject) raises -> PythonObject:
    # Identity stamp of everything the class admission reads: the layout
    # table and plan, the ancestry tuple, and every key and value of each
    # mutable ancestry row (immutable builtin rows cannot change).
    ref cpy = Python().cpython()
    var stamp = b.list()
    _ = py_call(stamp.append, layouts)
    _ = py_call(stamp.append, plan)
    _ = py_call(stamp.append, ancestry)
    var dict_get = b.type.__dict__["__dict__"].__get__
    for i in range(len(ancestry)):  # driver-loop: over the estimator class ancestry rows
        var row = ancestry[i]
        _ = py_call(stamp.append, row)
        # Py_TPFLAGS_IMMUTABLETYPE (1 << 8): attributes cannot be set.
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(row._obj_ptr)
        if (flags & UInt64(1 << 8)) != 0:
            continue
        var items = py_call(b.list, py_call(py_call(dict_get, row).items))
        for k in range(len(items)):  # driver-loop: over one class namespace, identity only
            _ = py_call(stamp.append, items[k][0])
            _ = py_call(stamp.append, items[k][1])
    return stamp
