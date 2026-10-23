# SPDX-License-Identifier: Apache-2.0
# Native model_selection._Resident lifecycle: constructor/reinitialization,
# entry, take, release, close and context enter/exit, one native call each.
# Lifetime and error order follow bindings/arena_cache_release.mojo: admission
# first, the emptied mapping published before the first GIL-releasing free,
# every slot attempted, the first failure retained, failed entries published
# again for a retry, and the owners (NumPy, pandas, Arrow-backed pandas, ...)
# released last, after every free and publication. No owner graph is
# inspected: an owner's own finalizer runs after the native work and cannot
# observe a half-closed resident or replace the first error.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.arena_cache_release import FirstFailure
from bindings.arena_cache_upload import _native_call
from bindings.arena_materialize_native import arena_resident_take, arena_resident_entry


def _resident_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var cls = py_type(model)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: resident requires native class metadata")
    var ancestry = py_call(b["type"].__dict__["__mro__"].__get__, cls)
    var expected = context["arena_resident_class"]
    var descriptor = context["arena_resident_dict"]
    if len(ancestry) < 2 or ancestry[len(ancestry)-2] is not expected or ancestry[len(ancestry)-1] is not b["object"]:
        raise Error("TypeError: resident requires its ordinary class or an ordinary subclass")
    for i in range(len(ancestry)-1):  # driver-loop: native instance metadata bypasses application descriptors and attribute hooks
        var mapping = py_call(b["type"].__dict__["__dict__"].__get__, ancestry[i])
        var keys = py_call(b["list"], mapping)
        for j in range(len(keys)):  # driver-loop: class lookup cannot trigger key equality
            if py_type(keys[j]) is not b["str"]:
                raise Error("TypeError: resident class keys must be native strings")
        if py_contains(mapping, "__getattribute__") or py_contains(mapping, "__setattr__"):
            raise Error("TypeError: resident cannot invoke application attribute hooks")
        var dictionary = py_get(mapping, "__dict__")
        if dictionary is not None and dictionary is not descriptor:
            raise Error("TypeError: resident requires its native dictionary descriptor")
    var fields = py_call(descriptor.__get__, model, expected)
    if py_type(fields) is not b["dict"]:
        raise Error("TypeError: resident requires a native dictionary")
    var keys = py_call(b["list"], fields)
    for i in range(len(keys)):  # driver-loop: instance metadata key admission precedes lookup and mutation
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: resident fields require native string keys")
    return fields


def _connected(fields: PythonObject, context: PythonObject) raises:
    if py_get(fields, "_put") is not context["arena_msel_put"] or py_get(fields, "_free") is not context["arena_msel_free"] or py_get(fields, "_take") is not context["arena_msel_take_rows"]:
        raise Error("TypeError: resident lifecycle requires its connected native entries")


def _entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var entries = py_get(fields, "_entries")
    if py_type(entries) is not b["dict"]:
        raise Error("TypeError: resident entries require their native dictionary")
    var keys = py_call(b["list"], entries)
    for i in range(len(keys)):  # driver-loop: lookup by identity cannot invoke a custom integer key protocol
        if py_type(keys[i]) is not b["int"]:
            raise Error("TypeError: resident entry keys must be native integers")
    return entries


def _checked(key: PythonObject, entry: PythonObject, context: PythonObject) raises:
    var b = context["arena_native_types"]
    if py_type(entry) is not b["tuple"] or len(entry) != 3 or py_type(entry[2]) is not b["int"]:
        raise Error("TypeError: resident entries require (input, array, native integer id)")
    if not Bool(py=key == py_call(context["arena_id"], entry[0])):
        raise Error("TypeError: resident entry identity differs from its retained input")


def _free(sid: PythonObject, context: PythonObject) raises:
    if Int(py=sid) >= 0:
        # The backend's own exception (type and text) is retained unchanged.
        _ = _native_call(context["arena_msel_free"], Python.tuple(sid))


def _republish(fields: PythonObject, failed: PythonObject, context: PythonObject) raises:
    var current = _entries(fields, context)
    for i in range(len(failed)):  # driver-loop: failed entries keep their owner for a retry
        if not py_contains(current, failed[i][0]):
            py_setitem(current, failed[i][0], failed[i][1])


def _close(fields: PythonObject, context: PythonObject) raises:
    var b = context["arena_native_types"]
    var pairs = py_call(b["list"], py_call(_entries(fields, context).items))
    var slots = List[Int]()
    for i in range(len(pairs)):  # driver-loop: validate the complete cleanup before releasing its first slot
        _checked(pairs[i][0], pairs[i][1], context)
        var sid = Int(py=pairs[i][1][2])
        for j in range(len(slots)):  # driver-loop: malformed duplicate ownership cannot free one native slot twice
            if sid >= 0 and sid == slots[j]:
                raise Error("TypeError: resident slots must have unique owners")
        slots.append(sid)
    # The old method's swap order: publish the new mapping, then free. An
    # application holding the old dictionary keeps its unchanged snapshot.
    py_setitem(fields, "_entries", Python.dict())
    var first = FirstFailure()
    var failed = Python.list()
    for i in range(len(pairs)):  # driver-loop: attempt every independent release
        try:
            _free(pairs[i][1][2], context)
        except error:
            first.keep(error)
            _ = py_call(failed.append, pairs[i])
    if len(failed):
        try:
            _republish(fields, failed, context)
        except error:
            first.keep(error)
    # Owners are released here, after every free and publication.
    _ = failed^
    _ = pairs^
    first.finish()


def _release(fields: PythonObject, values: PythonObject, context: PythonObject) raises:
    var entries = _entries(fields, context)
    var key = py_call(context["arena_id"], values)
    var hit = py_get(entries, key)
    if hit is None:
        return
    _checked(key, hit, context)
    if hit[0] is not values:
        return
    # Unpublish before the GIL-releasing free; the existing dictionary is
    # mutated in place, as before.
    _ = py_call(entries.pop, key)
    _ = entries^
    var first = FirstFailure()
    try:
        _free(hit[2], context)
    except error:
        first.keep(error)
        try:
            _republish(fields, Python.list(Python.tuple(key, hit)), context)
        except republish_error:
            first.keep(republish_error)
    _ = hit^
    first.finish()


def arena_resident_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    var fields = _resident_fields(model, context)
    if operation == "msel_resident_enter":
        return model
    if operation == "msel_resident_init":
        if py_contains(fields, "_entries"):
            # Reinitialization frees the live entries (the old constructor
            # lost them). A failed free keeps the old fields and the failed
            # entries published.
            if len(_entries(fields, context)):
                _connected(fields, context)
                _close(fields, context)
        py_setitem(fields, "_put", context["arena_msel_put"])
        py_setitem(fields, "_free", context["arena_msel_free"])
        py_setitem(fields, "_take", context["arena_msel_take_rows"])
        py_setitem(fields, "_entries", Python.dict())
        return PythonObject(None)
    _connected(fields, context)
    if operation == "msel_resident_take":
        return arena_resident_take(fields, args[0], args[1], context)
    if operation == "msel_resident_entry":
        var dtype = args[1]
        var indices = False
        if dtype is not None:
            if py_type(dtype) is not context["arena_native_types"]["str"] or String(py=dtype) != "<i8":
                raise Error("ValueError: resident entries cast only to the int64 index layout")
            indices = True
        return arena_resident_entry(fields, args[0], indices, context)
    if operation == "msel_resident_release":
        _release(fields, args[0], context)
        return PythonObject(None)
    if operation == "msel_resident_close":
        _close(fields, context)
        return PythonObject(None)
    if operation == "msel_resident_exit":
        _close(fields, context)
        return PythonObject(False)
    raise Error("ValueError: unknown native resident lifecycle operation")
