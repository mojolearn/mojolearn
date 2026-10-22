# SPDX-License-Identifier: Apache-2.0
# Native model-selection resident lifetime. Row preparation remains a separate
# operation; these entries own constructor state, release and deterministic close.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.native_owner_lifetime import native_release_admit
from bindings.pandas_owner_lifetime import resident_release_admit
from bindings.optimizer_public_words import stash_python_error


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
        if py_contains(mapping, "__del__") or py_contains(mapping, "__getattribute__") or py_contains(mapping, "__setattr__"):
            raise Error("TypeError: resident cannot invoke application attribute or lifecycle hooks")
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


def _selected(entries: PythonObject, key: PythonObject, all_entries: Bool,
              context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var pairs = py_call(b["list"], entries.items())
    var selected = Python.list()
    var slots = List[Int]()
    for i in range(len(pairs)):  # driver-loop: validate the complete chosen cleanup before releasing its first slot
        if not all_entries and not Bool(py=pairs[i][0] == key):
            continue
        var entry = pairs[i][1]
        if py_type(entry) is not b["tuple"] or len(entry) != 3 or py_type(entry[2]) is not b["int"]:
            raise Error("TypeError: resident entries require (input, array, native integer id)")
        if not Bool(py=pairs[i][0] == py_call(context["arena_id"], entry[0])):
            raise Error("TypeError: resident entry identity differs from its retained input")
        var sid = Int(py=entry[2])
        for j in range(len(slots)):  # driver-loop: malformed duplicate ownership cannot free the same native slot twice
            if sid >= 0 and sid == slots[j]:
                raise Error("TypeError: resident slots must have unique owners")
        slots.append(sid)
        _ = py_call(selected.append, Python.tuple(pairs[i][0], entry, Python.tuple(entry[2])))
    return selected


def _release_selected(fields: PythonObject, entries: PythonObject, selected: PythonObject,
                      replace: Bool, context: PythonObject) raises:
    var guard = resident_release_admit(selected, context)
    # close/reinitialization replace the mapping, as before. An application
    # holding the old dictionary keeps its snapshot; release mutates only the
    # existing dictionary. Failed entries remain in the published new mapping.
    var remaining = py_call(context["arena_native_types"]["dict"], entries) if replace else entries
    var entry_name = PythonObject("_entries")
    # Resolve every public C API before the first store mutation. The first
    # backend exception has its own three owned references, so a later failure
    # cannot overwrite the shared pending-exception bridge.
    ref cpy = Python().cpython()
    var call = cpy.lib.get_function[PyObjectPtr]("PyObject_CallObject")
    var fetch = cpy.lib.get_function[NoneType]("PyErr_Fetch")
    var restore = cpy.lib.get_function[NoneType]("PyErr_Restore")
    var clear = cpy.lib.get_function[NoneType]("PyErr_Clear")
    var remove = cpy.lib.get_function[Int32]("PyDict_DelItem")
    var publish = cpy.lib.get_function[Int32]("PyDict_SetItem")
    var first = List[Int](length=3, fill=0)
    var first_ptr = first.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var free = context["arena_msel_free"]
    for i in range(len(selected)):  # driver-loop: attempt every independent release, retaining failed entries for a later retry
        var pair = selected[i]
        var sid = Int(py=pair[1][2])
        var freed = True
        if sid >= 0:
            var result = call(free._obj_ptr, pair[2]._obj_ptr)
            if not result:
                freed = False
                if first[0] == 0:
                    fetch(first_ptr, first_ptr + 1, first_ptr + 2)
                else:
                    clear()
            else:
                var ignored = PythonObject(from_owned=result)
                _ = ignored^
        if freed:
            # Keys and dictionary are exact builtins. Deletion neither hashes
            # application objects nor releases an unadmitted retained owner.
            if remove(remaining._obj_ptr, pair[0]._obj_ptr) != 0:
                if first[0] == 0:
                    fetch(first_ptr, first_ptr + 1, first_ptr + 2)
                else:
                    clear()
    if replace:
        if publish(fields._obj_ptr, entry_name._obj_ptr, remaining._obj_ptr) != 0:
            if first[0] == 0:
                fetch(first_ptr, first_ptr + 1, first_ptr + 2)
            else:
                clear()
    _ = guard^
    if first[0] != 0:
        restore(first[0], first[1], first[2])
        _ = first^
        var failure = stash_python_error()
        raise failure^
    _ = first^


def arena_resident_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    var fields = _resident_fields(model, context)
    if operation == "resident_enter":
        return model
    if operation == "resident_init":
        var replacements = Python.dict(_put=context["arena_msel_put"],
            _free=context["arena_msel_free"], _take=context["arena_msel_take_rows"])
        for name in ["_put", "_free", "_take"]:  # driver-loop: admit changed entry owners before constructor publication
            var old = py_get(fields, name)
            if old is not replacements[name]:
                native_release_admit(old, context)
        var fresh = Python.dict()
        if py_contains(fields, "_entries"):
            var entries = _entries(fields, context)
            _release_selected(fields, entries, _selected(entries, PythonObject(None), True, context), True, context)
        for name in ["_put", "_free", "_take"]:  # driver-loop: publication follows complete old ownership admission and cleanup
            py_setitem(fields, name, replacements[name])
        py_setitem(fields, "_entries", fresh)
        return PythonObject(None)
    if fields["_put"] is not context["arena_msel_put"] or fields["_free"] is not context["arena_msel_free"] or fields["_take"] is not context["arena_msel_take_rows"]:
        raise Error("TypeError: resident lifecycle requires its connected native entries")
    var entries = _entries(fields, context)
    if operation == "resident_release":
        var key = py_call(context["arena_id"], args[0])
        var hit = py_get(entries, key)
        if hit is not None:
            var selected = _selected(entries, key, False, context)
            _release_selected(fields, entries, selected, False, context)
        return PythonObject(None)
    if operation == "resident_close" or operation == "resident_exit":
        _release_selected(fields, entries, _selected(entries, PythonObject(None), True, context), True, context)
        return PythonObject(False) if operation == "resident_exit" else PythonObject(None)
    raise Error("ValueError: unknown native resident lifecycle operation")
