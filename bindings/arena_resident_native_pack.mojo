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
# The PACK half of the former `bindings/arena_resident_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/arena_resident_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.arena_cache_release_unpack import FirstFailure
from bindings.arena_cache_upload_door import _native_call
from bindings.arena_resident_rows_door import arena_resident_take, arena_resident_entry
from bindings.native_loaded_module import native_namespace_string_keys
from bindings.p3_array_arena_records_unpack import (
    P3_RESIDENT_NATIVE, p3_layout_admitted, p3_layout_admit, P3_DICT_INT_IDS, p3_dict_admitted, p3_dict_admit,
)


from bindings.arena_resident_native_unpack import (
    _close,
    _connected,
    _entries,
    _free,
    _release,
    _resident_fields,
)


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
