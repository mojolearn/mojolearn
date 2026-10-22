# SPDX-License-Identifier: Apache-2.0
# Native public cache lifecycle operations. Allocation, release and resident
# scope cleanup are separate callers; this entry owns only the named operation.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type
from bindings.optimizer_public_words import stash_python_error
from bindings.arena_cache_stats import _attribute
from bindings.arena_cache_upload import _native_call


def arena_cache_lifecycle(model: PythonObject, operation: String,
                          context: PythonObject) raises -> PythonObject:
    if operation == "cache_enter":
        # Context entry has always returned self without inspecting fields.
        return model
    if operation == "cache_live":
        var missing = Python.list()
        var entry = _attribute(model,"_live",missing,False,context)
        if entry is missing:
            # Admission above ruled out attribute hooks. Let the native slot
            # preserve its exact missing/unset attribute exception text.
            ref cpy = Python().cpython()
            var name = PythonObject("_live")
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(
                model._obj_ptr,name._obj_ptr)
            _ = name^
            if not pointer:
                var failure = stash_python_error()
                raise failure^
            entry = PythonObject(from_owned=pointer)
        if entry is None:
            return PythonObject(None)
        var entries = context["cache_live_entries"]
        var admitted = False
        for i in range(len(entries)):  # driver-loop: fixed import-connected resident-store counters
            admitted = admitted or entry is entries[i]
        if not admitted:
            raise Error("TypeError: DeviceCache.live requires a connected native resident-store counter; Python callbacks cannot execute")
        # Both connected counter entries return an owned Python integer and
        # read the same DeviceStore.live used before. They do not release the
        # GIL, run NumPy, or call application protocols; no conversion is owed.
        var result = _native_call(entry,Python.tuple())
        if py_type(result) is not context["arena_native_types"]["int"]:
            raise Error("TypeError: native resident-store counter returned a non-integer")
        return result
    raise Error("ValueError: unknown native cache lifecycle operation")
