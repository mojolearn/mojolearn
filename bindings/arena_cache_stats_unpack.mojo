# SPDX-License-Identifier: Apache-2.0
# Read-only cache metadata admission: the string-key walk of a class or instance
# mapping. (The entry-key hashing admission went with the entries dictionary, lane arena 2026-11-21.)
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module_door import native_namespace_string_keys, native_mapping_string_keys


def _string_keys(mapping: PythonObject, context: PythonObject) raises:
    # p3-array-arena: exact dicts and class mappingproxies are walked in place
    # (no key list); any other mapping keeps the list walk. Same refusal.
    var b = context["arena_native_types"]
    var cls = py_type(mapping)
    if cls is b["dict"]:
        if not native_namespace_string_keys(mapping._obj_ptr, b["str"]):
            raise Error("TypeError: cache metadata keys must be native strings")
        return
    if cls is py_type(py_call(context["cache_type_dict"], b["object"])):
        if not native_mapping_string_keys(mapping, b["str"]):
            raise Error("TypeError: cache metadata keys must be native strings")
        return
    var keys = py_call(b["list"],mapping)
    for i in range(len(keys)):  # driver-loop: admit metadata keys before lookup can invoke equality
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: cache metadata keys must be native strings")


def _mapping(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = py_call(context["cache_type_dict"],cls)
    _string_keys(result,context)
    return result
