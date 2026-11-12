# SPDX-License-Identifier: Apache-2.0
# Cache binding capability metadata, including the connected CPU facade's
# existing by-name refusal. No module __getattr__ or input protocol is called.
# The PACK half of the former `bindings/arena_cache_binding.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/arena_cache_binding_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item
from bindings.arena_cache_stats_unpack import _mapping, _string_keys
from bindings.arena_cache_stats_pack import _attribute
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.cpython_table import cpython_table


from bindings.arena_cache_binding_unpack import (
    _cache_cpu_refusal,
)


def _cache_entry_name(prefix: PythonObject, suffix: String,
                      context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var cls = py_type(prefix)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: DeviceCache prefix cannot invoke a custom metaclass")
    var ancestry = py_call(context["cache_type_mro"],cls)
    var unicode = False
    # Optional, not a PythonObject(None) sentinel: its presence is the old `found` flag (modular/modular#7546).
    var selected = Optional[PythonObject]()
    for i in range(len(ancestry)):  # driver-loop: admit inherited native Unicode concatenation, including NumPy str_
        unicode = unicode or ancestry[i] is b["str"]
        var mapping = _mapping(ancestry[i],context)
        if not selected and py_contains(mapping,"__add__"):
            selected = mapping["__add__"]
    var admitted = unicode and Bool(selected) and selected.value() is context["cache_str_add"]
    var types = context["cache_prefix_error_types"]
    for i in range(len(types)):  # driver-loop: exact native error-producing prefix forms cannot dispatch user arithmetic
        admitted = admitted or cls is types[i]
    if not admitted:
        raise Error("TypeError: DeviceCache prefix requires native string concatenation without Python hooks")
    ref cpy = Python().cpython()
    var right = PythonObject(suffix)
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Add")(
        prefix._obj_ptr,right._obj_ptr)
    _ = right^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def cache_binding_attribute(binding: PythonObject, name: PythonObject,
                             missing: PythonObject, required: Bool,
                             context: PythonObject) raises -> PythonObject:
    var cls = py_type(binding)
    var host = cls is context["cache_cpu_host_proxy"]
    var absent = cls is context["cache_cpu_missing_proxy"]
    if cls is context["native_module_type"] or host or absent:
        ref cpy = Python().cpython()
        var pointer = cpython_table()[].PyModule_GetDict(binding._obj_ptr)
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        var fields = PythonObject(from_borrowed=pointer)
        _string_keys(fields,context)
        if py_contains(fields,name):
            return py_item(fields, name)  # tail-call fix (2026-11-12): a std subscript returned directly is emitted as an LLVM tail call; LLVM may drop the borrowed key's stack store (Mojo 1.2.0). Read through the C API.
        # The connected CPU facades deliberately use AttributeError for
        # unknown dunder names; hasattr therefore returns false, even on a
        # missing family that raises ImportError for ordinary entry names.
        if (host or absent) and Bool(py=py_call(name.startswith,"__")):
            if not required:
                return missing
            var exception = py_call(context["cache_error_new"],context["cache_attribute_error"],name)
            cpy.lib.get_function[NoneType]("PyErr_SetObject")(
                context["cache_attribute_error"]._obj_ptr,exception._obj_ptr)
            _ = exception^
            var failure = stash_python_error()
            raise failure^
        if host and not required:
            return missing
        if host or absent:
            _cache_cpu_refusal(fields,name,host,context)
        if py_contains(fields,"__getattr__"):
            raise Error("TypeError: DeviceCache binding metadata cannot invoke module __getattr__")
        if not required:
            return missing
        # Exact ordinary module, and its dictionary has no fallback hook.
        var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetAttr")(
            binding._obj_ptr,name._obj_ptr)
        if not result:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=result)
    var result = _attribute(binding,name,missing,False,context)
    if result is not missing or not required:
        return result
    # All ordinary attribute and descriptor paths were admitted above.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(
        binding._obj_ptr,name._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def arena_cache_supports(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = args[0]
    var prefix = args[1]
    var missing = Python.list()
    var put = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_put",context),missing,False,context)
    if put is missing:
        return PythonObject(False)
    var free = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_free",context),missing,False,context)
    return PythonObject(free is not missing)
