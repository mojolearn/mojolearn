# SPDX-License-Identifier: Apache-2.0
# Cache binding capability metadata, including the connected CPU facade's
# existing by-name refusal. No module __getattr__ or input protocol is called.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.arena_cache_stats import _attribute, _mapping, _string_keys
from bindings.optimizer_public_words import stash_python_error


def _cache_entry_name(prefix: PythonObject, suffix: String,
                      context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var cls = py_type(prefix)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: DeviceCache prefix cannot invoke a custom metaclass")
    var ancestry = py_call(context["cache_type_mro"],cls)
    var unicode = False
    var selected = PythonObject(None)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: admit inherited native Unicode concatenation, including NumPy str_
        unicode = unicode or ancestry[i] is b["str"]
        var mapping = _mapping(ancestry[i],context)
        if not found and py_contains(mapping,"__add__"):
            selected = mapping["__add__"]
            found = True
    var admitted = unicode and selected is context["cache_str_add"]
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


def _cache_cpu_refusal(fields: PythonObject, name: PythonObject, host: Bool,
                        context: PythonObject) raises:
    var marker = String("_HostBinding__") if host else String("_NoGpuBinding__")
    var module_name = fields[PythonObject(marker + "name")]
    var reason = fields[PythonObject(marker + "reason")]
    var b = context["arena_native_types"]
    if py_type(module_name) is not b["str"] or py_type(reason) is not b["str"]:
        raise Error("TypeError: native CPU binding refusal requires stored string metadata")
    var basename = PythonObject("")
    if host:
        basename = fields["_HostBinding__basename"]
        if py_type(basename) is not b["str"]:
            raise Error("TypeError: native CPU binding refusal requires a stored host module name")
    var values = Python.tuple(module_name,name,reason,basename)
    var parts = context["cache_cpu_error_templates"][0 if host else 1]
    var text = PythonObject("")
    for i in range(len(parts)):  # driver-loop: native concatenation of import-tokenized literals and exact native Unicode values
        var part = parts[i]
        if py_type(part) is not b["str"]:
            part = values[Int(part)]
        text = py_call(context["cache_str_add"],text,part)
    var category = context["cache_no_cpu_entry"] if host else context["cache_import_error"]
    if host:
        var mapping = _mapping(category,context)
        if py_contains(mapping,"__new__") or py_contains(mapping,"__init__") or py_contains(mapping,"__del__") or py_contains(mapping,"__getattribute__"):
            raise Error("TypeError: DeviceCache cannot invoke changed NoCpuEntry exception hooks")
    # Invoke the immutable BaseException allocator directly. Calling the
    # Python heap exception class would execute a replaced __init__/__new__.
    var exception = py_call(context["cache_error_new"],category,text)
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr,exception._obj_ptr)
    _ = exception^
    var failure = stash_python_error()
    raise failure^


def cache_binding_attribute(binding: PythonObject, name: PythonObject,
                             missing: PythonObject, required: Bool,
                             context: PythonObject) raises -> PythonObject:
    var cls = py_type(binding)
    var host = cls is context["cache_cpu_host_proxy"]
    var absent = cls is context["cache_cpu_missing_proxy"]
    if cls is context["native_module_type"] or host or absent:
        ref cpy = Python().cpython()
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(binding._obj_ptr)
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        var fields = PythonObject(from_borrowed=pointer)
        _string_keys(fields,context)
        if py_contains(fields,name):
            return fields[name]
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
