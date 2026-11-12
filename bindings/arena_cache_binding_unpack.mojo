# SPDX-License-Identifier: Apache-2.0
# Cache binding capability metadata, including the connected CPU facade's
# existing by-name refusal. No module __getattr__ or input protocol is called.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item
from bindings.arena_cache_stats_unpack import _mapping, _string_keys
from bindings.arena_cache_stats_pack import _attribute
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_cpython_table import cpython_table


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
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var joined = Optional[PythonObject](PythonObject(""))
    for i in range(len(parts)):  # driver-loop: native concatenation of import-tokenized literals and exact native Unicode values
        var part = parts[i]
        if py_type(part) is not b["str"]:
            part = values[Int(py=part)]
        joined = Optional[PythonObject](py_call(context["cache_str_add"],joined.value(),part))
    var text = joined.value()
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
