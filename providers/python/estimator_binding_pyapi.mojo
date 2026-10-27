# SPDX-License-Identifier: Apache-2.0
# Shared import-connected binding admission for estimator frontends. The caller
# enters compiled code once; runtime selection reads current mode/registry data
# and never invokes Python _bind/load_set/_ModeSet.__getattr__ callbacks.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields


def estimator_binding(model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var backend = context["backend_fields"]
    var mode = py_get(py_fields(model), "numeric_mode")
    if mode is None:
        mode = backend["_DEFAULT_MODE"]
    if py_type(mode) is not bi.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    mode = mode.strip().lower()
    var requested = String(mode)
    if requested != "identical" and requested != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var record = context["family_binding_connections"][name][requested]
    if backend["_CPU_ONLY"] is not None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        # Existing host/stub behavior remains explicit. The native operation
        # table separately decides whether this engine has a host provider.
        return record[0]
    var sets = backend["_SETS"]
    if py_type(sets) is not bi.dict:
        raise Error("TypeError: native binding registry must be an exact dictionary")
    var selected = py_get(sets, requested)
    if selected is None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        raise Error("TypeError: native binding requires its import-connected mode set")
    if py_type(selected) is not context["family_mode_set_type"]:
        raise Error("TypeError: native binding requires its import-connected mode set")
    # Direct native dictionary descriptor, pinned at import. Replacing a
    # Python __getattribute__/__dict__ hook cannot run through this lookup.
    var fields = py_call(context["family_mode_set_fields"], selected, context["family_mode_set_type"])
    var modules = fields["_modules"]
    if py_type(modules) is not bi.dict:
        raise Error("TypeError: native binding modules must be an exact dictionary")
    var module = py_get(modules, name)
    if module is None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        raise Error("ImportError: native binding was removed after import-time connection: " + name)
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: native binding must be an exact extension module")
    return module


def estimator_arguments(model: PythonObject, name: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # Preserve explicit native/internal module arguments. Thin public calls
    # reserve None for binding admission here, once after entering native code.
    if len(args) == 0 or args[0] is not None:
        return args
    var selected = estimator_binding(model, name, context)
    var values = Python.list(selected)
    for i in range(1, len(args)):  # driver-loop: over fixed public method arguments
        _ = py_call(values.append, args[i])
    return py_call(native_loaded_module("builtins").tuple, values)
