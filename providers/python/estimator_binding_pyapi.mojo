# SPDX-License-Identifier: Apache-2.0
# Shared import-connected binding admission for estimator frontends. The caller
# enters compiled code once; runtime selection reads current mode/registry data
# and never invokes Python _bind/load_set/_ModeSet.__getattr__ callbacks.
from std.python import Python, PythonObject
from std.os import getenv
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry


def estimator_binding(model: PythonObject, name: String, context: PythonObject, check_mode: Bool = True) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var backend = context["backend_fields"]
    var mode = py_get(py_fields(model), "numeric_mode") if model is not None else PythonObject(None)
    if mode is None:
        mode = backend["_DEFAULT_MODE"]
    if py_type(mode) is not bi.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    mode = mode.strip().lower()
    var requested = String(mode)
    if requested != "identical" and requested != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var record = context["family_binding_connections"][name][requested]
    if not Bool(py=context["family_binding_offerings"][name][requested]):
        raise Error(String(record[0]))
    if backend["_CPU_ONLY"] is not None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        # Import setup resolves the lazy HostBinding facade to its exact native
        # module. Read the live module slot so replacement/removal is observed.
        var module = py_get(native_loaded_module("sys").modules, record[1])
        if module is None:
            raise Error("ImportError: native host binding was removed after import-time connection: " + String(record[1]))
        if py_type(module) is not py_type(bi):
            raise Error("TypeError: native host binding must be an exact extension module")
        return estimator_checked_mode(module, String(record[2]), name, requested) if check_mode else module
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
    # Getter names are immutable import metadata, but the function is read
    # from the live replacement module at every public operation.
    var reader = String(record[2]) if len(record) > 2 else String(context["family_binding_readers"][name])
    return estimator_checked_mode(module, reader, name, requested) if check_mode else module


def estimator_checked_mode(module: PythonObject, reader: String, name: String,
                           requested: String) raises -> PythonObject:
    # These families read mode from their admitted native table once per
    # operation. Preserve that single read instead of calling the legacy door
    # first; plain frontdoors below still perform the original checked readback.
    if (getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") == "1"
            and (name == "_mojolearn_svm" or name == "_mojolearn_kernel_methods" or name == "_mojolearn_x_neighbors")):
        if requested != "identical":
            raise Error("native dispatch candidate requires identical mode")
        return module
    var getter = neural_descriptor_entry(module, reader, False)
    if getter is not None:
        var got = Int(py=py_call(getter))
        var expected = 1 if requested == "identical" else 0
        if got != expected:
            var compiled = String("identical") if got == 1 else (String("fast") if got == 0 else String("unknown"))
            raise Error("RuntimeError: mojolearn: " + name + " was compiled for " + compiled
                        + ", but this call requested " + requested + "; rebuild the " + requested + " binding")
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
