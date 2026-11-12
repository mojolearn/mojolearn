# SPDX-License-Identifier: Apache-2.0
# Native PCA constructor and scalar-parameter metadata. Public activation is
# separate: this private owner does not replace inherited Python API wrappers.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.sequential_description_unpack import sequential_fields
from bindings.sequential_admission_door import sequential_admit_value
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.pca_descriptor_boundary_door import pca_connection


def pca_dictionary(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["pca_class"]
    var descriptor = context["pca_dictionary_descriptor"]
    # The finalizer and dictionary-descriptor checks read only the PCA class
    # rows: run them once per class state (native_class_admission) and
    # descriptor identity, not on every PCA call.
    var mods = NativeModules()
    var record = class_record(mods, _PCA_DICTIONARY_SLOT, cls)
    if record is not None and record is descriptor:
        return descriptor
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found = False
    for k in range(len(mro)):  # driver-loop: over native MRO dictionary descriptors before stored-field access
        var class_fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_get(class_fields, "__del__") is not None:
            raise Error("TypeError: PCA native owner cannot invoke Python finalizers")
        if not found and py_contains(class_fields, "__dict__"):
            if class_fields["__dict__"] is not descriptor:
                raise Error("TypeError: PCA instance dictionary descriptor was replaced")
            found = True
    if not found:
        raise Error("TypeError: PCA native instance dictionary is unavailable")
    class_admit(mods, _PCA_DICTIONARY_SLOT, cls, descriptor)
    return descriptor


comptime _PCA_DICTIONARY_SLOT = "pca_dictionary_descriptor_v1"


def pca_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["pca_class"]
    var descriptor = pca_dictionary(context)
    var fields = py_call(descriptor.__get__, model, cls)
    var keys = py_call(b.list, fields.keys())
    for k in range(len(keys)):  # driver-loop: over stored model field names before keyed access
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: PCA stored field names must be native strings")
    var names = context["pca_parameters"] + context["pca_state"]
    for k in range(len(names)):  # driver-loop: over connected PCA option and fitted-array metadata
        if py_contains(fields, names[k]):
            sequential_admit_value(fields[names[k]], context)
    return fields


def pca_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not native_loaded_module("builtins").str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    return mode


def pca_entries(fields: PythonObject, context: PythonObject) raises -> Int:
    return pca_connection(String(pca_mode(fields, context)), context)
