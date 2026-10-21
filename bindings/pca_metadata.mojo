# SPDX-License-Identifier: Apache-2.0
# Native PCA constructor and scalar-parameter metadata. Public activation is
# separate: this private owner does not replace inherited Python API wrappers.
from std.python import Python, PythonObject
from bindings.sequential_description import sequential_fields
from bindings.sequential_admission import sequential_admit_value


def pca_dictionary(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = context["pca_class"]
    var mro = b.type.__dict__["__mro__"].__get__(cls)
    var descriptor = context["pca_dictionary_descriptor"]
    var found = False
    for k in range(len(mro)):  # driver-loop: over native MRO dictionary descriptors before stored-field access
        var class_fields = b.type.__dict__["__dict__"].__get__(mro[k])
        if class_fields.get("__del__") is not None:
            raise Error("TypeError: PCA native owner cannot invoke Python finalizers")
        if not found and Bool(py=class_fields.__contains__("__dict__")):
            if class_fields["__dict__"] is not descriptor:
                raise Error("TypeError: PCA instance dictionary descriptor was replaced")
            found = True
    if not found:
        raise Error("TypeError: PCA native instance dictionary is unavailable")
    return descriptor


def pca_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = context["pca_class"]
    var descriptor = pca_dictionary(context)
    var fields = descriptor.__get__(model, cls)
    var keys = b.list(fields.keys())
    for k in range(len(keys)):  # driver-loop: over stored model field names before keyed access
        if b.type(keys[k]) is not b.str:
            raise Error("TypeError: PCA stored field names must be native strings")
    var names = context["pca_parameters"] + context["pca_state"]
    for k in range(len(names)):  # driver-loop: over connected PCA option and fitted-array metadata
        if Bool(py=fields.__contains__(names[k])):
            sequential_admit_value(fields[names[k]], context)
    return fields


def pca_initialize(model: PythonObject, args: PythonObject) raises -> PythonObject:
    if len(args) != 10:
        raise Error("TypeError: native PCA initialization requires its ten constructor arguments")
    var fields = sequential_fields(model)
    var names = Python.tuple("n_components", "copy", "whiten", "svd_solver", "tol",
                             "iterated_power", "n_oversamples", "power_iteration_normalizer",
                             "random_state", "numeric_mode")
    for k in range(len(names)):  # driver-loop: over fixed constructor argument metadata
        if k != 9 or args[k] is not None or not Bool(py=fields.__contains__("numeric_mode")):
            fields[names[k]] = args[k]
    return PythonObject(None)


def pca_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = pca_fields(model, context)
    var names = context["pca_parameters"]
    var result = Python.dict()
    for k in range(len(names)):  # driver-loop: over alphabetized public constructor parameters
        if not Bool(py=fields.__contains__(names[k])):
            raise Error("TypeError: constructor parameter is not stored under its own name")
        result[names[k]] = fields[names[k]]
    return result


def pca_set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if b.type(values) is not b.dict:
        raise Error("TypeError: PCA parameters must be a native dict")
    if len(values) == 0:
        return model
    var current = pca_get_params(model, context)
    var keys = b.list(values.keys())
    for k in range(len(keys)):  # driver-loop: over caller keyword names before mutation
        var key = keys[k]
        if b.type(key) is not b.str:
            raise Error("TypeError: PCA parameter names must be native strings")
        var parts = key.partition("__")
        if not Bool(py=current.__contains__(parts[0])):
            raise Error("Invalid parameter '" + String(parts[0]) + "' for PCA")
        if String(parts[1]) != "":
            raise Error("AttributeError: scalar parameter has no set_params")
        current[key] = values[key]
    var fields = sequential_fields(model)
    var stored = b.list(fields.values())
    for k in range(len(stored)):  # driver-loop: over owned values before resetting the estimator dictionary
        sequential_admit_value(stored[k], context)
    fields.clear()
    fields.update(current)
    return model


def pca_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = fields.get("numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if Python.import_module("builtins").type(mode) is not Python.import_module("builtins").str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    return mode


def pca_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = pca_mode(fields, context)
    var entries = context["pca_bindings"][mode]
    if Python.import_module("builtins").type(entries) is Python.import_module("builtins").str:
        raise Error("ImportError: " + String(entries))
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    return entries
