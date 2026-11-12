# SPDX-License-Identifier: Apache-2.0
# Native PCA constructor and scalar-parameter metadata. Public activation is
# separate: this private owner does not replace inherited Python API wrappers.
# The PACK half of the former `bindings/pca_metadata.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/pca_metadata_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.sequential_description_unpack import sequential_fields
from bindings.sequential_admission_door import sequential_admit_value
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.pca_descriptor_boundary_door import pca_connection


from bindings.pca_metadata_unpack import (
    pca_fields,
)


def pca_initialize(model: PythonObject, args: PythonObject) raises -> PythonObject:
    if len(args) != 10:
        raise Error("TypeError: native PCA initialization requires its ten constructor arguments")
    var fields = sequential_fields(model)
    var names = Python.tuple("n_components", "copy", "whiten", "svd_solver", "tol",
                             "iterated_power", "n_oversamples", "power_iteration_normalizer",
                             "random_state", "numeric_mode")
    for k in range(len(names)):  # driver-loop: over fixed constructor argument metadata
        if k != 9 or args[k] is not None or not py_contains(fields, "numeric_mode"):
            py_setitem(fields, names[k], args[k])
    return PythonObject(None)


def pca_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = pca_fields(model, context)
    var names = context["pca_parameters"]
    var result = Python.dict()
    for k in range(len(names)):  # driver-loop: over alphabetized public constructor parameters
        if not py_contains(fields, names[k]):
            raise Error("TypeError: constructor parameter is not stored under its own name")
        py_setitem(result, names[k], fields[names[k]])
    return result


def pca_set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.dict:
        raise Error("TypeError: PCA parameters must be a native dict")
    if len(values) == 0:
        return model
    var current = pca_get_params(model, context)
    var keys = py_call(b.list, values.keys())
    for k in range(len(keys)):  # driver-loop: over caller keyword names before mutation
        var key = keys[k]
        if py_type(key) is not b.str:
            raise Error("TypeError: PCA parameter names must be native strings")
        var parts = py_call(key.partition, "__")
        if not py_contains(current, parts[0]):
            raise Error("Invalid parameter '" + String(parts[0]) + "' for PCA")
        if String(parts[1]) != "":
            raise Error("AttributeError: scalar parameter has no set_params")
        py_setitem(current, key, values[key])
    var fields = sequential_fields(model)
    var stored = py_call(b.list, fields.values())
    for k in range(len(stored)):  # driver-loop: over owned values before resetting the estimator dictionary
        sequential_admit_value(stored[k], context)
    fields.clear()
    _ = py_call(fields.update, current)
    return model
