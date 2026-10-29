# SPDX-License-Identifier: Apache-2.0
# Import-connected saved-host routes. An inherited public method must keep
# its CPU owner even when the installation also has a GPU numeric-mode set.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.py_balanced import py_type, py_get
from bindings.mode_metadata_native import _fields
from bindings.classical_attributes import _type_mro
from providers.python.estimator_binding_pyapi import estimator_binding_mode, estimator_checked_mode


def host_estimator_binding(model: PythonObject, name: String, requested: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var records = py_get(context, "family_host_binding_connections")
    if records is None or model is None:
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    if py_type(records) is not b.tuple:
        raise Error("TypeError: saved host binding records must retain their native tuple")
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: admitted class ancestry, once per public operation
        for j in range(len(records)):  # driver-loop: fixed import-connected host family inventory
            var record = records[j]
            if py_type(record) is not b.tuple or len(record) != 5:
                raise Error("TypeError: saved host binding record changed")
            if ancestry[i] is not record[0]:
                continue
            if py_type(record[1]) is not b.str or String(record[1]) != name:
                raise Error("ImportError: saved host class has no connected CPU binding for " + name)
            if requested is not None and (py_type(requested) is not b.str or String(requested) != "identical"):
                raise Error("ValueError: saved model runs IDENTICAL only on the host")
            var module = record[2]
            if py_type(module) is b.str:
                raise Error(String(module))
            # Native namespace access admits all keys and rejects module hooks.
            # Retained export identities prevent a replacement from changing
            # this CPU route or introducing a Python callback.
            var fields = _loaded_fields(module)
            var exports = record[3]
            if py_type(exports) is not b.tuple:
                raise Error("TypeError: saved host binding exports changed")
            for k in range(len(exports)):  # driver-loop: connected native export identities
                var item = exports[k]
                if py_type(item) is not b.tuple or len(item) != 2 or py_type(item[0]) is not b.str:
                    raise Error("TypeError: saved host binding export record changed")
                if py_get(fields, item[0]) is not item[1]:
                    raise Error("TypeError: saved host binding export changed after connection")
            if py_type(record[4]) is not b.str:
                raise Error("TypeError: saved host binding mode reader changed")
            return estimator_checked_mode(module, String(record[4]), name, "identical")
    return PythonObject(None)


def host_estimator_binding_mode(model: PythonObject, requested: PythonObject, name: String,
                                context: PythonObject) raises -> PythonObject:
    var host = host_estimator_binding(model, name, requested, context)
    if host is not None:
        return host
    return estimator_binding_mode(requested, name, context)


def host_estimator_binding_current(model: PythonObject, name: String,
                                   context: PythonObject) raises -> PythonObject:
    var mode = py_get(_fields(model), "numeric_mode") if model is not None else PythonObject(None)
    return host_estimator_binding_mode(model, mode, name, context)
