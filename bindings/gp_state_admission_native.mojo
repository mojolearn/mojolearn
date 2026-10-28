# SPDX-License-Identifier: Apache-2.0
# Shared GP consumed-state checks for execution and saved-model serialization.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.mode_metadata_native import _class_fields
from bindings.classical_attributes import _type_mro
from bindings.host_inference_scalars import host_inference_scalar
from bindings.classical_pandas import classical_admit_value


def _gp_scalar_fields(fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    for i in range(len(names)):  # driver-loop: fixed scalar-state field schema
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context, True)


def _gp_array_fields(fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    for i in range(len(names)):  # driver-loop: fixed retained-buffer schema
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)


def _gp_admit_binary_class(context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var cls = context["family_gp_binary_laplace"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: GP binary state requires ordinary native class metadata")
    var ancestry = _type_mro(cls)
    if len(ancestry) != 2 or ancestry[1] is not b.object:
        raise Error("TypeError: GP binary state ancestry changed after connection")
    var names = _class_fields(cls)
    var snapshot = context["family_gp_binary_namespace"]
    if py_type(snapshot) is not b.dict or len(names) != len(snapshot):
        raise Error("TypeError: GP binary state class changed after connection")
    var keys = py_call(b.list, names)
    for i in range(len(keys)):  # driver-loop: fixed binary-state namespace before allocation/release
        if py_type(keys[i]) is not b.str or not py_contains(snapshot, keys[i]) or names[keys[i]] is not snapshot[keys[i]]:
            raise Error("TypeError: GP binary state cannot invoke changed Python descriptors or lifecycle hooks")


