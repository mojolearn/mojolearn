# SPDX-License-Identifier: Apache-2.0
"""Shared exact host model layout and compiled-family UNPACK."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes import classical_connect_numpy
from bindings.host_tree_native import host_tree_constructor_fields

comptime _HOST_KIND_SLOT = "host_inference_kind_v1"


def host_inference_state(model: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, String]:
    classical_connect_numpy(context)
    var fields = host_tree_constructor_fields(model, context)
    var b = native_loaded_module("builtins")
    var mods = NativeModules()
    var cls = py_type(model)
    # The existing watched class record stores only the family name. It is
    # invalidated when the class or an ancestor changes, not by a new cache.
    var kind = class_record(mods, _HOST_KIND_SLOT, cls)
    if kind is None:
        var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
        for i in range(len(mro)):  # driver-loop: fixed explicit CPU class ancestry
            kind = py_get(context["host_inference_kinds"], mro[i])
            if kind is not None:
                break
        if kind is None:
            raise Error("TypeError: no native explicit CPU prediction owner for this class")
        class_admit(mods, _HOST_KIND_SLOT, cls, kind)
    return (fields^, String(kind))
