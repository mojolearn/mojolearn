# SPDX-License-Identifier: Apache-2.0
"""Shared exact host model layout and compiled-family UNPACK."""
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.host_tree_native_door import host_tree_constructor_fields
from bindings.host_rest_keys_unpack import host_rest_keys
from bindings.py_balanced import py_item

comptime _HOST_KIND_SLOT = "host_inference_kind_v1"


def host_inference_state(model: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, String]:
    var nk = host_rest_keys()
    classical_connect_numpy(context)
    var fields = host_tree_constructor_fields(model, context)
    var b = native_loaded_module("builtins")
    var mods = NativeModules()
    var cls = py_type(model)
    # The existing watched class record stores only the family name. It is
    # invalidated when the class or an ancestor changes, not by a new cache.
    var kind = class_record(mods, _HOST_KIND_SLOT, cls)
    if kind is None:
        var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
        for i in range(len(mro)):  # driver-loop: fixed explicit CPU class ancestry
            kind = py_get(py_item(context, nk[].k_host_inference_kinds), mro[i])
            if kind is not None:
                break
        if kind is None:
            raise Error("TypeError: no native explicit CPU prediction owner for this class")
        class_admit(mods, _HOST_KIND_SLOT, cls, kind)
    return (fields^, String(kind))
