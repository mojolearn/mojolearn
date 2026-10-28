# SPDX-License-Identifier: Apache-2.0
# Public GBDT adapter lifecycle reuses the shared native signature binder,
# parameter reader and tree-owner release admission. Constructor parameters
# retain object identity; numerical value validation remains at fit.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.params_metadata_native import _read, _resolved
from bindings.trees_constructor_native import tree_constructor_values, tree_store_parameters


def gbdt_adapter_construct(model: PythonObject, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var prepared = tree_constructor_values(model, args, context["trees_gbdt_adapter_constructors"])
    if fields is not prepared[0]:
        raise Error("TypeError: native GBDT constructor requires stable instance storage")
    var values = prepared[1]
    # Calling __init__ again stores all supplied/default parameters but keeps
    # existing non-parameter state, just as the former field-only constructor.
    tree_store_parameters(fields, values, context, False)
    return PythonObject(None)


def gbdt_adapter_set_params(model: PythonObject, values: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.dict:
        raise Error("TypeError: native GBDT parameter updates require an ordinary dictionary")
    # The public empty update was an identity/no-op even on incomplete state.
    if len(values) == 0:
        return model
    var keys = py_call(b.list, values)
    for i in range(len(keys)):  # driver-loop: admit keyword names before lookup/sorting
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native GBDT parameter names must be exact strings")
    var replacement = _read(model, False, context["trees_gbdt_parameter_metadata"], Python.list())
    var unknown = Python.list()
    for i in range(len(keys)):  # driver-loop: the existing shallow schema rejects unknown names before mutation
        if not py_contains(replacement, keys[i]):
            _ = py_call(unknown.append, keys[i])
    if len(unknown):
        raise Error("Invalid " + String(py_type(model).__name__) + " parameters: " + String(py_call(b.repr, py_call(b.sorted, unknown))))
    _ = py_call(replacement.update, values)
    if _resolved(py_type(model), "__new__", PythonObject(None)) is not b.object.__new__:
        raise Error("TypeError: native GBDT parameter reset cannot invoke a custom allocator")
    # Reuse constructor binding and its storage-hook checks; reconstruction
    # cannot silently bypass an application's custom __setattr__ or descriptors.
    var prepared = tree_constructor_values(model,
        Python.tuple(context["trees_gbdt_adapter_tables"]["base"], Python.tuple(), replacement),
        context["trees_gbdt_adapter_constructors"])
    replacement = prepared[1]
    var fields = _fields(model)
    # Reconstruction is parameter-only: changing even an identical parameter
    # resets learned state, while retaining every selected parameter by identity.
    tree_store_parameters(fields, replacement, context, True)
    return model
