# SPDX-License-Identifier: Apache-2.0
# Public GBDT adapter lifecycle reuses the shared native signature binder,
# parameter reader and tree-owner release admission. Constructor parameters
# retain object identity; numerical value validation remains at fit.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.mode_metadata_native_door import _fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.params_metadata_native_door import _read, _resolved
from bindings.trees_constructor_native_unpack import tree_store_parameters
from bindings.trees_constructor_native_pack import tree_constructor_values
from bindings.trees_keys_unpack import trees_keys
from bindings.parameter_names_unpack import names_in_table, resolve_parameter_names
from bindings.py_balanced import py_item
from bindings.trees_route_boundary_door import trees_device_admit
from providers.python.device_route_boundary_door import device_route_clear


def gbdt_adapter_construct(model: PythonObject, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var nk = trees_keys()
    var constructors = py_item(context, nk[].k_trees_gbdt_adapter_constructors)
    var prepared: PythonObject
    try:
        prepared = tree_constructor_values(model, args, constructors)
    except error:
        # A keyword the signature does not hold may be a CatBoost / XGBoost /
        # LightGBM name (lane api-boost): only on this refused path is the
        # name table read; the binder refused before storing anything.
        var table = py_get(context, PythonObject("trees_gbdt_adapter_parameter_names"))
        if not names_in_table(args[2], table):
            raise error^
        prepared = tree_constructor_values(
            model, Python.tuple(args[0], args[1], resolve_parameter_names(args[2], table)), constructors)
    if fields is not prepared[0]:
        raise Error("TypeError: native GBDT constructor requires stable instance storage")
    var values = prepared[1]
    # `device` is checked where it is given, in the shared words, before any
    # field is published (None is one identity compare).
    trees_device_admit(py_get(values, nk[].k_device), context)
    # Calling __init__ again stores all supplied/default parameters but keeps
    # existing non-parameter state, just as the former field-only constructor.
    tree_store_parameters(fields, values, context, False)
    # The next operation reads the device once and resolves the route again.
    device_route_clear(model)
    return PythonObject(None)


def gbdt_adapter_set_params(model: PythonObject, var values: PythonObject,
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
    var nk = trees_keys()
    var replacement = _read(model, False, py_item(context, nk[].k_trees_gbdt_parameter_metadata), Python.list())
    var unknown = Python.list()
    for i in range(len(keys)):  # driver-loop: the existing shallow schema rejects unknown names before mutation
        if not py_contains(replacement, keys[i]):
            _ = py_call(unknown.append, keys[i])
    if len(unknown):
        # CatBoost / XGBoost / LightGBM names (lane api-boost), read only when
        # a name is not a parameter: stored under the canonical name, or
        # refused in the table's words; the rest stay unknown.
        var table = py_get(context, PythonObject("trees_gbdt_adapter_parameter_names"))
        if names_in_table(unknown, table):
            values = resolve_parameter_names(values, table)
            keys = py_call(b.list, values)
            unknown = Python.list()
            for i in range(len(keys)):  # driver-loop: the resolved names, against the same schema
                if not py_contains(replacement, keys[i]):
                    _ = py_call(unknown.append, keys[i])
    if len(unknown):
        raise Error("Invalid " + String(py_type(model).__name__) + " parameters: " + String(py_call(b.repr, py_call(b.sorted, unknown))))
    # `device` is checked where it is given, before any field changes.
    trees_device_admit(py_get(values, nk[].k_device), context)
    _ = py_call(replacement.update, values)
    if _resolved(py_type(model), "__new__", PythonObject(None)) is not b.object.__new__:
        raise Error("TypeError: native GBDT parameter reset cannot invoke a custom allocator")
    # Reuse constructor binding and its storage-hook checks; reconstruction
    # cannot silently bypass an application's custom __setattr__ or descriptors.
    var prepared = tree_constructor_values(model,
        Python.tuple(py_item(py_item(context, nk[].k_trees_gbdt_adapter_tables), nk[].k_base), Python.tuple(), replacement),
        py_item(context, nk[].k_trees_gbdt_adapter_constructors))
    replacement = prepared[1]
    var fields = _fields(model)
    # Reconstruction is parameter-only: changing even an identical parameter
    # resets learned state, while retaining every selected parameter by identity.
    tree_store_parameters(fields, replacement, context, True)
    device_route_clear(model)
    return model
