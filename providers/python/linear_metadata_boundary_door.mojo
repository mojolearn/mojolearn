# SPDX-License-Identifier: Apache-2.0
"""Public parameter/attribute storage UNPACK and PACK for native linear metadata.

The public dictionaries retain exact user parameter objects for sklearn
compatibility. Native metadata code owns names, validation and fitted-state
decisions; this boundary only admits and transfers those external owners.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.native_model_records import bump_model_generation
from bindings.native_owner_lifetime import native_release_admit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields
from core.linear_metadata import (
    linear_parameter_names, linear_constructor_names, linear_constructor_check,
    linear_parameter_head, linear_parameter_index, linear_parameter_refusal,
    linear_public_fitted_key, linear_fitted_names,
)
from core.linear_model_state import LinearModelState, linear_state_is_fitted
from providers.python.linear_state_boundary import linear_state_clear


def _linear_dict_items(value: PythonObject) raises -> Tuple[List[PythonObject], List[PythonObject]]:
    ref cpy = Python().cpython()
    var table = cpython_table()
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    var keys = List[PythonObject]()
    var values = List[PythonObject]()
    # Native C-API iteration of an admitted exact dictionary invokes no hooks.
    while table[].PyDict_Next(value._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: over external dictionary fields
        keys.append(PythonObject(from_owned=cpy.Py_NewRef(pair[0])))
        values.append(PythonObject(from_owned=cpy.Py_NewRef(pair[1])))
    _ = position
    _ = pair
    return (keys^, values^)


def _linear_names_pack(ridge: Bool, list_result: Bool) raises -> PythonObject:
    var names = linear_parameter_names(ridge)
    ref cpy = Python().cpython()
    var raw = cpy.PyList_New(len(names)) if list_result else cpy.PyTuple_New(len(names))
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    for k in range(len(names)):  # driver-loop: pack fixed public parameter names
        var name = PythonObject(names[k])
        if list_result:
            _ = cpy.PyList_SetItem(raw, k, cpy.Py_NewRef(name._obj_ptr))
        else:
            _ = cpy.PyTuple_SetItem(raw, k, cpy.Py_NewRef(name._obj_ptr))
    return result^


def classical_parameter_names(ridge: Bool) raises -> PythonObject:
    return _linear_names_pack(ridge, False)


def classical_parameter_list(ridge: Bool) raises -> PythonObject:
    return _linear_names_pack(ridge, True)


def classical_initialize(model: PythonObject, args: PythonObject, ridge: Bool,
    context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    linear_constructor_check(len(args), ridge)
    var names = linear_constructor_names(ridge)
    for k in range(len(names)):  # driver-loop: admit replaced constructor owners before publishing fields
        var name = PythonObject(names[k])
        var replacing = names[k] != "numeric_mode" or args[k] is not None or not py_contains(fields, name)
        if replacing and py_contains(fields, name) and fields[name] is not args[k]:
            native_release_admit(fields[name], context)
    for k in range(len(names)):  # driver-loop: pack constructor fields, preserving numeric_mode on None
        var name = PythonObject(names[k])
        if names[k] != "numeric_mode" or args[k] is not None or not py_contains(fields, name):
            py_setitem(fields, name, args[k])
    return PythonObject(None)


def classical_fit_replacements_admit(fields: PythonObject, ridge: Bool, context: PythonObject) raises:
    var names = linear_fitted_names(ridge)
    for k in range(len(names)):  # driver-loop: admit owners of replaced public fitted attributes
        var name = PythonObject(names[k])
        if py_contains(fields, name):
            native_release_admit(fields[name], context)


def classical_get_params(model: PythonObject, ridge: Bool) raises -> PythonObject:
    var fields = sequential_fields(model)
    var result = Python.dict()
    var names = linear_parameter_names(ridge)
    for k in range(len(names)):  # driver-loop: pack exact public constructor parameter owners
        var name = PythonObject(names[k])
        if not py_contains(fields, name):
            raise Error("TypeError: constructor parameter is not stored under its own name: " + names[k])
        py_setitem(result, name, fields[name])
    return result^


def classical_set_params(model: PythonObject, params: PythonObject, ridge: Bool,
    context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(params) is not bi.dict:
        raise Error("TypeError: parameter keywords must be a native dict")
    if len(params) == 0:
        return model
    var values = classical_get_params(model, ridge)
    var pairs = _linear_dict_items(params)
    var names = linear_parameter_names(ridge)
    for k in range(len(pairs[0])):  # driver-loop: unpack public parameter keyword metadata
        var key = pairs[0][k]
        if py_type(key) is not bi.str:
            raise Error("TypeError: parameter names must be native strings")
        var word = String(key)
        var index = linear_parameter_index(word, ridge)
        var shown = String("")
        var estimator = String("")
        if index < 0:
            estimator = String(py_type(model).__name__)
            shown = String(py_call(bi.repr, PythonObject(linear_parameter_head(word))))
        linear_parameter_refusal(word, index, shown, estimator, ridge)
        py_setitem(values, PythonObject(names[index]), pairs[1][k])
    var fields = sequential_fields(model)
    var old = _linear_dict_items(fields)
    for k in range(len(old[0])):  # driver-loop: admit reset's external dictionary keys before mutation
        if py_type(old[0][k]) is not bi.str:
            raise Error("TypeError: native linear state requires exact string keys")
    for k in range(len(old[0])):  # driver-loop: admit every removed external owner before reset
        var key = old[0][k]
        if not py_contains(values, key) or values[key] is not old[1][k]:
            native_release_admit(old[1][k], context)
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyDict_Clear")(fields._obj_ptr)
    if cpy.lib.get_function[Int32]("PyDict_Update")(fields._obj_ptr, values._obj_ptr) < 0:
        raise cpy.unsafe_get_error()
    linear_state_clear(model)
    _ = bump_model_generation(model)
    return model


def linear_fitted_unpack(model: PythonObject) raises -> Bool:
    var fields = sequential_fields(model)
    var state = LinearModelState()
    state.fitted = Bool(py=py_get(fields, "_mojolearn_fitted", False))
    if linear_state_is_fitted(state, False):
        return True
    state.fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    if linear_state_is_fitted(state, False):
        return True
    var pairs = _linear_dict_items(fields)
    var native_str = native_loaded_module("builtins").str
    for k in range(len(pairs[0])):  # driver-loop: unpack public attribute names for native fitted-state decisions
        var key = pairs[0][k]
        if py_type(key) is native_str and pairs[1][k] is not None:
            if linear_state_is_fitted(state, linear_public_fitted_key(String(key))):
                return True
    return False


def classical_is_fitted(model: PythonObject) raises -> PythonObject:
    return PythonObject(linear_fitted_unpack(model))
