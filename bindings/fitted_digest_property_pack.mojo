# SPDX-License-Identifier: Apache-2.0
# `digest_` for the class layouts that bind only connected descriptors
# (bindings/classical_attributes_door.mojo: Ridge, LinearRegression, the
# scalers, the neighbor queries, LogisticRegression, the model-selection
# helpers, the explainers).
#
# `digest_` is ONE property on the shared estimator base
# (python/mojolearn/_mode.py `CheckMixin.digest_`). A class read through
# CPython's generic lookup runs the property, whose body is one native call. A
# class with a connected public attribute plan reads it as a native property
# recipe (python/mojolearn/_native_public_attributes.py, kind "fitted_digest").
# The layouts above refuse every Python property: this file admits that one
# property object by identity and makes the same native call the recipe makes
# (`classical_native_call(model, "fitted_digest", (), context)`), through the
# connection python/mojolearn/_digest_fields.py holds. No Python code of the
# property runs here.
#
# Reached only where the lookup already found a data descriptor the class
# layout does not know (the branch that used to raise): no ordinary attribute
# read enters it. It imports no model owner (the call goes through the
# connected compiled entry), so it adds none to its importers' closures.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_mode_recipe_pack import mode_recipe_call
from bindings.py_balanced import py_fields, py_get, py_item


def fitted_digest_property(member: PythonObject) raises -> PythonObject:
    """The `digest_` connection when `member` is the shared base's `digest_`
    property object, else None."""
    var fields = py_fields(native_loaded_module("mojolearn._digest_fields"))
    var connection = py_item(fields, PythonObject("CONNECTION"))
    if py_get(connection, "property") is not member:
        return PythonObject(None)
    return connection


def fitted_digest_property_read(connection: PythonObject, model: PythonObject) raises -> PythonObject:
    """`model.digest_` through the connected compiled entry."""
    var entry = py_get(connection, "entry")
    if entry is None:
        raise Error("AttributeError: digest_ is unavailable: the saved-model owner is not connected in this installation")
    var context = py_item(connection, PythonObject("context"))
    var result = mode_recipe_call(entry, Python.tuple(model, PythonObject("fitted_digest"), Python.tuple(), context))
    return result
