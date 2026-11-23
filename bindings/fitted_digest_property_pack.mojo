# SPDX-License-Identifier: Apache-2.0
# `digest_` and `training_digest_` for the class layouts that bind only connected descriptors
# (bindings/classical_attributes_door.mojo: Ridge, LinearRegression, the
# scalers, the neighbor queries, LogisticRegression, the model-selection
# helpers, the explainers).
#
# `digest_` is ONE property on the shared estimator base
# (python/mojolearn/_mode.py `CheckMixin.digest_`). A class read through
# CPython's generic lookup runs the property, whose body is one native call. A
# class with a connected public attribute plan reads it as a native property
# recipe (python/mojolearn/_native_public_attributes.py, kind "fitted_digest").
# The layouts above refuse every Python property: this file admits those two
# property objects by identity and makes the same native call the recipe makes
# (`classical_native_call(model, "fitted_digest", (), context)`, or with the
# arguments ("training",) for `training_digest_`), through the
# connection python/mojolearn/_digest_fields.py fills (held in `_mode._MODE_METADATA_CONTEXT`). No Python code of the
# property runs here.
#
# Reached only where the lookup already found a data descriptor the class
# layout does not know (the branch that used to raise): no ordinary attribute
# read enters it. It imports no model owner (the call goes through the
# connected compiled entry), so it adds none to its importers' closures.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_class_admission_door import static_builtin_type
from bindings.py_balanced import py_call, py_fields, py_get, py_item, py_type


def fitted_digest_property(member: PythonObject) raises -> PythonObject:
    """`(connection, arguments)` when `member` is the shared base's `digest_`
    property object (arguments ()) or its `training_digest_` property object
    (arguments ("training",)), else None."""
    # The connection rides in the mode metadata context, an export the
    # module registry already holds for `mojolearn._mode` (no new owner).
    var fields = py_fields(native_loaded_module("mojolearn._mode"))
    var metadata = py_item(fields, PythonObject("_MODE_METADATA_CONTEXT"))
    var connection = py_get(metadata, "digest_connection")
    if connection is None:
        return PythonObject(None)
    if py_get(connection, "property") is member:
        return Python.tuple(connection, Python.tuple())
    if py_get(connection, "training_property") is member:
        return Python.tuple(connection, Python.tuple(PythonObject("training")))
    return PythonObject(None)


def fitted_digest_property_read(found: PythonObject, model: PythonObject) raises -> PythonObject:
    """`model.digest_` or `model.training_digest_` through the connected
    compiled entry (`found` is what `fitted_digest_property` returned)."""
    var connection = found[0]
    var arguments = found[1]
    var entry = py_get(connection, "entry")
    if entry is None:
        raise Error("AttributeError: digest_ is unavailable: the saved-model owner is not connected in this installation")
    if py_type(entry) is not static_builtin_type("types", "BuiltinFunctionType"):
        raise Error("TypeError: digest_ requires its connected native entry")
    var context = py_item(connection, PythonObject("context"))
    var args = Python.tuple(model, PythonObject("fitted_digest"), arguments, context)
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    _ = args  # the argument tuple (and the model it holds) lives through the call
    if result:
        return PythonObject(from_owned=result)
    # The entry's refusals are Python exceptions in words: AttributeError for
    # an unfitted model, a fast model or a field with no canonical bytes. The
    # attribute doors that reach this branch (classical_attributes_door
    # `classical_attribute_c`, scaler_lifecycle_native_door
    # `scaler_attribute_c`, neighbor_query_native_door) map an Error by its
    # class prefix and do not restore a stashed pending exception: the stash
    # (`mode_recipe_call`) reached them as the text "mojolearn-optimizer: a
    # Python exception is pending" raised as TypeError, so `hasattr(model,
    # "digest_")` raised on an unfitted Ridge, LogisticRegression, scaler or
    # forest (proving run 2026-11-22). Fetch the exception here (clearing the
    # indicator) and hand it on as "<class>: <text>", which every door maps
    # back to the same class.
    var triple = cpy.PyErr_FetchTriple()
    var kind = PythonObject(from_owned=triple[0]) if triple[0] else PythonObject(None)
    var value = PythonObject(from_owned=triple[1]) if triple[1] else PythonObject(None)
    var traceback = PythonObject(from_owned=triple[2]) if triple[2] else PythonObject(None)
    _ = traceback
    if kind is None:
        raise Error("SystemError: digest_ failed without setting an exception")
    var b = native_loaded_module("builtins")
    var text = String("") if value is None else String(py=py_call(b.str, value))
    var class_name = String(py=kind.__name__)
    raise Error(class_name + ": " + text)
