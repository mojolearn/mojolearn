# SPDX-License-Identifier: Apache-2.0
# Other libraries' parameter names, resolved onto an estimator's canonical
# names at construction and in `set_params` only (lane api-boost, plan
# 2026-11-21 section 3). A leaf: no owner imports, so the tree constructors
# (bindings/trees_gbdt_parameters_pack.mojo, trees_gbdt_parameter_names_unpack.mojo)
# and the shared setter (bindings/full_reset_params_native_door.mojo) share it.
#
# A table is `{name: (canonical or None, refusal text, index, default)}`,
# built once at import (python/mojolearn/_boosting_param_names.py). Callers
# read it only after a keyword failed the canonical schema, so a call with
# canonical names never reaches this file.
#
# Values are compared without invoking any Python protocol: identity, or
# equality for exact int, float, str and bool only; anything else given twice
# under two names is refused as two values.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem


def same_parameter_value(a: PythonObject, b: PythonObject, bi: PythonObject) raises -> Bool:
    if a is b:
        return True
    var ta = py_type(a)
    if ta is not py_type(b):
        return False
    if ta is bi.int or ta is bi.float or ta is bi.str or ta is bi.bool:
        return Bool(py=a == b)
    return False


def given_twice(canonical: PythonObject, first: PythonObject, second: PythonObject) raises -> Error:
    return Error("ValueError: " + String(canonical) + " was given twice with different values, as "
                 + String(first) + " and " + String(second) + "; give one of them")


def names_in_table(keywords: PythonObject, table: PythonObject) raises -> Bool:
    """True when any key of `keywords` is a row of `table` (a None table holds
    none). Read only after the canonical binding refused a keyword."""
    if table is None:
        return False
    var names = py_call(native_loaded_module("builtins").list, keywords)
    for i in range(len(names)):  # driver-loop: over the caller's keyword names, never their values
        if py_contains(table, names[i]):
            return True
    return False


def resolve_parameter_names(keywords: PythonObject, table: PythonObject) raises -> PythonObject:
    """A new keyword dict with every table name stored under its canonical
    name; names that are not in the table are kept as given (the caller's
    binder refuses them in its own words). Refuses a different-meaning name
    and two names for one parameter with different values."""
    var bi = native_loaded_module("builtins")
    var names = py_call(bi.list, keywords)
    var resolved = Python.dict()
    var given_as = Python.dict()
    for i in range(len(names)):  # driver-loop: canonical and unknown names first, unchanged
        if not py_contains(table, names[i]):
            py_setitem(resolved, names[i], keywords[names[i]])
            py_setitem(given_as, names[i], names[i])
    for i in range(len(names)):  # driver-loop: then the other libraries' names, onto their canonical names
        var name = names[i]
        var row = py_get(table, name)
        if row is None:
            continue
        if row[0] is None:
            raise Error(String(row[1]))
        var canonical = row[0]
        var value = keywords[name]
        if py_contains(resolved, canonical):
            if not same_parameter_value(resolved[canonical], value, bi):
                raise given_twice(canonical, given_as[canonical], name)
            continue
        py_setitem(resolved, canonical, value)
        py_setitem(given_as, canonical, name)
    return resolved
