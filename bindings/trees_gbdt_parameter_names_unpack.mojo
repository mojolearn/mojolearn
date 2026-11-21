# SPDX-License-Identifier: Apache-2.0
# CatBoost / XGBoost / LightGBM parameter names on the gradient-boosting
# classes (plan 2026-11-21, section 3, items 3 and 5; lane api-boost).
#
# UNPACK work of the constructor and `set_params` only: a keyword the class's
# signature does not hold is looked up in the class's one name table
# (python/mojolearn/_boosting_param_names.py, `parameter_name_table`, built
# once at import) and stored under its canonical name. A name whose meaning
# differs is refused with the table's words; two names for one parameter with
# different values are refused. Callers reach these functions only on the path
# where a keyword is not a canonical name, so a call with canonical names does
# no lookup here. `fit` and `predict` never read the table.
#
# Values are compared without invoking any Python protocol: identity, or
# equality for exact int, float, str and bool only; anything else given twice
# under two names is refused as two values.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_item
from bindings.trees_keys_unpack import trees_keys


def _same_value(a: PythonObject, b: PythonObject, bi: PythonObject) raises -> Bool:
    if a is b:
        return True
    var ta = py_type(a)
    if ta is not py_type(b):
        return False
    if ta is bi.int or ta is bi.float or ta is bi.str or ta is bi.bool:
        return Bool(py=a == b)
    return False


def _twice(canonical: PythonObject, first: PythonObject, second: PythonObject) raises -> Error:
    return Error("ValueError: " + String(canonical) + " was given twice with different values, as "
                 + String(first) + " and " + String(second) + "; give one of them")


def gbdt_names_in_table(keywords: PythonObject, table: PythonObject) raises -> Bool:
    """True when any key of `keywords` is a row of `table` (a None table holds
    none). Read only after the canonical binding refused a keyword."""
    if table is None:
        return False
    var names = py_call(native_loaded_module("builtins").list, keywords)
    for i in range(len(names)):  # driver-loop: over the caller's keyword names, never their values
        if py_contains(table, names[i]):
            return True
    return False


def gbdt_resolve_keywords(keywords: PythonObject, table: PythonObject) raises -> PythonObject:
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
            if not _same_value(resolved[canonical], value, bi):
                raise _twice(canonical, given_as[canonical], name)
            continue
        py_setitem(resolved, canonical, value)
        py_setitem(given_as, canonical, name)
    return resolved


def gbdt_resolve_arguments(args: PythonObject, table: PythonObject) raises -> PythonObject:
    """`GradientBoosting.__init__`'s argument tuple, whose last item is the
    dict of keywords its signature does not hold. When that dict is empty
    (the common call) `args` is returned as is, with no copy: the constructor
    reads its arguments by position and ignores the extra item. Otherwise a
    new tuple, without the dict, with each table name's value at its
    canonical position.

    The positional signature cannot tell an explicit canonical value from its
    default, so a canonical value that differs from its default and from the
    other name's value is refused as given twice; a canonical value equal to
    its default yields to the other name."""
    var extra = args[len(args) - 1]
    if len(extra) == 0:
        return args
    var bi = native_loaded_module("builtins")
    var names = py_call(bi.list, extra)
    var values = py_call(bi.list, args)
    _ = py_call(values.pop)
    var given_as = Python.dict()
    for i in range(len(names)):  # driver-loop: over the caller's extra keyword names
        var name = names[i]
        if py_type(name) is not bi.str:
            raise Error("TypeError: keywords must be strings")
        var row = PythonObject(None)
        if table is not None:
            row = py_get(table, name)
        if row is None:
            raise Error("TypeError: __init__() got an unexpected keyword argument '" + String(name) + "'")
        if row[0] is None:
            raise Error(String(row[1]))
        var canonical = row[0]
        var index = Int(py=row[2])
        var value = extra[name]
        var current = values[index]
        if py_contains(given_as, canonical):
            if not _same_value(current, value, bi):
                raise _twice(canonical, given_as[canonical], name)
            continue
        if not _same_value(current, row[3], bi) and not _same_value(current, value, bi):
            raise _twice(canonical, canonical, name)
        py_setitem(values, PythonObject(index), value)
        py_setitem(given_as, canonical, name)
    return py_call(bi.tuple, values)


def gbdt_init_arguments(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The `trees_gbdt_init` arguments with the other libraries' names
    resolved. Only `GradientBoosting.__init__` passes the extra keyword dict
    (one item past its parameters and numeric_mode); the set_params rebuild
    (bindings/full_reset_params_native_door.mojo `_replacement`) passes
    canonical values only and is returned unchanged. Cost on the common call:
    two table reads, one length compare and one length of an empty dict."""
    var nk = trees_keys()
    var count = len(py_item(py_item(context, nk[].k_trees_gbdt_tables), nk[].k_init_names))
    if len(args) != count + 2:
        return args
    var extra = args[count + 1]
    if len(extra) == 0:
        return args
    return gbdt_resolve_arguments(args, py_get(context, PythonObject("trees_gbdt_parameter_names")))
