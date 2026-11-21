# SPDX-License-Identifier: Apache-2.0
# CatBoost / XGBoost / LightGBM parameter names on the gradient-boosting
# classes (plan 2026-11-21, section 3, items 3 and 5; lane api-boost).
#
# The positional half, for `GradientBoosting.__init__` (`trees_gbdt_init`):
# the keyword half and the value comparison are the leaf
# bindings/parameter_names_unpack.mojo. UNPACK work of the constructor only;
# `fit` and `predict` never read the table
# (python/mojolearn/_boosting_param_names.py, built once at import).
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_item
from bindings.trees_keys_unpack import trees_keys
from bindings.parameter_names_unpack import same_parameter_value, given_twice


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
            if not same_parameter_value(current, value, bi):
                var refusal_54 = given_twice(canonical, given_as[canonical], name)
                raise refusal_54^
            continue
        if not same_parameter_value(current, row[3], bi) and not same_parameter_value(current, value, bi):
            var refusal_57 = given_twice(canonical, canonical, name)
            raise refusal_57^
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
