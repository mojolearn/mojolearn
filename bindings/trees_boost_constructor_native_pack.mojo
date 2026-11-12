# SPDX-License-Identifier: Apache-2.0
# DARTRegressor / DARTClassifier constructors (python/mojolearn/_expansion_trees.py):
# the public __init__ is one call here (`trees_dart_construct`). The argument
# checks, their order and wording, the stored parameters and the seed checks
# are the former Python `_DARTBase.__init__` and `_trees_seed`.
# The PACK half of the former `bindings/trees_boost_constructor_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/trees_boost_constructor_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.params_metadata_native_door import _resolved
from bindings.trees_constructor_native_unpack import tree_parameter_storage


from bindings.trees_boost_constructor_native_unpack import (
    _float,
    _int_below,
    tree_constructor_storage,
    trees_seed_check,
)


def dart_construct(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`DARTRegressor.__init__` / `DARTClassifier.__init__`: `args` holds the
    22 constructor values in the stored order below. The class is native
    public, so NumericModeMixin injects no numeric_mode: it is the last
    constructor value and is stored as given."""
    var names = Python.tuple(
        "n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples", "reg_lambda",
        "max_bin", "drop_rate", "max_drop", "skip_drop", "xgboost_dart_mode", "uniform_drop", "drop_seed",
        "random_state", "reg_alpha", "max_delta_step", "colsample_bytree", "subsample", "subsample_freq",
        "feature_fraction_seed", "bagging_seed", "numeric_mode")
    if py_type(args) is not native_loaded_module("builtins").tuple or len(args) != len(names):
        raise Error("TypeError: the DART constructor takes its 22 parameters")
    if _int_below(args[0], 1):
        raise Error("n_estimators must be >= 1")
    if not _float(args[1]) > 0:
        raise Error("learning_rate must be > 0")
    var drop_rate = _float(args[7])
    var drop_ok = drop_rate >= 0.0 and drop_rate <= 1.0
    if drop_ok:
        var skip_drop = _float(args[9])
        drop_ok = skip_drop >= 0.0 and skip_drop <= 1.0
    if not drop_ok:
        raise Error("drop_rate and skip_drop must be in [0, 1]")
    var reg_bad = _float(args[5]) < 0
    if not reg_bad:
        reg_bad = _float(args[14]) < 0
    if reg_bad:
        raise Error("reg_lambda and reg_alpha must be >= 0")
    var colsample = _float(args[16])
    var sample_ok = colsample > 0.0 and colsample <= 1.0
    if sample_ok:
        var subsample = _float(args[17])
        sample_ok = subsample > 0.0 and subsample <= 1.0
    if not sample_ok:
        raise Error("colsample_bytree and subsample must be in (0, 1]")
    if _int_below(args[18], 0):
        raise Error("subsample_freq must be >= 0")
    tree_constructor_storage(model, names)
    var fields = py_fields(model)
    for i in range(len(names)):  # driver-loop: over the constructor's 22 stored parameters
        py_setitem(fields, names[i], args[i])
    # The seeds are checked after they are stored, as the constructor did.
    trees_seed_check(args[13], context)
    trees_seed_check(args[12], context)
    trees_seed_check(args[19], context)
    trees_seed_check(args[20], context)
    return PythonObject(None)
