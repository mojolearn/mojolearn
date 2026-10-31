# SPDX-License-Identifier: Apache-2.0
# DARTRegressor / DARTClassifier constructors (python/mojolearn/_expansion_trees.py):
# the public __init__ is one call here (`trees_dart_construct`). The argument
# checks, their order and wording, the stored parameters and the seed checks
# are the former Python `_DARTBase.__init__` and `_trees_seed`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.params_metadata_native import _resolved
from bindings.trees_constructor_native import tree_parameter_storage


def _is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    var t = py_type(value)
    return t is native_loaded_module("builtins").bool or String(t.__name__) == "bool_"


def _float(value: PythonObject) raises -> Float64:
    """`float(value)`; a refused conversion keeps its Python class."""
    return Float64(py=py_call_typed(native_loaded_module("builtins").float, value))


def _int_below(value: PythonObject, bound: Int) raises -> Bool:
    """`int(value) < bound`, compared as Python integers (no overflow)."""
    var parsed = py_call_typed(native_loaded_module("builtins").int, value)
    return Bool(py=parsed < PythonObject(bound))


def tree_constructor_storage(model: PythonObject, names: PythonObject) raises:
    """Refuse a modified `__setattr__` or a descriptor on a stored name before
    the constructor writes the instance dictionary directly: the writes must
    not skip a Python storage hook, and must not call one either."""
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native tree constructors require ordinary class metadata")
    if _resolved(cls, PythonObject("__setattr__"), PythonObject(None)) is not b.object.__dict__["__setattr__"]:
        raise Error("TypeError: native tree constructors cannot invoke modified Python storage hooks")
    tree_parameter_storage(model, names)


def trees_seed_check(value: PythonObject, context: PythonObject) raises:
    """`_trees_seed(value)`'s refusal: None or a nonnegative integer."""
    if value is None:
        return
    var b = native_loaded_module("builtins")
    var ok = not _is_bool(value) and Bool(py=py_call(b.isinstance, value, context["trees_integral_type"]))
    if ok:
        ok = not Bool(py=value < PythonObject(0))
    if not ok:
        raise Error("random_state must be None or a nonnegative int (a RandomState object is refused:"
                    + " draws come from the lane's own counter RNG, xtrees/ops.mojo)")


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
