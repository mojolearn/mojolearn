# SPDX-License-Identifier: Apache-2.0
# DARTRegressor / DARTClassifier constructors (python/mojolearn/_expansion_trees.py):
# the public __init__ is one call here (`trees_dart_construct`). The argument
# checks, their order and wording, the stored parameters and the seed checks
# are the former Python `_DARTBase.__init__` and `_trees_seed`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.params_metadata_native_door import _resolved
from bindings.trees_constructor_native_unpack import tree_parameter_storage


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
