# SPDX-License-Identifier: Apache-2.0
# Completed selector metadata at the preprocessing boundary. This operation
# has no numerical provider call: it reads the already fitted support once
# and builds the final independent public list in native code.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.mode_metadata_native import _fields, _truth
from bindings.params_metadata_native import _resolved, _value
from bindings.sequential_admission import _one_of


def _support_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    # Exact connected NumPy scalar types have native truth implementations.
    # Array/object truth and Python subclasses are not scalar flag protocols.
    var scalars = py_get(context, "scalar_types")
    if scalars is not None and _one_of(py_type(value), scalars):
        return Bool(py=value)
    return _truth(value)


def prep_support(model: PythonObject, indices: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native preprocessing refuses Python metaclass hooks")
    # The shared native dictionary admission refuses __dict__ descriptors,
    # dict subclasses and non-string keys before any model field is read.
    var fields = _fields(model)
    var missing = Python.list()
    var lookup = Python.dict(missing=missing)
    var feature_name = PythonObject("n_features_in_")
    if not py_contains(fields, feature_name) and _resolved(py_type(model), feature_name, missing) is missing:
        raise Error("RuntimeError: mojolearn: this " + String(py_type(model).__name__) + " instance is not fitted yet")
    # Preserve ordinary class/instance field precedence while refusing Python
    # descriptors. _value uses the documented native instance-item repair.
    _ = _value(model, fields, feature_name, lookup, "AttributeError")
    var mask = _value(model, fields, PythonObject("_mask"), lookup, "AttributeError")
    if py_type(mask) is not b.list and py_type(mask) is not b.tuple:
        raise Error("TypeError: native selector support requires an exact list or tuple mask")
    var return_indices = _support_truth(indices, context)
    var result = Python.list()
    for i in range(len(mask)):  # driver-loop: completed support metadata, no input arithmetic
        var item = mask[i]
        var selected = _support_truth(item, context)
        if not return_indices:
            _ = py_call(result.append, item)
        elif selected:
            _ = py_call(result.append, PythonObject(i))
    # Retain the original mask owner until the completed list is constructed.
    _ = mask^
    return result
