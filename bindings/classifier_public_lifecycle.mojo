# SPDX-License-Identifier: Apache-2.0
# Native equivalents of the classifier mixin's completed-fit metadata. Keep
# class labels in ordinary native storage while the solver/score consumes them;
# expose the established ClassLabels object only after the operation completes.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get
from bindings.classical_labels import _flatten_labels


def classifier_label_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_get(context, "classifier_labels_class")
    if cls is None or py_type(value) is not cls:
        return value
    # list.copy is the native list implementation, not an iterable protocol.
    return py_call(py_type(Python.list()).copy, value)


def _public_labels(values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(values) is not b.list:
        return values
    var cls = context["classifier_labels_class"]
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var connected = context["classifier_labels_namespace"]
    var names = py_call(b.list, fields)
    if len(names) != len(connected):
        raise Error("TypeError: native class-label output metadata changed")
    for i in range(len(names)):  # driver-loop: immutable connected output class metadata
        if py_type(names[i]) is not b.str or not py_contains(connected, names[i]) or fields[names[i]] is not connected[names[i]]:
            raise Error("TypeError: native class-label output cannot invoke modified Python hooks")
    var result = py_call(b.list.__new__, cls)
    _ = py_call(b.list.extend, result, values)
    return result


def _feature_names(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # Only an admitted exact DataFrame has column-name semantics. Reading its
    # manager/index storage avoids DataFrame.columns and Index iteration.
    if py_type(source) is not _pd_class("pandas.core.frame", "DataFrame"):
        return PythonObject(None)
    var b = Python.import_module("builtins")
    var manager = py_get(_pd_fields(source), "_mgr")
    if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
        raise Error("TypeError: native classifier requires ordinary pandas manager storage")
    var axes = _pd_get("pandas._libs.internals", "BlockManager", "axes", manager)
    if py_type(axes) is not b.list or len(axes) != 2:
        raise Error("TypeError: native pandas column axes are unavailable")
    var columns = axes[0]
    # Range/numeric/multi-index names never set feature_names_in_. Ordinary
    # string Index owns either a native ndarray or StringArray's native slot.
    if py_type(columns) is not _pd_class("pandas.core.indexes.base", "Index"):
        return PythonObject(None)
    var storage = py_get(_pd_fields(columns), "_data")
    if py_type(storage) is _pd_class("pandas.core.arrays.string_", "StringArray"):
        storage = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", storage)
    if py_type(storage) is not context["ndarray"]:
        raise Error("TypeError: native classifier feature names require native pandas string-index storage")
    var names = _flatten_labels(storage, context)
    if len(names) == 0:
        return PythonObject(None)
    for i in range(len(names)):  # driver-loop: feature-name type metadata, no conversion hooks
        if py_type(names[i]) is not b.str:
            return PythonObject(None)
    var result = py_call(context["ndarray"], Python.tuple(len(names)), dtype="O")
    for i in range(len(names)):  # driver-loop: copy admitted string references into native object storage
        py_setitem(result, i, names[i])
    return result


def classifier_after_fit(fields: PythonObject, source: PythonObject, context: PythonObject) raises:
    var names = _feature_names(source, context)
    if names is None:
        _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
    else:
        py_setitem(fields, "feature_names_in_", names)
    py_setitem(fields, "classes_", _public_labels(fields["classes_"], context))
    py_setitem(fields, "_mojolearn_fitted", True)
