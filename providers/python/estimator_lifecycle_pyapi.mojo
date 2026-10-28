# SPDX-License-Identifier: Apache-2.0
# Shared native estimator lifecycle/output boundary. Algorithm owners perform
# their numerical work; this common helper preserves the former public wrappers.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_estimator_name
from bindings.classical_labels import _flatten_labels
from bindings.classifier_public_lifecycle import classifier_after_fit, _feature_names
from bindings.metrics_native_support import metric_warn
from bindings.native_caller_frame import native_caller_globals


def estimator_outside() raises -> Bool:
    # One Python public wrapper hands over to this owner; inspect its caller
    # using CPython frame storage, preserving the established marker order.
    var globals = native_caller_globals(1)
    if globals is None:
        return True
    var name = py_get(globals, "__name__")
    var outside = True
    if py_type(name) is py_type(PythonObject("")):
        var module = String(name)
        outside = module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
    if py_get(globals, "__mojolearn_user_boundary__") is PythonObject(True):
        return True
    if py_get(globals, "__mojolearn_internal__") is PythonObject(True):
        return False
    return outside



def estimator_feature_check(model: PythonObject, fields: PythonObject, source: PythonObject,
                   context: PythonObject) raises:
    var names = _feature_names(source, context)
    var fitted = py_get(fields, "feature_names_in_")
    var b = Python.import_module("builtins")
    var name = classical_estimator_name(model)
    if fitted is None:
        if names is not None:
            metric_warn("X has feature names, but " + name + " was fitted without feature names", b.UserWarning, context)
        return
    if names is None:
        metric_warn("X does not have valid feature names, but " + name + " was fitted with feature names", b.UserWarning, context)
        return
    classical_admit_value(fitted, context)
    var before = _flatten_labels(fitted, context)
    var after = _flatten_labels(names, context)
    classical_admit_value(before, context)
    classical_admit_value(after, context)
    if not Bool(py=before == after):
        raise Error("The feature names should match those that were passed during fit.")



def estimator_fit_begin(fields: PythonObject) raises -> Int:
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fit depth must be an exact integer")
    var value = Int(py=depth)
    py_setitem(fields, "_mojolearn_fitting", PythonObject(value + 1))
    return value


def estimator_fit_end(model: PythonObject, depth: Int) raises:
    # A set_params nested inside fit can replace the dictionary's contents.
    var fields = py_fields(model)
    if depth > 0:
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
    else:
        _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))


def estimator_fit_complete(model: PythonObject, source: PythonObject, classifier: Bool,
                           context: PythonObject) raises:
    var fields = py_fields(model)
    if classifier:
        classifier_after_fit(fields, source, context)
    else:
        var names = _feature_names(source, context)
        if names is None:
            _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
        else:
            py_setitem(fields, "feature_names_in_", names)
        py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
