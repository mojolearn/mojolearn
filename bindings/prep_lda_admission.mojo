# SPDX-License-Identifier: Apache-2.0
# The LDA owner reads only connected native class storage and admitted values.
from std.math import isfinite, floor
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import _flatten_labels, labels_shape_of
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_metadata import classical_is_fitted
from bindings.optimizer_public_words import stash_python_error
from bindings.metrics_native_support import metric_float, metric_is_real, metric_is_integral, metric_warn
from bindings.classical_warnings import COLUMN_WARNING, _warning_category


def _lda_storage(model: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(model) is not py_get(context, "lda_native_class"):
        raise Error("TypeError: native LDA requires its connected class without Python subclass hooks")
    classical_check_layout(model, context["lda_admission_context"])
    return py_call(context["lda_fields_get"], model, py_type(model))


def _lda_parameter(name: PythonObject, value: PythonObject, context: PythonObject, configure: Bool = False) raises:
    # The constructor, get_params and set_params store values verbatim and
    # validate nothing beyond native admission (scikit-learn's convention,
    # check_do_not_raise_errors_in_init_or_set_params); the refusal below
    # runs when a consuming operation (fit, predict, ...) reads the value.
    if String(name) == "covariance_estimator" and value is not None:
        if configure:
            # Stored and returned as the same reference; nothing reads it
            # (no attribute, protocol or call), so no Python runs. A consuming
            # operation refuses it by name below.
            return
        # There is no compiled covariance-estimator object API in this
        # package. Calling sklearn/custom fit from Mojo would execute Python.
        raise Error("TypeError: covariance_estimator cannot invoke Python callbacks; use covariance_estimator=None and the native solver/shrinkage configuration")
    classical_admit_value(value, context)


def lda_native_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _lda_storage(model, context)
    var names = context["lda_parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native LDA initializer requires its complete constructor configuration")
    for i in range(len(names)):  # driver-loop: seven unchanged constructor values, before publication
        _lda_parameter(names[i], args[i], context, True)
    for i in range(len(names)):  # driver-loop: native instance dictionary writes, no __setattr__ hooks
        py_setitem(fields, names[i], args[i])
    return PythonObject(None)


def lda_native_fields(model: PythonObject, context: PythonObject, configure: Bool = False) raises -> PythonObject:
    var fields = _lda_storage(model, context)
    var names = context["lda_parameters"]
    for i in range(len(names)):  # driver-loop: fixed constructor configuration, before comparisons or conversion
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: LDA constructor parameter is missing: " + String(names[i]))
        _lda_parameter(names[i], fields[names[i]], context, configure)
    names = context["lda_fitted_fields"]
    for i in range(len(names)):  # driver-loop: consumed fitted state cannot introduce attribute or scalar hooks
        if py_contains(fields, names[i]):
            classical_admit_value(classifier_label_list(fields[names[i]], context) if String(names[i]) == "classes_" else fields[names[i]], context)
    return fields


def lda_native_get_params(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = Python.dict()
    var names = context["lda_parameters"]
    for i in range(len(names)):  # driver-loop: the fixed constructor parameter schema
        py_setitem(result, names[i], fields[names[i]])
    return result


def lda_native_set_params(model: PythonObject, fields: PythonObject, values: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    if py_type(values) is not py_type(Python.dict()):
        raise Error("TypeError: native LDA parameter updates require an ordinary dictionary")
    var names = context["lda_parameters"]
    var keys = py_call(py_type(Python.list()), values)
    for i in range(len(keys)):  # driver-loop: argument names before equality or conversion
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: LDA parameter names must be native strings")
    for i in range(len(keys)):  # driver-loop: preserve the existing ordered parameter update
        var name = keys[i]
        if not py_contains(names, name):
            raise Error("mojolearn: invalid parameter " + String(py_call(native_loaded_module("builtins").repr, name))
                        + " for LinearDiscriminantAnalysis")
        _lda_parameter(name, values[name], context, True)
        py_setitem(fields, name, values[name])
    return model


def lda_native_labels(value: PythonObject, context: PythonObject, fitting: Bool = False, name: String = "LinearDiscriminantAnalysis") raises:
    if fitting and value is None:
        raise Error(name + " requires y to be passed, but the target y is None.")
    classical_admit_value(value, context)
    # An exact object ndarray may still hold arbitrary Python scalar objects.
    # Reading/flattening the references is native; admission precedes hashing,
    # sorting, equality and numeric conversion in the label encoder.
    var flat = _flatten_labels(value, context)
    classical_admit_value(flat, context)
    if fitting:
        var b = native_loaded_module("builtins")
        var shape = labels_shape_of(value, context)
        if len(shape) == 2 and shape[1] == 1 and py_type(value) is not b.list and py_type(value) is not b.tuple and py_type(value) is not b.memoryview:
            metric_warn(COLUMN_WARNING, _warning_category(context), context)
        var numeric_float = True
        if py_type(value) is context["array_class"]:
            numeric_float = String(value.dtype).startswith("<f")
        elif py_type(value) is py_get(context, "ndarray"):
            numeric_float = String(value.dtype.kind) == "f"
        elif py_type(value) is b.list or py_type(value) is b.tuple:
            numeric_float = len(flat) > 0 and metric_is_real(flat[0], context) and not metric_is_integral(flat[0], context)
        for i in range(len(flat)):  # driver-loop: preserve the public classifier target admission before encoding
            if numeric_float and metric_is_real(flat[i], context) and not metric_is_integral(flat[i], context):
                var number = metric_float(flat[i], context)
                if not isfinite(number):
                    raise Error("Input y contains NaN or infinity (" + name + ").")
                if floor(number) != number:
                    raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")


def lda_require_fitted(model: PythonObject, operation: String, context: PythonObject) raises:
    if not Bool(py=classical_is_fitted(model)):
        var cls = classical_not_fitted_class(context)
        var message = "This LinearDiscriminantAnalysis instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + "."
        ref cpy = Python().cpython()
        cpy.PyErr_SetString(cls._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
        var failure = stash_python_error()
        raise failure^


def lda_native_context(context: PythonObject) raises -> PythonObject:
    # The established label algorithms use isinstance against their connected
    # real/integral classes. Concrete native type tuples preserve those tests
    # without executing numbers.Real/Integral's Python ABC machinery.
    var b = native_loaded_module("builtins")
    var result = py_call(b.dict, context)
    var types = Python.list()
    _ = py_call(types.extend, Python.tuple(b.int, b.float, b.bool))
    _ = py_call(types.extend, context["scalar_types"])
    py_setitem(result, "labels_real_type", py_call(b.tuple, types))
    py_setitem(result, "labels_integral_type", context["integer_types"])
    return result
