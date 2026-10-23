# SPDX-License-Identifier: Apache-2.0
# The LDA owner reads only connected native class storage and admitted values.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import _flatten_labels


def _lda_storage(model: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(model) is not py_get(context, "lda_native_class"):
        raise Error("TypeError: native LDA requires its connected class without Python subclass hooks")
    classical_check_layout(model, context["lda_admission_context"])
    return py_call(context["lda_fields_get"], model, py_type(model))


def _lda_parameter(name: PythonObject, value: PythonObject, context: PythonObject) raises:
    if String(name) == "covariance_estimator" and value is not None:
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
        _lda_parameter(names[i], args[i], context)
    for i in range(len(names)):  # driver-loop: native instance dictionary writes, no __setattr__ hooks
        py_setitem(fields, names[i], args[i])
    return PythonObject(None)


def lda_native_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _lda_storage(model, context)
    var names = context["lda_parameters"]
    for i in range(len(names)):  # driver-loop: fixed constructor configuration, before comparisons or conversion
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: LDA constructor parameter is missing: " + String(names[i]))
        _lda_parameter(names[i], fields[names[i]], context)
    names = context["lda_fitted_fields"]
    for i in range(len(names)):  # driver-loop: consumed fitted state cannot introduce attribute or scalar hooks
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)
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
            raise Error("mojolearn: invalid parameter " + String(py_call(Python.import_module("builtins").repr, name))
                        + " for LinearDiscriminantAnalysis")
        _lda_parameter(name, values[name], context)
        py_setitem(fields, name, values[name])
    return model


def lda_native_labels(value: PythonObject, context: PythonObject) raises:
    classical_admit_value(value, context)
    # An exact object ndarray may still hold arbitrary Python scalar objects.
    # Reading/flattening the references is native; admission precedes hashing,
    # sorting, equality and numeric conversion in the label encoder.
    var flat = _flatten_labels(value, context)
    classical_admit_value(flat, context)


def lda_native_context(context: PythonObject) raises -> PythonObject:
    # The established label algorithms use isinstance against their connected
    # real/integral classes. Concrete native type tuples preserve those tests
    # without executing numbers.Real/Integral's Python ABC machinery.
    var b = Python.import_module("builtins")
    var result = py_call(b.dict, context)
    var types = Python.list()
    _ = py_call(types.extend, Python.tuple(b.int, b.float, b.bool))
    _ = py_call(types.extend, context["scalar_types"])
    py_setitem(result, "labels_real_type", py_call(b.tuple, types))
    py_setitem(result, "labels_integral_type", context["integer_types"])
    return result
