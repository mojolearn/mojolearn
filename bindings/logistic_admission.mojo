# SPDX-License-Identifier: Apache-2.0
# Fixed LogisticRegression metadata; never invoke ParamsMixin, a Python
# constructor, user scalar/protocol hooks, or mutable class descriptors.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.sequential_admission import _one_of
from bindings.classifier_public_lifecycle import classifier_label_list


def _parameter(name: PythonObject, value: PythonObject, context: PythonObject, configure: Bool = False) raises:
    classical_admit_value(value, context)
    if configure:
        # Constructor, get_params and set_params store and return values
        # verbatim and validate nothing (scikit-learn's convention, its
        # check_do_not_raise_errors_in_init_or_set_params). Admission above
        # already refuses Python hooks; the scalar check runs when a value is
        # consumed (fit, predict, ...).
        return
    var b = Python.import_module("builtins")
    if String(name) == "class_weight" and _one_of(py_type(value), context["mapping_types"]):
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: native scalar class labels and scalar weights only
            _parameter(PythonObject("class label"), keys[i], context)
            _parameter(PythonObject("class weight"), value[keys[i]], context)
        return
    var cls = py_type(value)
    if value is not None and cls is not b.str and cls is not b.bool and cls is not b.int and cls is not b.float and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: LogisticRegression parameter " + String(name) + " requires native scalar configuration without conversion hooks")


def logistic_storage(model: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var layout = py_get(context["classical_public_layouts"], py_type(model))
    if layout is None:
        raise Error("TypeError: LogisticRegression requires its connected native class")
    return py_call(layout["fields_get"], model, py_type(model))


def logistic_fields(model: PythonObject, context: PythonObject, copy_state: Bool = True,
                    configure: Bool = False) raises -> PythonObject:
    var fields = logistic_storage(model, context)
    var names = context["lr_parameters"]
    for i in range(len(names)):  # driver-loop: fixed constructor parameter values before comparisons/conversion
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: missing LogisticRegression parameter " + String(names[i]))
        _parameter(names[i], fields[names[i]], context, configure)
    names = context["lr_consumed_fields"]
    for i in range(len(names)):  # driver-loop: consumed fitted state and optional native solver configuration
        if py_contains(fields, names[i]):
            var value = classifier_label_list(fields[names[i]], context) if String(names[i]) == "classes_" else fields[names[i]]
            classical_admit_value(value, context)
    if copy_state:
        fields = py_call(py_type(Python.dict()), fields)
        if py_contains(fields, "classes_"):
            py_setitem(fields, "classes_", classifier_label_list(fields["classes_"], context))
    return fields


def logistic_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = logistic_storage(model, context)
    var names = context["lr_parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native LogisticRegression requires its full constructor configuration")
    for i in range(len(names)):  # driver-loop: refusal before publishing any constructor parameters
        _parameter(names[i], args[i], context, True)
    for i in range(len(names)):  # driver-loop: unchanged native constructor assignments
        if String(names[i]) != "numeric_mode" or args[i] is not None or not py_contains(fields, names[i]):
            py_setitem(fields, names[i], args[i])
    return PythonObject(None)


def logistic_get_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = logistic_fields(model, context, True, True)
    var out = Python.dict()
    var names = context["lr_sorted_parameters"]
    for i in range(len(names)):  # driver-loop: ParamsMixin's sorted parameter names
        py_setitem(out, names[i], fields[names[i]])
    return out


def logistic_set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(values) is not b.dict:
        raise Error("TypeError: native LogisticRegression updates require ordinary keyword storage")
    if len(values) == 0:
        return model
    var params = logistic_get_params(model, context)
    var names = py_call(b.list, values)
    for i in range(len(names)):  # driver-loop: names before key comparison or formatting
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native LogisticRegression parameter names must be strings")
    for i in range(len(names)):  # driver-loop: ordered scalar/native-container parameter updates
        var key = names[i]
        var parts = py_call(key.partition, "__")
        if not py_contains(params, parts[0]):
            raise Error("Invalid parameter " + String(py_call(b.repr, parts[0])) + " for estimator "
                        + String(py_type(model).__name__) + ". Valid parameters are: "
                        + String(py_call(b.repr, py_call(b.list, context["lr_sorted_parameters"]))) + ".")
        if String(parts[1]) != "":
            raise Error("AttributeError: scalar parameter has no set_params")
        _parameter(parts[0], values[key], context, True)
        py_setitem(params, key, values[key])
    # Preserve ParamsMixin's successful reconstruction: a new configuration
    # removes fitted state. A rejected update leaves the original model alone.
    var fields = logistic_storage(model, context)
    _ = py_call(fields.clear)
    _ = py_call(fields.update, params)
    return model
