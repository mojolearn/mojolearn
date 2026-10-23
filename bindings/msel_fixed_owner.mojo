# SPDX-License-Identifier: Apache-2.0
# Admission/parameter updates for exact built-in estimator schemas whose
# constructors only store the supplied parameter objects. Numerical work
# continues through each estimator's existing native owner.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout, _type_fields
from bindings.classical_pandas import classical_admit_value
from bindings.prep_lda_admission import lda_native_context
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.sequential_admission import _one_of


def _parameter(name: PythonObject, value: PythonObject, context: PythonObject) raises:
    classical_admit_value(value, context)
    if py_contains(context["cv_fixed_owner"]["sequence_parameters"], name):
        return
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is not None and cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.str and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: native CV scalar configuration cannot invoke conversion or array computation hooks")


def fixed_owner_context(context: PythonObject) raises -> PythonObject:
    var result = lda_native_context(context)
    py_setitem(result, "trees_real_type", result["labels_real_type"])
    py_setitem(result, "trees_integral_type", result["labels_integral_type"])
    return result


def fixed_owner_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var plan = context["cv_fixed_owner"]
    if py_type(model) is not plan["class"]:
        raise Error("TypeError: native CV requires its connected built-in estimator class")
    classical_check_layout(model, plan["admission"])
    if py_get(plan, "loss") is not None and py_get(_type_fields(plan["class"]), "_loss") is not plan["loss"]:
        raise Error("TypeError: native gradient-boosting loss metadata changed")
    var fields = py_call(plan["fields_get"], model, py_type(model))
    var names = plan["parameters"]
    for i in range(len(names)):  # driver-loop: fixed constructor state, before native option comparisons
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: missing native estimator constructor parameter " + String(names[i]))
        _parameter(names[i], fields[names[i]], context)
    names = plan["consumed"]
    for i in range(len(names)):  # driver-loop: native fitted buffers/scalars before prediction or scoring
        if py_contains(fields, names[i]):
            var value = classifier_label_list(fields[names[i]], context) if String(names[i]) == "classes_" else fields[names[i]]
            classical_admit_value(value, context)
    if py_get(plan, "learner_class") is not None:
        classical_check_layout(plan["learner_class"], plan["learner_admission"])
    var learner = py_get(fields, "_learner_")
    if learner is not None:
        if py_type(learner) is not py_get(plan, "learner_class"):
            raise Error("TypeError: gradient-boosting CV requires its native learner state")
        classical_check_layout(learner, plan["learner_admission"])
        var stored = py_call(plan["learner_fields_get"], learner, py_type(learner))
        var consumed = plan["learner_consumed"]
        for i in range(len(consumed)):  # driver-loop: learner scalars/model text consumed by native prediction
            if py_contains(stored, consumed[i]):
                classical_admit_value(stored[consumed[i]], context)
    return fields


def fixed_owner_set_params(model: PythonObject, values: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = fixed_owner_fields(model, context)
    var plan = context["cv_fixed_owner"]
    if py_type(values) is not b.dict:
        raise Error("TypeError: native parameter updates require an ordinary dictionary")
    var names = plan["parameters"]
    var keys = py_call(b.list, values)
    for i in range(len(keys)):  # driver-loop: validate all names before hashing or formatting
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native parameter names must be exact strings")
    if len(keys) == 0:
        return model
    if Bool(py=plan["reset"]):
        var unknown = Python.list()
        for i in range(len(keys)):  # driver-loop: adapter rejects unknown names before any mutation
            if not py_contains(names, keys[i]):
                _ = py_call(unknown.append, keys[i])
        if len(unknown):
            raise Error("Invalid " + String(plan["name"]) + " parameters: " + String(py_call(b.repr, py_call(b.sorted, unknown))))
        var replacement = Python.dict()
        for i in range(len(names)):  # driver-loop: constructor reconstruction keeps only parameters
            var value = values[names[i]] if py_contains(values, names[i]) else fields[names[i]]
            _parameter(names[i], value, context)
            py_setitem(replacement, names[i], value)
        _ = py_call(fields.clear)
        _ = py_call(fields.update, replacement)
    else:
        for i in range(len(keys)):  # driver-loop: preserve _PrepBase's ordered parameter assignments
            if not py_contains(names, keys[i]):
                raise Error("mojolearn: invalid parameter " + String(py_call(b.repr, keys[i])) + " for " + String(plan["name"]))
            _parameter(keys[i], values[keys[i]], context)
            py_setitem(fields, keys[i], values[keys[i]])
    return model
