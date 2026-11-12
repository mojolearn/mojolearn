# SPDX-License-Identifier: Apache-2.0
# Admission/parameter updates for exact built-in estimator schemas whose
# constructors only store the supplied parameter objects. Numerical work
# continues through each estimator's existing native owner.
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_model_records import (
    RECORDS_PREPARED, ContentRecord, content_fields, content_object, records_stamped_get, records_stamped_put,
)
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes_door import classical_check_layout, _type_fields
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_typed_context_pack import native_typed_context
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.sequential_admission_door import _one_of
from bindings.native_model_records import bump_model_generation


def _parameter(name: PythonObject, value: PythonObject, context: PythonObject) raises:
    classical_admit_value(value, context)
    if py_contains(context["cv_fixed_owner"]["sequence_parameters"], name):
        return
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is not None and cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.str and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: native CV scalar configuration cannot invoke conversion or array computation hooks")


#: The stamped-record name of the fixed-owner admission (family RECORDS_PREPARED).
comptime _FIXED_OWNER_RECORD = "msel_fixed_owner_fields_v1"
#: Content walk depth: the admission's own nesting bound (classical_admit_value).
comptime _FIXED_OWNER_DEPTH = 64


def _fixed_owner_stamp(model: PythonObject, fields: PythonObject, learner: PythonObject,
                       plan: PythonObject) raises:
    # After a full admission passed: stamp it (nothing is stored when the
    # model has no generation or the walk meets a value it cannot prove).
    # The payload is the import-time plan, which never holds the model.
    var content = ContentRecord()
    content_fields(content, fields, _FIXED_OWNER_DEPTH)
    if learner is not None:
        content_object(content, learner, _FIXED_OWNER_DEPTH)
    _ = records_stamped_put(RECORDS_PREPARED, model, _FIXED_OWNER_RECORD, content^, plan)


def fixed_owner_context(context: PythonObject) raises -> PythonObject:
    # Cached per connected context (bindings/native_typed_context.mojo).
    return native_typed_context(context, True, True)


def fixed_owner_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var plan = context["cv_fixed_owner"]
    if py_type(model) is not plan["class"]:
        raise Error("TypeError: native CV requires its connected built-in estimator class")
    classical_check_layout(model, plan["admission"])
    if py_get(plan, "loss") is not None and py_get(_type_fields(plan["class"]), "_loss") is not plan["loss"]:
        raise Error("TypeError: native gradient-boosting loss metadata changed")
    var fields = py_call(plan["fields_get"], model, py_type(model))
    # N069: the parameter, fitted-buffer and learner admission below depends
    # only on the instance dictionary state, the identities and classes its
    # values hold, and the learner's own state. It is kept as one stamped
    # record per model (bindings/native_model_records.mojo: the model's
    # generation word plus a ContentRecord of every field value to the
    # admission depth and of the learner), so a predict / score / set_params
    # with no state change skips it; any assignment, in-place container edit,
    # class reassignment or learner change misses and re-admits. Without
    # dictionary watchers (Python < 3.12) every call re-admits, as before.
    var stamped = records_stamped_get(RECORDS_PREPARED, model, _FIXED_OWNER_RECORD)
    if stamped is not None and stamped is plan:
        var consumed_held = plan["consumed"]
        for i in range(len(consumed_held)):  # driver-loop: fitted buffer owners (their storage is not in the record)
            # The fitted Arrays are re-admitted as values every call (their
            # native storage slots are not words of the record); the label
            # list of `classes_` is not rebuilt: its identity and class are
            # proven and its labels were admitted when the record was made.
            if py_contains(fields, consumed_held[i]) and String(consumed_held[i]) != "classes_":
                classical_admit_value(fields[consumed_held[i]], context)
        # The layout checks stay per call (the record proves values, not class rows).
        if py_get(plan, "learner_class") is not None:
            classical_check_layout(plan["learner_class"], plan["learner_admission"])
        var held_learner = py_get(fields, "_learner_")
        if held_learner is not None:
            classical_check_layout(held_learner, plan["learner_admission"])
        return fields
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
    _fixed_owner_stamp(model, fields, learner, plan)
    return fields
