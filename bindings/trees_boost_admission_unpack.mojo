# SPDX-License-Identifier: Apache-2.0
# Passive storage admission for the boost owner and its fixed tree members.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_contains, py_call, py_fields
from bindings.classical_pandas_door import classical_admit_value
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.host_inference_scalars_door import host_inference_scalar
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from bindings.native_loaded_module_door import NativeModules
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put, trees_side_drop, trees_key_same, trees_key_fields
from bindings.native_model_records import (
    records_stamped_get, records_stamped_put, records_stampable, records_drop, ContentRecord, content_fields,
    content_object, RECORDS_TREES,
)

comptime _FITTED_STAMP = "trees_boost_fitted_stamp_v1"


def _fitted_stamp(model: PythonObject, fields: PythonObject, reps: PythonObject) raises:
    """Stamp the fitted admission with the owner's generation and the native
    content words of what `_fitted_key` reads (lane ne1-e-records, row N050):
    the owner's field contents, and each member's class, generation word and
    field contents. Only when `fields` is the owner's own instance
    dictionary (the one its generation watches; the record holds it)."""
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var content = ContentRecord()
    content_fields(content, fields, 4)
    var member = py_get(fields, "forest_")
    if member is not None:
        content_object(content, member, 4)
    for name in ["_estimators", "estimators_", "trees_", "_batched"]:  # small-loop(fixed stored member sequences)
        var members = py_get(fields, name)
        if members is not None:
            for i in range(len(members)):  # driver-loop: each fitted member's content words (once per record)
                content_object(content, members[i], 4)
    _ = records_stamped_put(RECORDS_TREES, model, _FITTED_STAMP, content^, Python.tuple(reps, fields))


def _fitted_key(mut mods: NativeModules, fields: PythonObject, key: PythonObject, reps: PythonObject) raises -> Bool:
    """The identity key of everything the fitted admission below reads: the
    owner's stored fields and every member's class and stored fields. `reps`
    receives one member per distinct member class (their class admission
    still runs on every call). False when a member sequence is not an
    ordinary list or tuple (the full admission refuses it)."""
    var b = mods.module("builtins")
    trees_key_fields(mods, key, fields)
    var groups = Python.list()
    var member = py_get(fields, "forest_")
    if member is not None:
        _ = py_call(groups.append, Python.list(member))
    for name in ["_estimators", "estimators_", "trees_", "_batched"]:  # small-loop(fixed stored member sequences)
        var members = py_get(fields, name)
        if members is not None:
            if py_type(members) is not b.list and py_type(members) is not b.tuple:
                return False
            _ = py_call(groups.append, members)
    var classes = Python.list()
    for g in range(len(groups)):  # small-loop(groups: the fixed member slots)
        var members = groups[g]
        for i in range(len(members)):  # driver-loop: each fitted member's identity record
            var m = members[i]
            var cls = py_type(m)
            _ = py_call(key.append, m)
            _ = py_call(key.append, cls)
            var member_fields = PythonObject(None)
            try:
                member_fields = py_fields(m)
            except:
                # Not an instance with stored fields: the full admission
                # refuses it with its own wording.
                return False
            if py_type(member_fields) is not b.dict:
                return False
            trees_key_fields(mods, key, member_fields)
            var seen = False
            for j in range(len(classes)):  # small-loop(classes: distinct member classes, usually one)
                if classes[j] is cls:
                    seen = True
                    break
            if not seen:
                _ = py_call(classes.append, cls)
                _ = py_call(reps.append, m)
    return True


def _scalar_storage(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 16:
        raise Error("TypeError: native boosting scalar state exceeds its supported nesting")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.dict:
        var names = py_call(b.list, value)
        for i in range(len(names)):  # driver-loop: native configuration key/value storage
            host_inference_scalar(names[i], context)
            _scalar_storage(value[names[i]], context, depth + 1)
    elif kind is b.list or kind is b.tuple:
        for i in range(len(value)):  # driver-loop: stored scalar coefficients and bias words
            _scalar_storage(value[i], context, depth + 1)
    else:
        host_inference_scalar(value, context, True)


def boost_tree_admit(model: PythonObject, context: PythonObject, cloning: Bool = False) raises:
    var fields = classical_family_admit(model)
    var b = native_loaded_module("builtins")
    var defaults = context["trees_defaults"]
    var cls = py_type(model)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    # Optional, not a PythonObject(None) sentinel read and reassigned in the
    # loop (modular/modular#7546 pattern; runtime-fix-save-forest).
    var found = Optional[PythonObject]()
    for i in range(len(mro)):  # driver-loop: connected tree parameter schema for passive subclasses
        if not found:
            var candidate = py_get(defaults, mro[i])
            if candidate is not None:
                found = candidate
    if not found:
        raise Error("TypeError: native boosting requires a connected tree member")
    var parameters = found.value()
    if cloning:
        var names = py_call(b.list, parameters)
        for i in range(len(names)):  # driver-loop: parameters consumed by native member clone/fit
            var value = py_get(fields, names[i])
            classical_admit_value(value, context)
            _scalar_storage(value, context)
    for name in ["_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_cfg", "_fit_device", "_fit_numeric_mode", "n_features_in_", "n_classes_", "_n_trees", "_num_outputs"]:  # small-loop(fixed native forest storage)
        if py_contains(fields, name):
            classical_admit_value(fields[name], context)
    for name in ["_cfg", "n_features_in_", "n_classes_", "_n_trees", "_num_outputs", "_fit_numeric_mode", "_fit_device"]:  # small-loop(forest scalar state)
        if py_contains(fields, name):
            _scalar_storage(fields[name], context)
    if py_contains(fields, "classes_"):
        classical_admit_value(classifier_label_list(fields["classes_"], context), context)


def _require_connected_member(model: PythonObject, estimator: PythonObject, context: PythonObject) raises:
    """The documented refusal of a member that is not a mojolearn tree
    estimator (trees_public_boost.mojo `_require_tree`), before the member's
    storage admission refuses it as an unconnected class. The class is named
    by its type name: a foreign object's Python __repr__ never runs."""
    var classes = py_get(context, "trees_classes")
    if classes is None:
        return
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(estimator))
    for i in range(len(mro)):  # driver-loop: over the member class ancestry metadata
        if py_contains(classes, mro[i]):
            return
    raise Error("NotImplementedError: " + String(py_type(model).__name__) + ": estimator="
                + String(py_type(estimator).__name__) + "() is not a mojolearn DecisionTree*, RandomForest* "
                + "or ExtraTrees* estimator; the native ensemble fits only those members. Refused by name "
                + "rather than accepted and ignored.")
