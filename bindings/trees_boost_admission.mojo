# SPDX-License-Identifier: Apache-2.0
# Passive storage admission for the boost owner and its fixed tree members.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_contains, py_call, py_fields
from bindings.classical_pandas import classical_admit_value
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.native_loaded_module import NativeModules
from bindings.trees_side_records import trees_side_get, trees_side_put, trees_side_drop, trees_key_same, trees_key_fields

comptime _FITTED_SLOT = "trees_boost_fitted_v1"


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
    var parameters = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: connected tree parameter schema for passive subclasses
        if parameters is None:
            parameters = py_get(defaults, mro[i])
    if parameters is None:
        raise Error("TypeError: native boosting requires a connected tree member")
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


def boost_admit(model: PythonObject, context: PythonObject, fitting: Bool = False) raises -> PythonObject:
    var fields = classical_family_admit(model)
    var b = native_loaded_module("builtins")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native boosting refuses Python fit callbacks")
    if py_contains(fields, "_mojolearn_fitting") and py_type(fields["_mojolearn_fitting"]) is not b.int:
        raise Error("TypeError: native boosting fit depth must be an exact integer")
    for name in ["numeric_mode", "_mojolearn_fitted"]:  # small-loop(public lifecycle scalar metadata)
        if py_contains(fields, name):
            host_inference_scalar(fields[name], context, True)
    if fitting:
        for name in ["algorithm", "bagging_seed", "bootstrap", "bootstrap_features", "colsample_bytree", "drop_rate", "drop_seed", "feature_fraction_seed", "learning_rate", "loss", "max_bin", "max_delta_step", "max_depth", "max_drop", "max_features", "max_leaf_nodes", "max_samples", "min_child_samples", "min_impurity_decrease", "min_samples_leaf", "min_samples_split", "n_estimators", "n_jobs", "num_leaves", "oob_score", "random_state", "reg_alpha", "reg_lambda", "skip_drop", "sparse_output", "subsample", "subsample_freq", "uniform_drop", "verbose", "warm_start", "xgboost_dart_mode"]:  # small-loop(fixed consumed scalar fields)
            if py_contains(fields, name):
                host_inference_scalar(fields[name], context, True)
        var estimator = py_get(fields, "estimator")
        if estimator is not None:
            _require_connected_member(model, estimator, context)
            boost_tree_admit(estimator, context, True)
        # A refit replaces the fitted state: forget the record now so the
        # old members are not kept alive until the next call.
        trees_side_drop(_FITTED_SLOT, model)
        return fields
    # Fitted admission once per fitted state (lane perf-trees-owed,
    # 2026-11-06): every predict admitted every member's arrays and scalars
    # again (DART: T*K one-tree members, each ~12 full value admissions plus
    # its class stamp). The identity key of the owner's fields and of every
    # member's class and fields is recorded in the native side table; while
    # it matches, the full admission below would accept again, and only one
    # member per distinct member class repeats its class admission. Any
    # assignment, member replacement or container edit changes the key.
    var mods = NativeModules()
    var key = Python.list()
    var reps = Python.list()
    var recordable = _fitted_key(mods, fields, key, reps)
    if recordable:
        var held = trees_side_get(_FITTED_SLOT, model)
        if held is not None and trees_key_same(held, key):
            for i in range(len(reps)):  # small-loop(reps: one member per distinct member class)
                _ = classical_family_admit(reps[i])
            return fields
    for name in ["n_classes_", "n_features_in_", "n_output_features_", "n_trees_"]:  # small-loop(fitted dimensions)
        if py_contains(fields, name):
            host_inference_scalar(fields[name], context)
    for name in ["forest_"]:  # small-loop(fixed individual tree members)
        var member = py_get(fields, name)
        if member is not None:
            boost_tree_admit(member, context)
    for name in ["_estimators", "estimators_", "trees_", "_batched"]:  # small-loop(fixed stored member sequences)
        var members = py_get(fields, name)
        if members is not None:
            if py_type(members) is not b.list and py_type(members) is not b.tuple:
                raise Error("TypeError: native boosting requires ordinary member sequences")
            for i in range(len(members)):  # driver-loop: each member's native storage before execution
                boost_tree_admit(members[i], context)
    for name in ["_node_col", "_oob_rows", "_tree_base", "estimator_weights_", "estimators_features_", "init_score_", "tree_coefs_", "tree_values_"]:  # small-loop(fixed fitted numeric arrays and lists)
        if py_contains(fields, name):
            classical_admit_value(fields[name], context)
    if py_contains(fields, "classes_"):
        classical_admit_value(classifier_label_list(fields["classes_"], context), context)
    for name in ["init_score_", "tree_coefs_"]:  # small-loop(fitted scalar sequences)
        if py_contains(fields, name):
            _scalar_storage(fields[name], context)
    if recordable:
        _ = trees_side_put(_FITTED_SLOT, model, key)
    return fields
