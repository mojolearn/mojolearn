# SPDX-License-Identifier: Apache-2.0
# Passive storage admission for the boost owner and its fixed tree members.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_contains, py_call
from bindings.classical_pandas import classical_admit_value
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.classical_family_admission_pyapi import classical_family_admit


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
            boost_tree_admit(estimator, context, True)
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
    return fields
