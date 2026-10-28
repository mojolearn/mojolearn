# SPDX-License-Identifier: Apache-2.0
# The tree-ensemble owner's operations (`classical_native_call` operations
# `trees_*`, connected by python/mojolearn/_native_trees_bridge.py). Each
# public method of a connected tree class is one call here.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get
from bindings.classical_pandas import classical_admit_value
from bindings.msel_fixed_owner import fixed_owner_context, fixed_owner_fields, fixed_owner_set_params
from bindings.classical_attributes import classical_connect_numpy
from bindings.trees_public_forest import forest_fit, forest_predict, forest_predict_proba, forest_get_params, forest_refresh
from bindings.trees_constructor_native import forest_construct
from bindings.trees_public_lifecycle_native import forest_public_operation
from bindings.trees_public_score import forest_score
from bindings.trees_public_iforest import iforest_operation
from bindings.trees_public_boost import boost_operation
from bindings.trees_public_auxiliary import embedding_apply, tree_export_state
from bindings.trees_public_meta import meta_operation
from bindings.trees_public_meta_lifecycle import meta_public_operation
from bindings.trees_public_mixin_score import boost_score, gbdt_adapter_score
from bindings.trees_public_gbdt_fit import (
    gbdt_fit, catboost_auto_learning_rate_op, gbdt_inspect, gbdt_initialize, adapter_classifier_fit,
    adapter_regressor_fit,
)
from bindings.trees_public_gbdt import (
    gbdt_predict, gbdt_predict_proba, gbdt_predict_classes, gbdt_tree_metadata_op, ordered_rmse_fit,
    feature_freq_fit, adapter_decision_function, adapter_predict, adapter_predict_proba, adapter_codes,
)


def trees_public_operation(model: PythonObject, operation: String, args: PythonObject,
                           var context: PythonObject) raises -> PythonObject:
    if py_unavailable(context):
        raise Error("ImportError: mojolearn: the tree ensembles are not connected")
    # Connect a loaded NumPy's native types (ndarray inputs, pandas
    # storage); outputs stay mojolearn Arrays.
    classical_connect_numpy(context)
    if operation == "trees_forest_getstate" or operation == "trees_iforest_getstate" or operation == "trees_gbdt_getstate":
        return tree_export_state(model, operation)
    if operation == "trees_embedding_apply":
        return embedding_apply(model, args, context)
    if operation == "trees_forest_construct":
        return forest_construct(model, args, context)
    if operation.startswith("trees_public_forest_"):
        return forest_public_operation(model, operation, args, context)
    if operation.startswith("trees_public_meta_"):
        return meta_public_operation(model, operation, args, context)
    if py_get(context, "cv_fixed_owner") is not None:
        context = fixed_owner_context(context)
        _ = fixed_owner_fields(model, context)
        classical_admit_value(args, context)
        if operation == "trees_gbdt_adapter_set_params":
            return fixed_owner_set_params(model, args[0], context)
    if operation == "trees_forest_get_params":
        return forest_get_params(model, context)
    if operation == "trees_forest_initialize":
        _ = forest_refresh(model, String(args[0]), context)
        return PythonObject(None)
    if operation == "trees_forest_fit":
        return forest_fit(model, args, context)
    if operation == "trees_forest_predict":
        return forest_predict(model, args, context)
    if operation == "trees_forest_predict_proba":
        return forest_predict_proba(model, args, context)
    if operation == "trees_score_forest":
        return forest_score(model, args, context)
    if operation.startswith("trees_iforest_"):
        return iforest_operation(model, operation, args, context)
    if operation == "trees_boost_score":
        return boost_score(model, args, context)
    if operation.startswith("trees_boost_"):
        return boost_operation(model, operation, args, context)
    if operation == "trees_gbdt_adapter_score":
        return gbdt_adapter_score(model, args, context)
    if operation.startswith("trees_gbdt_"):
        return _gbdt_operation(model, operation, args, context)
    if operation.startswith("trees_meta_"):
        # Voting*, Stacking*, MultiOutput*, OneVsRest, CalibratedClassifierCV
        return meta_operation(model, operation, args, context)
    raise Error("unknown tree ensemble operation " + operation)


def _gbdt_operation(model: PythonObject, operation: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """GradientBoosting, OrderedRMSE, ExperimentalTwoLevelFeatureFreq and
    the scikit-learn GBDT adapters (bindings/trees_public_gbdt.mojo)."""
    if operation == "trees_gbdt_init":
        return gbdt_initialize(model, args, context)
    if operation == "trees_gbdt_adapter_classifier_fit":
        return adapter_classifier_fit(model, args, context)
    if operation == "trees_gbdt_adapter_regressor_fit":
        return adapter_regressor_fit(model, args, context)
    if operation == "trees_gbdt_fit":
        return gbdt_fit(model, args, context)
    if operation == "trees_gbdt_params" or operation == "trees_gbdt_learning_rate" or operation == "trees_gbdt_boosting_type":
        return gbdt_inspect(model, operation, args, context)
    if operation == "trees_gbdt_auto_lr":
        return catboost_auto_learning_rate_op(args, context)
    if operation == "trees_gbdt_predict":
        return gbdt_predict(model, args[0], context)
    if operation == "trees_gbdt_predict_proba":
        return gbdt_predict_proba(model, args[0], context)
    if operation == "trees_gbdt_predict_classes":
        return gbdt_predict_classes(model, args[0], context)
    if operation == "trees_gbdt_leaf_counts":
        return gbdt_tree_metadata_op(model, "counts", context)
    if operation == "trees_gbdt_leaf_values":
        return gbdt_tree_metadata_op(model, "values", context)
    if operation == "trees_gbdt_ordered_fit":
        return ordered_rmse_fit(model, args, context)
    if operation == "trees_gbdt_feature_freq_fit":
        return feature_freq_fit(model, args, context)
    if operation == "trees_gbdt_adapter_decision":
        return adapter_decision_function(model, args[0], context)
    if operation == "trees_gbdt_adapter_predict":
        return adapter_predict(model, args[0], context)
    if operation == "trees_gbdt_adapter_proba":
        return adapter_predict_proba(model, args[0], context)
    if operation == "trees_gbdt_adapter_codes":
        return adapter_codes(model, args[0], context)
    raise Error("unknown gradient boosting operation " + operation)


def py_unavailable(context: PythonObject) raises -> Bool:
    from bindings.py_balanced import py_contains
    return py_contains(context, "trees_unavailable")
