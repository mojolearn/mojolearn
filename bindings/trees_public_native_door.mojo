# SPDX-License-Identifier: Apache-2.0
# The tree-ensemble owner's operations (`classical_native_call` operations
# `trees_*`, connected by python/mojolearn/_native_trees_bridge.py). Each
# public method of a connected tree class is one call here.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get
from bindings.classical_pandas_door import classical_admit_value
from bindings.msel_fixed_owner_unpack import fixed_owner_context, fixed_owner_fields
from bindings.msel_fixed_owner_pack import fixed_owner_set_params
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.py_trees_resident import trees_resident_stats
from bindings.trees_public_forest_door import forest_fit, forest_predict, forest_predict_proba, forest_get_params, forest_refresh
from bindings.trees_constructor_native_pack import forest_construct
from bindings.trees_boost_constructor_native_pack import dart_construct
from bindings.trees_public_lifecycle_native_door import forest_public_operation
from bindings.trees_public_score_door import forest_score
from bindings.trees_public_iforest_door import iforest_operation
from bindings.iforest_public_lifecycle_pack import iforest_public_operation
from bindings.trees_public_iforest_door import iforest_getstate
from bindings.trees_bagging_members_native_pack import bagging_members, bagging_members_set
from bindings.trees_public_boost_door import boost_operation
from providers.python.tree_boost_public_lifecycle_pyapi_pack import tree_boost_public_operation
from bindings.trees_public_auxiliary_door import tree_export_state
from bindings.trees_gbdt_adapter_lifecycle_door import gbdt_adapter_public_operation, gbdt_model_public_operation
from bindings.trees_gbdt_parameters_pack import gbdt_adapter_construct, gbdt_adapter_set_params
from bindings.trees_public_meta_door import meta_operation
from bindings.trees_public_meta_lifecycle_pack import meta_public_operation
from bindings.trees_public_mixin_score_unpack import gbdt_adapter_score
from bindings.trees_public_mixin_score_pack import boost_score
from bindings.trees_explainer_native_door import trees_explainer_operation
from bindings.trees_agnostic_native_door import trees_agnostic_operation
from bindings.trees_public_gbdt_fit_door import (
    gbdt_fit, catboost_auto_learning_rate_op, gbdt_inspect, gbdt_initialize, adapter_classifier_fit,
    adapter_regressor_fit, gbdt_ordered_construct, gbdt_feature_freq_construct,
)
from bindings.trees_public_gbdt_door import (
    gbdt_predict, gbdt_predict_classes, gbdt_tree_metadata_op, ordered_rmse_fit,
    feature_freq_fit, adapter_decision_function, adapter_predict, adapter_predict_proba,
)
from bindings.trees_keys_unpack import trees_keys
from bindings.native_operation_codes import OP_LEAF_NONE
from bindings.operation_names_trees import (
    TREES_OPN_TREES_RESIDENT_STATS, TREES_OPN_TREES_GBDT_ADAPTER_CONSTRUCT, TREES_OPN_TREES_GBDT_ADAPTER_PUBLIC_SET_PARAMS,
    TREES_OPN_TREES_IFOREST_GETSTATE, TREES_OPN_TREES_FOREST_GETSTATE, TREES_OPN_TREES_GBDT_GETSTATE,
    TREES_OPN_TREES_FOREST_CONSTRUCT, TREES_OPN_TREES_DART_CONSTRUCT, TREES_OPN_TREES_GBDT_ADAPTER_SET_PARAMS,
    TREES_OPN_TREES_FOREST_GET_PARAMS, TREES_OPN_TREES_FOREST_INITIALIZE, TREES_OPN_TREES_FOREST_FIT,
    TREES_OPN_TREES_FOREST_PREDICT, TREES_OPN_TREES_FOREST_PREDICT_PROBA, TREES_OPN_TREES_SCORE_FOREST,
    TREES_OPN_TREES_BAGGING_MEMBERS, TREES_OPN_TREES_BAGGING_MEMBERS_SET, TREES_OPN_TREES_BOOST_SCORE,
    TREES_OPN_TREES_GBDT_ADAPTER_SCORE, TREES_OPN_TREES_GBDT_INIT, TREES_OPN_TREES_GBDT_ORDERED_INIT,
    TREES_OPN_TREES_GBDT_FEATURE_FREQ_INIT, TREES_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT,
    TREES_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT, TREES_OPN_TREES_GBDT_FIT, TREES_OPN_TREES_GBDT_PARAMS,
    TREES_OPN_TREES_GBDT_LEARNING_RATE, TREES_OPN_TREES_GBDT_BOOSTING_TYPE, TREES_OPN_TREES_GBDT_AUTO_LR,
    TREES_OPN_TREES_GBDT_PREDICT, TREES_OPN_TREES_GBDT_PREDICT_CLASSES, TREES_OPN_TREES_GBDT_LEAF_COUNTS,
    TREES_OPN_TREES_GBDT_LEAF_VALUES, TREES_OPN_TREES_GBDT_ORDERED_FIT, TREES_OPN_TREES_GBDT_FEATURE_FREQ_FIT,
    TREES_OPN_TREES_GBDT_ADAPTER_DECISION, TREES_OPN_TREES_GBDT_ADAPTER_PREDICT, TREES_OPN_TREES_GBDT_ADAPTER_PROBA,
    TREES_FIRST_LATE, TREES_END_PUBLIC_IFOREST, TREES_END_PUBLIC_FOREST, TREES_END_PUBLIC_GBDT_ADAPTER,
    TREES_END_PUBLIC_GBDT_MODEL, TREES_END_PUBLIC_META, TREES_END_EXPLAINER, TREES_FIRST_IFOREST, TREES_END_IFOREST,
    TREES_FIRST_PUBLIC_BOOST, TREES_END_PUBLIC_BOOST, TREES_FIRST_BOOST, TREES_END_BOOST, TREES_FIRST_GBDT,
    TREES_END_GBDT, TREES_FIRST_META, TREES_END_META,
)


def trees_public_operation(model: PythonObject, operation: Int, name: String, args: PythonObject,
                           var context: PythonObject) raises -> PythonObject:
    """One `trees_*` operation by its leaf code
    (bindings/operation_names_trees.mojo; lane bw3-trees: the 47 String
    tests of the former name chain are integer compares, in the same order).
    `name` is the operation table's own String, borrowed: the sub-owners that
    still dispatch on the name (iforest, boost, forest/gbdt/meta lifecycles,
    explainer, agnostic) receive it unchanged. OP_LEAF_NONE is a `trees_*`
    name the table does not hold: the old prefix route, for its owner's error
    text only."""
    if py_unavailable(context):
        raise Error("ImportError: mojolearn: the tree ensembles are not connected")
    # Connect a loaded NumPy's native types (ndarray inputs, pandas
    # storage); outputs stay mojolearn Arrays.
    classical_connect_numpy(context)
    if operation == OP_LEAF_NONE:
        return _trees_prefixed(model, name, args, context)
    if operation == TREES_OPN_TREES_RESIDENT_STATS:
        # (snapshots kept, bytes kept, uploads so far): tools/ checks only.
        return trees_resident_stats()
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_CONSTRUCT:
        return gbdt_adapter_construct(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_PUBLIC_SET_PARAMS:
        return gbdt_adapter_set_params(model, args[0], context)
    if operation == TREES_OPN_TREES_IFOREST_GETSTATE:
        # Lane fix-iforest-save-device-forest: a fitted IsolationForest
        # pickles its forest, not the (possibly borrowed) training matrix.
        return iforest_getstate(model, context)
    if operation == TREES_OPN_TREES_FOREST_GETSTATE or operation == TREES_OPN_TREES_GBDT_GETSTATE:
        return tree_export_state(model, operation == TREES_OPN_TREES_FOREST_GETSTATE)
    if operation == TREES_OPN_TREES_FOREST_CONSTRUCT:
        return forest_construct(model, args, context)
    if operation == TREES_OPN_TREES_DART_CONSTRUCT:
        return dart_construct(model, args, context)
    if operation < TREES_FIRST_LATE:
        # The public lifecycles and the explainers (groups in chain order).
        if operation < TREES_END_PUBLIC_IFOREST:
            return iforest_public_operation(model, name, args, context)
        if operation < TREES_END_PUBLIC_FOREST:
            return forest_public_operation(model, operation, name, args, context)
        if operation < TREES_END_PUBLIC_GBDT_ADAPTER:
            return gbdt_adapter_public_operation(model, name, args, context)
        if operation < TREES_END_PUBLIC_GBDT_MODEL:
            return gbdt_model_public_operation(model, name, args, context)
        if operation < TREES_END_PUBLIC_META:
            return meta_public_operation(model, name, args, context)
        if operation < TREES_END_EXPLAINER:
            return trees_explainer_operation(model, name, args, context)
        return trees_agnostic_operation(model, name, args, context)
    var nk = trees_keys()
    if py_get(context, nk[].k_cv_fixed_owner) is not None:
        context = fixed_owner_context(context)
        _ = fixed_owner_fields(model, context)
        classical_admit_value(args, context)
        if operation == TREES_OPN_TREES_GBDT_ADAPTER_SET_PARAMS:
            return fixed_owner_set_params(model, args[0], context)
    if operation == TREES_OPN_TREES_FOREST_GET_PARAMS:
        return forest_get_params(model, context)
    if operation == TREES_OPN_TREES_FOREST_INITIALIZE:
        _ = forest_refresh(model, String(args[0]), context)
        return PythonObject(None)
    if operation == TREES_OPN_TREES_FOREST_FIT:
        return forest_fit(model, args, context)
    if operation == TREES_OPN_TREES_FOREST_PREDICT:
        return forest_predict(model, args, context)
    if operation == TREES_OPN_TREES_FOREST_PREDICT_PROBA:
        return forest_predict_proba(model, args, context)
    if operation == TREES_OPN_TREES_SCORE_FOREST:
        return forest_score(model, args, context)
    if operation >= TREES_FIRST_IFOREST and operation < TREES_END_IFOREST:
        return iforest_operation(model, name, args, context)
    if operation == TREES_OPN_TREES_BAGGING_MEMBERS:
        return bagging_members(model)
    if operation == TREES_OPN_TREES_BAGGING_MEMBERS_SET:
        return bagging_members_set(model, args[0], context)
    if operation >= TREES_FIRST_PUBLIC_BOOST and operation < TREES_END_PUBLIC_BOOST:
        return tree_boost_public_operation(model, name, args, context)
    if operation == TREES_OPN_TREES_BOOST_SCORE:
        return boost_score(model, args, context)
    if operation >= TREES_FIRST_BOOST and operation < TREES_END_BOOST:
        return boost_operation(model, name, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_SCORE:
        return gbdt_adapter_score(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_SET_PARAMS or (operation >= TREES_FIRST_GBDT and operation < TREES_END_GBDT):
        # `trees_gbdt_adapter_set_params` without a fixed owner reached the
        # GBDT chain by its prefix and was refused there.
        return _gbdt_operation(model, operation, name, args, context)
    if operation >= TREES_FIRST_META and operation < TREES_END_META:
        # Voting*, Stacking*, MultiOutput*, OneVsRest, CalibratedClassifierCV
        return meta_operation(model, name, args, context)
    raise Error("unknown tree ensemble operation " + name)


def _trees_prefixed(model: PythonObject, operation: String, args: PythonObject,
                    var context: PythonObject) raises -> PythonObject:
    """A `trees_*` name the operation table does not hold: the prefix tests
    of the former chain, in its order (every exact name is in the table), so
    the owner a prefix reaches raises its own unknown-operation text."""
    if operation.startswith("trees_public_iforest_"):
        return iforest_public_operation(model, operation, args, context)
    if operation.startswith("trees_public_forest_"):
        return forest_public_operation(model, OP_LEAF_NONE, operation, args, context)
    if operation.startswith("trees_public_gbdt_adapter_"):
        return gbdt_adapter_public_operation(model, operation, args, context)
    if operation.startswith("trees_public_gbdt_model_"):
        return gbdt_model_public_operation(model, operation, args, context)
    if operation.startswith("trees_public_meta_"):
        return meta_public_operation(model, operation, args, context)
    if operation.startswith("trees_explainer_"):
        return trees_explainer_operation(model, operation, args, context)
    if operation.startswith("trees_agnostic_"):
        return trees_agnostic_operation(model, operation, args, context)
    var nk = trees_keys()
    if py_get(context, nk[].k_cv_fixed_owner) is not None:
        context = fixed_owner_context(context)
        _ = fixed_owner_fields(model, context)
        classical_admit_value(args, context)
    if operation.startswith("trees_iforest_"):
        return iforest_operation(model, operation, args, context)
    if operation.startswith("trees_public_boost_"):
        return tree_boost_public_operation(model, operation, args, context)
    if operation.startswith("trees_boost_"):
        return boost_operation(model, operation, args, context)
    if operation.startswith("trees_gbdt_"):
        raise Error("unknown gradient boosting operation " + operation)
    if operation.startswith("trees_meta_"):
        return meta_operation(model, operation, args, context)
    raise Error("unknown tree ensemble operation " + operation)


def _gbdt_operation(model: PythonObject, operation: Int, name: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """GradientBoosting, OrderedRMSE, ExperimentalTwoLevelFeatureFreq and
    the scikit-learn GBDT adapters (bindings/trees_public_gbdt.mojo), by
    leaf code; `name` is for the unknown-operation text only."""
    if operation == TREES_OPN_TREES_GBDT_INIT:
        return gbdt_initialize(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ORDERED_INIT:
        return gbdt_ordered_construct(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_FEATURE_FREQ_INIT:
        return gbdt_feature_freq_construct(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT:
        return adapter_classifier_fit(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT:
        return adapter_regressor_fit(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_FIT:
        return gbdt_fit(model, args, context)
    if (operation == TREES_OPN_TREES_GBDT_PARAMS or operation == TREES_OPN_TREES_GBDT_LEARNING_RATE
            or operation == TREES_OPN_TREES_GBDT_BOOSTING_TYPE):
        return gbdt_inspect(model, operation, args, context)
    if operation == TREES_OPN_TREES_GBDT_AUTO_LR:
        return catboost_auto_learning_rate_op(args, context)
    if operation == TREES_OPN_TREES_GBDT_PREDICT:
        return gbdt_predict(model, args[0], context)
    if operation == TREES_OPN_TREES_GBDT_PREDICT_CLASSES:
        return gbdt_predict_classes(model, args[0], context)
    if operation == TREES_OPN_TREES_GBDT_LEAF_COUNTS:
        return gbdt_tree_metadata_op(model, "counts", context)
    if operation == TREES_OPN_TREES_GBDT_LEAF_VALUES:
        return gbdt_tree_metadata_op(model, "values", context)
    if operation == TREES_OPN_TREES_GBDT_ORDERED_FIT:
        return ordered_rmse_fit(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_FEATURE_FREQ_FIT:
        return feature_freq_fit(model, args, context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_DECISION:
        return adapter_decision_function(model, args[0], context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_PREDICT:
        return adapter_predict(model, args[0], context)
    if operation == TREES_OPN_TREES_GBDT_ADAPTER_PROBA:
        return adapter_predict_proba(model, args[0], context)
    raise Error("unknown gradient boosting operation " + name)


def py_unavailable(context: PythonObject) raises -> Bool:
    from bindings.py_balanced import py_contains
    var nk = trees_keys()
    return py_contains(context, nk[].k_trees_unavailable)
