# SPDX-License-Identifier: Apache-2.0
# The `trees_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw3-trees, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# Each name has a leaf code, numbered in the order of the tree owner's former
# name chain (bindings/trees_public_native_door.mojo `trees_public_operation`)
# and grouped by the sub-owner it forwards to, so each prefix test of that chain
# is one range test (`TREES_FIRST_<GROUP>` <= code < `TREES_END_<GROUP>`). The tree
# door resolves a caller's str to the code with ONE dictionary lookup
# (bindings/native_operation_names_unpack.mojo); a `trees_*` name the table does
# not hold keeps its String for the old prefix route and its error text only.
# A constant is `TREES_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Every name a caller
# spells (the Python shells, model selection, the owners' own compares) is here.

comptime TREES_OPN_TREES_RESIDENT_STATS = 0
comptime TREES_OPN_TREES_GBDT_ADAPTER_CONSTRUCT = 1
comptime TREES_OPN_TREES_GBDT_ADAPTER_PUBLIC_SET_PARAMS = 2
comptime TREES_OPN_TREES_IFOREST_GETSTATE = 3
comptime TREES_OPN_TREES_FOREST_GETSTATE = 4
comptime TREES_OPN_TREES_GBDT_GETSTATE = 5
comptime TREES_OPN_TREES_FOREST_CONSTRUCT = 6
comptime TREES_OPN_TREES_DART_CONSTRUCT = 7
comptime TREES_OPN_TREES_PUBLIC_IFOREST_DECISION_FUNCTION = 8
comptime TREES_OPN_TREES_PUBLIC_IFOREST_FIT = 9
comptime TREES_OPN_TREES_PUBLIC_IFOREST_FIT_PREDICT = 10
comptime TREES_OPN_TREES_PUBLIC_IFOREST_PREDICT = 11
comptime TREES_OPN_TREES_PUBLIC_IFOREST_SCORE_SAMPLES = 12
comptime TREES_OPN_TREES_PUBLIC_FOREST_APPLY = 13
comptime TREES_OPN_TREES_PUBLIC_FOREST_CAPTURE_MODE = 14
comptime TREES_OPN_TREES_PUBLIC_FOREST_DEPTH = 15
comptime TREES_OPN_TREES_PUBLIC_FOREST_EFFECTIVE_MODE = 16
comptime TREES_OPN_TREES_PUBLIC_FOREST_FIT = 17
comptime TREES_OPN_TREES_PUBLIC_FOREST_IS_FITTED = 18
comptime TREES_OPN_TREES_PUBLIC_FOREST_N_LEAVES = 19
comptime TREES_OPN_TREES_PUBLIC_FOREST_ORDERED_RESIDENT_AUTO = 20
comptime TREES_OPN_TREES_PUBLIC_FOREST_PREDICT = 21
comptime TREES_OPN_TREES_PUBLIC_FOREST_PREDICT_WITH_LEAVES = 22
comptime TREES_OPN_TREES_PUBLIC_FOREST_PREDICTION_ENGINE = 23
comptime TREES_OPN_TREES_PUBLIC_FOREST_PROBA = 24
comptime TREES_OPN_TREES_PUBLIC_FOREST_REFRESH = 25
comptime TREES_OPN_TREES_PUBLIC_FOREST_SCORE = 26
comptime TREES_OPN_TREES_PUBLIC_FOREST_SET_PARAMS = 27
comptime TREES_OPN_TREES_PUBLIC_FOREST_SETSTATE = 28
comptime TREES_OPN_TREES_PUBLIC_FOREST_STAGED_PREDICT = 29
comptime TREES_OPN_TREES_PUBLIC_FOREST_VALIDATE_PARAMS = 30
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_CLASSIFIER_FIT = 31
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_DECISION = 32
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_IS_FITTED = 33
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_LOAD = 34
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_NEW_LEARNER = 35
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_PREDICT = 36
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_PROBA = 37
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_REGRESSION_PREDICT = 38
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_REGRESSOR_FIT = 39
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_SAVE = 40
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_SCORE = 41
comptime TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_VALIDATE_PARAMS = 42
comptime TREES_OPN_TREES_PUBLIC_GBDT_MODEL_FEATURE_FREQ_FIT = 43
comptime TREES_OPN_TREES_PUBLIC_GBDT_MODEL_FIT = 44
comptime TREES_OPN_TREES_PUBLIC_GBDT_MODEL_ORDERED_FIT = 45
comptime TREES_OPN_TREES_PUBLIC_GBDT_MODEL_PREDICT = 46
comptime TREES_OPN_TREES_PUBLIC_GBDT_MODEL_PROBA = 47
comptime TREES_OPN_TREES_PUBLIC_META_FIT = 48
comptime TREES_OPN_TREES_PUBLIC_META_FIT_TRANSFORM = 49
comptime TREES_OPN_TREES_PUBLIC_META_PREDICT = 50
comptime TREES_OPN_TREES_PUBLIC_META_PREDICT_PROBA = 51
comptime TREES_OPN_TREES_PUBLIC_META_SCORE = 52
comptime TREES_OPN_TREES_PUBLIC_META_TRANSFORM = 53
comptime TREES_OPN_TREES_EXPLAINER_CLOSE = 54
comptime TREES_OPN_TREES_EXPLAINER_INIT = 55
comptime TREES_OPN_TREES_EXPLAINER_VALUES = 56
comptime TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT = 57
comptime TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES = 58
comptime TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT = 59
comptime TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES = 60
comptime TREES_OPN_TREES_GBDT_ADAPTER_SET_PARAMS = 61
comptime TREES_OPN_TREES_FOREST_GET_PARAMS = 62
comptime TREES_OPN_TREES_FOREST_INITIALIZE = 63
comptime TREES_OPN_TREES_FOREST_FIT = 64
comptime TREES_OPN_TREES_FOREST_PREDICT = 65
comptime TREES_OPN_TREES_FOREST_PREDICT_PROBA = 66
comptime TREES_OPN_TREES_SCORE_FOREST = 67
comptime TREES_OPN_TREES_IFOREST_DECISION_FUNCTION = 68
comptime TREES_OPN_TREES_IFOREST_FIT = 69
comptime TREES_OPN_TREES_IFOREST_FIT_PREDICT = 70
comptime TREES_OPN_TREES_IFOREST_PREDICT = 71
comptime TREES_OPN_TREES_IFOREST_SCORE_SAMPLES = 72
comptime TREES_OPN_TREES_BAGGING_MEMBERS = 73
comptime TREES_OPN_TREES_BAGGING_MEMBERS_SET = 74
comptime TREES_OPN_TREES_PUBLIC_BOOST_APPLY = 75
comptime TREES_OPN_TREES_PUBLIC_BOOST_DECISION_FUNCTION = 76
comptime TREES_OPN_TREES_PUBLIC_BOOST_FIT = 77
comptime TREES_OPN_TREES_PUBLIC_BOOST_FIT_TRANSFORM = 78
comptime TREES_OPN_TREES_PUBLIC_BOOST_PREDICT = 79
comptime TREES_OPN_TREES_PUBLIC_BOOST_PREDICT_PROBA = 80
comptime TREES_OPN_TREES_PUBLIC_BOOST_SCORE = 81
comptime TREES_OPN_TREES_PUBLIC_BOOST_TRANSFORM = 82
comptime TREES_OPN_TREES_BOOST_SCORE = 83
comptime TREES_OPN_TREES_BOOST_DECISION_FUNCTION = 84
comptime TREES_OPN_TREES_BOOST_FIT = 85
comptime TREES_OPN_TREES_BOOST_FIT_TRANSFORM = 86
comptime TREES_OPN_TREES_BOOST_PREDICT = 87
comptime TREES_OPN_TREES_BOOST_PREDICT_PROBA = 88
comptime TREES_OPN_TREES_BOOST_TRANSFORM = 89
comptime TREES_OPN_TREES_GBDT_ADAPTER_SCORE = 90
comptime TREES_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT = 91
comptime TREES_OPN_TREES_GBDT_ADAPTER_DECISION = 92
comptime TREES_OPN_TREES_GBDT_ADAPTER_PREDICT = 93
comptime TREES_OPN_TREES_GBDT_ADAPTER_PROBA = 94
comptime TREES_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT = 95
comptime TREES_OPN_TREES_GBDT_AUTO_LR = 96
comptime TREES_OPN_TREES_GBDT_BOOSTING_TYPE = 97
comptime TREES_OPN_TREES_GBDT_FEATURE_FREQ_FIT = 98
comptime TREES_OPN_TREES_GBDT_FEATURE_FREQ_INIT = 99
comptime TREES_OPN_TREES_GBDT_FIT = 100
comptime TREES_OPN_TREES_GBDT_INIT = 101
comptime TREES_OPN_TREES_GBDT_LEAF_COUNTS = 102
comptime TREES_OPN_TREES_GBDT_LEAF_VALUES = 103
comptime TREES_OPN_TREES_GBDT_LEARNING_RATE = 104
comptime TREES_OPN_TREES_GBDT_ORDERED_FIT = 105
comptime TREES_OPN_TREES_GBDT_ORDERED_INIT = 106
comptime TREES_OPN_TREES_GBDT_PARAMS = 107
comptime TREES_OPN_TREES_GBDT_PREDICT = 108
comptime TREES_OPN_TREES_GBDT_PREDICT_CLASSES = 109
comptime TREES_OPN_TREES_META_FIT = 110
comptime TREES_OPN_TREES_META_FIT_TRANSFORM = 111
comptime TREES_OPN_TREES_META_PREDICT = 112
comptime TREES_OPN_TREES_META_PREDICT_PROBA = 113
comptime TREES_OPN_TREES_META_SCORE = 114
comptime TREES_OPN_TREES_META_TRANSFORM = 115
# lane mf-imp6: `feature_importances_` of GradientBoosting, its subclasses and
# the adapters, after the chain's groups (routed beside the GBDT group).
comptime TREES_OPN_TREES_GBDT_FEATURE_IMPORTANCES = 116

# The first code of each group and its end (exclusive), in chain order.
comptime TREES_FIRST_PUBLIC_IFOREST = 8
comptime TREES_END_PUBLIC_IFOREST = 13
comptime TREES_FIRST_PUBLIC_FOREST = 13
comptime TREES_END_PUBLIC_FOREST = 31
comptime TREES_FIRST_PUBLIC_GBDT_ADAPTER = 31
comptime TREES_END_PUBLIC_GBDT_ADAPTER = 43
comptime TREES_FIRST_PUBLIC_GBDT_MODEL = 43
comptime TREES_END_PUBLIC_GBDT_MODEL = 48
comptime TREES_FIRST_PUBLIC_META = 48
comptime TREES_END_PUBLIC_META = 54
comptime TREES_FIRST_EXPLAINER = 54
comptime TREES_END_EXPLAINER = 57
comptime TREES_FIRST_AGNOSTIC = 57
comptime TREES_END_AGNOSTIC = 61
comptime TREES_FIRST_LATE = 61
comptime TREES_FIRST_IFOREST = 68
comptime TREES_END_IFOREST = 73
comptime TREES_FIRST_PUBLIC_BOOST = 75
comptime TREES_END_PUBLIC_BOOST = 83
comptime TREES_FIRST_BOOST = 84
comptime TREES_END_BOOST = 90
comptime TREES_FIRST_GBDT = 91
comptime TREES_END_GBDT = 110
comptime TREES_FIRST_META = 110
comptime TREES_END_META = 116
comptime TREES_COUNT = 117


def operation_names_trees(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("trees_resident_stats")
    leaves.append(TREES_OPN_TREES_RESIDENT_STATS)
    names.append("trees_gbdt_adapter_construct")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_CONSTRUCT)
    names.append("trees_gbdt_adapter_public_set_params")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_PUBLIC_SET_PARAMS)
    names.append("trees_iforest_getstate")
    leaves.append(TREES_OPN_TREES_IFOREST_GETSTATE)
    names.append("trees_forest_getstate")
    leaves.append(TREES_OPN_TREES_FOREST_GETSTATE)
    names.append("trees_gbdt_getstate")
    leaves.append(TREES_OPN_TREES_GBDT_GETSTATE)
    names.append("trees_forest_construct")
    leaves.append(TREES_OPN_TREES_FOREST_CONSTRUCT)
    names.append("trees_dart_construct")
    leaves.append(TREES_OPN_TREES_DART_CONSTRUCT)
    names.append("trees_public_iforest_decision_function")
    leaves.append(TREES_OPN_TREES_PUBLIC_IFOREST_DECISION_FUNCTION)
    names.append("trees_public_iforest_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_IFOREST_FIT)
    names.append("trees_public_iforest_fit_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_IFOREST_FIT_PREDICT)
    names.append("trees_public_iforest_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_IFOREST_PREDICT)
    names.append("trees_public_iforest_score_samples")
    leaves.append(TREES_OPN_TREES_PUBLIC_IFOREST_SCORE_SAMPLES)
    names.append("trees_public_forest_apply")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_APPLY)
    names.append("trees_public_forest_capture_mode")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_CAPTURE_MODE)
    names.append("trees_public_forest_depth")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_DEPTH)
    names.append("trees_public_forest_effective_mode")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_EFFECTIVE_MODE)
    names.append("trees_public_forest_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_FIT)
    names.append("trees_public_forest_is_fitted")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_IS_FITTED)
    names.append("trees_public_forest_n_leaves")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_N_LEAVES)
    names.append("trees_public_forest_ordered_resident_auto")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_ORDERED_RESIDENT_AUTO)
    names.append("trees_public_forest_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_PREDICT)
    names.append("trees_public_forest_predict_with_leaves")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_PREDICT_WITH_LEAVES)
    names.append("trees_public_forest_prediction_engine")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_PREDICTION_ENGINE)
    names.append("trees_public_forest_proba")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_PROBA)
    names.append("trees_public_forest_refresh")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_REFRESH)
    names.append("trees_public_forest_score")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_SCORE)
    names.append("trees_public_forest_set_params")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_SET_PARAMS)
    names.append("trees_public_forest_setstate")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_SETSTATE)
    names.append("trees_public_forest_staged_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_STAGED_PREDICT)
    names.append("trees_public_forest_validate_params")
    leaves.append(TREES_OPN_TREES_PUBLIC_FOREST_VALIDATE_PARAMS)
    names.append("trees_public_gbdt_adapter_classifier_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_CLASSIFIER_FIT)
    names.append("trees_public_gbdt_adapter_decision")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_DECISION)
    names.append("trees_public_gbdt_adapter_is_fitted")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_IS_FITTED)
    names.append("trees_public_gbdt_adapter_load")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_LOAD)
    names.append("trees_public_gbdt_adapter_new_learner")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_NEW_LEARNER)
    names.append("trees_public_gbdt_adapter_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_PREDICT)
    names.append("trees_public_gbdt_adapter_proba")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_PROBA)
    names.append("trees_public_gbdt_adapter_regression_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_REGRESSION_PREDICT)
    names.append("trees_public_gbdt_adapter_regressor_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_REGRESSOR_FIT)
    names.append("trees_public_gbdt_adapter_save")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_SAVE)
    names.append("trees_public_gbdt_adapter_score")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_SCORE)
    names.append("trees_public_gbdt_adapter_validate_params")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_ADAPTER_VALIDATE_PARAMS)
    names.append("trees_public_gbdt_model_feature_freq_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_MODEL_FEATURE_FREQ_FIT)
    names.append("trees_public_gbdt_model_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_MODEL_FIT)
    names.append("trees_public_gbdt_model_ordered_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_MODEL_ORDERED_FIT)
    names.append("trees_public_gbdt_model_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_MODEL_PREDICT)
    names.append("trees_public_gbdt_model_proba")
    leaves.append(TREES_OPN_TREES_PUBLIC_GBDT_MODEL_PROBA)
    names.append("trees_public_meta_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_FIT)
    names.append("trees_public_meta_fit_transform")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_FIT_TRANSFORM)
    names.append("trees_public_meta_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_PREDICT)
    names.append("trees_public_meta_predict_proba")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_PREDICT_PROBA)
    names.append("trees_public_meta_score")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_SCORE)
    names.append("trees_public_meta_transform")
    leaves.append(TREES_OPN_TREES_PUBLIC_META_TRANSFORM)
    names.append("trees_explainer_close")
    leaves.append(TREES_OPN_TREES_EXPLAINER_CLOSE)
    names.append("trees_explainer_init")
    leaves.append(TREES_OPN_TREES_EXPLAINER_INIT)
    names.append("trees_explainer_values")
    leaves.append(TREES_OPN_TREES_EXPLAINER_VALUES)
    names.append("trees_agnostic_kernel_init")
    leaves.append(TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT)
    names.append("trees_agnostic_kernel_values")
    leaves.append(TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES)
    names.append("trees_agnostic_permutation_init")
    leaves.append(TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT)
    names.append("trees_agnostic_permutation_values")
    leaves.append(TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES)
    names.append("trees_gbdt_adapter_set_params")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_SET_PARAMS)
    names.append("trees_forest_get_params")
    leaves.append(TREES_OPN_TREES_FOREST_GET_PARAMS)
    names.append("trees_forest_initialize")
    leaves.append(TREES_OPN_TREES_FOREST_INITIALIZE)
    names.append("trees_forest_fit")
    leaves.append(TREES_OPN_TREES_FOREST_FIT)
    names.append("trees_forest_predict")
    leaves.append(TREES_OPN_TREES_FOREST_PREDICT)
    names.append("trees_forest_predict_proba")
    leaves.append(TREES_OPN_TREES_FOREST_PREDICT_PROBA)
    names.append("trees_score_forest")
    leaves.append(TREES_OPN_TREES_SCORE_FOREST)
    names.append("trees_iforest_decision_function")
    leaves.append(TREES_OPN_TREES_IFOREST_DECISION_FUNCTION)
    names.append("trees_iforest_fit")
    leaves.append(TREES_OPN_TREES_IFOREST_FIT)
    names.append("trees_iforest_fit_predict")
    leaves.append(TREES_OPN_TREES_IFOREST_FIT_PREDICT)
    names.append("trees_iforest_predict")
    leaves.append(TREES_OPN_TREES_IFOREST_PREDICT)
    names.append("trees_iforest_score_samples")
    leaves.append(TREES_OPN_TREES_IFOREST_SCORE_SAMPLES)
    names.append("trees_bagging_members")
    leaves.append(TREES_OPN_TREES_BAGGING_MEMBERS)
    names.append("trees_bagging_members_set")
    leaves.append(TREES_OPN_TREES_BAGGING_MEMBERS_SET)
    names.append("trees_public_boost_apply")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_APPLY)
    names.append("trees_public_boost_decision_function")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_DECISION_FUNCTION)
    names.append("trees_public_boost_fit")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_FIT)
    names.append("trees_public_boost_fit_transform")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_FIT_TRANSFORM)
    names.append("trees_public_boost_predict")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_PREDICT)
    names.append("trees_public_boost_predict_proba")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_PREDICT_PROBA)
    names.append("trees_public_boost_score")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_SCORE)
    names.append("trees_public_boost_transform")
    leaves.append(TREES_OPN_TREES_PUBLIC_BOOST_TRANSFORM)
    names.append("trees_boost_score")
    leaves.append(TREES_OPN_TREES_BOOST_SCORE)
    names.append("trees_boost_decision_function")
    leaves.append(TREES_OPN_TREES_BOOST_DECISION_FUNCTION)
    names.append("trees_boost_fit")
    leaves.append(TREES_OPN_TREES_BOOST_FIT)
    names.append("trees_boost_fit_transform")
    leaves.append(TREES_OPN_TREES_BOOST_FIT_TRANSFORM)
    names.append("trees_boost_predict")
    leaves.append(TREES_OPN_TREES_BOOST_PREDICT)
    names.append("trees_boost_predict_proba")
    leaves.append(TREES_OPN_TREES_BOOST_PREDICT_PROBA)
    names.append("trees_boost_transform")
    leaves.append(TREES_OPN_TREES_BOOST_TRANSFORM)
    names.append("trees_gbdt_adapter_score")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_SCORE)
    names.append("trees_gbdt_adapter_classifier_fit")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT)
    names.append("trees_gbdt_adapter_decision")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_DECISION)
    names.append("trees_gbdt_adapter_predict")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_PREDICT)
    names.append("trees_gbdt_adapter_proba")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_PROBA)
    names.append("trees_gbdt_adapter_regressor_fit")
    leaves.append(TREES_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT)
    names.append("trees_gbdt_auto_lr")
    leaves.append(TREES_OPN_TREES_GBDT_AUTO_LR)
    names.append("trees_gbdt_boosting_type")
    leaves.append(TREES_OPN_TREES_GBDT_BOOSTING_TYPE)
    names.append("trees_gbdt_feature_freq_fit")
    leaves.append(TREES_OPN_TREES_GBDT_FEATURE_FREQ_FIT)
    names.append("trees_gbdt_feature_freq_init")
    leaves.append(TREES_OPN_TREES_GBDT_FEATURE_FREQ_INIT)
    names.append("trees_gbdt_fit")
    leaves.append(TREES_OPN_TREES_GBDT_FIT)
    names.append("trees_gbdt_init")
    leaves.append(TREES_OPN_TREES_GBDT_INIT)
    names.append("trees_gbdt_leaf_counts")
    leaves.append(TREES_OPN_TREES_GBDT_LEAF_COUNTS)
    names.append("trees_gbdt_leaf_values")
    leaves.append(TREES_OPN_TREES_GBDT_LEAF_VALUES)
    names.append("trees_gbdt_learning_rate")
    leaves.append(TREES_OPN_TREES_GBDT_LEARNING_RATE)
    names.append("trees_gbdt_ordered_fit")
    leaves.append(TREES_OPN_TREES_GBDT_ORDERED_FIT)
    names.append("trees_gbdt_ordered_init")
    leaves.append(TREES_OPN_TREES_GBDT_ORDERED_INIT)
    names.append("trees_gbdt_params")
    leaves.append(TREES_OPN_TREES_GBDT_PARAMS)
    names.append("trees_gbdt_predict")
    leaves.append(TREES_OPN_TREES_GBDT_PREDICT)
    names.append("trees_gbdt_predict_classes")
    leaves.append(TREES_OPN_TREES_GBDT_PREDICT_CLASSES)
    names.append("trees_meta_fit")
    leaves.append(TREES_OPN_TREES_META_FIT)
    names.append("trees_meta_fit_transform")
    leaves.append(TREES_OPN_TREES_META_FIT_TRANSFORM)
    names.append("trees_meta_predict")
    leaves.append(TREES_OPN_TREES_META_PREDICT)
    names.append("trees_meta_predict_proba")
    leaves.append(TREES_OPN_TREES_META_PREDICT_PROBA)
    names.append("trees_meta_score")
    leaves.append(TREES_OPN_TREES_META_SCORE)
    names.append("trees_meta_transform")
    leaves.append(TREES_OPN_TREES_META_TRANSFORM)
    names.append("trees_gbdt_feature_importances")
    leaves.append(TREES_OPN_TREES_GBDT_FEATURE_IMPORTANCES)
