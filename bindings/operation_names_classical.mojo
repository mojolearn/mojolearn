# SPDX-License-Identifier: Apache-2.0
# The string operations of `classical_native_call` and of its tree door, in
# the classical owner's code table (bindings/native_operation_codes.mojo reads
# this file once per process, at table fill, and gives these names the route
# OP_ROUTE_CLASSICAL; lane bw2-classical, 2026-11-20). A leaf: no Python, no
# imports of an owner.
#
# WHAT IT REPLACES. `classical_c` (bindings/classical_native_pack.mojo) copied
# the caller's str into a String and walked up to 26 exact compares plus the
# tree door's 23 name and prefix tests on every call; `classical_trees_c`
# (bindings/classical_trees_door.mojo) did the same again for its own names.
# Now both resolve the caller's own str with ONE dictionary lookup
# (bindings/native_operation_names_unpack.mojo) and switch on the leaf below.
# Only a name the table does not hold builds its String (the prefix routes of
# unlisted `trees_*`, `host_tree_*`, `host_infer_*` and `family_*` names, and
# the unchanged unknown-operation errors).
#
# Leaves are numbered by owner: the classical door's own operations 0-25, the
# tree door's exact names 26-40, the `host_infer_*` names the tree door
# forwards by name 41-48 (so "the tree door's" is one range test). These
# names are never passed by code: `operation_code_names` (the tuple Python
# maps to `_OP`) stops before them, and the coded doors refuse their route.
# A constant is `CLASSICAL_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime CLASSICAL_OPN_OPERATION_CODES = 0
comptime CLASSICAL_OPN_RESTORE_NOT_FITTED = 1
comptime CLASSICAL_OPN_REDUCE_NOT_FITTED = 2
# 3 was "linear_regressor_score" (retired 2026-11-20, lane dw-xlinear: the x_linear door's "regressor_score").
# 3 is reused (lane digest, 2026-11-21): `model.digest_` (bindings/fitted_digest_door.mojo).
comptime CLASSICAL_OPN_FITTED_DIGEST = 3
comptime CLASSICAL_OPN_ARCHIVE_SAVE = 4
comptime CLASSICAL_OPN_ARCHIVE_NPZ = 5
comptime CLASSICAL_OPN_HOST_MODEL = 6
comptime CLASSICAL_OPN_LOAD = 7
comptime CLASSICAL_OPN_PARAM_NAMES = 8
comptime CLASSICAL_OPN_GETATTRIBUTE = 9
comptime CLASSICAL_OPN_RIDGE_FIT = 10
comptime CLASSICAL_OPN_OLS_FIT = 11
comptime CLASSICAL_OPN_PREDICT_PUBLIC = 12
comptime CLASSICAL_OPN_PREDICT_SCORE = 13
comptime CLASSICAL_OPN_SCORE_PUBLIC = 14
comptime CLASSICAL_OPN_SAVE = 15
comptime CLASSICAL_OPN_INITIALIZE = 16
comptime CLASSICAL_OPN_VALIDATE_PARAMS = 17
comptime CLASSICAL_OPN_GET_PARAMS = 18
comptime CLASSICAL_OPN_SET_PARAMS = 19
comptime CLASSICAL_OPN_IS_FITTED = 20
comptime CLASSICAL_OPN_MODEL_SHA256 = 21
comptime CLASSICAL_OPN_PREDICT = 22
comptime CLASSICAL_OPN_SCORE = 23
comptime CLASSICAL_OPN_NUMERIC_MODE_USED = 24
comptime CLASSICAL_OPN_VENDOR_USED = 25
# The tree door's (bindings/classical_trees_door.mojo `_trees_leaf`).
comptime CLASSICAL_OPN_HOST_KDE_FIT = 26
comptime CLASSICAL_OPN_HOST_IFOREST_FIT = 27
comptime CLASSICAL_OPN_HOST_DENSITY_SAVE = 28
comptime CLASSICAL_OPN_HOST_DENSITY_HASH = 29
comptime CLASSICAL_OPN_HOST_BOUND_HASH = 30
comptime CLASSICAL_OPN_HOST_DENSITY_INITIALIZE = 31
comptime CLASSICAL_OPN_ARCHIVE_INFER_PREDICT = 32
comptime CLASSICAL_OPN_ARCHIVE_INFER_PREDICT_PROBA = 33
comptime CLASSICAL_OPN_ARCHIVE_KNN_CLASSIFIER = 34
comptime CLASSICAL_OPN_ARCHIVE_KNN_REGRESSOR = 35
comptime CLASSICAL_OPN_HOST_ARCHIVE_PREDICT = 36
comptime CLASSICAL_OPN_HOST_ARCHIVE_PREDICT_PROBA = 37
comptime CLASSICAL_OPN_HOST_ARCHIVE_MODEL = 38
comptime CLASSICAL_OPN_HOST_ARCHIVE_LOAD = 39
comptime CLASSICAL_OPN_HOST_ARCHIVE_CLASSICAL_MODEL = 40
# Forwarded by name to host_inference_operation (the table's String, borrowed).
comptime CLASSICAL_OPN_HOST_INFER_PREDICT = 41
comptime CLASSICAL_OPN_HOST_INFER_PREDICT_PROBA = 42
comptime CLASSICAL_OPN_HOST_INFER_PREDICT_LOG_PROBA = 43
comptime CLASSICAL_OPN_HOST_INFER_DECISION_FUNCTION = 44
comptime CLASSICAL_OPN_HOST_INFER_TRANSFORM = 45
comptime CLASSICAL_OPN_HOST_INFER_INVERSE_TRANSFORM = 46
comptime CLASSICAL_OPN_HOST_INFER_SCORE = 47
comptime CLASSICAL_OPN_HOST_INFER_SCORE_SAMPLES = 48

#: The first leaf the classical door forwards to the tree door.
comptime CLASSICAL_TREES_FIRST = CLASSICAL_OPN_HOST_KDE_FIT
#: The first leaf the tree door forwards by name (host_inference_operation).
comptime CLASSICAL_INFER_FIRST = CLASSICAL_OPN_HOST_INFER_PREDICT
#: `classical_operation_leaf` answers for a table name of another door: a
#: `trees_*` / `host_tree_*` name (the tree door) or a `family_*` name.
comptime CLASSICAL_LEAF_TREES_DOOR = -2
comptime CLASSICAL_LEAF_FAMILY_DOOR = -3


def operation_names_classical(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("operation_codes")
    leaves.append(CLASSICAL_OPN_OPERATION_CODES)
    names.append("restore_not_fitted")
    leaves.append(CLASSICAL_OPN_RESTORE_NOT_FITTED)
    names.append("reduce_not_fitted")
    leaves.append(CLASSICAL_OPN_REDUCE_NOT_FITTED)
    names.append("fitted_digest")
    leaves.append(CLASSICAL_OPN_FITTED_DIGEST)
    names.append("archive_save")
    leaves.append(CLASSICAL_OPN_ARCHIVE_SAVE)
    names.append("archive_npz")
    leaves.append(CLASSICAL_OPN_ARCHIVE_NPZ)
    names.append("host_model")
    leaves.append(CLASSICAL_OPN_HOST_MODEL)
    names.append("load")
    leaves.append(CLASSICAL_OPN_LOAD)
    names.append("param_names")
    leaves.append(CLASSICAL_OPN_PARAM_NAMES)
    names.append("getattribute")
    leaves.append(CLASSICAL_OPN_GETATTRIBUTE)
    names.append("ridge_fit")
    leaves.append(CLASSICAL_OPN_RIDGE_FIT)
    names.append("ols_fit")
    leaves.append(CLASSICAL_OPN_OLS_FIT)
    names.append("predict_public")
    leaves.append(CLASSICAL_OPN_PREDICT_PUBLIC)
    names.append("predict_score")
    leaves.append(CLASSICAL_OPN_PREDICT_SCORE)
    names.append("score_public")
    leaves.append(CLASSICAL_OPN_SCORE_PUBLIC)
    names.append("save")
    leaves.append(CLASSICAL_OPN_SAVE)
    names.append("initialize")
    leaves.append(CLASSICAL_OPN_INITIALIZE)
    names.append("validate_params")
    leaves.append(CLASSICAL_OPN_VALIDATE_PARAMS)
    names.append("get_params")
    leaves.append(CLASSICAL_OPN_GET_PARAMS)
    names.append("set_params")
    leaves.append(CLASSICAL_OPN_SET_PARAMS)
    names.append("is_fitted")
    leaves.append(CLASSICAL_OPN_IS_FITTED)
    names.append("model_sha256")
    leaves.append(CLASSICAL_OPN_MODEL_SHA256)
    names.append("predict")
    leaves.append(CLASSICAL_OPN_PREDICT)
    names.append("score")
    leaves.append(CLASSICAL_OPN_SCORE)
    names.append("numeric_mode_used")
    leaves.append(CLASSICAL_OPN_NUMERIC_MODE_USED)
    names.append("vendor_used")
    leaves.append(CLASSICAL_OPN_VENDOR_USED)
    names.append("host_kde_fit")
    leaves.append(CLASSICAL_OPN_HOST_KDE_FIT)
    names.append("host_iforest_fit")
    leaves.append(CLASSICAL_OPN_HOST_IFOREST_FIT)
    names.append("host_density_save")
    leaves.append(CLASSICAL_OPN_HOST_DENSITY_SAVE)
    names.append("host_density_hash")
    leaves.append(CLASSICAL_OPN_HOST_DENSITY_HASH)
    names.append("host_bound_hash")
    leaves.append(CLASSICAL_OPN_HOST_BOUND_HASH)
    names.append("host_density_initialize")
    leaves.append(CLASSICAL_OPN_HOST_DENSITY_INITIALIZE)
    names.append("archive_infer_predict")
    leaves.append(CLASSICAL_OPN_ARCHIVE_INFER_PREDICT)
    names.append("archive_infer_predict_proba")
    leaves.append(CLASSICAL_OPN_ARCHIVE_INFER_PREDICT_PROBA)
    names.append("archive_knn_classifier")
    leaves.append(CLASSICAL_OPN_ARCHIVE_KNN_CLASSIFIER)
    names.append("archive_knn_regressor")
    leaves.append(CLASSICAL_OPN_ARCHIVE_KNN_REGRESSOR)
    names.append("host_archive_predict")
    leaves.append(CLASSICAL_OPN_HOST_ARCHIVE_PREDICT)
    names.append("host_archive_predict_proba")
    leaves.append(CLASSICAL_OPN_HOST_ARCHIVE_PREDICT_PROBA)
    names.append("host_archive_model")
    leaves.append(CLASSICAL_OPN_HOST_ARCHIVE_MODEL)
    names.append("host_archive_load")
    leaves.append(CLASSICAL_OPN_HOST_ARCHIVE_LOAD)
    names.append("host_archive_classical_model")
    leaves.append(CLASSICAL_OPN_HOST_ARCHIVE_CLASSICAL_MODEL)
    names.append("host_infer_predict")
    leaves.append(CLASSICAL_OPN_HOST_INFER_PREDICT)
    names.append("host_infer_predict_proba")
    leaves.append(CLASSICAL_OPN_HOST_INFER_PREDICT_PROBA)
    names.append("host_infer_predict_log_proba")
    leaves.append(CLASSICAL_OPN_HOST_INFER_PREDICT_LOG_PROBA)
    names.append("host_infer_decision_function")
    leaves.append(CLASSICAL_OPN_HOST_INFER_DECISION_FUNCTION)
    names.append("host_infer_transform")
    leaves.append(CLASSICAL_OPN_HOST_INFER_TRANSFORM)
    names.append("host_infer_inverse_transform")
    leaves.append(CLASSICAL_OPN_HOST_INFER_INVERSE_TRANSFORM)
    names.append("host_infer_score")
    leaves.append(CLASSICAL_OPN_HOST_INFER_SCORE)
    names.append("host_infer_score_samples")
    leaves.append(CLASSICAL_OPN_HOST_INFER_SCORE_SAMPLES)
