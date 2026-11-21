# SPDX-License-Identifier: Apache-2.0
# The logistic family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime LOGISTIC_OPN_INITIALIZE = 0
comptime LOGISTIC_OPN_GET_PARAMS = 1
comptime LOGISTIC_OPN_SET_PARAMS = 2
comptime LOGISTIC_OPN_IS_FITTED = 3
comptime LOGISTIC_OPN_FIT = 4
comptime LOGISTIC_OPN_DECISION_FUNCTION = 5
comptime LOGISTIC_OPN_PREDICT = 6
comptime LOGISTIC_OPN_PREDICT_PROBA = 7
comptime LOGISTIC_OPN_PREDICT_LOG_PROBA = 8
comptime LOGISTIC_OPN_SCORE = 9
comptime LOGISTIC_OPN_VENDOR_USED = 10
comptime LOGISTIC_OPN_NUMERIC_MODE_USED = 11


struct LogisticKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_C: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__addr: PythonObject
    var k__l1_ratio: PythonObject
    var k__w: PythonObject
    var k_array_class: PythonObject
    var k_array_fields: PythonObject
    var k_backend_fields: PythonObject
    var k_class_weight: PythonObject
    var k_classes_: PythonObject
    var k_classical_public_layouts: PythonObject
    var k_classifier_labels_class: PythonObject
    var k_coef_: PythonObject
    var k_dtype: PythonObject
    var k_fields_get: PythonObject
    var k_fit_intercept: PythonObject
    var k_input_copied_: PythonObject
    var k_intercept_: PythonObject
    var k_l1_ratio: PythonObject
    var k_linesearch_max_iter: PythonObject
    var k_logistic_fold_native_api_v1: PythonObject
    var k_logistic_native_api_v1: PythonObject
    var k_lr_bindings: PythonObject
    var k_lr_complex_type: PythonObject
    var k_lr_consumed_fields: PythonObject
    var k_lr_constructor_parameters: PythonObject
    var k_lr_host_contexts: PythonObject
    var k_lr_host_model: PythonObject
    var k_lr_host_score_context: PythonObject
    var k_lr_parameters: PythonObject
    var k_lr_sorted_parameters: PythonObject
    var k_lr_x_trees: PythonObject
    var k_mapping_types: PythonObject
    var k_max_iter: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_native_value_types: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_dtypes: PythonObject
    var k_numeric_mode: PythonObject
    var k_objective_: PythonObject
    var k_penalty: PythonObject
    var k_retcode_: PythonObject
    var k_scalar_types: PythonObject
    var k_sequence_array_warnings: PythonObject
    var k_size: PythonObject
    var k_solver: PythonObject
    var k_tol: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_C = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__addr = PythonObject(None)
        self.k__l1_ratio = PythonObject(None)
        self.k__w = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_fields = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_class_weight = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_classical_public_layouts = PythonObject(None)
        self.k_classifier_labels_class = PythonObject(None)
        self.k_coef_ = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_fields_get = PythonObject(None)
        self.k_fit_intercept = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_intercept_ = PythonObject(None)
        self.k_l1_ratio = PythonObject(None)
        self.k_linesearch_max_iter = PythonObject(None)
        self.k_logistic_fold_native_api_v1 = PythonObject(None)
        self.k_logistic_native_api_v1 = PythonObject(None)
        self.k_lr_bindings = PythonObject(None)
        self.k_lr_complex_type = PythonObject(None)
        self.k_lr_consumed_fields = PythonObject(None)
        self.k_lr_constructor_parameters = PythonObject(None)
        self.k_lr_host_contexts = PythonObject(None)
        self.k_lr_host_model = PythonObject(None)
        self.k_lr_host_score_context = PythonObject(None)
        self.k_lr_parameters = PythonObject(None)
        self.k_lr_sorted_parameters = PythonObject(None)
        self.k_lr_x_trees = PythonObject(None)
        self.k_mapping_types = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_native_value_types = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_dtypes = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_objective_ = PythonObject(None)
        self.k_penalty = PythonObject(None)
        self.k_retcode_ = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_sequence_array_warnings = PythonObject(None)
        self.k_size = PythonObject(None)
        self.k_solver = PythonObject(None)
        self.k_tol = PythonObject(None)


comptime _LOGISTIC_KEYS = SharedSlot[LogisticKeys, "MojolearnLogisticKeysV1"]


def _logistic_keys_fill(nk: UnsafePointer[LogisticKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_C = py_key("C")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__addr = py_key("_addr")
    nk[].k__l1_ratio = py_key("_l1_ratio")
    nk[].k__w = py_key("_w")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_fields = py_key("array_fields")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_class_weight = py_key("class_weight")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_classical_public_layouts = py_key("classical_public_layouts")
    nk[].k_classifier_labels_class = py_key("classifier_labels_class")
    nk[].k_coef_ = py_key("coef_")
    nk[].k_dtype = py_key("dtype")
    nk[].k_fields_get = py_key("fields_get")
    nk[].k_fit_intercept = py_key("fit_intercept")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_intercept_ = py_key("intercept_")
    nk[].k_l1_ratio = py_key("l1_ratio")
    nk[].k_linesearch_max_iter = py_key("linesearch_max_iter")
    nk[].k_logistic_fold_native_api_v1 = py_key("logistic_fold_native_api_v1")
    nk[].k_logistic_native_api_v1 = py_key("logistic_native_api_v1")
    nk[].k_lr_bindings = py_key("lr_bindings")
    nk[].k_lr_complex_type = py_key("lr_complex_type")
    nk[].k_lr_consumed_fields = py_key("lr_consumed_fields")
    nk[].k_lr_constructor_parameters = py_key("lr_constructor_parameters")
    nk[].k_lr_host_contexts = py_key("lr_host_contexts")
    nk[].k_lr_host_model = py_key("lr_host_model")
    nk[].k_lr_host_score_context = py_key("lr_host_score_context")
    nk[].k_lr_parameters = py_key("lr_parameters")
    nk[].k_lr_sorted_parameters = py_key("lr_sorted_parameters")
    nk[].k_lr_x_trees = py_key("lr_x_trees")
    nk[].k_mapping_types = py_key("mapping_types")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_native_value_types = py_key("native_value_types")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_dtypes = py_key("numeric_dtypes")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_objective_ = py_key("objective_")
    nk[].k_penalty = py_key("penalty")
    nk[].k_retcode_ = py_key("retcode_")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_sequence_array_warnings = py_key("sequence_array_warnings")
    nk[].k_size = py_key("size")
    nk[].k_solver = py_key("solver")
    nk[].k_tol = py_key("tol")
    var ops = Python.dict()
    py_op_add(ops, "initialize", LOGISTIC_OPN_INITIALIZE)
    py_op_add(ops, "get_params", LOGISTIC_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", LOGISTIC_OPN_SET_PARAMS)
    py_op_add(ops, "is_fitted", LOGISTIC_OPN_IS_FITTED)
    py_op_add(ops, "fit", LOGISTIC_OPN_FIT)
    py_op_add(ops, "decision_function", LOGISTIC_OPN_DECISION_FUNCTION)
    py_op_add(ops, "predict", LOGISTIC_OPN_PREDICT)
    py_op_add(ops, "predict_proba", LOGISTIC_OPN_PREDICT_PROBA)
    py_op_add(ops, "predict_log_proba", LOGISTIC_OPN_PREDICT_LOG_PROBA)
    py_op_add(ops, "score", LOGISTIC_OPN_SCORE)
    py_op_add(ops, "vendor_used", LOGISTIC_OPN_VENDOR_USED)
    py_op_add(ops, "numeric_mode_used", LOGISTIC_OPN_NUMERIC_MODE_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def logistic_keys() raises -> UnsafePointer[LogisticKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _LOGISTIC_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _logistic_keys_fill(nk)
    return nk
