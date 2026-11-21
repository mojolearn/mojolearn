# SPDX-License-Identifier: Apache-2.0
# The prep family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime PREP_OPN_INITIALIZE = 0
comptime PREP_OPN_IS_FITTED = 1
comptime PREP_OPN_GET_PARAMS = 2
comptime PREP_OPN_SET_PARAMS = 3
comptime PREP_OPN_FIT = 4
comptime PREP_OPN_FIT_TRANSFORM = 5
comptime PREP_OPN_SCORE = 6
comptime PREP_OPN_CV_SET_PARAMS = 7
comptime PREP_OPN_PREDICT = 8
comptime PREP_OPN_PREDICT_PROBA = 9
comptime PREP_OPN_PREDICT_LOG_PROBA = 10
comptime PREP_OPN_PREDICT_JOINT_LOG_PROBA = 11
comptime PREP_OPN_DECISION_FUNCTION = 12
comptime PREP_OPN_PARTIAL_FIT = 13
comptime PREP_OPN_TRANSFORM = 14
comptime PREP_OPN_NB_CSR_READY = 15
comptime PREP_OPN_INVERSE_TRANSFORM = 16
comptime PREP_OPN_GET_SUPPORT = 17
comptime PREP_OPN_MUTUAL_INFO_CLASSIF = 18
comptime PREP_OPN_MUTUAL_INFO_REGRESSION = 19
comptime PREP_OPN_F_CLASSIF = 20
comptime PREP_OPN_CHI2 = 21
comptime PREP_OPN_F_REGRESSION = 22
comptime PREP_OPN_R_REGRESSION = 23


struct PrepKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_SimpleImputer: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__ROWS_CONTEXT: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k___str__: PythonObject
    var k__bounds_k: PythonObject
    var k__categories: PythonObject
    var k__cats: PythonObject
    var k__classes: PythonObject
    var k__cmax: PythonObject
    var k__data: PythonObject
    var k__estimator_admitted_: PythonObject
    var k__given: PythonObject
    var k__indicator: PythonObject
    var k__keep: PythonObject
    var k__mask: PythonObject
    var k__max_components: PythonObject
    var k__mgr: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__mojolearn_params_reset: PythonObject
    var k__post_step: PythonObject
    var k__posterior: PythonObject
    var k__range: PythonObject
    var k__rank: PythonObject
    var k__rng: PythonObject
    var k__shape: PythonObject
    var k__support_k_: PythonObject
    var k__support_mask_: PythonObject
    var k__support_words_: PythonObject
    var k_add_indicator: PythonObject
    var k_allow_nan: PythonObject
    var k_alpha: PythonObject
    var k_arena_io_fields: PythonObject
    var k_args: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_backend_fields: PythonObject
    var k_base: PythonObject
    var k_binarize: PythonObject
    var k_blocks: PythonObject
    var k_bool_types: PythonObject
    var k_class_log_prior_: PythonObject
    var k_class_prior: PythonObject
    var k_classes: PythonObject
    var k_classes_: PythonObject
    var k_classifier_labels_class: PythonObject
    var k_classifier_labels_namespace: PythonObject
    var k_cls_bindings: PythonObject
    var k_cls_kinds: PythonObject
    var k_copy: PythonObject
    var k_covariance_estimator: PythonObject
    var k_cv_fixed_owner: PythonObject
    var k_dtype: PythonObject
    var k_estimator: PythonObject
    var k_estimator_: PythonObject
    var k_family_msel_estimators: PythonObject
    var k_feature_log_prob_: PythonObject
    var k_feature_names_in_: PythonObject
    var k_fields_get: PythonObject
    var k_fill_value: PythonObject
    var k_fit_prior: PythonObject
    var k_importance_getter: PythonObject
    var k_imputation_order: PythonObject
    var k_imputation_sequence_: PythonObject
    var k_indices: PythonObject
    var k_initial_imputer_: PythonObject
    var k_initial_strategy: PythonObject
    var k_integral_type: PythonObject
    var k_keep_empty_features: PythonObject
    var k_device: PythonObject
    var k_labels_integral_type: PythonObject
    var k_lda_admission_context: PythonObject
    var k_lda_fields_get: PythonObject
    var k_lda_fitted_fields: PythonObject
    var k_lda_native_class: PythonObject
    var k_lda_parameters: PythonObject
    var k_max_iter: PythonObject
    var k_max_value: PythonObject
    var k_maxabs_rows_class: PythonObject
    var k_methods: PythonObject
    var k_min_categories: PythonObject
    var k_min_value: PythonObject
    var k_missing_values: PythonObject
    var k_modules: PythonObject
    var k_multi_output: PythonObject
    var k_n_categories_: PythonObject
    var k_n_components: PythonObject
    var k_n_features_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_features_to_select: PythonObject
    var k_n_features_with_missing_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_nearest_features: PythonObject
    var k_native_estimator_context: PythonObject
    var k_native_estimator_entry: PythonObject
    var k_ndarray: PythonObject
    var k_neg_label: PythonObject
    var k_norm: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_one_d_array: PythonObject
    var k_output_context: PythonObject
    var k_parameters: PythonObject
    var k_pos_label: PythonObject
    var k_prep_composition_plans: PythonObject
    var k_prep_owner_classes: PythonObject
    var k_prep_owner_kinds: PythonObject
    var k_priors: PythonObject
    var k_random_state: PythonObject
    var k_ranking_: PythonObject
    var k_reg_param: PythonObject
    var k_required: PythonObject
    var k_resident_class: PythonObject
    var k_resident_min_words: PythonObject
    var k_sample_posterior: PythonObject
    var k_scalar_types: PythonObject
    var k_select_context: PythonObject
    var k_select_mi_work: PythonObject
    var k_shape: PythonObject
    var k_shrinkage: PythonObject
    var k_skip_complete: PythonObject
    var k_solver: PythonObject
    var k_sparse: PythonObject
    var k_sparse_output: PythonObject
    var k_step: PythonObject
    var k_store_covariance: PythonObject
    var k_strategy: PythonObject
    var k_support_: PythonObject
    var k_tol: PythonObject
    var k_two_d_array: PythonObject
    var k_var_smoothing: PythonObject
    var k_y_type_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_SimpleImputer = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__ROWS_CONTEXT = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k___str__ = PythonObject(None)
        self.k__bounds_k = PythonObject(None)
        self.k__categories = PythonObject(None)
        self.k__cats = PythonObject(None)
        self.k__classes = PythonObject(None)
        self.k__cmax = PythonObject(None)
        self.k__data = PythonObject(None)
        self.k__estimator_admitted_ = PythonObject(None)
        self.k__given = PythonObject(None)
        self.k__indicator = PythonObject(None)
        self.k__keep = PythonObject(None)
        self.k__mask = PythonObject(None)
        self.k__max_components = PythonObject(None)
        self.k__mgr = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__mojolearn_params_reset = PythonObject(None)
        self.k__post_step = PythonObject(None)
        self.k__posterior = PythonObject(None)
        self.k__range = PythonObject(None)
        self.k__rank = PythonObject(None)
        self.k__rng = PythonObject(None)
        self.k__shape = PythonObject(None)
        self.k__support_k_ = PythonObject(None)
        self.k__support_mask_ = PythonObject(None)
        self.k__support_words_ = PythonObject(None)
        self.k_add_indicator = PythonObject(None)
        self.k_allow_nan = PythonObject(None)
        self.k_alpha = PythonObject(None)
        self.k_arena_io_fields = PythonObject(None)
        self.k_args = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_base = PythonObject(None)
        self.k_binarize = PythonObject(None)
        self.k_blocks = PythonObject(None)
        self.k_bool_types = PythonObject(None)
        self.k_class_log_prior_ = PythonObject(None)
        self.k_class_prior = PythonObject(None)
        self.k_classes = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_classifier_labels_class = PythonObject(None)
        self.k_classifier_labels_namespace = PythonObject(None)
        self.k_cls_bindings = PythonObject(None)
        self.k_cls_kinds = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_covariance_estimator = PythonObject(None)
        self.k_cv_fixed_owner = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_estimator = PythonObject(None)
        self.k_estimator_ = PythonObject(None)
        self.k_family_msel_estimators = PythonObject(None)
        self.k_feature_log_prob_ = PythonObject(None)
        self.k_feature_names_in_ = PythonObject(None)
        self.k_fields_get = PythonObject(None)
        self.k_fill_value = PythonObject(None)
        self.k_fit_prior = PythonObject(None)
        self.k_importance_getter = PythonObject(None)
        self.k_imputation_order = PythonObject(None)
        self.k_imputation_sequence_ = PythonObject(None)
        self.k_indices = PythonObject(None)
        self.k_initial_imputer_ = PythonObject(None)
        self.k_initial_strategy = PythonObject(None)
        self.k_integral_type = PythonObject(None)
        self.k_keep_empty_features = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_labels_integral_type = PythonObject(None)
        self.k_lda_admission_context = PythonObject(None)
        self.k_lda_fields_get = PythonObject(None)
        self.k_lda_fitted_fields = PythonObject(None)
        self.k_lda_native_class = PythonObject(None)
        self.k_lda_parameters = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_max_value = PythonObject(None)
        self.k_maxabs_rows_class = PythonObject(None)
        self.k_methods = PythonObject(None)
        self.k_min_categories = PythonObject(None)
        self.k_min_value = PythonObject(None)
        self.k_missing_values = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_multi_output = PythonObject(None)
        self.k_n_categories_ = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_features_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_features_to_select = PythonObject(None)
        self.k_n_features_with_missing_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_nearest_features = PythonObject(None)
        self.k_native_estimator_context = PythonObject(None)
        self.k_native_estimator_entry = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_neg_label = PythonObject(None)
        self.k_norm = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_one_d_array = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_parameters = PythonObject(None)
        self.k_pos_label = PythonObject(None)
        self.k_prep_composition_plans = PythonObject(None)
        self.k_prep_owner_classes = PythonObject(None)
        self.k_prep_owner_kinds = PythonObject(None)
        self.k_priors = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_ranking_ = PythonObject(None)
        self.k_reg_param = PythonObject(None)
        self.k_required = PythonObject(None)
        self.k_resident_class = PythonObject(None)
        self.k_resident_min_words = PythonObject(None)
        self.k_sample_posterior = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_select_context = PythonObject(None)
        self.k_select_mi_work = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_shrinkage = PythonObject(None)
        self.k_skip_complete = PythonObject(None)
        self.k_solver = PythonObject(None)
        self.k_sparse = PythonObject(None)
        self.k_sparse_output = PythonObject(None)
        self.k_step = PythonObject(None)
        self.k_store_covariance = PythonObject(None)
        self.k_strategy = PythonObject(None)
        self.k_support_ = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_two_d_array = PythonObject(None)
        self.k_var_smoothing = PythonObject(None)
        self.k_y_type_ = PythonObject(None)


comptime _PREP_KEYS = SharedSlot[PrepKeys, "MojolearnPrepKeysV1"]


def _prep_keys_fill(nk: UnsafePointer[PrepKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_SimpleImputer = py_key("SimpleImputer")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__ROWS_CONTEXT = py_key("_ROWS_CONTEXT")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k___str__ = py_key("__str__")
    nk[].k__bounds_k = py_key("_bounds_k")
    nk[].k__categories = py_key("_categories")
    nk[].k__cats = py_key("_cats")
    nk[].k__classes = py_key("_classes")
    nk[].k__cmax = py_key("_cmax")
    nk[].k__data = py_key("_data")
    nk[].k__estimator_admitted_ = py_key("_estimator_admitted_")
    nk[].k__given = py_key("_given")
    nk[].k__indicator = py_key("_indicator")
    nk[].k__keep = py_key("_keep")
    nk[].k__mask = py_key("_mask")
    nk[].k__max_components = py_key("_max_components")
    nk[].k__mgr = py_key("_mgr")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__mojolearn_params_reset = py_key("_mojolearn_params_reset")
    nk[].k__post_step = py_key("_post_step")
    nk[].k__posterior = py_key("_posterior")
    nk[].k__range = py_key("_range")
    nk[].k__rank = py_key("_rank")
    nk[].k__rng = py_key("_rng")
    nk[].k__shape = py_key("_shape")
    nk[].k__support_k_ = py_key("_support_k_")
    nk[].k__support_mask_ = py_key("_support_mask_")
    nk[].k__support_words_ = py_key("_support_words_")
    nk[].k_add_indicator = py_key("add_indicator")
    nk[].k_allow_nan = py_key("allow_nan")
    nk[].k_alpha = py_key("alpha")
    nk[].k_arena_io_fields = py_key("arena_io_fields")
    nk[].k_args = py_key("args")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_base = py_key("base")
    nk[].k_binarize = py_key("binarize")
    nk[].k_blocks = py_key("blocks")
    nk[].k_bool_types = py_key("bool_types")
    nk[].k_class_log_prior_ = py_key("class_log_prior_")
    nk[].k_class_prior = py_key("class_prior")
    nk[].k_classes = py_key("classes")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_classifier_labels_class = py_key("classifier_labels_class")
    nk[].k_classifier_labels_namespace = py_key("classifier_labels_namespace")
    nk[].k_cls_bindings = py_key("cls_bindings")
    nk[].k_cls_kinds = py_key("cls_kinds")
    nk[].k_copy = py_key("copy")
    nk[].k_covariance_estimator = py_key("covariance_estimator")
    nk[].k_cv_fixed_owner = py_key("cv_fixed_owner")
    nk[].k_dtype = py_key("dtype")
    nk[].k_estimator = py_key("estimator")
    nk[].k_estimator_ = py_key("estimator_")
    nk[].k_family_msel_estimators = py_key("family_msel_estimators")
    nk[].k_feature_log_prob_ = py_key("feature_log_prob_")
    nk[].k_feature_names_in_ = py_key("feature_names_in_")
    nk[].k_fields_get = py_key("fields_get")
    nk[].k_fill_value = py_key("fill_value")
    nk[].k_fit_prior = py_key("fit_prior")
    nk[].k_importance_getter = py_key("importance_getter")
    nk[].k_imputation_order = py_key("imputation_order")
    nk[].k_imputation_sequence_ = py_key("imputation_sequence_")
    nk[].k_indices = py_key("indices")
    nk[].k_initial_imputer_ = py_key("initial_imputer_")
    nk[].k_initial_strategy = py_key("initial_strategy")
    nk[].k_integral_type = py_key("integral_type")
    nk[].k_keep_empty_features = py_key("keep_empty_features")
    nk[].k_device = py_key("device")
    nk[].k_labels_integral_type = py_key("labels_integral_type")
    nk[].k_lda_admission_context = py_key("lda_admission_context")
    nk[].k_lda_fields_get = py_key("lda_fields_get")
    nk[].k_lda_fitted_fields = py_key("lda_fitted_fields")
    nk[].k_lda_native_class = py_key("lda_native_class")
    nk[].k_lda_parameters = py_key("lda_parameters")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_max_value = py_key("max_value")
    nk[].k_maxabs_rows_class = py_key("maxabs_rows_class")
    nk[].k_methods = py_key("methods")
    nk[].k_min_categories = py_key("min_categories")
    nk[].k_min_value = py_key("min_value")
    nk[].k_missing_values = py_key("missing_values")
    nk[].k_modules = py_key("modules")
    nk[].k_multi_output = py_key("multi_output")
    nk[].k_n_categories_ = py_key("n_categories_")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_features_ = py_key("n_features_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_features_to_select = py_key("n_features_to_select")
    nk[].k_n_features_with_missing_ = py_key("n_features_with_missing_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_nearest_features = py_key("n_nearest_features")
    nk[].k_native_estimator_context = py_key("native_estimator_context")
    nk[].k_native_estimator_entry = py_key("native_estimator_entry")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_neg_label = py_key("neg_label")
    nk[].k_norm = py_key("norm")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_one_d_array = py_key("one_d_array")
    nk[].k_output_context = py_key("output_context")
    nk[].k_parameters = py_key("parameters")
    nk[].k_pos_label = py_key("pos_label")
    nk[].k_prep_composition_plans = py_key("prep_composition_plans")
    nk[].k_prep_owner_classes = py_key("prep_owner_classes")
    nk[].k_prep_owner_kinds = py_key("prep_owner_kinds")
    nk[].k_priors = py_key("priors")
    nk[].k_random_state = py_key("random_state")
    nk[].k_ranking_ = py_key("ranking_")
    nk[].k_reg_param = py_key("reg_param")
    nk[].k_required = py_key("required")
    nk[].k_resident_class = py_key("resident_class")
    nk[].k_resident_min_words = py_key("resident_min_words")
    nk[].k_sample_posterior = py_key("sample_posterior")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_select_context = py_key("select_context")
    nk[].k_select_mi_work = py_key("select_mi_work")
    nk[].k_shape = py_key("shape")
    nk[].k_shrinkage = py_key("shrinkage")
    nk[].k_skip_complete = py_key("skip_complete")
    nk[].k_solver = py_key("solver")
    nk[].k_sparse = py_key("sparse")
    nk[].k_sparse_output = py_key("sparse_output")
    nk[].k_step = py_key("step")
    nk[].k_store_covariance = py_key("store_covariance")
    nk[].k_strategy = py_key("strategy")
    nk[].k_support_ = py_key("support_")
    nk[].k_tol = py_key("tol")
    nk[].k_two_d_array = py_key("two_d_array")
    nk[].k_var_smoothing = py_key("var_smoothing")
    nk[].k_y_type_ = py_key("y_type_")
    var ops = Python.dict()
    py_op_add(ops, "initialize", PREP_OPN_INITIALIZE)
    py_op_add(ops, "is_fitted", PREP_OPN_IS_FITTED)
    py_op_add(ops, "get_params", PREP_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", PREP_OPN_SET_PARAMS)
    py_op_add(ops, "fit", PREP_OPN_FIT)
    py_op_add(ops, "fit_transform", PREP_OPN_FIT_TRANSFORM)
    py_op_add(ops, "score", PREP_OPN_SCORE)
    py_op_add(ops, "cv_set_params", PREP_OPN_CV_SET_PARAMS)
    py_op_add(ops, "predict", PREP_OPN_PREDICT)
    py_op_add(ops, "predict_proba", PREP_OPN_PREDICT_PROBA)
    py_op_add(ops, "predict_log_proba", PREP_OPN_PREDICT_LOG_PROBA)
    py_op_add(ops, "predict_joint_log_proba", PREP_OPN_PREDICT_JOINT_LOG_PROBA)
    py_op_add(ops, "decision_function", PREP_OPN_DECISION_FUNCTION)
    py_op_add(ops, "partial_fit", PREP_OPN_PARTIAL_FIT)
    py_op_add(ops, "transform", PREP_OPN_TRANSFORM)
    py_op_add(ops, "nb_csr_ready", PREP_OPN_NB_CSR_READY)
    py_op_add(ops, "inverse_transform", PREP_OPN_INVERSE_TRANSFORM)
    py_op_add(ops, "get_support", PREP_OPN_GET_SUPPORT)
    py_op_add(ops, "mutual_info_classif", PREP_OPN_MUTUAL_INFO_CLASSIF)
    py_op_add(ops, "mutual_info_regression", PREP_OPN_MUTUAL_INFO_REGRESSION)
    py_op_add(ops, "f_classif", PREP_OPN_F_CLASSIF)
    py_op_add(ops, "chi2", PREP_OPN_CHI2)
    py_op_add(ops, "f_regression", PREP_OPN_F_REGRESSION)
    py_op_add(ops, "r_regression", PREP_OPN_R_REGRESSION)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def prep_keys() raises -> UnsafePointer[PrepKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PREP_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _prep_keys_fill(nk)
    return nk
