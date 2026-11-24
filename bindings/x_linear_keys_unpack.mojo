# SPDX-License-Identifier: Apache-2.0
# The x_linear family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime X_LINEAR_OPN_REGRESSOR_PREDICT = 0
comptime X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE = 1
comptime X_LINEAR_OPN_DECISION_FUNCTION = 2
comptime X_LINEAR_OPN_CLASSIFIER_PREDICT = 3
comptime X_LINEAR_OPN_CLASSIFIER_SCORE = 4
comptime X_LINEAR_OPN_SGD_PREDICT_PROBA = 5
comptime X_LINEAR_OPN_LRCV_PREDICT_PROBA = 6
comptime X_LINEAR_OPN_FIT_SGD_CLASSIFIER = 7
comptime X_LINEAR_OPN_FIT_PERCEPTRON = 8
comptime X_LINEAR_OPN_FIT_PA_CLASSIFIER = 9
comptime X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER = 10
comptime X_LINEAR_OPN_FIT_SGD_REGRESSOR = 11
comptime X_LINEAR_OPN_FIT_PA_REGRESSOR = 12
comptime X_LINEAR_OPN_FIT_ONE_CLASS = 13
comptime X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION = 14
comptime X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES = 15
comptime X_LINEAR_OPN_ONE_CLASS_PREDICT = 16
comptime X_LINEAR_OPN_FIT_GLM = 17
comptime X_LINEAR_OPN_FIT_HUBER = 18
comptime X_LINEAR_OPN_FIT_BAYES = 19
comptime X_LINEAR_OPN_FIT_ARD = 20
comptime X_LINEAR_OPN_FIT_LARS = 21
comptime X_LINEAR_OPN_FIT_LASSO_LARS = 22
comptime X_LINEAR_OPN_FIT_QUANTILE = 23
comptime X_LINEAR_OPN_ISOTONIC_FIT = 24
comptime X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM = 25
comptime X_LINEAR_OPN_ISOTONIC_TRANSFORM = 26
comptime X_LINEAR_OPN_ISOTONIC_PREDICT = 27
comptime X_LINEAR_OPN_ISOTONIC_SCORE = 28
comptime X_LINEAR_OPN_FIT_RIDGE_CV = 29
comptime X_LINEAR_OPN_FIT_LASSO_CV = 30
comptime X_LINEAR_OPN_FIT_ELASTIC_NET_CV = 31
comptime X_LINEAR_OPN_FIT_LOGISTIC_CV = 32
comptime X_LINEAR_OPN_CD_FIT = 33
comptime X_LINEAR_OPN_CD_PREDICT = 34
comptime X_LINEAR_OPN_CD_SCORE = 35
# Lane dw-xlinear (2026-11-20). "regressor_score": the regressors' score in
# this door, on the estimator's route (it was a second door that selected the
# process images by name). "device_admit": a `device` checked where it is
# given (a constructor, `set_params`) and the estimator's route dropped.
comptime X_LINEAR_OPN_REGRESSOR_SCORE = 36
comptime X_LINEAR_OPN_DEVICE_ADMIT = 37
# "cd_vendor_used": `vendor_used()` of Lasso / ElasticNet, the vendor word of
# the solver image their route holds (metadata, no input, no lifecycle step).
comptime X_LINEAR_OPN_CD_VENDOR_USED = 38
# BayesianRidge.sigma_'s first read (lane mf-iterimp2): formed natively from the fitted factor.
comptime X_LINEAR_OPN_BAYES_SIGMA = 39


struct XLinearKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_C: PythonObject
    var k_C_: PythonObject
    var k_Cs: PythonObject
    var k_Cs_: PythonObject
    var k_X_max_: PythonObject
    var k_X_min_: PythonObject
    var k_X_thresholds_: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__HOST_INFERENCE_ONLY: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k__link: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k_active_: PythonObject
    var k_alpha: PythonObject
    var k_alpha_: PythonObject
    var k_alpha_1: PythonObject
    var k_alpha_2: PythonObject
    var k_alpha_init: PythonObject
    var k_alpha_per_target: PythonObject
    var k_alphas: PythonObject
    var k_alphas_: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_average: PythonObject
    var k_backend_fields: PythonObject
    var k_batch_size: PythonObject
    var k_best_score_: PythonObject
    var k_cd_host_predict: PythonObject
    var k_cd_selection: PythonObject
    var k_cd_solver: PythonObject
    var k_cd_solver_modes: PythonObject
    var k_class_weight: PythonObject
    var k_classes_: PythonObject
    var k_coef_: PythonObject
    var k_compute_score: PythonObject
    var k_copy_X: PythonObject
    var k_cv: PythonObject
    var k_cv_results_: PythonObject
    var k_dual: PythonObject
    var k_early_stopping: PythonObject
    var k_eps: PythonObject
    var k_epsilon: PythonObject
    var k_epsilon_insensitive: PythonObject
    var k_eta0: PythonObject
    var k_family_msel_splitter_layouts: PythonObject
    var k_fit_intercept: PythonObject
    var k_fit_loop_: PythonObject
    var k_fortran_copied_: PythonObject
    var k_hinge: PythonObject
    var k_increasing: PythonObject
    var k_increasing_: PythonObject
    var k_input_copied_: PythonObject
    var k_intercept_: PythonObject
    var k_l1_ratio: PythonObject
    var k_l1_ratio_: PythonObject
    var k_l1_ratios: PythonObject
    var k_lambda_: PythonObject
    var k_lambda_1: PythonObject
    var k_lambda_2: PythonObject
    var k_lambda_init: PythonObject
    var k_learning_rate: PythonObject
    var k_link: PythonObject
    var k_loss: PythonObject
    var k_max_iter: PythonObject
    var k_modules: PythonObject
    var k_mse_path_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_iter_no_change: PythonObject
    var k_n_nonzero_coefs: PythonObject
    var k_ndarray: PythonObject
    var k_nu: PythonObject
    var k_numeric_mode: PythonObject
    var k_offset_: PythonObject
    var k_out_of_bounds: PythonObject
    var k_outliers_: PythonObject
    var k_penalty: PythonObject
    var k_perceptron: PythonObject
    var k_positive: PythonObject
    var k_power: PythonObject
    var k_power_t: PythonObject
    var k_precompute: PythonObject
    var k_quantile: PythonObject
    var k_random_state: PythonObject
    var k_refit: PythonObject
    var k_scale_: PythonObject
    var k_scores_: PythonObject
    var k_scoring: PythonObject
    var k_selection: PythonObject
    var k_shuffle: PythonObject
    var k_solver: PythonObject
    var k_store_cv_results: PythonObject
    var k_t_: PythonObject
    var k_threshold_lambda: PythonObject
    var k_sigma_: PythonObject
    var k_X_offset_: PythonObject
    var k_X_scale_: PythonObject
    var k__bayes_v: PythonObject
    var k__bayes_ev: PythonObject
    var k__bayes_sigma: PythonObject
    var k_tol: PythonObject
    var k_warm_start: PythonObject
    var k_xl_bindings: PythonObject
    var k_xl_decimal: PythonObject
    var k_xl_helpers: PythonObject
    var k_xl_pow_context: PythonObject
    var k_xl_sgd_clf_loss: PythonObject
    var k_xl_sgd_lr: PythonObject
    var k_xl_sgd_penalty: PythonObject
    var k_xl_sgd_reg_loss: PythonObject
    var k_xl_sgd_sum_losses: PythonObject
    var k_xl_undefined_warning: PythonObject
    var k_y_max: PythonObject
    var k_y_min: PythonObject
    var k_y_thresholds_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_C = PythonObject(None)
        self.k_C_ = PythonObject(None)
        self.k_Cs = PythonObject(None)
        self.k_Cs_ = PythonObject(None)
        self.k_X_max_ = PythonObject(None)
        self.k_X_min_ = PythonObject(None)
        self.k_X_thresholds_ = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__HOST_INFERENCE_ONLY = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k__link = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k_active_ = PythonObject(None)
        self.k_alpha = PythonObject(None)
        self.k_alpha_ = PythonObject(None)
        self.k_alpha_1 = PythonObject(None)
        self.k_alpha_2 = PythonObject(None)
        self.k_alpha_init = PythonObject(None)
        self.k_alpha_per_target = PythonObject(None)
        self.k_alphas = PythonObject(None)
        self.k_alphas_ = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_average = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_best_score_ = PythonObject(None)
        self.k_cd_host_predict = PythonObject(None)
        self.k_cd_selection = PythonObject(None)
        self.k_cd_solver = PythonObject(None)
        self.k_cd_solver_modes = PythonObject(None)
        self.k_class_weight = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_coef_ = PythonObject(None)
        self.k_compute_score = PythonObject(None)
        self.k_copy_X = PythonObject(None)
        self.k_cv = PythonObject(None)
        self.k_cv_results_ = PythonObject(None)
        self.k_dual = PythonObject(None)
        self.k_early_stopping = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_epsilon = PythonObject(None)
        self.k_epsilon_insensitive = PythonObject(None)
        self.k_eta0 = PythonObject(None)
        self.k_family_msel_splitter_layouts = PythonObject(None)
        self.k_fit_intercept = PythonObject(None)
        self.k_fit_loop_ = PythonObject(None)
        self.k_fortran_copied_ = PythonObject(None)
        self.k_hinge = PythonObject(None)
        self.k_increasing = PythonObject(None)
        self.k_increasing_ = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_intercept_ = PythonObject(None)
        self.k_l1_ratio = PythonObject(None)
        self.k_l1_ratio_ = PythonObject(None)
        self.k_l1_ratios = PythonObject(None)
        self.k_lambda_ = PythonObject(None)
        self.k_lambda_1 = PythonObject(None)
        self.k_lambda_2 = PythonObject(None)
        self.k_lambda_init = PythonObject(None)
        self.k_learning_rate = PythonObject(None)
        self.k_link = PythonObject(None)
        self.k_loss = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_mse_path_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_iter_no_change = PythonObject(None)
        self.k_n_nonzero_coefs = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_nu = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_offset_ = PythonObject(None)
        self.k_out_of_bounds = PythonObject(None)
        self.k_outliers_ = PythonObject(None)
        self.k_penalty = PythonObject(None)
        self.k_perceptron = PythonObject(None)
        self.k_positive = PythonObject(None)
        self.k_power = PythonObject(None)
        self.k_power_t = PythonObject(None)
        self.k_precompute = PythonObject(None)
        self.k_quantile = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_refit = PythonObject(None)
        self.k_scale_ = PythonObject(None)
        self.k_scores_ = PythonObject(None)
        self.k_scoring = PythonObject(None)
        self.k_selection = PythonObject(None)
        self.k_shuffle = PythonObject(None)
        self.k_solver = PythonObject(None)
        self.k_store_cv_results = PythonObject(None)
        self.k_t_ = PythonObject(None)
        self.k_threshold_lambda = PythonObject(None)
        self.k_sigma_ = PythonObject(None)
        self.k_X_offset_ = PythonObject(None)
        self.k_X_scale_ = PythonObject(None)
        self.k__bayes_v = PythonObject(None)
        self.k__bayes_ev = PythonObject(None)
        self.k__bayes_sigma = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_warm_start = PythonObject(None)
        self.k_xl_bindings = PythonObject(None)
        self.k_xl_decimal = PythonObject(None)
        self.k_xl_helpers = PythonObject(None)
        self.k_xl_pow_context = PythonObject(None)
        self.k_xl_sgd_clf_loss = PythonObject(None)
        self.k_xl_sgd_lr = PythonObject(None)
        self.k_xl_sgd_penalty = PythonObject(None)
        self.k_xl_sgd_reg_loss = PythonObject(None)
        self.k_xl_sgd_sum_losses = PythonObject(None)
        self.k_xl_undefined_warning = PythonObject(None)
        self.k_y_max = PythonObject(None)
        self.k_y_min = PythonObject(None)
        self.k_y_thresholds_ = PythonObject(None)


comptime _X_LINEAR_KEYS = SharedSlot[XLinearKeys, "MojolearnXLinearKeysV1"]


def _x_linear_keys_fill(nk: UnsafePointer[XLinearKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_C = py_key("C")
    nk[].k_C_ = py_key("C_")
    nk[].k_Cs = py_key("Cs")
    nk[].k_Cs_ = py_key("Cs_")
    nk[].k_X_max_ = py_key("X_max_")
    nk[].k_X_min_ = py_key("X_min_")
    nk[].k_X_thresholds_ = py_key("X_thresholds_")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__HOST_INFERENCE_ONLY = py_key("_HOST_INFERENCE_ONLY")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k__link = py_key("_link")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k_active_ = py_key("active_")
    nk[].k_alpha = py_key("alpha")
    nk[].k_alpha_ = py_key("alpha_")
    nk[].k_alpha_1 = py_key("alpha_1")
    nk[].k_alpha_2 = py_key("alpha_2")
    nk[].k_alpha_init = py_key("alpha_init")
    nk[].k_alpha_per_target = py_key("alpha_per_target")
    nk[].k_alphas = py_key("alphas")
    nk[].k_alphas_ = py_key("alphas_")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_average = py_key("average")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_best_score_ = py_key("best_score_")
    nk[].k_cd_host_predict = py_key("cd_host_predict")
    nk[].k_cd_selection = py_key("cd_selection")
    nk[].k_cd_solver = py_key("cd_solver")
    nk[].k_cd_solver_modes = py_key("cd_solver_modes")
    nk[].k_class_weight = py_key("class_weight")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_coef_ = py_key("coef_")
    nk[].k_compute_score = py_key("compute_score")
    nk[].k_copy_X = py_key("copy_X")
    nk[].k_cv = py_key("cv")
    nk[].k_cv_results_ = py_key("cv_results_")
    nk[].k_dual = py_key("dual")
    nk[].k_early_stopping = py_key("early_stopping")
    nk[].k_eps = py_key("eps")
    nk[].k_epsilon = py_key("epsilon")
    nk[].k_epsilon_insensitive = py_key("epsilon_insensitive")
    nk[].k_eta0 = py_key("eta0")
    nk[].k_family_msel_splitter_layouts = py_key("family_msel_splitter_layouts")
    nk[].k_fit_intercept = py_key("fit_intercept")
    nk[].k_fit_loop_ = py_key("fit_loop_")
    nk[].k_fortran_copied_ = py_key("fortran_copied_")
    nk[].k_hinge = py_key("hinge")
    nk[].k_increasing = py_key("increasing")
    nk[].k_increasing_ = py_key("increasing_")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_intercept_ = py_key("intercept_")
    nk[].k_l1_ratio = py_key("l1_ratio")
    nk[].k_l1_ratio_ = py_key("l1_ratio_")
    nk[].k_l1_ratios = py_key("l1_ratios")
    nk[].k_lambda_ = py_key("lambda_")
    nk[].k_lambda_1 = py_key("lambda_1")
    nk[].k_lambda_2 = py_key("lambda_2")
    nk[].k_lambda_init = py_key("lambda_init")
    nk[].k_learning_rate = py_key("learning_rate")
    nk[].k_link = py_key("link")
    nk[].k_loss = py_key("loss")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_modules = py_key("modules")
    nk[].k_mse_path_ = py_key("mse_path_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_iter_no_change = py_key("n_iter_no_change")
    nk[].k_n_nonzero_coefs = py_key("n_nonzero_coefs")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_nu = py_key("nu")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_offset_ = py_key("offset_")
    nk[].k_out_of_bounds = py_key("out_of_bounds")
    nk[].k_outliers_ = py_key("outliers_")
    nk[].k_penalty = py_key("penalty")
    nk[].k_perceptron = py_key("perceptron")
    nk[].k_positive = py_key("positive")
    nk[].k_power = py_key("power")
    nk[].k_power_t = py_key("power_t")
    nk[].k_precompute = py_key("precompute")
    nk[].k_quantile = py_key("quantile")
    nk[].k_random_state = py_key("random_state")
    nk[].k_refit = py_key("refit")
    nk[].k_scale_ = py_key("scale_")
    nk[].k_scores_ = py_key("scores_")
    nk[].k_scoring = py_key("scoring")
    nk[].k_selection = py_key("selection")
    nk[].k_shuffle = py_key("shuffle")
    nk[].k_solver = py_key("solver")
    nk[].k_store_cv_results = py_key("store_cv_results")
    nk[].k_t_ = py_key("t_")
    nk[].k_threshold_lambda = py_key("threshold_lambda")
    nk[].k_sigma_ = py_key("sigma_")
    nk[].k_X_offset_ = py_key("X_offset_")
    nk[].k_X_scale_ = py_key("X_scale_")
    nk[].k__bayes_v = py_key("_mojolearn_bayes_v")
    nk[].k__bayes_ev = py_key("_mojolearn_bayes_ev")
    nk[].k__bayes_sigma = py_key("_mojolearn_bayes_sigma")
    nk[].k_tol = py_key("tol")
    nk[].k_warm_start = py_key("warm_start")
    nk[].k_xl_bindings = py_key("xl_bindings")
    nk[].k_xl_decimal = py_key("xl_decimal")
    nk[].k_xl_helpers = py_key("xl_helpers")
    nk[].k_xl_pow_context = py_key("xl_pow_context")
    nk[].k_xl_sgd_clf_loss = py_key("xl_sgd_clf_loss")
    nk[].k_xl_sgd_lr = py_key("xl_sgd_lr")
    nk[].k_xl_sgd_penalty = py_key("xl_sgd_penalty")
    nk[].k_xl_sgd_reg_loss = py_key("xl_sgd_reg_loss")
    nk[].k_xl_sgd_sum_losses = py_key("xl_sgd_sum_losses")
    nk[].k_xl_undefined_warning = py_key("xl_undefined_warning")
    nk[].k_y_max = py_key("y_max")
    nk[].k_y_min = py_key("y_min")
    nk[].k_y_thresholds_ = py_key("y_thresholds_")
    var ops = Python.dict()
    py_op_add(ops, "regressor_predict", X_LINEAR_OPN_REGRESSOR_PREDICT)
    py_op_add(ops, "regressor_predict_score", X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE)
    py_op_add(ops, "decision_function", X_LINEAR_OPN_DECISION_FUNCTION)
    py_op_add(ops, "classifier_predict", X_LINEAR_OPN_CLASSIFIER_PREDICT)
    py_op_add(ops, "classifier_score", X_LINEAR_OPN_CLASSIFIER_SCORE)
    py_op_add(ops, "sgd_predict_proba", X_LINEAR_OPN_SGD_PREDICT_PROBA)
    py_op_add(ops, "lrcv_predict_proba", X_LINEAR_OPN_LRCV_PREDICT_PROBA)
    py_op_add(ops, "fit_sgd_classifier", X_LINEAR_OPN_FIT_SGD_CLASSIFIER)
    py_op_add(ops, "fit_perceptron", X_LINEAR_OPN_FIT_PERCEPTRON)
    py_op_add(ops, "fit_pa_classifier", X_LINEAR_OPN_FIT_PA_CLASSIFIER)
    py_op_add(ops, "fit_ridge_classifier", X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER)
    py_op_add(ops, "fit_sgd_regressor", X_LINEAR_OPN_FIT_SGD_REGRESSOR)
    py_op_add(ops, "fit_pa_regressor", X_LINEAR_OPN_FIT_PA_REGRESSOR)
    py_op_add(ops, "fit_one_class", X_LINEAR_OPN_FIT_ONE_CLASS)
    py_op_add(ops, "one_class_decision_function", X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION)
    py_op_add(ops, "one_class_score_samples", X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES)
    py_op_add(ops, "one_class_predict", X_LINEAR_OPN_ONE_CLASS_PREDICT)
    py_op_add(ops, "fit_glm", X_LINEAR_OPN_FIT_GLM)
    py_op_add(ops, "fit_huber", X_LINEAR_OPN_FIT_HUBER)
    py_op_add(ops, "fit_bayes", X_LINEAR_OPN_FIT_BAYES)
    py_op_add(ops, "fit_ard", X_LINEAR_OPN_FIT_ARD)
    py_op_add(ops, "fit_lars", X_LINEAR_OPN_FIT_LARS)
    py_op_add(ops, "fit_lasso_lars", X_LINEAR_OPN_FIT_LASSO_LARS)
    py_op_add(ops, "fit_quantile", X_LINEAR_OPN_FIT_QUANTILE)
    py_op_add(ops, "isotonic_fit", X_LINEAR_OPN_ISOTONIC_FIT)
    py_op_add(ops, "isotonic_fit_transform", X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM)
    py_op_add(ops, "isotonic_transform", X_LINEAR_OPN_ISOTONIC_TRANSFORM)
    py_op_add(ops, "isotonic_predict", X_LINEAR_OPN_ISOTONIC_PREDICT)
    py_op_add(ops, "isotonic_score", X_LINEAR_OPN_ISOTONIC_SCORE)
    py_op_add(ops, "fit_ridge_cv", X_LINEAR_OPN_FIT_RIDGE_CV)
    py_op_add(ops, "fit_lasso_cv", X_LINEAR_OPN_FIT_LASSO_CV)
    py_op_add(ops, "fit_elastic_net_cv", X_LINEAR_OPN_FIT_ELASTIC_NET_CV)
    py_op_add(ops, "fit_logistic_cv", X_LINEAR_OPN_FIT_LOGISTIC_CV)
    py_op_add(ops, "cd_fit", X_LINEAR_OPN_CD_FIT)
    py_op_add(ops, "cd_predict", X_LINEAR_OPN_CD_PREDICT)
    py_op_add(ops, "cd_score", X_LINEAR_OPN_CD_SCORE)
    py_op_add(ops, "regressor_score", X_LINEAR_OPN_REGRESSOR_SCORE)
    py_op_add(ops, "device_admit", X_LINEAR_OPN_DEVICE_ADMIT)
    py_op_add(ops, "cd_vendor_used", X_LINEAR_OPN_CD_VENDOR_USED)
    py_op_add(ops, "bayes_sigma", X_LINEAR_OPN_BAYES_SIGMA)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def x_linear_keys() raises -> UnsafePointer[XLinearKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _X_LINEAR_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _x_linear_keys_fill(nk)
    return nk
