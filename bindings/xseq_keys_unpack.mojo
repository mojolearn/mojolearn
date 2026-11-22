# SPDX-License-Identifier: Apache-2.0
# The xseq family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

# bw2-sequence (2026-11-20): codes are numbered by owner, so `xseq_c` routes
# by range tests: recurrent state 0-1, recurrent models 2-7, perceptrons 8-12,
# Prophet 13-16, the VAR/STL matrix owner 17-24, then one block of four per
# forecaster kind (theta, croston, ets, garch: initialize, forecast, fit,
# predict; garch has no predict), so kind = (code - 25) // 4 and the step is
# (code - 25) % 4 (bindings/xseq_forecast_native_door.mojo).
comptime XSEQ_OPN_RNN_STATE_DICT = 0
comptime XSEQ_OPN_RNN_LOAD_STATE_DICT = 1
comptime XSEQ_OPN_RNN_FIT = 2
comptime XSEQ_OPN_RNN_PREDICT = 3
comptime XSEQ_OPN_RNN_PREDICT_PROBA = 4
comptime XSEQ_OPN_RNN_HIDDEN_SEQUENCE = 5
comptime XSEQ_OPN_RNN_SCORE = 6
comptime XSEQ_OPN_RNN_CLASSIFIER_SCORE = 7
comptime XSEQ_OPN_MLP_FIT = 8
comptime XSEQ_OPN_MLP_PREDICT = 9
comptime XSEQ_OPN_MLP_PREDICT_PROBA = 10
comptime XSEQ_OPN_MLP_SCORE = 11
comptime XSEQ_OPN_MLP_CLASSIFIER_SCORE = 12
comptime XSEQ_OPN_PROPHET_INITIALIZE = 13
comptime XSEQ_OPN_PROPHET_ADD_SEASONALITY = 14
comptime XSEQ_OPN_PROPHET_FIT = 15
comptime XSEQ_OPN_PROPHET_PREDICT = 16
comptime XSEQ_OPN_VAR_INITIALIZE = 17
comptime XSEQ_OPN_VAR_RESULTS_INITIALIZE = 18
comptime XSEQ_OPN_VAR_FORECAST = 19
comptime XSEQ_OPN_VAR_FIT = 20
comptime XSEQ_OPN_STL_INITIALIZE = 21
comptime XSEQ_OPN_STL_CONFIG = 22
comptime XSEQ_OPN_STL_FIT = 23
comptime XSEQ_OPN_DECOMPOSE_INITIALIZE = 24
comptime XSEQ_OPN_THETA_INITIALIZE = 25
comptime XSEQ_OPN_THETA_FORECAST = 26
comptime XSEQ_OPN_THETA_FIT = 27
comptime XSEQ_OPN_THETA_PREDICT = 28
comptime XSEQ_OPN_CROSTON_INITIALIZE = 29
comptime XSEQ_OPN_CROSTON_FORECAST = 30
comptime XSEQ_OPN_CROSTON_FIT = 31
comptime XSEQ_OPN_CROSTON_PREDICT = 32
comptime XSEQ_OPN_ETS_INITIALIZE = 33
comptime XSEQ_OPN_ETS_FORECAST = 34
comptime XSEQ_OPN_ETS_FIT = 35
comptime XSEQ_OPN_ETS_PREDICT = 36
comptime XSEQ_OPN_GARCH_INITIALIZE = 37
comptime XSEQ_OPN_GARCH_FORECAST = 38
comptime XSEQ_OPN_GARCH_FIT = 39
# `vendor_used` of the MLP and recurrent estimators: metadata, no input,
# answered from the estimator's route before any lifecycle step.
comptime XSEQ_OPN_XSEQ_VENDOR_USED = 40


struct XseqKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__K: PythonObject
    var k__LOSS_CODE: PythonObject
    var k__fast_stall: PythonObject
    var k__fc: PythonObject
    var k__fitted_h: PythonObject
    var k__last: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__nh: PythonObject
    var k__one: PythonObject
    var k__season: PythonObject
    var k__y: PythonObject
    var k__y_1d: PythonObject
    var k_activation: PythonObject
    var k_alpha: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_backend_fields: PythonObject
    var k_batch_size: PythonObject
    var k_best_loss_: PythonObject
    var k_beta: PythonObject
    var k_beta_1: PythonObject
    var k_beta_2: PythonObject
    var k_betas: PythonObject
    var k_centered: PythonObject
    var k_changepoint_prior_scale: PythonObject
    var k_changepoint_range: PythonObject
    var k_changepoints: PythonObject
    var k_changepoints_t_: PythonObject
    var k_classes_: PythonObject
    var k_coefs_: PythonObject
    var k_daily_seasonality: PythonObject
    var k_damped: PythonObject
    var k_dampening: PythonObject
    var k_decomposition_type: PythonObject
    var k_decoupled_weight_decay: PythonObject
    var k_eps: PythonObject
    var k_epsilon: PythonObject
    var k_float32: PythonObject
    var k_float64: PythonObject
    var k_fourier_order: PythonObject
    var k_gamma: PythonObject
    var k_hidden_layer_sizes: PythonObject
    var k_hidden_size: PythonObject
    var k_holidays_prior_scale: PythonObject
    var k_info_: PythonObject
    var k_initial_accumulator_value: PythonObject
    var k_initial_smoothed: PythonObject
    var k_int64: PythonObject
    var k_int_types: PythonObject
    var k_intercepts_: PythonObject
    var k_learning_rate: PythonObject
    var k_learning_rate_init: PythonObject
    var k_loss_: PythonObject
    var k_loss_curve_: PythonObject
    var k_lr_decay: PythonObject
    var k_lr_schedule: PythonObject
    var k_max_epochs: PythonObject
    var k_max_iter: PythonObject
    var k_mean: PythonObject
    var k_mean_: PythonObject
    var k_missing: PythonObject
    var k_model: PythonObject
    var k_model_native_api_v1: PythonObject
    var k_momentum: PythonObject
    var k_momentum_decay: PythonObject
    var k_n_changepoints: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_iter_no_change: PythonObject
    var k_n_layers_: PythonObject
    var k_n_outputs_: PythonObject
    var k_n_timesteps_: PythonObject
    var k_names: PythonObject
    var k_native_model_provider: PythonObject
    var k_ndarray: PythonObject
    var k_nesterov: PythonObject
    var k_nesterovs_momentum: PythonObject
    var k_nonlinearity: PythonObject
    var k_num_layers: PythonObject
    var k_numeric_mode: PythonObject
    var k_o: PythonObject
    var k_objective_: PythonObject
    var k_optimizer: PythonObject
    var k_optimizer_context: PythonObject
    var k_optimizer_options: PythonObject
    var k_out_activation_: PythonObject
    var k_p: PythonObject
    var k_params_: PythonObject
    var k_period: PythonObject
    var k_phi: PythonObject
    var k_power_t: PythonObject
    var k_predict_chunk: PythonObject
    var k_prior_scale: PythonObject
    var k_q: PythonObject
    var k_random_state: PythonObject
    var k_scalar_types: PythonObject
    var k_season_length: PythonObject
    var k_seasonalities: PythonObject
    var k_seasonality_mode: PythonObject
    var k_seasonality_prior_scale: PythonObject
    var k_shuffle: PythonObject
    var k_solver: PythonObject
    var k_start_: PythonObject
    var k_t_: PythonObject
    var k_t_scale_: PythonObject
    var k_theta: PythonObject
    var k_tol: PythonObject
    var k_trend_: PythonObject
    var k_weekly_seasonality: PythonObject
    var k_weight_decay: PythonObject
    var k_xseq_base: PythonObject
    var k_xseq_bindings: PythonObject
    var k_xseq_croston_classes: PythonObject
    var k_xseq_ets_classes: PythonObject
    var k_xseq_forecast_ownership: PythonObject
    var k_xseq_garch_classes: PythonObject
    var k_xseq_lifecycle_context: PythonObject
    var k_xseq_metrics: PythonObject
    var k_xseq_mlp_activations: PythonObject
    var k_xseq_mlp_outputs: PythonObject
    var k_xseq_mlp_rates: PythonObject
    var k_xseq_mlp_solvers: PythonObject
    var k_xseq_native_keys: PythonObject
    var k_xseq_ndarray: PythonObject
    var k_xseq_owner_context: PythonObject
    var k_xseq_r2_options: PythonObject
    var k_xseq_rnn_defaults: PythonObject
    var k_xseq_rnn_optimizers: PythonObject
    var k_xseq_theta_classes: PythonObject
    var k_xseq_theta_codes: PythonObject
    var k_xseq_theta_models: PythonObject
    var k_yearly_seasonality: PythonObject
    var k_conditional_volatility_: PythonObject
    var k_dist: PythonObject
    var k_loglikelihood_: PythonObject
    var k_model_: PythonObject
    var k_power: PythonObject
    var k_seasonal_states_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__K = PythonObject(None)
        self.k__LOSS_CODE = PythonObject(None)
        self.k__fast_stall = PythonObject(None)
        self.k__fc = PythonObject(None)
        self.k__fitted_h = PythonObject(None)
        self.k__last = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__nh = PythonObject(None)
        self.k__one = PythonObject(None)
        self.k__season = PythonObject(None)
        self.k__y = PythonObject(None)
        self.k__y_1d = PythonObject(None)
        self.k_activation = PythonObject(None)
        self.k_alpha = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_best_loss_ = PythonObject(None)
        self.k_beta = PythonObject(None)
        self.k_beta_1 = PythonObject(None)
        self.k_beta_2 = PythonObject(None)
        self.k_betas = PythonObject(None)
        self.k_centered = PythonObject(None)
        self.k_changepoint_prior_scale = PythonObject(None)
        self.k_changepoint_range = PythonObject(None)
        self.k_changepoints = PythonObject(None)
        self.k_changepoints_t_ = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_coefs_ = PythonObject(None)
        self.k_daily_seasonality = PythonObject(None)
        self.k_damped = PythonObject(None)
        self.k_dampening = PythonObject(None)
        self.k_decomposition_type = PythonObject(None)
        self.k_decoupled_weight_decay = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_epsilon = PythonObject(None)
        self.k_float32 = PythonObject(None)
        self.k_float64 = PythonObject(None)
        self.k_fourier_order = PythonObject(None)
        self.k_gamma = PythonObject(None)
        self.k_hidden_layer_sizes = PythonObject(None)
        self.k_hidden_size = PythonObject(None)
        self.k_holidays_prior_scale = PythonObject(None)
        self.k_info_ = PythonObject(None)
        self.k_initial_accumulator_value = PythonObject(None)
        self.k_initial_smoothed = PythonObject(None)
        self.k_int64 = PythonObject(None)
        self.k_int_types = PythonObject(None)
        self.k_intercepts_ = PythonObject(None)
        self.k_learning_rate = PythonObject(None)
        self.k_learning_rate_init = PythonObject(None)
        self.k_loss_ = PythonObject(None)
        self.k_loss_curve_ = PythonObject(None)
        self.k_lr_decay = PythonObject(None)
        self.k_lr_schedule = PythonObject(None)
        self.k_max_epochs = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_mean = PythonObject(None)
        self.k_mean_ = PythonObject(None)
        self.k_missing = PythonObject(None)
        self.k_model = PythonObject(None)
        self.k_model_native_api_v1 = PythonObject(None)
        self.k_momentum = PythonObject(None)
        self.k_momentum_decay = PythonObject(None)
        self.k_n_changepoints = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_iter_no_change = PythonObject(None)
        self.k_n_layers_ = PythonObject(None)
        self.k_n_outputs_ = PythonObject(None)
        self.k_n_timesteps_ = PythonObject(None)
        self.k_names = PythonObject(None)
        self.k_native_model_provider = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_nesterov = PythonObject(None)
        self.k_nesterovs_momentum = PythonObject(None)
        self.k_nonlinearity = PythonObject(None)
        self.k_num_layers = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_o = PythonObject(None)
        self.k_objective_ = PythonObject(None)
        self.k_optimizer = PythonObject(None)
        self.k_optimizer_context = PythonObject(None)
        self.k_optimizer_options = PythonObject(None)
        self.k_out_activation_ = PythonObject(None)
        self.k_p = PythonObject(None)
        self.k_params_ = PythonObject(None)
        self.k_period = PythonObject(None)
        self.k_phi = PythonObject(None)
        self.k_power_t = PythonObject(None)
        self.k_predict_chunk = PythonObject(None)
        self.k_prior_scale = PythonObject(None)
        self.k_q = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_season_length = PythonObject(None)
        self.k_seasonalities = PythonObject(None)
        self.k_seasonality_mode = PythonObject(None)
        self.k_seasonality_prior_scale = PythonObject(None)
        self.k_shuffle = PythonObject(None)
        self.k_solver = PythonObject(None)
        self.k_start_ = PythonObject(None)
        self.k_t_ = PythonObject(None)
        self.k_t_scale_ = PythonObject(None)
        self.k_theta = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_trend_ = PythonObject(None)
        self.k_weekly_seasonality = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_xseq_base = PythonObject(None)
        self.k_xseq_bindings = PythonObject(None)
        self.k_xseq_croston_classes = PythonObject(None)
        self.k_xseq_ets_classes = PythonObject(None)
        self.k_xseq_forecast_ownership = PythonObject(None)
        self.k_xseq_garch_classes = PythonObject(None)
        self.k_xseq_lifecycle_context = PythonObject(None)
        self.k_xseq_metrics = PythonObject(None)
        self.k_xseq_mlp_activations = PythonObject(None)
        self.k_xseq_mlp_outputs = PythonObject(None)
        self.k_xseq_mlp_rates = PythonObject(None)
        self.k_xseq_mlp_solvers = PythonObject(None)
        self.k_xseq_native_keys = PythonObject(None)
        self.k_xseq_ndarray = PythonObject(None)
        self.k_xseq_owner_context = PythonObject(None)
        self.k_xseq_r2_options = PythonObject(None)
        self.k_xseq_rnn_defaults = PythonObject(None)
        self.k_xseq_rnn_optimizers = PythonObject(None)
        self.k_xseq_theta_classes = PythonObject(None)
        self.k_xseq_theta_codes = PythonObject(None)
        self.k_xseq_theta_models = PythonObject(None)
        self.k_yearly_seasonality = PythonObject(None)
        self.k_conditional_volatility_ = PythonObject(None)
        self.k_dist = PythonObject(None)
        self.k_loglikelihood_ = PythonObject(None)
        self.k_model_ = PythonObject(None)
        self.k_power = PythonObject(None)
        self.k_seasonal_states_ = PythonObject(None)


comptime _XSEQ_KEYS = SharedSlot[XseqKeys, "MojolearnXseqKeysV1"]


def _xseq_keys_fill(nk: UnsafePointer[XseqKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__K = py_key("_K")
    nk[].k__LOSS_CODE = py_key("_LOSS_CODE")
    nk[].k__fast_stall = py_key("_fast_stall")
    nk[].k__fc = py_key("_fc")
    nk[].k__fitted_h = py_key("_fitted_h")
    nk[].k__last = py_key("_last")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__nh = py_key("_nh")
    nk[].k__one = py_key("_one")
    nk[].k__season = py_key("_season")
    nk[].k__y = py_key("_y")
    nk[].k__y_1d = py_key("_y_1d")
    nk[].k_activation = py_key("activation")
    nk[].k_alpha = py_key("alpha")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_best_loss_ = py_key("best_loss_")
    nk[].k_beta = py_key("beta")
    nk[].k_beta_1 = py_key("beta_1")
    nk[].k_beta_2 = py_key("beta_2")
    nk[].k_betas = py_key("betas")
    nk[].k_centered = py_key("centered")
    nk[].k_changepoint_prior_scale = py_key("changepoint_prior_scale")
    nk[].k_changepoint_range = py_key("changepoint_range")
    nk[].k_changepoints = py_key("changepoints")
    nk[].k_changepoints_t_ = py_key("changepoints_t_")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_coefs_ = py_key("coefs_")
    nk[].k_daily_seasonality = py_key("daily_seasonality")
    nk[].k_damped = py_key("damped")
    nk[].k_dampening = py_key("dampening")
    nk[].k_decomposition_type = py_key("decomposition_type")
    nk[].k_decoupled_weight_decay = py_key("decoupled_weight_decay")
    nk[].k_eps = py_key("eps")
    nk[].k_epsilon = py_key("epsilon")
    nk[].k_float32 = py_key("float32")
    nk[].k_float64 = py_key("float64")
    nk[].k_fourier_order = py_key("fourier_order")
    nk[].k_gamma = py_key("gamma")
    nk[].k_hidden_layer_sizes = py_key("hidden_layer_sizes")
    nk[].k_hidden_size = py_key("hidden_size")
    nk[].k_holidays_prior_scale = py_key("holidays_prior_scale")
    nk[].k_info_ = py_key("info_")
    nk[].k_initial_accumulator_value = py_key("initial_accumulator_value")
    nk[].k_initial_smoothed = py_key("initial_smoothed")
    nk[].k_int64 = py_key("int64")
    nk[].k_int_types = py_key("int_types")
    nk[].k_intercepts_ = py_key("intercepts_")
    nk[].k_learning_rate = py_key("learning_rate")
    nk[].k_learning_rate_init = py_key("learning_rate_init")
    nk[].k_loss_ = py_key("loss_")
    nk[].k_loss_curve_ = py_key("loss_curve_")
    nk[].k_lr_decay = py_key("lr_decay")
    nk[].k_lr_schedule = py_key("lr_schedule")
    nk[].k_max_epochs = py_key("max_epochs")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_mean = py_key("mean")
    nk[].k_mean_ = py_key("mean_")
    nk[].k_missing = py_key("missing")
    nk[].k_model = py_key("model")
    nk[].k_model_native_api_v1 = py_key("model_native_api_v1")
    nk[].k_momentum = py_key("momentum")
    nk[].k_momentum_decay = py_key("momentum_decay")
    nk[].k_n_changepoints = py_key("n_changepoints")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_iter_no_change = py_key("n_iter_no_change")
    nk[].k_n_layers_ = py_key("n_layers_")
    nk[].k_n_outputs_ = py_key("n_outputs_")
    nk[].k_n_timesteps_ = py_key("n_timesteps_")
    nk[].k_names = py_key("names")
    nk[].k_native_model_provider = py_key("native_model_provider")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_nesterov = py_key("nesterov")
    nk[].k_nesterovs_momentum = py_key("nesterovs_momentum")
    nk[].k_nonlinearity = py_key("nonlinearity")
    nk[].k_num_layers = py_key("num_layers")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_o = py_key("o")
    nk[].k_objective_ = py_key("objective_")
    nk[].k_optimizer = py_key("optimizer")
    nk[].k_optimizer_context = py_key("optimizer_context")
    nk[].k_optimizer_options = py_key("optimizer_options")
    nk[].k_out_activation_ = py_key("out_activation_")
    nk[].k_p = py_key("p")
    nk[].k_params_ = py_key("params_")
    nk[].k_period = py_key("period")
    nk[].k_phi = py_key("phi")
    nk[].k_power_t = py_key("power_t")
    nk[].k_predict_chunk = py_key("predict_chunk")
    nk[].k_prior_scale = py_key("prior_scale")
    nk[].k_q = py_key("q")
    nk[].k_random_state = py_key("random_state")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_season_length = py_key("season_length")
    nk[].k_seasonalities = py_key("seasonalities")
    nk[].k_seasonality_mode = py_key("seasonality_mode")
    nk[].k_seasonality_prior_scale = py_key("seasonality_prior_scale")
    nk[].k_shuffle = py_key("shuffle")
    nk[].k_solver = py_key("solver")
    nk[].k_start_ = py_key("start_")
    nk[].k_t_ = py_key("t_")
    nk[].k_t_scale_ = py_key("t_scale_")
    nk[].k_theta = py_key("theta")
    nk[].k_tol = py_key("tol")
    nk[].k_trend_ = py_key("trend_")
    nk[].k_weekly_seasonality = py_key("weekly_seasonality")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_xseq_base = py_key("xseq_base")
    nk[].k_xseq_bindings = py_key("xseq_bindings")
    nk[].k_xseq_croston_classes = py_key("xseq_croston_classes")
    nk[].k_xseq_ets_classes = py_key("xseq_ets_classes")
    nk[].k_xseq_forecast_ownership = py_key("xseq_forecast_ownership")
    nk[].k_xseq_garch_classes = py_key("xseq_garch_classes")
    nk[].k_xseq_lifecycle_context = py_key("xseq_lifecycle_context")
    nk[].k_xseq_metrics = py_key("xseq_metrics")
    nk[].k_xseq_mlp_activations = py_key("xseq_mlp_activations")
    nk[].k_xseq_mlp_outputs = py_key("xseq_mlp_outputs")
    nk[].k_xseq_mlp_rates = py_key("xseq_mlp_rates")
    nk[].k_xseq_mlp_solvers = py_key("xseq_mlp_solvers")
    nk[].k_xseq_native_keys = py_key("xseq_native_keys")
    nk[].k_xseq_ndarray = py_key("xseq_ndarray")
    nk[].k_xseq_owner_context = py_key("xseq_owner_context")
    nk[].k_xseq_r2_options = py_key("xseq_r2_options")
    nk[].k_xseq_rnn_defaults = py_key("xseq_rnn_defaults")
    nk[].k_xseq_rnn_optimizers = py_key("xseq_rnn_optimizers")
    nk[].k_xseq_theta_classes = py_key("xseq_theta_classes")
    nk[].k_xseq_theta_codes = py_key("xseq_theta_codes")
    nk[].k_xseq_theta_models = py_key("xseq_theta_models")
    nk[].k_yearly_seasonality = py_key("yearly_seasonality")
    nk[].k_conditional_volatility_ = py_key("conditional_volatility_")
    nk[].k_dist = py_key("dist")
    nk[].k_loglikelihood_ = py_key("loglikelihood_")
    nk[].k_model_ = py_key("model_")
    nk[].k_power = py_key("power")
    nk[].k_seasonal_states_ = py_key("seasonal_states_")
    var ops = Python.dict()
    py_op_add(ops, "rnn_state_dict", XSEQ_OPN_RNN_STATE_DICT)
    py_op_add(ops, "rnn_load_state_dict", XSEQ_OPN_RNN_LOAD_STATE_DICT)
    py_op_add(ops, "rnn_fit", XSEQ_OPN_RNN_FIT)
    py_op_add(ops, "rnn_predict", XSEQ_OPN_RNN_PREDICT)
    py_op_add(ops, "rnn_predict_proba", XSEQ_OPN_RNN_PREDICT_PROBA)
    py_op_add(ops, "rnn_hidden_sequence", XSEQ_OPN_RNN_HIDDEN_SEQUENCE)
    py_op_add(ops, "rnn_score", XSEQ_OPN_RNN_SCORE)
    py_op_add(ops, "rnn_classifier_score", XSEQ_OPN_RNN_CLASSIFIER_SCORE)
    py_op_add(ops, "xseq_vendor_used", XSEQ_OPN_XSEQ_VENDOR_USED)
    py_op_add(ops, "mlp_fit", XSEQ_OPN_MLP_FIT)
    py_op_add(ops, "mlp_predict", XSEQ_OPN_MLP_PREDICT)
    py_op_add(ops, "mlp_predict_proba", XSEQ_OPN_MLP_PREDICT_PROBA)
    py_op_add(ops, "mlp_score", XSEQ_OPN_MLP_SCORE)
    py_op_add(ops, "mlp_classifier_score", XSEQ_OPN_MLP_CLASSIFIER_SCORE)
    py_op_add(ops, "prophet_initialize", XSEQ_OPN_PROPHET_INITIALIZE)
    py_op_add(ops, "prophet_add_seasonality", XSEQ_OPN_PROPHET_ADD_SEASONALITY)
    py_op_add(ops, "prophet_fit", XSEQ_OPN_PROPHET_FIT)
    py_op_add(ops, "prophet_predict", XSEQ_OPN_PROPHET_PREDICT)
    py_op_add(ops, "var_initialize", XSEQ_OPN_VAR_INITIALIZE)
    py_op_add(ops, "var_results_initialize", XSEQ_OPN_VAR_RESULTS_INITIALIZE)
    py_op_add(ops, "var_forecast", XSEQ_OPN_VAR_FORECAST)
    py_op_add(ops, "var_fit", XSEQ_OPN_VAR_FIT)
    py_op_add(ops, "stl_initialize", XSEQ_OPN_STL_INITIALIZE)
    py_op_add(ops, "stl_config", XSEQ_OPN_STL_CONFIG)
    py_op_add(ops, "stl_fit", XSEQ_OPN_STL_FIT)
    py_op_add(ops, "decompose_initialize", XSEQ_OPN_DECOMPOSE_INITIALIZE)
    py_op_add(ops, "theta_initialize", XSEQ_OPN_THETA_INITIALIZE)
    py_op_add(ops, "theta_forecast", XSEQ_OPN_THETA_FORECAST)
    py_op_add(ops, "theta_fit", XSEQ_OPN_THETA_FIT)
    py_op_add(ops, "theta_predict", XSEQ_OPN_THETA_PREDICT)
    py_op_add(ops, "croston_initialize", XSEQ_OPN_CROSTON_INITIALIZE)
    py_op_add(ops, "croston_forecast", XSEQ_OPN_CROSTON_FORECAST)
    py_op_add(ops, "croston_fit", XSEQ_OPN_CROSTON_FIT)
    py_op_add(ops, "croston_predict", XSEQ_OPN_CROSTON_PREDICT)
    py_op_add(ops, "ets_initialize", XSEQ_OPN_ETS_INITIALIZE)
    py_op_add(ops, "ets_forecast", XSEQ_OPN_ETS_FORECAST)
    py_op_add(ops, "ets_fit", XSEQ_OPN_ETS_FIT)
    py_op_add(ops, "ets_predict", XSEQ_OPN_ETS_PREDICT)
    py_op_add(ops, "garch_initialize", XSEQ_OPN_GARCH_INITIALIZE)
    py_op_add(ops, "garch_forecast", XSEQ_OPN_GARCH_FORECAST)
    py_op_add(ops, "garch_fit", XSEQ_OPN_GARCH_FIT)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def xseq_keys() raises -> UnsafePointer[XseqKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _XSEQ_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _xseq_keys_fill(nk)
    return nk
