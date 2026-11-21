# SPDX-License-Identifier: Apache-2.0
# The parallel family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
# Lane bw2-parallel (2026-11-20): one table for the parallel family's doors.
# The dispatchers' names collide across doors ("init", "close", "worker"), so
# each dispatcher has its own `ops_<door>` dictionary and `PARALLEL_<DOOR>_OPN_*`
# codes; the cooperative drivers' names and owner operations are filled from
# bindings/parallel_plan_specs.mojo (Python-free, where their codes live).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add
from bindings.parallel_plan_specs import (
    PARALLEL_DRIVER_COUNT, PARALLEL_OWNER_COUNT, par_spec_driver_name, par_spec_owner_name,
)

# `byte` dispatcher codes (`ops_byte`).
comptime PARALLEL_BYTE_OPN_INITIALIZE = 0
comptime PARALLEL_BYTE_OPN_ADMIT_CLASS = 1
comptime PARALLEL_BYTE_OPN_INIT = 2
comptime PARALLEL_BYTE_OPN_FROM_CHECKPOINT = 3
comptime PARALLEL_BYTE_OPN_STEP = 4
comptime PARALLEL_BYTE_OPN_ENTER = 5
comptime PARALLEL_BYTE_OPN_ADMITTED_STATE = 6
comptime PARALLEL_BYTE_OPN_CLOSE = 7
comptime PARALLEL_BYTE_OPN_OPEN = 8
comptime PARALLEL_BYTE_OPN_TRAIN_STEP = 9
comptime PARALLEL_BYTE_OPN_STATE = 10
comptime PARALLEL_BYTE_OPN_EXPORT_RAW = 11
comptime PARALLEL_BYTE_OPN_EXPORT_GRADIENTS = 12
comptime PARALLEL_BYTE_OPN_CHECKPOINT = 13
comptime PARALLEL_BYTE_OPN_SET_LR = 14
comptime PARALLEL_BYTE_OPN_REQUIRE_FOLD = 15
comptime PARALLEL_BYTE_OPN_HAS_DEVICE_FOLD = 16
comptime PARALLEL_BYTE_OPN_OWNERSHIP = 17
comptime PARALLEL_BYTE_OPN_MODEL_OWNERSHIP = 18
comptime PARALLEL_BYTE_OPN_SHARD_GRADIENT = 19
comptime PARALLEL_BYTE_OPN_SHARD_GRADIENT_FOLD = 20
comptime PARALLEL_BYTE_OPN_APPLY_GRADIENT = 21
comptime PARALLEL_BYTE_OPN_FOLD_RESET = 22
comptime PARALLEL_BYTE_OPN_FOLD_ADD = 23
comptime PARALLEL_BYTE_OPN_FOLD_EXPORT = 24

# `mlp` dispatcher codes (`ops_mlp`).
comptime PARALLEL_MLP_OPN_INITIALIZE = 0
comptime PARALLEL_MLP_OPN_SNAPSHOT = 1
comptime PARALLEL_MLP_OPN_VALIDATE = 2
comptime PARALLEL_MLP_OPN_FROM_STATE = 3
comptime PARALLEL_MLP_OPN_PUBLISH = 4
comptime PARALLEL_MLP_OPN_GRADIENT = 5
comptime PARALLEL_MLP_OPN_ORDERED_SUM = 6
comptime PARALLEL_MLP_OPN_APPLY = 7

# `neural` dispatcher codes (`ops_neural`).
comptime PARALLEL_NEURAL_OPN_INITIALIZE = 0
comptime PARALLEL_NEURAL_OPN_INIT = 1
comptime PARALLEL_NEURAL_OPN_INIT_WITH_WORKER = 2
comptime PARALLEL_NEURAL_OPN_RESTORE = 3
comptime PARALLEL_NEURAL_OPN_ORDERED_SUM = 4
comptime PARALLEL_NEURAL_OPN_WORKER = 5
comptime PARALLEL_NEURAL_OPN_TRAIN = 6
comptime PARALLEL_NEURAL_OPN_EXPORT_GRADIENTS = 7
comptime PARALLEL_NEURAL_OPN_CHECKPOINT = 8
comptime PARALLEL_NEURAL_OPN_CLOSE = 9
comptime PARALLEL_NEURAL_OPN_EXIT = 10
comptime PARALLEL_NEURAL_OPN_ENTER = 11


# `forecast` dispatcher codes (`ops_forecast`).
comptime PARALLEL_FORECAST_OPN_INITIALIZE = 0
comptime PARALLEL_FORECAST_OPN_PREDICT_ARIMA = 1
comptime PARALLEL_FORECAST_OPN_FORECAST_ARIMA = 2
comptime PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING = 3
comptime PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING = 4
comptime PARALLEL_FORECAST_OPN_FIT_ARIMA = 5
comptime PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING = 6
comptime PARALLEL_FORECAST_OPN_WORKER = 7

# `forecast_worker` dispatcher codes (`ops_forecast_worker`).
comptime PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD = 0
comptime PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD = 1
comptime PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD = 2
comptime PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD = 3
comptime PARALLEL_FORECAST_WORKER_OPN_IDENTITY = 4

# `shard` dispatcher codes (`ops_shard`).
comptime PARALLEL_SHARD_OPN_REGISTER = 0
comptime PARALLEL_SHARD_OPN_TRANSFORM_RBF_SAMPLER = 1
comptime PARALLEL_SHARD_OPN_FIT_SCALER = 2
comptime PARALLEL_SHARD_OPN_TRANSFORM_SCALER = 3
comptime PARALLEL_SHARD_OPN_FIT_GAUSSIAN_PROCESS_CLASSIFIER = 4
comptime PARALLEL_SHARD_OPN_PREDICT_GAUSSIAN_PROCESS_CLASSIFIER = 5
comptime PARALLEL_SHARD_OPN_FIT_FOREST = 6
comptime PARALLEL_SHARD_OPN_TRACE_TAKE = 7
comptime PARALLEL_SHARD_OPN_FAULTS_TAKE = 8
comptime PARALLEL_SHARD_OPN_FAULTS_ARM = 9
comptime PARALLEL_SHARD_OPN_SABOTAGE_ARM = 10
comptime PARALLEL_SHARD_OPN_REFERENCE_INIT = 11
comptime PARALLEL_SHARD_OPN_QUERIES_INIT = 12
comptime PARALLEL_SHARD_OPN_HOLDER_ENTER = 13
comptime PARALLEL_SHARD_OPN_HOLDER_CLOSE = 14
comptime PARALLEL_SHARD_OPN_IVF_INIT = 15
comptime PARALLEL_SHARD_OPN_IVF_ENTER = 16
comptime PARALLEL_SHARD_OPN_REFERENCE_QUERY = 17
comptime PARALLEL_SHARD_OPN_PARALLEL_QUERIES = 18
comptime PARALLEL_SHARD_OPN_IVF_FROM_INDEX = 19
comptime PARALLEL_SHARD_OPN_IVF_SEARCH = 20
comptime PARALLEL_SHARD_OPN_IVF_CLOSE = 21
comptime PARALLEL_SHARD_OPN_WORKER = 22

# `shard_worker` dispatcher codes (`ops_shard_worker`).
comptime PARALLEL_SHARD_WORKER_OPN_IDENTITY = 0
comptime PARALLEL_SHARD_WORKER_OPN_SHARD_STORE = 1
comptime PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE = 2
comptime PARALLEL_SHARD_WORKER_OPN_RBF_ROWS = 3
comptime PARALLEL_SHARD_WORKER_OPN_SCALER_FIT = 4
comptime PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM = 5
comptime PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT = 6
comptime PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT = 7
comptime PARALLEL_SHARD_WORKER_OPN_FOREST_FIT = 8
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD = 9
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE = 10
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE = 11
comptime PARALLEL_SHARD_WORKER_OPN_QUERY_STATE = 12
comptime PARALLEL_SHARD_WORKER_OPN_RELEASE = 13
comptime PARALLEL_SHARD_WORKER_OPN_IVF_STORE = 14
comptime PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH = 15
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY = 16
comptime PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS = 17

# `numerical` dispatcher codes (`ops_numerical`).
comptime PARALLEL_NUMERICAL_OPN_PY = 0
comptime PARALLEL_NUMERICAL_OPN_ARIMA_PREDICT_SHARD = 1
comptime PARALLEL_NUMERICAL_OPN_HW_PREDICT_SHARD = 2
comptime PARALLEL_NUMERICAL_OPN_GPC_CLASS_FIT = 3
comptime PARALLEL_NUMERICAL_OPN_GPC_CLASS_PREDICT = 4
comptime PARALLEL_NUMERICAL_OPN_IVF_SEARCH = 5
comptime PARALLEL_NUMERICAL_OPN_RBF_ROWS = 6
comptime PARALLEL_NUMERICAL_OPN_SCALER_FIT = 7
comptime PARALLEL_NUMERICAL_OPN_SCALER_TRANSFORM = 8
comptime PARALLEL_NUMERICAL_OPN_FOREST_FIT = 9
comptime PARALLEL_NUMERICAL_OPN_ARIMA_FIT_SHARD = 10
comptime PARALLEL_NUMERICAL_OPN_HW_FIT_SHARD = 11
comptime PARALLEL_NUMERICAL_OPN_REFERENCE_SHARD = 12
comptime PARALLEL_NUMERICAL_OPN_REFERENCE_QUERY = 13
comptime PARALLEL_NUMERICAL_OPN_QUERY_ROWS = 14

# `causal` dispatcher codes (`ops_causal`).
comptime PARALLEL_CAUSAL_OPN_INITIALIZE = 0
comptime PARALLEL_CAUSAL_OPN_WORKER = 1
comptime PARALLEL_CAUSAL_OPN_INIT = 2
comptime PARALLEL_CAUSAL_OPN_LOAD = 3
comptime PARALLEL_CAUSAL_OPN_LOAD_WITH_WORKER = 4
comptime PARALLEL_CAUSAL_OPN_RPC = 5
comptime PARALLEL_CAUSAL_OPN_CAPTURE_MODEL = 6
comptime PARALLEL_CAUSAL_OPN_CAPTURE_STATE = 7
comptime PARALLEL_CAUSAL_OPN_PARAMETERS = 8
comptime PARALLEL_CAUSAL_OPN_BLOCK_FIELDS = 9
comptime PARALLEL_CAUSAL_OPN_ALLOCATE_LAYER = 10
comptime PARALLEL_CAUSAL_OPN_BLOCK_ALLOCATE = 11
comptime PARALLEL_CAUSAL_OPN_BLOCK_RUN = 12
comptime PARALLEL_CAUSAL_OPN_BLOCK_CALL = 13
comptime PARALLEL_CAUSAL_OPN_BLOCK_INIT = 14
comptime PARALLEL_CAUSAL_OPN_STATE_INIT = 15
comptime PARALLEL_CAUSAL_OPN_STATE_SNAPSHOT = 16
comptime PARALLEL_CAUSAL_OPN_STATE_PUBLISH = 17
comptime PARALLEL_CAUSAL_OPN_STATE_CLOSE = 18
comptime PARALLEL_CAUSAL_OPN_RELEASE_LAYERS = 19
comptime PARALLEL_CAUSAL_OPN_RUN = 20
comptime PARALLEL_CAUSAL_OPN_CLOSE = 21
comptime PARALLEL_CAUSAL_OPN_CLOSE_QUIET = 22
comptime PARALLEL_CAUSAL_OPN_EXIT = 23
comptime PARALLEL_CAUSAL_OPN_ENTER = 24
comptime PARALLEL_CAUSAL_OPN_PRIMITIVES_INIT = 25
comptime PARALLEL_CAUSAL_OPN_PRIMITIVE = 26

# `predictor` dispatcher codes (`ops_predictor`).
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_INIT = 0
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_CLOSE = 1
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_EXIT = 2
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_ENTER = 3
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT_PROBA = 4
comptime PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT = 5



struct ParallelKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops_byte: PythonObject
    var ops_mlp: PythonObject
    var ops_neural: PythonObject
    var ops_forecast: PythonObject
    var ops_forecast_worker: PythonObject
    var ops_shard: PythonObject
    var ops_shard_worker: PythonObject
    var ops_numerical: PythonObject
    var ops_causal: PythonObject
    var ops_predictor: PythonObject
    var ops_par_driver: PythonObject
    var ops_par_owner: PythonObject
    var k_L_: PythonObject
    var k_MOJOLEARN_NATIVE_CV_NUMPY: PythonObject
    var k_W_sr_: PythonObject
    var k_X_train_: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k__blocks: PythonObject
    var k__causal_admission_failure: PythonObject
    var k__causal_initial_buffers: PythonObject
    var k__causal_live_fields: PythonObject
    var k__causal_original: PythonObject
    var k__classes_list: PythonObject
    var k__closed: PythonObject
    var k__components_len: PythonObject
    var k__comps: PythonObject
    var k__data: PythonObject
    var k__device_list_data: PythonObject
    var k__embed: PythonObject
    var k__exog: PythonObject
    var k__failed: PythonObject
    var k__fit_numeric_mode: PythonObject
    var k__handle: PythonObject
    var k__head: PythonObject
    var k__head_int15: PythonObject
    var k__id_maps: PythonObject
    var k__index: PythonObject
    var k__kernel: PythonObject
    var k__label_names: PythonObject
    var k__lifetime: PythonObject
    var k__lock: PythonObject
    var k__max_iter_predict: PythonObject
    var k__metric_arg: PythonObject
    var k__metric_value: PythonObject
    var k__model: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__n_trees: PythonObject
    var k__native: PythonObject
    var k__native_entries: PythonObject
    var k__native_numeric_mode: PythonObject
    var k__native_parallel_block: PythonObject
    var k__native_parallel_blocks: PythonObject
    var k__native_parallel_entries_live: PythonObject
    var k__native_parallel_extra_blocks: PythonObject
    var k__native_parallel_model: PythonObject
    var k__native_state_identity: PythonObject
    var k__native_vendor: PythonObject
    var k__norm: PythonObject
    var k__num_outputs: PythonObject
    var k__offsets: PythonObject
    var k__pool: PythonObject
    var k__pools: PythonObject
    var k__resident: PythonObject
    var k__resident_holder: PythonObject
    var k__session: PythonObject
    var k__shape: PythonObject
    var k__token: PythonObject
    var k__transform_mode: PythonObject
    var k__update_pool: PythonObject
    var k__w: PythonObject
    var k__worker_index: PythonObject
    var k__y: PythonObject
    var k__y_cols: PythonObject
    var k__y_ndim: PythonObject
    var k_accumulation_steps: PythonObject
    var k_admission: PythonObject
    var k_algorithm: PythonObject
    var k_architecture: PythonObject
    var k_arima_fit_shard: PythonObject
    var k_arima_predict_shard: PythonObject
    var k_array_class: PythonObject
    var k_assignments: PythonObject
    var k_backend: PythonObject
    var k_backend_fields: PythonObject
    var k_batch_size_: PythonObject
    var k_beta1: PythonObject
    var k_beta2: PythonObject
    var k_block: PythonObject
    var k_block_kwargs: PythonObject
    var k_block_options: PythonObject
    var k_calls: PythonObject
    var k_capture_classes: PythonObject
    var k_center_norms_: PythonObject
    var k_centers_: PythonObject
    var k_check: PythonObject
    var k_checked_of: PythonObject
    var k_checkpoint_class: PythonObject
    var k_checkpoint_context: PythonObject
    var k_chunked_lm_head_v2: PythonObject
    var k_classes: PythonObject
    var k_classes_: PythonObject
    var k_clip: PythonObject
    var k_clip_: PythonObject
    var k_closed: PythonObject
    var k_completed_steps: PythonObject
    var k_complexity_: PythonObject
    var k_config: PythonObject
    var k_config_class: PythonObject
    var k_config_context: PythonObject
    var k_copy: PythonObject
    var k_counter: PythonObject
    var k_cpu_only: PythonObject
    var k_create: PythonObject
    var k_criterion_: PythonObject
    var k_data_schedule: PythonObject
    var k_decompositions: PythonObject
    var k_dense_solvers: PythonObject
    var k_device: PythonObject
    var k_device_arg: PythonObject
    var k_devices: PythonObject
    var k_dims: PythonObject
    var k_driver: PythonObject
    var k_dropout: PythonObject
    var k_end: PythonObject
    var k_endog: PythonObject
    var k_entry: PythonObject
    var k_entry_mode: PythonObject
    var k_entry_vendor: PythonObject
    var k_environ: PythonObject
    var k_environment: PythonObject
    var k_epoch: PythonObject
    var k_eps: PythonObject
    var k_error: PythonObject
    var k_estimator: PythonObject
    var k_estimators_: PythonObject
    var k_exception_args: PythonObject
    var k_executable: PythonObject
    var k_family: PythonObject
    var k_family_msel_parallel: PythonObject
    var k_feature_range: PythonObject
    var k_feature_range_: PythonObject
    var k_fields_get: PythonObject
    var k_fit_executed_flag: PythonObject
    var k_flags: PythonObject
    var k_forest_classifier: PythonObject
    var k_forest_fit: PythonObject
    var k_forests: PythonObject
    var k_full: PythonObject
    var k_generation_classes: PythonObject
    var k_gpc_class_fit: PythonObject
    var k_gpc_class_predict: PythonObject
    var k_handle: PythonObject
    var k_head: PythonObject
    var k_history: PythonObject
    var k_hooks: PythonObject
    var k_hw_fit_shard: PythonObject
    var k_hw_predict_shard: PythonObject
    var k_identical: PythonObject
    var k_identities: PythonObject
    var k_identity: PythonObject
    var k_index: PythonObject
    var k_inference_engine: PythonObject
    var k_input_copied_: PythonObject
    var k_integer_types: PythonObject
    var k_inventory: PythonObject
    var k_ivf_class: PythonObject
    var k_ivf_search: PythonObject
    var k_ivf_store: PythonObject
    var k_kde_class: PythonObject
    var k_kind: PythonObject
    var k_knc_class: PythonObject
    var k_knr_class: PythonObject
    var k_last_run: PythonObject
    var k_last_shards_: PythonObject
    var k_layer: PythonObject
    var k_layer_devices: PythonObject
    var k_layers: PythonObject
    var k_list_data_: PythonObject
    var k_list_indices_: PythonObject
    var k_list_offsets_: PythonObject
    var k_lock_type: PythonObject
    var k_log_marginal_likelihood_value_: PythonObject
    var k_logical_shards: PythonObject
    var k_losses: PythonObject
    var k_lr: PythonObject
    var k_m: PythonObject
    var k_mamba1_profile: PythonObject
    var k_max_positions: PythonObject
    var k_mean_: PythonObject
    var k_method_arg: PythonObject
    var k_metric: PythonObject
    var k_metric_code_: PythonObject
    var k_min_: PythonObject
    var k_mlp_slot: PythonObject
    var k_mode: PythonObject
    var k_model: PythonObject
    var k_model_class: PythonObject
    var k_model_shape: PythonObject
    var k_modules: PythonObject
    var k_mro: PythonObject
    var k_n: PythonObject
    var k_n_candidates_: PythonObject
    var k_n_classes_: PythonObject
    var k_n_estimators: PythonObject
    var k_n_estimators_: PythonObject
    var k_n_exog_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_layers: PythonObject
    var k_n_lists_: PythonObject
    var k_n_neighbors: PythonObject
    var k_n_obs_: PythonObject
    var k_n_probes: PythonObject
    var k_n_rows_: PythonObject
    var k_n_samples_fit_: PythonObject
    var k_n_samples_seen_: PythonObject
    var k_n_tensors: PythonObject
    var k_n_total: PythonObject
    var k_n_vocab: PythonObject
    var k_names: PythonObject
    var k_nb_: PythonObject
    var k_nearest_class: PythonObject
    var k_nesterov: PythonObject
    var k_next_batch_index: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_arg: PythonObject
    var k_numeric_profile: PythonObject
    var k_offsets: PythonObject
    var k_operation: PythonObject
    var k_optimizer: PythonObject
    var k_output: PythonObject
    var k_outputs_2d_: PythonObject
    var k_oversampling_factor: PythonObject
    var k_owner: PythonObject
    var k_owner_fields: PythonObject
    var k_p: PythonObject
    var k_parallel_call: PythonObject
    var k_parallel_public: PythonObject
    var k_parameter_names: PythonObject
    var k_parameter_offsets: PythonObject
    var k_parameter_order: PythonObject
    var k_parameter_shapes: PythonObject
    var k_parameters: PythonObject
    var k_params_: PythonObject
    var k_pca: PythonObject
    var k_pi_: PythonObject
    var k_pid: PythonObject
    var k_plan_class: PythonObject
    var k_plans: PythonObject
    var k_pool_optimizer: PythonObject
    var k_positions: PythonObject
    var k_predictor_class: PythonObject
    var k_predictor_layout: PythonObject
    var k_primitives: PythonObject
    var k_profile: PythonObject
    var k_profile_base: PythonObject
    var k_provider: PythonObject
    var k_query_classes: PythonObject
    var k_query_end: PythonObject
    var k_query_rows: PythonObject
    var k_query_rows_per_shard: PythonObject
    var k_query_start: PythonObject
    var k_query_state: PythonObject
    var k_query_tile: PythonObject
    var k_radius: PythonObject
    var k_random_weights_: PythonObject
    var k_rbf_rows: PythonObject
    var k_recipe: PythonObject
    var k_record: PythonObject
    var k_reduction: PythonObject
    var k_reference_bytes: PythonObject
    var k_reference_end: PythonObject
    var k_reference_query: PythonObject
    var k_reference_rows_per_shard: PythonObject
    var k_reference_start: PythonObject
    var k_reference_store: PythonObject
    var k_reference_vote: PythonObject
    var k_release: PythonObject
    var k_return_distance: PythonObject
    var k_rms_norm_forward: PythonObject
    var k_rng: PythonObject
    var k_route: PythonObject
    var k_rows_per_shard: PythonObject
    var k_scale_: PythonObject
    var k_scaler_fit: PythonObject
    var k_scaler_transform: PythonObject
    var k_schema: PythonObject
    var k_seasonal_periods: PythonObject
    var k_selected_vendor: PythonObject
    var k_sha256: PythonObject
    var k_shapes: PythonObject
    var k_shard: PythonObject
    var k_shard_release: PythonObject
    var k_shard_store: PythonObject
    var k_sort_results: PythonObject
    var k_stages: PythonObject
    var k_start: PythonObject
    var k_state: PythonObject
    var k_status: PythonObject
    var k_step: PythonObject
    var k_suffix: PythonObject
    var k_svd_solver: PythonObject
    var k_t: PythonObject
    var k_token: PythonObject
    var k_trace_record: PythonObject
    var k_training_numeric_mode: PythonObject
    var k_training_vendor: PythonObject
    var k_trees_context: PythonObject
    var k_trees_entry: PythonObject
    var k_ts_num: PythonObject
    var k_v: PythonObject
    var k_vendor: PythonObject
    var k_version: PythonObject
    var k_vocab_size: PythonObject
    var k_vocabulary: PythonObject
    var k_waves: PythonObject
    var k_weight_decay: PythonObject
    var k_weight_format: PythonObject
    var k_weight_names: PythonObject
    var k_weights: PythonObject
    var k_weights_: PythonObject
    var k_with_mean: PythonObject
    var k_with_mean_: PythonObject
    var k_with_std: PythonObject
    var k_with_std_: PythonObject
    var k_worker_bindings: PythonObject
    var k_workers: PythonObject
    var k_y_train_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops_byte = PythonObject(None)
        self.ops_mlp = PythonObject(None)
        self.ops_neural = PythonObject(None)
        self.ops_forecast = PythonObject(None)
        self.ops_forecast_worker = PythonObject(None)
        self.ops_shard = PythonObject(None)
        self.ops_shard_worker = PythonObject(None)
        self.ops_numerical = PythonObject(None)
        self.ops_causal = PythonObject(None)
        self.ops_predictor = PythonObject(None)
        self.ops_par_driver = PythonObject(None)
        self.ops_par_owner = PythonObject(None)
        self.k_L_ = PythonObject(None)
        self.k_MOJOLEARN_NATIVE_CV_NUMPY = PythonObject(None)
        self.k_W_sr_ = PythonObject(None)
        self.k_X_train_ = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k__blocks = PythonObject(None)
        self.k__causal_admission_failure = PythonObject(None)
        self.k__causal_initial_buffers = PythonObject(None)
        self.k__causal_live_fields = PythonObject(None)
        self.k__causal_original = PythonObject(None)
        self.k__classes_list = PythonObject(None)
        self.k__closed = PythonObject(None)
        self.k__components_len = PythonObject(None)
        self.k__comps = PythonObject(None)
        self.k__data = PythonObject(None)
        self.k__device_list_data = PythonObject(None)
        self.k__embed = PythonObject(None)
        self.k__exog = PythonObject(None)
        self.k__failed = PythonObject(None)
        self.k__fit_numeric_mode = PythonObject(None)
        self.k__handle = PythonObject(None)
        self.k__head = PythonObject(None)
        self.k__head_int15 = PythonObject(None)
        self.k__id_maps = PythonObject(None)
        self.k__index = PythonObject(None)
        self.k__kernel = PythonObject(None)
        self.k__label_names = PythonObject(None)
        self.k__lifetime = PythonObject(None)
        self.k__lock = PythonObject(None)
        self.k__max_iter_predict = PythonObject(None)
        self.k__metric_arg = PythonObject(None)
        self.k__metric_value = PythonObject(None)
        self.k__model = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__n_trees = PythonObject(None)
        self.k__native = PythonObject(None)
        self.k__native_entries = PythonObject(None)
        self.k__native_numeric_mode = PythonObject(None)
        self.k__native_parallel_block = PythonObject(None)
        self.k__native_parallel_blocks = PythonObject(None)
        self.k__native_parallel_entries_live = PythonObject(None)
        self.k__native_parallel_extra_blocks = PythonObject(None)
        self.k__native_parallel_model = PythonObject(None)
        self.k__native_state_identity = PythonObject(None)
        self.k__native_vendor = PythonObject(None)
        self.k__norm = PythonObject(None)
        self.k__num_outputs = PythonObject(None)
        self.k__offsets = PythonObject(None)
        self.k__pool = PythonObject(None)
        self.k__pools = PythonObject(None)
        self.k__resident = PythonObject(None)
        self.k__resident_holder = PythonObject(None)
        self.k__session = PythonObject(None)
        self.k__shape = PythonObject(None)
        self.k__token = PythonObject(None)
        self.k__transform_mode = PythonObject(None)
        self.k__update_pool = PythonObject(None)
        self.k__w = PythonObject(None)
        self.k__worker_index = PythonObject(None)
        self.k__y = PythonObject(None)
        self.k__y_cols = PythonObject(None)
        self.k__y_ndim = PythonObject(None)
        self.k_accumulation_steps = PythonObject(None)
        self.k_admission = PythonObject(None)
        self.k_algorithm = PythonObject(None)
        self.k_architecture = PythonObject(None)
        self.k_arima_fit_shard = PythonObject(None)
        self.k_arima_predict_shard = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_assignments = PythonObject(None)
        self.k_backend = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_batch_size_ = PythonObject(None)
        self.k_beta1 = PythonObject(None)
        self.k_beta2 = PythonObject(None)
        self.k_block = PythonObject(None)
        self.k_block_kwargs = PythonObject(None)
        self.k_block_options = PythonObject(None)
        self.k_calls = PythonObject(None)
        self.k_capture_classes = PythonObject(None)
        self.k_center_norms_ = PythonObject(None)
        self.k_centers_ = PythonObject(None)
        self.k_check = PythonObject(None)
        self.k_checked_of = PythonObject(None)
        self.k_checkpoint_class = PythonObject(None)
        self.k_checkpoint_context = PythonObject(None)
        self.k_chunked_lm_head_v2 = PythonObject(None)
        self.k_classes = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_clip = PythonObject(None)
        self.k_clip_ = PythonObject(None)
        self.k_closed = PythonObject(None)
        self.k_completed_steps = PythonObject(None)
        self.k_complexity_ = PythonObject(None)
        self.k_config = PythonObject(None)
        self.k_config_class = PythonObject(None)
        self.k_config_context = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_counter = PythonObject(None)
        self.k_cpu_only = PythonObject(None)
        self.k_create = PythonObject(None)
        self.k_criterion_ = PythonObject(None)
        self.k_data_schedule = PythonObject(None)
        self.k_decompositions = PythonObject(None)
        self.k_dense_solvers = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_device_arg = PythonObject(None)
        self.k_devices = PythonObject(None)
        self.k_dims = PythonObject(None)
        self.k_driver = PythonObject(None)
        self.k_dropout = PythonObject(None)
        self.k_end = PythonObject(None)
        self.k_endog = PythonObject(None)
        self.k_entry = PythonObject(None)
        self.k_entry_mode = PythonObject(None)
        self.k_entry_vendor = PythonObject(None)
        self.k_environ = PythonObject(None)
        self.k_environment = PythonObject(None)
        self.k_epoch = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_error = PythonObject(None)
        self.k_estimator = PythonObject(None)
        self.k_estimators_ = PythonObject(None)
        self.k_exception_args = PythonObject(None)
        self.k_executable = PythonObject(None)
        self.k_family = PythonObject(None)
        self.k_family_msel_parallel = PythonObject(None)
        self.k_feature_range = PythonObject(None)
        self.k_feature_range_ = PythonObject(None)
        self.k_fields_get = PythonObject(None)
        self.k_fit_executed_flag = PythonObject(None)
        self.k_flags = PythonObject(None)
        self.k_forest_classifier = PythonObject(None)
        self.k_forest_fit = PythonObject(None)
        self.k_forests = PythonObject(None)
        self.k_full = PythonObject(None)
        self.k_generation_classes = PythonObject(None)
        self.k_gpc_class_fit = PythonObject(None)
        self.k_gpc_class_predict = PythonObject(None)
        self.k_handle = PythonObject(None)
        self.k_head = PythonObject(None)
        self.k_history = PythonObject(None)
        self.k_hooks = PythonObject(None)
        self.k_hw_fit_shard = PythonObject(None)
        self.k_hw_predict_shard = PythonObject(None)
        self.k_identical = PythonObject(None)
        self.k_identities = PythonObject(None)
        self.k_identity = PythonObject(None)
        self.k_index = PythonObject(None)
        self.k_inference_engine = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_integer_types = PythonObject(None)
        self.k_inventory = PythonObject(None)
        self.k_ivf_class = PythonObject(None)
        self.k_ivf_search = PythonObject(None)
        self.k_ivf_store = PythonObject(None)
        self.k_kde_class = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_knc_class = PythonObject(None)
        self.k_knr_class = PythonObject(None)
        self.k_last_run = PythonObject(None)
        self.k_last_shards_ = PythonObject(None)
        self.k_layer = PythonObject(None)
        self.k_layer_devices = PythonObject(None)
        self.k_layers = PythonObject(None)
        self.k_list_data_ = PythonObject(None)
        self.k_list_indices_ = PythonObject(None)
        self.k_list_offsets_ = PythonObject(None)
        self.k_lock_type = PythonObject(None)
        self.k_log_marginal_likelihood_value_ = PythonObject(None)
        self.k_logical_shards = PythonObject(None)
        self.k_losses = PythonObject(None)
        self.k_lr = PythonObject(None)
        self.k_m = PythonObject(None)
        self.k_mamba1_profile = PythonObject(None)
        self.k_max_positions = PythonObject(None)
        self.k_mean_ = PythonObject(None)
        self.k_method_arg = PythonObject(None)
        self.k_metric = PythonObject(None)
        self.k_metric_code_ = PythonObject(None)
        self.k_min_ = PythonObject(None)
        self.k_mlp_slot = PythonObject(None)
        self.k_mode = PythonObject(None)
        self.k_model = PythonObject(None)
        self.k_model_class = PythonObject(None)
        self.k_model_shape = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_mro = PythonObject(None)
        self.k_n = PythonObject(None)
        self.k_n_candidates_ = PythonObject(None)
        self.k_n_classes_ = PythonObject(None)
        self.k_n_estimators = PythonObject(None)
        self.k_n_estimators_ = PythonObject(None)
        self.k_n_exog_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_layers = PythonObject(None)
        self.k_n_lists_ = PythonObject(None)
        self.k_n_neighbors = PythonObject(None)
        self.k_n_obs_ = PythonObject(None)
        self.k_n_probes = PythonObject(None)
        self.k_n_rows_ = PythonObject(None)
        self.k_n_samples_fit_ = PythonObject(None)
        self.k_n_samples_seen_ = PythonObject(None)
        self.k_n_tensors = PythonObject(None)
        self.k_n_total = PythonObject(None)
        self.k_n_vocab = PythonObject(None)
        self.k_names = PythonObject(None)
        self.k_nb_ = PythonObject(None)
        self.k_nearest_class = PythonObject(None)
        self.k_nesterov = PythonObject(None)
        self.k_next_batch_index = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_arg = PythonObject(None)
        self.k_numeric_profile = PythonObject(None)
        self.k_offsets = PythonObject(None)
        self.k_operation = PythonObject(None)
        self.k_optimizer = PythonObject(None)
        self.k_output = PythonObject(None)
        self.k_outputs_2d_ = PythonObject(None)
        self.k_oversampling_factor = PythonObject(None)
        self.k_owner = PythonObject(None)
        self.k_owner_fields = PythonObject(None)
        self.k_p = PythonObject(None)
        self.k_parallel_call = PythonObject(None)
        self.k_parallel_public = PythonObject(None)
        self.k_parameter_names = PythonObject(None)
        self.k_parameter_offsets = PythonObject(None)
        self.k_parameter_order = PythonObject(None)
        self.k_parameter_shapes = PythonObject(None)
        self.k_parameters = PythonObject(None)
        self.k_params_ = PythonObject(None)
        self.k_pca = PythonObject(None)
        self.k_pi_ = PythonObject(None)
        self.k_pid = PythonObject(None)
        self.k_plan_class = PythonObject(None)
        self.k_plans = PythonObject(None)
        self.k_pool_optimizer = PythonObject(None)
        self.k_positions = PythonObject(None)
        self.k_predictor_class = PythonObject(None)
        self.k_predictor_layout = PythonObject(None)
        self.k_primitives = PythonObject(None)
        self.k_profile = PythonObject(None)
        self.k_profile_base = PythonObject(None)
        self.k_provider = PythonObject(None)
        self.k_query_classes = PythonObject(None)
        self.k_query_end = PythonObject(None)
        self.k_query_rows = PythonObject(None)
        self.k_query_rows_per_shard = PythonObject(None)
        self.k_query_start = PythonObject(None)
        self.k_query_state = PythonObject(None)
        self.k_query_tile = PythonObject(None)
        self.k_radius = PythonObject(None)
        self.k_random_weights_ = PythonObject(None)
        self.k_rbf_rows = PythonObject(None)
        self.k_recipe = PythonObject(None)
        self.k_record = PythonObject(None)
        self.k_reduction = PythonObject(None)
        self.k_reference_bytes = PythonObject(None)
        self.k_reference_end = PythonObject(None)
        self.k_reference_query = PythonObject(None)
        self.k_reference_rows_per_shard = PythonObject(None)
        self.k_reference_start = PythonObject(None)
        self.k_reference_store = PythonObject(None)
        self.k_reference_vote = PythonObject(None)
        self.k_release = PythonObject(None)
        self.k_return_distance = PythonObject(None)
        self.k_rms_norm_forward = PythonObject(None)
        self.k_rng = PythonObject(None)
        self.k_route = PythonObject(None)
        self.k_rows_per_shard = PythonObject(None)
        self.k_scale_ = PythonObject(None)
        self.k_scaler_fit = PythonObject(None)
        self.k_scaler_transform = PythonObject(None)
        self.k_schema = PythonObject(None)
        self.k_seasonal_periods = PythonObject(None)
        self.k_selected_vendor = PythonObject(None)
        self.k_sha256 = PythonObject(None)
        self.k_shapes = PythonObject(None)
        self.k_shard = PythonObject(None)
        self.k_shard_release = PythonObject(None)
        self.k_shard_store = PythonObject(None)
        self.k_sort_results = PythonObject(None)
        self.k_stages = PythonObject(None)
        self.k_start = PythonObject(None)
        self.k_state = PythonObject(None)
        self.k_status = PythonObject(None)
        self.k_step = PythonObject(None)
        self.k_suffix = PythonObject(None)
        self.k_svd_solver = PythonObject(None)
        self.k_t = PythonObject(None)
        self.k_token = PythonObject(None)
        self.k_trace_record = PythonObject(None)
        self.k_training_numeric_mode = PythonObject(None)
        self.k_training_vendor = PythonObject(None)
        self.k_trees_context = PythonObject(None)
        self.k_trees_entry = PythonObject(None)
        self.k_ts_num = PythonObject(None)
        self.k_v = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_version = PythonObject(None)
        self.k_vocab_size = PythonObject(None)
        self.k_vocabulary = PythonObject(None)
        self.k_waves = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_weight_format = PythonObject(None)
        self.k_weight_names = PythonObject(None)
        self.k_weights = PythonObject(None)
        self.k_weights_ = PythonObject(None)
        self.k_with_mean = PythonObject(None)
        self.k_with_mean_ = PythonObject(None)
        self.k_with_std = PythonObject(None)
        self.k_with_std_ = PythonObject(None)
        self.k_worker_bindings = PythonObject(None)
        self.k_workers = PythonObject(None)
        self.k_y_train_ = PythonObject(None)


comptime _PARALLEL_KEYS = SharedSlot[ParallelKeys, "MojolearnParallelKeysV1"]


def _parallel_keys_fill(nk: UnsafePointer[ParallelKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_L_ = py_key("L_")
    nk[].k_MOJOLEARN_NATIVE_CV_NUMPY = py_key("MOJOLEARN_NATIVE_CV_NUMPY")
    nk[].k_W_sr_ = py_key("W_sr_")
    nk[].k_X_train_ = py_key("X_train_")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k__blocks = py_key("_blocks")
    nk[].k__causal_admission_failure = py_key("_causal_admission_failure")
    nk[].k__causal_initial_buffers = py_key("_causal_initial_buffers")
    nk[].k__causal_live_fields = py_key("_causal_live_fields")
    nk[].k__causal_original = py_key("_causal_original")
    nk[].k__classes_list = py_key("_classes_list")
    nk[].k__closed = py_key("_closed")
    nk[].k__components_len = py_key("_components_len")
    nk[].k__comps = py_key("_comps")
    nk[].k__data = py_key("_data")
    nk[].k__device_list_data = py_key("_device_list_data")
    nk[].k__embed = py_key("_embed")
    nk[].k__exog = py_key("_exog")
    nk[].k__failed = py_key("_failed")
    nk[].k__fit_numeric_mode = py_key("_fit_numeric_mode")
    nk[].k__handle = py_key("_handle")
    nk[].k__head = py_key("_head")
    nk[].k__head_int15 = py_key("_head_int15")
    nk[].k__id_maps = py_key("_id_maps")
    nk[].k__index = py_key("_index")
    nk[].k__kernel = py_key("_kernel")
    nk[].k__label_names = py_key("_label_names")
    nk[].k__lifetime = py_key("_lifetime")
    nk[].k__lock = py_key("_lock")
    nk[].k__max_iter_predict = py_key("_max_iter_predict")
    nk[].k__metric_arg = py_key("_metric_arg")
    nk[].k__metric_value = py_key("_metric_value")
    nk[].k__model = py_key("_model")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__n_trees = py_key("_n_trees")
    nk[].k__native = py_key("_native")
    nk[].k__native_entries = py_key("_native_entries")
    nk[].k__native_numeric_mode = py_key("_native_numeric_mode")
    nk[].k__native_parallel_block = py_key("_native_parallel_block")
    nk[].k__native_parallel_blocks = py_key("_native_parallel_blocks")
    nk[].k__native_parallel_entries_live = py_key("_native_parallel_entries_live")
    nk[].k__native_parallel_extra_blocks = py_key("_native_parallel_extra_blocks")
    nk[].k__native_parallel_model = py_key("_native_parallel_model")
    nk[].k__native_state_identity = py_key("_native_state_identity")
    nk[].k__native_vendor = py_key("_native_vendor")
    nk[].k__norm = py_key("_norm")
    nk[].k__num_outputs = py_key("_num_outputs")
    nk[].k__offsets = py_key("_offsets")
    nk[].k__pool = py_key("_pool")
    nk[].k__pools = py_key("_pools")
    nk[].k__resident = py_key("_resident")
    nk[].k__resident_holder = py_key("_resident_holder")
    nk[].k__session = py_key("_session")
    nk[].k__shape = py_key("_shape")
    nk[].k__token = py_key("_token")
    nk[].k__transform_mode = py_key("_transform_mode")
    nk[].k__update_pool = py_key("_update_pool")
    nk[].k__w = py_key("_w")
    nk[].k__worker_index = py_key("_worker_index")
    nk[].k__y = py_key("_y")
    nk[].k__y_cols = py_key("_y_cols")
    nk[].k__y_ndim = py_key("_y_ndim")
    nk[].k_accumulation_steps = py_key("accumulation_steps")
    nk[].k_admission = py_key("admission")
    nk[].k_algorithm = py_key("algorithm")
    nk[].k_architecture = py_key("architecture")
    nk[].k_arima_fit_shard = py_key("arima_fit_shard")
    nk[].k_arima_predict_shard = py_key("arima_predict_shard")
    nk[].k_array_class = py_key("array_class")
    nk[].k_assignments = py_key("assignments")
    nk[].k_backend = py_key("backend")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_batch_size_ = py_key("batch_size_")
    nk[].k_beta1 = py_key("beta1")
    nk[].k_beta2 = py_key("beta2")
    nk[].k_block = py_key("block")
    nk[].k_block_kwargs = py_key("block_kwargs")
    nk[].k_block_options = py_key("block_options")
    nk[].k_calls = py_key("calls")
    nk[].k_capture_classes = py_key("capture_classes")
    nk[].k_center_norms_ = py_key("center_norms_")
    nk[].k_centers_ = py_key("centers_")
    nk[].k_check = py_key("check")
    nk[].k_checked_of = py_key("checked_of")
    nk[].k_checkpoint_class = py_key("checkpoint_class")
    nk[].k_checkpoint_context = py_key("checkpoint_context")
    nk[].k_chunked_lm_head_v2 = py_key("chunked_lm_head_v2")
    nk[].k_classes = py_key("classes")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_clip = py_key("clip")
    nk[].k_clip_ = py_key("clip_")
    nk[].k_closed = py_key("closed")
    nk[].k_completed_steps = py_key("completed_steps")
    nk[].k_complexity_ = py_key("complexity_")
    nk[].k_config = py_key("config")
    nk[].k_config_class = py_key("config_class")
    nk[].k_config_context = py_key("config_context")
    nk[].k_copy = py_key("copy")
    nk[].k_counter = py_key("counter")
    nk[].k_cpu_only = py_key("cpu_only")
    nk[].k_create = py_key("create")
    nk[].k_criterion_ = py_key("criterion_")
    nk[].k_data_schedule = py_key("data_schedule")
    nk[].k_decompositions = py_key("decompositions")
    nk[].k_dense_solvers = py_key("dense_solvers")
    nk[].k_device = py_key("device")
    nk[].k_device_arg = py_key("device_arg")
    nk[].k_devices = py_key("devices")
    nk[].k_dims = py_key("dims")
    nk[].k_driver = py_key("driver")
    nk[].k_dropout = py_key("dropout")
    nk[].k_end = py_key("end")
    nk[].k_endog = py_key("endog")
    nk[].k_entry = py_key("entry")
    nk[].k_entry_mode = py_key("entry_mode")
    nk[].k_entry_vendor = py_key("entry_vendor")
    nk[].k_environ = py_key("environ")
    nk[].k_environment = py_key("environment")
    nk[].k_epoch = py_key("epoch")
    nk[].k_eps = py_key("eps")
    nk[].k_error = py_key("error")
    nk[].k_estimator = py_key("estimator")
    nk[].k_estimators_ = py_key("estimators_")
    nk[].k_exception_args = py_key("exception_args")
    nk[].k_executable = py_key("executable")
    nk[].k_family = py_key("family")
    nk[].k_family_msel_parallel = py_key("family_msel_parallel")
    nk[].k_feature_range = py_key("feature_range")
    nk[].k_feature_range_ = py_key("feature_range_")
    nk[].k_fields_get = py_key("fields_get")
    nk[].k_fit_executed_flag = py_key("fit_executed_flag")
    nk[].k_flags = py_key("flags")
    nk[].k_forest_classifier = py_key("forest_classifier")
    nk[].k_forest_fit = py_key("forest_fit")
    nk[].k_forests = py_key("forests")
    nk[].k_full = py_key("full")
    nk[].k_generation_classes = py_key("generation_classes")
    nk[].k_gpc_class_fit = py_key("gpc_class_fit")
    nk[].k_gpc_class_predict = py_key("gpc_class_predict")
    nk[].k_handle = py_key("handle")
    nk[].k_head = py_key("head")
    nk[].k_history = py_key("history")
    nk[].k_hooks = py_key("hooks")
    nk[].k_hw_fit_shard = py_key("hw_fit_shard")
    nk[].k_hw_predict_shard = py_key("hw_predict_shard")
    nk[].k_identical = py_key("identical")
    nk[].k_identities = py_key("identities")
    nk[].k_identity = py_key("identity")
    nk[].k_index = py_key("index")
    nk[].k_inference_engine = py_key("inference_engine")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_integer_types = py_key("integer_types")
    nk[].k_inventory = py_key("inventory")
    nk[].k_ivf_class = py_key("ivf_class")
    nk[].k_ivf_search = py_key("ivf_search")
    nk[].k_ivf_store = py_key("ivf_store")
    nk[].k_kde_class = py_key("kde_class")
    nk[].k_kind = py_key("kind")
    nk[].k_knc_class = py_key("knc_class")
    nk[].k_knr_class = py_key("knr_class")
    nk[].k_last_run = py_key("last_run")
    nk[].k_last_shards_ = py_key("last_shards_")
    nk[].k_layer = py_key("layer")
    nk[].k_layer_devices = py_key("layer_devices")
    nk[].k_layers = py_key("layers")
    nk[].k_list_data_ = py_key("list_data_")
    nk[].k_list_indices_ = py_key("list_indices_")
    nk[].k_list_offsets_ = py_key("list_offsets_")
    nk[].k_lock_type = py_key("lock_type")
    nk[].k_log_marginal_likelihood_value_ = py_key("log_marginal_likelihood_value_")
    nk[].k_logical_shards = py_key("logical_shards")
    nk[].k_losses = py_key("losses")
    nk[].k_lr = py_key("lr")
    nk[].k_m = py_key("m")
    nk[].k_mamba1_profile = py_key("mamba1_profile")
    nk[].k_max_positions = py_key("max_positions")
    nk[].k_mean_ = py_key("mean_")
    nk[].k_method_arg = py_key("method_arg")
    nk[].k_metric = py_key("metric")
    nk[].k_metric_code_ = py_key("metric_code_")
    nk[].k_min_ = py_key("min_")
    nk[].k_mlp_slot = py_key("mlp_slot")
    nk[].k_mode = py_key("mode")
    nk[].k_model = py_key("model")
    nk[].k_model_class = py_key("model_class")
    nk[].k_model_shape = py_key("model_shape")
    nk[].k_modules = py_key("modules")
    nk[].k_mro = py_key("mro")
    nk[].k_n = py_key("n")
    nk[].k_n_candidates_ = py_key("n_candidates_")
    nk[].k_n_classes_ = py_key("n_classes_")
    nk[].k_n_estimators = py_key("n_estimators")
    nk[].k_n_estimators_ = py_key("n_estimators_")
    nk[].k_n_exog_ = py_key("n_exog_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_layers = py_key("n_layers")
    nk[].k_n_lists_ = py_key("n_lists_")
    nk[].k_n_neighbors = py_key("n_neighbors")
    nk[].k_n_obs_ = py_key("n_obs_")
    nk[].k_n_probes = py_key("n_probes")
    nk[].k_n_rows_ = py_key("n_rows_")
    nk[].k_n_samples_fit_ = py_key("n_samples_fit_")
    nk[].k_n_samples_seen_ = py_key("n_samples_seen_")
    nk[].k_n_tensors = py_key("n_tensors")
    nk[].k_n_total = py_key("n_total")
    nk[].k_n_vocab = py_key("n_vocab")
    nk[].k_names = py_key("names")
    nk[].k_nb_ = py_key("nb_")
    nk[].k_nearest_class = py_key("nearest_class")
    nk[].k_nesterov = py_key("nesterov")
    nk[].k_next_batch_index = py_key("next_batch_index")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_arg = py_key("numeric_mode_arg")
    nk[].k_numeric_profile = py_key("numeric_profile")
    nk[].k_offsets = py_key("offsets")
    nk[].k_operation = py_key("operation")
    nk[].k_optimizer = py_key("optimizer")
    nk[].k_output = py_key("output")
    nk[].k_outputs_2d_ = py_key("outputs_2d_")
    nk[].k_oversampling_factor = py_key("oversampling_factor")
    nk[].k_owner = py_key("owner")
    nk[].k_owner_fields = py_key("owner_fields")
    nk[].k_p = py_key("p")
    nk[].k_parallel_call = py_key("parallel_call")
    nk[].k_parallel_public = py_key("parallel_public")
    nk[].k_parameter_names = py_key("parameter_names")
    nk[].k_parameter_offsets = py_key("parameter_offsets")
    nk[].k_parameter_order = py_key("parameter_order")
    nk[].k_parameter_shapes = py_key("parameter_shapes")
    nk[].k_parameters = py_key("parameters")
    nk[].k_params_ = py_key("params_")
    nk[].k_pca = py_key("pca")
    nk[].k_pi_ = py_key("pi_")
    nk[].k_pid = py_key("pid")
    nk[].k_plan_class = py_key("plan_class")
    nk[].k_plans = py_key("plans")
    nk[].k_pool_optimizer = py_key("pool_optimizer")
    nk[].k_positions = py_key("positions")
    nk[].k_predictor_class = py_key("predictor_class")
    nk[].k_predictor_layout = py_key("predictor_layout")
    nk[].k_primitives = py_key("primitives")
    nk[].k_profile = py_key("profile")
    nk[].k_profile_base = py_key("profile_base")
    nk[].k_provider = py_key("provider")
    nk[].k_query_classes = py_key("query_classes")
    nk[].k_query_end = py_key("query_end")
    nk[].k_query_rows = py_key("query_rows")
    nk[].k_query_rows_per_shard = py_key("query_rows_per_shard")
    nk[].k_query_start = py_key("query_start")
    nk[].k_query_state = py_key("query_state")
    nk[].k_query_tile = py_key("query_tile")
    nk[].k_radius = py_key("radius")
    nk[].k_random_weights_ = py_key("random_weights_")
    nk[].k_rbf_rows = py_key("rbf_rows")
    nk[].k_recipe = py_key("recipe")
    nk[].k_record = py_key("record")
    nk[].k_reduction = py_key("reduction")
    nk[].k_reference_bytes = py_key("reference_bytes")
    nk[].k_reference_end = py_key("reference_end")
    nk[].k_reference_query = py_key("reference_query")
    nk[].k_reference_rows_per_shard = py_key("reference_rows_per_shard")
    nk[].k_reference_start = py_key("reference_start")
    nk[].k_reference_store = py_key("reference_store")
    nk[].k_reference_vote = py_key("reference_vote")
    nk[].k_release = py_key("release")
    nk[].k_return_distance = py_key("return_distance")
    nk[].k_rms_norm_forward = py_key("rms_norm_forward")
    nk[].k_rng = py_key("rng")
    nk[].k_route = py_key("route")
    nk[].k_rows_per_shard = py_key("rows_per_shard")
    nk[].k_scale_ = py_key("scale_")
    nk[].k_scaler_fit = py_key("scaler_fit")
    nk[].k_scaler_transform = py_key("scaler_transform")
    nk[].k_schema = py_key("schema")
    nk[].k_seasonal_periods = py_key("seasonal_periods")
    nk[].k_selected_vendor = py_key("selected_vendor")
    nk[].k_sha256 = py_key("sha256")
    nk[].k_shapes = py_key("shapes")
    nk[].k_shard = py_key("shard")
    nk[].k_shard_release = py_key("shard_release")
    nk[].k_shard_store = py_key("shard_store")
    nk[].k_sort_results = py_key("sort_results")
    nk[].k_stages = py_key("stages")
    nk[].k_start = py_key("start")
    nk[].k_state = py_key("state")
    nk[].k_status = py_key("status")
    nk[].k_step = py_key("step")
    nk[].k_suffix = py_key("suffix")
    nk[].k_svd_solver = py_key("svd_solver")
    nk[].k_t = py_key("t")
    nk[].k_token = py_key("token")
    nk[].k_trace_record = py_key("trace_record")
    nk[].k_training_numeric_mode = py_key("training_numeric_mode")
    nk[].k_training_vendor = py_key("training_vendor")
    nk[].k_trees_context = py_key("trees_context")
    nk[].k_trees_entry = py_key("trees_entry")
    nk[].k_ts_num = py_key("ts_num")
    nk[].k_v = py_key("v")
    nk[].k_vendor = py_key("vendor")
    nk[].k_version = py_key("version")
    nk[].k_vocab_size = py_key("vocab_size")
    nk[].k_vocabulary = py_key("vocabulary")
    nk[].k_waves = py_key("waves")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_weight_format = py_key("weight_format")
    nk[].k_weight_names = py_key("weight_names")
    nk[].k_weights = py_key("weights")
    nk[].k_weights_ = py_key("weights_")
    nk[].k_with_mean = py_key("with_mean")
    nk[].k_with_mean_ = py_key("with_mean_")
    nk[].k_with_std = py_key("with_std")
    nk[].k_with_std_ = py_key("with_std_")
    nk[].k_worker_bindings = py_key("worker_bindings")
    nk[].k_workers = py_key("workers")
    nk[].k_y_train_ = py_key("y_train_")
    var ops_byte = Python.dict()
    py_op_add(ops_byte, "initialize", PARALLEL_BYTE_OPN_INITIALIZE)
    py_op_add(ops_byte, "admit_class", PARALLEL_BYTE_OPN_ADMIT_CLASS)
    py_op_add(ops_byte, "init", PARALLEL_BYTE_OPN_INIT)
    py_op_add(ops_byte, "from_checkpoint", PARALLEL_BYTE_OPN_FROM_CHECKPOINT)
    py_op_add(ops_byte, "step", PARALLEL_BYTE_OPN_STEP)
    py_op_add(ops_byte, "enter", PARALLEL_BYTE_OPN_ENTER)
    py_op_add(ops_byte, "admitted_state", PARALLEL_BYTE_OPN_ADMITTED_STATE)
    py_op_add(ops_byte, "close", PARALLEL_BYTE_OPN_CLOSE)
    py_op_add(ops_byte, "open", PARALLEL_BYTE_OPN_OPEN)
    py_op_add(ops_byte, "train_step", PARALLEL_BYTE_OPN_TRAIN_STEP)
    py_op_add(ops_byte, "state", PARALLEL_BYTE_OPN_STATE)
    py_op_add(ops_byte, "export_raw", PARALLEL_BYTE_OPN_EXPORT_RAW)
    py_op_add(ops_byte, "export_gradients", PARALLEL_BYTE_OPN_EXPORT_GRADIENTS)
    py_op_add(ops_byte, "checkpoint", PARALLEL_BYTE_OPN_CHECKPOINT)
    py_op_add(ops_byte, "set_lr", PARALLEL_BYTE_OPN_SET_LR)
    py_op_add(ops_byte, "require_fold", PARALLEL_BYTE_OPN_REQUIRE_FOLD)
    py_op_add(ops_byte, "has_device_fold", PARALLEL_BYTE_OPN_HAS_DEVICE_FOLD)
    py_op_add(ops_byte, "ownership", PARALLEL_BYTE_OPN_OWNERSHIP)
    py_op_add(ops_byte, "model_ownership", PARALLEL_BYTE_OPN_MODEL_OWNERSHIP)
    py_op_add(ops_byte, "shard_gradient", PARALLEL_BYTE_OPN_SHARD_GRADIENT)
    py_op_add(ops_byte, "shard_gradient_fold", PARALLEL_BYTE_OPN_SHARD_GRADIENT_FOLD)
    py_op_add(ops_byte, "apply_gradient", PARALLEL_BYTE_OPN_APPLY_GRADIENT)
    py_op_add(ops_byte, "fold_reset", PARALLEL_BYTE_OPN_FOLD_RESET)
    py_op_add(ops_byte, "fold_add", PARALLEL_BYTE_OPN_FOLD_ADD)
    py_op_add(ops_byte, "fold_export", PARALLEL_BYTE_OPN_FOLD_EXPORT)
    nk[].ops_byte = ops_byte
    var ops_mlp = Python.dict()
    py_op_add(ops_mlp, "initialize", PARALLEL_MLP_OPN_INITIALIZE)
    py_op_add(ops_mlp, "snapshot", PARALLEL_MLP_OPN_SNAPSHOT)
    py_op_add(ops_mlp, "validate", PARALLEL_MLP_OPN_VALIDATE)
    py_op_add(ops_mlp, "from_state", PARALLEL_MLP_OPN_FROM_STATE)
    py_op_add(ops_mlp, "publish", PARALLEL_MLP_OPN_PUBLISH)
    py_op_add(ops_mlp, "gradient", PARALLEL_MLP_OPN_GRADIENT)
    py_op_add(ops_mlp, "ordered_sum", PARALLEL_MLP_OPN_ORDERED_SUM)
    py_op_add(ops_mlp, "apply", PARALLEL_MLP_OPN_APPLY)
    nk[].ops_mlp = ops_mlp
    var ops_neural = Python.dict()
    py_op_add(ops_neural, "initialize", PARALLEL_NEURAL_OPN_INITIALIZE)
    py_op_add(ops_neural, "init", PARALLEL_NEURAL_OPN_INIT)
    py_op_add(ops_neural, "init_with_worker", PARALLEL_NEURAL_OPN_INIT_WITH_WORKER)
    py_op_add(ops_neural, "restore", PARALLEL_NEURAL_OPN_RESTORE)
    py_op_add(ops_neural, "ordered_sum", PARALLEL_NEURAL_OPN_ORDERED_SUM)
    py_op_add(ops_neural, "worker", PARALLEL_NEURAL_OPN_WORKER)
    py_op_add(ops_neural, "train", PARALLEL_NEURAL_OPN_TRAIN)
    py_op_add(ops_neural, "export_gradients", PARALLEL_NEURAL_OPN_EXPORT_GRADIENTS)
    py_op_add(ops_neural, "checkpoint", PARALLEL_NEURAL_OPN_CHECKPOINT)
    py_op_add(ops_neural, "close", PARALLEL_NEURAL_OPN_CLOSE)
    py_op_add(ops_neural, "exit", PARALLEL_NEURAL_OPN_EXIT)
    py_op_add(ops_neural, "enter", PARALLEL_NEURAL_OPN_ENTER)
    nk[].ops_neural = ops_neural
    var ops_forecast = Python.dict()
    py_op_add(ops_forecast, "initialize", PARALLEL_FORECAST_OPN_INITIALIZE)
    py_op_add(ops_forecast, "predict_arima", PARALLEL_FORECAST_OPN_PREDICT_ARIMA)
    py_op_add(ops_forecast, "forecast_arima", PARALLEL_FORECAST_OPN_FORECAST_ARIMA)
    py_op_add(ops_forecast, "predict_exponential_smoothing", PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING)
    py_op_add(ops_forecast, "forecast_exponential_smoothing", PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING)
    py_op_add(ops_forecast, "fit_arima", PARALLEL_FORECAST_OPN_FIT_ARIMA)
    py_op_add(ops_forecast, "fit_exponential_smoothing", PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING)
    py_op_add(ops_forecast, "worker", PARALLEL_FORECAST_OPN_WORKER)
    nk[].ops_forecast = ops_forecast
    var ops_forecast_worker = Python.dict()
    py_op_add(ops_forecast_worker, "arima_predict_shard", PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD)
    py_op_add(ops_forecast_worker, "hw_predict_shard", PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD)
    py_op_add(ops_forecast_worker, "arima_fit_shard", PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD)
    py_op_add(ops_forecast_worker, "hw_fit_shard", PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD)
    py_op_add(ops_forecast_worker, "identity", PARALLEL_FORECAST_WORKER_OPN_IDENTITY)
    nk[].ops_forecast_worker = ops_forecast_worker
    var ops_shard = Python.dict()
    py_op_add(ops_shard, "register", PARALLEL_SHARD_OPN_REGISTER)
    py_op_add(ops_shard, "transform_rbf_sampler", PARALLEL_SHARD_OPN_TRANSFORM_RBF_SAMPLER)
    py_op_add(ops_shard, "fit_scaler", PARALLEL_SHARD_OPN_FIT_SCALER)
    py_op_add(ops_shard, "transform_scaler", PARALLEL_SHARD_OPN_TRANSFORM_SCALER)
    py_op_add(ops_shard, "fit_gaussian_process_classifier", PARALLEL_SHARD_OPN_FIT_GAUSSIAN_PROCESS_CLASSIFIER)
    py_op_add(ops_shard, "predict_gaussian_process_classifier", PARALLEL_SHARD_OPN_PREDICT_GAUSSIAN_PROCESS_CLASSIFIER)
    py_op_add(ops_shard, "fit_forest", PARALLEL_SHARD_OPN_FIT_FOREST)
    py_op_add(ops_shard, "trace_take", PARALLEL_SHARD_OPN_TRACE_TAKE)
    py_op_add(ops_shard, "faults_take", PARALLEL_SHARD_OPN_FAULTS_TAKE)
    py_op_add(ops_shard, "faults_arm", PARALLEL_SHARD_OPN_FAULTS_ARM)
    py_op_add(ops_shard, "sabotage_arm", PARALLEL_SHARD_OPN_SABOTAGE_ARM)
    py_op_add(ops_shard, "reference_init", PARALLEL_SHARD_OPN_REFERENCE_INIT)
    py_op_add(ops_shard, "queries_init", PARALLEL_SHARD_OPN_QUERIES_INIT)
    py_op_add(ops_shard, "holder_enter", PARALLEL_SHARD_OPN_HOLDER_ENTER)
    py_op_add(ops_shard, "holder_close", PARALLEL_SHARD_OPN_HOLDER_CLOSE)
    py_op_add(ops_shard, "ivf_init", PARALLEL_SHARD_OPN_IVF_INIT)
    py_op_add(ops_shard, "ivf_enter", PARALLEL_SHARD_OPN_IVF_ENTER)
    py_op_add(ops_shard, "reference_query", PARALLEL_SHARD_OPN_REFERENCE_QUERY)
    py_op_add(ops_shard, "parallel_queries", PARALLEL_SHARD_OPN_PARALLEL_QUERIES)
    py_op_add(ops_shard, "ivf_from_index", PARALLEL_SHARD_OPN_IVF_FROM_INDEX)
    py_op_add(ops_shard, "ivf_search", PARALLEL_SHARD_OPN_IVF_SEARCH)
    py_op_add(ops_shard, "ivf_close", PARALLEL_SHARD_OPN_IVF_CLOSE)
    py_op_add(ops_shard, "worker", PARALLEL_SHARD_OPN_WORKER)
    nk[].ops_shard = ops_shard
    var ops_shard_worker = Python.dict()
    py_op_add(ops_shard_worker, "identity", PARALLEL_SHARD_WORKER_OPN_IDENTITY)
    py_op_add(ops_shard_worker, "shard_store", PARALLEL_SHARD_WORKER_OPN_SHARD_STORE)
    py_op_add(ops_shard_worker, "shard_release", PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE)
    py_op_add(ops_shard_worker, "rbf_rows", PARALLEL_SHARD_WORKER_OPN_RBF_ROWS)
    py_op_add(ops_shard_worker, "scaler_fit", PARALLEL_SHARD_WORKER_OPN_SCALER_FIT)
    py_op_add(ops_shard_worker, "scaler_transform", PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM)
    py_op_add(ops_shard_worker, "gpc_class_fit", PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT)
    py_op_add(ops_shard_worker, "gpc_class_predict", PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT)
    py_op_add(ops_shard_worker, "forest_fit", PARALLEL_SHARD_WORKER_OPN_FOREST_FIT)
    py_op_add(ops_shard_worker, "reference_shard", PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD)
    py_op_add(ops_shard_worker, "reference_store", PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE)
    py_op_add(ops_shard_worker, "reference_vote", PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE)
    py_op_add(ops_shard_worker, "query_state", PARALLEL_SHARD_WORKER_OPN_QUERY_STATE)
    py_op_add(ops_shard_worker, "release", PARALLEL_SHARD_WORKER_OPN_RELEASE)
    py_op_add(ops_shard_worker, "ivf_store", PARALLEL_SHARD_WORKER_OPN_IVF_STORE)
    py_op_add(ops_shard_worker, "ivf_search", PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH)
    py_op_add(ops_shard_worker, "reference_query", PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY)
    py_op_add(ops_shard_worker, "query_rows", PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS)
    nk[].ops_shard_worker = ops_shard_worker
    var ops_numerical = Python.dict()
    py_op_add(ops_numerical, "py", PARALLEL_NUMERICAL_OPN_PY)
    py_op_add(ops_numerical, "arima_predict_shard", PARALLEL_NUMERICAL_OPN_ARIMA_PREDICT_SHARD)
    py_op_add(ops_numerical, "hw_predict_shard", PARALLEL_NUMERICAL_OPN_HW_PREDICT_SHARD)
    py_op_add(ops_numerical, "gpc_class_fit", PARALLEL_NUMERICAL_OPN_GPC_CLASS_FIT)
    py_op_add(ops_numerical, "gpc_class_predict", PARALLEL_NUMERICAL_OPN_GPC_CLASS_PREDICT)
    py_op_add(ops_numerical, "ivf_search", PARALLEL_NUMERICAL_OPN_IVF_SEARCH)
    py_op_add(ops_numerical, "rbf_rows", PARALLEL_NUMERICAL_OPN_RBF_ROWS)
    py_op_add(ops_numerical, "scaler_fit", PARALLEL_NUMERICAL_OPN_SCALER_FIT)
    py_op_add(ops_numerical, "scaler_transform", PARALLEL_NUMERICAL_OPN_SCALER_TRANSFORM)
    py_op_add(ops_numerical, "forest_fit", PARALLEL_NUMERICAL_OPN_FOREST_FIT)
    py_op_add(ops_numerical, "arima_fit_shard", PARALLEL_NUMERICAL_OPN_ARIMA_FIT_SHARD)
    py_op_add(ops_numerical, "hw_fit_shard", PARALLEL_NUMERICAL_OPN_HW_FIT_SHARD)
    py_op_add(ops_numerical, "reference_shard", PARALLEL_NUMERICAL_OPN_REFERENCE_SHARD)
    py_op_add(ops_numerical, "reference_query", PARALLEL_NUMERICAL_OPN_REFERENCE_QUERY)
    py_op_add(ops_numerical, "query_rows", PARALLEL_NUMERICAL_OPN_QUERY_ROWS)
    nk[].ops_numerical = ops_numerical
    var ops_causal = Python.dict()
    py_op_add(ops_causal, "initialize", PARALLEL_CAUSAL_OPN_INITIALIZE)
    py_op_add(ops_causal, "worker", PARALLEL_CAUSAL_OPN_WORKER)
    py_op_add(ops_causal, "init", PARALLEL_CAUSAL_OPN_INIT)
    py_op_add(ops_causal, "load", PARALLEL_CAUSAL_OPN_LOAD)
    py_op_add(ops_causal, "load_with_worker", PARALLEL_CAUSAL_OPN_LOAD_WITH_WORKER)
    py_op_add(ops_causal, "rpc", PARALLEL_CAUSAL_OPN_RPC)
    py_op_add(ops_causal, "capture_model", PARALLEL_CAUSAL_OPN_CAPTURE_MODEL)
    py_op_add(ops_causal, "capture_state", PARALLEL_CAUSAL_OPN_CAPTURE_STATE)
    py_op_add(ops_causal, "parameters", PARALLEL_CAUSAL_OPN_PARAMETERS)
    py_op_add(ops_causal, "block_fields", PARALLEL_CAUSAL_OPN_BLOCK_FIELDS)
    py_op_add(ops_causal, "allocate_layer", PARALLEL_CAUSAL_OPN_ALLOCATE_LAYER)
    py_op_add(ops_causal, "block_allocate", PARALLEL_CAUSAL_OPN_BLOCK_ALLOCATE)
    py_op_add(ops_causal, "block_run", PARALLEL_CAUSAL_OPN_BLOCK_RUN)
    py_op_add(ops_causal, "block_call", PARALLEL_CAUSAL_OPN_BLOCK_CALL)
    py_op_add(ops_causal, "block_init", PARALLEL_CAUSAL_OPN_BLOCK_INIT)
    py_op_add(ops_causal, "state_init", PARALLEL_CAUSAL_OPN_STATE_INIT)
    py_op_add(ops_causal, "state_snapshot", PARALLEL_CAUSAL_OPN_STATE_SNAPSHOT)
    py_op_add(ops_causal, "state_publish", PARALLEL_CAUSAL_OPN_STATE_PUBLISH)
    py_op_add(ops_causal, "state_close", PARALLEL_CAUSAL_OPN_STATE_CLOSE)
    py_op_add(ops_causal, "release_layers", PARALLEL_CAUSAL_OPN_RELEASE_LAYERS)
    py_op_add(ops_causal, "run", PARALLEL_CAUSAL_OPN_RUN)
    py_op_add(ops_causal, "close", PARALLEL_CAUSAL_OPN_CLOSE)
    py_op_add(ops_causal, "close_quiet", PARALLEL_CAUSAL_OPN_CLOSE_QUIET)
    py_op_add(ops_causal, "exit", PARALLEL_CAUSAL_OPN_EXIT)
    py_op_add(ops_causal, "enter", PARALLEL_CAUSAL_OPN_ENTER)
    py_op_add(ops_causal, "primitives_init", PARALLEL_CAUSAL_OPN_PRIMITIVES_INIT)
    py_op_add(ops_causal, "primitive", PARALLEL_CAUSAL_OPN_PRIMITIVE)
    nk[].ops_causal = ops_causal
    var ops_predictor = Python.dict()
    py_op_add(ops_predictor, "predictor_init", PARALLEL_PREDICTOR_OPN_PREDICTOR_INIT)
    py_op_add(ops_predictor, "predictor_close", PARALLEL_PREDICTOR_OPN_PREDICTOR_CLOSE)
    py_op_add(ops_predictor, "predictor_exit", PARALLEL_PREDICTOR_OPN_PREDICTOR_EXIT)
    py_op_add(ops_predictor, "predictor_enter", PARALLEL_PREDICTOR_OPN_PREDICTOR_ENTER)
    py_op_add(ops_predictor, "predictor_predict_proba", PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT_PROBA)
    py_op_add(ops_predictor, "predictor_predict", PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT)
    nk[].ops_predictor = ops_predictor
    var ops_par_driver = Python.dict()
    for i in range(PARALLEL_DRIVER_COUNT):  # small-loop(fill: the driver specs, once per process)
        py_op_add(ops_par_driver, String(par_spec_driver_name(i)), i)
    nk[].ops_par_driver = ops_par_driver
    var ops_par_owner = Python.dict()
    for i in range(PARALLEL_OWNER_COUNT):  # small-loop(fill: the driver specs' owner operations, once per process)
        py_op_add(ops_par_owner, String(par_spec_owner_name(i)), i)
    nk[].ops_par_owner = ops_par_owner
    nk[].ready = True


@always_inline
def parallel_keys() raises -> UnsafePointer[ParallelKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PARALLEL_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _parallel_keys_fill(nk)
    return nk
