# SPDX-License-Identifier: Apache-2.0
# The optimizer family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add
from bindings.native_loaded_module_door import native_interpreter_finalizing

# bw2-optim-training (2026-11-20): `optimizer_native_c` resolves the
# operation by one lookup in `ops`. The codes are numbered by the owner that
# receives them, so the router's old prefix tests (`seqdev_`, `emblayer_`,
# `seqln_`/`seqmoe_`, `seq_`, `device_`, `schedule_`) are the range tests of
# `optimizer_route` (bindings/optimizer_native_router_door.mojo). A name that
# is not here (-1) is routed by its prefix exactly as before.

comptime OPTIMIZER_OPN_TRAINING_NUMERIC_MODE = 0
comptime OPTIMIZER_OPN_TRAINING_VENDOR = 1
comptime OPTIMIZER_OPN_SEQDEV_FINALIZE = 2
comptime OPTIMIZER_OPN_SEQDEV_ARRAY = 3
comptime OPTIMIZER_OPN_SEQDEV_CONSTRUCT = 4
comptime OPTIMIZER_OPN_SEQDEV_COPY = 5
comptime OPTIMIZER_OPN_SEQDEV_HOST = 6
comptime OPTIMIZER_OPN_SEQDEV_NDIM = 7
comptime OPTIMIZER_OPN_SEQDEV_NEW = 8
comptime OPTIMIZER_OPN_SEQDEV_RESHAPE = 9
comptime OPTIMIZER_OPN_SEQDEV_SIZE = 10
comptime OPTIMIZER_OPN_SEQDEV_SIZE_OF = 11
comptime OPTIMIZER_OPN_SEQDEV_TO = 12
comptime OPTIMIZER_OPN_SEQDEV_UPLOAD = 13
comptime OPTIMIZER_OPN_SEQDEV_WRAP = 14
comptime OPTIMIZER_OPN_RESIDUAL_DROPOUT = 15
comptime OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD = 16
comptime OPTIMIZER_OPN_EMBLAYER_INIT = 17
comptime OPTIMIZER_OPN_EMBLAYER_PRETRAINED = 18
comptime OPTIMIZER_OPN_EMBLAYER_SET_PARAMS = 19
comptime OPTIMIZER_OPN_EMBLAYER_STATE = 20
comptime OPTIMIZER_OPN_EMBLAYER_TO_DEVICE = 21
comptime OPTIMIZER_OPN_EMBLAYER_FORWARD = 22
comptime OPTIMIZER_OPN_EMBLAYER_BACKWARD = 23
comptime OPTIMIZER_OPN_EMBLAYER_LOAD = 24
comptime OPTIMIZER_OPN_SEQLN_INITIALIZE = 25
comptime OPTIMIZER_OPN_SEQLN_FORWARD = 26
comptime OPTIMIZER_OPN_SEQLN_BACKWARD = 27
comptime OPTIMIZER_OPN_SEQLN_DEVICE_IO = 28
comptime OPTIMIZER_OPN_SEQLN_TO_DEVICE = 29
comptime OPTIMIZER_OPN_SEQLN_LAYER = 30
comptime OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD = 31
comptime OPTIMIZER_OPN_SEQMOE_INITIALIZE = 32
comptime OPTIMIZER_OPN_SEQMOE_FORWARD = 33
comptime OPTIMIZER_OPN_SEQMOE_BACKWARD = 34
comptime OPTIMIZER_OPN_SEQMOE_LOAD = 35
comptime OPTIMIZER_OPN_SEQMOE_RELEASE = 36
comptime OPTIMIZER_OPN_SEQMOE_ROUTING = 37
comptime OPTIMIZER_OPN_SEQMOE_SELECTED = 38
comptime OPTIMIZER_OPN_SEQMOE_SETATTR = 39
comptime OPTIMIZER_OPN_SEQMOE_TO_DEVICE = 40
comptime OPTIMIZER_OPN_SEQ_CONSTRUCT = 41
comptime OPTIMIZER_OPN_SEQ_STEP = 42
comptime OPTIMIZER_OPN_SEQ_STATE = 43
comptime OPTIMIZER_OPN_SEQ_STATE_SET = 44
comptime OPTIMIZER_OPN_SEQ_WORD = 45
comptime OPTIMIZER_OPN_SEQ_WORD_SET = 46
comptime OPTIMIZER_OPN_SEQ_SLOT = 47
comptime OPTIMIZER_OPN_SEQ_HANDLE = 48
comptime OPTIMIZER_OPN_SEQ_BINDING = 49
comptime OPTIMIZER_OPN_SEQ_RESIDENT = 50
comptime OPTIMIZER_OPN_SEQ_EXPORT = 51
comptime OPTIMIZER_OPN_SEQ_IMPORT = 52
comptime OPTIMIZER_OPN_SEQ_FINALIZE = 53
comptime OPTIMIZER_OPN_DEVICE_CONSTRUCT = 54
comptime OPTIMIZER_OPN_DEVICE_NEW = 55
comptime OPTIMIZER_OPN_DEVICE_UPLOAD = 56
comptime OPTIMIZER_OPN_DEVICE_RESHAPE = 57
comptime OPTIMIZER_OPN_DEVICE_HOST = 58
comptime OPTIMIZER_OPN_SCHEDULE_CONSTRUCT = 59
comptime OPTIMIZER_OPN_SCHEDULE_FROM_CONFIG = 60
comptime OPTIMIZER_OPN_SCHEDULE_GETSTATE = 61
comptime OPTIMIZER_OPN_SCHEDULE_SETATTR = 62
comptime OPTIMIZER_OPN_SCHEDULE_DELATTR = 63
comptime OPTIMIZER_OPN_SCHEDULE_SETSTATE = 64
comptime OPTIMIZER_OPN_SCHEDULE_SPEC = 65
comptime OPTIMIZER_OPN_SCHEDULE_CONFIG = 66
comptime OPTIMIZER_OPN_SCHEDULE_VALUES = 67
comptime OPTIMIZER_OPN_SCHEDULE_LIST = 68
comptime OPTIMIZER_OPN_SCHEDULE_AT = 69
comptime OPTIMIZER_OPN_SCHEDULE_BITS = 70
comptime OPTIMIZER_OPN_RESTORE_OWNER = 71
comptime OPTIMIZER_OPN_CONSTRUCT = 72
comptime OPTIMIZER_OPN_GET_PARAMS = 73
comptime OPTIMIZER_OPN_SET_PARAMS = 74
comptime OPTIMIZER_OPN_EXPORT_OWNER = 75
comptime OPTIMIZER_OPN_LINEAR_FORWARD = 76
comptime OPTIMIZER_OPN_LINEAR_BACKWARD = 77
comptime OPTIMIZER_OPN_RMS_NORM_FORWARD = 78
comptime OPTIMIZER_OPN_RMS_NORM_BACKWARD = 79
comptime OPTIMIZER_OPN_EMBEDDING_FORWARD = 80
comptime OPTIMIZER_OPN_EMBEDDING_BACKWARD = 81
comptime OPTIMIZER_OPN_CROSS_ENTROPY = 82
comptime OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS = 83
comptime OPTIMIZER_OPN_SAMBA_HEAD_LOSS = 84
comptime OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS = 85
comptime OPTIMIZER_OPN_BINDING = 86
comptime OPTIMIZER_OPN_RAW_IO = 87
comptime OPTIMIZER_OPN_PARAM_DEVICE = 88
comptime OPTIMIZER_OPN_STEP = 89
comptime OPTIMIZER_OPN_STATE = 90
comptime OPTIMIZER_OPN_STATE_SET = 91
comptime OPTIMIZER_OPN_LOAD = 92
comptime OPTIMIZER_OPN_ZERO = 93
comptime OPTIMIZER_OPN_RESIDENT = 94
comptime OPTIMIZER_OPN_IO = 95
comptime OPTIMIZER_OPN_CLIP = 96
comptime OPTIMIZER_OPN_ACCUMULATE = 97
comptime OPTIMIZER_OPN_ALIGNED = 98


struct OptimizerKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_ComplexWarning: PythonObject
    var k__CPU_ONLY: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__OPT_CONTEXT: PythonObject
    var k__addr: PythonObject
    var k__b: PythonObject
    var k__base: PythonObject
    var k__m: PythonObject
    var k__mv: PythonObject
    var k__native_optimizer_owner: PythonObject
    var k__native_optimizer_pristine: PythonObject
    var k__pristine: PythonObject
    var k__readonly: PythonObject
    var k__resident_wanted: PythonObject
    var k__step: PythonObject
    var k__v: PythonObject
    var k_accumulation_steps: PythonObject
    var k_anneal_strategy: PythonObject
    var k_array_class: PythonObject
    var k_array_getters: PythonObject
    var k_array_type: PythonObject
    var k_backend: PythonObject
    var k_base_lr: PythonObject
    var k_betas: PythonObject
    var k_bindings: PythonObject
    var k_bits_at: PythonObject
    var k_buf_initialized: PythonObject
    var k_buffer_types: PythonObject
    var k_ce_loss: PythonObject
    var k_ce_loss_dev: PythonObject
    var k_chunked_lm_head_v2_loss: PythonObject
    var k_chunked_lm_head_v2_train: PythonObject
    var k_container_types: PythonObject
    var k_cross_entropy: PythonObject
    var k_dampening: PythonObject
    var k_device_class: PythonObject
    var k_device_fields_getter: PythonObject
    var k_device_finalizer_types: PythonObject
    var k_div_factor: PythonObject
    var k_dtype: PythonObject
    var k_embedding_index_cast: PythonObject
    var k_eps: PythonObject
    var k_exp_avg: PythonObject
    var k_exp_avg_sq: PythonObject
    var k_final_div_factor: PythonObject
    var k_gamma: PythonObject
    var k_guards: PythonObject
    var k_h: PythonObject
    var k_input_context: PythonObject
    var k_itemsize: PythonObject
    var k_kind: PythonObject
    var k_linear_backward_dev: PythonObject
    var k_linear_forward_dev: PythonObject
    var k_lr: PythonObject
    var k_lr_at: PythonObject
    var k_lr_schedule: PythonObject
    var k_mapping_types: PythonObject
    var k_max_lr: PythonObject
    var k_maximize: PythonObject
    var k_min_lr: PythonObject
    var k_mlp_sessions: PythonObject
    var k_module: PythonObject
    var k_module_type: PythonObject
    var k_modules: PythonObject
    var k_momentum: PythonObject
    var k_n_total: PythonObject
    var k_nbytes: PythonObject
    var k_ndarray: PythonObject
    var k_ndim: PythonObject
    var k_nesterov: PythonObject
    var k_numeric_mode: PythonObject
    var k_numpy: PythonObject
    var k_offsets: PythonObject
    var k_optimizer_native_api_v1: PythonObject
    var k_order: PythonObject
    var k_owners: PythonObject
    var k_params: PythonObject
    var k_pct_start: PythonObject
    var k_peak_lr: PythonObject
    var k_profile_defaults: PythonObject
    var k_profiles: PythonObject
    var k_residual_dropout: PythonObject
    var k_residual_dropout_backward: PythonObject
    var k_residual_dropout_backward_dev: PythonObject
    var k_residual_dropout_dev: PythonObject
    var k_scalar_types: PythonObject
    var k_schedules: PythonObject
    var k_shape: PythonObject
    var k_size: PythonObject
    var k_step_size: PythonObject
    var k_strides: PythonObject
    var k_t: PythonObject
    var k_three_phase: PythonObject
    var k_total_steps: PythonObject
    var k_training: PythonObject
    var k_training_metadata_bindings: PythonObject
    var k_training_metadata_lower: PythonObject
    var k_training_metadata_strip: PythonObject
    var k_training_metadata_types: PythonObject
    var k_training_scalar_locations: PythonObject
    var k_training_scalar_warnings: PythonObject
    var k_type_mro: PythonObject
    var k_type_name: PythonObject
    var k_warmup_steps: PythonObject
    var k_weight_decay: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_ComplexWarning = PythonObject(None)
        self.k__CPU_ONLY = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__OPT_CONTEXT = PythonObject(None)
        self.k__addr = PythonObject(None)
        self.k__b = PythonObject(None)
        self.k__base = PythonObject(None)
        self.k__m = PythonObject(None)
        self.k__mv = PythonObject(None)
        self.k__native_optimizer_owner = PythonObject(None)
        self.k__native_optimizer_pristine = PythonObject(None)
        self.k__pristine = PythonObject(None)
        self.k__readonly = PythonObject(None)
        self.k__resident_wanted = PythonObject(None)
        self.k__step = PythonObject(None)
        self.k__v = PythonObject(None)
        self.k_accumulation_steps = PythonObject(None)
        self.k_anneal_strategy = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_getters = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend = PythonObject(None)
        self.k_base_lr = PythonObject(None)
        self.k_betas = PythonObject(None)
        self.k_bindings = PythonObject(None)
        self.k_bits_at = PythonObject(None)
        self.k_buf_initialized = PythonObject(None)
        self.k_buffer_types = PythonObject(None)
        self.k_ce_loss = PythonObject(None)
        self.k_ce_loss_dev = PythonObject(None)
        self.k_chunked_lm_head_v2_loss = PythonObject(None)
        self.k_chunked_lm_head_v2_train = PythonObject(None)
        self.k_container_types = PythonObject(None)
        self.k_cross_entropy = PythonObject(None)
        self.k_dampening = PythonObject(None)
        self.k_device_class = PythonObject(None)
        self.k_device_fields_getter = PythonObject(None)
        self.k_device_finalizer_types = PythonObject(None)
        self.k_div_factor = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_embedding_index_cast = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_exp_avg = PythonObject(None)
        self.k_exp_avg_sq = PythonObject(None)
        self.k_final_div_factor = PythonObject(None)
        self.k_gamma = PythonObject(None)
        self.k_guards = PythonObject(None)
        self.k_h = PythonObject(None)
        self.k_input_context = PythonObject(None)
        self.k_itemsize = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_linear_backward_dev = PythonObject(None)
        self.k_linear_forward_dev = PythonObject(None)
        self.k_lr = PythonObject(None)
        self.k_lr_at = PythonObject(None)
        self.k_lr_schedule = PythonObject(None)
        self.k_mapping_types = PythonObject(None)
        self.k_max_lr = PythonObject(None)
        self.k_maximize = PythonObject(None)
        self.k_min_lr = PythonObject(None)
        self.k_mlp_sessions = PythonObject(None)
        self.k_module = PythonObject(None)
        self.k_module_type = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_momentum = PythonObject(None)
        self.k_n_total = PythonObject(None)
        self.k_nbytes = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_ndim = PythonObject(None)
        self.k_nesterov = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_offsets = PythonObject(None)
        self.k_optimizer_native_api_v1 = PythonObject(None)
        self.k_order = PythonObject(None)
        self.k_owners = PythonObject(None)
        self.k_params = PythonObject(None)
        self.k_pct_start = PythonObject(None)
        self.k_peak_lr = PythonObject(None)
        self.k_profile_defaults = PythonObject(None)
        self.k_profiles = PythonObject(None)
        self.k_residual_dropout = PythonObject(None)
        self.k_residual_dropout_backward = PythonObject(None)
        self.k_residual_dropout_backward_dev = PythonObject(None)
        self.k_residual_dropout_dev = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_schedules = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_size = PythonObject(None)
        self.k_step_size = PythonObject(None)
        self.k_strides = PythonObject(None)
        self.k_t = PythonObject(None)
        self.k_three_phase = PythonObject(None)
        self.k_total_steps = PythonObject(None)
        self.k_training = PythonObject(None)
        self.k_training_metadata_bindings = PythonObject(None)
        self.k_training_metadata_lower = PythonObject(None)
        self.k_training_metadata_strip = PythonObject(None)
        self.k_training_metadata_types = PythonObject(None)
        self.k_training_scalar_locations = PythonObject(None)
        self.k_training_scalar_warnings = PythonObject(None)
        self.k_type_mro = PythonObject(None)
        self.k_type_name = PythonObject(None)
        self.k_warmup_steps = PythonObject(None)
        self.k_weight_decay = PythonObject(None)


comptime _OPTIMIZER_KEYS = SharedSlot[OptimizerKeys, "MojolearnOptimizerKeysV1"]


def _optimizer_keys_fill(nk: UnsafePointer[OptimizerKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_ComplexWarning = py_key("ComplexWarning")
    nk[].k__CPU_ONLY = py_key("_CPU_ONLY")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__OPT_CONTEXT = py_key("_OPT_CONTEXT")
    nk[].k__addr = py_key("_addr")
    nk[].k__b = py_key("_b")
    nk[].k__base = py_key("_base")
    nk[].k__m = py_key("_m")
    nk[].k__mv = py_key("_mv")
    nk[].k__native_optimizer_owner = py_key("_native_optimizer_owner")
    nk[].k__native_optimizer_pristine = py_key("_native_optimizer_pristine")
    nk[].k__pristine = py_key("_pristine")
    nk[].k__readonly = py_key("_readonly")
    nk[].k__resident_wanted = py_key("_resident_wanted")
    nk[].k__step = py_key("_step")
    nk[].k__v = py_key("_v")
    nk[].k_accumulation_steps = py_key("accumulation_steps")
    nk[].k_anneal_strategy = py_key("anneal_strategy")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_getters = py_key("array_getters")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend = py_key("backend")
    nk[].k_base_lr = py_key("base_lr")
    nk[].k_betas = py_key("betas")
    nk[].k_bindings = py_key("bindings")
    nk[].k_bits_at = py_key("bits_at")
    nk[].k_buf_initialized = py_key("buf_initialized")
    nk[].k_buffer_types = py_key("buffer_types")
    nk[].k_ce_loss = py_key("ce_loss")
    nk[].k_ce_loss_dev = py_key("ce_loss_dev")
    nk[].k_chunked_lm_head_v2_loss = py_key("chunked_lm_head_v2_loss")
    nk[].k_chunked_lm_head_v2_train = py_key("chunked_lm_head_v2_train")
    nk[].k_container_types = py_key("container_types")
    nk[].k_cross_entropy = py_key("cross_entropy")
    nk[].k_dampening = py_key("dampening")
    nk[].k_device_class = py_key("device_class")
    nk[].k_device_fields_getter = py_key("device_fields_getter")
    nk[].k_device_finalizer_types = py_key("device_finalizer_types")
    nk[].k_div_factor = py_key("div_factor")
    nk[].k_dtype = py_key("dtype")
    nk[].k_embedding_index_cast = py_key("embedding_index_cast")
    nk[].k_eps = py_key("eps")
    nk[].k_exp_avg = py_key("exp_avg")
    nk[].k_exp_avg_sq = py_key("exp_avg_sq")
    nk[].k_final_div_factor = py_key("final_div_factor")
    nk[].k_gamma = py_key("gamma")
    nk[].k_guards = py_key("guards")
    nk[].k_h = py_key("h")
    nk[].k_input_context = py_key("input_context")
    nk[].k_itemsize = py_key("itemsize")
    nk[].k_kind = py_key("kind")
    nk[].k_linear_backward_dev = py_key("linear_backward_dev")
    nk[].k_linear_forward_dev = py_key("linear_forward_dev")
    nk[].k_lr = py_key("lr")
    nk[].k_lr_at = py_key("lr_at")
    nk[].k_lr_schedule = py_key("lr_schedule")
    nk[].k_mapping_types = py_key("mapping_types")
    nk[].k_max_lr = py_key("max_lr")
    nk[].k_maximize = py_key("maximize")
    nk[].k_min_lr = py_key("min_lr")
    nk[].k_mlp_sessions = py_key("mlp_sessions")
    nk[].k_module = py_key("module")
    nk[].k_module_type = py_key("module_type")
    nk[].k_modules = py_key("modules")
    nk[].k_momentum = py_key("momentum")
    nk[].k_n_total = py_key("n_total")
    nk[].k_nbytes = py_key("nbytes")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_ndim = py_key("ndim")
    nk[].k_nesterov = py_key("nesterov")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numpy = py_key("numpy")
    nk[].k_offsets = py_key("offsets")
    nk[].k_optimizer_native_api_v1 = py_key("optimizer_native_api_v1")
    nk[].k_order = py_key("order")
    nk[].k_owners = py_key("owners")
    nk[].k_params = py_key("params")
    nk[].k_pct_start = py_key("pct_start")
    nk[].k_peak_lr = py_key("peak_lr")
    nk[].k_profile_defaults = py_key("profile_defaults")
    nk[].k_profiles = py_key("profiles")
    nk[].k_residual_dropout = py_key("residual_dropout")
    nk[].k_residual_dropout_backward = py_key("residual_dropout_backward")
    nk[].k_residual_dropout_backward_dev = py_key("residual_dropout_backward_dev")
    nk[].k_residual_dropout_dev = py_key("residual_dropout_dev")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_schedules = py_key("schedules")
    nk[].k_shape = py_key("shape")
    nk[].k_size = py_key("size")
    nk[].k_step_size = py_key("step_size")
    nk[].k_strides = py_key("strides")
    nk[].k_t = py_key("t")
    nk[].k_three_phase = py_key("three_phase")
    nk[].k_total_steps = py_key("total_steps")
    nk[].k_training = py_key("training")
    nk[].k_training_metadata_bindings = py_key("training_metadata_bindings")
    nk[].k_training_metadata_lower = py_key("training_metadata_lower")
    nk[].k_training_metadata_strip = py_key("training_metadata_strip")
    nk[].k_training_metadata_types = py_key("training_metadata_types")
    nk[].k_training_scalar_locations = py_key("training_scalar_locations")
    nk[].k_training_scalar_warnings = py_key("training_scalar_warnings")
    nk[].k_type_mro = py_key("type_mro")
    nk[].k_type_name = py_key("type_name")
    nk[].k_warmup_steps = py_key("warmup_steps")
    nk[].k_weight_decay = py_key("weight_decay")
    var ops = Python.dict()
    py_op_add(ops, "training_numeric_mode", OPTIMIZER_OPN_TRAINING_NUMERIC_MODE)
    py_op_add(ops, "training_vendor", OPTIMIZER_OPN_TRAINING_VENDOR)
    py_op_add(ops, "seqdev_finalize", OPTIMIZER_OPN_SEQDEV_FINALIZE)
    py_op_add(ops, "seqdev_array", OPTIMIZER_OPN_SEQDEV_ARRAY)
    py_op_add(ops, "seqdev_construct", OPTIMIZER_OPN_SEQDEV_CONSTRUCT)
    py_op_add(ops, "seqdev_copy", OPTIMIZER_OPN_SEQDEV_COPY)
    py_op_add(ops, "seqdev_host", OPTIMIZER_OPN_SEQDEV_HOST)
    py_op_add(ops, "seqdev_ndim", OPTIMIZER_OPN_SEQDEV_NDIM)
    py_op_add(ops, "seqdev_new", OPTIMIZER_OPN_SEQDEV_NEW)
    py_op_add(ops, "seqdev_reshape", OPTIMIZER_OPN_SEQDEV_RESHAPE)
    py_op_add(ops, "seqdev_size", OPTIMIZER_OPN_SEQDEV_SIZE)
    py_op_add(ops, "seqdev_size_of", OPTIMIZER_OPN_SEQDEV_SIZE_OF)
    py_op_add(ops, "seqdev_to", OPTIMIZER_OPN_SEQDEV_TO)
    py_op_add(ops, "seqdev_upload", OPTIMIZER_OPN_SEQDEV_UPLOAD)
    py_op_add(ops, "seqdev_wrap", OPTIMIZER_OPN_SEQDEV_WRAP)
    py_op_add(ops, "residual_dropout", OPTIMIZER_OPN_RESIDUAL_DROPOUT)
    py_op_add(ops, "residual_dropout_backward", OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD)
    py_op_add(ops, "emblayer_init", OPTIMIZER_OPN_EMBLAYER_INIT)
    py_op_add(ops, "emblayer_pretrained", OPTIMIZER_OPN_EMBLAYER_PRETRAINED)
    py_op_add(ops, "emblayer_set_params", OPTIMIZER_OPN_EMBLAYER_SET_PARAMS)
    py_op_add(ops, "emblayer_state", OPTIMIZER_OPN_EMBLAYER_STATE)
    py_op_add(ops, "emblayer_to_device", OPTIMIZER_OPN_EMBLAYER_TO_DEVICE)
    py_op_add(ops, "emblayer_forward", OPTIMIZER_OPN_EMBLAYER_FORWARD)
    py_op_add(ops, "emblayer_backward", OPTIMIZER_OPN_EMBLAYER_BACKWARD)
    py_op_add(ops, "emblayer_load", OPTIMIZER_OPN_EMBLAYER_LOAD)
    py_op_add(ops, "seqln_initialize", OPTIMIZER_OPN_SEQLN_INITIALIZE)
    py_op_add(ops, "seqln_forward", OPTIMIZER_OPN_SEQLN_FORWARD)
    py_op_add(ops, "seqln_backward", OPTIMIZER_OPN_SEQLN_BACKWARD)
    py_op_add(ops, "seqln_device_io", OPTIMIZER_OPN_SEQLN_DEVICE_IO)
    py_op_add(ops, "seqln_to_device", OPTIMIZER_OPN_SEQLN_TO_DEVICE)
    py_op_add(ops, "seqln_layer", OPTIMIZER_OPN_SEQLN_LAYER)
    py_op_add(ops, "seqln_layer_backward", OPTIMIZER_OPN_SEQLN_LAYER_BACKWARD)
    py_op_add(ops, "seqmoe_initialize", OPTIMIZER_OPN_SEQMOE_INITIALIZE)
    py_op_add(ops, "seqmoe_forward", OPTIMIZER_OPN_SEQMOE_FORWARD)
    py_op_add(ops, "seqmoe_backward", OPTIMIZER_OPN_SEQMOE_BACKWARD)
    py_op_add(ops, "seqmoe_load", OPTIMIZER_OPN_SEQMOE_LOAD)
    py_op_add(ops, "seqmoe_release", OPTIMIZER_OPN_SEQMOE_RELEASE)
    py_op_add(ops, "seqmoe_routing", OPTIMIZER_OPN_SEQMOE_ROUTING)
    py_op_add(ops, "seqmoe_selected", OPTIMIZER_OPN_SEQMOE_SELECTED)
    py_op_add(ops, "seqmoe_setattr", OPTIMIZER_OPN_SEQMOE_SETATTR)
    py_op_add(ops, "seqmoe_to_device", OPTIMIZER_OPN_SEQMOE_TO_DEVICE)
    py_op_add(ops, "seq_construct", OPTIMIZER_OPN_SEQ_CONSTRUCT)
    py_op_add(ops, "seq_step", OPTIMIZER_OPN_SEQ_STEP)
    py_op_add(ops, "seq_state", OPTIMIZER_OPN_SEQ_STATE)
    py_op_add(ops, "seq_state_set", OPTIMIZER_OPN_SEQ_STATE_SET)
    py_op_add(ops, "seq_word", OPTIMIZER_OPN_SEQ_WORD)
    py_op_add(ops, "seq_word_set", OPTIMIZER_OPN_SEQ_WORD_SET)
    py_op_add(ops, "seq_slot", OPTIMIZER_OPN_SEQ_SLOT)
    py_op_add(ops, "seq_handle", OPTIMIZER_OPN_SEQ_HANDLE)
    py_op_add(ops, "seq_binding", OPTIMIZER_OPN_SEQ_BINDING)
    py_op_add(ops, "seq_resident", OPTIMIZER_OPN_SEQ_RESIDENT)
    py_op_add(ops, "seq_export", OPTIMIZER_OPN_SEQ_EXPORT)
    py_op_add(ops, "seq_import", OPTIMIZER_OPN_SEQ_IMPORT)
    py_op_add(ops, "seq_finalize", OPTIMIZER_OPN_SEQ_FINALIZE)
    py_op_add(ops, "device_construct", OPTIMIZER_OPN_DEVICE_CONSTRUCT)
    py_op_add(ops, "device_new", OPTIMIZER_OPN_DEVICE_NEW)
    py_op_add(ops, "device_upload", OPTIMIZER_OPN_DEVICE_UPLOAD)
    py_op_add(ops, "device_reshape", OPTIMIZER_OPN_DEVICE_RESHAPE)
    py_op_add(ops, "device_host", OPTIMIZER_OPN_DEVICE_HOST)
    py_op_add(ops, "schedule_construct", OPTIMIZER_OPN_SCHEDULE_CONSTRUCT)
    py_op_add(ops, "schedule_from_config", OPTIMIZER_OPN_SCHEDULE_FROM_CONFIG)
    py_op_add(ops, "schedule_getstate", OPTIMIZER_OPN_SCHEDULE_GETSTATE)
    py_op_add(ops, "schedule_setattr", OPTIMIZER_OPN_SCHEDULE_SETATTR)
    py_op_add(ops, "schedule_delattr", OPTIMIZER_OPN_SCHEDULE_DELATTR)
    py_op_add(ops, "schedule_setstate", OPTIMIZER_OPN_SCHEDULE_SETSTATE)
    py_op_add(ops, "schedule_spec", OPTIMIZER_OPN_SCHEDULE_SPEC)
    py_op_add(ops, "schedule_config", OPTIMIZER_OPN_SCHEDULE_CONFIG)
    py_op_add(ops, "schedule_values", OPTIMIZER_OPN_SCHEDULE_VALUES)
    py_op_add(ops, "schedule_list", OPTIMIZER_OPN_SCHEDULE_LIST)
    py_op_add(ops, "schedule_at", OPTIMIZER_OPN_SCHEDULE_AT)
    py_op_add(ops, "schedule_bits", OPTIMIZER_OPN_SCHEDULE_BITS)
    py_op_add(ops, "restore_owner", OPTIMIZER_OPN_RESTORE_OWNER)
    py_op_add(ops, "construct", OPTIMIZER_OPN_CONSTRUCT)
    py_op_add(ops, "get_params", OPTIMIZER_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", OPTIMIZER_OPN_SET_PARAMS)
    py_op_add(ops, "export_owner", OPTIMIZER_OPN_EXPORT_OWNER)
    py_op_add(ops, "linear_forward", OPTIMIZER_OPN_LINEAR_FORWARD)
    py_op_add(ops, "linear_backward", OPTIMIZER_OPN_LINEAR_BACKWARD)
    py_op_add(ops, "rms_norm_forward", OPTIMIZER_OPN_RMS_NORM_FORWARD)
    py_op_add(ops, "rms_norm_backward", OPTIMIZER_OPN_RMS_NORM_BACKWARD)
    py_op_add(ops, "embedding_forward", OPTIMIZER_OPN_EMBEDDING_FORWARD)
    py_op_add(ops, "embedding_backward", OPTIMIZER_OPN_EMBEDDING_BACKWARD)
    py_op_add(ops, "cross_entropy", OPTIMIZER_OPN_CROSS_ENTROPY)
    py_op_add(ops, "chunked_lm_head_loss", OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS)
    py_op_add(ops, "samba_head_loss", OPTIMIZER_OPN_SAMBA_HEAD_LOSS)
    py_op_add(ops, "mlp_inference_sessions", OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS)
    py_op_add(ops, "binding", OPTIMIZER_OPN_BINDING)
    py_op_add(ops, "raw_io", OPTIMIZER_OPN_RAW_IO)
    py_op_add(ops, "param_device", OPTIMIZER_OPN_PARAM_DEVICE)
    py_op_add(ops, "step", OPTIMIZER_OPN_STEP)
    py_op_add(ops, "state", OPTIMIZER_OPN_STATE)
    py_op_add(ops, "state_set", OPTIMIZER_OPN_STATE_SET)
    py_op_add(ops, "load", OPTIMIZER_OPN_LOAD)
    py_op_add(ops, "zero", OPTIMIZER_OPN_ZERO)
    py_op_add(ops, "resident", OPTIMIZER_OPN_RESIDENT)
    py_op_add(ops, "io", OPTIMIZER_OPN_IO)
    py_op_add(ops, "clip", OPTIMIZER_OPN_CLIP)
    py_op_add(ops, "accumulate", OPTIMIZER_OPN_ACCUMULATE)
    py_op_add(ops, "aligned", OPTIMIZER_OPN_ALIGNED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def optimizer_keys() raises -> UnsafePointer[OptimizerKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _OPTIMIZER_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _optimizer_keys_fill(nk)
    return nk


def optimizer_keys_in_finalizer() raises -> UnsafePointer[OptimizerKeys, MutUntrackedOrigin]:
    """The keys for a native finalizer (`__del__` descriptor) and the router
    entry a Python `__del__` reaches. The first fill interns names through
    `sys` (py_key -> native_loaded_module), a fatal read once the interpreter
    tears its modules down (fx-smoke-b59, rc -6). So during finalization an
    unfilled table is returned unfilled (`ready` False) and the caller
    releases nothing: no call of this process ever filled it, so no native
    owner it could release was made through these doors. Filled tables cost
    what `optimizer_keys()` costs: one slot fetch and one flag test."""
    var nk = _OPTIMIZER_KEYS.get_or_create_ptr()
    if not nk[].ready and not native_interpreter_finalizing():
        _optimizer_keys_fill(nk)
    return nk
