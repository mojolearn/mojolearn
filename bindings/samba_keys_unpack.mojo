# SPDX-License-Identifier: Apache-2.0
# The samba family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime SAMBA_OPN_DECODE_SESSION_DROP = 0
comptime SAMBA_OPN_MODEL_TRAIN_STEP = 1
comptime SAMBA_OPN_FORWARD = 2
comptime SAMBA_OPN_STEP = 3
comptime SAMBA_OPN_ALLOCATE_STATE = 4
comptime SAMBA_OPN_STATE_CONSTRUCT = 5
comptime SAMBA_OPN_LOSS = 6
comptime SAMBA_OPN_CHECKPOINT_SAVE = 7
comptime SAMBA_OPN_CHECKPOINT_LOAD = 8
comptime SAMBA_OPN_CHECKPOINT_READ = 9
comptime SAMBA_OPN_INFERENCE_CONSTRUCT = 10
comptime SAMBA_OPN_INFERENCE_PARAMETERS = 11
comptime SAMBA_OPN_INFERENCE_CHECKPOINT = 12
comptime SAMBA_OPN_INFERENCE_FORWARD = 13
comptime SAMBA_OPN_INFERENCE_STEP = 14
comptime SAMBA_OPN_DECODE_SESSION_OPEN = 15
comptime SAMBA_OPN_DECODE_SESSION_INIT = 16
comptime SAMBA_OPN_DECODE_SESSION_STEP = 17
comptime SAMBA_OPN_DECODE_SESSION_DECODE = 18
comptime SAMBA_OPN_DECODE_SESSION_SYNC = 19
comptime SAMBA_OPN_DECODE_SESSION_LOAD = 20
comptime SAMBA_OPN_DECODE_SESSION_CLOSE = 21
comptime SAMBA_OPN_DECODE_SESSION_EXIT = 22
comptime SAMBA_OPN_DECODE_SESSION_ENTER = 23
comptime SAMBA_OPN_DECODE_SESSION_STATE = 24
comptime SAMBA_OPN_DECODE_SESSION_IS_OPEN = 25
comptime SAMBA_OPN_DECODE_SESSION_DEVICE = 26
comptime SAMBA_OPN_DECODE_SESSION_REPR = 27
comptime SAMBA_OPN_INFERENCE_REPR = 28
comptime SAMBA_OPN_INFERENCE_REFUSE = 29
comptime SAMBA_OPN_CONSTRUCT = 30
comptime SAMBA_OPN_LOAD_WEIGHTS = 31
comptime SAMBA_OPN_PARAMETERS = 32
comptime SAMBA_OPN_GENERATOR_INIT = 33
comptime SAMBA_OPN_GENERATOR_SNAPSHOT = 34
comptime SAMBA_OPN_GENERATOR_RESTORE = 35
comptime SAMBA_OPN_GENERATOR_NEXT = 36
comptime SAMBA_OPN_SNAPSHOT = 37
comptime SAMBA_OPN_VALIDATE = 38
comptime SAMBA_OPN_FROM_STATE = 39
comptime SAMBA_OPN_PUBLISH = 40
comptime SAMBA_OPN_APPLY = 41
comptime SAMBA_OPN_GRADIENTS = 42
comptime SAMBA_OPN_MODEL_GRADIENTS = 43
comptime SAMBA_OPN_INITIALIZE = 44
comptime SAMBA_OPN_DICT = 45
comptime SAMBA_OPN_FROM_DICT = 46
comptime SAMBA_OPN_BLOCK_SHAPES = 47
comptime SAMBA_OPN_REGISTRY = 48
comptime SAMBA_OPN_PARALLEL_LAYOUT = 49


struct SambaKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__blocks: PythonObject
    var k__discard_reports: PythonObject
    var k__handle: PythonObject
    var k__m: PythonObject
    var k__model: PythonObject
    var k__mv: PythonObject
    var k__native_cpu_inference: PythonObject
    var k__native_optimizer_owner: PythonObject
    var k__native_primitives: PythonObject
    var k__open: PythonObject
    var k__private: PythonObject
    var k__resident_session: PythonObject
    var k__resident_wanted: PythonObject
    var k__rows: PythonObject
    var k__state: PythonObject
    var k__v: PythonObject
    var k__vocab: PythonObject
    var k__w: PythonObject
    var k_accumulation_steps: PythonObject
    var k_admission: PythonObject
    var k_all_finite_f32: PythonObject
    var k_array_class: PythonObject
    var k_array_getters: PythonObject
    var k_array_metadata: PythonObject
    var k_arrays: PythonObject
    var k_batch_size: PythonObject
    var k_beta1: PythonObject
    var k_beta2: PythonObject
    var k_betas: PythonObject
    var k_bf16_class: PythonObject
    var k_buf_initialized: PythonObject
    var k_buffer_types: PythonObject
    var k_clip_coef_: PythonObject
    var k_config: PythonObject
    var k_config_class: PythonObject
    var k_config_context: PythonObject
    var k_counter: PythonObject
    var k_d_model: PythonObject
    var k_decode_session_class: PythonObject
    var k_defaultdict_type: PythonObject
    var k_device_finalizer_types: PythonObject
    var k_dropout: PythonObject
    var k_dtype: PythonObject
    var k_entries: PythonObject
    var k_eps: PythonObject
    var k_exp_avg: PythonObject
    var k_exp_avg_sq: PythonObject
    var k_experiment_profile: PythonObject
    var k_flat: PythonObject
    var k_generation_classes: PythonObject
    var k_generator: PythonObject
    var k_generator_class: PythonObject
    var k_guards: PythonObject
    var k_head_dim: PythonObject
    var k_hex: PythonObject
    var k_int8_class: PythonObject
    var k_intermediate: PythonObject
    var k_kind: PythonObject
    var k_last_: PythonObject
    var k_layers: PythonObject
    var k_linear_forward: PythonObject
    var k_loss: PythonObject
    var k_lr: PythonObject
    var k_lr_: PythonObject
    var k_lr_schedule: PythonObject
    var k_mapping_types: PythonObject
    var k_max_norm: PythonObject
    var k_max_tokens: PythonObject
    var k_maximize: PythonObject
    var k_metadata_fields: PythonObject
    var k_min_lr: PythonObject
    var k_mode: PythonObject
    var k_model_class: PythonObject
    var k_module: PythonObject
    var k_modules: PythonObject
    var k_n_heads: PythonObject
    var k_n_kv_heads: PythonObject
    var k_n_total: PythonObject
    var k_name: PythonObject
    var k_names: PythonObject
    var k_native_factories: PythonObject
    var k_nbytes: PythonObject
    var k_ndarray: PythonObject
    var k_neural: PythonObject
    var k_nheads: PythonObject
    var k_norm_eps: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_profile: PythonObject
    var k_numpy: PythonObject
    var k_offset: PythonObject
    var k_offsets: PythonObject
    var k_opaque_types: PythonObject
    var k_optimizer: PythonObject
    var k_optimizer_class: PythonObject
    var k_optimizer_maximize_device: PythonObject
    var k_optimizer_step_io: PythonObject
    var k_packed_: PythonObject
    var k_parameters: PythonObject
    var k_params: PythonObject
    var k_path_types: PythonObject
    var k_payload: PythonObject
    var k_payload_sha256: PythonObject
    var k_peak_lr: PythonObject
    var k_plan_class: PythonObject
    var k_profile: PythonObject
    var k_records: PythonObject
    var k_registry: PythonObject
    var k_rng: PythonObject
    var k_scalar_types: PythonObject
    var k_schedule: PythonObject
    var k_schedule_classes: PythonObject
    var k_schedule_context: PythonObject
    var k_schedule_native: PythonObject
    var k_schedule_types: PythonObject
    var k_schema: PythonObject
    var k_seed: PythonObject
    var k_sha256: PythonObject
    var k_shape: PythonObject
    var k_shapes: PythonObject
    var k_size: PythonObject
    var k_state: PythonObject
    var k_state_class: PythonObject
    var k_step: PythonObject
    var k_t: PythonObject
    var k_tie_embeddings: PythonObject
    var k_total_norm: PythonObject
    var k_total_norm_: PythonObject
    var k_total_steps: PythonObject
    var k_type_mro: PythonObject
    var k_vendor: PythonObject
    var k_vocab: PythonObject
    var k_warmup_steps: PythonObject
    var k_weight_decay: PythonObject
    var k_weight_format: PythonObject
    var k_x: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__blocks = PythonObject(None)
        self.k__discard_reports = PythonObject(None)
        self.k__handle = PythonObject(None)
        self.k__m = PythonObject(None)
        self.k__model = PythonObject(None)
        self.k__mv = PythonObject(None)
        self.k__native_cpu_inference = PythonObject(None)
        self.k__native_optimizer_owner = PythonObject(None)
        self.k__native_primitives = PythonObject(None)
        self.k__open = PythonObject(None)
        self.k__private = PythonObject(None)
        self.k__resident_session = PythonObject(None)
        self.k__resident_wanted = PythonObject(None)
        self.k__rows = PythonObject(None)
        self.k__state = PythonObject(None)
        self.k__v = PythonObject(None)
        self.k__vocab = PythonObject(None)
        self.k__w = PythonObject(None)
        self.k_accumulation_steps = PythonObject(None)
        self.k_admission = PythonObject(None)
        self.k_all_finite_f32 = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_getters = PythonObject(None)
        self.k_array_metadata = PythonObject(None)
        self.k_arrays = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_beta1 = PythonObject(None)
        self.k_beta2 = PythonObject(None)
        self.k_betas = PythonObject(None)
        self.k_bf16_class = PythonObject(None)
        self.k_buf_initialized = PythonObject(None)
        self.k_buffer_types = PythonObject(None)
        self.k_clip_coef_ = PythonObject(None)
        self.k_config = PythonObject(None)
        self.k_config_class = PythonObject(None)
        self.k_config_context = PythonObject(None)
        self.k_counter = PythonObject(None)
        self.k_d_model = PythonObject(None)
        self.k_decode_session_class = PythonObject(None)
        self.k_defaultdict_type = PythonObject(None)
        self.k_device_finalizer_types = PythonObject(None)
        self.k_dropout = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_entries = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_exp_avg = PythonObject(None)
        self.k_exp_avg_sq = PythonObject(None)
        self.k_experiment_profile = PythonObject(None)
        self.k_flat = PythonObject(None)
        self.k_generation_classes = PythonObject(None)
        self.k_generator = PythonObject(None)
        self.k_generator_class = PythonObject(None)
        self.k_guards = PythonObject(None)
        self.k_head_dim = PythonObject(None)
        self.k_hex = PythonObject(None)
        self.k_int8_class = PythonObject(None)
        self.k_intermediate = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_last_ = PythonObject(None)
        self.k_layers = PythonObject(None)
        self.k_linear_forward = PythonObject(None)
        self.k_loss = PythonObject(None)
        self.k_lr = PythonObject(None)
        self.k_lr_ = PythonObject(None)
        self.k_lr_schedule = PythonObject(None)
        self.k_mapping_types = PythonObject(None)
        self.k_max_norm = PythonObject(None)
        self.k_max_tokens = PythonObject(None)
        self.k_maximize = PythonObject(None)
        self.k_metadata_fields = PythonObject(None)
        self.k_min_lr = PythonObject(None)
        self.k_mode = PythonObject(None)
        self.k_model_class = PythonObject(None)
        self.k_module = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_n_heads = PythonObject(None)
        self.k_n_kv_heads = PythonObject(None)
        self.k_n_total = PythonObject(None)
        self.k_name = PythonObject(None)
        self.k_names = PythonObject(None)
        self.k_native_factories = PythonObject(None)
        self.k_nbytes = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_neural = PythonObject(None)
        self.k_nheads = PythonObject(None)
        self.k_norm_eps = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_profile = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_offset = PythonObject(None)
        self.k_offsets = PythonObject(None)
        self.k_opaque_types = PythonObject(None)
        self.k_optimizer = PythonObject(None)
        self.k_optimizer_class = PythonObject(None)
        self.k_optimizer_maximize_device = PythonObject(None)
        self.k_optimizer_step_io = PythonObject(None)
        self.k_packed_ = PythonObject(None)
        self.k_parameters = PythonObject(None)
        self.k_params = PythonObject(None)
        self.k_path_types = PythonObject(None)
        self.k_payload = PythonObject(None)
        self.k_payload_sha256 = PythonObject(None)
        self.k_peak_lr = PythonObject(None)
        self.k_plan_class = PythonObject(None)
        self.k_profile = PythonObject(None)
        self.k_records = PythonObject(None)
        self.k_registry = PythonObject(None)
        self.k_rng = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_schedule = PythonObject(None)
        self.k_schedule_classes = PythonObject(None)
        self.k_schedule_context = PythonObject(None)
        self.k_schedule_native = PythonObject(None)
        self.k_schedule_types = PythonObject(None)
        self.k_schema = PythonObject(None)
        self.k_seed = PythonObject(None)
        self.k_sha256 = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_shapes = PythonObject(None)
        self.k_size = PythonObject(None)
        self.k_state = PythonObject(None)
        self.k_state_class = PythonObject(None)
        self.k_step = PythonObject(None)
        self.k_t = PythonObject(None)
        self.k_tie_embeddings = PythonObject(None)
        self.k_total_norm = PythonObject(None)
        self.k_total_norm_ = PythonObject(None)
        self.k_total_steps = PythonObject(None)
        self.k_type_mro = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_vocab = PythonObject(None)
        self.k_warmup_steps = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_weight_format = PythonObject(None)
        self.k_x = PythonObject(None)


comptime _SAMBA_KEYS = SharedSlot[SambaKeys, "MojolearnSambaKeysV1"]


def _samba_keys_fill(nk: UnsafePointer[SambaKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__blocks = py_key("_blocks")
    nk[].k__discard_reports = py_key("_discard_reports")
    nk[].k__handle = py_key("_handle")
    nk[].k__m = py_key("_m")
    nk[].k__model = py_key("_model")
    nk[].k__mv = py_key("_mv")
    nk[].k__native_cpu_inference = py_key("_native_cpu_inference")
    nk[].k__native_optimizer_owner = py_key("_native_optimizer_owner")
    nk[].k__native_primitives = py_key("_native_primitives")
    nk[].k__open = py_key("_open")
    nk[].k__private = py_key("_private")
    nk[].k__resident_session = py_key("_resident_session")
    nk[].k__resident_wanted = py_key("_resident_wanted")
    nk[].k__rows = py_key("_rows")
    nk[].k__state = py_key("_state")
    nk[].k__v = py_key("_v")
    nk[].k__vocab = py_key("_vocab")
    nk[].k__w = py_key("_w")
    nk[].k_accumulation_steps = py_key("accumulation_steps")
    nk[].k_admission = py_key("admission")
    nk[].k_all_finite_f32 = py_key("all_finite_f32")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_getters = py_key("array_getters")
    nk[].k_array_metadata = py_key("array_metadata")
    nk[].k_arrays = py_key("arrays")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_beta1 = py_key("beta1")
    nk[].k_beta2 = py_key("beta2")
    nk[].k_betas = py_key("betas")
    nk[].k_bf16_class = py_key("bf16_class")
    nk[].k_buf_initialized = py_key("buf_initialized")
    nk[].k_buffer_types = py_key("buffer_types")
    nk[].k_clip_coef_ = py_key("clip_coef_")
    nk[].k_config = py_key("config")
    nk[].k_config_class = py_key("config_class")
    nk[].k_config_context = py_key("config_context")
    nk[].k_counter = py_key("counter")
    nk[].k_d_model = py_key("d_model")
    nk[].k_decode_session_class = py_key("decode_session_class")
    nk[].k_defaultdict_type = py_key("defaultdict_type")
    nk[].k_device_finalizer_types = py_key("device_finalizer_types")
    nk[].k_dropout = py_key("dropout")
    nk[].k_dtype = py_key("dtype")
    nk[].k_entries = py_key("entries")
    nk[].k_eps = py_key("eps")
    nk[].k_exp_avg = py_key("exp_avg")
    nk[].k_exp_avg_sq = py_key("exp_avg_sq")
    nk[].k_experiment_profile = py_key("experiment_profile")
    nk[].k_flat = py_key("flat")
    nk[].k_generation_classes = py_key("generation_classes")
    nk[].k_generator = py_key("generator")
    nk[].k_generator_class = py_key("generator_class")
    nk[].k_guards = py_key("guards")
    nk[].k_head_dim = py_key("head_dim")
    nk[].k_hex = py_key("hex")
    nk[].k_int8_class = py_key("int8_class")
    nk[].k_intermediate = py_key("intermediate")
    nk[].k_kind = py_key("kind")
    nk[].k_last_ = py_key("last_")
    nk[].k_layers = py_key("layers")
    nk[].k_linear_forward = py_key("linear_forward")
    nk[].k_loss = py_key("loss")
    nk[].k_lr = py_key("lr")
    nk[].k_lr_ = py_key("lr_")
    nk[].k_lr_schedule = py_key("lr_schedule")
    nk[].k_mapping_types = py_key("mapping_types")
    nk[].k_max_norm = py_key("max_norm")
    nk[].k_max_tokens = py_key("max_tokens")
    nk[].k_maximize = py_key("maximize")
    nk[].k_metadata_fields = py_key("metadata_fields")
    nk[].k_min_lr = py_key("min_lr")
    nk[].k_mode = py_key("mode")
    nk[].k_model_class = py_key("model_class")
    nk[].k_module = py_key("module")
    nk[].k_modules = py_key("modules")
    nk[].k_n_heads = py_key("n_heads")
    nk[].k_n_kv_heads = py_key("n_kv_heads")
    nk[].k_n_total = py_key("n_total")
    nk[].k_name = py_key("name")
    nk[].k_names = py_key("names")
    nk[].k_native_factories = py_key("native_factories")
    nk[].k_nbytes = py_key("nbytes")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_neural = py_key("neural")
    nk[].k_nheads = py_key("nheads")
    nk[].k_norm_eps = py_key("norm_eps")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_profile = py_key("numeric_profile")
    nk[].k_numpy = py_key("numpy")
    nk[].k_offset = py_key("offset")
    nk[].k_offsets = py_key("offsets")
    nk[].k_opaque_types = py_key("opaque_types")
    nk[].k_optimizer = py_key("optimizer")
    nk[].k_optimizer_class = py_key("optimizer_class")
    nk[].k_optimizer_maximize_device = py_key("optimizer_maximize_device")
    nk[].k_optimizer_step_io = py_key("optimizer_step_io")
    nk[].k_packed_ = py_key("packed_")
    nk[].k_parameters = py_key("parameters")
    nk[].k_params = py_key("params")
    nk[].k_path_types = py_key("path_types")
    nk[].k_payload = py_key("payload")
    nk[].k_payload_sha256 = py_key("payload_sha256")
    nk[].k_peak_lr = py_key("peak_lr")
    nk[].k_plan_class = py_key("plan_class")
    nk[].k_profile = py_key("profile")
    nk[].k_records = py_key("records")
    nk[].k_registry = py_key("registry")
    nk[].k_rng = py_key("rng")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_schedule = py_key("schedule")
    nk[].k_schedule_classes = py_key("schedule_classes")
    nk[].k_schedule_context = py_key("schedule_context")
    nk[].k_schedule_native = py_key("schedule_native")
    nk[].k_schedule_types = py_key("schedule_types")
    nk[].k_schema = py_key("schema")
    nk[].k_seed = py_key("seed")
    nk[].k_sha256 = py_key("sha256")
    nk[].k_shape = py_key("shape")
    nk[].k_shapes = py_key("shapes")
    nk[].k_size = py_key("size")
    nk[].k_state = py_key("state")
    nk[].k_state_class = py_key("state_class")
    nk[].k_step = py_key("step")
    nk[].k_t = py_key("t")
    nk[].k_tie_embeddings = py_key("tie_embeddings")
    nk[].k_total_norm = py_key("total_norm")
    nk[].k_total_norm_ = py_key("total_norm_")
    nk[].k_total_steps = py_key("total_steps")
    nk[].k_type_mro = py_key("type_mro")
    nk[].k_vendor = py_key("vendor")
    nk[].k_vocab = py_key("vocab")
    nk[].k_warmup_steps = py_key("warmup_steps")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_weight_format = py_key("weight_format")
    nk[].k_x = py_key("x")
    var ops = Python.dict()
    py_op_add(ops, "decode_session_drop", SAMBA_OPN_DECODE_SESSION_DROP)
    py_op_add(ops, "model_train_step", SAMBA_OPN_MODEL_TRAIN_STEP)
    py_op_add(ops, "forward", SAMBA_OPN_FORWARD)
    py_op_add(ops, "step", SAMBA_OPN_STEP)
    py_op_add(ops, "allocate_state", SAMBA_OPN_ALLOCATE_STATE)
    py_op_add(ops, "state_construct", SAMBA_OPN_STATE_CONSTRUCT)
    py_op_add(ops, "loss", SAMBA_OPN_LOSS)
    py_op_add(ops, "checkpoint_save", SAMBA_OPN_CHECKPOINT_SAVE)
    py_op_add(ops, "checkpoint_load", SAMBA_OPN_CHECKPOINT_LOAD)
    py_op_add(ops, "checkpoint_read", SAMBA_OPN_CHECKPOINT_READ)
    py_op_add(ops, "inference_construct", SAMBA_OPN_INFERENCE_CONSTRUCT)
    py_op_add(ops, "inference_parameters", SAMBA_OPN_INFERENCE_PARAMETERS)
    py_op_add(ops, "inference_checkpoint", SAMBA_OPN_INFERENCE_CHECKPOINT)
    py_op_add(ops, "inference_forward", SAMBA_OPN_INFERENCE_FORWARD)
    py_op_add(ops, "inference_step", SAMBA_OPN_INFERENCE_STEP)
    py_op_add(ops, "decode_session_open", SAMBA_OPN_DECODE_SESSION_OPEN)
    py_op_add(ops, "decode_session_init", SAMBA_OPN_DECODE_SESSION_INIT)
    py_op_add(ops, "decode_session_step", SAMBA_OPN_DECODE_SESSION_STEP)
    py_op_add(ops, "decode_session_decode", SAMBA_OPN_DECODE_SESSION_DECODE)
    py_op_add(ops, "decode_session_sync", SAMBA_OPN_DECODE_SESSION_SYNC)
    py_op_add(ops, "decode_session_load", SAMBA_OPN_DECODE_SESSION_LOAD)
    py_op_add(ops, "decode_session_close", SAMBA_OPN_DECODE_SESSION_CLOSE)
    py_op_add(ops, "decode_session_exit", SAMBA_OPN_DECODE_SESSION_EXIT)
    py_op_add(ops, "decode_session_enter", SAMBA_OPN_DECODE_SESSION_ENTER)
    py_op_add(ops, "decode_session_state", SAMBA_OPN_DECODE_SESSION_STATE)
    py_op_add(ops, "decode_session_is_open", SAMBA_OPN_DECODE_SESSION_IS_OPEN)
    py_op_add(ops, "decode_session_device", SAMBA_OPN_DECODE_SESSION_DEVICE)
    py_op_add(ops, "decode_session_repr", SAMBA_OPN_DECODE_SESSION_REPR)
    py_op_add(ops, "inference_repr", SAMBA_OPN_INFERENCE_REPR)
    py_op_add(ops, "inference_refuse", SAMBA_OPN_INFERENCE_REFUSE)
    py_op_add(ops, "construct", SAMBA_OPN_CONSTRUCT)
    py_op_add(ops, "load_weights", SAMBA_OPN_LOAD_WEIGHTS)
    py_op_add(ops, "parameters", SAMBA_OPN_PARAMETERS)
    py_op_add(ops, "generator_init", SAMBA_OPN_GENERATOR_INIT)
    py_op_add(ops, "generator_snapshot", SAMBA_OPN_GENERATOR_SNAPSHOT)
    py_op_add(ops, "generator_restore", SAMBA_OPN_GENERATOR_RESTORE)
    py_op_add(ops, "generator_next", SAMBA_OPN_GENERATOR_NEXT)
    py_op_add(ops, "snapshot", SAMBA_OPN_SNAPSHOT)
    py_op_add(ops, "validate", SAMBA_OPN_VALIDATE)
    py_op_add(ops, "from_state", SAMBA_OPN_FROM_STATE)
    py_op_add(ops, "publish", SAMBA_OPN_PUBLISH)
    py_op_add(ops, "apply", SAMBA_OPN_APPLY)
    py_op_add(ops, "gradients", SAMBA_OPN_GRADIENTS)
    py_op_add(ops, "model_gradients", SAMBA_OPN_MODEL_GRADIENTS)
    py_op_add(ops, "initialize", SAMBA_OPN_INITIALIZE)
    py_op_add(ops, "dict", SAMBA_OPN_DICT)
    py_op_add(ops, "from_dict", SAMBA_OPN_FROM_DICT)
    py_op_add(ops, "block_shapes", SAMBA_OPN_BLOCK_SHAPES)
    py_op_add(ops, "registry", SAMBA_OPN_REGISTRY)
    py_op_add(ops, "parallel_layout", SAMBA_OPN_PARALLEL_LAYOUT)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def samba_keys() raises -> UnsafePointer[SambaKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _SAMBA_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _samba_keys_fill(nk)
    return nk
