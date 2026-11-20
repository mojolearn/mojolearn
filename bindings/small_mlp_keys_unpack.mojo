# SPDX-License-Identifier: Apache-2.0
# The small_mlp family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime SMALL_MLP_OPN_TRAIN_STEP = 0
comptime SMALL_MLP_OPN_TRAIN_STEPS = 1
comptime SMALL_MLP_OPN_PREDICT_LOGITS = 2
comptime SMALL_MLP_OPN_PREDICT_LOGITS_SESSIONS = 3
comptime SMALL_MLP_OPN_LOSS_AND_GRADS = 4
comptime SMALL_MLP_OPN_APPLY_GRADIENTS = 5
comptime SMALL_MLP_OPN_STATE_DICT = 6
comptime SMALL_MLP_OPN_LOAD_STATE_DICT = 7
comptime SMALL_MLP_OPN_STEP = 8
comptime SMALL_MLP_OPN_WEIGHTS = 9
comptime SMALL_MLP_OPN_SAVE_CHECKPOINT = 10
comptime SMALL_MLP_OPN_OWNER_TRAIN_STEP = 11
comptime SMALL_MLP_OPN_OWNER_TRAIN_STEPS = 12
comptime SMALL_MLP_OPN_OWNER_PREDICT_LOGITS = 13
comptime SMALL_MLP_OPN_OWNER_PREDICT_LOGITS_SESSIONS = 14
comptime SMALL_MLP_OPN_OWNER_LOSS_AND_GRADS = 15
comptime SMALL_MLP_OPN_OWNER_APPLY_GRADIENTS = 16
comptime SMALL_MLP_OPN_OWNER_STATE_DICT = 17
comptime SMALL_MLP_OPN_OWNER_LOAD_STATE_DICT = 18
comptime SMALL_MLP_OPN_OWNER_STEP = 19
comptime SMALL_MLP_OPN_OWNER_WEIGHTS = 20
comptime SMALL_MLP_OPN_OWNER_SAVE_CHECKPOINT = 21
comptime SMALL_MLP_OPN_OWNER_CREATE = 22
comptime SMALL_MLP_OPN_OWNER_CLOSE = 23
comptime SMALL_MLP_OPN_INFERENCE_CHECKPOINT = 24
comptime SMALL_MLP_OPN_INFERENCE_REPR = 25
comptime SMALL_MLP_OPN_INFERENCE_INITIALIZE = 26
comptime SMALL_MLP_OPN_INFERENCE_WEIGHTS = 27
comptime SMALL_MLP_OPN_INFERENCE_PREDICT = 28
comptime SMALL_MLP_OPN_INITIALIZE = 29
comptime SMALL_MLP_OPN_INFERENCE_CONNECT = 30
comptime SMALL_MLP_OPN_FROM_CHECKPOINT = 31
comptime SMALL_MLP_OPN_NATIVE_PROVIDER_INFO = 32
comptime SMALL_MLP_OPN_CREATE = 33
comptime SMALL_MLP_OPN_CLOSE = 34


struct SmallMlpKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__ARRAY_CONTEXT: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___del__: PythonObject
    var k___mro__: PythonObject
    var k__handle: PythonObject
    var k__m: PythonObject
    var k__weights: PythonObject
    var k_admission: PythonObject
    var k_architecture: PythonObject
    var k_arithmetic_profile: PythonObject
    var k_array: PythonObject
    var k_array_type: PythonObject
    var k_backend: PythonObject
    var k_beta1: PythonObject
    var k_beta2: PythonObject
    var k_class: PythonObject
    var k_codes: PythonObject
    var k_config: PythonObject
    var k_data: PythonObject
    var k_data_schedule: PythonObject
    var k_dtype: PythonObject
    var k_entries: PythonObject
    var k_environ: PythonObject
    var k_eps: PythonObject
    var k_family_low_bf16: PythonObject
    var k_family_low_int8: PythonObject
    var k_flags: PythonObject
    var k_gradients: PythonObject
    var k_helpers: PythonObject
    var k_hex: PythonObject
    var k_input_grad: PythonObject
    var k_itemsizes: PythonObject
    var k_kind: PythonObject
    var k_logits: PythonObject
    var k_loss: PythonObject
    var k_losses: PythonObject
    var k_lowbit: PythonObject
    var k_lr: PythonObject
    var k_m: PythonObject
    var k_mlp_forward_logits: PythonObject
    var k_mlp_sessions: PythonObject
    var k_mlp_sessions_rows: PythonObject
    var k_modes: PythonObject
    var k_module: PythonObject
    var k_neural_arithmetic_profile: PythonObject
    var k_numeric_mode: PythonObject
    var k_optimizer: PythonObject
    var k_parameter_order: PythonObject
    var k_payload: PythonObject
    var k_payload_sha256: PythonObject
    var k_schema: PythonObject
    var k_shape: PythonObject
    var k_small_mlp_program: PythonObject
    var k_step: PythonObject
    var k_strided_copy_bytes: PythonObject
    var k_strides: PythonObject
    var k_struct_unpack: PythonObject
    var k_trainer: PythonObject
    var k_trainer_finalizer: PythonObject
    var k_training_numeric_mode: PythonObject
    var k_training_vendor: PythonObject
    var k_unbind: PythonObject
    var k_v: PythonObject
    var k_weight_decay: PythonObject
    var k_weight_format: PythonObject
    var k_weights: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__ARRAY_CONTEXT = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__handle = PythonObject(None)
        self.k__m = PythonObject(None)
        self.k__weights = PythonObject(None)
        self.k_admission = PythonObject(None)
        self.k_architecture = PythonObject(None)
        self.k_arithmetic_profile = PythonObject(None)
        self.k_array = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend = PythonObject(None)
        self.k_beta1 = PythonObject(None)
        self.k_beta2 = PythonObject(None)
        self.k_class = PythonObject(None)
        self.k_codes = PythonObject(None)
        self.k_config = PythonObject(None)
        self.k_data = PythonObject(None)
        self.k_data_schedule = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_entries = PythonObject(None)
        self.k_environ = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_family_low_bf16 = PythonObject(None)
        self.k_family_low_int8 = PythonObject(None)
        self.k_flags = PythonObject(None)
        self.k_gradients = PythonObject(None)
        self.k_helpers = PythonObject(None)
        self.k_hex = PythonObject(None)
        self.k_input_grad = PythonObject(None)
        self.k_itemsizes = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_logits = PythonObject(None)
        self.k_loss = PythonObject(None)
        self.k_losses = PythonObject(None)
        self.k_lowbit = PythonObject(None)
        self.k_lr = PythonObject(None)
        self.k_m = PythonObject(None)
        self.k_mlp_forward_logits = PythonObject(None)
        self.k_mlp_sessions = PythonObject(None)
        self.k_mlp_sessions_rows = PythonObject(None)
        self.k_modes = PythonObject(None)
        self.k_module = PythonObject(None)
        self.k_neural_arithmetic_profile = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_optimizer = PythonObject(None)
        self.k_parameter_order = PythonObject(None)
        self.k_payload = PythonObject(None)
        self.k_payload_sha256 = PythonObject(None)
        self.k_schema = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_small_mlp_program = PythonObject(None)
        self.k_step = PythonObject(None)
        self.k_strided_copy_bytes = PythonObject(None)
        self.k_strides = PythonObject(None)
        self.k_struct_unpack = PythonObject(None)
        self.k_trainer = PythonObject(None)
        self.k_trainer_finalizer = PythonObject(None)
        self.k_training_numeric_mode = PythonObject(None)
        self.k_training_vendor = PythonObject(None)
        self.k_unbind = PythonObject(None)
        self.k_v = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_weight_format = PythonObject(None)
        self.k_weights = PythonObject(None)


comptime _SMALL_MLP_KEYS = SharedSlot[SmallMlpKeys, "MojolearnSmallMlpKeysV1"]


def _small_mlp_keys_fill(nk: UnsafePointer[SmallMlpKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__ARRAY_CONTEXT = py_key("_ARRAY_CONTEXT")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___del__ = py_key("__del__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__handle = py_key("_handle")
    nk[].k__m = py_key("_m")
    nk[].k__weights = py_key("_weights")
    nk[].k_admission = py_key("admission")
    nk[].k_architecture = py_key("architecture")
    nk[].k_arithmetic_profile = py_key("arithmetic_profile")
    nk[].k_array = py_key("array")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend = py_key("backend")
    nk[].k_beta1 = py_key("beta1")
    nk[].k_beta2 = py_key("beta2")
    nk[].k_class = py_key("class")
    nk[].k_codes = py_key("codes")
    nk[].k_config = py_key("config")
    nk[].k_data = py_key("data")
    nk[].k_data_schedule = py_key("data_schedule")
    nk[].k_dtype = py_key("dtype")
    nk[].k_entries = py_key("entries")
    nk[].k_environ = py_key("environ")
    nk[].k_eps = py_key("eps")
    nk[].k_family_low_bf16 = py_key("family_low_bf16")
    nk[].k_family_low_int8 = py_key("family_low_int8")
    nk[].k_flags = py_key("flags")
    nk[].k_gradients = py_key("gradients")
    nk[].k_helpers = py_key("helpers")
    nk[].k_hex = py_key("hex")
    nk[].k_input_grad = py_key("input_grad")
    nk[].k_itemsizes = py_key("itemsizes")
    nk[].k_kind = py_key("kind")
    nk[].k_logits = py_key("logits")
    nk[].k_loss = py_key("loss")
    nk[].k_losses = py_key("losses")
    nk[].k_lowbit = py_key("lowbit")
    nk[].k_lr = py_key("lr")
    nk[].k_m = py_key("m")
    nk[].k_mlp_forward_logits = py_key("mlp_forward_logits")
    nk[].k_mlp_sessions = py_key("mlp_sessions")
    nk[].k_mlp_sessions_rows = py_key("mlp_sessions_rows")
    nk[].k_modes = py_key("modes")
    nk[].k_module = py_key("module")
    nk[].k_neural_arithmetic_profile = py_key("neural_arithmetic_profile")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_optimizer = py_key("optimizer")
    nk[].k_parameter_order = py_key("parameter_order")
    nk[].k_payload = py_key("payload")
    nk[].k_payload_sha256 = py_key("payload_sha256")
    nk[].k_schema = py_key("schema")
    nk[].k_shape = py_key("shape")
    nk[].k_small_mlp_program = py_key("small_mlp_program")
    nk[].k_step = py_key("step")
    nk[].k_strided_copy_bytes = py_key("strided_copy_bytes")
    nk[].k_strides = py_key("strides")
    nk[].k_struct_unpack = py_key("struct_unpack")
    nk[].k_trainer = py_key("trainer")
    nk[].k_trainer_finalizer = py_key("trainer_finalizer")
    nk[].k_training_numeric_mode = py_key("training_numeric_mode")
    nk[].k_training_vendor = py_key("training_vendor")
    nk[].k_unbind = py_key("unbind")
    nk[].k_v = py_key("v")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_weight_format = py_key("weight_format")
    nk[].k_weights = py_key("weights")
    var ops = Python.dict()
    py_op_add(ops, "train_step", SMALL_MLP_OPN_TRAIN_STEP)
    py_op_add(ops, "train_steps", SMALL_MLP_OPN_TRAIN_STEPS)
    py_op_add(ops, "predict_logits", SMALL_MLP_OPN_PREDICT_LOGITS)
    py_op_add(ops, "predict_logits_sessions", SMALL_MLP_OPN_PREDICT_LOGITS_SESSIONS)
    py_op_add(ops, "loss_and_grads", SMALL_MLP_OPN_LOSS_AND_GRADS)
    py_op_add(ops, "apply_gradients", SMALL_MLP_OPN_APPLY_GRADIENTS)
    py_op_add(ops, "state_dict", SMALL_MLP_OPN_STATE_DICT)
    py_op_add(ops, "load_state_dict", SMALL_MLP_OPN_LOAD_STATE_DICT)
    py_op_add(ops, "step", SMALL_MLP_OPN_STEP)
    py_op_add(ops, "weights", SMALL_MLP_OPN_WEIGHTS)
    py_op_add(ops, "save_checkpoint", SMALL_MLP_OPN_SAVE_CHECKPOINT)
    py_op_add(ops, "owner_train_step", SMALL_MLP_OPN_OWNER_TRAIN_STEP)
    py_op_add(ops, "owner_train_steps", SMALL_MLP_OPN_OWNER_TRAIN_STEPS)
    py_op_add(ops, "owner_predict_logits", SMALL_MLP_OPN_OWNER_PREDICT_LOGITS)
    py_op_add(ops, "owner_predict_logits_sessions", SMALL_MLP_OPN_OWNER_PREDICT_LOGITS_SESSIONS)
    py_op_add(ops, "owner_loss_and_grads", SMALL_MLP_OPN_OWNER_LOSS_AND_GRADS)
    py_op_add(ops, "owner_apply_gradients", SMALL_MLP_OPN_OWNER_APPLY_GRADIENTS)
    py_op_add(ops, "owner_state_dict", SMALL_MLP_OPN_OWNER_STATE_DICT)
    py_op_add(ops, "owner_load_state_dict", SMALL_MLP_OPN_OWNER_LOAD_STATE_DICT)
    py_op_add(ops, "owner_step", SMALL_MLP_OPN_OWNER_STEP)
    py_op_add(ops, "owner_weights", SMALL_MLP_OPN_OWNER_WEIGHTS)
    py_op_add(ops, "owner_save_checkpoint", SMALL_MLP_OPN_OWNER_SAVE_CHECKPOINT)
    py_op_add(ops, "owner_create", SMALL_MLP_OPN_OWNER_CREATE)
    py_op_add(ops, "owner_close", SMALL_MLP_OPN_OWNER_CLOSE)
    py_op_add(ops, "inference_checkpoint", SMALL_MLP_OPN_INFERENCE_CHECKPOINT)
    py_op_add(ops, "inference_repr", SMALL_MLP_OPN_INFERENCE_REPR)
    py_op_add(ops, "inference_initialize", SMALL_MLP_OPN_INFERENCE_INITIALIZE)
    py_op_add(ops, "inference_weights", SMALL_MLP_OPN_INFERENCE_WEIGHTS)
    py_op_add(ops, "inference_predict", SMALL_MLP_OPN_INFERENCE_PREDICT)
    py_op_add(ops, "initialize", SMALL_MLP_OPN_INITIALIZE)
    py_op_add(ops, "inference_connect", SMALL_MLP_OPN_INFERENCE_CONNECT)
    py_op_add(ops, "from_checkpoint", SMALL_MLP_OPN_FROM_CHECKPOINT)
    py_op_add(ops, "native_provider_info", SMALL_MLP_OPN_NATIVE_PROVIDER_INFO)
    py_op_add(ops, "create", SMALL_MLP_OPN_CREATE)
    py_op_add(ops, "close", SMALL_MLP_OPN_CLOSE)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def small_mlp_keys() raises -> UnsafePointer[SmallMlpKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _SMALL_MLP_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _small_mlp_keys_fill(nk)
    return nk
