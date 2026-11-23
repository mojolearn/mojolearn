# SPDX-License-Identifier: Apache-2.0
# The sequential family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime SEQUENTIAL_OPN_INITIALIZE = 0
comptime SEQUENTIAL_OPN_CONSTRUCT = 1
comptime SEQUENTIAL_OPN_ADD = 2
comptime SEQUENTIAL_OPN_APPEND = 3
comptime SEQUENTIAL_OPN_COMPILE = 4
comptime SEQUENTIAL_OPN_LAYERS = 5
comptime SEQUENTIAL_OPN_PROGRAM = 6
comptime SEQUENTIAL_OPN_TABLE = 7
comptime SEQUENTIAL_OPN_FIT = 8
comptime SEQUENTIAL_OPN_PREDICT = 9
comptime SEQUENTIAL_OPN_PREDICT_PROBA = 10
comptime SEQUENTIAL_OPN_FORWARD = 11
comptime SEQUENTIAL_OPN_TARGETS = 12
comptime SEQUENTIAL_OPN_LOAD_WEIGHTS = 13
comptime SEQUENTIAL_OPN_SAVE = 14
comptime SEQUENTIAL_OPN_LOAD_CHECKPOINT = 15
# lane dw-autograd-seq (2026-11-21): the vendor of the binary the model's route runs on
comptime SEQUENTIAL_OPN_VENDOR_USED = 16
# lane api-torch round 2 (2026-11-21): state_dict / parameters /
# named_parameters (one export, three result forms) and to(device, dtype)
comptime SEQUENTIAL_OPN_EXPORT_WEIGHTS = 17
comptime SEQUENTIAL_OPN_TO = 18


struct SequentialKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___bases__: PythonObject
    var k___buffer__: PythonObject
    var k___dict__: PythonObject
    var k___release_buffer__: PythonObject
    var k__curve: PythonObject
    var k__fit_batch: PythonObject
    var k__fit_loss: PythonObject
    var k__fit_shuffle: PythonObject
    var k__fit_state: PythonObject
    var k__imported: PythonObject
    var k__layers: PythonObject
    var k__options: PythonObject
    var k__sample_shape: PythonObject
    var k__sequential_checked_vendor: PythonObject
    var k__target: PythonObject
    var k__y_1d: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_backend_fields: PythonObject
    var k_best_epoch_: PythonObject
    var k_bindings: PythonObject
    var k_buffer_codes: PythonObject
    var k_buffer_owner_types: PythonObject
    var k_class_dtypes: PythonObject
    var k_classes_: PythonObject
    var k_ctypes_array_base: PythonObject
    var k_data: PythonObject
    var k_defaultdict_class: PythonObject
    var k_device: PythonObject
    var k_epochs_run_: PythonObject
    var k_eps: PythonObject
    var k_f1: PythonObject
    var k_f2: PythonObject
    var k_flags: PythonObject
    var k_float32: PythonObject
    var k_gamma: PythonObject
    var k_huber_delta: PythonObject
    var k_input_shape: PythonObject
    var k_int64: PythonObject
    var k_integer_types: PythonObject
    var k_layer_codes: PythonObject
    var k_layer_names: PythonObject
    var k_loss: PythonObject
    var k_loss_codes: PythonObject
    var k_loss_curve_: PythonObject
    var k_loss_names: PythonObject
    var k_lr: PythonObject
    var k_mapping_types: PythonObject
    var k_model_fields: PythonObject
    var k_model_native_api_v1: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_iter_no_change: PythonObject
    var k_n_outputs_: PythonObject
    var k_native_bytearray_type: PythonObject
    var k_native_default_factories: PythonObject
    var k_native_model_provider: PythonObject
    var k_native_value_types: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_dtypes: PythonObject
    var k_numeric_mode: PythonObject
    var k_opt: PythonObject
    var k_optimizer_codes: PythonObject
    var k_optimizer_m_: PythonObject
    var k_optimizer_v_: PythonObject
    var k_params_: PythonObject
    var k_scalar_types: PythonObject
    var k_sched: PythonObject
    var k_schedule_codes: PythonObject
    var k_sequential_class: PythonObject
    var k_sequential_cpu_bindings: PythonObject
    var k_step_size: PythonObject
    var k_tol: PythonObject
    var k_uint64_max: PythonObject
    var k_uname: PythonObject
    var k_validation_loss_curve_: PythonObject
    var k_vendor: PythonObject
    var k_version: PythonObject
    var k_weight_decay: PythonObject
    var k_x_sequence_numeric_mode: PythonObject
    var k_x_sequence_vendor: PythonObject
    # nn-losses begin
    var k_reduction: PythonObject
    var k_reduction_codes: PythonObject
    # nn-losses end

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___bases__ = PythonObject(None)
        self.k___buffer__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___release_buffer__ = PythonObject(None)
        self.k__curve = PythonObject(None)
        self.k__fit_batch = PythonObject(None)
        self.k__fit_loss = PythonObject(None)
        self.k__fit_shuffle = PythonObject(None)
        self.k__fit_state = PythonObject(None)
        self.k__imported = PythonObject(None)
        self.k__layers = PythonObject(None)
        self.k__options = PythonObject(None)
        self.k__sample_shape = PythonObject(None)
        self.k__sequential_checked_vendor = PythonObject(None)
        self.k__target = PythonObject(None)
        self.k__y_1d = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_best_epoch_ = PythonObject(None)
        self.k_bindings = PythonObject(None)
        self.k_buffer_codes = PythonObject(None)
        self.k_buffer_owner_types = PythonObject(None)
        self.k_class_dtypes = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_ctypes_array_base = PythonObject(None)
        self.k_data = PythonObject(None)
        self.k_defaultdict_class = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_epochs_run_ = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_f1 = PythonObject(None)
        self.k_f2 = PythonObject(None)
        self.k_flags = PythonObject(None)
        self.k_float32 = PythonObject(None)
        self.k_gamma = PythonObject(None)
        self.k_huber_delta = PythonObject(None)
        self.k_input_shape = PythonObject(None)
        self.k_int64 = PythonObject(None)
        self.k_integer_types = PythonObject(None)
        self.k_layer_codes = PythonObject(None)
        self.k_layer_names = PythonObject(None)
        self.k_loss = PythonObject(None)
        self.k_loss_codes = PythonObject(None)
        self.k_loss_curve_ = PythonObject(None)
        self.k_loss_names = PythonObject(None)
        self.k_lr = PythonObject(None)
        self.k_mapping_types = PythonObject(None)
        self.k_model_fields = PythonObject(None)
        self.k_model_native_api_v1 = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_iter_no_change = PythonObject(None)
        self.k_n_outputs_ = PythonObject(None)
        self.k_native_bytearray_type = PythonObject(None)
        self.k_native_default_factories = PythonObject(None)
        self.k_native_model_provider = PythonObject(None)
        self.k_native_value_types = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_dtypes = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_opt = PythonObject(None)
        self.k_optimizer_codes = PythonObject(None)
        self.k_optimizer_m_ = PythonObject(None)
        self.k_optimizer_v_ = PythonObject(None)
        self.k_params_ = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_sched = PythonObject(None)
        self.k_schedule_codes = PythonObject(None)
        self.k_sequential_class = PythonObject(None)
        self.k_sequential_cpu_bindings = PythonObject(None)
        self.k_step_size = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_uint64_max = PythonObject(None)
        self.k_uname = PythonObject(None)
        self.k_validation_loss_curve_ = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_version = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_x_sequence_numeric_mode = PythonObject(None)
        self.k_x_sequence_vendor = PythonObject(None)
        # nn-losses begin
        self.k_reduction = PythonObject(None)
        self.k_reduction_codes = PythonObject(None)
        # nn-losses end


comptime _SEQUENTIAL_KEYS = SharedSlot[SequentialKeys, "MojolearnSequentialKeysV1"]


def _sequential_keys_fill(nk: UnsafePointer[SequentialKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___bases__ = py_key("__bases__")
    nk[].k___buffer__ = py_key("__buffer__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___release_buffer__ = py_key("__release_buffer__")
    nk[].k__curve = py_key("_curve")
    nk[].k__fit_batch = py_key("_fit_batch")
    nk[].k__fit_loss = py_key("_fit_loss")
    nk[].k__fit_shuffle = py_key("_fit_shuffle")
    nk[].k__fit_state = py_key("_fit_state")
    nk[].k__imported = py_key("_imported")
    nk[].k__layers = py_key("_layers")
    nk[].k__options = py_key("_options")
    nk[].k__sample_shape = py_key("_sample_shape")
    nk[].k__sequential_checked_vendor = py_key("_sequential_checked_vendor")
    nk[].k__target = py_key("_target")
    nk[].k__y_1d = py_key("_y_1d")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_best_epoch_ = py_key("best_epoch_")
    nk[].k_bindings = py_key("bindings")
    nk[].k_buffer_codes = py_key("buffer_codes")
    nk[].k_buffer_owner_types = py_key("buffer_owner_types")
    nk[].k_class_dtypes = py_key("class_dtypes")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_ctypes_array_base = py_key("ctypes_array_base")
    nk[].k_data = py_key("data")
    nk[].k_defaultdict_class = py_key("defaultdict_class")
    nk[].k_device = py_key("device")
    nk[].k_epochs_run_ = py_key("epochs_run_")
    nk[].k_eps = py_key("eps")
    nk[].k_f1 = py_key("f1")
    nk[].k_f2 = py_key("f2")
    nk[].k_flags = py_key("flags")
    nk[].k_float32 = py_key("float32")
    nk[].k_gamma = py_key("gamma")
    nk[].k_huber_delta = py_key("huber_delta")
    nk[].k_input_shape = py_key("input_shape")
    nk[].k_int64 = py_key("int64")
    nk[].k_integer_types = py_key("integer_types")
    nk[].k_layer_codes = py_key("layer_codes")
    nk[].k_layer_names = py_key("layer_names")
    nk[].k_loss = py_key("loss")
    nk[].k_loss_codes = py_key("loss_codes")
    nk[].k_loss_curve_ = py_key("loss_curve_")
    nk[].k_loss_names = py_key("loss_names")
    nk[].k_lr = py_key("lr")
    nk[].k_mapping_types = py_key("mapping_types")
    nk[].k_model_fields = py_key("model_fields")
    nk[].k_model_native_api_v1 = py_key("model_native_api_v1")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_iter_no_change = py_key("n_iter_no_change")
    nk[].k_n_outputs_ = py_key("n_outputs_")
    nk[].k_native_bytearray_type = py_key("native_bytearray_type")
    nk[].k_native_default_factories = py_key("native_default_factories")
    nk[].k_native_model_provider = py_key("native_model_provider")
    nk[].k_native_value_types = py_key("native_value_types")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_dtypes = py_key("numeric_dtypes")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_opt = py_key("opt")
    nk[].k_optimizer_codes = py_key("optimizer_codes")
    nk[].k_optimizer_m_ = py_key("optimizer_m_")
    nk[].k_optimizer_v_ = py_key("optimizer_v_")
    nk[].k_params_ = py_key("params_")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_sched = py_key("sched")
    nk[].k_schedule_codes = py_key("schedule_codes")
    nk[].k_sequential_class = py_key("sequential_class")
    nk[].k_sequential_cpu_bindings = py_key("sequential_cpu_bindings")
    nk[].k_step_size = py_key("step_size")
    nk[].k_tol = py_key("tol")
    nk[].k_uint64_max = py_key("uint64_max")
    nk[].k_uname = py_key("uname")
    nk[].k_validation_loss_curve_ = py_key("validation_loss_curve_")
    nk[].k_vendor = py_key("vendor")
    nk[].k_version = py_key("version")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_x_sequence_numeric_mode = py_key("x_sequence_numeric_mode")
    nk[].k_x_sequence_vendor = py_key("x_sequence_vendor")
    # nn-losses begin
    nk[].k_reduction = py_key("reduction")
    nk[].k_reduction_codes = py_key("reduction_codes")
    # nn-losses end
    var ops = Python.dict()
    py_op_add(ops, "initialize", SEQUENTIAL_OPN_INITIALIZE)
    py_op_add(ops, "construct", SEQUENTIAL_OPN_CONSTRUCT)
    py_op_add(ops, "add", SEQUENTIAL_OPN_ADD)
    py_op_add(ops, "append", SEQUENTIAL_OPN_APPEND)
    py_op_add(ops, "compile", SEQUENTIAL_OPN_COMPILE)
    py_op_add(ops, "layers", SEQUENTIAL_OPN_LAYERS)
    py_op_add(ops, "program", SEQUENTIAL_OPN_PROGRAM)
    py_op_add(ops, "table", SEQUENTIAL_OPN_TABLE)
    py_op_add(ops, "fit", SEQUENTIAL_OPN_FIT)
    py_op_add(ops, "predict", SEQUENTIAL_OPN_PREDICT)
    py_op_add(ops, "predict_proba", SEQUENTIAL_OPN_PREDICT_PROBA)
    py_op_add(ops, "forward", SEQUENTIAL_OPN_FORWARD)
    py_op_add(ops, "targets", SEQUENTIAL_OPN_TARGETS)
    py_op_add(ops, "load_weights", SEQUENTIAL_OPN_LOAD_WEIGHTS)
    py_op_add(ops, "save", SEQUENTIAL_OPN_SAVE)
    py_op_add(ops, "load_checkpoint", SEQUENTIAL_OPN_LOAD_CHECKPOINT)
    py_op_add(ops, "vendor_used", SEQUENTIAL_OPN_VENDOR_USED)
    py_op_add(ops, "export_weights", SEQUENTIAL_OPN_EXPORT_WEIGHTS)
    py_op_add(ops, "to", SEQUENTIAL_OPN_TO)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def sequential_keys() raises -> UnsafePointer[SequentialKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _SEQUENTIAL_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _sequential_keys_fill(nk)
    return nk
