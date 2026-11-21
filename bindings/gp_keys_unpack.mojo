# SPDX-License-Identifier: Apache-2.0
# The gp family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct GpKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_L_: PythonObject
    var k_W_sr_: PythonObject
    var k_X_train_: PythonObject
    var k__CODE_MODE: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__MODE_METADATA_CONTEXT: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k__alpha: PythonObject
    var k__kernel: PythonObject
    var k__logdet: PythonObject
    var k__max_iter_predict: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__n_restarts_optimizer: PythonObject
    var k__nodes: PythonObject
    var k__normalize_y: PythonObject
    var k__optimizer: PythonObject
    var k__random_state: PythonObject
    var k__theta_fit: PythonObject
    var k__y_train_mean: PythonObject
    var k__y_train_std: PythonObject
    var k__ydotalpha: PythonObject
    var k_a: PythonObject
    var k_alpha_: PythonObject
    var k_array_class: PythonObject
    var k_b: PythonObject
    var k_backend_fields: PythonObject
    var k_clamped_: PythonObject
    var k_classes_: PythonObject
    var k_combined: PythonObject
    var k_const: PythonObject
    var k_constant_value: PythonObject
    var k_constant_value_bounds: PythonObject
    var k_error: PythonObject
    var k_estimators_: PythonObject
    var k_family_gp_binary_laplace: PythonObject
    var k_family_gp_binary_namespace: PythonObject
    var k_family_gp_classes: PythonObject
    var k_family_gp_default_bounds: PythonObject
    var k_family_gp_kernel_base: PythonObject
    var k_family_gp_kernel_bounds: PythonObject
    var k_family_gp_kernel_nodes: PythonObject
    var k_family_gp_mode_codes: PythonObject
    var k_family_gp_opt_stops: PythonObject
    var k_family_gp_scaler: PythonObject
    var k_family_gp_seed_bound: PythonObject
    var k_family_metrics_prog_context: PythonObject
    var k_gp_native_key: PythonObject
    var k_identical: PythonObject
    var k_info_: PythonObject
    var k_input_copied_: PythonObject
    var k_kernel_: PythonObject
    var k_length_scale: PythonObject
    var k_length_scale_bounds: PythonObject
    var k_log_marginal_likelihood_value_: PythonObject
    var k_metrics_program_native_api_v1: PythonObject
    var k_mode: PythonObject
    var k_modes: PythonObject
    var k_n_clamped_: PythonObject
    var k_n_classes_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_native_program_provider: PythonObject
    var k_native_scaler_provider: PythonObject
    var k_nb_: PythonObject
    var k_numeric_mode: PythonObject
    var k_ndarray: PythonObject
    var k_normalize_y_: PythonObject
    var k_op: PythonObject
    var k_optimizer_runs_: PythonObject
    var k_pi_: PythonObject
    var k_rbf: PythonObject
    var k_route: PythonObject
    var k_scalar_types: PythonObject
    var k_scaler_bindings: PythonObject
    var k_scaler_training_error: PythonObject
    var k_sym: PythonObject
    var k_vendor: PythonObject
    var k_y_train_: PythonObject
    var k__CPU_ONLY: PythonObject
    var k_device: PythonObject
    var k_gp_cpu_bindings: PythonObject
    var k__mojolearn_gp: PythonObject
    var k__mojolearn: PythonObject
    var k_metrics: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_L_ = PythonObject(None)
        self.k_W_sr_ = PythonObject(None)
        self.k_X_train_ = PythonObject(None)
        self.k__CODE_MODE = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__MODE_METADATA_CONTEXT = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k__alpha = PythonObject(None)
        self.k__kernel = PythonObject(None)
        self.k__logdet = PythonObject(None)
        self.k__max_iter_predict = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__n_restarts_optimizer = PythonObject(None)
        self.k__nodes = PythonObject(None)
        self.k__normalize_y = PythonObject(None)
        self.k__optimizer = PythonObject(None)
        self.k__random_state = PythonObject(None)
        self.k__theta_fit = PythonObject(None)
        self.k__y_train_mean = PythonObject(None)
        self.k__y_train_std = PythonObject(None)
        self.k__ydotalpha = PythonObject(None)
        self.k_a = PythonObject(None)
        self.k_alpha_ = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_b = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_clamped_ = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_combined = PythonObject(None)
        self.k_const = PythonObject(None)
        self.k_constant_value = PythonObject(None)
        self.k_constant_value_bounds = PythonObject(None)
        self.k_error = PythonObject(None)
        self.k_estimators_ = PythonObject(None)
        self.k_family_gp_binary_laplace = PythonObject(None)
        self.k_family_gp_binary_namespace = PythonObject(None)
        self.k_family_gp_classes = PythonObject(None)
        self.k_family_gp_default_bounds = PythonObject(None)
        self.k_family_gp_kernel_base = PythonObject(None)
        self.k_family_gp_kernel_bounds = PythonObject(None)
        self.k_family_gp_kernel_nodes = PythonObject(None)
        self.k_family_gp_mode_codes = PythonObject(None)
        self.k_family_gp_opt_stops = PythonObject(None)
        self.k_family_gp_scaler = PythonObject(None)
        self.k_family_gp_seed_bound = PythonObject(None)
        self.k_family_metrics_prog_context = PythonObject(None)
        self.k_gp_native_key = PythonObject(None)
        self.k_identical = PythonObject(None)
        self.k_info_ = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_kernel_ = PythonObject(None)
        self.k_length_scale = PythonObject(None)
        self.k_length_scale_bounds = PythonObject(None)
        self.k_log_marginal_likelihood_value_ = PythonObject(None)
        self.k_metrics_program_native_api_v1 = PythonObject(None)
        self.k_mode = PythonObject(None)
        self.k_modes = PythonObject(None)
        self.k_n_clamped_ = PythonObject(None)
        self.k_n_classes_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_native_program_provider = PythonObject(None)
        self.k_native_scaler_provider = PythonObject(None)
        self.k_nb_ = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_normalize_y_ = PythonObject(None)
        self.k_op = PythonObject(None)
        self.k_optimizer_runs_ = PythonObject(None)
        self.k_pi_ = PythonObject(None)
        self.k_rbf = PythonObject(None)
        self.k_route = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_scaler_bindings = PythonObject(None)
        self.k_scaler_training_error = PythonObject(None)
        self.k_sym = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_y_train_ = PythonObject(None)
        self.k__CPU_ONLY = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_gp_cpu_bindings = PythonObject(None)
        self.k__mojolearn_gp = PythonObject(None)
        self.k__mojolearn = PythonObject(None)
        self.k_metrics = PythonObject(None)


comptime _GP_KEYS = SharedSlot[GpKeys, "MojolearnGpKeysV1"]


def _gp_keys_fill(nk: UnsafePointer[GpKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_L_ = py_key("L_")
    nk[].k_W_sr_ = py_key("W_sr_")
    nk[].k_X_train_ = py_key("X_train_")
    nk[].k__CODE_MODE = py_key("_CODE_MODE")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__MODE_METADATA_CONTEXT = py_key("_MODE_METADATA_CONTEXT")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k__alpha = py_key("_alpha")
    nk[].k__kernel = py_key("_kernel")
    nk[].k__logdet = py_key("_logdet")
    nk[].k__max_iter_predict = py_key("_max_iter_predict")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__n_restarts_optimizer = py_key("_n_restarts_optimizer")
    nk[].k__nodes = py_key("_nodes")
    nk[].k__normalize_y = py_key("_normalize_y")
    nk[].k__optimizer = py_key("_optimizer")
    nk[].k__random_state = py_key("_random_state")
    nk[].k__theta_fit = py_key("_theta_fit")
    nk[].k__y_train_mean = py_key("_y_train_mean")
    nk[].k__y_train_std = py_key("_y_train_std")
    nk[].k__ydotalpha = py_key("_ydotalpha")
    nk[].k_a = py_key("a")
    nk[].k_alpha_ = py_key("alpha_")
    nk[].k_array_class = py_key("array_class")
    nk[].k_b = py_key("b")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_clamped_ = py_key("clamped_")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_combined = py_key("combined")
    nk[].k_const = py_key("const")
    nk[].k_constant_value = py_key("constant_value")
    nk[].k_constant_value_bounds = py_key("constant_value_bounds")
    nk[].k_error = py_key("error")
    nk[].k_estimators_ = py_key("estimators_")
    nk[].k_family_gp_binary_laplace = py_key("family_gp_binary_laplace")
    nk[].k_family_gp_binary_namespace = py_key("family_gp_binary_namespace")
    nk[].k_family_gp_classes = py_key("family_gp_classes")
    nk[].k_family_gp_default_bounds = py_key("family_gp_default_bounds")
    nk[].k_family_gp_kernel_base = py_key("family_gp_kernel_base")
    nk[].k_family_gp_kernel_bounds = py_key("family_gp_kernel_bounds")
    nk[].k_family_gp_kernel_nodes = py_key("family_gp_kernel_nodes")
    nk[].k_family_gp_mode_codes = py_key("family_gp_mode_codes")
    nk[].k_family_gp_opt_stops = py_key("family_gp_opt_stops")
    nk[].k_family_gp_scaler = py_key("family_gp_scaler")
    nk[].k_family_gp_seed_bound = py_key("family_gp_seed_bound")
    nk[].k_family_metrics_prog_context = py_key("family_metrics_prog_context")
    nk[].k_gp_native_key = py_key("gp_native_key")
    nk[].k_identical = py_key("identical")
    nk[].k_info_ = py_key("info_")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_kernel_ = py_key("kernel_")
    nk[].k_length_scale = py_key("length_scale")
    nk[].k_length_scale_bounds = py_key("length_scale_bounds")
    nk[].k_log_marginal_likelihood_value_ = py_key("log_marginal_likelihood_value_")
    nk[].k_metrics_program_native_api_v1 = py_key("metrics_program_native_api_v1")
    nk[].k_mode = py_key("mode")
    nk[].k_modes = py_key("modes")
    nk[].k_n_clamped_ = py_key("n_clamped_")
    nk[].k_n_classes_ = py_key("n_classes_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_native_program_provider = py_key("native_program_provider")
    nk[].k_native_scaler_provider = py_key("native_scaler_provider")
    nk[].k_nb_ = py_key("nb_")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_normalize_y_ = py_key("normalize_y_")
    nk[].k_op = py_key("op")
    nk[].k_optimizer_runs_ = py_key("optimizer_runs_")
    nk[].k_pi_ = py_key("pi_")
    nk[].k_rbf = py_key("rbf")
    nk[].k_route = py_key("route")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_scaler_bindings = py_key("scaler_bindings")
    nk[].k_scaler_training_error = py_key("scaler_training_error")
    nk[].k_sym = py_key("sym")
    nk[].k_vendor = py_key("vendor")
    nk[].k_y_train_ = py_key("y_train_")
    nk[].k__CPU_ONLY = py_key("_CPU_ONLY")
    nk[].k_device = py_key("device")
    nk[].k_gp_cpu_bindings = py_key("gp_cpu_bindings")
    nk[].k__mojolearn_gp = py_key("_mojolearn_gp")
    nk[].k__mojolearn = py_key("_mojolearn")
    nk[].k_metrics = py_key("metrics")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def gp_keys() raises -> UnsafePointer[GpKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _GP_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _gp_keys_fill(nk)
    return nk
