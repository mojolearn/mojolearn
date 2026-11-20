# SPDX-License-Identifier: Apache-2.0
# The scaler family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime SCALER_OPN_INIT = 0
comptime SCALER_OPN_GET_PARAMS = 1
comptime SCALER_OPN_SET_PARAMS = 2
comptime SCALER_OPN_IS_FITTED = 3
comptime SCALER_OPN_TRANSFORM = 4
comptime SCALER_OPN_PARTIAL_FIT = 5
comptime SCALER_OPN_FIT_TRANSFORM = 6
comptime SCALER_OPN_SAVE = 7
comptime SCALER_OPN_LOAD = 8


struct ScalerKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___mojolearn_internal__: PythonObject
    var k___mojolearn_user_boundary__: PythonObject
    var k___name__: PythonObject
    var k_backend_fields: PythonObject
    var k_bool_types: PythonObject
    var k_copy: PythonObject
    var k_mean_: PythonObject
    var k_minmax_transform: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_samples_seen_: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_output_context: PythonObject
    var k_sample_weight: PythonObject
    var k_scalar_types: PythonObject
    var k_scale_: PythonObject
    var k_scaler_bindings: PythonObject
    var k_scaler_explicit_host: PythonObject
    var k_scaler_host_contexts: PythonObject
    var k_scaler_training_error: PythonObject
    var k_standard_fit: PythonObject
    var k_standard_transform: PythonObject
    var k_var_: PythonObject
    var k_with_mean: PythonObject
    var k_with_mean_: PythonObject
    var k_with_std: PythonObject
    var k_with_std_: PythonObject
    var k_x_prep_run: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___mojolearn_internal__ = PythonObject(None)
        self.k___mojolearn_user_boundary__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_bool_types = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_mean_ = PythonObject(None)
        self.k_minmax_transform = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_samples_seen_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_sample_weight = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_scale_ = PythonObject(None)
        self.k_scaler_bindings = PythonObject(None)
        self.k_scaler_explicit_host = PythonObject(None)
        self.k_scaler_host_contexts = PythonObject(None)
        self.k_scaler_training_error = PythonObject(None)
        self.k_standard_fit = PythonObject(None)
        self.k_standard_transform = PythonObject(None)
        self.k_var_ = PythonObject(None)
        self.k_with_mean = PythonObject(None)
        self.k_with_mean_ = PythonObject(None)
        self.k_with_std = PythonObject(None)
        self.k_with_std_ = PythonObject(None)
        self.k_x_prep_run = PythonObject(None)


comptime _SCALER_KEYS = SharedSlot[ScalerKeys, "MojolearnScalerKeysV1"]


def _scaler_keys_fill(nk: UnsafePointer[ScalerKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___mojolearn_internal__ = py_key("__mojolearn_internal__")
    nk[].k___mojolearn_user_boundary__ = py_key("__mojolearn_user_boundary__")
    nk[].k___name__ = py_key("__name__")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_bool_types = py_key("bool_types")
    nk[].k_copy = py_key("copy")
    nk[].k_mean_ = py_key("mean_")
    nk[].k_minmax_transform = py_key("minmax_transform")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_samples_seen_ = py_key("n_samples_seen_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_sample_weight = py_key("sample_weight")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_scale_ = py_key("scale_")
    nk[].k_scaler_bindings = py_key("scaler_bindings")
    nk[].k_scaler_explicit_host = py_key("scaler_explicit_host")
    nk[].k_scaler_host_contexts = py_key("scaler_host_contexts")
    nk[].k_scaler_training_error = py_key("scaler_training_error")
    nk[].k_standard_fit = py_key("standard_fit")
    nk[].k_standard_transform = py_key("standard_transform")
    nk[].k_var_ = py_key("var_")
    nk[].k_with_mean = py_key("with_mean")
    nk[].k_with_mean_ = py_key("with_mean_")
    nk[].k_with_std = py_key("with_std")
    nk[].k_with_std_ = py_key("with_std_")
    nk[].k_x_prep_run = py_key("x_prep_run")
    var ops = Python.dict()
    py_op_add(ops, "init", SCALER_OPN_INIT)
    py_op_add(ops, "get_params", SCALER_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", SCALER_OPN_SET_PARAMS)
    py_op_add(ops, "is_fitted", SCALER_OPN_IS_FITTED)
    py_op_add(ops, "transform", SCALER_OPN_TRANSFORM)
    py_op_add(ops, "partial_fit", SCALER_OPN_PARTIAL_FIT)
    py_op_add(ops, "fit_transform", SCALER_OPN_FIT_TRANSFORM)
    py_op_add(ops, "save", SCALER_OPN_SAVE)
    py_op_add(ops, "load", SCALER_OPN_LOAD)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def scaler_keys() raises -> UnsafePointer[ScalerKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _SCALER_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _scaler_keys_fill(nk)
    return nk
