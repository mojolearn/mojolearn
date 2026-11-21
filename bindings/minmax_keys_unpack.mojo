# SPDX-License-Identifier: Apache-2.0
# The minmax family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime MINMAX_OPN_INIT = 0
comptime MINMAX_OPN_GET_PARAMS = 1
comptime MINMAX_OPN_SET_PARAMS = 2
comptime MINMAX_OPN_IS_FITTED = 3
comptime MINMAX_OPN_FIT = 4
comptime MINMAX_OPN_PARTIAL_FIT = 5
comptime MINMAX_OPN_FIT_TRANSFORM = 6
comptime MINMAX_OPN_TRANSFORM = 7
comptime MINMAX_OPN_SAVE = 8
comptime MINMAX_OPN_LOAD = 9
comptime MINMAX_OPN_VENDOR_USED = 10


struct MinMaxKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k_archive_memoryview_type: PythonObject
    var k_archive_python_micro: PythonObject
    var k_archive_python_minor: PythonObject
    var k_backend_fields: PythonObject
    var k_bool_types: PythonObject
    var k_clip: PythonObject
    var k_clip_: PythonObject
    var k_close: PythonObject
    var k_copy: PythonObject
    var k_device: PythonObject
    var k_data_max_: PythonObject
    var k_data_min_: PythonObject
    var k_feature_range: PythonObject
    var k_feature_range_: PythonObject
    var k_host_archive_dict_descriptor_type: PythonObject
    var k_host_archive_finalizer_descriptor: PythonObject
    var k_host_archive_peer_finalizer_descriptor: PythonObject
    var k_min_: PythonObject
    var k_minmax_fit: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_samples_seen_: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_output_context: PythonObject
    var k_scalar_types: PythonObject
    var k_scale_: PythonObject
    var k_scaler_class: PythonObject
    var k_scaler_explicit_host: PythonObject
    var k_scaler_host_contexts: PythonObject
    var k_scaler_training_error: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k_archive_memoryview_type = PythonObject(None)
        self.k_archive_python_micro = PythonObject(None)
        self.k_archive_python_minor = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_bool_types = PythonObject(None)
        self.k_clip = PythonObject(None)
        self.k_clip_ = PythonObject(None)
        self.k_close = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_data_max_ = PythonObject(None)
        self.k_data_min_ = PythonObject(None)
        self.k_feature_range = PythonObject(None)
        self.k_feature_range_ = PythonObject(None)
        self.k_host_archive_dict_descriptor_type = PythonObject(None)
        self.k_host_archive_finalizer_descriptor = PythonObject(None)
        self.k_host_archive_peer_finalizer_descriptor = PythonObject(None)
        self.k_min_ = PythonObject(None)
        self.k_minmax_fit = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_samples_seen_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_scale_ = PythonObject(None)
        self.k_scaler_class = PythonObject(None)
        self.k_scaler_explicit_host = PythonObject(None)
        self.k_scaler_host_contexts = PythonObject(None)
        self.k_scaler_training_error = PythonObject(None)


comptime _MINMAX_KEYS = SharedSlot[MinMaxKeys, "MojolearnMinMaxKeysV1"]


def _minmax_keys_fill(nk: UnsafePointer[MinMaxKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k_archive_memoryview_type = py_key("archive_memoryview_type")
    nk[].k_archive_python_micro = py_key("archive_python_micro")
    nk[].k_archive_python_minor = py_key("archive_python_minor")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_bool_types = py_key("bool_types")
    nk[].k_clip = py_key("clip")
    nk[].k_clip_ = py_key("clip_")
    nk[].k_close = py_key("close")
    nk[].k_copy = py_key("copy")
    nk[].k_device = py_key("device")
    nk[].k_data_max_ = py_key("data_max_")
    nk[].k_data_min_ = py_key("data_min_")
    nk[].k_feature_range = py_key("feature_range")
    nk[].k_feature_range_ = py_key("feature_range_")
    nk[].k_host_archive_dict_descriptor_type = py_key("host_archive_dict_descriptor_type")
    nk[].k_host_archive_finalizer_descriptor = py_key("host_archive_finalizer_descriptor")
    nk[].k_host_archive_peer_finalizer_descriptor = py_key("host_archive_peer_finalizer_descriptor")
    nk[].k_min_ = py_key("min_")
    nk[].k_minmax_fit = py_key("minmax_fit")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_samples_seen_ = py_key("n_samples_seen_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_scale_ = py_key("scale_")
    nk[].k_scaler_class = py_key("scaler_class")
    nk[].k_scaler_explicit_host = py_key("scaler_explicit_host")
    nk[].k_scaler_host_contexts = py_key("scaler_host_contexts")
    nk[].k_scaler_training_error = py_key("scaler_training_error")
    var ops = Python.dict()
    py_op_add(ops, "init", MINMAX_OPN_INIT)
    py_op_add(ops, "get_params", MINMAX_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", MINMAX_OPN_SET_PARAMS)
    py_op_add(ops, "is_fitted", MINMAX_OPN_IS_FITTED)
    py_op_add(ops, "fit", MINMAX_OPN_FIT)
    py_op_add(ops, "partial_fit", MINMAX_OPN_PARTIAL_FIT)
    py_op_add(ops, "fit_transform", MINMAX_OPN_FIT_TRANSFORM)
    py_op_add(ops, "transform", MINMAX_OPN_TRANSFORM)
    py_op_add(ops, "save", MINMAX_OPN_SAVE)
    py_op_add(ops, "load", MINMAX_OPN_LOAD)
    py_op_add(ops, "vendor_used", MINMAX_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def minmax_keys() raises -> UnsafePointer[MinMaxKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _MINMAX_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _minmax_keys_fill(nk)
    return nk
