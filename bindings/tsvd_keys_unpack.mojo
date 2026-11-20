# SPDX-License-Identifier: Apache-2.0
# The tsvd family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime TSVD_OPN_FIT = 0
comptime TSVD_OPN_FIT_TRANSFORM = 1
comptime TSVD_OPN_TRANSFORM = 2
comptime TSVD_OPN_INVERSE_TRANSFORM = 3


struct TsvdKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__HOST_INFERENCE_ONLY: PythonObject
    var k___and__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k_algorithm: PythonObject
    var k_array_class: PythonObject
    var k_backend_fields: PythonObject
    var k_components_: PythonObject
    var k_explained_variance_: PythonObject
    var k_explained_variance_ratio_: PythonObject
    var k_input_copied_: PythonObject
    var k_n_components: PythonObject
    var k_n_components_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter: PythonObject
    var k_n_oversamples: PythonObject
    var k_numeric_mode: PythonObject
    var k_output_context: PythonObject
    var k_power_iteration_normalizer: PythonObject
    var k_random_state: PythonObject
    var k_singular_values_: PythonObject
    var k_tsvd_class: PythonObject
    var k_tsvd_fold_factory: PythonObject
    var k_tsvd_host_factory: PythonObject
    var k_vendor: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__HOST_INFERENCE_ONLY = PythonObject(None)
        self.k___and__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k_algorithm = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_components_ = PythonObject(None)
        self.k_explained_variance_ = PythonObject(None)
        self.k_explained_variance_ratio_ = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_components_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter = PythonObject(None)
        self.k_n_oversamples = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_power_iteration_normalizer = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_singular_values_ = PythonObject(None)
        self.k_tsvd_class = PythonObject(None)
        self.k_tsvd_fold_factory = PythonObject(None)
        self.k_tsvd_host_factory = PythonObject(None)
        self.k_vendor = PythonObject(None)


comptime _TSVD_KEYS = SharedSlot[TsvdKeys, "MojolearnTsvdKeysV1"]


def _tsvd_keys_fill(nk: UnsafePointer[TsvdKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__HOST_INFERENCE_ONLY = py_key("_HOST_INFERENCE_ONLY")
    nk[].k___and__ = py_key("__and__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k_algorithm = py_key("algorithm")
    nk[].k_array_class = py_key("array_class")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_components_ = py_key("components_")
    nk[].k_explained_variance_ = py_key("explained_variance_")
    nk[].k_explained_variance_ratio_ = py_key("explained_variance_ratio_")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_components_ = py_key("n_components_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter = py_key("n_iter")
    nk[].k_n_oversamples = py_key("n_oversamples")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_output_context = py_key("output_context")
    nk[].k_power_iteration_normalizer = py_key("power_iteration_normalizer")
    nk[].k_random_state = py_key("random_state")
    nk[].k_singular_values_ = py_key("singular_values_")
    nk[].k_tsvd_class = py_key("tsvd_class")
    nk[].k_tsvd_fold_factory = py_key("tsvd_fold_factory")
    nk[].k_tsvd_host_factory = py_key("tsvd_host_factory")
    nk[].k_vendor = py_key("vendor")
    var ops = Python.dict()
    py_op_add(ops, "fit", TSVD_OPN_FIT)
    py_op_add(ops, "fit_transform", TSVD_OPN_FIT_TRANSFORM)
    py_op_add(ops, "transform", TSVD_OPN_TRANSFORM)
    py_op_add(ops, "inverse_transform", TSVD_OPN_INVERSE_TRANSFORM)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def tsvd_keys() raises -> UnsafePointer[TsvdKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _TSVD_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _tsvd_keys_fill(nk)
    return nk
