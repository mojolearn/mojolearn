# SPDX-License-Identifier: Apache-2.0
# The mcd family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime MCD_OPN_FIT = 0
comptime MCD_OPN_FIT_PREDICT = 1
comptime MCD_OPN_GET_PRECISION = 2
comptime MCD_OPN_SCORE = 3
comptime MCD_OPN_MAHALANOBIS = 4
comptime MCD_OPN_SCORE_SAMPLES = 5
comptime MCD_OPN_DECISION_FUNCTION = 6
comptime MCD_OPN_PREDICT = 7
comptime MCD_OPN_ELLIPTIC_SCORE = 8
comptime MCD_OPN_DEVICE_SET = 9
comptime MCD_OPN_VENDOR_USED = 10


struct McdKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__draws: PythonObject
    var k__precision: PythonObject
    var k__seed: PythonObject
    var k_array_type: PythonObject
    var k_assume_centered: PythonObject
    var k_contamination: PythonObject
    var k_covariance_: PythonObject
    var k_dist_: PythonObject
    var k_location_: PythonObject
    var k_n_features_in_: PythonObject
    var k_native_bytes_type: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_offset_: PythonObject
    var k_output_context: PythonObject
    var k_elliptic_envelope_class: PythonObject
    var k_precision_: PythonObject
    var k_random_state: PythonObject
    var k_raw_covariance_: PythonObject
    var k_raw_location_: PythonObject
    var k_raw_support_: PythonObject
    var k_store_precision: PythonObject
    var k_support_: PythonObject
    var k_support_fraction: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__draws = PythonObject(None)
        self.k__precision = PythonObject(None)
        self.k__seed = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_assume_centered = PythonObject(None)
        self.k_contamination = PythonObject(None)
        self.k_covariance_ = PythonObject(None)
        self.k_dist_ = PythonObject(None)
        self.k_location_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_native_bytes_type = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_offset_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_elliptic_envelope_class = PythonObject(None)
        self.k_precision_ = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_raw_covariance_ = PythonObject(None)
        self.k_raw_location_ = PythonObject(None)
        self.k_raw_support_ = PythonObject(None)
        self.k_store_precision = PythonObject(None)
        self.k_support_ = PythonObject(None)
        self.k_support_fraction = PythonObject(None)


comptime _MCD_KEYS = SharedSlot[McdKeys, "MojolearnMcdKeysV1"]


def _mcd_keys_fill(nk: UnsafePointer[McdKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__draws = py_key("_draws")
    nk[].k__precision = py_key("_precision")
    nk[].k__seed = py_key("_seed")
    nk[].k_array_type = py_key("array_type")
    nk[].k_assume_centered = py_key("assume_centered")
    nk[].k_contamination = py_key("contamination")
    nk[].k_covariance_ = py_key("covariance_")
    nk[].k_dist_ = py_key("dist_")
    nk[].k_location_ = py_key("location_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_native_bytes_type = py_key("native_bytes_type")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_offset_ = py_key("offset_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_elliptic_envelope_class = py_key("elliptic_envelope_class")
    nk[].k_precision_ = py_key("precision_")
    nk[].k_random_state = py_key("random_state")
    nk[].k_raw_covariance_ = py_key("raw_covariance_")
    nk[].k_raw_location_ = py_key("raw_location_")
    nk[].k_raw_support_ = py_key("raw_support_")
    nk[].k_store_precision = py_key("store_precision")
    nk[].k_support_ = py_key("support_")
    nk[].k_support_fraction = py_key("support_fraction")
    var ops = Python.dict()
    py_op_add(ops, "fit", MCD_OPN_FIT)
    py_op_add(ops, "fit_predict", MCD_OPN_FIT_PREDICT)
    py_op_add(ops, "get_precision", MCD_OPN_GET_PRECISION)
    py_op_add(ops, "score", MCD_OPN_SCORE)
    py_op_add(ops, "mahalanobis", MCD_OPN_MAHALANOBIS)
    py_op_add(ops, "score_samples", MCD_OPN_SCORE_SAMPLES)
    py_op_add(ops, "decision_function", MCD_OPN_DECISION_FUNCTION)
    py_op_add(ops, "predict", MCD_OPN_PREDICT)
    py_op_add(ops, "elliptic_score", MCD_OPN_ELLIPTIC_SCORE)
    py_op_add(ops, "device_set", MCD_OPN_DEVICE_SET)
    py_op_add(ops, "vendor_used", MCD_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def mcd_keys() raises -> UnsafePointer[McdKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _MCD_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _mcd_keys_fill(nk)
    return nk
