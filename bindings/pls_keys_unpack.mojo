# SPDX-License-Identifier: Apache-2.0
# The pls family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime PLS_OPN_FIT = 0
comptime PLS_OPN_FIT_TRANSFORM = 1
comptime PLS_OPN_SCORE = 2
comptime PLS_OPN_PREDICT = 3
comptime PLS_OPN_TRANSFORM = 4
comptime PLS_OPN_INVERSE_TRANSFORM = 5
comptime PLS_OPN_VENDOR_USED = 6


struct PlsKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__deflation: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__pls_canonical: PythonObject
    var k__pls_name: PythonObject
    var k__pls_regression: PythonObject
    var k__pmode: PythonObject
    var k__predict_1d: PythonObject
    var k__x_scores: PythonObject
    var k__y_scores: PythonObject
    var k_algorithm: PythonObject
    var k_backend_fields: PythonObject
    var k_coef_: PythonObject
    var k_copy: PythonObject
    var k_feature_names_in_: PythonObject
    var k_max_iter: PythonObject
    var k_n_components: PythonObject
    var k_n_features_in_: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_output_context: PythonObject
    var k_pls_canonical_class: PythonObject
    var k_pls_metric_owner: PythonObject
    var k_pls_regression_class: PythonObject
    var k_scalar_types: PythonObject
    var k_scale: PythonObject
    var k_tol: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__deflation = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__pls_canonical = PythonObject(None)
        self.k__pls_name = PythonObject(None)
        self.k__pls_regression = PythonObject(None)
        self.k__pmode = PythonObject(None)
        self.k__predict_1d = PythonObject(None)
        self.k__x_scores = PythonObject(None)
        self.k__y_scores = PythonObject(None)
        self.k_algorithm = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_coef_ = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_feature_names_in_ = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_pls_canonical_class = PythonObject(None)
        self.k_pls_metric_owner = PythonObject(None)
        self.k_pls_regression_class = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_scale = PythonObject(None)
        self.k_tol = PythonObject(None)


comptime _PLS_KEYS = SharedSlot[PlsKeys, "MojolearnPlsKeysV1"]


def _pls_keys_fill(nk: UnsafePointer[PlsKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__deflation = py_key("_deflation")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__pls_canonical = py_key("_pls_canonical")
    nk[].k__pls_name = py_key("_pls_name")
    nk[].k__pls_regression = py_key("_pls_regression")
    nk[].k__pmode = py_key("_pmode")
    nk[].k__predict_1d = py_key("_predict_1d")
    nk[].k__x_scores = py_key("_x_scores")
    nk[].k__y_scores = py_key("_y_scores")
    nk[].k_algorithm = py_key("algorithm")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_coef_ = py_key("coef_")
    nk[].k_copy = py_key("copy")
    nk[].k_feature_names_in_ = py_key("feature_names_in_")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_pls_canonical_class = py_key("pls_canonical_class")
    nk[].k_pls_metric_owner = py_key("pls_metric_owner")
    nk[].k_pls_regression_class = py_key("pls_regression_class")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_scale = py_key("scale")
    nk[].k_tol = py_key("tol")
    var ops = Python.dict()
    py_op_add(ops, "fit", PLS_OPN_FIT)
    py_op_add(ops, "fit_transform", PLS_OPN_FIT_TRANSFORM)
    py_op_add(ops, "score", PLS_OPN_SCORE)
    py_op_add(ops, "predict", PLS_OPN_PREDICT)
    py_op_add(ops, "transform", PLS_OPN_TRANSFORM)
    py_op_add(ops, "inverse_transform", PLS_OPN_INVERSE_TRANSFORM)
    py_op_add(ops, "vendor_used", PLS_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def pls_keys() raises -> UnsafePointer[PlsKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PLS_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _pls_keys_fill(nk)
    return nk
