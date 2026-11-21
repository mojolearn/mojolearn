# SPDX-License-Identifier: Apache-2.0
# The qn family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime QN_OPN_INITIALIZE = 0
comptime QN_OPN_FIT = 1
comptime QN_OPN_PREDICT = 2
comptime QN_OPN_SCORE = 3
comptime QN_OPN_VENDOR_USED = 4
comptime QN_OPN_MODE_USED = 5


struct QnKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__module: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__w: PythonObject
    var k_backend_fields: PythonObject
    var k_classical_metrics: PythonObject
    var k_coef_: PythonObject
    var k_device: PythonObject
    var k_delta: PythonObject
    var k_fit_intercept: PythonObject
    var k_input_copied_: PythonObject
    var k_intercept_: PythonObject
    var k_l1_strength: PythonObject
    var k_l2_strength: PythonObject
    var k_lbfgs_memory: PythonObject
    var k_linesearch_max_iter: PythonObject
    var k_loss: PythonObject
    var k_max_iter: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_native_key: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_mode: PythonObject
    var k_objective_: PythonObject
    var k_output_context: PythonObject
    var k_penalty_normalized: PythonObject
    var k_qn_bindings: PythonObject
    var k_qn_cpu_bindings: PythonObject
    var k_qn_fold_key: PythonObject
    var k_qn_losses: PythonObject
    var k_qn_metric_admitted: PythonObject
    var k_qn_parameters: PythonObject
    var k_qn_public_class: PythonObject
    var k_retcode_: PythonObject
    var k_tol: PythonObject
    var k_vendor: PythonObject
    var k_warm_start: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__module = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__w = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_classical_metrics = PythonObject(None)
        self.k_coef_ = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_delta = PythonObject(None)
        self.k_fit_intercept = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_intercept_ = PythonObject(None)
        self.k_l1_strength = PythonObject(None)
        self.k_l2_strength = PythonObject(None)
        self.k_lbfgs_memory = PythonObject(None)
        self.k_linesearch_max_iter = PythonObject(None)
        self.k_loss = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_native_key = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_objective_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_penalty_normalized = PythonObject(None)
        self.k_qn_bindings = PythonObject(None)
        self.k_qn_cpu_bindings = PythonObject(None)
        self.k_qn_fold_key = PythonObject(None)
        self.k_qn_losses = PythonObject(None)
        self.k_qn_metric_admitted = PythonObject(None)
        self.k_qn_parameters = PythonObject(None)
        self.k_qn_public_class = PythonObject(None)
        self.k_retcode_ = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_warm_start = PythonObject(None)


comptime _QN_KEYS = SharedSlot[QnKeys, "MojolearnQnKeysV1"]


def _qn_keys_fill(nk: UnsafePointer[QnKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__module = py_key("_module")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__w = py_key("_w")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_classical_metrics = py_key("classical_metrics")
    nk[].k_coef_ = py_key("coef_")
    nk[].k_device = py_key("device")
    nk[].k_delta = py_key("delta")
    nk[].k_fit_intercept = py_key("fit_intercept")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_intercept_ = py_key("intercept_")
    nk[].k_l1_strength = py_key("l1_strength")
    nk[].k_l2_strength = py_key("l2_strength")
    nk[].k_lbfgs_memory = py_key("lbfgs_memory")
    nk[].k_linesearch_max_iter = py_key("linesearch_max_iter")
    nk[].k_loss = py_key("loss")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_native_key = py_key("native_key")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_objective_ = py_key("objective_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_penalty_normalized = py_key("penalty_normalized")
    nk[].k_qn_bindings = py_key("qn_bindings")
    nk[].k_qn_cpu_bindings = py_key("qn_cpu_bindings")
    nk[].k_qn_fold_key = py_key("qn_fold_key")
    nk[].k_qn_losses = py_key("qn_losses")
    nk[].k_qn_metric_admitted = py_key("qn_metric_admitted")
    nk[].k_qn_parameters = py_key("qn_parameters")
    nk[].k_qn_public_class = py_key("qn_public_class")
    nk[].k_retcode_ = py_key("retcode_")
    nk[].k_tol = py_key("tol")
    nk[].k_vendor = py_key("vendor")
    nk[].k_warm_start = py_key("warm_start")
    var ops = Python.dict()
    py_op_add(ops, "initialize", QN_OPN_INITIALIZE)
    py_op_add(ops, "fit", QN_OPN_FIT)
    py_op_add(ops, "predict", QN_OPN_PREDICT)
    py_op_add(ops, "score", QN_OPN_SCORE)
    py_op_add(ops, "vendor_used", QN_OPN_VENDOR_USED)
    py_op_add(ops, "numeric_mode_used", QN_OPN_MODE_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def qn_keys() raises -> UnsafePointer[QnKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _QN_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _qn_keys_fill(nk)
    return nk
