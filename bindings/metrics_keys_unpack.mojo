# SPDX-License-Identifier: Apache-2.0
# The metrics family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct MetricsKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_UndefinedMetricWarning: PythonObject
    var k___name__: PythonObject
    var k_accuracy: PythonObject
    var k_array_type: PythonObject
    var k_average_precision: PythonObject
    var k_bool_types: PythonObject
    var k_digits: PythonObject
    var k_family_metrics_prog_context: PythonObject
    var k_family_metrics_undefined: PythonObject
    var k_mean_absolute_error: PythonObject
    var k_mean_squared_error: PythonObject
    var k_modules: PythonObject
    var k_native_bytes_type: PythonObject
    var k_native_module_type: PythonObject
    var k_ndarray: PythonObject
    var k_r2_score: PythonObject
    var k_roc_auc: PythonObject
    var k_scalar_types: PythonObject
    var k_width: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_UndefinedMetricWarning = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k_accuracy = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_average_precision = PythonObject(None)
        self.k_bool_types = PythonObject(None)
        self.k_digits = PythonObject(None)
        self.k_family_metrics_prog_context = PythonObject(None)
        self.k_family_metrics_undefined = PythonObject(None)
        self.k_mean_absolute_error = PythonObject(None)
        self.k_mean_squared_error = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_native_bytes_type = PythonObject(None)
        self.k_native_module_type = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_r2_score = PythonObject(None)
        self.k_roc_auc = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_width = PythonObject(None)


comptime _METRICS_KEYS = SharedSlot[MetricsKeys, "MojolearnMetricsKeysV1"]


def _metrics_keys_fill(nk: UnsafePointer[MetricsKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_UndefinedMetricWarning = py_key("UndefinedMetricWarning")
    nk[].k___name__ = py_key("__name__")
    nk[].k_accuracy = py_key("accuracy")
    nk[].k_array_type = py_key("array_type")
    nk[].k_average_precision = py_key("average_precision")
    nk[].k_bool_types = py_key("bool_types")
    nk[].k_digits = py_key("digits")
    nk[].k_family_metrics_prog_context = py_key("family_metrics_prog_context")
    nk[].k_family_metrics_undefined = py_key("family_metrics_undefined")
    nk[].k_mean_absolute_error = py_key("mean_absolute_error")
    nk[].k_mean_squared_error = py_key("mean_squared_error")
    nk[].k_modules = py_key("modules")
    nk[].k_native_bytes_type = py_key("native_bytes_type")
    nk[].k_native_module_type = py_key("native_module_type")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_r2_score = py_key("r2_score")
    nk[].k_roc_auc = py_key("roc_auc")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_width = py_key("width")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def metrics_keys() raises -> UnsafePointer[MetricsKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _METRICS_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _metrics_keys_fill(nk)
    return nk
