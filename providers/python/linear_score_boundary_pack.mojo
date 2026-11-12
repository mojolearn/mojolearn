# SPDX-License-Identifier: Apache-2.0
"""Unpack/pin R² inputs and pack the native scalar result."""
# The PACK half of the former `providers/python/linear_score_boundary.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/linear_score_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_input import admit_x
from bindings.classical_arrays_door import classical_address
from bindings.classical_validate import classical_check_x_snapshot, classical_estimator_name
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.sequential_description import sequential_fields
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients
from core.linear_score_run import linear_score_run, linear_r2_run, linear_r2_check_target
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_decomp.descriptor_schema import xd_vendor_code
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT

from providers.python.linear_entry_boundary import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_METRICS, linear_table_entry


from providers.python.linear_score_boundary_unpack import (
    _linear_metrics_provider,
)


def classical_r2_score(prediction: PythonObject, target_source: PythonObject,
    context: PythonObject) raises -> PythonObject:
    var rows = Int(py=prediction.size)
    var target = admit_x(target_source, 1, "y", context)
    var target_rows = Int(py=target.size)
    linear_r2_check_target(rows, target_rows)
    var target_address = Int(py=classical_address(target))
    var prediction_address = Int(py=classical_address(prediction))
    var provider = _linear_metrics_provider(context)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var vendor = xd_vendor_code(String(py_get(context, "classical_metric_vendor", context["vendor"])))
    var mode = 1 if String(context["backend_fields"]["_DEFAULT_MODE"]) == "identical" else 0
    var result = Float64(0.0)
    var py = Python()
    try:
        with GILReleased(py):
            result = linear_r2_run(entry, prediction_address, rows, target_address, target_rows, vendor, mode)
    finally:
        _ = prediction
        _ = target
        _ = provider
    return PythonObject(result)
