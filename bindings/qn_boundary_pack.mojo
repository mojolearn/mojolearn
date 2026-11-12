# SPDX-License-Identifier: Apache-2.0
# UNPACK and PACK of QNRegressor's public fit and predict: the only part of
# the QN door that reads or writes Python objects. UNPACK reads the
# estimator's fields and admits X, y and the weights into one typed request
# (glm/qn_door_run.mojo) plus the Python owners that keep every addressed
# buffer alive; PACK publishes the fitted attributes. The registered door,
# bindings/qn_native.mojo, calls UNPACK, releases the GIL for RUN, then PACK
# (lane doorsplit-linear, 2026-11-11).
# The PACK half of the former `bindings/qn_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/qn_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_address, classical_typed, classical_dtype_str, classical_c_order
from bindings.admitted_input import admit_x, admit_x_unrecorded, admitted_vector
from bindings.device_rows_door import door_is_device_rows
from bindings.classical_validate_unpack import classical_view, classical_check_weights_input
from core.estimators_entry_table import EstimatorsEntryKey, estimators_vendor_word
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QnFoldStore
from glm.qn_door_run import (
    QnDoorFitRequest, QnDoorFitResult, QnDoorPredictRequest,
    qn_check_lengths, qn_check_features, qn_check_coefficients, qn_change_tolerance,
)


from bindings.qn_boundary_unpack import (
    QnFitUnpacked,
)


def qn_fit_pack(model: PythonObject, var fit: QnFitUnpacked, result: QnDoorFitResult) raises -> PythonObject:
    var fields = py_fields(model)
    py_setitem(fields, "input_copied_", PythonObject(fit.copied))
    py_setitem(fields, "_w", fit.w)
    py_setitem(fields, "coef_", fit.coef)
    py_setitem(fields, "intercept_", PythonObject(result.intercept))
    py_setitem(fields, "n_iter_", fit.iterations)
    py_setitem(fields, "objective_", PythonObject(result.objective))
    py_setitem(fields, "retcode_", PythonObject(result.retcode))
    py_setitem(fields, "n_features_in_", fit.run.request.features)
    _ = fit^
    return model
