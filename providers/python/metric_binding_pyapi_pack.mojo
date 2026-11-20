# SPDX-License-Identifier: Apache-2.0
# Public metric binding selection after the single native call. Import setup
# connects available modules; operation execution never invokes a Python loader.
from bindings.operation_names_mcls import (
    MCLS_OPN_FAMILY_MCLS_ACCURACY,
    MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION,
    MCLS_OPN_FAMILY_MCLS_CONFUSION,
    MCLS_OPN_FAMILY_MCLS_LOG_LOSS,
    MCLS_OPN_FAMILY_MCLS_PRF,
    MCLS_OPN_FAMILY_MCLS_PR_CURVE,
    MCLS_OPN_FAMILY_MCLS_R2,
    MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR,
    MCLS_OPN_FAMILY_MCLS_ROC_AUC,
)
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from providers.python.estimator_binding_pyapi_door import estimator_binding_mode


def metric_binding_arguments(operation: Int, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    # These two internal/public routes have no reserved binding argument.
    # `operation` is the leaf code of bindings/operation_names_mcls.mojo.
    if operation == MCLS_OPN_FAMILY_MCLS_LOG_LOSS or operation == MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION:
        return args
    if len(args) == 0 or args[0] is not None:
        return args
    var mode = PythonObject(None)
    if (operation == MCLS_OPN_FAMILY_MCLS_ACCURACY or operation == MCLS_OPN_FAMILY_MCLS_R2
            or operation == MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR or operation == MCLS_OPN_FAMILY_MCLS_ROC_AUC
            or operation == MCLS_OPN_FAMILY_MCLS_PR_CURVE or operation == MCLS_OPN_FAMILY_MCLS_CONFUSION
            or operation == MCLS_OPN_FAMILY_MCLS_PRF):
        mode = args[len(args) - 1]
    var binding = estimator_binding_mode(mode, "_mojolearn_metrics", context)
    var values = Python.list(binding)
    for i in range(1, len(args)):  # driver-loop: fixed public metric arguments
        _ = py_call(values.append, args[i])
    return py_call(py_type(Python.tuple()), values)

