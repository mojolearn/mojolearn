# SPDX-License-Identifier: Apache-2.0
# Public metric binding selection after the single native call. Import setup
# connects available modules; operation execution never invokes a Python loader.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from providers.python.estimator_binding_pyapi import estimator_binding_mode


def metric_binding_arguments(operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    # These two internal/public routes have no reserved binding argument.
    if operation == "family_mcls_log_loss" or operation == "family_mcls_accuracy_fraction":
        return args
    if len(args) == 0 or args[0] is not None:
        return args
    var mode = PythonObject(None)
    if (operation == "family_mcls_accuracy" or operation == "family_mcls_r2"
            or operation == "family_mcls_regression_error" or operation == "family_mcls_roc_auc"
            or operation == "family_mcls_pr_curve" or operation == "family_mcls_confusion"
            or operation == "family_mcls_prf"):
        mode = args[len(args) - 1]
    var binding = estimator_binding_mode(mode, "_mojolearn_metrics", context)
    var values = Python.list(binding)
    for i in range(1, len(args)):  # driver-loop: fixed public metric arguments
        _ = py_call(values.append, args[i])
    return py_call(py_type(Python.tuple()), values)

