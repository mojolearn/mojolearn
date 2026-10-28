# SPDX-License-Identifier: Apache-2.0
# Import-connected resampling binding selection inside the public operation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from providers.python.estimator_binding_pyapi import estimator_binding_mode


def resample_arguments(operation: String, args: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    if len(args) == 0 or args[0] is not None:
        return args
    var index = len(args) - 1
    if operation == "family_rs_bootstrap" or operation == "family_rs_permutation_test":
        index -= 1
    var binding = estimator_binding_mode(args[index], "_mojolearn_resample", context, False)
    var values = Python.list(binding)
    for i in range(1, len(args)):  # driver-loop: fixed public resampling arguments
        _ = py_call(values.append, args[i])
    return py_call(py_type(Python.tuple()), values)
