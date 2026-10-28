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


def resample_result_initialize(args: PythonObject, context: PythonObject) raises -> PythonObject:
    from bindings.mode_metadata_native import _fields, _class_fields
    from bindings.py_balanced import py_contains, py_setitem
    from bindings.native_loaded_module import native_loaded_module
    var b = native_loaded_module("builtins")
    var model = args[0]
    if py_type(args[1]) is not b.str:
        raise Error("TypeError: resampling result kind must be a native string")
    var kind = String(args[1])
    var classes = context["family_rs_result_classes"]
    var index = 0 if kind == "bootstrap" else 1 if kind == "permutation" else 2
    if kind != "bootstrap" and kind != "permutation" and kind != "monte_carlo":
        raise Error("ValueError: unknown resampling result kind")
    if py_type(model) is not classes[index]:
        raise Error("TypeError: resampling result initialization requires the connected result class")
    var names = (Python.tuple("point_estimate", "distribution", "sorted_distribution", "standard_error", "confidence_interval", "order_low", "order_high")
                 if index == 0 else (Python.tuple("statistic", "null_distribution", "pvalue", "count_less", "count_greater")
                 if index == 1 else Python.tuple("integral", "mean", "volume", "closed_form")))
    var namespace = _class_fields(classes[index])
    for i in range(len(names)):  # driver-loop: fixed result fields cannot invoke replacement descriptors
        if py_contains(namespace, names[i]):
            raise Error("TypeError: resampling result fields require ordinary instance storage")
    var fields = _fields(model)
    if index == 0:
        for i in range(4):  # driver-loop: fixed BootstrapResult fields
            py_setitem(fields, names[i], args[i + 2])
        var interval = py_call(b.tuple.__new__, classes[3], Python.tuple(args[6], args[7]))
        py_setitem(fields, names[4], interval)
        py_setitem(fields, names[5], args[8])
        py_setitem(fields, names[6], args[9])
    else:
        for i in range(len(names)):  # driver-loop: fixed PermutationTestResult/MonteCarloResult fields
            py_setitem(fields, names[i], args[i + 2])
    return PythonObject(None)
