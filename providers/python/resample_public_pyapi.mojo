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


def resample_result_layout(index: Int, context: PythonObject) raises -> PythonObject:
    from bindings.mode_metadata_native import _class_fields
    from bindings.py_balanced import py_get
    from bindings.native_loaded_module import native_loaded_module
    var b = native_loaded_module("builtins")
    var cls = context["family_rs_result_classes"][index]
    if py_type(cls) is not b.type:
        raise Error("TypeError: resampling result metadata requires the connected ordinary type")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not (b.tuple if index == 3 else b.object):
        raise Error("TypeError: resampling result class ancestry changed")
    var fields = _class_fields(cls)
    if py_get(fields, "__del__") is not None:
        raise Error("TypeError: resampling result storage cannot invoke Python finalizers")
    return cls


def resample_result_initialize(args: PythonObject, context: PythonObject) raises -> PythonObject:
    from bindings.mode_metadata_native import _fields, _class_fields
    from bindings.py_balanced import py_contains, py_setitem, py_get
    from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
    from bindings.native_instance_item import native_instance_item
    from bindings.classical_attributes import classical_connect_numpy
    from bindings.py_attr import py_attr
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
    _ = resample_result_layout(index, context)
    var namespace = _class_fields(classes[index])
    for i in range(len(names)):  # driver-loop: fixed result fields cannot invoke replacement descriptors
        if py_contains(namespace, names[i]):
            raise Error("TypeError: resampling result fields require ordinary instance storage")
    var fields = _fields(model)
    var replacements = Python.list()
    # Prepare every incoming owner, including the interval, before examining
    # releases. A late refusal cannot leave a partially initialized result.
    if index == 0:
        for i in range(4):  # driver-loop: fixed BootstrapResult fields
            py_call(replacements.append, args[i + 2])
        var interval = py_call(b.tuple.__new__, resample_result_layout(3, context), Python.tuple(args[6], args[7]))
        py_call(replacements.append, interval)
        py_call(replacements.append, args[8])
        py_call(replacements.append, args[9])
    else:
        for i in range(len(names)):  # driver-loop: fixed PermutationTestResult/MonteCarloResult fields
            py_call(replacements.append, args[i + 2])
    classical_connect_numpy(context)
    for i in range(len(names)):  # driver-loop: admit every replaced owner before the first field mutation
        if not py_contains(fields, names[i]):
            continue
        var previous = native_instance_item(fields, names[i])
        if previous is replacements[i]:
            continue
        if py_type(previous) is classes[3]:
            # ConfidenceInterval is the package's tuple subclass. Inspect its
            # native tuple storage and class finalizer rather than invoking
            # its Python namedtuple accessors or broadening owner admission.
            var interval_fields = _class_fields(classes[3])
            if py_get(interval_fields, "__del__") is not None:
                raise Error("TypeError: resampling result replacement cannot release interval finalizers")
            native_release_weakrefs(previous, context)
            var count = Int(py=py_call(py_attr(b.tuple, "__len__"), previous))
            for j in range(count):  # driver-loop: native interval elements can own callback-bearing values
                native_release_admit(py_call(py_attr(b.tuple, "__getitem__"), previous, j), context)
        else:
            native_release_admit(previous, context)
    for i in range(len(names)):  # driver-loop: commit only after all old owners passed release admission
        py_setitem(fields, names[i], replacements[i])
    return PythonObject(None)
