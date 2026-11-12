# SPDX-License-Identifier: Apache-2.0
# Scalar ndarray parameter errors must not enter NumPy's Python arrayprint.
# The PACK half of the former `bindings/host_iforest_scalar_repr.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/host_iforest_scalar_repr_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.math import isfinite, isnan
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_numpy_metadata_unpack import native_numpy_dtype


def _if_repr_options(context: PythonObject) raises -> PythonObject:
    var modules = context["modules"]
    var module = py_get(modules, "numpy._core.arrayprint")
    if module is None:
        module = py_get(modules, "numpy.core.arrayprint")
    if py_type(module) is not context["host_iforest_module_type"]:
        raise Error("TypeError: IsolationForest parameter text requires native NumPy print metadata")
    var fields = py_call(context["host_iforest_module_dict"], module)
    var options = py_get(fields, "_format_options")
    if options is None:
        var variable = py_get(fields, "format_options")
        if py_type(variable) is not context["host_iforest_context_type"]:
            raise Error("TypeError: IsolationForest parameter text requires a native print ContextVar")
        options = py_call(context["host_iforest_context_get"], variable)
    if py_type(options) is not py_type(Python.dict()):
        raise Error("TypeError: IsolationForest parameter text requires native print option data")
    var keys = py_call(py_type(Python.list()), options)
    for i in range(len(keys)):  # driver-loop: validate print metadata keys before dictionary lookups
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest print option names must be native strings")
    var integers = Python.tuple("legacy", "linewidth", "threshold")
    for i in range(len(integers)):  # driver-loop: exact integer print options
        if py_type(options[integers[i]]) is not py_type(PythonObject(0)):
            raise Error("TypeError: IsolationForest print options require native integers")
    var strings = Python.tuple("sign", "floatmode", "nanstr", "infstr")
    for i in range(len(strings)):  # driver-loop: exact text print options
        if py_type(options[strings[i]]) is not py_type(PythonObject("")):
            raise Error("TypeError: IsolationForest print options require native strings")
    if py_type(options["suppress"]) is not py_type(PythonObject(True)) or (options["precision"] is not None and py_type(options["precision"]) is not py_type(PythonObject(0))):
        raise Error("TypeError: IsolationForest print precision requires native scalar metadata")
    if py_get(options, "override_repr") is not None:
        raise Error("TypeError: IsolationForest parameter errors cannot invoke Python print overrides; use native print options")
    return options
