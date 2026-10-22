# SPDX-License-Identifier: Apache-2.0
# Preserve the existing wrapper's feature-name warning after native buffer
# preparation. Its _array_like materializes pandas before checking names;
# supported native buffers therefore have no incoming name sequence.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_warnings import classical_warning


def host_inference_feature_names(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    if py_get(fields, "feature_names_in_") is None:
        return
    var b = Python.import_module("builtins")
    var name = String(py_call(b.type.__getattribute__, py_type(model), "__name__"))
    var message = "X does not have valid feature names, but " + name + " was fitted with feature names"
    var frame = py_call(context["classical_getframe"], 1)
    var module = py_get(frame.f_globals, "__name__", "")
    if py_type(module) is not b.str:
        module = PythonObject("")
    classical_warning(message, b.UserWarning, context, Python.tuple(frame.f_code.co_filename, frame.f_lineno, module), "HostInferenceWarning")
