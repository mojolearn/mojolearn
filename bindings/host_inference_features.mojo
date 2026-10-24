# SPDX-License-Identifier: Apache-2.0
# Preserve the existing wrapper's feature-name warning after native buffer
# preparation. Its _array_like materializes pandas before checking names;
# supported native buffers therefore have no incoming name sequence.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_warnings import classical_warning
from bindings.native_caller_frame import native_frame_globals


def host_inference_feature_names(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    if py_get(fields, "feature_names_in_") is None:
        return
    var b = Python.import_module("builtins")
    var name = String(py_call(b.type.__getattribute__, py_type(model), "__name__"))
    var message = "X does not have valid feature names, but " + name + " was fitted with feature names"
    # Match the old _getframe(1) attribution using public native CPython
    # accessors. Neither sys._getframe nor frame.f_code may run here: both
    # emit audit events which invoke application Python hooks.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        raise Error("ValueError: call stack is not deep enough")
    var current = PythonObject(from_borrowed=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(current._obj_ptr)
    _ = current^
    if not pointer:
        if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
            raise cpy.unsafe_get_error()
        raise Error("ValueError: call stack is not deep enough")
    var frame = PythonObject(from_owned=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var code = PythonObject(from_owned=pointer)
    var line = cpy.lib.get_function[Int32]("PyFrame_GetLineNumber")(frame._obj_ptr)
    var module = py_get(native_frame_globals(frame), "__name__", "")
    if py_type(module) is not b.str:
        module = PythonObject("")
    classical_warning(message, b.UserWarning, context, Python.tuple(code.co_filename, Int(line), module), "HostInferenceWarning")
