# SPDX-License-Identifier: Apache-2.0
# Read a suspended caller without sys._getframe's Python audit callbacks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def native_caller_globals(depth: Int) raises -> PythonObject:
    # Public CPython APIs: depth zero is the current Python frame, matching
    # sys._getframe called from a native entry. Missing callers are reported
    # as None so each boundary can preserve its existing outside fallback.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        return PythonObject(None)
    var frame = PythonObject(from_borrowed=pointer)
    for _ in range(max(0,depth)):  # driver-loop: requested suspended caller depth
        pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(frame._obj_ptr)
        _ = frame  # keep the inspected frame alive through the C accessor
        if not pointer:
            if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
                raise cpy.unsafe_get_error()
            return PythonObject(None)
        frame = PythonObject(from_owned=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetGlobals")(frame._obj_ptr)
    _ = frame  # the globals accessor returns a new reference
    if not pointer:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=pointer)
