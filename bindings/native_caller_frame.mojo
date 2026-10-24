# SPDX-License-Identifier: Apache-2.0
# Read a suspended caller without sys._getframe's Python audit callbacks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def native_frame_globals(frame: PythonObject) raises -> PythonObject:
    # Callers supply only frames obtained from PyEval_GetFrame/PyFrame_GetBack,
    # never user objects. The exact CPython frame's f_globals is a native,
    # read-only member/getter without an audit event (unlike f_code). Reading
    # it via the stable object API also supports our Python 3.10 floor;
    # PyFrame_GetGlobals itself was added only in Python 3.11.
    ref cpy = Python().cpython()
    var name = PythonObject("f_globals")
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(frame._obj_ptr,name._obj_ptr)
    _ = frame
    _ = name
    if not pointer:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=pointer)


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
    return native_frame_globals(frame)
