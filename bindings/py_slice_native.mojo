# SPDX-License-Identifier: Apache-2.0
# Supported CPython slices with owned slice/result references. The caller must
# admit the receiver and bounds before entering a product operation; this is
# ownership machinery, not permission to invoke arbitrary __getitem__/__index__.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def py_slice(value: PythonObject, start: PythonObject, stop: PythonObject,
             step: PythonObject) raises -> PythonObject:
    # PySlice_New borrows its arguments and returns a new reference. Explicit
    # owned PythonObject(None) bounds are balanced; no reference is adjusted
    # to compensate for the stdlib Slice conversion's borrowed-None defect.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PySlice_New")(
        start._obj_ptr, stop._obj_ptr, step._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var bounds = PythonObject(from_owned=pointer)
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(
        value._obj_ptr, bounds._obj_ptr)
    _ = bounds^
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def py_slice(value: PythonObject, start: Int, stop: Int) raises -> PythonObject:
    return py_slice(value, PythonObject(start), PythonObject(stop), PythonObject(None))


def py_slice_from(value: PythonObject, start: Int) raises -> PythonObject:
    return py_slice(value, PythonObject(start), PythonObject(None), PythonObject(None))


def py_slice_to(value: PythonObject, stop: Int) raises -> PythonObject:
    return py_slice(value, PythonObject(None), PythonObject(stop), PythonObject(None))
