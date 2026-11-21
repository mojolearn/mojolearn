# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a neural constructor's `device` value into its device code.

The rule and the words are the Ridge pilot's (core/linear_route.mojo); this
is `linear_device_unpack` (providers/python/linear_route_boundary_door.mojo)
without that door's import closure, for the neural model classes. Called at
construction, `set_params` and an owner record's miss; never per call.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type
from core.linear_route import LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal


def neural_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in words
    before any state changes. `b` is the builtins module the calling door
    already admitted."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL
