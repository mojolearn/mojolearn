# SPDX-License-Identifier: Apache-2.0
# Native pickle state restoration for the kernel SVM family. Incoming state
# remains owned by its caller; replaced values are admitted before publication.
# The PACK half of the former `providers/python/svc_public_state_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/svc_public_state_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_attributes import classical_connect_numpy
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.mode_metadata_native import _truth
from bindings.svc_kernel_native import _svm_validate


from providers.python.svc_public_state_pyapi_unpack import (
    _state_fitted,
    _state_keys,
)


def svc_public_state_operation(model: PythonObject, operation: String, args: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    if operation != "family_svc_public_svc_setstate" and operation != "family_svc_public_svr_setstate":
        raise Error("RuntimeError: unknown SVM state operation")
    classical_connect_numpy(context)
    var state = args[0]
    var keys = _state_keys(state)
    var fields = py_fields(model)
    var staged = py_call(native_loaded_module("builtins").dict, fields)
    for i in range(len(keys)):  # driver-loop: stage restoration without mutating or releasing model state
        py_setitem(staged, keys[i], state[keys[i]])
    if not py_contains(state, "_hp_C") and _state_fitted(staged):
        _svm_validate(model, staged, operation == "family_svc_public_svc_setstate", context)
    var publish = _state_keys(staged)
    for i in range(len(publish)):  # driver-loop: all old owners remain intact until full admission succeeds
        if py_contains(fields, publish[i]) and fields[publish[i]] is not staged[publish[i]]:
            native_release_admit(fields[publish[i]], context)
    for i in range(len(publish)):  # driver-loop: publish only admitted replacements
        py_setitem(fields, publish[i], staged[publish[i]])
    return PythonObject(None)
