# SPDX-License-Identifier: Apache-2.0
# Native pickle state restoration for the kernel SVM family. Incoming state
# remains owned by its caller; replaced values are admitted before publication.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_attributes import classical_connect_numpy
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.mode_metadata_native import _truth
from bindings.svc_kernel_native import _svm_validate


def _state_keys(state: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(state) is not b.dict:
        raise Error("TypeError: SVM state requires an exact dictionary")
    var keys = py_call(b.list, state)
    for i in range(len(keys)):  # driver-loop: keys are admitted before dictionary lookup or update
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: SVM state requires exact string keys")
    return keys


def _state_fitted(fields: PythonObject) raises -> Bool:
    if _truth(py_get(fields, "_mojolearn_fitted")) or _truth(py_get(fields, "_mojolearn_fitting")):
        return True
    var b = native_loaded_module("builtins")
    var names = py_get(fields, "_mojolearn_params_reset")
    if names is not None:
        if py_type(names) is not b.tuple:
            raise Error("TypeError: SVM reset metadata requires an exact tuple")
    else:
        names = py_call(b.list, fields)
    for i in range(len(names)):  # driver-loop: same learned-field rule as native mode metadata
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: SVM reset metadata requires exact string names")
        var name = String(names[i])
        if name.endswith("_") and not name.startswith("_") and py_get(fields, names[i]) is not None:
            return True
    return False


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
