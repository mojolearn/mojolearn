# SPDX-License-Identifier: Apache-2.0
# `host_predict(path, X)` / `host_predict_proba(path, X)`: the saved model
# rebuilt by the native archive reader, then the compiled owner entry its
# own public method reaches, selected from fixed import-time recipes
# (python/mojolearn/_native_host_archive_predict.py). No public Python
# method, method wrapper, availability lambda or `_bind` runs here.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.family_common_unpack import family_repr
from bindings.family_common_pack import family_call
from bindings.host_archive_native_door import host_archive_build
from bindings.host_svc_native_door import host_svc_probability
from bindings.native_loaded_module_door import NativeModules
from bindings.native_class_admission_door import class_record, class_admit

def _class_name(record: PythonObject) raises -> String:
    return String(record[2])


def _unavailable(record: PythonObject, name: String) raises -> Error:
    # available_if's refusal text (python/mojolearn/_mode.py:available_if).
    return Error("AttributeError: This '" + _class_name(record) + "' has no attribute '" + name + "'")


def _check_module(module: PythonObject, context: PythonObject) raises:
    # A native entry pointer alone does not authorize a module __getattr__
    # hook or a replaced helper export: the owner looks its kernels up by
    # name in this module, so its retained exports must be unchanged.
    var exports = py_get(context["host_archive_predict_modules"], module)
    if exports is None:
        raise Error("TypeError: saved host prediction binding was not connected at import")
    var namespace = _loaded_fields(module)
    for k in range(len(exports)):  # driver-loop: over the retained native export identities of one binding
        if py_get(namespace, exports[k][0]) is not exports[k][1]:
            raise Error("TypeError: saved host prediction binding export changed")
