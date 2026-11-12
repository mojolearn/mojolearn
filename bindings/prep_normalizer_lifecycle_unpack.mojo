# SPDX-License-Identifier: Apache-2.0
# Normalizer operations own their complete previously decorated public calls.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, native_namespace_string_keys
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native_door import _fields, _truth
from bindings.params_metadata_native_door import _resolved, _value
from bindings.native_loaded_module_door import NativeModules
from bindings.native_class_admission_door import class_record, class_admit, class_watch_ready
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.metrics_native_support_door import metric_warn
from core.preprocessing_program import prep_program_stage as _stage
from bindings.x_prep_native_owner_door import xprep_storage, xprep_take, xprep_run
from bindings.parallel_public_result_door import parallel_public_result
from bindings.prep_entry_cache_unpack import prep_mode_checked
from providers.python.prep_transform_pyapi_unpack import prep_transform_snapshot
from providers.python.prep_transform_pyapi_pack import prep_transform_values
from providers.python.array_metadata_pyapi_door import ArrayPin
from core.shared_slot import SharedSlot


def _fitted(fields: PythonObject) raises -> Bool:
    if _truth(py_get(fields, "_mojolearn_fitted")) or _truth(py_get(fields, "_mojolearn_fitting")):
        return True
    var b = native_loaded_module("builtins")
    var names = py_get(fields, "_mojolearn_params_reset")
    if names is None:
        names = py_call(b.list, fields)
    elif py_type(names) is not b.tuple:
        raise Error("TypeError: native parameter reset metadata requires an ordinary tuple")
    for i in range(len(names)):  # driver-loop: native fitted field names
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native parameter reset metadata requires exact string names")
        var key = String(names[i])
        if key.endswith("_") and not key.startswith("_") and py_contains(fields, names[i]) and fields[names[i]] is not None:
            return True
    return False


def _repr_admit(value: PythonObject, depth: Int = 0) raises:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex:
        return
    if depth < 64 and (cls is b.list or cls is b.tuple):
        for i in range(len(value)):  # driver-loop: admit native invalid-setting repr before formatting
            _repr_admit(value[i], depth + 1)
        return
    if depth < 64 and cls is b.dict:
        var pairs = py_call(b.list, value.items())
        for i in range(len(pairs)):  # driver-loop: native invalid-setting keys/values before repr
            _repr_admit(pairs[i][0], depth + 1)
            _repr_admit(pairs[i][1], depth + 1)
        return
    raise Error("TypeError: native Normalizer settings cannot invoke Python representation or conversion hooks")


def _safe_repr(value: PythonObject) raises -> String:
    _repr_admit(value)
    return String(py_call(native_loaded_module("builtins").repr, value))


comptime _CLASS_SETTING_NAMES = 7


def _plain_setting(value: PythonObject, b: PythonObject) raises -> Bool:
    # A resolved class value a record may hold: plain values only, so the
    # record never retains a function (its globals reach the class) or a
    # descriptor.
    var cls = py_type(value)
    return value is None or cls is b.bool or cls is b.int or cls is b.str or cls is b.dict


struct _PinnedEntries(Defaultable, Movable):
    # The import-time entry identities of one numeric mode as a name -> entry
    # dict, built once from the immutable pins tuple it came from.
    var pins: PythonObject
    var mode: String
    var expected: PythonObject

    def __init__(out self):
        self.pins = PythonObject(None)
        self.mode = String("")
        self.expected = PythonObject(None)


def _normalizer_norm(norm: PythonObject) raises -> Int:
    var b = native_loaded_module("builtins")
    var code = -1
    if py_type(norm) is b.str:
        var word = String(norm)
        code = 0 if word == "l1" else (1 if word == "l2" else (2 if word == "max" else -1))
    if code < 0:
        raise Error("mojolearn: invalid norm " + _safe_repr(norm))
    return code
