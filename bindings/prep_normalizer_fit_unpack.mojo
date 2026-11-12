# SPDX-License-Identifier: Apache-2.0
# Native Normalizer fit/fit_transform envelope, including the former _mode
# validation and fitting-depth wrapper. The optional target is validated but
# never used in normalization, and fit_params remain intentionally ignored.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.mode_metadata_native_door import _fields, _truth
from bindings.prep_normalizer_lifecycle_unpack import _normalizer_norm
from bindings.prep_normalizer_lifecycle_pack import _setting, _class_setting, _tags, _native_entries, _normalizer_apply
from bindings.prep_fit_error_pack import prep_fit_error
from bindings.parallel_public_result_door import parallel_public_result
from providers.python.prep_transform_pyapi_unpack import prep_transform_snapshot
from providers.python.prep_transform_pyapi_pack import prep_transform_values
from providers.python.array_metadata_pyapi_door import ArrayPin
from providers.python.prep_target_pyapi_pack import prep_ignored_target


def _process_mode(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = context["backend_fields"]
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native Normalizer mode settings require an ordinary dictionary")
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: native namespace keys before default-mode lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native Normalizer mode keys require exact strings")
    return fields["_DEFAULT_MODE"]


def _publication(state: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, state)
    for i in range(len(keys)):  # driver-loop: refuse concurrent destination-key hooks before publication
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native Normalizer state keys must remain exact strings")


def _finish_depth(state: PythonObject, remaining: PythonObject) raises:
    # remaining comes from exact native bool/int/float arithmetic before execution.
    # Reuse the snapshot, not a mutable model setting reread after a kernel.
    _publication(state)
    if Bool(py=remaining > PythonObject(0)):
        py_setitem(state, "_mojolearn_fitting", remaining)
    else:
        _ = py_call(state.pop, "_mojolearn_fitting", PythonObject(None))
