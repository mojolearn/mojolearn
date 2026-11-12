# SPDX-License-Identifier: Apache-2.0
# Fixed native CV wire records. No Python class/function, pickle reducer or
# import name from the request is executed. Import-time maps select all owners.
from bindings.native_array_storage_door import native_array_empty
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.native_buffer_owner_pack import native_storage_allocate, native_buffer_view
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_numpy_metadata_pack import native_numpy_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.msel_estimator_native import msel_native_clone, msel_estimator_plan, msel_clone_parameter
from bindings.msel_scorer_native import msel_scorer_fields, msel_metric_plan
from bindings.family_common import family_call
from core.msel_parallel_wire_run import _depth

def _record_size(record: PythonObject, size: Int) raises:
    if len(record) != size:
        raise Error("ValueError: malformed native CV value record length")


def _sequence(record: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.list and py_type(record) is not b.tuple:
        raise Error("TypeError: native CV wire sequences require exact list/tuple metadata")


def msel_wire_needs_numpy(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.tuple or cls is b.list:
        if len(value) and py_type(value[0]) is b.str:
            if String(value[0]) == "numpy_scalar" or String(value[0]) == "ndarray":
                return True
        for i in range(len(value)):  # driver-loop: generated wire metadata, never application iterators
            if msel_wire_needs_numpy(value[i]):
                return True
    elif cls is b.dict:
        var values = py_call(b.list, value.values())
        for i in range(len(values)):  # driver-loop: generated configuration record fields
            if msel_wire_needs_numpy(values[i]):
                return True
    return False
