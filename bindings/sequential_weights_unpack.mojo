# SPDX-License-Identifier: Apache-2.0
# Sequential weight admission and transfer use the existing import ABI.
# Ordinary CPU torch tensors use their native dtype/device/shape metadata and
# C data_ptr/stride entries, without __array__ or a conversion helper. Custom
# mapping/tensor subclasses can define caller protocols; those are not a
# qualification of arbitrary Python callbacks as native execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.py_typed_call import py_call_typed
from providers.python.autograd_weight_frontend_pyapi_pack import _checkpoint, _record
from autograd.program_io import ProgramInput
from bindings.sequential_description_unpack import sequential_fields
from bindings.sequential_description_pack import sequential_program, sequential_table
from bindings.sequential_execution_pack import sequential_entries
from bindings.sequential_arrays_door import sequential_array, sequential_address, sequential_input
from bindings.native_model_records import bump_model_generation


def _exact_mapping(value: PythonObject, mapping_types: PythonObject) raises -> Bool:
    var kind = py_type(value)
    return kind is mapping_types[0] or kind is mapping_types[1]
