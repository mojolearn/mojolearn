# SPDX-License-Identifier: Apache-2.0
# Completed selector metadata at the preprocessing boundary. This operation
# has no numerical provider call: it reads the already fitted support once
# and builds the final independent public list in native code.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.mode_metadata_native_door import _fields, _truth
from bindings.params_metadata_native_door import _resolved, _value
from bindings.sequential_admission_door import _one_of
from bindings.py_cpython_table import cpython_table
from bindings.prep_keys_unpack import prep_keys


def _support_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = prep_keys()
    # Exact connected NumPy scalar types have native truth implementations.
    # Array/object truth and Python subclasses are not scalar flag protocols.
    var scalars = py_get(context, nk[].k_scalar_types)
    if scalars is not None and _one_of(py_type(value), scalars):
        return Bool(py=value)
    return _truth(value)
