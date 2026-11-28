# SPDX-License-Identifier: Apache-2.0
# Sequential weight admission and transfer use the existing import ABI.
# Ordinary CPU torch tensors use their native dtype/device/shape metadata and
# C data_ptr/stride entries, without __array__ or a conversion helper. Custom
# mapping/tensor subclasses can define caller protocols; those are not a
# qualification of arbitrary Python callbacks as native execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type


def _exact_mapping(value: PythonObject, mapping_types: PythonObject) raises -> Bool:
    var kind = py_type(value)
    return kind is mapping_types[0] or kind is mapping_types[1]
