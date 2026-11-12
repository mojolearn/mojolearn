# SPDX-License-Identifier: Apache-2.0
# Shared exact label scalar and connected ClassLabels admission. No numbers
# ABC or Python subtype/iteration/conversion hook is part of this boundary.
# The PACK half of the former `bindings/label_admission_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/label_admission_native_unpack.mojo`.
from std.collections import Dict
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_numpy_metadata_pack import native_numpy_module_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_attr import py_attr
from std.python._cpython import PyObjectPtr
from bindings.p3_labels_metrics_native_door import p3_type_word

from bindings.label_admission_native_unpack import (
    _labels_class,
)


def classifier_label_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_get(context, "classifier_labels_class")
    if cls is None or py_type(value) is not cls:
        return value
    _ = _labels_class(context)
    # list.copy is the native list implementation, not an iterable protocol.
    return py_call(py_attr(py_type(Python.list()), "copy"), value)
