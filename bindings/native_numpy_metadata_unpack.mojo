# SPDX-License-Identifier: Apache-2.0
# Admit NumPy's native dtype constructor before any live-module field call.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.cpython_table import cpython_table


from bindings.native_numpy_metadata_pack import (
    native_numpy_type,
)


def native_numpy_dtype(context: PythonObject) raises -> PythonObject:
    return native_numpy_type("dtype", context)
