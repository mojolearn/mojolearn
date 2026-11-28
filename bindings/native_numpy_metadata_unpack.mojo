# SPDX-License-Identifier: Apache-2.0
# Admit NumPy's native dtype constructor before any live-module field call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


from bindings.native_numpy_metadata_pack import (
    native_numpy_type,
)


def native_numpy_dtype(context: PythonObject) raises -> PythonObject:
    return native_numpy_type("dtype", context)
