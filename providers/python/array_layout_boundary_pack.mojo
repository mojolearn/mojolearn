# SPDX-License-Identifier: Apache-2.0
"""External shape UNPACK/PACK; native layout code owns size and strides."""
# The PACK half of the former `providers/python/array_layout_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/array_layout_boundary_unpack.mojo`.
from std.python import Python, PythonObject


def native_shape_pack(shape: List[Int]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var raw = cpy.PyTuple_New(len(shape))
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    for k in range(len(shape)):  # driver-loop: pack native public rank metadata
        var dimension = PythonObject(shape[k])
        _ = cpy.PyTuple_SetItem(raw, k, cpy.Py_NewRef(dimension._obj_ptr))
    return result^
