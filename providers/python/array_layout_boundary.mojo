# SPDX-License-Identifier: Apache-2.0
"""External shape UNPACK/PACK; native layout code owns size and strides."""
from std.python import Python, PythonObject


def native_shape_unpack(shape: PythonObject) raises -> List[Int]:
    var result = List[Int](capacity=len(shape))
    for k in range(len(shape)):  # driver-loop: unpack admitted rank metadata
        result.append(Int(py=shape[k]))
    return result^


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
