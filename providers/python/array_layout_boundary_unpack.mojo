# SPDX-License-Identifier: Apache-2.0
"""External shape UNPACK/PACK; native layout code owns size and strides."""
from std.python import Python, PythonObject


def native_shape_unpack(shape: PythonObject) raises -> List[Int]:
    var result = List[Int](capacity=len(shape))
    for k in range(len(shape)):  # driver-loop: unpack admitted rank metadata
        result.append(Int(py=shape[k]))
    return result^
