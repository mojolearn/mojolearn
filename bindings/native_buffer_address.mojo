# SPDX-License-Identifier: Apache-2.0
# Read the address of an already-admitted, owned native contiguous buffer.
from std.python import Python, PythonObject


def native_buffer_address(value: PythonObject) raises -> Int:
    # The caller admits a native contiguous buffer and keeps its owner alive
    # through pointer consumption. This temporary export owns a reference only
    # until the matching release, as in sequence_array_address.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0))
    if status != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return address


