# SPDX-License-Identifier: Apache-2.0
# One native copy of a buffer's logical C-order bytes into a fresh array.array
# store. Replaces the `array.frombytes(view.tobytes("C"))` pair (two copies:
# an intermediate bytes object, then the store) used by row materialization
# and sequence admission. PyBuffer_ToContiguous packs signed/zero/indirect
# strides in logical C order exactly as memoryview.tobytes("C") does.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_array_storage_door import native_array_zeros
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.cpython_table import cpython_table


def _typecode_width(code: String) raises -> Int:
    # p3-array-arena: the item width of the fixed-width array.array typecodes
    # without allocating a probe store; any other code keeps the probe.
    if code == "b" or code == "B":
        return 1
    if code == "h" or code == "H":
        return 2
    if code == "i" or code == "I" or code == "f":
        return 4
    if code == "d" or code == "q" or code == "Q":
        return 8
    return Int(py=native_array_zeros(code, 0).itemsize)
