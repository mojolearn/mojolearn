# SPDX-License-Identifier: Apache-2.0
# One native copy of a buffer's logical C-order bytes into a fresh array.array
# store. Replaces the `array.frombytes(view.tobytes("C"))` pair (two copies:
# an intermediate bytes object, then the store) used by row materialization
# and sequence admission. PyBuffer_ToContiguous packs signed/zero/indirect
# strides in logical C order exactly as memoryview.tobytes("C") does.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_array_storage import native_array_zeros
from bindings.native_buffer_address import native_buffer_address
from bindings.optimizer_public_words import stash_python_error
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


def buffer_copy_to_array(code: String, source: PythonObject) raises -> PythonObject:
    """A new array.array of typecode `code` holding the C-order bytes of
    `source` (any buffer exporter, contiguous or strided). The byte count must
    be a multiple of the store's item size, as array.frombytes requires."""
    ref cpy = Python().cpython()
    var width = _typecode_width(code)
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    # PyBUF_INDIRECT without PyBUF_FORMAT: shape, strides and suboffsets, so
    # any strided exporter is accepted; the raw bytes are copied unchanged.
    if cpython_table()[].PyObject_GetBuffer(source._obj_ptr, pointer, Int32(0x118)) != 0:
        var failure = stash_python_error()
        raise failure^
    var result = PythonObject(None)
    try:
        var size = view[2]
        if size % width != 0:
            raise Error("ValueError: bytes length not a multiple of item size")
        result = native_array_zeros(code, size // width)
        if size > 0:
            var address = native_buffer_address(result)
            if cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(address, pointer, size, Int8(67)) != 0:
                var failure = stash_python_error()
                raise failure^
    finally:
        cpython_table()[].PyBuffer_Release(pointer)
        # Keep the Py_buffer storage alive past the release on every path
        # (ASAP destruction could free it before PyBuffer_Release reads it).
        _ = len(view)
    _ = view^
    return result
