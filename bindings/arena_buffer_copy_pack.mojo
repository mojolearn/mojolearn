# SPDX-License-Identifier: Apache-2.0
# One native copy of a buffer's logical C-order bytes into a fresh array.array
# store. Replaces the `array.frombytes(view.tobytes("C"))` pair (two copies:
# an intermediate bytes object, then the store) used by row materialization
# and sequence admission. PyBuffer_ToContiguous packs signed/zero/indirect
# strides in logical C order exactly as memoryview.tobytes("C") does.
# The PACK half of the former `bindings/arena_buffer_copy.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/arena_buffer_copy_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_array_storage_door import native_array_zeros
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.cpython_table import cpython_table


from bindings.arena_buffer_copy_unpack import (
    _typecode_width,
)


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
