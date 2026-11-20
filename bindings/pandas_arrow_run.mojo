# SPDX-License-Identifier: Apache-2.0
"""RUN of the pandas Arrow reader (bindings/pandas_arrow_native.mojo): the
typestr item sizes and the typed copy of one Arrow chunk's values and missing
slots into the output column. No Python import (tools/python_free_dirs.sh).
Lane rs-shared-infra-b (2026-11-12): moved word for word; the per-chunk copies
are the existing storage unpack, unchanged."""
from std.memory import unsafe_memcpy


def _itemsize(typestr: String) -> Int:
    if typestr == "<f8" or typestr == "<i8" or typestr == "<u8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<i2" or typestr == "<u2":
        return 2
    return 1


def _arrow_copy_values(data: Int, dst: Int, offset: Int, length: Int, item: Int, boolean: Bool):
    """`pandas_arrow_ndarray`'s copy of one chunk's `length` values, starting
    at element `offset` of the Arrow data buffer at `data`, to `dst`: one byte
    per bit for a boolean column, else the raw `item`-byte words."""
    if boolean:
        var bits = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=data)
        var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst)
        for r in range(length):  # driver-loop: storage unpack only, one Arrow bit per output byte
            var bit = offset + r
            target[r] = (bits[bit >> 3] >> UInt8(bit & 7)) & UInt8(1)
    else:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=data + offset * item),
               count=length * item)


def _arrow_fill_missing(valid_address: Int, dst: Int, offset: Int, length: Int, item: Int):
    """`pandas_arrow_ndarray`'s missing float slots: every row of the chunk
    whose validity bit (at `valid_address`, from element `offset`) is clear
    becomes NumPy's canonical quiet NaN in `dst`."""
    var valid = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid_address)
    for r in range(length):  # driver-loop: storage only, missing float slots become NaN
        var bit = offset + r
        if ((valid[bit >> 3] >> UInt8(bit & 7)) & UInt8(1)) == UInt8(0):
            # NumPy's canonical quiet NaN, the value pandas exports.
            if item == 8:
                MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=dst)[r] = UInt64(0x7FF8000000000000)
            else:
                MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)[r] = UInt32(0x7FC00000)
