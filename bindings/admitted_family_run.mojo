# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/admitted_family.mojo: typed Mojo only (no Python imports).


def admitted_reverse_word(src_address: Int, dst_address: Int, itemsize: Int):
    """Copy one scalar word of `itemsize` bytes with its byte order reversed."""
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src_address)
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_address)
    for byte in range(itemsize):  # small-loop(one admitted scalar word)
        dst[byte] = src[itemsize - 1 - byte]
