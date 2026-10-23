# SPDX-License-Identifier: Apache-2.0
"""Typed output marshaling for the native decomposition owners
(bindings/mcd_native.mojo): the exact dtype casts `Array.astype` made of a
fitted 0/1 mask or a +-1 label column, and the count of a 0/1 mask."""


def decomp_f32_to_u8(src: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[UInt8, MutUntrackedOrigin], n: Int):
    # native output marshaling: one exact cast per mask entry (0.0 / 1.0)
    for i in range(n):
        dst[i] = src[i].cast[DType.uint8]()


def decomp_i32_to_u8(src: MutPointer[Int32, MutUntrackedOrigin], dst: MutPointer[UInt8, MutUntrackedOrigin], n: Int):
    # native output marshaling: one exact cast per mask entry (0 / 1)
    for i in range(n):
        dst[i] = src[i].cast[DType.uint8]()


def decomp_u8_to_f32(src: MutPointer[UInt8, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin], n: Int):
    # native input marshaling: a 0 / 1 mask as exact float32 values
    for i in range(n):
        dst[i] = src[i].cast[DType.float32]()


def decomp_f32_to_i32(src: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Int32, MutUntrackedOrigin], n: Int):
    # native output marshaling: one exact cast per label (+1.0 / -1.0)
    for i in range(n):
        dst[i] = src[i].cast[DType.int32]()


def decomp_count_u8(src: MutPointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    # the sum of a 0 / 1 mask (`Array.sum` of the support)
    var total = 0
    for i in range(n):
        total += Int(src[i])
    return total


def decomp_piv_valid(src: MutPointer[Int32, MutUntrackedOrigin], n: Int) -> Bool:
    # native input validation: one row index in [0, n) per pivot
    for i in range(n):
        if src[i] < 0 or Int(src[i]) >= n:
            return False
    return True
