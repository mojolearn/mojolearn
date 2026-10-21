# SPDX-License-Identifier: Apache-2.0
"""Typed generation id conversion moved from bindings/generation_buffers.mojo."""


def generation_ids_into(
    addr: Int, item: Int, signed: Bool, b: Int, l: Int, column_major: Bool, vocab: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Native little-endian integer ids (`item` bytes each, C order in the buffer) to int32,
    refusing any id outside [0, vocab) before conversion."""
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(b * l):
        var source = (i // l) + (i % l) * b if column_major else i
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(bytes[source * item + k]) << UInt64(8 * k)
        var negative = signed and ((raw >> UInt64(8 * item - 1)) & UInt64(1)) == UInt64(1)
        if negative or raw >= UInt64(vocab):
            raise Error("generation: ids must be in [0, " + String(vocab) + "); invalid id at flat index " + String(i))
        dst[i] = Int32(Int(raw))
