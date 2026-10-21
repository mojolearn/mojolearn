# SPDX-License-Identifier: Apache-2.0
"""Typed SambaStack token-id conversion moved from bindings/parallel_neural_native.mojo."""
from std.memory import bitcast


def samba_ids_into(
    addr: Int, item: Int, signed: Bool, rows: Int, columns: Int, column_major: Bool,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Native little-endian integer ids (`item` bytes each, C order in the buffer) to int32,
    refusing any value outside int32 before conversion."""
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(rows * columns):
        var source = i % columns * rows + i // columns if column_major else i
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(bytes[source * item + k]) << UInt64(8 * k)
        var negative = signed and ((raw >> UInt64(8 * item - 1)) & UInt64(1)) == UInt64(1)
        var number = Int64(0)
        if negative:
            if item < 8:
                raw |= ~((UInt64(1) << UInt64(8 * item)) - UInt64(1))
            number = bitcast[DType.int64](raw)
            if number < Int64(-2147483648):
                raise Error("OverflowError: Samba token id is outside int32")
        else:
            if raw > UInt64(2147483647):
                raise Error("OverflowError: Samba token id is outside int32")
            number = Int64(raw)
        dst[i] = Int32(number)
