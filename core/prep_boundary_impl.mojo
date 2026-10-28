# SPDX-License-Identifier: Apache-2.0
# Public preprocessing validation and row-buffer preparation. No model math.
from core.cast_in import CastIn


def prep_float_finite(source: CastIn, shape: List[Int], strides: List[Int],
                      count: Int, native: Bool, dtype: Int) -> Bool:
    # _sk_validate scans Float32/Float64 only, before any narrowing to f32.
    if dtype != 11 and dtype != 12:
        return True
    var width = source.itemsize
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.addr)
    var mask = UInt64(0x7F800000) if width == 4 else UInt64(0x7FF0000000000000)
    for linear in range(count):  # native input marshaling: public finiteness check on original float words
        var remainder = linear
        var offset = 0
        for k in range(len(shape)):  # native input marshaling: captured rank metadata
            var axis = len(shape) - 1 - k
            offset += (remainder % shape[axis]) * strides[axis]
            remainder //= shape[axis]
        var bits = UInt64(0)
        for byte in range(width):  # native input marshaling: supported four/eight-byte float words
            var index = byte if native else width - 1 - byte
            bits |= UInt64(pointer[offset + index]) << UInt64(8 * byte)
        if bits & mask == mask:
            return False
    return True


def prep_gather_rows(source: CastIn, indices: Int, rows: Int, output: Int) raises:
    var index = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=indices)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.addr)
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=output)
    for r in range(rows):  # native input marshaling: validate selected row indices before any output write
        var selected = Int(index[r])
        if selected < 0 or selected >= source.rows:
            raise Error("gather_rows_bytes: row index out of bounds")
    for r in range(rows):  # native input marshaling: copy selected rows without numeric conversion
        var selected = Int(index[r])
        for c in range(source.cols):  # native input marshaling: captured source columns
            for byte in range(source.itemsize):  # native input marshaling: raw primitive-buffer bytes
                dst[(r * source.cols + c) * source.itemsize + byte] = src[selected * source.s0 + c * source.s1 + byte]
