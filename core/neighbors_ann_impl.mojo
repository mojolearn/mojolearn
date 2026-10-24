# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling of the approximate-index owner
(bindings/neighbors_ann_native.mojo, lane native-neighbors-20261024): the
candidate-id step `refine` made with NumPy before the kernel ran, the int32
range refusal (`c.min() < -(1 << 31) or c.max() >= (1 << 31)`) and the
narrowing copy `np.ascontiguousarray(c, dtype=np.int32)`. Integer ids are
read exactly; no arithmetic on distances happens here."""
from std.memory import bitcast


def _ann_item(addr: Int, item: Int, signed: Bool, swap: Bool) -> Tuple[Int64, Bool]:
    """Integer element at `addr` (`item` bytes, little-endian, or reversed
    when `swap`; sign-extended when `signed`) and whether it fits int32."""
    var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    var raw = UInt64(0)
    for k in range(item):  # native input marshaling: the bytes of one item
        var byte = bytes[item - 1 - k] if swap else bytes[k]
        raw |= UInt64(byte) << UInt64(k * 8)
    if not signed:
        return (Int64(0) if raw > UInt64(2147483647) else bitcast[DType.int64](raw),
                raw <= UInt64(2147483647))
    if item < 8:
        var top = UInt64(1) << UInt64(item * 8 - 1)
        if (raw & top) != 0:
            raw |= ~((top << 1) - 1)
    var value = bitcast[DType.int64](raw)
    return (value, value >= Int64(-2147483648) and value <= Int64(2147483647))


def ann_candidates_i32(
    src: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, signed: Bool, swap: Bool,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) -> Bool:
    """The (rows, cols) integer id matrix at `src` (byte strides s0, s1)
    narrowed into the C-order int32 `dst`. False, with nothing written,
    when any id lies outside int32 (the whole matrix is checked first, as
    the min/max refusal ran before the copy)."""
    for r in range(rows):  # native input marshaling: candidate rows, the range refusal
        for c in range(cols):  # native input marshaling: candidate ids of one row
            var got = _ann_item(src + r * s0 + c * s1, item, signed, swap)
            if not got[1]:
                return False
    for r in range(rows):  # native input marshaling: candidate rows, the narrowing copy
        for c in range(cols):  # native input marshaling: candidate ids of one row
            var got = _ann_item(src + r * s0 + c * s1, item, signed, swap)
            dst[r * cols + c] = Int32(got[0])
    return True
