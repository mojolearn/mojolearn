# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling of the classification metrics owner
(bindings/metrics_cls_native.mojo, lane mcb2-mcls): the integer label reads
`_buffer.as_i64_c` and `Array.astype("<i4")` made for `_metrics_impl._as_i32_1d`
(exact widening of a signed, unsigned or bool buffer, then the int32 range
rule), the label range `lab.min()` / `lab.max()` the entropy entry takes,
and the copies of a kernel's leading outputs (`precision[:size + 1].copy()`).
No arithmetic on metric values happens here: the kernels compute."""


def mcls_read_i64(addr: Int, stride: Int, count: Int, item: Int, signed: Bool,
                  dst: MutPointer[Int64, MutUntrackedOrigin]) -> Bool:
    """`dst[i]` = integer element i of a strided buffer (little-endian
    items of `item` bytes, sign-extended when `signed`). False when an
    unsigned 64-bit value does not fit in int64 (as_i64_c refuses it)."""
    for i in range(count):  # native input marshaling: integer label items, read exactly
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + i * stride)
        var raw = UInt64(0)
        for k in range(item):  # native input marshaling: the bytes of one item
            raw |= UInt64(bytes[k]) << UInt64(k * 8)
        if signed and item < 8:
            var top = UInt64(1) << UInt64(item * 8 - 1)
            if (raw & top) != 0:
                raw |= ~((top << 1) - 1)
        if not signed and item == 8 and raw > UInt64(9223372036854775807):
            return False
        dst[i] = Int64(raw)
    return True


def mcls_narrow_i32(src: MutPointer[Int64, MutUntrackedOrigin], count: Int,
                    dst: MutPointer[Int32, MutUntrackedOrigin]) -> Bool:
    """`a.astype("<i4")` after `_as_i32_1d`'s range check: False (nothing
    written past the first refusal) when a label is outside int32."""
    for i in range(count):  # native input marshaling: the int32 range rule over the labels
        var v = src[i]
        if v < Int64(-2147483648) or v > Int64(2147483647):
            return False
        dst[i] = Int32(v)
    return True


def mcls_min_i32(src: MutPointer[Int32, MutUntrackedOrigin], count: Int) -> Int:
    """`lab.min()` of `count` >= 1 int32 labels."""
    var lo = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked labels
        if src[i] < lo:
            lo = src[i]
    return Int(lo)


def mcls_max_i32(src: MutPointer[Int32, MutUntrackedOrigin], count: Int) -> Int:
    """`lab.max()` of `count` >= 1 int32 labels."""
    var hi = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked labels
        if src[i] > hi:
            hi = src[i]
    return Int(hi)


def mcls_copy_words(src: Int, dst: Int, words: Int):
    """Copy `words` 4-byte words (a kernel output's kept prefix)."""
    var s = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=src)
    var d = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(words):  # native output marshaling: the kept prefix of a kernel output
        d[i] = s[i]
