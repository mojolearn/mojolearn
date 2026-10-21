# SPDX-License-Identifier: Apache-2.0
"""Typed Samba token-id conversion and row copies moved from bindings/samba_inference_native.mojo
and bindings/samba_training_native.mojo. Ids arrive as native little-endian integers of
`item` bytes each, in C order; every range check runs on the full-width value."""
from std.memory import bitcast, memcpy


@always_inline
def _id_word(bytes: Pointer[UInt8, ImmUntrackedOrigin], index: Int, item: Int) -> UInt64:
    var raw = UInt64(0)
    for k in range(item):
        raw |= UInt64(bytes[index * item + k]) << UInt64(8 * k)
    return raw


@always_inline
def _id_negative(raw: UInt64, item: Int, signed: Bool) -> Bool:
    return signed and ((raw >> UInt64(8 * item - 1)) & UInt64(1)) == UInt64(1)


@always_inline
def _id_signed(raw: UInt64, item: Int) -> Int64:
    var word = raw
    if item < 8:
        word |= ~((UInt64(1) << UInt64(8 * item)) - UInt64(1))
    return bitcast[DType.int64](word)


def samba_inference_ids_into(
    addr: Int, item: Int, signed: Bool, b: Int, l: Int, column_major: Bool, vocab: Int,
    has_lengths: Bool, sizes: List[Int], dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(b * l):
        if has_lengths and i % l >= sizes[i // l]:
            continue
        var source = i // l + (i % l) * b if column_major else i
        var raw = _id_word(bytes, source, item)
        if _id_negative(raw, item, signed) or raw >= UInt64(vocab):
            raise Error("Samba: input id outside vocabulary")
        dst[i] = Int32(Int(raw))


def samba_training_ids_into(
    addr: Int, item: Int, signed: Bool, b: Int, l: Int, column_major: Bool, vocab: Int,
    targets: Bool, dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(b * l):
        var source = i // l + (i % l) * b if column_major else i
        var raw = _id_word(bytes, source, item)
        var negative = _id_negative(raw, item, signed)
        if negative:
            var value = _id_signed(raw, item)
            if not (targets and value == Int64(-100)):
                raise Error("Samba: token id outside vocabulary at flat index " + String(i))
            dst[i] = Int32(value)
        else:
            if raw >= UInt64(vocab):
                raise Error("Samba: token id outside vocabulary at flat index " + String(i))
            dst[i] = Int32(Int(raw))


def samba_copy_row_prefixes(dst_addr: Int, src_addr: Int, sizes: List[Int], b: Int, l: Int, v: Int):
    for row in range(b):
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr) + row * l * v,
               src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=src_addr) + row * l * v, count=sizes[row] * v)
