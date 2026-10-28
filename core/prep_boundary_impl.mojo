# SPDX-License-Identifier: Apache-2.0
# Public preprocessing validation and row-buffer preparation. No model math.
from core.cast_in import CastIn, cast_w64_bits
from std.memory import bitcast
from core.classical_pandas_impl import pd_raw_at, _pd_double


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


def prep_mask_any(source: CastIn, count: Int) -> Bool:
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.addr)
    for i in range(count):  # native input marshaling: captured one-byte nullable mask
        if pointer[i * source.s0] != 0:
            return True
    return False


def prep_category_flags(codes: CastIn, count: Int, codes_native: Bool,
                        categories: CastIn, category_count: Int,
                        category_native: Bool, dtype: Int) raises -> Tuple[Bool, Bool]:
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=codes.addr)
    var values = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=categories.addr)
    var missing = False
    var finite = True
    for i in range(count):  # native input marshaling: categorical label codes, no Python take/reshape
        var bits = UInt64(0)
        for byte in range(codes.itemsize):  # native input marshaling: admitted signed index widths
            var offset = byte if codes_native else codes.itemsize - 1 - byte
            bits |= UInt64(pointer[i * codes.s0 + offset]) << UInt64(8 * byte)
        var index = Int(bitcast[Int64](cast_w64_bits(bits, codes.kind, codes.itemsize)))
        if index == -1:
            missing = True
            continue
        if index < 0 or index >= category_count:
            raise Error("TypeError: pandas categorical code exceeds its category storage")
        if dtype == 11 or dtype == 12:
            var raw = UInt64(0)
            for byte in range(categories.itemsize):  # native input marshaling: original float category words
                var offset = byte if category_native else categories.itemsize - 1 - byte
                raw |= UInt64(values[index * categories.s0 + offset]) << UInt64(8 * byte)
            var mask = UInt64(0x7F800000) if dtype == 11 else UInt64(0x7FF0000000000000)
            finite = finite and (raw & mask != mask)
    return (missing, finite)


def prep_arrow_finite(address: Int, length: Int, offset: Int, width: Int) -> Bool:
    var source = CastIn(address + offset * width, 2, width, length, 1, width, 0)
    var shape = List[Int]()
    shape.append(length)
    var strides = List[Int]()
    strides.append(width)
    return prep_float_finite(source, shape, strides, length, True, 11 if width == 4 else 12)


def _prep_export_word(raw: UInt64, source: CastIn, number: Int) -> UInt64:
    if number == 12 and (source.kind != 2 or source.itemsize != 8):
        return bitcast[UInt64](_pd_double(raw, source.kind, source.itemsize))
    return raw


def _prep_missing_word(width: Int) -> UInt64:
    return UInt64(0x7E00) if width == 2 else (UInt64(0x7FC00000) if width == 4 else UInt64(0x7FF8000000000000))


def _prep_store_word(address: Int, row: Int, width: Int, word: UInt64):
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address + row * width)
    for byte in range(width):  # native input marshaling: original primitive export width
        out[byte] = UInt8(word >> UInt64(8 * byte))


def prep_mask_export(source: CastIn, mask: CastIn, rows: Int, native: Bool,
                     output: CastIn, number: Int):
    var flags = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=mask.addr)
    for row in range(rows):  # native input marshaling: nullable export before public validation
        var word = _prep_missing_word(output.itemsize)
        if flags[row * mask.s0] == 0:
            word = _prep_export_word(pd_raw_at(source.addr, row * source.s0, source.itemsize, native), source, number)
        _prep_store_word(output.addr, row, output.itemsize, word)


def prep_category_export(codes: CastIn, rows: Int, codes_native: Bool,
                         labels: CastIn, label_count: Int, labels_native: Bool,
                         output: CastIn, number: Int) raises:
    for row in range(rows):  # native input marshaling: native categorical take, preserving export width
        var raw = pd_raw_at(codes.addr, row * codes.s0, codes.itemsize, codes_native)
        var index = Int(bitcast[Int64](cast_w64_bits(raw, codes.kind, codes.itemsize)))
        if index < -1 or index >= label_count:
            raise Error("TypeError: pandas categorical code exceeds its category storage")
        var word = _prep_missing_word(output.itemsize)
        if index >= 0:
            word = _prep_export_word(pd_raw_at(labels.addr, index * labels.s0, labels.itemsize, labels_native), labels, number)
        _prep_store_word(output.addr, row, output.itemsize, word)
