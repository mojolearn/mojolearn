# SPDX-License-Identifier: Apache-2.0
# Public preprocessing validation and row-buffer preparation. No model math.
from core.cast_in import CastIn, cast_w64_bits
from std.memory import bitcast, memcpy
from core.admitted_input import admitted_raw_word
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.classical_pandas_impl import pd_raw_at, _pd_double
from core.host_finite_scan import host_all_finite_f32, host_all_finite_f64, host_all_finite_f32_strided


comptime _PREP_IP = MutPointer[Int, MutUntrackedOrigin]


@always_inline
def _prep_word_offset(linear: Int, shape: _PREP_IP, strides: _PREP_IP, rank: Int) -> Int:
    """The byte offset of C-order element `linear` of a strided N-d view."""
    var remainder = linear
    var offset = 0
    for k in range(rank):  # small-loop(the array's rank metadata)
        var axis = rank - 1 - k
        var extent = shape.unsafe_load(axis)
        offset += (remainder % extent) * strides.unsafe_load(axis)
        remainder //= extent
    return offset


def prep_float_finite(source: CastIn, shape: List[Int], strides: List[Int],
                      count: Int, native: Bool, dtype: Int) -> Bool:
    """True when no float32 (dtype 11) or float64 (dtype 12) word of the
    strided view is a NaN or an infinity; True for any other dtype
    (_sk_validate scans Float32/Float64 only, before any narrowing to f32).
    The verdict is the exponent-bits test on the original words, an AND over
    tasks, so the split cannot change it."""
    if dtype != 11 and dtype != 12:
        return True
    var width = source.itemsize
    if native and (width == 4 or width == 8):
        var contiguous = True
        var expected = width
        for k in range(len(shape)):  # driver-loop: over the array's rank metadata
            var axis = len(shape) - 1 - k
            if shape[axis] != 1 and strides[axis] != expected:
                contiguous = False
            expected *= shape[axis]
        if contiguous:
            if width == 4:
                return host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source.addr), count)
            return host_all_finite_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=source.addr), count)
        if width == 4 and len(shape) == 2:
            return host_all_finite_f32_strided(source.addr, strides[0], strides[1], shape[0], shape[1])
    # parallel: the strided / non-native finiteness test over contiguous
    # element ranges across host cores (it was a serial byte walk)
    if count <= 0:
        return True
    var mask = UInt64(0x7F800000) if width == 4 else UInt64(0x7FF0000000000000)
    var tasks = host_predict_task_count_work(count, count * len(shape))
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())
    var base = source.addr
    var rank = len(shape)
    var shape_ptr = shape.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var strides_ptr = strides.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()

    def _span(t: Int) {imm base, imm width, imm native, imm mask, imm count, imm chunk, imm flag_ptr, imm shape_ptr, imm strides_ptr, imm rank}:
        var lo = t * chunk
        var hi = min(lo + chunk, count)
        var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base)
        for linear in range(lo, hi):  # admitted input: this task's elements, finiteness on original float words
            var offset = _prep_word_offset(linear, shape_ptr, strides_ptr, rank)
            var bits = UInt64(0)
            if native:
                bits = admitted_raw_word(base + offset, width)
            else:
                for byte in range(width):  # small-loop(bytes of one non-native float word): 4 or 8
                    bits |= UInt64(pointer[offset + width - 1 - byte]) << UInt64(8 * byte)
            if bits & mask == mask:
                flag_ptr.unsafe_store(t, UInt8(0))
                return

    if tasks <= 1:
        _span(0)
    else:
        host_parallelize(_span, tasks)
    var finite = True
    for t in range(tasks):  # driver-loop: over the scan tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            finite = False
    _ = flags^
    return finite


def prep_gather_rows(source: CastIn, indices: Int, rows: Int, output: Int) raises:
    """The selected rows of the source matrix, raw bytes in C order, at
    `output` (`rows` x source.cols items). Every index is checked before any
    row is written; no numeric conversion."""
    # parallel: index check, then one copy per selected row (a memcpy when the
    # row is contiguous), row tasks across host cores (it was a serial byte walk)
    if rows <= 0:
        return
    var index = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=indices)
    var limit = source.rows
    var item = source.itemsize
    var cols = source.cols
    var tasks = host_predict_task_count_work(rows, rows * max(cols * item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _check(t: Int) {imm index, imm limit, imm rows, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # admitted input: this task's selected row indices
            var selected = Int(index.unsafe_load(r))
            if selected < 0 or selected >= limit:
                flag_ptr.unsafe_store(t, UInt8(0))
                return

    if tasks <= 1:
        _check(0)
    else:
        host_parallelize(_check, tasks)
    for t in range(tasks):  # driver-loop: over the check tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            raise Error("gather_rows_bytes: row index out of bounds")
    _ = flags^
    var base = source.addr
    var s0 = source.s0
    var s1 = source.s1
    var row_bytes = cols * item

    def _copy(t: Int) {imm index, imm base, imm s0, imm s1, imm item, imm cols, imm row_bytes, imm output, imm rows, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=output)
        for r in range(lo, hi):  # admitted input: this task's selected rows
            var start = base + Int(index.unsafe_load(r)) * s0
            if s1 == item or cols == 1:
                memcpy(dest=dst.unsafe_offset(r * row_bytes),
                       src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=start), count=row_bytes)
            else:
                for c in range(cols):  # admitted input: strided items of one selected row
                    memcpy(dest=dst.unsafe_offset(r * row_bytes + c * item),
                           src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=start + c * s1), count=item)

    if tasks <= 1:
        _copy(0)
    else:
        host_parallelize(_copy, tasks)


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
        var index = Int(bitcast[DType.int64](cast_w64_bits(bits, codes.kind, codes.itemsize)))
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
        return bitcast[DType.uint64](_pd_double(raw, source.kind, source.itemsize))
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
        var index = Int(bitcast[DType.int64](cast_w64_bits(raw, codes.kind, codes.itemsize)))
        if index < -1 or index >= label_count:
            raise Error("TypeError: pandas categorical code exceeds its category storage")
        var word = _prep_missing_word(output.itemsize)
        if index >= 0:
            word = _prep_export_word(pd_raw_at(labels.addr, index * labels.s0, labels.itemsize, labels_native), labels, number)
        _prep_store_word(output.addr, row, output.itemsize, word)
