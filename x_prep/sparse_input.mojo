# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Native sparse normalization shared by host and device entry points.

No Python, NumPy or SciPy dependency. CSR/CSC/COO/BSR/DIA and dense input
are described by borrowed, possibly strided byte buffers. LIL is marshalled
as CSR and DOK as COO at the foreign-object boundary. This module does the
index validation, format expansion, duplicate fold and dtype conversion.

CSR/BSR retain stored order and duplicates. CSC retains column-major source
order within each output row. COO/DIA use a stable (row, column, source-slot)
order and coalesce duplicates IN THE INPUT DTYPE before narrowing to f32.
That order is explicit; equality with a particular SciPy duplicate-sort order
is NOT asserted. All columns must use the same new normalization version.
New source: NOT COMPILED / NOT EXECUTED / IDENTITY AND QUALITY UNVERIFIED.
"""
from std.memory import bitcast
from checks.soft_f64 import sf64_add, sf64_from_f32, sf64_from_int, sf64_to_f32

comptime SP_CSR = 0
comptime SP_CSC = 1
comptime SP_COO = 2
comptime SP_BSR = 3
comptime SP_DIA = 4
comptime SP_DENSE = 5
comptime SP_COO_KEEP = 6
comptime SP_MAX = 2147483647
comptime SP_BAD = UInt32(0xFFFFFFFF)
comptime BP = MutPointer[UInt8, MutAnyOrigin]
comptime UP = MutPointer[UInt32, MutAnyOrigin]
comptime WP = MutPointer[UInt64, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]
comptime FP = MutPointer[Float32, MutAnyOrigin]


@fieldwise_init
struct SparseView(Copyable, Movable, TrivialRegisterPassable):
    """Byte buffer, flattened in C index order. origin accounts for negative strides.

    Codes: 0 f32, 1 f64, 2 i32, 3 i64, 4 u32, 5 u8, 6 u16, 7 i8,
    8 f16, 9 i16, 10 u64, 11 bool. width's sign specifies byte order.
    """
    var data: BP
    var code: Int
    var width: Int
    var dim1: Int
    var dim2: Int
    var stride0: Int
    var stride1: Int
    var stride2: Int
    var origin: Int
    var items: Int


@fieldwise_init
struct SparseShape(Copyable, Movable, TrivialRegisterPassable):
    var kind: Int
    var rows: Int
    var cols: Int
    var entries: Int
    var segments: Int
    var block_rows: Int
    var block_cols: Int


def validate_shape(s: SparseShape) raises:
    if s.kind < SP_CSR or s.kind > SP_COO_KEEP or s.rows < 1 or s.cols < 1:
        raise Error("sparse input: expected a supported, nonempty 2-D matrix")
    if s.rows >= SP_MAX or s.cols > SP_MAX or s.entries < 0 or s.entries > SP_MAX:
        raise Error("sparse input: exceeds the Int32 indexing bound")
    if s.segments < 0 or s.segments >= SP_MAX:
        raise Error("sparse input: invalid compressed pointer length")
    if s.kind == SP_BSR and (s.block_rows < 1 or s.block_cols < 1
                            or s.rows % s.block_rows != 0 or s.cols % s.block_cols != 0):
        raise Error("sparse input: invalid BSR block shape")
    if s.kind == SP_DENSE and (s.rows > SP_MAX // s.cols or s.entries != s.rows * s.cols):
        raise Error("sparse input: invalid or oversized dense shape")


def validate_parts(s: SparseShape, a: SparseView, b: SparseView, v: SparseView) raises:
    validate_shape(s)
    validate_view(a, True)
    validate_view(b, True)
    validate_view(v)
    if v.items != s.entries:
        raise Error("sparse input: data buffer length does not match storage shape")
    var compressed = s.kind == SP_CSR or s.kind == SP_CSC or s.kind == SP_BSR
    var slots = s.entries
    if s.kind == SP_BSR:
        var area = s.block_rows * s.block_cols
        if slots % area != 0:
            raise Error("sparse input: BSR data buffer has a partial block")
        slots //= area
    if compressed:
        var segments = s.cols if s.kind == SP_CSC else s.rows
        if s.kind == SP_BSR:
            segments //= s.block_rows
        if s.segments != segments or a.items != segments + 1 or b.items != slots:
            raise Error("sparse input: compressed buffer lengths do not match shape")
    elif s.kind == SP_COO or s.kind == SP_COO_KEEP:
        if a.items != s.entries or b.items != s.entries:
            raise Error("sparse input: coordinate buffer lengths do not match data")
    elif s.kind == SP_DIA:
        if v.dim1 < 1 or v.dim2 != 1 or s.entries % v.dim1 != 0 or a.items != s.entries // v.dim1:
            raise Error("sparse input: diagonal storage shape does not match offsets")


def validate_view(v: SparseView, indices: Bool = False) raises:
    if v.code < 0 or v.code > 11 or v.items < 0 or v.items > SP_MAX or v.dim1 < 1 or v.dim2 < 1:
        raise Error("sparse input: invalid native buffer descriptor")
    var width = 1
    if v.code == 0 or v.code == 2 or v.code == 4:
        width = 4
    elif v.code == 1 or v.code == 3 or v.code == 10:
        width = 8
    elif v.code == 6 or v.code == 8 or v.code == 9:
        width = 2
    if abs(v.width) != width:
        raise Error("sparse input: dtype and storage width differ")
    if indices and (v.code == 0 or v.code == 1 or v.code == 8):
        raise Error("sparse input: integer indices required")


@always_inline
def sparse_bits(v: SparseView, i: Int) -> UInt64:
    var j = i // v.dim2
    var offset = v.origin + (j // v.dim1) * v.stride0 + (j % v.dim1) * v.stride1 + (i % v.dim2) * v.stride2
    var width = abs(v.width)
    var bits = UInt64(0)
    for b in range(width):
        var shift = b if v.width > 0 else width - 1 - b
        bits |= UInt64(v.data.unsafe_load(offset + b)) << UInt64(8 * shift)
    return bits


@always_inline
def sparse_index(v: SparseView, i: Int) -> Int:
    var bits = sparse_bits(v, i)
    if v.code == 2 or v.code == 3 or v.code == 7 or v.code == 9:
        var shift = 64 - 8 * abs(v.width)
        return Int(bitcast[DType.int64](bits << UInt64(shift)) >> shift)
    if bits > UInt64(SP_MAX):
        return -1
    return Int(bits)


@always_inline
def sparse_f32(bits: UInt64, code: Int, width: Int) -> Float32:
    if code == 0:
        return bitcast[DType.float32](UInt32(bits))
    if code == 1:
        return sf64_to_f32(bits)
    if code == 8:
        return Float32(bitcast[DType.float16](UInt16(bits)))
    if code == 2 or code == 3 or code == 7 or code == 9:
        var shift = 64 - 8 * abs(width)
        var value = Int(bitcast[DType.int64](bits << UInt64(shift)) >> shift)
        return sf64_to_f32(sf64_from_int(value))
    # u64's sign bit is refused by sparse_expand_unit: the incumbent boundary
    # refused values outside its signed-int64 Array representation too.
    return sf64_to_f32(sf64_from_int(Int(bits)))


@always_inline
def sparse_nonzero(bits: UInt64, code: Int) -> Bool:
    if code == 0:
        return (bits & UInt64(0x7FFFFFFF)) != 0
    if code == 1:
        return (bits & UInt64(0x7FFFFFFFFFFFFFFF)) != 0
    if code == 8:
        return (bits & UInt64(0x7FFF)) != 0
    return bits != 0


@always_inline
def sparse_add(a: UInt64, b: UInt64, code: Int, width: Int) -> UInt64:
    if code == 1:
        return sf64_add(a, b)
    if code == 0 or code == 8:
        var total = sf64_to_f32(sf64_add(sf64_from_f32(sparse_f32(a, code, width)),
                                       sf64_from_f32(sparse_f32(b, code, width))))
        if code == 0:
            return UInt64(bitcast[DType.uint32](total))
        return UInt64(bitcast[DType.uint16](Float16(total)))
    if code == 11:
        return UInt64(1) if a != 0 or b != 0 else UInt64(0)
    var total = a + b
    if abs(width) < 8:
        total &= (UInt64(1) << UInt64(8 * abs(width))) - 1
    return total


@always_inline
def sparse_coalesce(s: SparseShape) -> Bool:
    return s.kind == SP_COO or s.kind == SP_DIA


def sparse_pointer_unit(t: Int, s: SparseShape, a: SparseView, bad: IP):
    """One status per pointer; never a racing shared status write."""
    if s.kind != SP_CSR and s.kind != SP_CSC and s.kind != SP_BSR:
        bad.unsafe_store(t, Int32(0))
        return
    var cap = s.entries
    if s.kind == SP_BSR:
        cap //= s.block_rows * s.block_cols
    var p = sparse_index(a, t)
    var fail = p < 0 or p > cap or (t == 0 and p != 0)
    if t < s.segments:
        fail = fail or p > sparse_index(a, t + 1)
    bad.unsafe_store(t, Int32(1) if fail else Int32(0))


@always_inline
def _sparse_segment(a: SparseView, segments: Int, e: Int) -> Int:
    var lo = 0
    var hi = segments
    while lo < hi:
        var mid = (lo + hi) // 2
        if sparse_index(a, mid + 1) <= e:
            lo = mid + 1
        else:
            hi = mid
    return lo


def sparse_expand_unit(t: Int, s: SparseShape, a: SparseView, b: SparseView, v: SparseView,
                       rows: UP, cols: UP, vals: WP, bad: IP):
    var r = 0
    var c = 0
    var keep = True
    var bits = sparse_bits(v, t)
    if s.kind == SP_CSR:
        r = _sparse_segment(a, s.segments, t)
        c = sparse_index(b, t)
        keep = t < sparse_index(a, s.segments)
    elif s.kind == SP_CSC:
        c = _sparse_segment(a, s.segments, t)
        r = sparse_index(b, t)
        keep = t < sparse_index(a, s.segments)
    elif s.kind == SP_COO or s.kind == SP_COO_KEEP:
        r = sparse_index(a, t)
        c = sparse_index(b, t)
    elif s.kind == SP_BSR:
        var area = s.block_rows * s.block_cols
        var e = t // area
        r = _sparse_segment(a, s.segments, e) * s.block_rows + (t % area) // s.block_cols
        var block_col = sparse_index(b, e)
        c = block_col * s.block_cols + t % s.block_cols if block_col >= 0 and block_col < s.cols // s.block_cols else -1
        keep = e < sparse_index(a, s.segments)
    elif s.kind == SP_DIA:
        # DIA storage's column j represents (j - offset, j); padding and
        # explicit zeros are absent from its COO/CSR representation.
        c = t % v.dim1
        r = c - sparse_index(a, t // v.dim1)
        keep = r >= 0 and r < s.rows and c < s.cols and sparse_nonzero(bits, v.code)
    else:
        r = t // s.cols
        c = t % s.cols
        keep = sparse_nonzero(bits, v.code)
    var fail = keep and (r < 0 or r >= s.rows or c < 0 or c >= s.cols)
    fail = fail or (keep and v.code == 10 and bits > UInt64(0x7FFFFFFFFFFFFFFF))
    bad.unsafe_store(t, Int32(1) if fail else Int32(0))
    rows.unsafe_store(t, UInt32(r) if keep and not fail else SP_BAD)
    cols.unsafe_store(t, UInt32(c) if keep and not fail else SP_BAD)
    vals.unsafe_store(t, bits)


@always_inline
def sparse_lower(keys: UP, n: Int, key: UInt32) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if keys.unsafe_load(mid) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def sparse_count_unit(r: Int, s: SparseShape, keys: UP, order: UP, cols: UP, counts: IP):
    if r == s.rows:
        counts.unsafe_store(r, Int32(0))
        return
    var lo = sparse_lower(keys, s.entries, UInt32(r))
    var hi = sparse_lower(keys, s.entries, UInt32(r + 1))
    var count = hi - lo
    if sparse_coalesce(s):
        count = 0
        var prev = SP_BAD
        for p in range(lo, hi):
            var c = cols.unsafe_load(Int(order.unsafe_load(p)))
            if c != prev:
                count += 1
            prev = c
    counts.unsafe_store(r, Int32(count))


def sparse_write_unit(r: Int, s: SparseShape, v: SparseView, keys: UP, order: UP,
                      cols: UP, vals: WP, indptr: IP, indices: IP, data: FP):
    var lo = sparse_lower(keys, s.entries, UInt32(r))
    var hi = sparse_lower(keys, s.entries, UInt32(r + 1))
    var out = Int(indptr.unsafe_load(r))
    var p = lo
    while p < hi:
        var e = Int(order.unsafe_load(p))
        var c = cols.unsafe_load(e)
        var value = vals.unsafe_load(e)
        var code = v.code
        var width = v.width
        if sparse_coalesce(s) and code == 8:
            # The sparse converter's half input is promoted to float32
            # before duplicate accumulation, as in the incumbent COO path.
            value = UInt64(bitcast[DType.uint32](sparse_f32(value, code, width)))
            code = 0
            width = 4
        p += 1
        if sparse_coalesce(s):
            while p < hi:
                e = Int(order.unsafe_load(p))
                if cols.unsafe_load(e) != c:
                    break
                var next_value = vals.unsafe_load(e)
                if v.code == 8:
                    next_value = UInt64(bitcast[DType.uint32](sparse_f32(next_value, v.code, v.width)))
                value = sparse_add(value, next_value, code, width)
                p += 1
        indices.unsafe_store(out, Int32(c))
        data.unsafe_store(out, sparse_f32(value, code, width))
        out += 1


def sparse_direct_count_unit(r: Int, s: SparseShape, a: SparseView, v: SparseView, counts: IP):
    """CSR preserves pointers; dense uses a row count followed by an exclusive scan.

    These two common input forms need no triples or radix-sort workspace.
    In particular a dense input does NOT allocate O(rows*cols) index scratch.
    """
    if s.kind == SP_CSR:
        counts.unsafe_store(r, Int32(sparse_index(a, r)))
        return
    var count = 0
    if r < s.rows:
        for c in range(s.cols):
            if sparse_nonzero(sparse_bits(v, r * s.cols + c), v.code):
                count += 1
    counts.unsafe_store(r, Int32(count))


def sparse_direct_write_unit(r: Int, s: SparseShape, b: SparseView, v: SparseView,
                             indptr: IP, indices: IP, data: FP, bad: IP):
    var out = Int(indptr.unsafe_load(r))
    var lo = out if s.kind == SP_CSR else r * s.cols
    var hi = Int(indptr.unsafe_load(r + 1)) if s.kind == SP_CSR else (r + 1) * s.cols
    var fail = False
    for e in range(lo, hi):
        var bits = sparse_bits(v, e)
        if s.kind == SP_DENSE and not sparse_nonzero(bits, v.code):
            continue
        var c = sparse_index(b, e) if s.kind == SP_CSR else e - lo
        var invalid = c < 0 or c >= s.cols or (v.code == 10 and bits > UInt64(0x7FFFFFFFFFFFFFFF))
        fail = fail or invalid
        if not invalid:
            indices.unsafe_store(out, Int32(c))
            data.unsafe_store(out, sparse_f32(bits, v.code, v.width))
        out += 1
    bad.unsafe_store(r, Int32(1) if fail else Int32(0))
