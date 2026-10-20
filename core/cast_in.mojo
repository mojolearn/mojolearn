# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A caller's array as it is: its address, NumPy dtype number, item size,
shape and byte strides (`CastIn`), and the element conversions the entries
that take one apply in Mojo (lane ag-thin-glue, 2026-10-19). Python passes
the user's buffer untouched; no copy, cast or reorder happens there.

Accepted element types (NumPy `dtype.num` with the item size checked
against it): bool (0), the signed integers int8..int64 (1, 3, 5, 7, 9), the
unsigned integers uint8..uint64 (2, 4, 6, 8, 10), float16 (23), float32 (11)
and float64 (12). Accepted layouts: any byte strides, negative (a reversed
view) included, as long as no two elements overlap (lane ag-thin-glue2).
The span the cast reads starts at the lowest byte of the four corner
elements (`start()`); element (r, c) sits at `origin() + r * s0 + c * s1`
from it.

Conversions, integer operations only, so every backend (and the host
column) gives the same bits:
  * to float32 (`cast_f32_bits`): float32 as is (the bits, subnormals
    included); float16 by its exact bit expansion (`f16_to_f32_bits`, NaN ->
    the quiet NaN 0x7FC00000); bfloat16 (weight import's CAST_KIND_BFLOAT)
    by the exact 16-bit shift (`bf16_to_f32_bits`); float64 by round-to-nearest-even narrowing
    (`checks/soft_f64.mojo::sf64_to_f32`'s statements, the native
    `cast_f64_to_f32` the estimators' input path takes, NaN -> the quiet
    NaN 0x7FC00000); integers by round-to-nearest-even from their exact
    value (`numpy.astype(float32)`'s result); bool to 0.0 or 1.0.
  * to a 64-bit label word (`cast_w64_bits`, the label sort's input): float
    -> float64 bits (float32 widened exactly), signed -> sign-extended
    int64, unsigned and bool -> zero-extended (uint64 is refused for labels:
    its upper half does not fit the int64 key; float16 labels are refused).
  * back from a 64-bit label word to the caller's item (`narrow_w64`): the
    exact inverse of `cast_w64_bits` for every value it produced.

Byte access. The array's bytes travel in 4-byte executor words (the span is
uploaded as it is; a 1..3-byte tail goes in one zero-padded word). An item
is read from the one to three words its bytes cover (little-endian, as on
every target), so a stride that is not a multiple of the item size (a field
of a structured array) reads the right bytes too; only the words the item
covers are read, never one past the span."""
from std.memory import bitcast
from sequence.ops import FP

comptime CAST_KIND_UNSIGNED = 0
comptime CAST_KIND_SIGNED = 1
comptime CAST_KIND_FLOAT = 2
comptime CAST_KIND_BOOL = 3
# bfloat16 (2-byte items): weight import only (autograd/weight_import.mojo
# names it; NumPy has no bfloat16 dtype number, so `cast_in_kind` never
# returns it and no other entry sees it)
comptime CAST_KIND_BFLOAT = 4

# NumPy's dtype numbers (numpy/_core/include/numpy/ndarraytypes.h NPY_TYPES)
comptime NPY_BOOL = 0
comptime NPY_FLOAT = 11
comptime NPY_DOUBLE = 12
comptime NPY_ULONGLONG = 10
comptime NPY_HALF = 23


def cast_in_kind(dtype_num: Int, itemsize: Int, what: String) raises -> Int:
    """The element kind of NumPy dtype number `dtype_num` (CAST_KIND_*), or a
    refusal naming `what`."""
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        raise Error(what + ": item size " + String(itemsize) + " is not supported")
    if dtype_num == NPY_BOOL:
        if itemsize != 1:
            raise Error(what + ": a bool array must have 1-byte items")
        return CAST_KIND_BOOL
    if dtype_num >= 1 and dtype_num <= NPY_ULONGLONG:
        # 1 byte, 3 short, 5 int, 7 long, 9 longlong are signed; the even
        # numbers their unsigned twins (the width comes from the item size,
        # so long's platform size needs no table)
        return CAST_KIND_SIGNED if dtype_num % 2 == 1 else CAST_KIND_UNSIGNED
    if dtype_num == NPY_HALF:
        if itemsize != 2:
            raise Error(what + ": float item size does not match its dtype")
        return CAST_KIND_FLOAT
    if dtype_num == NPY_FLOAT or dtype_num == NPY_DOUBLE:
        if itemsize != (4 if dtype_num == NPY_FLOAT else 8):
            raise Error(what + ": float item size does not match its dtype")
        return CAST_KIND_FLOAT
    raise Error(
        what + ": dtype number " + String(dtype_num)
        + " is not supported; pass a float16, float32, float64, integer or bool array"
    )


struct CastIn(ImplicitlyCopyable, Movable):
    """One caller array: base address, element kind and size, (rows, cols)
    and byte strides (signed: a reversed view has a negative stride; `addr`
    is element (0, 0)). A 1-D array of n is (n, 1); a scalar (1, 1)."""

    var addr: Int
    var kind: Int
    var itemsize: Int
    var rows: Int
    var cols: Int
    var s0: Int
    var s1: Int

    def __init__(out self, addr: Int, kind: Int, itemsize: Int, rows: Int, cols: Int, s0: Int, s1: Int):
        self.addr = addr
        self.kind = kind
        self.itemsize = itemsize
        self.rows = rows
        self.cols = cols
        self.s0 = s0
        self.s1 = s1

    @staticmethod
    def f32_c(addr: Int, rows: Int, cols: Int) -> CastIn:
        """A C-contiguous float32 (rows, cols) buffer (the entries' own
        arrays, such as the class codes)."""
        return CastIn(addr, CAST_KIND_FLOAT, 4, rows, cols, cols * 4, 4)

    def count(self) -> Int:
        return self.rows * self.cols

    def lo_off(self) -> Int:
        """Byte offset (<= 0) from element (0, 0) to the lowest byte of the
        array: the sum of the negative corner steps."""
        var d0 = (self.rows - 1) * self.s0
        var d1 = (self.cols - 1) * self.s1
        return (d0 if d0 < 0 else 0) + (d1 if d1 < 0 else 0)

    def hi_end(self) -> Int:
        """Byte offset from element (0, 0) to the end of the highest item."""
        var d0 = (self.rows - 1) * self.s0
        var d1 = (self.cols - 1) * self.s1
        return (d0 if d0 > 0 else 0) + (d1 if d1 > 0 else 0) + self.itemsize

    def start(self) -> Int:
        """Host address of the span's first byte (the lowest one)."""
        return self.addr + self.lo_off()

    def origin(self) -> Int:
        """Byte offset of element (0, 0) from `start()`: element (r, c) is
        at origin() + r * s0 + c * s1 of the span, never negative."""
        return -self.lo_off()

    def nbytes(self) -> Int:
        """Bytes from the lowest byte to the end of the highest item: the
        span the cast reads (column slices, row-stepped and reversed views
        included)."""
        return self.hi_end() - self.lo_off()

    def words(self) -> Int:
        """4-byte words holding the span (the last one zero-padded)."""
        return (self.nbytes() + 3) // 4

    def is_f32_c(self) -> Bool:
        """float32 in C order: uploaded as it is, no cast launch."""
        return self.kind == CAST_KIND_FLOAT and self.itemsize == 4 and self.is_c_order()

    def is_c_order(self) -> Bool:
        return (self.cols == 1 or self.s1 == self.itemsize) and (
            self.rows == 1 or self.s0 == self.cols * self.itemsize
        )

    def is_f_order(self) -> Bool:
        return (self.rows == 1 or self.s0 == self.itemsize) and (
            self.cols == 1 or self.s1 == self.rows * self.itemsize
        )

    def check(self, what: String) raises:
        """Refuses a null buffer, an empty or oversized shape, and
        overlapping elements (a stride narrower than the item)."""
        if self.addr == 0:
            raise Error(what + ": null buffer address")
        if self.rows < 1 or self.cols < 1 or self.rows > ((1 << 40) // self.cols):
            raise Error(what + ": the array must not be empty")
        # Any strides are fine, negative (a reversed view) included: the span starts at the lowest
        # corner byte and the kernel addresses by signed byte stride. A stride narrower than the item
        # along a dimension of more than one element (overlapping elements) is refused.
        var a0 = self.s0 if self.s0 >= 0 else -self.s0
        var a1 = self.s1 if self.s1 >= 0 else -self.s1
        if (self.rows > 1 and a0 < self.itemsize) or (self.cols > 1 and a1 < self.itemsize):
            raise Error(what + ": the array's elements overlap (a stride is narrower than one item); pass numpy.ascontiguousarray of it")
        if self.nbytes() >= (1 << 33):
            raise Error(what + ": the array is larger than 8 GiB")

    def byte_offset(self, t: Int) -> Int:
        """Byte offset from `start()` of row-major element t (row t // cols,
        column t % cols)."""
        var r = t // self.cols
        var c = t - r * self.cols
        return self.origin() + r * self.s0 + c * self.s1


@always_inline
def _word(w: FP, i: Int) -> UInt64:
    return UInt64(bitcast[DType.uint32](w.unsafe_load(i)))


@always_inline
def cast_raw_bits(w: FP, off: Int, itemsize: Int) -> UInt64:
    """The item at byte offset `off` (>= 0) of the words `w`, zero-extended.
    Reads only the one to three words the item's bytes cover; an item at a
    multiple of its size reads exactly the words it did before."""
    var wi = off >> 2
    var b = off & 3
    var last = (off + itemsize - 1) >> 2
    var v = _word(w, wi) >> UInt64(b * 8)
    if last > wi:
        v |= _word(w, wi + 1) << UInt64(32 - b * 8)
    if last > wi + 1:  # an 8-byte item off a word boundary (b > 0)
        v |= _word(w, wi + 2) << UInt64(64 - b * 8)
    if itemsize == 8:
        return v
    return v & ((UInt64(1) << UInt64(itemsize * 8)) - 1)


@always_inline
def _sign_extend(raw: UInt64, itemsize: Int) -> UInt64:
    """`raw`'s low itemsize bytes as a two's-complement int64 (bits)."""
    if itemsize == 8:
        return raw
    var bits = UInt64(itemsize * 8)
    var top = UInt64(1) << (bits - 1)
    if (raw & top) != 0:
        return raw | (~UInt64(0) << bits)
    return raw


@always_inline
def _top_bit(x: UInt64) -> Int:
    """Index of x's highest set bit (x != 0), by shifts only."""
    var e = 0
    var v = x
    if (v >> 32) != 0:
        e += 32
        v >>= 32
    if (v >> 16) != 0:
        e += 16
        v >>= 16
    if (v >> 8) != 0:
        e += 8
        v >>= 8
    if (v >> 4) != 0:
        e += 4
        v >>= 4
    if (v >> 2) != 0:
        e += 2
        v >>= 2
    if (v >> 1) != 0:
        e += 1
    return e


@always_inline
def int_to_f32_bits(neg: Bool, mag: UInt64) -> UInt32:
    """The float32 nearest (ties to even) to +-mag, as bits."""
    if mag == 0:
        return UInt32(0)
    var e = _top_bit(mag)
    var sig: UInt64
    if e <= 23:
        sig = mag << UInt64(23 - e)
    else:
        var sh = UInt64(e - 23)
        sig = mag >> sh
        var rem = mag & ((UInt64(1) << sh) - 1)
        var half = UInt64(1) << (sh - 1)
        if rem > half or (rem == half and (sig & 1) != 0):
            sig += 1
            if sig == (UInt64(1) << 24):
                sig >>= 1
                e += 1
    var s = UInt32(0x80000000) if neg else UInt32(0)
    return s | (UInt32(e + 127) << 23) | UInt32(sig & UInt64(0x7FFFFF))


@always_inline
def _jam(a: UInt64, dist: Int) -> UInt64:
    """`softfloat_shiftRightJam64` (checks/soft_f64.mojo `_jam`)."""
    if dist <= 0:
        return a
    if dist < 63:
        var lost = a << UInt64(64 - dist)
        return (a >> UInt64(dist)) | (UInt64(1) if lost != 0 else UInt64(0))
    return UInt64(1) if a != 0 else UInt64(0)


@always_inline
def f64_to_f32_bits(a: UInt64) -> UInt32:
    """Round-to-nearest-even narrowing: checks/soft_f64.mojo `sf64_to_f32`'s
    statements, inlined for kernel bodies."""
    var s = a >> 63
    var e = Int((a >> 52) & UInt64(0x7FF))
    var f = a & UInt64(0x000FFFFFFFFFFFFF)
    if e == 0x7FF:
        if f != 0:
            return UInt32(0x7FC00000)
        return UInt32((s << 31) | UInt64(0x7F800000))
    var f30 = _jam(f, 22)
    if e == 0 and f30 == 0:
        return UInt32(s << 31)
    var exp = e - 0x381
    var sig = f30 | UInt64(0x40000000)
    var round_bits = sig & UInt64(0x7F)
    if exp < 0:
        sig = _jam(sig, -exp)
        exp = 0
        round_bits = sig & UInt64(0x7F)
    elif exp > 0xFD or (exp == 0xFD and sig + UInt64(0x40) >= UInt64(0x80000000)):
        return UInt32((s << 31) | UInt64(0x7F800000))
    sig = (sig + UInt64(0x40)) >> 7
    if round_bits == UInt64(0x40):
        sig &= ~UInt64(1)
    if sig == 0:
        exp = 0
    return UInt32((s << 31) + (UInt64(exp) << 23) + sig)


@always_inline
def f32_to_f64_bits(b32: UInt64) -> UInt64:
    """The exact widening (checks/soft_f64.mojo `sf64_from_f32`'s statements;
    a NaN becomes the quiet NaN 0x7FF8000000000000)."""
    var s = b32 >> 31
    var e = Int((b32 >> 23) & UInt64(0xFF))
    var f = b32 & UInt64(0x7FFFFF)
    if e == 0xFF:
        if f != 0:
            return UInt64(0x7FF8000000000000)
        return (s << 63) | (UInt64(0x7FF) << 52)
    if e == 0:
        if f == 0:
            return s << 63
        var lead = _top_bit(f)  # 0..22
        var frac = (f << UInt64(52 - lead)) & UInt64(0x000FFFFFFFFFFFFF)
        return (s << 63) | (UInt64(lead - 149 + 1023) << 52) | frac
    return (s << 63) | (UInt64(e + 896) << 52) | (f << 29)


@always_inline
def f16_to_f32_bits(h: UInt64) -> UInt32:
    """The exact float32 of IEEE half `h` (low 16 bits), by bit expansion:
    every half is a float32, so nothing rounds. A NaN becomes the quiet NaN
    0x7FC00000 (as `f64_to_f32_bits`); subnormal halves are normalized."""
    var s = UInt32((h >> 15) & 1) << 31
    var e = Int((h >> 10) & UInt64(0x1F))
    var f = h & UInt64(0x3FF)
    if e == 0x1F:
        if f != 0:
            return UInt32(0x7FC00000)
        return s | UInt32(0x7F800000)
    if e == 0:
        if f == 0:
            return s
        # f * 2^-24 with its top bit at `lead` (0..9): exponent lead - 24
        var lead = _top_bit(f)
        var frac = UInt32((f << UInt64(23 - lead)) & UInt64(0x7FFFFF))
        return s | (UInt32(lead + 103) << 23) | frac
    return s | (UInt32(e + 112) << 23) | (UInt32(f) << 13)


@always_inline
def bf16_to_f32_bits(h: UInt64) -> UInt32:
    """The exact float32 of bfloat16 `h` (low 16 bits): a bfloat16 is the
    top half of a float32, so the widening is a 16-bit shift and nothing
    rounds (infinities, NaN payloads and subnormals keep their bits)."""
    return UInt32(h & UInt64(0xFFFF)) << 16


@always_inline
def cast_f32_bits(raw: UInt64, kind: Int, itemsize: Int) -> UInt32:
    """The float32 (bits) of one item `raw` (`cast_raw_bits`)."""
    if kind == CAST_KIND_BFLOAT:
        return bf16_to_f32_bits(raw)
    if kind == CAST_KIND_FLOAT:
        if itemsize == 4:
            return UInt32(raw)
        if itemsize == 2:
            return f16_to_f32_bits(raw)
        return f64_to_f32_bits(raw)
    if kind == CAST_KIND_BOOL:
        return UInt32(0x3F800000) if raw != 0 else UInt32(0)
    if kind == CAST_KIND_SIGNED:
        var v = _sign_extend(raw, itemsize)
        if (v >> 63) != 0:
            return int_to_f32_bits(True, ~v + 1)
        return int_to_f32_bits(False, v)
    return int_to_f32_bits(False, raw)


@always_inline
def cast_i32_bits(raw: UInt64, kind: Int, itemsize: Int) -> UInt32:
    """The int32 (bits) of one integer or bool item (`cast_raw_bits`),
    saturated: a value below -2^31 becomes -2^31 and one above 2^31 - 1
    becomes 2^31 - 1. Every int32 value is exact. The index consumers (the
    embedding op's id check) refuse both saturated values: no table has 2^31
    rows, so a value outside int32 is refused by position, never wrapped."""
    if kind == CAST_KIND_BOOL:
        return UInt32(1) if raw != 0 else UInt32(0)
    if kind == CAST_KIND_SIGNED:
        var v = _sign_extend(raw, itemsize)
        if (v >> 63) != 0:
            if v < UInt64(0xFFFFFFFF80000000):  # below -2^31
                return UInt32(0x80000000)
            return UInt32(v & UInt64(0xFFFFFFFF))
        if v > UInt64(0x7FFFFFFF):
            return UInt32(0x7FFFFFFF)
        return UInt32(v)
    if raw > UInt64(0x7FFFFFFF):
        return UInt32(0x7FFFFFFF)
    return UInt32(raw)


@always_inline
def cast_w64_bits(raw: UInt64, kind: Int, itemsize: Int) -> UInt64:
    """The 64-bit label word of one item: float64 bits for a float, int64
    bits for an integer or bool (core/label_encode.mojo LABEL_KIND_*)."""
    if kind == CAST_KIND_FLOAT:
        if itemsize == 8:
            return raw
        return f32_to_f64_bits(raw)
    if kind == CAST_KIND_BOOL:
        return UInt64(1) if raw != 0 else UInt64(0)
    if kind == CAST_KIND_SIGNED:
        return _sign_extend(raw, itemsize)
    return raw


@always_inline
def narrow_w64(w: UInt64, kind: Int, itemsize: Int) -> UInt64:
    """The caller's item (its low itemsize bytes) of a label word
    `cast_w64_bits` produced."""
    if kind == CAST_KIND_FLOAT and itemsize == 4:
        return UInt64(f64_to_f32_bits(w))
    if itemsize == 8:
        return w
    return w & ((UInt64(1) << UInt64(itemsize * 8)) - 1)


def label_kind_of(kind: Int, itemsize: Int, what: String) raises -> Int:
    """The label sort's kind (core/label_encode.mojo: 0 float64, 1 int64)."""
    if kind == CAST_KIND_FLOAT:
        if itemsize == 2:
            raise Error(what + ": float16 labels are not supported; pass float32 labels")
        return 0
    if kind == CAST_KIND_UNSIGNED and itemsize == 8:
        raise Error(what + ": uint64 labels are not supported; pass int64 labels")
    return 1
