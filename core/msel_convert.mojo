# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Element loads and widenings shared by the model-selection device kernels
(`core/msel_device.mojo`) and their host column (`core/msel_host.mojo`)
(lane cpu2-l4-modelsel, 2026-10-04).

Integers, shifts and bit moves only: float64 is handled as its 64-bit word
(the Apple GPU has no float64), through `checks/soft_f64.mojo` for the
float32 widening and narrowing and through `msel_i64_to_f64_bits` for an
integer, so the device and the host column write the same bytes on NVIDIA,
AMD, Apple and the CPU. Every function is `@always_inline` (a pointer must
not cross a non-inlined call on Metal).

The dtype codes are those of `bindings/hotpath_helpers.mojo` (HP_F32 ...).
"""

from std.memory import bitcast

from checks.soft_f64 import sf64_from_f32, sf64_to_f32

comptime MSEL_F32 = 0
comptime MSEL_F64 = 1
comptime MSEL_I32 = 2
comptime MSEL_I64 = 3
comptime MSEL_U32 = 4
comptime MSEL_U8 = 5

comptime _FRAC = UInt64(0x000FFFFFFFFFFFFF)


@always_inline
def msel_valid_code(code: Int) -> Bool:
    return code >= MSEL_F32 and code <= MSEL_U8


@always_inline
def msel_itemsize(code: Int) -> Int:
    if code == MSEL_F64 or code == MSEL_I64:
        return 8
    if code == MSEL_U8:
        return 1
    return 4


@always_inline
def msel_is_int(code: Int) -> Bool:
    return code == MSEL_I32 or code == MSEL_I64 or code == MSEL_U32 or code == MSEL_U8


@always_inline
def _clz64(x: UInt64) -> Int:
    """Leading zeros by shifts only (the same instructions on every backend)."""
    if x == 0:
        return 64
    var n = 0
    var v = x
    if (v & UInt64(0xFFFFFFFF00000000)) == 0:
        n += 32
        v <<= 32
    if (v & UInt64(0xFFFF000000000000)) == 0:
        n += 16
        v <<= 16
    if (v & UInt64(0xFF00000000000000)) == 0:
        n += 8
        v <<= 8
    if (v & UInt64(0xF000000000000000)) == 0:
        n += 4
        v <<= 4
    if (v & UInt64(0xC000000000000000)) == 0:
        n += 2
        v <<= 2
    if (v & UInt64(0x8000000000000000)) == 0:
        n += 1
    return n


@always_inline
def msel_i64_to_f64_bits(v: Int64) -> UInt64:
    """The float64 word of an int64, rounded to nearest even (the C cast,
    and numpy's `astype('<f8')`), for every int64 including |v| >= 2^53."""
    if v == 0:
        return UInt64(0)
    var neg = v < 0
    var u = bitcast[DType.uint64](v)
    if neg:
        u = (~u) + UInt64(1)
    var lz = _clz64(u)
    var e = UInt64(1023 + 63 - lz)
    var m: UInt64
    if lz >= 11:
        m = u << UInt64(lz - 11)
    else:
        var drop = 11 - lz
        m = u >> UInt64(drop)
        var rem = u & ((UInt64(1) << UInt64(drop)) - UInt64(1))
        var half = UInt64(1) << UInt64(drop - 1)
        if rem > half or (rem == half and (m & UInt64(1)) == UInt64(1)):
            m += UInt64(1)
            if m == (UInt64(1) << 53):
                m >>= 1
                e += UInt64(1)
    var s = (UInt64(1) << 63) if neg else UInt64(0)
    return s | (e << 52) | (m & _FRAC)


@always_inline
def msel_load_i64(src: MutPointer[UInt8, MutAnyOrigin], code: Int, e: Int) -> Int64:
    """Element `e` of an integer buffer of dtype `code`, widened to int64
    (sign extension for int32, zero extension for uint32 and uint8)."""
    if code == MSEL_I64:
        return src.bitcast[Int64]().unsafe_load(e)
    if code == MSEL_I32:
        return Int64(src.bitcast[Int32]().unsafe_load(e))
    if code == MSEL_U32:
        return Int64(src.bitcast[UInt32]().unsafe_load(e))
    return Int64(src.unsafe_load(e))


@always_inline
def msel_load_f64_bits(src: MutPointer[UInt8, MutAnyOrigin], code: Int, e: Int) -> UInt64:
    """Element `e` of a buffer of dtype `code` as a float64 word: float64 as
    is, float32 widened exactly (`sf64_from_f32`), integers rounded to
    nearest even (`msel_i64_to_f64_bits`)."""
    if code == MSEL_F64:
        return src.bitcast[UInt64]().unsafe_load(e)
    if code == MSEL_F32:
        return sf64_from_f32(src.bitcast[Float32]().unsafe_load(e))
    return msel_i64_to_f64_bits(msel_load_i64(src, code, e))


@always_inline
def msel_load_f32(src: MutPointer[UInt8, MutAnyOrigin], code: Int, e: Int) -> Float32:
    """Element `e` of a float32 or float64 buffer as float32: float32 as is,
    float64 narrowed to nearest even (`sf64_to_f32`)."""
    if code == MSEL_F32:
        return src.bitcast[Float32]().unsafe_load(e)
    return sf64_to_f32(src.bitcast[UInt64]().unsafe_load(e))
