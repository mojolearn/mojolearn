# SPDX-License-Identifier: Apache-2.0
"""Every float16 bit pattern widened to float32, ported from
`test_models_loader.py::test_f16_widening_is_bit_exact_over_every_half`
(lane tmig-lm-1, 2026-11-14).

`mojolearn.models.safetensors.widen_f16` hands its bits to the linalg
binding's `from_f16`: `decomposition/linalg_host_impl.mojo::from_f16_host_run`
on a CPU-only install, `decomposition/linalg_door_impl.mojo::from_f16_device`
on a GPU one. Both suites widen all 65,536 halves and compare each result
with an independent oracle that builds the half's VALUE (sign, exponent,
mantissa, in float64, exact) and converts it to float32, as Python's
`struct` oracle did: finite values and infinities by their float32 bits, a
NaN by its sign and NaN-ness. The halves split into independent chunks.
"""
from std.memory import bitcast

comptime F16_HALVES = 65536
comptime F16_CHUNKS = 64
comptime F16_CHUNK = F16_HALVES // F16_CHUNKS


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def all_halves() -> List[UInt16]:
    var out = List[UInt16](capacity=F16_HALVES)
    for h in range(F16_HALVES):  # driver-loop: over the fixture's 65,536 bit patterns
        out.append(UInt16(h))
    return out^


def half_matches(h: Int, got: Float32) -> Bool:
    """The independent value oracle for half `h`."""
    var g = bitcast[DType.uint32](got)
    var sign = UInt32(h >> 15) << 31
    var exponent = (h >> 10) & 31
    var mantissa = h & 1023
    if exponent == 31:
        if mantissa != 0:
            return ((g & UInt32(0x7F800000)) == UInt32(0x7F800000) and (g & UInt32(0x007FFFFF)) != 0
                    and (g & UInt32(0x80000000)) == sign)
        return g == (sign | UInt32(0x7F800000))
    var magnitude: Float64
    if exponent == 0:
        magnitude = Float64(mantissa) / Float64(16777216)  # mantissa * 2^-24
    else:
        var scale = exponent - 25  # (1024 + mantissa) * 2^(exponent - 25)
        if scale >= 0:
            magnitude = Float64(1024 + mantissa) * Float64(1 << scale)
        else:
            magnitude = Float64(1024 + mantissa) / Float64(1 << (0 - scale))
    return g == (bitcast[DType.uint32](Float32(magnitude)) | sign)


def check_chunk(widened: MutPointer[Float32, MutUntrackedOrigin], chunk: Int) -> Int:
    """The first failing half of `chunk`, or -1."""
    var first = chunk * F16_CHUNK
    for i in range(first, first + F16_CHUNK):  # driver-loop: over one chunk's bit patterns
        if not half_matches(i, widened[i]):
            return i
    return -1
