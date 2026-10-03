# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IEEE float16 to float32 by bit construction, the one spelling the linalg
binding's `from_f16` device kernel (`gemm/checks/gemm_lowbit.mojo`) and its
host twin (`gemm/host/gemm_lowbit_oracle.mojo`) share. Kept out of
`gemm/contract.mojo` so only the bindings that widen float16 import it."""

from std.memory import bitcast



def f16_bits_to_f32_bits(h_in: UInt16) -> UInt32:
    """IEEE binary16 bits widened to binary32 bits, by bit construction
    (pyglue-text-io, Oct 3: the safetensors float16 reader's widening,
    formerly a per-element Python loop in `models/safetensors.py`). Exact:
    every float16 is a float32. A normal value keeps its mantissa shifted up
    by 13 and rebiases the exponent by 112; a subnormal (exponent 0,
    mantissa nonzero) is renormalized into a NORMAL float32; zero keeps its
    sign; the infinities and every NaN keep sign and payload (no quieting,
    which a hardware cast would do). Integer operations only, so every
    column produces the same bits."""
    var h = UInt32(h_in)
    var s = (h & 0x8000) << 16
    var e = (h >> 10) & 0x1F
    var m = h & 0x03FF
    if e == 0:
        if m == 0:
            return s
        var shift: UInt32 = 0
        while (m & 0x0400) == 0:
            m <<= 1
            shift += 1
        m &= 0x03FF
        return s | ((113 - shift) << 23) | (m << 13)
    if e == 31:
        return s | 0x7F800000 | (m << 13)
    return s | ((e + 112) << 23) | (m << 13)


def f16_bits_to_f32(h: UInt16) -> Float32:
    """`f16_bits_to_f32_bits` as the float32 it spells."""
    return bitcast[DType.float32](f16_bits_to_f32_bits(h))
