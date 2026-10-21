# SPDX-License-Identifier: Apache-2.0
# Preserve the established classical Array.astype integer -> double -> float
# contract. CastIn's direct integer -> float conversion serves other APIs and
# deliberately remains unchanged. Narrow the binary64 word with integer logic
# so the intervening rounding cannot be collapsed into a direct conversion.
from std.memory import bitcast
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, cast_f32_bits, cast_w64_bits, f64_to_f32_bits


def classical_f32_bits(raw: UInt64, kind: Int, size: Int) -> UInt32:
    if kind == CAST_KIND_SIGNED or kind == CAST_KIND_UNSIGNED:
        var word = cast_w64_bits(raw, kind, size)
        var wide = Float64(bitcast[DType.int64](word)) if kind == CAST_KIND_SIGNED else Float64(word)
        return f64_to_f32_bits(bitcast[DType.uint64](wide))
    return cast_f32_bits(raw, kind, size)
