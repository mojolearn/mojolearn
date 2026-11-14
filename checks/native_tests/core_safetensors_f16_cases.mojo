# SPDX-License-Identifier: Apache-2.0
"""Every IEEE half pattern widened to float32, moved from
test_safetensors_native_ndarray.py::test_every_half_pattern_preserves_sign_and_nan_payload
(2026-11-14).

The safetensors reader's `widen_f16` is one native call of the linalg
`from_f16` entry (decomposition/linalg_host_impl.mojo from_f16_host_run on the
host column, decomposition/linalg_door_impl.mojo from_f16_device on a GPU).
All 65536 words, uint16 and float16 storage alike (the same bits), must give
the former test's integer oracle word for word: sign kept, exponent rebased,
subnormals normalized, infinities kept and every NaN payload kept (never
quieted). The source words must come back unchanged. The ndarray strides,
native Array descriptor and refusal arms stay in the Python test.
"""

comptime F16_WORDS = 65536
comptime F16_CHUNK = 1024
comptime F16_UNITS = F16_WORDS // F16_CHUNK


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def expected_word(half: Int) -> UInt32:
    """The former test's `expected_word`, written on integers."""
    var sign = UInt32((half & 0x8000) << 16)
    var exponent = (half >> 10) & 31
    var fraction = half & 1023
    if exponent == 31:
        return sign | UInt32(0x7F800000) | UInt32(fraction << 13)
    if exponent != 0:
        return sign | UInt32((exponent + 112) << 23) | UInt32(fraction << 13)
    if fraction == 0:
        return sign
    exponent = 113
    while (fraction & 1024) == 0:
        fraction <<= 1
        exponent -= 1
    return sign | UInt32(exponent << 23) | UInt32((fraction & 1023) << 13)


def check_words(start: Int, count: Int, source: List[UInt16], widened: List[UInt32], what: String) raises:
    for j in range(count):
        var word = start + j
        require(Int(source[j]) == word, what + ": the source word " + String(word) + " changed")
        require(widened[j] == expected_word(word),
                what + ": half " + String(word) + " widened to " + String(widened[j])
                + ", expected " + String(expected_word(word)))
