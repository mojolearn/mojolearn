# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
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
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.memory import bitcast
from core.hostptr import f32_ptr, u16_ptr
from decomposition.linalg_host_impl import from_f16_host_run
from max.gpu.host import DeviceContext
from decomposition.linalg_door_impl import from_f16_device


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime F16_WORDS = 65536
comptime F16_CHUNK = 1024
comptime F16_UNITS = F16_WORDS // F16_CHUNK


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


# ---- The host column (was core_safetensors_f16_cpu_suite.mojo) ----
# The host column's float16 widening over every half pattern, 64 chunks of
# 1024 words run across host cores, each with its own buffers.
def check_unit(unit: Int) raises:
    var start = unit * F16_CHUNK
    var source = List[UInt16](length=F16_CHUNK, fill=0)
    var widened = List[Float32](length=F16_CHUNK, fill=0)
    for j in range(F16_CHUNK):
        source[j] = UInt16(start + j)
    from_f16_host_run(f32_ptr(Int(widened.unsafe_ptr())), Int(source.unsafe_ptr()), F16_CHUNK)
    var words = List[UInt32](length=F16_CHUNK, fill=0)
    for j in range(F16_CHUNK):
        words[j] = bitcast[DType.uint32](widened[j])
    check_words(start, F16_CHUNK, source, words, "host from_f16")
    _ = source
    _ = widened


def test_widening() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("widening", F16_UNITS, check_unit)
    else:
        # The device float16 widening over every half pattern in one launch (one thread per word); the host checks
        # the words across cores.
        var ctx = DeviceContext()
        var source = List[UInt16](length=F16_WORDS, fill=0)
        var widened = List[Float32](length=F16_WORDS, fill=0)
        for j in range(F16_WORDS):
            source[j] = UInt16(j)
        from_f16_device(ctx, f32_ptr(Int(widened.unsafe_ptr())), u16_ptr(Int(source.unsafe_ptr())), F16_WORDS)
        var sp = source.unsafe_ptr()
        var wp = widened.unsafe_ptr()
        def chunk(unit: Int) raises {imm sp, imm wp}:
            var start = unit * F16_CHUNK
            var src = List[UInt16](length=F16_CHUNK, fill=0)
            var words = List[UInt32](length=F16_CHUNK, fill=0)
            for j in range(F16_CHUNK):
                src[j] = sp[start + j]
                words[j] = bitcast[DType.uint32](wp[start + j])
            check_words(start, F16_CHUNK, src, words, "device from_f16")
        check_cases("device widening", F16_UNITS, chunk)
        _ = source
        _ = widened


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
