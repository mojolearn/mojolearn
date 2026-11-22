# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The transfer kit's own check (`checks/native_tests/kit.mojo`).

Device column: every dtype the checks move goes up and comes back bit for
bit through `upload` / `download`, a prefix through `upload_into`, a list
through `download_into`, a caller's pinned buffer and a host address through
`upload_host` / `upload_host_into` / `upload_address` / `download_host_into`, and the undrained set
with the returned pinned buffer held past the caller's drain. The float
fixtures carry NaN payloads, both zeros, both infinities and subnormals, so
a transfer that goes through a float value instead of the bits fails.

Host column: the comparisons. A planted difference (+0.0 against -0.0 at
index 3) must be refused by `same_words` with the index and both words in
hex; `allclose` must refuse a value outside the caller's tolerance and a
NaN; `all_finite` must name the first non-finite index.

Planted run: `-D KIT_ROUNDTRIP_PLANT=1` flips the low bit of word 3 of the
float32 words read back (device column) and of the second list compared
(host column), so case 0 of each column MUST fail with `at 3`, the word it
got and the word it wanted (`pixi run check-kit-planted-gpu` and
`check-kit-planted-cpu`). A check that cannot fail proves nothing.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.testing import TestSuite

from max.gpu.host import DeviceContext

from checks.native_tests.kit import (
    all_finite, allclose, copy_device_into, download, download_buffer_into, download_host_into, download_into,
    download_undrained_keep, download_whole_into, same_words, upload, upload_address, upload_buffer_into,
    upload_host, upload_host_into, upload_into, upload_into_undrained_keep, upload_whole_into, word_hex,
)
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases, check_cases_serial, require


comptime KIT_WORDS = 1031
comptime KIT_GPU_CASES = 12
comptime KIT_CPU_CASES = 6
comptime KIT_PLANT = is_defined["KIT_ROUNDTRIP_PLANT"]()


def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _f32_words() -> List[Float32]:
    """Hashed float32 bit patterns behind eight fixed special words."""
    var special: List[UInt32] = [
        UInt32(0x00000000), UInt32(0x80000000), UInt32(0x7FC00001), UInt32(0xFFC12345),
        UInt32(0x7F800000), UInt32(0xFF800000), UInt32(0x00000001), UInt32(0x807FFFFF),
    ]
    var words = List[Float32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        var bits = UInt32(_mix(UInt64(i)) >> 32)
        if i < 8:
            bits = special[i]
        words[i] = bitcast[DType.float32](bits)
    return words^


def _f64_words() -> List[Float64]:
    var special: List[UInt64] = [
        UInt64(0x0000000000000000), UInt64(0x8000000000000000), UInt64(0x7FF8000000000001),
        UInt64(0xFFF8123456789ABC), UInt64(0x7FF0000000000000), UInt64(0xFFF0000000000000),
        UInt64(0x0000000000000001), UInt64(0x800FFFFFFFFFFFFF),
    ]
    var words = List[Float64](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        var bits = _mix(UInt64(i) + UInt64(7919))
        if i < 8:
            bits = special[i]
        words[i] = bitcast[DType.float64](bits)
    return words^


def _i32_words() -> List[Int32]:
    var words = List[Int32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = bitcast[DType.int32](UInt32(_mix(UInt64(i) + UInt64(101)) >> 32))
    return words^


def _u32_words() -> List[UInt32]:
    var words = List[UInt32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = UInt32(_mix(UInt64(i) + UInt64(211)) >> 32)
    return words^


def _u8_words() -> List[UInt8]:
    var words = List[UInt8](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = UInt8(_mix(UInt64(i) + UInt64(307)) >> 56)
    return words^


def _u16_words() -> List[UInt16]:
    var words = List[UInt16](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = UInt16(_mix(UInt64(i) + UInt64(401)) >> 48)
    return words^


def _i64_words() -> List[Int64]:
    var words = List[Int64](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = bitcast[DType.int64](_mix(UInt64(i) + UInt64(503)))
    return words^


def _u64_words() -> List[UInt64]:
    var words = List[UInt64](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        words[i] = _mix(UInt64(i) + UInt64(601))
    return words^


def _roundtrip[dt: DType](ctx: DeviceContext, values: List[Scalar[dt]]) raises:
    """Whole buffer up and down, a prefix into a wider zero buffer, and a
    download into a caller's list, each bit for bit."""
    var n = len(values)
    var device = upload[dt](ctx, values)
    var back = download[dt](ctx, device, n)
    same_words[dt](back, values, "upload then download")
    var into = List[Scalar[dt]](length=n, fill=Scalar[dt](0))
    download_into[dt](ctx, device, into)
    same_words[dt](into, values, "download_into")
    var short = download[dt](ctx, device, 5)
    var short_want = List[Scalar[dt]](length=5, fill=Scalar[dt](0))
    for i in range(5):
        short_want[i] = values[i]
    same_words[dt](short, short_want, "download of a prefix")
    # a prefix into a wider buffer: the tail keeps what the buffer held
    var padded = List[Scalar[dt]](length=n + 5, fill=Scalar[dt](1))
    var wider = upload[dt](ctx, padded)
    upload_into[dt](ctx, wider, values)
    for i in range(n):
        padded[i] = values[i]
    var whole = download[dt](ctx, wider, n + 5)
    same_words[dt](whole, padded, "upload_into of a prefix")
    # the whole-buffer forms: the list may be longer than the buffer
    var exact = ctx.enqueue_create_buffer[dt](n)
    upload_whole_into[dt](ctx, exact, padded)
    var longer = List[Scalar[dt]](length=n + 5, fill=Scalar[dt](1))
    download_whole_into[dt](ctx, exact, longer)
    same_words[dt](longer, padded, "upload_whole_into then download_whole_into")
    _ = exact^
    _ = device^
    _ = wider^


def _empty(ctx: DeviceContext) raises:
    var none = List[Float32]()
    var device = upload[DType.float32](ctx, none)
    require(len(device) == 1, "an empty upload is a one-element buffer")
    var zero = download[DType.float32](ctx, device, 1)
    require(bitcast[DType.uint32](zero[0]) == UInt32(0), "an empty upload holds one zero")
    require(len(download[DType.float32](ctx, device, 0)) == 0, "a download of nothing is empty")
    _ = device^


def _undrained(ctx: DeviceContext) raises:
    """The undrained set, used as its docstrings require: the returned
    pinned buffers are named after this function's own drain."""
    var values = _f32_words()
    var device = ctx.enqueue_create_buffer[DType.float32](KIT_WORDS)
    var up_keep = upload_into_undrained_keep[DType.float32](ctx, device, values)
    var down_keep = download_undrained_keep[DType.float32](ctx, device)
    ctx.synchronize()
    var got = List[Float32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        got[i] = down_keep.unsafe_ptr().unsafe_load(i)
    same_words[DType.float32](got, values, "undrained upload and download after the caller's drain")
    _ = up_keep^
    _ = down_keep^
    _ = device^


def _host_forms(ctx: DeviceContext) raises:
    """A caller's pinned buffer and a host address as the source."""
    var values = _u32_words()
    var host = ctx.enqueue_create_host_buffer[DType.uint32](KIT_WORDS)
    ctx.synchronize()
    for i in range(KIT_WORDS):
        host.unsafe_ptr().unsafe_store(i, values[i])
    var first = upload_host[DType.uint32](ctx, host, KIT_WORDS)
    same_words[DType.uint32](download[DType.uint32](ctx, first, KIT_WORDS), values, "upload_host")
    var second = ctx.enqueue_create_buffer[DType.uint32](KIT_WORDS)
    upload_host_into[DType.uint32](ctx, second, host)
    same_words[DType.uint32](download[DType.uint32](ctx, second, KIT_WORDS), values, "upload_host_into")
    var back = ctx.enqueue_create_host_buffer[DType.uint32](KIT_WORDS)
    download_host_into[DType.uint32](ctx, second, back)
    var read = List[UInt32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        read[i] = back.unsafe_ptr().unsafe_load(i)
    same_words[DType.uint32](read, values, "download_host_into")
    var third = upload_address[DType.uint32](ctx, Int(values.unsafe_ptr()), KIT_WORDS)
    same_words[DType.uint32](download[DType.uint32](ctx, third, KIT_WORDS), values, "upload_address")
    _ = values
    _ = host^
    _ = back^
    _ = first^
    _ = second^
    _ = third^


def _buffer_forms(ctx: DeviceContext) raises:
    """The buffer to buffer forms: pinned to device, device to device and
    device to pinned, each read back bit for bit."""
    var values = _u32_words()
    var host = ctx.enqueue_create_host_buffer[DType.uint32](KIT_WORDS)
    ctx.synchronize()
    for i in range(KIT_WORDS):
        host.unsafe_ptr().unsafe_store(i, values[i])
    var first = ctx.enqueue_create_buffer[DType.uint32](KIT_WORDS)
    upload_buffer_into[DType.uint32](ctx, first, host)
    same_words[DType.uint32](download[DType.uint32](ctx, first, KIT_WORDS), values, "upload_buffer_into")
    var second = ctx.enqueue_create_buffer[DType.uint32](KIT_WORDS)
    copy_device_into[DType.uint32](ctx, second, first)
    same_words[DType.uint32](download[DType.uint32](ctx, second, KIT_WORDS), values, "copy_device_into")
    var back = ctx.enqueue_create_host_buffer[DType.uint32](KIT_WORDS)
    download_buffer_into[DType.uint32](ctx, second, back)
    var read = List[UInt32](length=KIT_WORDS, fill=0)
    for i in range(KIT_WORDS):
        read[i] = back.unsafe_ptr().unsafe_load(i)
    same_words[DType.uint32](read, values, "download_buffer_into")
    _ = host^
    _ = back^
    _ = first^
    _ = second^


def _planted_device(ctx: DeviceContext) raises:
    """The planted run's device case: the float32 words up and down with the
    low bit of word 3 flipped on the way back; `same_words` must refuse it."""
    var values = _f32_words()
    var device = upload[DType.float32](ctx, values)
    var back = download[DType.float32](ctx, device, KIT_WORDS)
    back[3] = bitcast[DType.float32](bitcast[DType.uint32](back[3]) ^ UInt32(1))
    same_words[DType.float32](back, values, "planted device roundtrip")
    _ = device^


def _gpu_case(ctx: DeviceContext, which: Int) raises:
    if which == 0:
        _roundtrip[DType.float32](ctx, _f32_words())
        comptime if KIT_PLANT:
            _planted_device(ctx)
    elif which == 1:
        _roundtrip[DType.float64](ctx, _f64_words())
    elif which == 2:
        _roundtrip[DType.int32](ctx, _i32_words())
    elif which == 3:
        _roundtrip[DType.uint32](ctx, _u32_words())
    elif which == 4:
        _roundtrip[DType.uint8](ctx, _u8_words())
    elif which == 5:
        _roundtrip[DType.uint16](ctx, _u16_words())
    elif which == 6:
        _roundtrip[DType.int64](ctx, _i64_words())
    elif which == 7:
        _roundtrip[DType.uint64](ctx, _u64_words())
    elif which == 8:
        _empty(ctx)
    elif which == 9:
        _undrained(ctx)
    elif which == 10:
        _host_forms(ctx)
    else:
        _buffer_forms(ctx)


def _refusal(got: List[Float32], want: List[Float32]) -> String:
    """The message `same_words` refuses with, or empty when it accepts."""
    try:
        same_words[DType.float32](got, want, "planted")
    except error:
        return String(error)
    return String("")


def _cpu_case(which: Int) raises:
    if which == 0:
        # the same NaN bits are the same words
        var a = _f32_words()
        var b = _f32_words()
        comptime if KIT_PLANT:
            b[3] = bitcast[DType.float32](bitcast[DType.uint32](b[3]) ^ UInt32(1))
        same_words[DType.float32](a, b, "equal words")
        require(_refusal(a, b) == "", "equal words are accepted")
    elif which == 1:
        # the planted difference: +0.0 against -0.0 at index 3
        var got: List[Float32] = [1.0, 2.0, 3.0, 0.0, 5.0]
        var want: List[Float32] = [1.0, 2.0, 3.0, 0.0, 5.0]
        want[3] = bitcast[DType.float32](UInt32(0x80000000))
        var message = _refusal(got, want)
        require("at 3" in message, "the refusal names the index: " + message)
        require("got 0x00000000" in message, "the refusal prints the word it got: " + message)
        require("want 0x80000000" in message, "the refusal prints the word it wanted: " + message)
    elif which == 2:
        var got: List[Float32] = [1.0, 2.0]
        var want: List[Float32] = [1.0, 2.0, 3.0]
        require("extent 2 vs 3" in _refusal(got, want), "a length difference is refused by extent")
    elif which == 3:
        var actual: List[Float32] = [1.0, 2.0, 4.0]
        var near: List[Float64] = [1.0, 2.0000001, 4.0]
        allclose[DType.float32](actual, near, 1e-6, 0.0, "within tolerance")
        var far: List[Float64] = [1.0, 2.01, 4.0]
        var refused = False
        try:
            allclose[DType.float32](actual, far, 1e-6, 0.0, "outside tolerance")
        except error:
            refused = "at 1" in String(error)
        require(refused, "allclose refuses a value outside the tolerance, by index")
        actual[2] = bitcast[DType.float32](UInt32(0x7FC00000))
        refused = False
        try:
            allclose[DType.float32](actual, near, 1e-6, 1e30, "a NaN")
        except error:
            refused = "at 2" in String(error)
        require(refused, "allclose refuses a NaN at any tolerance")
    elif which == 4:
        var values: List[Float64] = [0.0, -1.5, 3.0]
        all_finite[DType.float64](values, "finite")
        values[2] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
        var refused = False
        try:
            all_finite[DType.float64](values, "an infinity")
        except error:
            refused = "non-finite at 2" in String(error)
        require(refused, "all_finite names the first non-finite index")
    else:
        var halves: List[UInt16] = [UInt16(0x0001), UInt16(0xBEEF)]
        require(word_hex[DType.uint16](halves, 1) == "0xbeef", "word_hex of a 16-bit word")
        var longs = List[UInt64](length=1, fill=UInt64(0x0123456789ABCDEF))
        require(word_hex[DType.uint64](longs, 0) == "0x0123456789abcdef", "word_hex of a 64-bit word")


def test_kit_roundtrip() raises:
    """The column's cases: the device transfers in order on the one device queue, the host comparisons across the
    cores."""
    comptime if CPU_COLUMN_BUILD:
        comptime if KIT_PLANT:
            print("PLANT    KIT_ROUNDTRIP_PLANT is set: host case 0 MUST fail with at 3, got and want")
        check_cases("kit_roundtrip host", KIT_CPU_CASES, _cpu_case)
    else:
        var ctx = DeviceContext()
        comptime if KIT_PLANT:
            print("PLANT    KIT_ROUNDTRIP_PLANT is set: device case 0 MUST fail with at 3, got and want")
        def device_case(which: Int) raises {imm ctx}:
            _gpu_case(ctx, which)
        check_cases_serial("kit_roundtrip device", KIT_GPU_CASES, device_case)


def main() raises:
    # Tests listed by name, not discovered: discovery elaborates every function of the module for the host column,
    # and the device helpers here are the device column's (docs/plans/TESTSUITE_CONVERSION.md, the two-column rule).
    var suite = TestSuite()
    suite.test[test_kit_roundtrip]()
    suite^.run()
