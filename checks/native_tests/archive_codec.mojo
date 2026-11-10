# SPDX-License-Identifier: Apache-2.0
"""Independent compressed fixtures and checked spans, one owner per pool task.

This exercises public system codec libraries, including optional Zstandard;
the orchestrator must provision/qualify those dependencies before running it.
"""
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_codec import archive_inflate
from checks.native_tests.archive_codec_vectors import archive_codec_fixture


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _codec(unit: Int) raises:
    var methods: List[Int] = [8, 12, 14, 93]
    var method = methods[unit // 7]
    var variant = unit % 7
    var vector = variant if variant < 3 else 0
    var source = archive_codec_fixture(method, vector)
    var expected = 9 if vector == 0 else (0 if vector == 1 else 65536)
    if variant == 3:
        expected -= 1
    elif variant == 4:
        expected += 1
    elif variant == 5:
        source.clear()
    elif variant == 6:
        # Incremental single-member decoding historically ignores trailing
        # bytes after the first completed stream. Keep that behavior.
        source.append(0xFF)
        source.append(0xFF)
    var before = source.copy()
    var refused = False
    var accepted = False
    try:
        var decoded = archive_inflate(ArchiveSpan(Int(source.unsafe_ptr()), len(source)), method, expected)
        accepted = True
        if variant < 3 or variant == 6:
            _require(len(decoded) == expected, "decoded owned size")
            for i in range(expected):
                var value = i + 49 if vector == 0 else 0
                _require(Int(decoded[i]) == value, "independent plaintext differs")
            # Output is independent of compressed input lifetime/mutation.
            source.clear()
            if expected:
                _require(Int(decoded[expected - 1]) == (57 if vector == 0 else 0), "decoder owns output")
    except problem:
        if variant < 3 or variant == 6:
            raise Error(String(problem))
        refused = True
    if variant >= 3 and variant < 6:
        _require(refused and not accepted, "invalid stream/length refusal")
        _require(ArchiveSpan(Int(source.unsafe_ptr()), len(source)).equal(
            ArchiveSpan(Int(before.unsafe_ptr()), len(before))), "failure modified compressed input")
    _ = source
    _ = before


def _span(unit: Int) raises:
    var source: List[UInt8] = [99, 1, 2, 3, 4, 5, 6, 7, 8, 99]
    var span = ArchiveSpan(Int(source.unsafe_ptr()) + 1, 8)
    if unit == 0:
        _require(span.uint(0, 8) == 0x0807060504030201, "independent little-endian value")
        _require(span.slice(2, 3).uint(0, 3) == 0x050403, "bounded interior span")
        _require(span.slice(8, 0).size == 0, "empty end span")
    else:
        var refused = False
        try:
            if unit == 1:
                _ = span.uint(1, 8)
            elif unit == 2:
                _ = span.uint(0, 9)
            elif unit == 3:
                _ = span.slice(-1, 2)
            elif unit == 4:
                _ = span.slice(7, 9223372036854775807)
            elif unit == 5:
                _ = span.byte(8)
            else:
                source[8] = 0x80
                _ = span.uint(0, 8)
        except:
            refused = True
        _require(refused, "span rejects overflow/out-of-bounds before access")
    _require(source[0] == 99 and source[9] == 99, "span canaries")
    _ = source


def main() raises:
    comptime FIXTURES = 35
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 28:
                _codec(unit)
            else:
                _span(unit - 28)
        except problem:
            print("FAIL archive_codec fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "archive codecs failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_codec:", FIXTURES, "native fixtures, parallel")
