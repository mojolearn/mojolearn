# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Independent filesystem byte and POSIX spelling fixtures, task-local storage."""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require as _require
from core.native_archive_span import ArchiveSpan
from core.native_archive_path import archive_filesystem_decode, archive_path_join, archive_path_normalize


def _equal(actual: List[UInt32], expected: List[UInt32]) raises:
    _require(len(actual) == len(expected), "filesystem scalar count")
    for i in range(len(expected)):
        _require(actual[i] == expected[i], "filesystem scalar " + String(i))


def _words(value: String) -> List[UInt32]:
    var result = List[UInt32]()
    for point in value.codepoints():
        result.append(UInt32(Int(point)))
    return result^


def _decode(unit: Int) raises:
    var raw = List[UInt8]()
    var expected = List[UInt32]()
    if unit == 1:
        raw = [97, 0, 47, 122]
        expected = [97, 0, 47, 122]
    elif unit == 2:
        raw = [0xC2, 0xA2, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98, 0x80]
        expected = [0xA2, 0x20AC, 0x1F600]
    elif unit == 3:
        raw = [0xC2, 0x80, 0xDF, 0xBF, 0xE0, 0xA0, 0x80, 0xEF, 0xBF, 0xBF]
        expected = [0x80, 0x7FF, 0x800, 0xFFFF]
    elif unit == 4:
        raw = [0xF0, 0x90, 0x80, 0x80, 0xF4, 0x8F, 0xBF, 0xBF]
        expected = [0x10000, 0x10FFFF]
    elif unit == 5:
        raw = [0x80, 0xBF, 0xFF]
        expected = [0xDC80, 0xDCBF, 0xDCFF]
    elif unit == 6:
        raw = [0xC0, 0xAF, 0xC1, 0xBF]
        expected = [0xDCC0, 0xDCAF, 0xDCC1, 0xDCBF]
    elif unit == 7:
        raw = [0xC2]
        expected = [0xDCC2]
    elif unit == 8:
        raw = [0xE2, 0x82]
        expected = [0xDCE2, 0xDC82]
    elif unit == 9:
        raw = [0xF0, 0x9F, 0x98]
        expected = [0xDCF0, 0xDC9F, 0xDC98]
    elif unit == 10:
        raw = [0xE2, 0x82, 47]
        expected = [0xDCE2, 0xDC82, 47]
    elif unit == 11:
        raw = [0xE2, 0xC2, 0xA2]
        expected = [0xDCE2, 0xA2]
    elif unit == 12:
        raw = [0xE0, 0x9F, 0xBF]
        expected = [0xDCE0, 0xDC9F, 0xDCBF]
    elif unit == 13:
        raw = [0xED, 0xA0, 0x80]
        expected = [0xDCED, 0xDCA0, 0xDC80]
    elif unit == 14:
        raw = [0xF0, 0x8F, 0xBF, 0xBF]
        expected = [0xDCF0, 0xDC8F, 0xDCBF, 0xDCBF]
    elif unit == 15:
        raw = [0xF4, 0x90, 0x80, 0x80]
        expected = [0xDCF4, 0xDC90, 0xDC80, 0xDC80]
    elif unit == 16:
        raw = [0xF5, 0x80, 0x80, 0x80]
        expected = [0xDCF5, 0xDC80, 0xDC80, 0xDC80]
    elif unit == 17:
        raw = [0xED, 0x9F, 0xBF, 0xEE, 0x80, 0x80]
        expected = [0xD7FF, 0xE000]
    var actual = archive_filesystem_decode(ArchiveSpan(Int(raw.unsafe_ptr()), len(raw)))
    _equal(actual, expected)
    _ = raw


def _join(unit: Int) raises:
    var parts = List[List[UInt32]]()
    var expected = String(".")
    if unit == 1:
        parts.append(_words(""))
    elif unit == 2:
        parts.append(_words("alpha"))
        parts.append(_words("beta"))
        expected = "alpha/beta"
    elif unit == 3:
        parts.append(_words("alpha"))
        parts.append(_words("/beta"))
        parts.append(_words("gamma"))
        expected = "/beta/gamma"
    elif unit == 4:
        parts.append(_words("/"))
        parts.append(_words("alpha"))
        expected = "/alpha"
    elif unit == 5:
        parts.append(_words("//"))
        parts.append(_words("alpha"))
        expected = "//alpha"
    elif unit == 6:
        parts.append(_words("///"))
        parts.append(_words("alpha"))
        expected = "/alpha"
    elif unit == 7:
        parts.append(_words("alpha///"))
        parts.append(_words("beta"))
        expected = "alpha/beta"
    elif unit == 8:
        parts.append(_words("alpha"))
        parts.append(_words(""))
        expected = "alpha"
    elif unit == 9:
        parts.append(_words("./alpha"))
        parts.append(_words("../beta"))
        expected = "./alpha/../beta"
    elif unit == 10:
        parts.append(_words(""))
        parts.append(_words("//"))
        expected = "//"
    elif unit == 11:
        parts.append(_words("/alpha"))
        parts.append(_words("//beta"))
        expected = "//beta"
    _equal(archive_path_join(parts), _words(expected))


def _normalize(unit: Int) raises:
    var raw = String("")
    var expected = String(".")
    if unit == 1:
        raw = "././"
    elif unit == 2:
        raw = "alpha///beta/./"
        expected = "alpha/beta"
    elif unit == 3:
        raw = "./alpha/../beta"
        expected = "alpha/../beta"
    elif unit == 4:
        raw = "/./alpha//beta/"
        expected = "/alpha/beta"
    elif unit == 5:
        raw = "//./alpha//beta/"
        expected = "//alpha/beta"
    elif unit == 6:
        raw = "///./alpha//beta/"
        expected = "/alpha/beta"
    elif unit == 7:
        raw = "//"
        expected = "//"
    elif unit == 8:
        raw = "////"
        expected = "/"
    elif unit == 9:
        raw = ".././.."
        expected = "../.."
    elif unit == 10:
        raw = ".../.hidden/é"
        expected = ".../.hidden/é"
    elif unit == 11:
        var points: List[UInt32] = [47, 46, 47, 0xDCFF, 47, 0, 47]
        var wanted: List[UInt32] = [47, 0xDCFF, 47, 0]
        _equal(archive_path_normalize(points), wanted)
        return
    _equal(archive_path_normalize(_words(raw)), _words(expected))


def test_decode() raises:
    check_cases("decode", 18, _decode)


def test_join() raises:
    check_cases("join", 12, _join)


def test_normalize() raises:
    check_cases("normalize", 12, _normalize)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
