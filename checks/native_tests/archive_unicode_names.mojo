# SPDX-License-Identifier: Apache-2.0
"""Independent UCD/version-boundary fixtures; no Python decoder or oracle.

Expected scalars are published Unicode code points. Version cutoffs come from
the pinned official UCD files and tagged CPython sources in the provenance doc.
Independent fixtures run in the native host pool; this is not model identity.
"""
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_unicode_names import archive_unicode_name, archive_unicode_digit


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _name(word: String, minor: Int, expected: Int, micro: Int = 0) raises:
    var bytes = word.as_bytes()
    var found = archive_unicode_name(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes)), minor, micro)
    _require(found == expected, "Unicode name/profile: " + word + " Python 3." + String(minor) + "." + String(micro))
    _ = word
    _ = bytes


def _version(unit: Int) raises:
    var minor = 10 + unit // 8
    var fixture = unit % 8
    if fixture == 0:
        _name("latin small letter a", minor, 0x61)
        _name("LATIN CAPITAL LETTER A", minor, 0x41)
    elif fixture == 1:
        _name("lf", minor, 0x0A)
        _name("BYTE ORDER MARK", minor, 0xFEFF)
    elif fixture == 2:
        _name("KEYCAP NUMBER SIGN", minor, -1)
        _name("NOT AN ASSIGNED CHARACTER NAME", minor, -1)
    elif fixture == 3:
        _name("TANGSA DIGIT ZERO", minor, 0x16AC0 if minor >= 11 else -1)
        _require(archive_unicode_digit(UInt32(0x16AC0), minor) == (0 if minor >= 11 else -1), "Tangsa digit version")
    elif fixture == 4:
        _name("KAWI DIGIT ZERO", minor, 0x11F50 if minor >= 12 else -1)
        _require(archive_unicode_digit(UInt32(0x11F50), minor) == (0 if minor >= 12 else -1), "Kawi digit version")
    elif fixture == 5:
        _name("GARAY DIGIT ZERO", minor, 0x10D40 if minor >= 14 else -1)
        _require(archive_unicode_digit(UInt32(0x10D40), minor) == (0 if minor >= 14 else -1), "Garay digit version")
    elif fixture == 6:
        _require(archive_unicode_digit(UInt32(0x00B2), minor) == -2, "superscript digit is not decimal")
        _require(archive_unicode_digit(UInt32(0x0663), minor) == 3, "Arabic-Indic decimal")
        _require(archive_unicode_digit(UInt32(0x41), minor) == -1, "letter is not digit")
        _require(archive_unicode_digit(UInt32(0xD800), minor) == -1, "surrogate is not digit")
    else:
        _name("HANGUL SYLLABLE GA", minor, 0xAC00)
        _name("HANGUL SYLLABLE HIH", minor, 0xD7A3)
        _name("CJK UNIFIED IDEOGRAPH-31350", minor, 0x31350 if minor >= 12 else -1)
        _name("CJK UNIFIED IDEOGRAPH-2EBF0", minor, 0x2EBF0 if minor >= 13 else -1)


def _profile(unit: Int) raises:
    var minor = 13 if unit < 2 else 14
    var modern = unit % 2 == 1
    var micro = (13 if modern else 12) if minor == 13 else (4 if modern else 3)
    _name("cjK Unified Ideograph-3aBc", minor, 0x3ABC if modern else -1, micro)
    _name("haNGul SYllABle waE", minor, 0xC65C if modern else -1, micro)
    _name("CJK UNIFIED IDEOGRAPH-04E00", minor, -1 if modern else 0x4E00, micro)
    _name("TANGUT IDEOGRAPH-17000", minor, 0x17000 if modern else -1, micro)
    _name("TANGUT IDEOGRAPH-18D08", minor, 0x18D08 if modern else -1, micro)
    _name("TANGUT IDEOGRAPH-18D09", minor, -1, micro)
    _name("TANGUT IDEOGRAPH-017000", minor, -1, micro)
    _name("nushu character-1b170", minor, 0x1B170, micro)


def _refusal(unit: Int) raises:
    if unit == 0:
        _name("HANGUL SYLLABLE ", 14, -1, 4)
        _name("CJK UNIFIED IDEOGRAPH-FFFF", 14, -1, 4)
        _name("CJK UNIFIED IDEOGRAPH-4E00 ", 14, -1, 4)
        return
    var failed = False
    try:
        if unit == 1:
            _ = archive_unicode_digit(UInt32(0x30), 9)
        elif unit == 2:
            _ = archive_unicode_digit(UInt32(0x30), 15)
        else:
            _name("LATIN CAPITAL LETTER A", 14, 0x41, -1)
    except problem:
        failed = String(problem).find("archive Unicode:") >= 0
    _require(failed, "unsupported Unicode interpreter profile must refuse")


def main() raises:
    comptime FIXTURES = 48
    var failed = List[Int](length=FIXTURES, fill=0)
    var flags = failed.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            if unit < 40:
                _version(unit)
            elif unit < 44:
                _profile(unit - 40)
            else:
                _refusal(unit - 44)
        except problem:
            print("FAIL archive_unicode_names fixture", unit, String(problem))
            flags.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for flag in failed:
        failures += flag
    _require(failures == 0, "archive Unicode failed " + String(failures) + " fixtures")
    print("PASS archive_unicode_names:", FIXTURES, "native fixtures, parallel")
