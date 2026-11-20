# SPDX-License-Identifier: Apache-2.0
"""Versioned native Unicode scalar names/digits for archive literal decoding.

UCD records are generated from pinned official data. No CPython internals,
Python decoder, runtime module import or external data loader is involved.
The compact records share one name/code point with a version-presence mask.
"""
from core.archive_unicode_names_generated import (
    ARCHIVE_UNICODE_TABLE_SHA256, ARCHIVE_UNICODE_NAME_COUNT,
    ARCHIVE_UNICODE_NAME_OFFSETS, ARCHIVE_UNICODE_NAMES, ARCHIVE_UNICODE_DIGITS,
    ARCHIVE_UNICODE_CJK_RANGES, ARCHIVE_UNICODE_TANGUT_RANGES,
)
from core.native_archive_span import ArchiveSpan
from core.shared_slot import SharedSlot

comptime ARCHIVE_UNICODE_TABLE_SHA256_PINNED = "008b4b54a8a01ae692b2a19f1d16995dcca1c2d112cff162eb74096805ee1179"


struct ArchiveUnicodeData(Defaultable, Movable):
    var names: String
    var offsets: String
    var digits: String
    var cjk: String
    var tangut: String
    var leading: List[String]
    var vowels: List[String]
    var trailing: List[String]

    def __init__(out self):
        self.names = String(ARCHIVE_UNICODE_NAMES)
        self.offsets = String(ARCHIVE_UNICODE_NAME_OFFSETS)
        self.digits = String(ARCHIVE_UNICODE_DIGITS)
        self.cjk = String(ARCHIVE_UNICODE_CJK_RANGES)
        self.tangut = String(ARCHIVE_UNICODE_TANGUT_RANGES)
        # Unicode's fixed Hangul composition vocabulary, in L/V/T index order.
        self.leading = ["G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "C", "K", "T", "P", "H"]
        self.vowels = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "YI", "I"]
        self.trailing = ["", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH", "M", "B", "BS", "S", "SS", "NG", "J", "C", "K", "T", "P", "H"]


comptime ARCHIVE_UNICODE_DATA = SharedSlot[ArchiveUnicodeData, "MojolearnArchiveUnicodeDataV1"]


def _archive_unicode_mask(python_minor: Int) raises -> Int:
    if python_minor < 10 or python_minor > 14:
        raise Error("archive Unicode: supported CPython versions are 3.10 through 3.14")
    if String(ARCHIVE_UNICODE_TABLE_SHA256) != String(ARCHIVE_UNICODE_TABLE_SHA256_PINNED):
        raise Error("archive Unicode: regenerate the pinned native Unicode table before compilation")
    return 1 << (python_minor - 10)


@always_inline
def _archive_upper(value: Int) -> Int:
    return value - 32 if 97 <= value <= 122 else value


def _archive_table_hex(text: String, start: Int, count: Int) -> Int:
    var bytes = text.as_bytes()
    var result = 0
    for i in range(count):  # small-loop(fixed-width generated hexadecimal record fields)
        var value = Int(bytes[start + i])
        result = result * 16 + (value - 48 if value <= 57 else value - 55)
    return result


def _archive_name_compare(raw: ArchiveSpan, text: String, start: Int, count: Int) raises -> Int:
    var bytes = text.as_bytes()
    for i in range(min(raw.size, count)):  # driver-loop: native comparison of one external character name
        var left = _archive_upper(raw.byte(i))
        var right = Int(bytes[start + i])
        if left != right:
            return -1 if left < right else 1
    return -1 if raw.size < count else (1 if raw.size > count else 0)


def _archive_name_prefix(raw: ArchiveSpan, at: Int, prefix: String, fold: Bool) raises -> Bool:
    var bytes = prefix.as_bytes()
    if at < 0 or at > raw.size or len(bytes) > raw.size - at:
        return False
    for i in range(len(bytes)):  # driver-loop: compare fixed Unicode name prefix or Hangul component
        var value = raw.byte(at + i)
        if (_archive_upper(value) if fold else value) != Int(bytes[i]):
            return False
    return True


def _archive_syllable(raw: ArchiveSpan, at: Int, names: List[String], fold: Bool) raises -> Tuple[Int, Int]:
    var selected = -1
    var width = -1
    for i in range(len(names)):  # driver-loop: choose longest match among fixed Hangul L/V/T components
        var length = names[i].byte_length()
        if length > width and _archive_name_prefix(raw, at, names[i], fold):
            selected = i
            width = length
    return (selected, max(width, 0))


def _archive_derived_hex(raw: ArchiveSpan, start: Int, modern: Bool) raises -> Int:
    var width = raw.size - start
    if width < 4 or width > (6 if modern else 5):
        return -1
    if modern and raw.byte(start) == 48:
        return -1
    var result = 0
    for i in range(start, raw.size):  # small-loop(algorithmic scalar name has four to six hexadecimal digits)
        var value = raw.byte(i)
        if modern:
            value = _archive_upper(value)
        if 48 <= value <= 57:
            result = result * 16 + value - 48
        elif 65 <= value <= 70:
            result = result * 16 + value - 55
        else:
            return -1
    return result if result <= 0x10FFFF else -1


def _archive_derived_range(point: Int, ranges: String, mask: Int) -> Bool:
    for record in range(ranges.byte_length() // 14):  # driver-loop: over small versioned ideograph range metadata
        var at = record * 14
        if ((_archive_table_hex(ranges, at + 12, 2) & mask) != 0 and
                _archive_table_hex(ranges, at, 6) <= point <= _archive_table_hex(ranges, at + 6, 6)):
            return True
    return False


def archive_unicode_name(raw: ArchiveSpan, python_minor: Int, python_micro: Int = 0) raises -> Int:
    """A scalar code point, or -1 for unknown names and named sequences.

    CPython 3.13.13/3.14.4 changed algorithmic lookup: case folding, canonical
    unpadded hex and Tangut lookup. Earlier versions retain exact Hangul/CJK
    spelling and the old four/five-digit CJK rule. Ordinary UCD names/aliases
    use ASCII case folding in every supported version. See the provenance doc.
    """
    var mask = _archive_unicode_mask(python_minor)
    if python_micro < 0:
        raise Error("archive Unicode: invalid CPython patch version")
    var modern = (python_minor == 13 and python_micro >= 13) or (python_minor == 14 and python_micro >= 4)
    var data = ARCHIVE_UNICODE_DATA.get_or_create_ptr()
    if _archive_name_prefix(raw, 0, "HANGUL SYLLABLE ", modern):
        var leading = _archive_syllable(raw, 16, data[].leading, modern)
        var vowel = _archive_syllable(raw, 16 + leading[1], data[].vowels, modern)
        var trailing = _archive_syllable(raw, 16 + leading[1] + vowel[1], data[].trailing, modern)
        if leading[0] < 0 or vowel[0] < 0 or trailing[0] < 0 or 16 + leading[1] + vowel[1] + trailing[1] != raw.size:
            return -1
        return 0xAC00 + (leading[0] * 21 + vowel[0]) * 28 + trailing[0]
    if _archive_name_prefix(raw, 0, "CJK UNIFIED IDEOGRAPH-", modern):
        var point = _archive_derived_hex(raw, 22, modern)
        return point if _archive_derived_range(point, data[].cjk, mask) else -1
    if modern and _archive_name_prefix(raw, 0, "TANGUT IDEOGRAPH-", True):
        var point = _archive_derived_hex(raw, 17, True)
        return point if _archive_derived_range(point, data[].tangut, mask) else -1
    var lo = 0
    var hi = Int(ARCHIVE_UNICODE_NAME_COUNT)
    while lo < hi:  # driver-loop: binary-search the immutable generated name index
        var mid = (lo + hi) // 2
        var start = _archive_table_hex(data[].offsets, mid * 8, 8)
        var end = _archive_table_hex(data[].offsets, (mid + 1) * 8, 8)
        var comparison = _archive_name_compare(raw, data[].names, start, end - start - 11)
        if comparison < 0:
            hi = mid
        elif comparison > 0:
            lo = mid + 1
        else:
            if (_archive_table_hex(data[].names, end - 3, 2) & mask) != 0:
                return _archive_table_hex(data[].names, end - 10, 6)
            return -1
    return -1


def archive_unicode_digit(point: UInt32, python_minor: Int) raises -> Int:
    """Decimal 0..9, -2 for nondecimal isdigit characters, or -1 otherwise."""
    var mask = _archive_unicode_mask(python_minor)
    var data = ARCHIVE_UNICODE_DATA.get_or_create_ptr()
    var lo = 0
    var hi = data[].digits.byte_length() // 9
    while lo < hi:  # driver-loop: binary-search versioned Unicode digit records
        var mid = (lo + hi) // 2
        var at = mid * 9
        var value = _archive_table_hex(data[].digits, at, 6)
        if Int(point) < value:
            hi = mid
        elif Int(point) > value:
            lo = mid + 1
        else:
            if (_archive_table_hex(data[].digits, at + 7, 2) & mask) == 0:
                return -1
            var digit = _archive_table_hex(data[].digits, at + 6, 1)
            return -2 if digit == 15 else digit
    return -1
