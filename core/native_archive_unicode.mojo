# SPDX-License-Identifier: Apache-2.0
"""Native archive text and escape decoding, with scalar values kept in Mojo.

UTF-8 and escape boundaries depend on preceding bytes. UTF-32 words are
independent and use the existing host pool. No Python codecs/error registry,
Unicode database capsule, or user callback is involved. Literal surrogates
remain scalar values until descriptor admission, as in unicode_escape.
"""
from core.native_archive_span import ArchiveSpan
from core.native_archive_unicode_names import archive_unicode_name
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def archive_hex(byte: Int) -> Int:
    if byte >= 48 and byte <= 57:
        return byte - 48
    if byte >= 65 and byte <= 70:
        return byte - 55
    if byte >= 97 and byte <= 102:
        return byte - 87
    return -1


def archive_unicode_error(raw: ArchiveSpan, encoding: String, start: Int, end: Int, reason: String) raises:
    var message = "'" + encoding + "' codec can't decode "
    if end == start + 1:
        var value = raw.byte(start)
        var digits = String("0123456789abcdef")
        message += "byte 0x" + chr(Int(digits[byte=value >> 4])) + chr(Int(digits[byte=value & 15])) + " in position " + String(start)
    else:
        message += "bytes in position " + String(start) + "-" + String(end - 1)
    raise Error(message + ": " + reason)


def archive_decode_utf8(raw: ArchiveSpan) raises -> List[UInt32]:
    var result = List[UInt32](capacity=raw.size)
    var i = 0
    while i < raw.size:
        var first = raw.byte(i)
        if first < 128:
            result.append(UInt32(first))
            i += 1
            continue
        var width = 2 if first >= 194 and first <= 223 else (3 if first >= 224 and first <= 239 else (4 if first >= 240 and first <= 244 else 0))
        if width == 0:
            archive_unicode_error(raw, "utf-8", i, i + 1, "invalid start byte")
        var value = first & (31 if width == 2 else (15 if width == 3 else 7))
        for offset in range(1, width):
            if i + offset >= raw.size:
                archive_unicode_error(raw, "utf-8", i, raw.size, "unexpected end of data")
            var byte = raw.byte(i + offset)
            var low = 128
            var high = 191
            if offset == 1:
                if first == 224:
                    low = 160
                elif first == 237:
                    high = 159
                elif first == 240:
                    low = 144
                elif first == 244:
                    high = 143
            if byte < low or byte > high:
                archive_unicode_error(raw, "utf-8", i, i + offset, "invalid continuation byte")
            value = (value << 6) | (byte & 63)
        result.append(UInt32(value))
        i += width
    return result^


def archive_decode_utf32(raw: ArchiveSpan) raises -> List[UInt32]:
    var count = raw.size // 4
    var result = List[UInt32](length=count, fill=0)
    var tasks = host_predict_task_count_work(count, raw.size)
    var chunk = host_predict_chunk(count, tasks)
    var invalid = List[Int](length=tasks, fill=count)
    var failures = invalid.unsafe_ptr()
    var dst = result.unsafe_ptr()
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=raw.address)
    def convert(unit: Int) {imm failures, imm dst, imm src, imm chunk, imm count}:
        var first_bad = count
        for i in range(unit * chunk, min(count, (unit + 1) * chunk)):
            var word = UInt32(src[4 * i]) | (UInt32(src[4 * i + 1]) << 8) | (UInt32(src[4 * i + 2]) << 16) | (UInt32(src[4 * i + 3]) << 24)
            dst[i] = word
            if ((word >= 0xD800 and word <= 0xDFFF) or word > 0x10FFFF) and i < first_bad:
                first_bad = i
        failures[unit] = first_bad
    host_parallelize(convert, tasks)
    var first_bad = count
    for value in invalid:
        first_bad = min(first_bad, value)
    if first_bad != count:
        var word = result[first_bad]
        var reason = "code point not in range(0x110000)" if word > 0x10FFFF else "code point in surrogate code point range(0xd800, 0xe000)"
        archive_unicode_error(raw, "utf-32-le", first_bad * 4, first_bad * 4 + 4, reason)
    if raw.size % 4:
        archive_unicode_error(raw, "utf-32-le", count * 4, raw.size, "truncated data")
    _ = invalid
    return result^


struct ArchiveLiteral(Movable):
    var points: List[UInt32]
    var warning_at: Int
    var warning_kind: Int
    var warning_value: Int

    def __init__(out self):
        self.points = List[UInt32]()
        self.warning_at = -1
        self.warning_kind = 0
        self.warning_value = 0


def archive_decode_literal(raw: ArchiveSpan, python_minor: Int, python_micro: Int = 0) raises -> ArchiveLiteral:
    var result = ArchiveLiteral()
    var i = 0
    # Escape token boundaries depend on the preceding token. The enclosing
    # member parses run independently in the reader's existing host pool.
    while i < raw.size:
        var letter = raw.byte(i)
        if letter != 92:
            result.points.append(UInt32(letter))
            i += 1
            continue
        var start = i
        i += 1
        if i == raw.size:
            archive_unicode_error(raw, "unicodeescape", start, i, "\\ at end of string")
        letter = raw.byte(i)
        i += 1
        if letter == 10:
            continue
        if letter == 120 or letter == 117 or letter == 85:
            var width = 2 if letter == 120 else (4 if letter == 117 else 8)
            var reason = "truncated \\xXX escape" if width == 2 else ("truncated \\uXXXX escape" if width == 4 else "truncated \\UXXXXXXXX escape")
            var value = UInt32(0)
            for k in range(width):
                if i == raw.size:
                    archive_unicode_error(raw, "unicodeescape", start, i, reason)
                var digit = archive_hex(raw.byte(i))
                if digit < 0:
                    archive_unicode_error(raw, "unicodeescape", start, i, reason)
                value = (value << 4) | UInt32(digit)
                i += 1
            if value > 0x10FFFF:
                archive_unicode_error(raw, "unicodeescape", start, i, "illegal Unicode character")
            result.points.append(value)
        elif letter == 78:
            if i == raw.size or raw.byte(i) != 123:
                archive_unicode_error(raw, "unicodeescape", start, i, "malformed \\N character escape")
            i += 1
            var begin = i
            while i < raw.size and raw.byte(i) != 125:
                i += 1
            if i == raw.size or i == begin:
                archive_unicode_error(raw, "unicodeescape", start, i, "malformed \\N character escape")
            var point = archive_unicode_name(raw.slice(begin, i - begin), python_minor, python_micro)
            i += 1
            if point < 0:
                archive_unicode_error(raw, "unicodeescape", start, i, "unknown Unicode character name")
            result.points.append(UInt32(point))
        elif letter >= 48 and letter <= 55:
            var value = letter - 48
            for k in range(2):
                if i < raw.size and raw.byte(i) >= 48 and raw.byte(i) <= 55:
                    value = (value << 3) | (raw.byte(i) - 48)
                    i += 1
                else:
                    break
            result.points.append(UInt32(value))
            if value > 255 and result.warning_at < 0:
                result.warning_at = start
                result.warning_kind = 2
                result.warning_value = value
        else:
            var value = letter
            if letter == 97:
                value = 7
            elif letter == 98:
                value = 8
            elif letter == 102:
                value = 12
            elif letter == 110:
                value = 10
            elif letter == 114:
                value = 13
            elif letter == 116:
                value = 9
            elif letter == 118:
                value = 11
            elif letter != 92 and letter != 34 and letter != 39:
                result.points.append(92)
                if result.warning_at < 0:
                    result.warning_at = start
                    result.warning_kind = 1
                    result.warning_value = letter
            result.points.append(UInt32(value))
    return result^


def archive_text_equal(points: List[UInt32], text: String) -> Bool:
    var bytes = text.as_bytes()
    if len(points) != len(bytes):
        return False
    for i in range(len(points)):
        if points[i] != UInt32(bytes[i]):
            return False
    return True


def archive_text_string(points: List[UInt32]) -> String:
    var result = String("")
    for point in points:
        result += chr(Int(point))
    return result
