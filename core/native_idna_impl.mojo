# SPDX-License-Identifier: Apache-2.0
"""Typed Punycode encoding moved from bindings/native_idna_door.mojo."""


def _idna_adapt(delta_in: Int, points: Int, first: Bool) -> Int:
    var delta = delta_in // 700 if first else delta_in // 2
    delta += delta // points
    var k = 0
    while delta > 455:
        delta //= 35
        k += 36
    return k + (36 * delta) // (delta + 38)


def _idna_digit(value: Int) raises -> String:
    if value < 0 or value >= 36:
        raise Error("idna: Punycode digit outside range")
    return String(chr(97 + value if value < 26 else 22 + value))


def idna_punycode(label: String) raises -> String:
    """RFC 3492 Punycode of one prepared label, `xn--` prefixed."""
    var points = List[Int]()
    var out = String()
    var basic = 0
    for codepoint in label.codepoints():
        var cp = Int(codepoint)
        points.append(cp)
        if cp < 128:
            out += String(chr(cp))
            basic += 1
    if basic != 0:
        out += "-"
    var handled = basic
    var n = 128
    var delta = 0
    var bias = 72
    while handled < len(points):  # driver-loop: over Punycode delta encoding of distinct non-ASCII code points
        var next = 0x110000
        for cp in points:
            if cp >= n and cp < next:
                next = cp
        delta += (next - n) * (handled + 1)
        n = next
        for cp in points:
            if cp < n:
                delta += 1
            elif cp == n:
                var q = delta
                var k = 36
                while True:
                    var threshold = 1 if k <= bias else (26 if k >= bias + 26 else k - bias)
                    if q < threshold:
                        break
                    out += _idna_digit(threshold + (q - threshold) % (36 - threshold))
                    q = (q - threshold) // (36 - threshold)
                    k += 36
                out += _idna_digit(q)
                bias = _idna_adapt(delta, handled + 1, handled == basic)
                delta = 0
                handled += 1
        if out.byte_length() + 4 >= 64:
            raise Error("idna: hostname label too long")
        delta += 1
        n += 1
    return "xn--" + out


# Moved verbatim from bindings/native_idna_run.mojo (lane bindings-inline, 2026-11-22).
# Typed IDNA code-point predicates (RUN): the RFC 3454 ignored and prohibited tables used by
# `bindings/native_idna_door.mojo` after it reads each code point and its Unicode category.

def _idna_ignored(cp: Int) -> Bool:
    return cp == 0xAD or cp == 0x34F or cp == 0x1806 or (0x180B <= cp <= 0x180D) or (
        0x200B <= cp <= 0x200D) or cp == 0x2060 or (0xFE00 <= cp <= 0xFE0F) or cp == 0xFEFF


def _idna_prohibited(cp: Int, category: String) -> Bool:
    return (category == "Zs" and cp != 32) or (cp >= 128 and category == "Cc") or (
        category == "Co" or category == "Cs") or (0xFDD0 <= cp < 0xFDF0) or (
        (cp & 0xFFFF) == 0xFFFE or (cp & 0xFFFF) == 0xFFFF) or (
        0xFFF9 <= cp <= 0xFFFD) or (0x2FF0 <= cp <= 0x2FFB) or (
        cp == 0x340 or cp == 0x341 or cp == 0x200E or cp == 0x200F or 0x202A <= cp <= 0x202E or
        0x206A <= cp <= 0x206F) or (cp == 0xE0001 or 0xE0020 <= cp <= 0xE007F) or (
        cp == 0x6DD or cp == 0x70F or cp == 0x180E or cp == 0x200C or cp == 0x200D or
        cp == 0x2028 or cp == 0x2029 or cp == 0xFEFF or 0x2060 <= cp <= 0x2063 or
        0x1D173 <= cp <= 0x1D17A)
