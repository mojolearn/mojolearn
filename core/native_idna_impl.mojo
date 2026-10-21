# SPDX-License-Identifier: Apache-2.0
"""Typed Punycode encoding moved from bindings/native_idna.mojo."""


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
