# SPDX-License-Identifier: Apache-2.0
"""Native RFC 3490/3491 hostname preparation and RFC 3492 Punycode.

Preserves CPython socket's IDNA 2003 query semantics (AllowUnassigned true,
UseSTD3ASCIIRules false). Unicode 3.2 properties/normalization are provided by
unicodedata's compiled C database; all mapping, validation and encoding control
runs in Mojo. Python codec registration/search/encode functions are never used.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.native_idna_data import IDNA_B3_EXCEPTIONS


struct IdnaBoundary(Defaultable, Movable):
    var bi: PythonObject
    var normalize: PythonObject
    var category: PythonObject
    var bidi: PythonObject
    var mappings: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.normalize = PythonObject(None)
        self.category = PythonObject(None)
        self.bidi = PythonObject(None)
        self.mappings = PythonObject(None)


comptime IDNA_BOUNDARY = _Global[
    StorageType=IdnaBoundary, name="MojoNativeIdnaBoundary",
    init_fn=IdnaBoundary.__init__,
]


def idna_initialize() raises:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    var unicode = Python.import_module("unicodedata").ucd_3_2_0
    b[].normalize = unicode.normalize
    b[].category = unicode.category
    b[].bidi = unicode.bidirectional
    for entry in [b[].normalize, b[].category, b[].bidi]:
        if b[].bi.type(entry) is not b[].bi.type(b[].bi.len):
            raise Error("idna: Unicode database must expose native C entries")
    b[].mappings = b[].bi.dict()
    var rows = PythonObject(IDNA_B3_EXCEPTIONS).split(";")
    for i in range(len(rows) - 1):  # driver-loop: over immutable Unicode 3.2 mapping records at ABI initialization
        var parts = rows[i].split(":")
        var points = parts[1].split(",")
        var text = b[].bi.list()
        for j in range(len(points)):
            text.append(b[].bi.chr(b[].bi.int(points[j])))
        b[].mappings[b[].bi.int(parts[0])] = PythonObject("").join(text)


def _idna_b3(char: PythonObject) raises -> PythonObject:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    var mapped = b[].mappings.get(b[].bi.ord(char))
    return char.lower() if mapped is None else mapped


def _idna_b2(char: PythonObject) raises -> PythonObject:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    var lower = _idna_b3(char)
    var first = b[].normalize("NFKC", lower)
    var mapped = b[].bi.list()
    for i in range(len(first)):  # driver-loop: over normalized Unicode mapping code points
        mapped.append(_idna_b3(first[i]))
    var second = b[].normalize("NFKC", PythonObject("").join(mapped))
    return second if Bool(py=first != second) else lower


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


def _idna_prepare(label: PythonObject) raises -> PythonObject:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    var parts = b[].bi.list()
    for i in range(len(label)):  # driver-loop: over hostname code points in RFC 3491 mapping order
        var char = label[i]
        if not _idna_ignored(Int(py=b[].bi.ord(char))):
            parts.append(_idna_b2(char))
    var prepared = b[].normalize("NFKC", PythonObject("").join(parts))
    var right = False
    var left = False
    var first_right = False
    var last_right = False
    for i in range(len(prepared)):  # driver-loop: over prepared hostname prohibition and bidirectional checks
        var char = prepared[i]
        var cp = Int(py=b[].bi.ord(char))
        if _idna_prohibited(cp, String(b[].category(char))):
            raise Error("idna: prohibited hostname code point")
        var direction = String(b[].bidi(char))
        var is_right = direction == "R" or direction == "AL"
        right = right or is_right
        left = left or direction == "L"
        if i == 0:
            first_right = is_right
        last_right = is_right
    if right and (left or not first_right or not last_right):
        raise Error("idna: bidirectional hostname requirement failed")
    return prepared


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
    return String(IDNA_BOUNDARY.get_or_create_ptr()[].bi.chr(97 + value if value < 26 else 22 + value))


def _idna_punycode(label: PythonObject) raises -> String:
    var bi = IDNA_BOUNDARY.get_or_create_ptr()[].bi
    var points = List[Int]()
    var out = String()
    var basic = 0
    for i in range(len(label)):
        var cp = Int(py=bi.ord(label[i]))
        points.append(cp)
        if cp < 128:
            out += String(label[i])
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


def _idna_label(label: PythonObject) raises -> String:
    if Bool(py=label.isascii()):
        if len(label) == 0 or len(label) >= 64:
            raise Error("idna: hostname label is empty or too long")
        return String(label)
    var prepared = _idna_prepare(label)
    if Bool(py=prepared.isascii()):
        if len(prepared) == 0 or len(prepared) >= 64:
            raise Error("idna: prepared hostname label is empty or too long")
        return String(prepared)
    if Bool(py=prepared.lower().startswith("xn--")):
        raise Error("idna: non-ASCII hostname label has reserved ACE prefix")
    return _idna_punycode(prepared)


def idna_encode(host: PythonObject) raises -> PythonObject:
    if len(host) == 0:
        return PythonObject("").encode("ascii")
    var normalized = host.replace("。", ".").replace("．", ".").replace("｡", ".")
    var labels = normalized.split(".")
    var count = len(labels)
    var trailing = len(labels[count - 1]) == 0
    if trailing:
        count -= 1
    var encoded = String()
    for i in range(count):  # driver-loop: over DNS labels in source order
        if i != 0:
            encoded += "."
        encoded += _idna_label(labels[i])
    if trailing:
        encoded += "."
    return PythonObject(encoded).encode("ascii")
