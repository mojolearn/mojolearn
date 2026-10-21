# SPDX-License-Identifier: Apache-2.0
"""Native RFC 3490/3491 hostname preparation and RFC 3492 Punycode.

Preserves CPython socket's IDNA 2003 query semantics (AllowUnassigned true,
UseSTD3ASCIIRules false). Unicode 3.2 properties/normalization are provided by
unicodedata's compiled C database; all mapping, validation and encoding control
runs in Mojo. Python codec registration/search/encode functions are never used.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from core.native_idna_impl import idna_punycode
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
    for entry in [b[].normalize, b[].category, b[].bidi]:  # small-loop(the three Unicode database entries)
        if py_type(entry) is not py_type(b[].bi.len):
            raise Error("idna: Unicode database must expose native C entries")
    b[].mappings = b[].bi.dict()
    var rows = py_call(PythonObject(IDNA_B3_EXCEPTIONS).split, ";")
    for i in range(len(rows) - 1):  # driver-loop: over immutable Unicode 3.2 mapping records at ABI initialization
        var parts = py_call(rows[i].split, ":")
        var points = py_call(parts[1].split, ",")
        var text = b[].bi.list()
        for j in range(len(points)):  # small-loop(the code points of one immutable mapping record)
            _ = py_call(text.append, py_call(b[].bi.chr, py_call(b[].bi.int, points[j])))
        py_setitem(b[].mappings, py_call(b[].bi.int, parts[0]), py_call(PythonObject("").join, text))


def _idna_b3(char: PythonObject) raises -> PythonObject:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    var mapped = py_get(b[].mappings, py_call(b[].bi.ord, char))
    return char.lower() if mapped is None else mapped


def _idna_b2(char: PythonObject) raises -> PythonObject:
    var b = IDNA_BOUNDARY.get_or_create_ptr()
    var lower = _idna_b3(char)
    var first = py_call(b[].normalize, "NFKC", lower)
    var mapped = b[].bi.list()
    for i in range(len(first)):  # driver-loop: over normalized Unicode mapping code points
        _ = py_call(mapped.append, _idna_b3(first[i]))
    var second = py_call(b[].normalize, "NFKC", py_call(PythonObject("").join, mapped))
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
        if not _idna_ignored(Int(py=py_call(b[].bi.ord, char))):
            _ = py_call(parts.append, _idna_b2(char))
    var prepared = py_call(b[].normalize, "NFKC", py_call(PythonObject("").join, parts))
    var right = False
    var left = False
    var first_right = False
    var last_right = False
    for i in range(len(prepared)):  # driver-loop: over prepared hostname prohibition and bidirectional checks
        var char = prepared[i]
        var cp = Int(py=py_call(b[].bi.ord, char))
        if _idna_prohibited(cp, String(py_call(b[].category, char))):
            raise Error("idna: prohibited hostname code point")
        var direction = String(py_call(b[].bidi, char))
        var is_right = direction == "R" or direction == "AL"
        right = right or is_right
        left = left or direction == "L"
        if i == 0:
            first_right = is_right
        last_right = is_right
    if right and (left or not first_right or not last_right):
        raise Error("idna: bidirectional hostname requirement failed")
    return prepared


def _idna_punycode(label: PythonObject) raises -> String:
    return idna_punycode(String(py=label))


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
    if Bool(py=py_call(prepared.lower().startswith, "xn--")):
        raise Error("idna: non-ASCII hostname label has reserved ACE prefix")
    return _idna_punycode(prepared)


def idna_encode(host: PythonObject) raises -> PythonObject:
    if len(host) == 0:
        return py_call(PythonObject("").encode, "ascii")
    var normalized = py_call(py_call(py_call(host.replace, "。", ".").replace, "．", ".").replace, "｡", ".")
    var labels = py_call(normalized.split, ".")
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
    return py_call(PythonObject(encoded).encode, "ascii")
