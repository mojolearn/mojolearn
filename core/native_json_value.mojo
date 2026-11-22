# SPDX-License-Identifier: Apache-2.0
"""A native JSON value: parsed from bytes, built from typed values, written
as the indented text the corpus manifests have always had. No Python.

`JsonDoc` is one arena of nodes; a node is an Int. A node that is already in
the arena may be placed under several parents (the tree is only ever added
to), so a sub-value read from one file (a vocabulary identity, a source URL)
is put into another document without a copy.

Numbers keep their source text. An integer token is read with
`json_number_int`; a token written by this module comes from `json_add_int`
or `json_add_float`. A float token read from a file is written back as it was
read (the files this serves were written with the shortest round-trip form).

`json_write` is the manifest writer's exact layout: two-space indent, one
member per line, `": "` after a name, `{}` and `[]` for empty containers,
names and strings escaped to ASCII (`\\uXXXX`, surrogate pairs above the
BMP), members in insertion order.

Serial by nature: a document is a few hundred bytes read or written once per
corpus; there is no independent unit to split.
"""

comptime JSON_NULL = 0
comptime JSON_FALSE = 1
comptime JSON_TRUE = 2
comptime JSON_NUMBER = 3
comptime JSON_STRING = 4
comptime JSON_ARRAY = 5
comptime JSON_OBJECT = 6
comptime _JSON_DEPTH = 128


struct JsonDoc(Movable):
    """The arena. `kind[n]`, `text[n]` (a number's token or a string's UTF-8
    text), `kids[n]` (member values or elements, in order) and `names[n]`
    (member names of an object, parallel to `kids[n]`)."""

    var kind: List[Int]
    var text: List[String]
    var kids: List[List[Int]]
    var names: List[List[String]]

    def __init__(out self):
        self.kind = List[Int]()
        self.text = List[String]()
        self.kids = List[List[Int]]()
        self.names = List[List[String]]()


def _json_node(mut doc: JsonDoc, kind: Int, text: String) -> Int:
    doc.kind.append(kind)
    doc.text.append(text.copy())
    doc.kids.append(List[Int]())
    doc.names.append(List[String]())
    return len(doc.kind) - 1


def json_add_null(mut doc: JsonDoc) -> Int:
    return _json_node(doc, JSON_NULL, String(""))


def json_add_bool(mut doc: JsonDoc, value: Bool) -> Int:
    return _json_node(doc, JSON_TRUE if value else JSON_FALSE, String(""))


def json_add_int(mut doc: JsonDoc, value: Int) -> Int:
    return _json_node(doc, JSON_NUMBER, String(value))


def json_add_number(mut doc: JsonDoc, token: String) -> Int:
    """A number by its JSON token (the caller's own spelling)."""
    return _json_node(doc, JSON_NUMBER, token)


def json_add_float(mut doc: JsonDoc, value: Float64) -> Int:
    """A binary64 by its shortest round-trip text; the non-finite values by
    the names the manifest writer has always used."""
    if value != value:
        return _json_node(doc, JSON_NUMBER, String("NaN"))
    if value - value != 0.0:
        return _json_node(doc, JSON_NUMBER, String("Infinity") if value > 0.0 else String("-Infinity"))
    return _json_node(doc, JSON_NUMBER, String(value))


def json_add_string(mut doc: JsonDoc, value: String) -> Int:
    return _json_node(doc, JSON_STRING, value)


def json_add_array(mut doc: JsonDoc) -> Int:
    return _json_node(doc, JSON_ARRAY, String(""))


def json_add_object(mut doc: JsonDoc) -> Int:
    return _json_node(doc, JSON_OBJECT, String(""))


def json_push(mut doc: JsonDoc, array: Int, value: Int):
    doc.kids[array].append(value)


def json_member(doc: JsonDoc, node: Int, name: String) -> Int:
    """The value of member `name` of an object, or -1 (also for a node that
    is not an object)."""
    if node < 0 or doc.kind[node] != JSON_OBJECT:
        return -1
    for i in range(len(doc.names[node])):  # small-loop(members of one manifest object)
        if doc.names[node][i] == name:
            return doc.kids[node][i]
    return -1


def json_put(mut doc: JsonDoc, target: Int, name: String, value: Int):
    """Set member `name`: a name already present keeps its place and takes
    the new value (a dictionary's rule)."""
    for i in range(len(doc.names[target])):  # small-loop(members of one manifest object)
        if doc.names[target][i] == name:
            doc.kids[target][i] = value
            return
    doc.names[target].append(name.copy())
    doc.kids[target].append(value)


def json_kind(doc: JsonDoc, node: Int) -> Int:
    return JSON_NULL if node < 0 else doc.kind[node]


def json_count(doc: JsonDoc, node: Int) -> Int:
    return 0 if node < 0 else len(doc.kids[node])


def json_item(doc: JsonDoc, node: Int, index: Int) -> Int:
    return doc.kids[node][index]


def json_string(doc: JsonDoc, node: Int) -> String:
    """The text of a string node; empty for anything else."""
    if node < 0 or doc.kind[node] != JSON_STRING:
        return String("")
    return doc.text[node].copy()


def json_string_is(doc: JsonDoc, node: Int, wanted: String) -> Bool:
    return node >= 0 and doc.kind[node] == JSON_STRING and doc.text[node] == wanted


def json_truthy(doc: JsonDoc, node: Int) -> Bool:
    """Python truth of the value: absent, null, false, zero, an empty string
    and an empty container are false."""
    if node < 0:
        return False
    var kind = doc.kind[node]
    if kind == JSON_NULL or kind == JSON_FALSE:
        return False
    if kind == JSON_TRUE:
        return True
    if kind == JSON_STRING:
        return doc.text[node].byte_length() > 0
    if kind == JSON_NUMBER:
        var view = doc.text[node].as_bytes()
        for i in range(len(view)):  # small-loop(bytes of one number token)
            var c = Int(view[i])
            if c >= 49 and c <= 57:
                return True
            if c == 101 or c == 69:
                return False
            if c == 78 or c == 73:
                return True
        return False
    return len(doc.kids[node]) > 0


def _json_token_int(token: String, mut exact: Bool) raises -> Int:
    """The integer a number token denotes, truncated toward zero; `exact`
    says whether nothing was cut off."""
    var view = token.as_bytes()
    var n = len(view)
    var at = 0
    var negative = False
    if at < n and Int(view[at]) == 45:
        negative = True
        at += 1
    var digits = List[Int]()
    var point = -1
    while at < n:  # small-loop(bytes of one number token)
        var c = Int(view[at])
        if c >= 48 and c <= 57:
            digits.append(c - 48)
        elif c == 46 and point < 0:
            point = len(digits)
        else:
            break
        at += 1
    if len(digits) == 0:
        raise Error("ValueError: corpus metadata number is not finite")
    var shift = 0
    if at < n:
        var c = Int(view[at])
        if c != 101 and c != 69:
            raise Error("ValueError: corpus metadata number is malformed")
        at += 1
        var sign = 1
        if at < n and (Int(view[at]) == 45 or Int(view[at]) == 43):
            sign = -1 if Int(view[at]) == 45 else 1
            at += 1
        var seen = False
        while at < n:  # small-loop(exponent digits of one number token)
            var d = Int(view[at]) - 48
            if d < 0 or d > 9:
                raise Error("ValueError: corpus metadata number is malformed")
            if shift < 100000:
                shift = shift * 10 + d
            seen = True
            at += 1
        if not seen:
            raise Error("ValueError: corpus metadata number is malformed")
        shift *= sign
    # The integer part ends `whole` digits into the digit string.
    var whole = (len(digits) if point < 0 else point) + shift
    var value = 0
    exact = True
    for i in range(len(digits)):  # small-loop(digits of one number token)
        if i < whole:
            if value > (9223372036854775807 - digits[i]) // 10:
                raise Error("OverflowError: corpus metadata integer exceeds the native word")
            value = value * 10 + digits[i]
        elif digits[i] != 0:
            exact = False
    for _ in range(max(whole - len(digits), 0)):  # small-loop(trailing zeros of a scaled token)
        if value > 9223372036854775807 // 10:
            raise Error("OverflowError: corpus metadata integer exceeds the native word")
        value *= 10
    return -value if negative else value


def json_number_int(doc: JsonDoc, node: Int) raises -> Int:
    """A number node as an integer, truncated toward zero (what `int()` of
    the parsed value gave)."""
    if node < 0 or doc.kind[node] != JSON_NUMBER:
        raise Error("TypeError: corpus metadata requires a number")
    var exact = True
    return _json_token_int(doc.text[node], exact)


def json_number_equals(doc: JsonDoc, node: Int, wanted: Int) -> Bool:
    """`value == wanted` for a number node: an integer, or a float token
    that denotes exactly that integer. Anything else is not equal."""
    if node < 0 or doc.kind[node] != JSON_NUMBER:
        return False
    try:
        var exact = True
        var value = _json_token_int(doc.text[node], exact)
        return exact and value == wanted
    except:
        return False


# ---------------------------------------------------------------- writer


def _json_put_text(mut dst: List[UInt8], text: String):
    var view = text.as_bytes()
    for i in range(len(view)):  # small-loop(bytes of one manifest token)
        dst.append(view[i])


def _json_put_hex4(mut dst: List[UInt8], unit: Int):
    dst.append(UInt8(92))
    dst.append(UInt8(117))
    for d in range(4):  # small-loop(hex digits of one UTF-16 unit)
        var nib = (unit >> (12 - 4 * d)) & 15
        dst.append(UInt8(48 + nib) if nib < 10 else UInt8(87 + nib))


def json_escape_into(mut dst: List[UInt8], text: String):
    """One JSON string literal, ASCII only: `"` `\\` and the five short
    escapes, every other byte outside space..tilde as `\\uXXXX` (a surrogate
    pair above U+FFFF), lowercase hex."""
    var view = text.as_bytes()
    var n = len(view)
    dst.append(UInt8(34))
    var i = 0
    while i < n:  # small-loop(bytes of one manifest string)
        var c = Int(view[i])
        if c == 34 or c == 92:
            dst.append(UInt8(92))
            dst.append(UInt8(c))
            i += 1
        elif c >= 32 and c <= 126:
            dst.append(UInt8(c))
            i += 1
        elif c == 10 or c == 13 or c == 9 or c == 8 or c == 12:
            dst.append(UInt8(92))
            dst.append(UInt8(110) if c == 10 else (UInt8(114) if c == 13 else (UInt8(116) if c == 9 else (UInt8(98) if c == 8 else UInt8(102)))))
            i += 1
        elif c < 128:
            _json_put_hex4(dst, c)
            i += 1
        else:
            # One UTF-8 sequence (the text came from an admitted str or from
            # this module's parser, both well formed).
            var extra = 1 if c < 0xE0 else (2 if c < 0xF0 else 3)
            var point = c & (0x1F if extra == 1 else (0x0F if extra == 2 else 0x07))
            for k in range(extra):  # small-loop(continuation bytes of one code point)
                if i + 1 + k < n:
                    point = (point << 6) | (Int(view[i + 1 + k]) & 0x3F)
            i += 1 + extra
            if point >= 0x10000:
                var v = point - 0x10000
                _json_put_hex4(dst, 0xD800 | ((v >> 10) & 0x3FF))
                _json_put_hex4(dst, 0xDC00 | (v & 0x3FF))
            else:
                _json_put_hex4(dst, point)
    dst.append(UInt8(34))


def _json_indent(mut dst: List[UInt8], depth: Int):
    for _ in range(2 * depth):  # small-loop(indent of one manifest line)
        dst.append(UInt8(32))


def json_write(doc: JsonDoc, node: Int, mut dst: List[UInt8], depth: Int = 0) raises:
    """Append the indented text of `node` (no trailing newline)."""
    if depth > _JSON_DEPTH:
        raise Error("ValueError: corpus metadata exceeds supported JSON nesting")
    var kind = json_kind(doc, node)
    if kind == JSON_NULL:
        _json_put_text(dst, "null")
        return
    if kind == JSON_TRUE or kind == JSON_FALSE:
        _json_put_text(dst, "true" if kind == JSON_TRUE else "false")
        return
    if kind == JSON_NUMBER:
        _json_put_text(dst, doc.text[node])
        return
    if kind == JSON_STRING:
        json_escape_into(dst, doc.text[node])
        return
    var mapping = kind == JSON_OBJECT
    var count = len(doc.kids[node])
    if count == 0:
        _json_put_text(dst, "{}" if mapping else "[]")
        return
    dst.append(UInt8(123) if mapping else UInt8(91))
    dst.append(UInt8(10))
    for i in range(count):  # driver-loop: over the members of one manifest container, in insertion order
        if i:
            dst.append(UInt8(44))
            dst.append(UInt8(10))
        _json_indent(dst, depth + 1)
        if mapping:
            json_escape_into(dst, doc.names[node][i])
            dst.append(UInt8(58))
            dst.append(UInt8(32))
        json_write(doc, doc.kids[node][i], dst, depth + 1)
    dst.append(UInt8(10))
    _json_indent(dst, depth)
    dst.append(UInt8(125) if mapping else UInt8(93))


# ---------------------------------------------------------------- parser


struct JsonReader(Movable):
    var at: Int

    def __init__(out self):
        self.at = 0


def _json_ws(raw: List[UInt8], mut reader: JsonReader):
    while reader.at < len(raw):  # small-loop(whitespace between manifest tokens)
        var c = Int(raw[reader.at])
        if c != 32 and c != 10 and c != 13 and c != 9:
            return
        reader.at += 1


def _json_peek(raw: List[UInt8], reader: JsonReader) raises -> Int:
    if reader.at >= len(raw):
        raise Error("JSON: unexpected end of document")
    return Int(raw[reader.at])


def _json_utf8(mut dst: List[UInt8], point: Int):
    if point < 0x80:
        dst.append(UInt8(point))
    elif point < 0x800:
        dst.append(UInt8(0xC0 | (point >> 6)))
        dst.append(UInt8(0x80 | (point & 0x3F)))
    elif point < 0x10000:
        dst.append(UInt8(0xE0 | (point >> 12)))
        dst.append(UInt8(0x80 | ((point >> 6) & 0x3F)))
        dst.append(UInt8(0x80 | (point & 0x3F)))
    else:
        dst.append(UInt8(0xF0 | (point >> 18)))
        dst.append(UInt8(0x80 | ((point >> 12) & 0x3F)))
        dst.append(UInt8(0x80 | ((point >> 6) & 0x3F)))
        dst.append(UInt8(0x80 | (point & 0x3F)))


def _json_hex4(raw: List[UInt8], mut reader: JsonReader) raises -> Int:
    var value = 0
    for _ in range(4):  # small-loop(the four hex digits of one Unicode escape)
        var c = _json_peek(raw, reader)
        reader.at += 1
        if c >= 48 and c <= 57:
            value = (value << 4) | (c - 48)
        elif c >= 97 and c <= 102:
            value = (value << 4) | (c - 87)
        elif c >= 65 and c <= 70:
            value = (value << 4) | (c - 55)
        else:
            raise Error("JSON: invalid Unicode escape at " + String(reader.at))
    return value


def _json_text(raw: List[UInt8], mut reader: JsonReader) raises -> String:
    """One string literal (the reader is at its opening quote) as UTF-8 text."""
    reader.at += 1
    var bytes = List[UInt8]()
    while True:  # small-loop(bytes of one manifest string)
        var c = _json_peek(raw, reader)
        reader.at += 1
        if c == 34:
            break
        if c != 92:
            bytes.append(UInt8(c))
            continue
        var e = _json_peek(raw, reader)
        reader.at += 1
        if e == 34 or e == 92 or e == 47:
            bytes.append(UInt8(e))
        elif e == 110:
            bytes.append(UInt8(10))
        elif e == 114:
            bytes.append(UInt8(13))
        elif e == 116:
            bytes.append(UInt8(9))
        elif e == 98:
            bytes.append(UInt8(8))
        elif e == 102:
            bytes.append(UInt8(12))
        elif e == 117:
            var point = _json_hex4(raw, reader)
            if point >= 0xD800 and point <= 0xDBFF:
                if reader.at + 1 < len(raw) and Int(raw[reader.at]) == 92 and Int(raw[reader.at + 1]) == 117:
                    reader.at += 2
                    var low = _json_hex4(raw, reader)
                    if low < 0xDC00 or low > 0xDFFF:
                        raise Error("JSON: unpaired surrogate escape at " + String(reader.at))
                    point = 0x10000 + ((point - 0xD800) << 10) + (low - 0xDC00)
                else:
                    raise Error("JSON: unpaired surrogate escape at " + String(reader.at))
            elif point >= 0xDC00 and point <= 0xDFFF:
                raise Error("JSON: unpaired surrogate escape at " + String(reader.at))
            _json_utf8(bytes, point)
        else:
            raise Error("JSON: invalid escape at " + String(reader.at))
    var result = String(StringSlice(unsafe_from_utf8=Span(bytes)))
    _ = len(bytes)
    return result


def _json_word(raw: List[UInt8], mut reader: JsonReader, word: String) -> Bool:
    var view = word.as_bytes()
    if reader.at + len(view) > len(raw):
        return False
    for i in range(len(view)):  # small-loop(bytes of one JSON literal)
        if raw[reader.at + i] != view[i]:
            return False
    reader.at += len(view)
    return True


def _json_value(mut doc: JsonDoc, raw: List[UInt8], mut reader: JsonReader, depth: Int) raises -> Int:
    if depth > _JSON_DEPTH:
        raise Error("JSON: nesting limit exceeded")
    _json_ws(raw, reader)
    var c = _json_peek(raw, reader)
    if c == 34:
        var text = _json_text(raw, reader)
        return json_add_string(doc, text)
    if c == 123 or c == 91:
        var mapping = c == 123
        var closer = 125 if mapping else 93
        var node = json_add_object(doc) if mapping else json_add_array(doc)
        reader.at += 1
        _json_ws(raw, reader)
        if _json_peek(raw, reader) == closer:
            reader.at += 1
            return node
        while True:  # driver-loop: over the members of one manifest container
            if mapping:
                _json_ws(raw, reader)
                if _json_peek(raw, reader) != 34:
                    raise Error("JSON: expected a member name at " + String(reader.at))
                var name = _json_text(raw, reader)
                _json_ws(raw, reader)
                if _json_peek(raw, reader) != 58:
                    raise Error("JSON: expected ':' at " + String(reader.at))
                reader.at += 1
                var member = _json_value(doc, raw, reader, depth + 1)
                json_put(doc, node, name, member)
            else:
                var element = _json_value(doc, raw, reader, depth + 1)
                json_push(doc, node, element)
            _json_ws(raw, reader)
            var d = _json_peek(raw, reader)
            reader.at += 1
            if d == closer:
                return node
            if d != 44:
                raise Error("JSON: expected ',' at " + String(reader.at))
    if _json_word(raw, reader, "null"):
        return json_add_null(doc)
    if _json_word(raw, reader, "true"):
        return json_add_bool(doc, True)
    if _json_word(raw, reader, "false"):
        return json_add_bool(doc, False)
    if _json_word(raw, reader, "NaN"):
        return json_add_number(doc, String("NaN"))
    if _json_word(raw, reader, "Infinity"):
        return json_add_number(doc, String("Infinity"))
    if _json_word(raw, reader, "-Infinity"):
        return json_add_number(doc, String("-Infinity"))
    var start = reader.at
    var token = List[UInt8]()
    while reader.at < len(raw):  # small-loop(bytes of one number token)
        var d = Int(raw[reader.at])
        if not ((d >= 48 and d <= 57) or d == 45 or d == 43 or d == 46 or d == 101 or d == 69):
            break
        token.append(UInt8(d))
        reader.at += 1
    if reader.at == start:
        raise Error("JSON: unexpected byte at " + String(start))
    var spelled = String(StringSlice(unsafe_from_utf8=Span(token)))
    _ = len(token)
    if spelled == "-0":
        spelled = String("0")
    return json_add_number(doc, spelled)


def json_parse(mut doc: JsonDoc, raw: List[UInt8]) raises -> Int:
    """Parse one document into the arena; returns its root node."""
    var reader = JsonReader()
    var root = _json_value(doc, raw, reader, 0)
    _json_ws(raw, reader)
    if reader.at != len(raw):
        raise Error("JSON: trailing data at " + String(reader.at))
    return root
