# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Just enough JSON to read `checks/fixtures/gpt2_reference.json`.

NOT a JSON library and not offered as one: it reads objects, arrays, strings,
integers, `true`/`false`/`null`, and it REFUSES everything else by position --
a float, a duplicate key it does not know, a trailing comma. There is no
number-to-float path at all, which is deliberate: the fixture is text and
integers, and a tokenizer gate that parsed floats could drift on the parse
rather than on the tokenizer.

STRINGS COME BACK AS BYTES, not as `String`. The fixture's `raw_bytes` case
is "\\u0000\\u0001\\u007f", so the text of a case contains NUL; the ids are
compared against bytes and the decode round trip is compared against bytes,
so nothing here ever needs those bytes inside a `String`. `\\uXXXX` is
decoded, including a surrogate PAIR, and a lone surrogate is refused rather
than encoded (it is not a codepoint, and CESU-8 is not UTF-8).
"""


struct JsonCase(Copyable, Movable):
    var name: String
    var text: List[UInt8]
    var ids: List[Int]
    var roundtrip_ok: Bool

    def __init__(out self):
        self.name = String("")
        self.text = List[UInt8]()
        self.ids = List[Int]()
        self.roundtrip_ok = False


struct JsonFixture(Copyable, Movable):
    var encoding: String
    var pat_str: String
    var tiktoken_version: String
    var n_vocab: Int
    var cases: List[JsonCase]

    def __init__(out self):
        self.encoding = String("")
        self.pat_str = String("")
        self.tiktoken_version = String("")
        self.n_vocab = -1
        self.cases = List[JsonCase]()


def _utf8_append(mut out: List[UInt8], cp: Int) raises:
    if cp < 0:
        raise Error("json: negative codepoint")
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp <= 0x10FFFF:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        raise Error("json: codepoint above U+10FFFF")


struct JsonCursor(Copyable, Movable):
    var b: List[UInt8]
    var i: Int

    def __init__(out self, var b: List[UInt8]):
        self.b = b^
        self.i = 0

    def _where(self) -> String:
        return "byte " + String(self.i)

    def skip_ws(mut self):
        while self.i < len(self.b):
            var c = Int(self.b[self.i])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.i += 1
            else:
                break

    def peek(mut self) raises -> Int:
        self.skip_ws()
        if self.i >= len(self.b):
            raise Error("json: end of input at " + self._where())
        return Int(self.b[self.i])

    def take(mut self) raises -> Int:
        var c = self.peek()
        self.i += 1
        return c

    def expect(mut self, c: Int) raises:
        var got = self.take()
        if got != c:
            raise Error(
                "json: expected byte "
                + String(c)
                + ", got "
                + String(got)
                + " at "
                + self._where()
            )

    def parse_string(mut self) raises -> List[UInt8]:
        self.expect(0x22)  # '"'
        var out = List[UInt8]()
        while True:
            if self.i >= len(self.b):
                raise Error("json: unterminated string")
            var c = Int(self.b[self.i])
            self.i += 1
            if c == 0x22:
                return out^
            if c != 0x5C:  # not a backslash
                out.append(UInt8(c))
                continue
            if self.i >= len(self.b):
                raise Error("json: trailing backslash")
            var e = Int(self.b[self.i])
            self.i += 1
            if e == 0x22 or e == 0x5C or e == 0x2F:  # " \ /
                out.append(UInt8(e))
            elif e == 0x6E:  # n
                out.append(UInt8(0x0A))
            elif e == 0x72:  # r
                out.append(UInt8(0x0D))
            elif e == 0x74:  # t
                out.append(UInt8(0x09))
            elif e == 0x62:  # b
                out.append(UInt8(0x08))
            elif e == 0x66:  # f
                out.append(UInt8(0x0C))
            elif e == 0x75:  # u
                var cp = self._hex4()
                if cp >= 0xD800 and cp <= 0xDBFF:
                    if (
                        self.i + 1 < len(self.b)
                        and Int(self.b[self.i]) == 0x5C
                        and Int(self.b[self.i + 1]) == 0x75
                    ):
                        self.i += 2
                        var lo = self._hex4()
                        if lo < 0xDC00 or lo > 0xDFFF:
                            raise Error(
                                "json: high surrogate followed by a"
                                " non-low-surrogate escape"
                            )
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    else:
                        raise Error("json: unpaired high surrogate")
                elif cp >= 0xDC00 and cp <= 0xDFFF:
                    raise Error("json: lone low surrogate")
                _utf8_append(out, cp)
            else:
                raise Error(
                    "json: unknown escape " + String(e) + " at " + self._where()
                )

    def _hex4(mut self) raises -> Int:
        var v = 0
        for _ in range(4):
            if self.i >= len(self.b):
                raise Error("json: truncated \\u escape")
            var c = Int(self.b[self.i])
            self.i += 1
            if c >= 48 and c <= 57:
                v = (v << 4) | (c - 48)
            elif c >= 97 and c <= 102:
                v = (v << 4) | (c - 97 + 10)
            elif c >= 65 and c <= 70:
                v = (v << 4) | (c - 65 + 10)
            else:
                raise Error("json: bad \\u hex digit at " + self._where())
        return v

    def parse_string_text(mut self) raises -> String:
        var raw = self.parse_string()
        var s = String(StringSlice(unsafe_from_utf8=Span(raw)))
        _ = raw
        return s^

    def parse_int(mut self) raises -> Int:
        self.skip_ws()
        var neg = False
        if self.peek() == 0x2D:  # '-'
            neg = True
            self.i += 1
        var digits = 0
        var v = 0
        while self.i < len(self.b):
            var c = Int(self.b[self.i])
            if c < 48 or c > 57:
                break
            v = v * 10 + (c - 48)
            digits += 1
            self.i += 1
        if digits == 0:
            raise Error("json: expected a digit at " + self._where())
        if self.i < len(self.b):
            var c = Int(self.b[self.i])
            if c == 0x2E or c == 0x65 or c == 0x45:  # . e E
                raise Error(
                    "json: this reader takes integers only; a float appears"
                    " at " + self._where()
                )
        return -v if neg else v

    def parse_bool(mut self) raises -> Bool:
        var c = self.peek()
        if c == 0x74:  # 't'
            self._literal("true")
            return True
        if c == 0x66:  # 'f'
            self._literal("false")
            return False
        raise Error("json: expected true or false at " + self._where())

    def _literal(mut self, word: String) raises:
        var w = String(word)
        var wb = w.as_bytes()
        for k in range(len(wb)):
            if self.i >= len(self.b) or self.b[self.i] != wb[k]:
                raise Error(
                    "json: expected '" + word + "' at " + self._where()
                )
            self.i += 1
        _ = w

    def parse_int_array(mut self) raises -> List[Int]:
        var out = List[Int]()
        self.expect(0x5B)  # '['
        if self.peek() == 0x5D:  # ']'
            self.i += 1
            return out^
        while True:
            out.append(self.parse_int())
            var c = self.take()
            if c == 0x5D:
                return out^
            if c != 0x2C:  # ','
                raise Error(
                    "json: expected ',' or ']' in an array at " + self._where()
                )

    def skip_value(mut self) raises:
        var c = self.peek()
        if c == 0x22:
            _ = self.parse_string()
        elif c == 0x7B:  # '{'
            self.i += 1
            if self.peek() == 0x7D:
                self.i += 1
                return
            while True:
                _ = self.parse_string()
                self.expect(0x3A)  # ':'
                self.skip_value()
                var sep = self.take()
                if sep == 0x7D:
                    return
                if sep != 0x2C:
                    raise Error("json: bad object at " + self._where())
        elif c == 0x5B:  # '['
            self.i += 1
            if self.peek() == 0x5D:
                self.i += 1
                return
            while True:
                self.skip_value()
                var sep = self.take()
                if sep == 0x5D:
                    return
                if sep != 0x2C:
                    raise Error("json: bad array at " + self._where())
        elif c == 0x74:
            self._literal("true")
        elif c == 0x66:
            self._literal("false")
        elif c == 0x6E:  # 'n'
            self._literal("null")
        else:
            _ = self.parse_int()


def load_fixture(path: String) raises -> JsonFixture:
    """Read the reference fixture. Unknown top-level and per-case keys are
    SKIPPED, not refused, so the fixture can gain a field; the five fields the
    gate needs are refused by name if absent."""
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()

    var cur = JsonCursor(raw^)
    var out = JsonFixture()
    cur.expect(0x7B)  # '{'
    if cur.peek() == 0x7D:
        raise Error(path + ": empty object")
    while True:
        var key = cur.parse_string_text()
        cur.expect(0x3A)
        if key == "encoding":
            out.encoding = cur.parse_string_text()
        elif key == "pat_str":
            out.pat_str = cur.parse_string_text()
        elif key == "tiktoken_version":
            out.tiktoken_version = cur.parse_string_text()
        elif key == "n_vocab":
            out.n_vocab = cur.parse_int()
        elif key == "cases":
            out.cases = _parse_cases(cur, path)
        else:
            cur.skip_value()
        var sep = cur.take()
        if sep == 0x7D:
            break
        if sep != 0x2C:
            raise Error(path + ": bad top-level object")

    if out.n_vocab < 0:
        raise Error(path + ": no 'n_vocab'")
    if out.pat_str.byte_length() == 0:
        raise Error(path + ": no 'pat_str'")
    if len(out.cases) == 0:
        raise Error(path + ": no 'cases'")
    return out^


def _parse_cases(mut cur: JsonCursor, path: String) raises -> List[JsonCase]:
    var cases = List[JsonCase]()
    cur.expect(0x5B)  # '['
    if cur.peek() == 0x5D:
        cur.i += 1
        return cases^
    while True:
        cases.append(_parse_case(cur, path))
        var sep = cur.take()
        if sep == 0x5D:
            return cases^
        if sep != 0x2C:
            raise Error(path + ": bad 'cases' array")


def _parse_case(mut cur: JsonCursor, path: String) raises -> JsonCase:
    var out = JsonCase()
    var saw_name = False
    var saw_text = False
    var saw_ids = False
    cur.expect(0x7B)
    while True:
        var key = cur.parse_string_text()
        cur.expect(0x3A)
        if key == "name":
            out.name = cur.parse_string_text()
            saw_name = True
        elif key == "text":
            out.text = cur.parse_string()
            saw_text = True
        elif key == "ids":
            out.ids = cur.parse_int_array()
            saw_ids = True
        elif key == "roundtrip_ok":
            out.roundtrip_ok = cur.parse_bool()
        else:
            cur.skip_value()
        var sep = cur.take()
        if sep == 0x7D:
            break
        if sep != 0x2C:
            raise Error(path + ": bad case object")
    if not saw_name or not saw_text or not saw_ids:
        raise Error(
            path
            + ": a case is missing one of 'name', 'text', 'ids' (read name '"
            + out.name
            + "')"
        )
    return out^
