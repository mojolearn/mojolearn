# SPDX-License-Identifier: Apache-2.0
"""Typed transport frame header (RUN, no Python).

A transport frame on the wire is: a 4-byte big-endian header length, the
header, the payload. The header is one ASCII JSON object whose `nbytes`
field is the payload length. `TransportHeader` writes that object from typed
fields and the `transport_header_*` readers take typed fields out of a
received one, so a RUN caller frames and reads a message without a Python
dictionary. The text is the one the boundary door writes for the same
fields (keys in code-point order, no spaces, the `encode_basestring_ascii`
escapes), so a typed end and a door end of one version read each other.

Field kinds: integer, boolean, text, a list of integers, null. A floating
value travels as its bit word in an integer field (or a list of them): no
decimal text is formatted or parsed here.
"""

comptime TRANSPORT_HEADER_LIMIT = 1048576
comptime _HeaderBytes = MutPointer[UInt8, MutUntrackedOrigin]


def _header_append_text(mut dst: List[UInt8], text: String):
    var view = text.as_bytes()
    for i in range(len(view)):  # small-loop(bytes of one short header token)
        dst.append(view[i])


def _header_hex_digit(value: Int) -> UInt8:
    return UInt8(48 + value) if value < 10 else UInt8(87 + value)


def _header_append_unit(mut dst: List[UInt8], unit: Int):
    # One UTF-16 code unit as \uXXXX, lower-case hex as CPython writes it.
    dst.append(UInt8(92))
    dst.append(UInt8(117))
    dst.append(_header_hex_digit((unit >> 12) & 15))
    dst.append(_header_hex_digit((unit >> 8) & 15))
    dst.append(_header_hex_digit((unit >> 4) & 15))
    dst.append(_header_hex_digit(unit & 15))


def _header_append_quoted(mut dst: List[UInt8], text: String) raises:
    """`text` as a JSON string with the `encode_basestring_ascii` escapes."""
    var view = text.as_bytes()
    var n = len(view)
    dst.append(UInt8(34))
    var i = 0
    while i < n:  # small-loop(bytes of one short header text)
        var b = Int(view[i])
        if b == 34 or b == 92:
            dst.append(UInt8(92))
            dst.append(UInt8(b))
            i += 1
        elif b >= 32 and b < 127:
            dst.append(UInt8(b))
            i += 1
        elif b == 10 or b == 13 or b == 9 or b == 8 or b == 12:
            dst.append(UInt8(92))
            dst.append(UInt8(110 if b == 10 else (114 if b == 13 else (116 if b == 9 else (98 if b == 8 else 102)))))
            i += 1
        elif b < 128:
            _header_append_unit(dst, b)
            i += 1
        else:
            var extra = 1 if b < 224 else (2 if b < 240 else 3)
            if b < 194 or b > 244 or i + extra >= n:
                raise Error("transport: header text is not valid UTF-8")
            var point = b & (31 if extra == 1 else (15 if extra == 2 else 7))
            for k in range(extra):  # small-loop(continuation bytes of one code point)
                var c = Int(view[i + 1 + k])
                if (c & 192) != 128:
                    raise Error("transport: header text is not valid UTF-8")
                point = (point << 6) | (c & 63)
            if point < 65536:
                _header_append_unit(dst, point)
            else:
                point -= 65536
                _header_append_unit(dst, 55296 + (point >> 10))
                _header_append_unit(dst, 56320 + (point & 1023))
            i += 1 + extra
    dst.append(UInt8(34))


def _header_key_less(a: String, b: String) -> Bool:
    var x = a.as_bytes()
    var y = b.as_bytes()
    var n = min(len(x), len(y))
    for i in range(n):  # small-loop(bytes of two short header keys)
        if x[i] != y[i]:
            return x[i] < y[i]
    return len(x) < len(y)


struct TransportHeader(Movable):
    """The typed fields of one frame header. Values are kept already encoded
    in one flat byte list; `order` lists the fields by key."""
    var keys: List[String]
    var starts: List[Int]
    var ends: List[Int]
    var order: List[Int]
    var values: List[UInt8]

    def __init__(out self):
        self.keys = List[String]()
        self.starts = List[Int]()
        self.ends = List[Int]()
        self.order = List[Int]()
        self.values = List[UInt8]()

    def _open(mut self, key: String) raises:
        var view = key.as_bytes()
        if len(view) == 0:
            raise Error("transport: header key is empty")
        for i in range(len(view)):  # small-loop(bytes of one short header key)
            var b = Int(view[i])
            if b < 32 or b > 126 or b == 34 or b == 92:
                raise Error("transport: header keys are plain ASCII names")
        if key == "nbytes":
            raise Error("transport: nbytes is written by the transport")
        var at = len(self.order)
        for i in range(len(self.order)):  # small-loop(fields of one header)
            if self.keys[self.order[i]] == key:
                raise Error("transport: header field written twice: " + key)
            if _header_key_less(key, self.keys[self.order[i]]):
                at = i
                break
        self.order.append(0)
        var j = len(self.order) - 1
        while j > at:  # small-loop(fields of one header)
            self.order[j] = self.order[j - 1]
            j -= 1
        self.order[at] = len(self.keys)
        self.keys.append(key)
        self.starts.append(len(self.values))

    def _close(mut self):
        self.ends.append(len(self.values))

    def put_int(mut self, key: String, value: Int) raises:
        self._open(key)
        _header_append_text(self.values, String(value))
        self._close()

    def put_bool(mut self, key: String, value: Bool) raises:
        self._open(key)
        _header_append_text(self.values, "true" if value else "false")
        self._close()

    def put_null(mut self, key: String) raises:
        self._open(key)
        _header_append_text(self.values, "null")
        self._close()

    def put_text(mut self, key: String, value: String) raises:
        self._open(key)
        _header_append_quoted(self.values, value)
        self._close()

    def put_words(mut self, key: String, words: List[Int]) raises:
        self._open(key)
        self.values.append(UInt8(91))
        for i in range(len(words)):  # driver-loop: over one header list field (shard ids, loss bit words)
            if i != 0:
                self.values.append(UInt8(44))
            _header_append_text(self.values, String(words[i]))
        self.values.append(UInt8(93))
        self._close()

    def encode(self, payload_bytes: Int) raises -> List[UInt8]:
        """The header bytes for a frame whose payload is `payload_bytes` long:
        the fields by key, `nbytes` in its place among them."""
        if payload_bytes < 0:
            raise Error("transport: negative payload size")
        var text = List[UInt8]()
        text.append(UInt8(123))
        var placed = False
        var first = True
        for i in range(len(self.order) + 1):  # small-loop(fields of one header)
            var last = i == len(self.order)
            if not placed and (last or _header_key_less("nbytes", self.keys[self.order[i]])):
                if not first:
                    text.append(UInt8(44))
                _header_append_text(text, "\"nbytes\":" + String(payload_bytes))
                placed = True
                first = False
            if last:
                break
            var f = self.order[i]
            if not first:
                text.append(UInt8(44))
            first = False
            text.append(UInt8(34))
            _header_append_text(text, self.keys[f])
            text.append(UInt8(34))
            text.append(UInt8(58))
            for b in range(self.starts[f], self.ends[f]):  # small-loop(bytes of one encoded field)
                text.append(self.values[b])
        text.append(UInt8(125))
        if len(text) > TRANSPORT_HEADER_LIMIT:
            raise Error("transport: oversized header")
        return text^


def _header_space(b: Int) -> Bool:
    return b == 32 or b == 9 or b == 10 or b == 13


def _header_string_end(text: _HeaderBytes, count: Int, start: Int) raises -> Int:
    """`start` is the opening quote; the index of the closing one."""
    var i = start + 1
    while i < count:  # small-loop(bytes of one header string)
        var b = Int(text[i])
        if b == 92:
            i += 2
        elif b == 34:
            return i
        else:
            i += 1
    raise Error("transport: framed header has an unterminated string")


def transport_header_find(address: Int, count: Int, key: String) raises -> Tuple[Int, Int]:
    """The byte range [start, end) of the value of the top-level field `key`
    in the header object at `address`, or (-1, -1) when it has no such field."""
    var text = _HeaderBytes(unsafe_from_address=address)
    var name = key.as_bytes()
    var i = 0
    while i < count and _header_space(Int(text[i])):  # small-loop(leading blanks)
        i += 1
    if i >= count or Int(text[i]) != 123:
        raise Error("transport: framed header must be an object")
    i += 1
    while True:  # small-loop(fields of one header)
        while i < count and (_header_space(Int(text[i])) or Int(text[i]) == 44):  # small-loop(separators)
            i += 1
        if i >= count:
            raise Error("transport: framed header is truncated")
        if Int(text[i]) == 125:
            return (-1, -1)
        if Int(text[i]) != 34:
            raise Error("transport: framed header field has no name")
        var close = _header_string_end(text, count, i)
        var matched = close - i - 1 == len(name)
        if matched:
            for k in range(len(name)):  # small-loop(bytes of one short header key)
                if text[i + 1 + k] != name[k]:
                    matched = False
                    break
        i = close + 1
        while i < count and _header_space(Int(text[i])):  # small-loop(blanks)
            i += 1
        if i >= count or Int(text[i]) != 58:
            raise Error("transport: framed header field has no value")
        i += 1
        while i < count and _header_space(Int(text[i])):  # small-loop(blanks)
            i += 1
        var start = i
        var depth = 0
        while i < count:  # small-loop(bytes of one header value)
            var b = Int(text[i])
            if b == 34:
                i = _header_string_end(text, count, i) + 1
            elif b == 91 or b == 123:
                depth += 1
                i += 1
            elif b == 93 or b == 125:
                if depth == 0:
                    break
                depth -= 1
                i += 1
            elif b == 44 and depth == 0:
                break
            else:
                i += 1
        if i >= count:
            raise Error("transport: framed header is truncated")
        if matched:
            var end = i
            while end > start and _header_space(Int(text[end - 1])):  # small-loop(trailing blanks)
                end -= 1
            return (start, end)


def _header_integer(text: _HeaderBytes, start: Int, end: Int, key: String) raises -> Int:
    var i = start
    var negative = i < end and Int(text[i]) == 45
    if negative:
        i += 1
    if i >= end or end - i > 19:
        raise Error("transport: header field " + key + " is not an integer")
    # The whole signed 64-bit range reads back (a float's bit word has up to
    # 19 digits). The digits accumulate as a negative value, whose range is
    # the wider one; a value outside 64 bits is refused, never wrapped.
    var value = 0
    while i < end:  # small-loop(digits of one integer)
        var d = Int(text[i]) - 48
        if d < 0 or d > 9:
            raise Error("transport: header field " + key + " is not an integer")
        if value < -922337203685477580 or (value == -922337203685477580 and d > 8):
            raise Error("transport: header field " + key + " is not an integer")
        value = value * 10 - d
        i += 1
    if negative:
        return value
    if value < -9223372036854775807:
        raise Error("transport: header field " + key + " is not an integer")
    return -value


def transport_header_has(address: Int, count: Int, key: String) raises -> Bool:
    return transport_header_find(address, count, key)[0] >= 0


def transport_header_int(address: Int, count: Int, key: String) raises -> Int:
    var span = transport_header_find(address, count, key)
    if span[0] < 0:
        raise Error("transport: header has no field " + key)
    return _header_integer(_HeaderBytes(unsafe_from_address=address), span[0], span[1], key)


def transport_header_payload_bytes(address: Int, count: Int) raises -> Int:
    """The frame's payload length: its `nbytes` field, 0 when absent."""
    var span = transport_header_find(address, count, "nbytes")
    if span[0] < 0:
        return 0
    return _header_integer(_HeaderBytes(unsafe_from_address=address), span[0], span[1], "nbytes")


def transport_header_bool(address: Int, count: Int, key: String) raises -> Bool:
    var span = transport_header_find(address, count, key)
    var text = _HeaderBytes(unsafe_from_address=address)
    if span[1] - span[0] == 4 and Int(text[span[0]]) == 116:
        return True
    if span[1] - span[0] == 5 and Int(text[span[0]]) == 102:
        return False
    raise Error("transport: header field " + key + " is not a boolean")


def transport_header_words(address: Int, count: Int, key: String) raises -> List[Int]:
    var span = transport_header_find(address, count, key)
    var text = _HeaderBytes(unsafe_from_address=address)
    if span[0] < 0 or span[1] - span[0] < 2 or Int(text[span[0]]) != 91 or Int(text[span[1] - 1]) != 93:
        raise Error("transport: header field " + key + " is not a list of integers")
    var words = List[Int]()
    var i = span[0] + 1
    var end = span[1] - 1
    while i < end:  # driver-loop: over one header list field (shard ids, loss bit words)
        while i < end and (_header_space(Int(text[i])) or Int(text[i]) == 44):  # small-loop(separators)
            i += 1
        if i >= end:
            break
        var stop = i
        while stop < end and Int(text[stop]) != 44 and not _header_space(Int(text[stop])):  # small-loop(digits of one integer)
            stop += 1
        words.append(_header_integer(text, i, stop, key))
        i = stop
    return words^


def _header_hex_value(b: Int) raises -> Int:
    if b >= 48 and b <= 57:
        return b - 48
    if b >= 97 and b <= 102:
        return b - 87
    if b >= 65 and b <= 70:
        return b - 55
    raise Error("transport: header text has a bad escape")


def _header_append_point(mut dst: List[UInt8], point: Int):
    if point < 128:
        dst.append(UInt8(point))
    elif point < 2048:
        dst.append(UInt8(192 | (point >> 6)))
        dst.append(UInt8(128 | (point & 63)))
    elif point < 65536:
        dst.append(UInt8(224 | (point >> 12)))
        dst.append(UInt8(128 | ((point >> 6) & 63)))
        dst.append(UInt8(128 | (point & 63)))
    else:
        dst.append(UInt8(240 | (point >> 18)))
        dst.append(UInt8(128 | ((point >> 12) & 63)))
        dst.append(UInt8(128 | ((point >> 6) & 63)))
        dst.append(UInt8(128 | (point & 63)))


def transport_header_text(address: Int, count: Int, key: String) raises -> String:
    var span = transport_header_find(address, count, key)
    var text = _HeaderBytes(unsafe_from_address=address)
    if span[0] < 0 or span[1] - span[0] < 2 or Int(text[span[0]]) != 34 or Int(text[span[1] - 1]) != 34:
        raise Error("transport: header field " + key + " is not text")
    var raw = List[UInt8]()
    var i = span[0] + 1
    var end = span[1] - 1
    while i < end:  # small-loop(bytes of one short header text)
        var b = Int(text[i])
        if b != 92:
            raw.append(UInt8(b))
            i += 1
            continue
        if i + 1 >= end:
            raise Error("transport: header text has a bad escape")
        var e = Int(text[i + 1])
        i += 2
        if e == 117:
            if i + 4 > end:
                raise Error("transport: header text has a bad escape")
            var unit = 0
            for k in range(4):  # small-loop(the four hex digits of one escape)
                unit = (unit << 4) | _header_hex_value(Int(text[i + k]))
            i += 4
            if unit >= 55296 and unit < 56320:
                if i + 6 > end or Int(text[i]) != 92 or Int(text[i + 1]) != 117:
                    raise Error("transport: header text has a lone surrogate")
                var low = 0
                for k in range(4):  # small-loop(the four hex digits of one escape)
                    low = (low << 4) | _header_hex_value(Int(text[i + 2 + k]))
                if low < 56320 or low > 57343:
                    raise Error("transport: header text has a lone surrogate")
                i += 6
                unit = 65536 + ((unit - 55296) << 10) + (low - 56320)
            elif unit >= 56320 and unit <= 57343:
                raise Error("transport: header text has a lone surrogate")
            _header_append_point(raw, unit)
        elif e == 110:
            raw.append(UInt8(10))
        elif e == 114:
            raw.append(UInt8(13))
        elif e == 116:
            raw.append(UInt8(9))
        elif e == 98:
            raw.append(UInt8(8))
        elif e == 102:
            raw.append(UInt8(12))
        elif e == 34 or e == 92 or e == 47:
            raw.append(UInt8(e))
        else:
            raise Error("transport: header text has a bad escape")
    if len(raw) == 0:
        return String("")
    var bytes = Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(raw.unsafe_ptr())), length=len(raw))
    var result = String(from_utf8=bytes)
    # Keep the byte list alive past the copy into the string.
    _ = len(raw)
    return result
