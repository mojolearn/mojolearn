# SPDX-License-Identifier: Apache-2.0
"""Typed JSON byte scans moved from bindings/native_json.mojo. Each scan reads the
document bytes and returns the new offset; the shell builds the Python containers."""


def json_latin1_bytes(text: String) -> List[UInt8]:
    """The bytes of a latin-1 decoded document: every codepoint is one byte (0..255)."""
    var out = List[UInt8]()
    for cp in text.codepoints():
        out.append(UInt8(Int(cp)))
    return out^


def json_peek(raw: List[UInt8], at: Int) -> Int:
    return Int(raw[at]) if at < len(raw) else -1


def json_skip_ws(raw: List[UInt8], start: Int) -> Int:
    var at = start
    while json_peek(raw, at) == 32 or json_peek(raw, at) == 9 or json_peek(raw, at) == 10 or json_peek(raw, at) == 13:
        at += 1
    return at


def json_digits_end(raw: List[UInt8], start: Int) -> Int:
    var at = start
    while json_peek(raw, at) >= 48 and json_peek(raw, at) <= 57:
        at += 1
    return at


def json_plain_end(raw: List[UInt8], start: Int) -> Int:
    """The end of a run of string bytes that need no escape handling: stops at a quote,
    a backslash, a control byte or the end of the document."""
    var at = start
    while json_peek(raw, at) >= 32 and json_peek(raw, at) != 34 and json_peek(raw, at) != 92:
        at += 1
    return at
