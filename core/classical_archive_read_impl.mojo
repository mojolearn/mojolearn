# SPDX-License-Identifier: Apache-2.0
"""Typed NPY header byte scans moved word for word from bindings/classical_archive_read.mojo.

The header bytes arrive as a List[UInt8]; each scan returns the new offset."""


def npy_latin1_bytes(text: String) -> List[UInt8]:
    """The bytes of a latin-1 decoded header: every codepoint is one byte (0..255)."""
    var out = List[UInt8]()
    for cp in text.codepoints():
        out.append(UInt8(Int(cp)))
    return out^


def npy_peek(raw: List[UInt8], at: Int) -> Int:
    return Int(raw[at]) if at < len(raw) else -1


def npy_skip_ws(raw: List[UInt8], start: Int) -> Int:
    var at = start
    while npy_peek(raw, at) == 32 or npy_peek(raw, at) == 9 or npy_peek(raw, at) == 10 or npy_peek(raw, at) == 13:
        at += 1
    return at


def npy_string_close(raw: List[UInt8], start: Int, quote: Int) -> Int:
    """The offset of the closing quote of a string opened before `start`, or -1 when unterminated."""
    var at = start
    while npy_peek(raw, at) >= 0:
        if npy_peek(raw, at) == quote:
            return at
        if npy_peek(raw, at) == 92:
            at += 1
        at += 1
    return -1


def npy_digits_end(raw: List[UInt8], start: Int) -> Int:
    var at = start
    while npy_peek(raw, at) >= 48 and npy_peek(raw, at) <= 57:
        at += 1
    return at
