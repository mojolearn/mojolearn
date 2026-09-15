# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The three character classes the GPT-2 pre-tokenizer pattern names, and the
UTF-8 decoder that feeds them.

Mojo has NO regex engine, so `tokenizer/impl/pretokenize.mojo` walks the
pattern by hand and needs exactly three predicates:

    \\p{L}  -- general category Letter (Lu Ll Lt Lm Lo). MARKS ARE NOT
              LETTERS: that is the Rust `regex` / `fancy-regex` meaning of
              `\\p{L}`, the syntax tiktoken compiles this pattern with, and
              it is why "e" followed by a combining acute pre-tokenizes as
              two pre-tokens while a precomposed U+00E9 is one.
    \\p{N}  -- general category Number (Nd Nl No). Not just ASCII digits.
    \\s    -- the Unicode WHITE_SPACE PROPERTY. Not Python's
              `str.isspace()`, which also answers True for U+001C..U+001F.

THE RANGES ARE GENERATED AT BUILD TIME, NOT TRACKED (2026-09-15).
`tokenizer/tools/gen_unicode_table.sh` runs
`tokenizer/tools/gen_unicode_categories.py`, which computes them from
Python's standard `unicodedata` at the version pinned below and writes
`tokenizer/impl/unicode_table_generated.mojo` (gitignored): the table as one
string constant, ascending and disjoint per class, so membership is a binary
search. The generator refuses to write a table whose Unicode version or
sha256 is not the pin, and `builtin_unicode_classes` refuses a generated
module whose constants disagree with it, so every build compiles the same
bytes. A codepoint whose category differs between Unicode versions is the
one way two pins could tokenize differently, which is why there is a pin.
"""

from tokenizer.impl.unicode_table_generated import (
    UNICODE_TABLE_SHA256,
    UNICODE_TABLE_TEXT,
    UNICODE_TABLE_VERSION,
)

comptime UNICODE_VERSION_PINNED = "16.0.0"
comptime UNICODE_TABLE_SHA256_PINNED = "ad4e106533e0f7efb20eb3f0bd7931e21ebf638415b525918e81d3abc104a339"


struct UnicodeClasses(Copyable, Movable):
    """Three flat ascending range tables: `[lo0, hi0, lo1, hi1, ...]`."""

    var letters: List[Int]
    var numbers: List[Int]
    var spaces: List[Int]
    var source_header: String
    """The generator's `# unicodedata <version>` line, printed by the check."""

    def __init__(out self):
        self.letters = List[Int]()
        self.numbers = List[Int]()
        self.spaces = List[Int]()
        self.source_header = String("")

    def _in(self, table: List[Int], cp: Int) -> Bool:
        var lo = 0
        var hi = len(table) // 2 - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            if cp < table[2 * mid]:
                hi = mid - 1
            elif cp > table[2 * mid + 1]:
                lo = mid + 1
            else:
                return True
        return False

    def is_letter(self, cp: Int) -> Bool:
        return self._in(self.letters, cp)

    def is_number(self, cp: Int) -> Bool:
        return self._in(self.numbers, cp)

    def is_space(self, cp: Int) -> Bool:
        return self._in(self.spaces, cp)

    def is_other(self, cp: Int) -> Bool:
        """`[^\\s\\p{L}\\p{N}]`, the pattern's fourth alternative's class."""
        return not (
            self.is_space(cp) or self.is_letter(cp) or self.is_number(cp)
        )


def _hex_digit(b: UInt8) raises -> Int:
    var v = Int(b)
    if v >= 48 and v <= 57:  # '0'..'9'
        return v - 48
    if v >= 97 and v <= 102:  # 'a'..'f'
        return v - 97 + 10
    if v >= 65 and v <= 70:  # 'A'..'F'
        return v - 65 + 10
    raise Error("unicode_categories.tsv: bad hex digit: " + String(v))


def _parse_hex(tok: StringSlice) raises -> Int:
    var v = 0
    var n = 0
    for b in tok.as_bytes():
        v = (v << 4) | _hex_digit(b)
        n += 1
    if n == 0:
        raise Error("unicode_categories.tsv: empty hex field")
    return v


def builtin_unicode_classes() raises -> UnicodeClasses:
    """The classes compiled into this build from the generated module,
    refused by name when its version or sha256 is not the pin (a stale
    generated file from an older pin)."""
    if String(UNICODE_TABLE_VERSION) != String(UNICODE_VERSION_PINNED) or String(
        UNICODE_TABLE_SHA256
    ) != String(UNICODE_TABLE_SHA256_PINNED):
        raise Error(
            "unicode classes: tokenizer/impl/unicode_table_generated.mojo is unicodedata "
            + String(UNICODE_TABLE_VERSION)
            + " sha256 "
            + String(UNICODE_TABLE_SHA256)
            + ", not the pinned "
            + String(UNICODE_VERSION_PINNED)
            + " "
            + String(UNICODE_TABLE_SHA256_PINNED)
            + "; rerun sh tokenizer/tools/gen_unicode_table.sh and rebuild"
        )
    return parse_unicode_classes(
        String(UNICODE_TABLE_TEXT), String("unicode_table_generated")
    )


def load_unicode_classes(path: String) raises -> UnicodeClasses:
    """A range table from a file in the same text format, for tools."""
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return parse_unicode_classes(text, path)


def parse_unicode_classes(text: String, path: String) raises -> UnicodeClasses:
    """Parse the range table text. Refuses a non-ascending or overlapping
    table BY POSITION rather than trusting the generator, because the binary
    search above is only correct on a sorted disjoint table and a silently
    wrong membership test would show up as a tokenizer mismatch a long way
    from here. `path` only labels the refusals."""
    var out = UnicodeClasses()
    var row = 0
    for line_slice in text.split("\n"):
        var line = String(line_slice)
        row += 1
        if line.byte_length() == 0:
            continue
        if line[byte=0] == "#":
            if out.source_header.byte_length() == 0 and (
                line.find("unicodedata") >= 0
            ):
                out.source_header = line
            continue
        var fields = line.split("\t")
        if len(fields) != 3:
            raise Error(
                path
                + ":"
                + String(row)
                + ": expected 'class<TAB>first_hex<TAB>last_hex', got "
                + String(len(fields))
                + " fields"
            )
        var first = _parse_hex(fields[1])
        var last = _parse_hex(fields[2])
        if last < first or last > 0x10FFFF:
            raise Error(
                path + ":" + String(row) + ": range is not a codepoint range"
            )
        var cls = String(fields[0])
        if cls == "L":
            _append_range(out.letters, first, last, path, row)
        elif cls == "N":
            _append_range(out.numbers, first, last, path, row)
        elif cls == "WS":
            _append_range(out.spaces, first, last, path, row)
        else:
            raise Error(
                path
                + ":"
                + String(row)
                + ": class '"
                + cls
                + "' is not one of L, N, WS"
            )
    if (
        len(out.letters) == 0
        or len(out.numbers) == 0
        or len(out.spaces) == 0
    ):
        raise Error(path + ": one of the three classes is empty")
    return out^


def _append_range(
    mut table: List[Int], first: Int, last: Int, path: String, row: Int
) raises:
    if len(table) > 0 and first <= table[len(table) - 1]:
        raise Error(
            path
            + ":"
            + String(row)
            + ": ranges must be ascending and disjoint within a class"
        )
    table.append(first)
    table.append(last)


def decode_codepoint(data: List[UInt8], i: Int) raises -> Tuple[Int, Int]:
    """`(codepoint, width)` at byte offset `i`, or `(-1, 1)` for a byte that
    does not begin a well-formed UTF-8 sequence.

    Well-formed means the Unicode definition, not just "the right number of
    continuation bytes": the overlong forms, the surrogate range
    (U+D800..U+DFFF) and everything above U+10FFFF are rejected by the
    per-lead-byte bounds below. tiktoken cannot meet an invalid sequence at
    all -- its input is a Rust `&str` -- so the `-1` path is ours alone and
    `NOT_IMPLEMENTED.tsv` records what it does with it.
    """
    var n = len(data)
    if i < 0 or i >= n:
        raise Error("decode_codepoint: offset out of range")
    var c = Int(data[i])
    if c < 0x80:
        return (c, 1)
    if c >= 0xC2 and c <= 0xDF and i + 1 < n:
        var b1 = Int(data[i + 1])
        if b1 >= 0x80 and b1 <= 0xBF:
            return (((c & 0x1F) << 6) | (b1 & 0x3F), 2)
        return (-1, 1)
    if c >= 0xE0 and c <= 0xEF and i + 2 < n:
        var b1 = Int(data[i + 1])
        var b2 = Int(data[i + 2])
        var lo = 0xA0 if c == 0xE0 else 0x80
        var hi = 0x9F if c == 0xED else 0xBF
        if b1 >= lo and b1 <= hi and b2 >= 0x80 and b2 <= 0xBF:
            return (
                ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F),
                3,
            )
        return (-1, 1)
    if c >= 0xF0 and c <= 0xF4 and i + 3 < n:
        var b1 = Int(data[i + 1])
        var b2 = Int(data[i + 2])
        var b3 = Int(data[i + 3])
        var lo = 0x90 if c == 0xF0 else 0x80
        var hi = 0x8F if c == 0xF4 else 0xBF
        if (
            b1 >= lo
            and b1 <= hi
            and b2 >= 0x80
            and b2 <= 0xBF
            and b3 >= 0x80
            and b3 <= 0xBF
        ):
            return (
                ((c & 0x07) << 18)
                | ((b1 & 0x3F) << 12)
                | ((b2 & 0x3F) << 6)
                | (b3 & 0x3F),
                4,
            )
        return (-1, 1)
    return (-1, 1)
