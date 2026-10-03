# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Building and checking a rank table from token bytes or byte-level
spellings (lane/pyglue-text-io, 2026-10-03).

These are the checks `python/mojolearn/tokenizer.py` (`from_token_bytes`,
`from_files`) and `python/mojolearn/models/tokenizer.py` (`_validate`) ran
in Python, moved here so the Python doors only pass the caller's objects:

  * every token non-empty and unique, and all 256 single bytes present
    (byte-level BPE needs every byte);
  * a byte-level spelling decodes through the GPT-2 byte-to-unicode
    bijection (`byte_unicode.mojo`);
  * every merge joins two tokens, its result is a token, and merge results
    take strictly increasing ids above both parts -- the conditions under
    which merging by rank gives the merge list's own result;
  * the LOOSE scan: every token no merge makes is a single byte (skipped
    when the file says `model.ignore_merges`).

Refusals start with the Python exception name they are re-raised as
(`ValueError: `); the Python door strips it and prefixes the file.
"""

from tokenizer.impl.byte_unicode import codepoint_to_byte, spell_bytes
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import decode_codepoint

comptime _HEX = "0123456789abcdef"


def hex_of(data: List[UInt8], start: Int, count: Int) -> String:
    var d = String(_HEX).as_bytes()
    var out = String("")
    for i in range(start, start + count):
        var b = Int(data[i])
        out += String(chr(Int(d[b >> 4])))
        out += String(chr(Int(d[b & 15])))
    return out^


def hex2(b: Int) -> String:
    var one = List[UInt8]()
    one.append(UInt8(b))
    return hex_of(one, 0, 1)


def table_from_tokens(arena: List[UInt8], lengths: List[Int]) raises -> RankTable:
    """A rank table from token bytes back to back in rank order (rank = id).
    Refuses an empty token, a repeated token and a missing single byte."""
    var n = len(lengths)
    var table = RankTable()
    table._reserve(n)
    var at = 0
    for k in range(n):
        var m = lengths[k]
        if m <= 0:
            raise Error("ValueError: token " + String(k) + " is empty")
        if at + m > len(arena):
            raise Error("ValueError: token " + String(k) + " runs past the token bytes")
        var other = table.rank(arena, at, m)
        if other >= 0:
            raise Error(
                "ValueError: token "
                + String(k)
                + " repeats token "
                + String(other)
                + " ("
                + hex_of(arena, at, m)
                + ")"
            )
        table.offset.append(len(table.arena))
        table.length.append(m)
        for i in range(m):
            table.arena.append(arena[at + i])
        table._insert(k)
        at += m
    var missing = 0
    var first = -1
    var one = List[UInt8](length=1, fill=UInt8(0))
    for b in range(256):
        one[0] = UInt8(b)
        if table.rank(one, 0, 1) < 0:
            missing += 1
            if first < 0:
                first = b
    if missing > 0:
        raise Error(
            "ValueError: the vocabulary lacks "
            + String(missing)
            + " of the 256 single-byte tokens (first 0x"
            + hex2(first)
            + "); byte-level BPE needs every byte"
        )
    return table^


def unspell(
    spelling: List[UInt8], inv: List[Int], mut out: List[UInt8]
) raises -> Int:
    """Append the bytes a byte-level spelling (UTF-8) names to `out`.
    Returns -1, or the first codepoint that is not in the bijection's image
    (-2 for a byte that begins no UTF-8 sequence); `out` is then partial."""
    var i = 0
    var n = len(spelling)
    while i < n:
        var s = decode_codepoint(spelling, i)
        var cp = s[0]
        if cp < 0:
            return -2
        if cp >= len(inv) or inv[cp] < 0:
            return cp
        out.append(UInt8(inv[cp]))
        i += s[1]
    return -1


def codepoint_text(cp: Int) -> String:
    """`'c' (U+XXXX)` for an error message."""
    if cp < 0:
        return String("an invalid UTF-8 byte")
    var d = String(_HEX).as_bytes()
    var u = String("")
    var started = False
    for shift in range(20, -1, -4):
        var nib = (cp >> shift) & 15
        if nib != 0 or started or shift <= 12:
            started = True
            u += String(chr(Int(d[nib])))
    return "'" + String(chr(cp)) + "' (U+" + u.upper() + ")"


def unspell_error(shown: String, cp: Int) -> String:
    """`shown` is the spelling as the caller wrote it (its Python repr)."""
    var why = String("not a byte-level spelling")
    if cp == 0x2581:
        why = "a SentencePiece spelling (▁), refused by name"
    return (
        "ValueError: "
        + shown
        + " holds "
        + codepoint_text(cp)
        + ", "
        + why
    )


struct MergeCheck(Movable):
    """The running state of the merge-list check: the previous result id
    and which ids a merge made."""

    var last: Int
    var made: List[Bool]
    var count: Int

    def __init__(out self, n_tokens: Int):
        self.last = -1
        self.made = List[Bool](length=n_tokens, fill=False)
        self.count = 0


def check_merge(
    table: RankTable,
    a: List[UInt8],
    b: List[UInt8],
    label: String,
    mut chk: MergeCheck,
) raises:
    """One merge `a + b`, refused by `label` unless both parts are tokens,
    the result is a token, and its id rises above the previous merge's and
    both parts'."""
    var ia = table.rank(a, 0, len(a))
    var ib = table.rank(b, 0, len(b))
    if ia < 0 or ib < 0:
        raise Error(
            "ValueError: "
            + label
            + " joins "
            + spell_bytes(a, 0, len(a))
            + " and "
            + spell_bytes(b, 0, len(b))
            + ", not both tokens"
        )
    var ab = List[UInt8](capacity=len(a) + len(b))
    for i in range(len(a)):
        ab.append(a[i])
    for i in range(len(b)):
        ab.append(b[i])
    var m = table.rank(ab, 0, len(ab))
    if m < 0:
        raise Error(
            "ValueError: "
            + label
            + ": the merge result "
            + spell_bytes(ab, 0, len(ab))
            + " is not a token"
        )
    if m <= chk.last or m <= ia or m <= ib:
        raise Error(
            "ValueError: "
            + label
            + ": merge result id "
            + String(m)
            + " does not rise above the previous merge ("
            + String(chk.last)
            + ") and both parts; merging by rank would not follow this"
            " merge list"
        )
    chk.last = m
    chk.made[m] = True
    chk.count += 1


def check_loose(table: RankTable, chk: MergeCheck, ignore_merges: Bool) raises:
    """Every token no merge makes is a single byte, unless the file's
    `ignore_merges` lets a whole-word lookup reach it."""
    if ignore_merges:
        return
    for i in range(table.n_tokens()):
        if not chk.made[i] and table.length[i] != 1:
            raise Error(
                "ValueError: id "
                + String(i)
                + " is neither a single byte nor made by a merge"
            )


def split_merge_line(line: List[UInt8], mut a: List[UInt8], mut b: List[UInt8]) -> Bool:
    """`a b` (one space, two non-empty parts) into the two spellings."""
    var cut = -1
    for i in range(len(line)):
        if Int(line[i]) == 0x20:
            if cut >= 0:
                return False
            cut = i
    if cut <= 0 or cut + 1 >= len(line):
        return False
    for i in range(cut):
        a.append(line[i])
    for i in range(cut + 1, len(line)):
        b.append(line[i])
    return True


def ranks_text(table: RankTable) -> String:
    """The canonical rank file: `rank<TAB>lowercase hex` per line, the text
    `TrainedBpeVocabulary.render_ranks` writes; the vocabulary identity is
    the SHA-256 of it."""
    var s = String("")
    for id in range(table.n_tokens()):
        s += String(id)
        s += "\t"
        s += hex_of(table.arena, table.offset[id], table.length[id])
        s += "\n"
    return s^
