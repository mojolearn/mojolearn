# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPT-2's byte-to-unicode bijection: every one of the 256 byte values as a
printable, non-space codepoint.

WHY IT EXISTS. GPT-2's published vocabulary is a JSON file of TEXT keys, and a
merge table is a text file, so the training pipeline needed every byte to be
spellable as a character that neither a JSON reader nor a whitespace-splitting
merge-file reader could mangle. The recipe in the original `encoder.py` takes
the three runs of already-printable Latin-1 codepoints

    0x21..0x7E  ('!' .. '~')        0xA1..0xAC  ('.'..'-')        0xAE..0xFF

as fixed points and maps the remaining 68 bytes -- the C0 controls, space,
DEL, the C1 controls, NBSP (0xA0) and the soft hyphen (0xAD) -- to
U+0100, U+0101, ... U+0143 in ASCENDING BYTE ORDER. Nothing else is assigned,
so the image is 256 distinct codepoints and the map is a bijection.

WHAT THIS MODULE IS FOR HERE. `tokenizer/data/gpt2_ranks.tsv` gives token
bytes as HEX, not as this spelling, so the rank lookup does NOT need the
bijection: `ranks.mojo` hashes raw bytes. The bijection is still the only
readable way to NAME a token whose bytes are not valid UTF-8 on their own
(half of a multi-byte character, a lone 0x80), and that is what a failure
report needs, so `spell_bytes` is what the check prints and
`check_byte_unicode_bijection` gates it both ways.

HOST ONLY, integers and tables. No float, no device, nothing to be identical
about across vendors beyond the table itself.
"""


def byte_to_codepoint() -> List[Int]:
    """256 entries: byte value -> the codepoint that spells it.

    Built from the recipe rather than pasted, so the three runs and the
    ascending assignment are visible and a transcription typo is impossible.
    """
    var is_fixed = List[Bool]()
    for _ in range(256):
        is_fixed.append(False)
    for b in range(0x21, 0x7F):
        is_fixed[b] = True
    for b in range(0xA1, 0xAD):
        is_fixed[b] = True
    for b in range(0xAE, 0x100):
        is_fixed[b] = True

    var table = List[Int]()
    for _ in range(256):
        table.append(-1)
    var next_extra = 256
    for b in range(256):
        if is_fixed[b]:
            table[b] = b
        else:
            table[b] = next_extra
            next_extra += 1
    return table^


def codepoint_to_byte() -> List[Int]:
    """The inverse, as a dense table indexed by codepoint; -1 where the
    codepoint is not in the image. Length is one past the largest codepoint
    the bijection uses (U+0143 for GPT-2's 68 non-printable bytes)."""
    var fwd = byte_to_codepoint()
    var top = 0
    for b in range(256):
        if fwd[b] > top:
            top = fwd[b]
    var rev = List[Int]()
    for _ in range(top + 1):
        rev.append(-1)
    for b in range(256):
        rev[fwd[b]] = b
    return rev^


def utf8_append(mut out: List[UInt8], cp: Int):
    """Append `cp` as UTF-8. The bijection's image tops out at U+0143, so one
    and two byte forms are all this needs; a three or four byte codepoint
    cannot arrive here and would be a table bug, not an input."""
    if cp < 0x80:
        out.append(UInt8(cp))
    else:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def spell_bytes(data: List[UInt8], start: Int, count: Int) -> String:
    """The GPT-2 vocabulary spelling of `data[start : start + count]`.

    Always valid UTF-8 and always printable, whatever the bytes were: that is
    the whole point of the bijection. Safe to put in an error message.
    """
    var fwd = byte_to_codepoint()
    var buf = List[UInt8]()
    for i in range(start, start + count):
        utf8_append(buf, fwd[Int(data[i])])
    var s = String(StringSlice(unsafe_from_utf8=Span(buf)))
    # `[[mojo-buffer-freed-at-last-use]]`: the slice views `buf`, so `buf`
    # has to outlive the String construction above.
    _ = buf
    return s^
