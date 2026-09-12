# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public Mojo surface: GPT-2 byte-level BPE `encode` and `decode`.

    var tok = load_gpt2_tokenizer()
    var ids = tok.encode("hello world", False)
    var back = tok.decode(ids)

WHAT THE PROPERTY IS. This is HOST-ONLY integer and table work: a byte
buffer, three range tables, a hash probe and an argmin over small integers.
There is no floating-point arithmetic and no device kernel anywhere in
`tokenizer/`, so cross-vendor bitwise identity is true BY CONSTRUCTION and is
not a claim worth making. The property that has to be earned is EXACT
AGREEMENT WITH THE REFERENCE IMPLEMENTATION -- the same id sequence as
tiktoken 0.14.0's `gpt2` encoding, for the same text, id for id -- and
`tokenizer/checks/tokenizer_check.mojo` is where that is asserted against 43
recorded cases.

THE SPECIAL TOKEN. `<|endoftext|>` is id 50256 and is not in the rank table.
It is a token ONLY when the caller passes `allow_endoftext=True`; otherwise
the thirteen characters are ordinary text and encode as seven ids. Both
readings are in the fixture (`endoftext_as_text`, `endoftext_as_special`),
which is why the flag is an argument and not a policy. There is no
"error on encountering a special token" mode: tiktoken's `encode` has one and
`NOT_IMPLEMENTED.tsv` records its absence.

BYTES, NOT STRINGS, ARE THE INTERFACE. `encode_bytes` / `decode_bytes` are
the real entry points. A GPT-2 token's bytes need not be valid UTF-8 on their
own, and a caller's text may hold NUL (the fixture's `raw_bytes` case does),
so the `String` wrappers are conveniences over the byte functions rather than
the other way round.
"""

from tokenizer.impl.bpe import bpe_append
from tokenizer.impl.byte_unicode import spell_bytes
from tokenizer.impl.pretokenize import pretokenize
from tokenizer.impl.ranks import RankTable, load_rank_table
from tokenizer.impl.unicode_class import (
    UnicodeClasses,
    load_unicode_classes,
)

comptime GPT2_RANKS_PATH = "tokenizer/data/gpt2_ranks.tsv"
comptime GPT2_UNICODE_PATH = "tokenizer/data/unicode_categories.tsv"

comptime GPT2_N_VOCAB = 50257
"""50256 ranks plus `<|endoftext|>`. Asserted against the fixture."""

comptime GPT2_ENDOFTEXT_ID = 50256

comptime GPT2_ENDOFTEXT = "<|endoftext|>"

comptime GPT2_PAT_STR = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s"
"""The pattern `impl/pretokenize.mojo` implements by hand, spelled so it can
be compared to the fixture's own `pat_str`. THE CHECK ASSERTS THEY ARE EQUAL:
a fixture regenerated from a different pattern must fail the gate rather than
be silently tokenized by this one."""


def string_bytes(text: String) -> List[UInt8]:
    var s = String(text)
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        out.append(b[i])
    # `[[mojo-buffer-freed-at-last-use]]`: `b` views `s`.
    _ = s
    return out^


def bytes_string(data: List[UInt8]) -> String:
    """The bytes as a `String`, unvalidated on purpose: `decode` of a token
    stream cut in the middle of a character yields incomplete UTF-8 and the
    byte sequence is still the right answer. Byte-exact comparisons belong on
    `decode_bytes`."""
    var s = String(StringSlice(unsafe_from_utf8=Span(data)))
    _ = data
    return s^


struct Gpt2Tokenizer(Copyable, Movable):
    var ranks: RankTable
    var classes: UnicodeClasses
    var eot: List[UInt8]

    def __init__(
        out self, var ranks: RankTable, var classes: UnicodeClasses
    ):
        self.ranks = ranks^
        self.classes = classes^
        self.eot = string_bytes(String(GPT2_ENDOFTEXT))

    def n_vocab(self) -> Int:
        return self.ranks.n_tokens() + 1

    def encode_ordinary_bytes(self, text: List[UInt8]) raises -> List[Int]:
        """Pre-tokenize, then merge each pre-token. No special token is
        recognized: `<|endoftext|>` here is thirteen ordinary characters."""
        var out = List[Int]()
        var bounds = pretokenize(text, self.classes)
        for k in range(len(bounds) - 1):
            bpe_append(out, self.ranks, text, bounds[k], bounds[k + 1])
        return out^

    def encode_bytes(
        self, text: List[UInt8], allow_endoftext: Bool
    ) raises -> List[Int]:
        """`allow_endoftext=True` splits the text on the literal
        `<|endoftext|>` and emits 50256 for each occurrence.

        The segments between occurrences are encoded INDEPENDENTLY, which is
        not a detail: the pattern's `\\s++$` alternative means end of text,
        and a special token ends the text that precedes it. A segment is
        copied rather than encoded in place so that "end of text" is a
        property of the buffer the pre-tokenizer sees.
        """
        if not allow_endoftext:
            return self.encode_ordinary_bytes(text)
        var out = List[Int]()
        var i = 0
        while True:
            var hit = self._find_eot(text, i)
            if hit < 0:
                out += self.encode_ordinary_bytes(
                    _slice(text, i, len(text))
                )
                return out^
            out += self.encode_ordinary_bytes(_slice(text, i, hit))
            out.append(GPT2_ENDOFTEXT_ID)
            i = hit + len(self.eot)

    def _find_eot(self, text: List[UInt8], start: Int) -> Int:
        var n = len(text)
        var m = len(self.eot)
        if m == 0 or n < m:
            return -1
        for i in range(start, n - m + 1):
            var hit = True
            for k in range(m):
                if text[i + k] != self.eot[k]:
                    hit = False
                    break
            if hit:
                return i
        return -1

    def encode(self, text: String, allow_endoftext: Bool) raises -> List[Int]:
        return self.encode_bytes(string_bytes(text), allow_endoftext)

    def decode_bytes(self, ids: List[Int]) raises -> List[UInt8]:
        var out = List[UInt8]()
        for k in range(len(ids)):
            var id = ids[k]
            if id == GPT2_ENDOFTEXT_ID:
                for j in range(len(self.eot)):
                    out.append(self.eot[j])
            elif id >= 0 and id < self.ranks.n_tokens():
                self.ranks.append_token_bytes(out, id)
            else:
                raise Error(
                    "decode: id "
                    + String(id)
                    + " at position "
                    + String(k)
                    + " is outside [0, "
                    + String(self.n_vocab())
                    + ")"
                )
        return out^

    def decode(self, ids: List[Int]) raises -> String:
        return bytes_string(self.decode_bytes(ids))

    def token_spelling(self, id: Int) raises -> String:
        """The token as GPT-2's own printable vocabulary spelling, through the
        byte-to-unicode bijection. For reports: a token whose bytes are half a
        character has no readable form otherwise."""
        if id == GPT2_ENDOFTEXT_ID:
            return String(GPT2_ENDOFTEXT)
        var b = self.ranks.token_bytes(id)
        return spell_bytes(b, 0, len(b))


def _slice(data: List[UInt8], start: Int, end: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(data[i])
    return out^


def load_gpt2_tokenizer() raises -> Gpt2Tokenizer:
    """Load both tables from their repository-relative paths. Run checks and
    tools from the repository root, as every other `pixi run check-*` does."""
    return load_gpt2_tokenizer_from(
        String(GPT2_RANKS_PATH), String(GPT2_UNICODE_PATH)
    )


def load_gpt2_tokenizer_from(
    ranks_path: String, unicode_path: String
) raises -> Gpt2Tokenizer:
    var ranks = load_rank_table(ranks_path)
    var classes = load_unicode_classes(unicode_path)
    return Gpt2Tokenizer(ranks^, classes^)
