# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pretrained-vocabulary tokenizer `mojolearn.models.Tokenizer` drives
(lane/pyglue-text-io, 2026-10-03): byte-level BPE over a rank table with
the pre-tokenization pattern a PARAMETER (GPT-2, Llama 3 or Qwen 2) and the
file's added (special) tokens.

Before this lane the Llama 3 / Qwen 2 cut, the merge loop, the special-token
split and the decode ran in Python (`models/tokenizer.py`); they are here
now, over the same pinned Unicode tables and the same merge loop
(`impl/bpe.mojo`) the GPT-2 door uses, so a GPT-2 `Tokenizer` and
`BpeTokenizer` give the same ids by construction.

SPECIAL TOKENS. With `allow_special` the text is split at the added tokens'
contents, leftmost first, the LONGER content winning at a tie in position
(then the smaller bytes); the segments between them are encoded alone (a
special token ends the text before it). Without it their characters are
ordinary text. `decode` renders an added token's content; an added id wins
over a rank with the same id.
"""

from tokenizer.impl.bpe import bpe_append
from tokenizer.impl.pretokenize import pretokenize_pattern
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import UnicodeClasses


struct VocabTokenizer(Movable):
    var ranks: RankTable
    var classes: UnicodeClasses
    var pattern: Int
    var sp_arena: List[UInt8]
    var sp_offset: List[Int]
    var sp_length: List[Int]
    var sp_id: List[Int]
    var added_of: List[Int]
    """id -> index into the specials, or -1; length `n_vocab()`."""

    def __init__(
        out self,
        var ranks: RankTable,
        var classes: UnicodeClasses,
        pattern: Int,
    ):
        self.ranks = ranks^
        self.classes = classes^
        self.pattern = pattern
        self.sp_arena = List[UInt8]()
        self.sp_offset = List[Int]()
        self.sp_length = List[Int]()
        self.sp_id = List[Int]()
        self.added_of = List[Int]()

    def set_specials(
        mut self, arena: List[UInt8], lengths: List[Int], ids: List[Int]
    ) raises:
        """The added tokens, `(content bytes, id)`, stored longest first and
        then by bytes, the order the split prefers at a tie in position."""
        var n = len(lengths)
        var offs = List[Int](capacity=n)
        var at = 0
        for k in range(n):
            if lengths[k] <= 0 or ids[k] < 0:
                raise Error(
                    "ValueError: added token "
                    + String(k)
                    + " is not a non-empty content to a non-negative id"
                )
            offs.append(at)
            at += lengths[k]
        # insertion sort of the indices by (-length, bytes)
        var order = List[Int](capacity=n)
        for k in range(n):
            var j = len(order)
            order.append(k)
            while j > 0 and _before(arena, offs, lengths, k, order[j - 1]):
                order[j] = order[j - 1]
                j -= 1
            order[j] = k
        var top = self.ranks.n_tokens()
        for k in range(n):
            if ids[k] + 1 > top:
                top = ids[k] + 1
        self.added_of = List[Int](length=top, fill=-1)
        for r in range(n):
            var k = order[r]
            if self.added_of[ids[k]] >= 0:
                raise Error(
                    "ValueError: two added tokens share id " + String(ids[k])
                )
            self.sp_offset.append(len(self.sp_arena))
            self.sp_length.append(lengths[k])
            self.sp_id.append(ids[k])
            for i in range(lengths[k]):
                self.sp_arena.append(arena[offs[k] + i])
            self.added_of[ids[k]] = r

    def n_ranks(self) -> Int:
        return self.ranks.n_tokens()

    def n_vocab(self) -> Int:
        if len(self.added_of) > self.ranks.n_tokens():
            return len(self.added_of)
        return self.ranks.n_tokens()

    def max_token_bytes(self) -> Int:
        var longest = 1
        for i in range(self.ranks.n_tokens()):
            if self.ranks.length[i] > longest:
                longest = self.ranks.length[i]
        for k in range(len(self.sp_length)):
            if self.sp_length[k] > longest:
                longest = self.sp_length[k]
        return longest

    def encode_ordinary(
        self, text: List[UInt8], start: Int, end: Int, mut out: List[Int]
    ) raises:
        if end <= start:
            return
        var seg = List[UInt8](capacity=end - start)
        for i in range(start, end):
            seg.append(text[i])
        var bounds = pretokenize_pattern(seg, self.classes, self.pattern)
        for k in range(len(bounds) - 1):
            bpe_append(out, self.ranks, seg, bounds[k], bounds[k + 1])

    def _find(self, text: List[UInt8], r: Int, start: Int, stop: Int) -> Int:
        """First position >= start of special `r` that ends at or before
        `stop`, or -1."""
        var m = self.sp_length[r]
        var at0 = self.sp_offset[r]
        var last = stop - m
        for i in range(start, last + 1):
            var hit = True
            for k in range(m):
                if text[i + k] != self.sp_arena[at0 + k]:
                    hit = False
                    break
            if hit:
                return i
        return -1

    def encode(self, text: List[UInt8], allow_special: Bool) raises -> List[Int]:
        var out = List[Int]()
        var n = len(text)
        if not allow_special or len(self.sp_id) == 0:
            self.encode_ordinary(text, 0, n, out)
            return out^
        var i = 0
        while i < n:
            var best_at = n
            var best = -1
            for r in range(len(self.sp_id)):
                var at = self._find(text, r, i, n)
                # longest first already: a later special wins only by an
                # earlier position
                if at >= 0 and at < best_at:
                    best_at = at
                    best = r
            if best < 0:
                self.encode_ordinary(text, i, n, out)
                break
            self.encode_ordinary(text, i, best_at, out)
            out.append(self.sp_id[best])
            i = best_at + self.sp_length[best]
        return out^

    def pretokenize(self, text: List[UInt8]) raises -> List[Int]:
        return pretokenize_pattern(text, self.classes, self.pattern)

    def append_decoded(self, mut out: List[UInt8], id: Int, k: Int) raises:
        if id >= 0 and id < len(self.added_of) and self.added_of[id] >= 0:
            var r = self.added_of[id]
            var at = self.sp_offset[r]
            for i in range(self.sp_length[r]):
                out.append(self.sp_arena[at + i])
        elif id >= 0 and id < self.ranks.n_tokens():
            self.ranks.append_token_bytes(out, id)
        else:
            raise Error(
                "ValueError: id "
                + String(id)
                + " at position "
                + String(k)
                + " is outside [0, "
                + String(self.n_vocab())
                + ") and no added token"
            )


def _before(
    arena: List[UInt8], offs: List[Int], lengths: List[Int], x: Int, y: Int
) -> Bool:
    """Special `x` sorts before `y`: longer first, then smaller bytes."""
    if lengths[x] != lengths[y]:
        return lengths[x] > lengths[y]
    for i in range(lengths[x]):
        var a = arena[offs[x] + i]
        var b = arena[offs[y] + i]
        if a != b:
            return a < b
    return False
