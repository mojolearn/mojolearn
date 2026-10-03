# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPT-2 pre-tokenizer, hand-rolled. Mojo has no regex engine.

THE PATTERN, verbatim from the fixture's `pat_str` (and asserted equal to it
by the check, so a fixture regenerated against a different pattern fails
loudly instead of quietly):

    '(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s

Seven alternatives, and tiktoken compiles it with a BACKTRACKING engine
(`fancy-regex`, which the lookahead forces), so the semantics are
LEFTMOST-FIRST, not leftmost-longest: at each position the alternatives are
tried IN ORDER and the first one that matches wins. `pretoken_end` is that
trial, in that order. Every position matches something -- a space matches
alternative 7 at worst, anything else matches 2, 3 or 4 -- so the scan never
skips a byte, and `pretokenize` asserts exactly that.

THE TWO SUBTLETIES, both of which change the output:

  1. THE QUANTIFIERS ARE POSSESSIVE (`++`), so a run is taken and never
     given back. This only matters where something follows the run inside
     one alternative, which is alternative 5 alone: `\\s++$` takes the
     MAXIMAL whitespace run and then requires end of text. If the run is
     followed by anything, alternative 5 fails as a whole rather than
     matching a shorter prefix -- that is what distinguishes it from
     alternative 6 and it is why the two are separate alternatives.
  2. `\\s+(?!\\S)` IS NOT POSSESSIVE and does backtrack. Greedy first: take
     the whole run, then look ahead. The lookahead succeeds only at end of
     text (already alternative 5's case) or before whitespace (impossible
     after a maximal run), so the surviving match is the run MINUS ITS LAST
     CODEPOINT whenever a run of two or more is followed by a non-space.
     That is the rule that makes "a  b   c" split as
     `a`, ` `, ` b`, ` `, ` `, ` c` -- the last space of each run joins the
     following word through alternative 2's optional leading space.

`$` is end of HAYSTACK, the Rust `regex` meaning, not Python's "also before a
final newline". No fixture case separates the two readings (a trailing
newline is whitespace either way); the Rust reading is implemented because
the Rust engine produced the fixture.

THE OPTIONAL LEADING SPACE (` ?`) in alternatives 2, 3 and 4 is greedy and
backtrackable, but trying it the other way is provably pointless: without
consuming the space, the class test would be applied TO the space, and a
space is not a letter, not a number, and excluded from
`[^\\s\\p{L}\\p{N}]`. So "consume it if the next codepoint satisfies the
class" is the whole of it, and the alternative simply fails otherwise.
"""

from tokenizer.impl.unicode_class import UnicodeClasses, decode_codepoint

comptime CLASS_LETTER = 0
comptime CLASS_NUMBER = 1
comptime CLASS_OTHER = 2
comptime CLASS_SPACE = 3


def _in_class(classes: UnicodeClasses, cp: Int, which: Int) -> Bool:
    # ASCII dominates ordinary training corpora.  Its membership in the
    # three Unicode properties is fixed, so do not pay three range-table
    # binary searches per punctuation byte (letter, number, then the three
    # tests in `is_other`).  This is exactly the pinned Unicode table's
    # answer, only stated directly for U+0000..U+007F.
    if cp >= 0 and cp < 0x80:
        var letter = (cp >= 0x41 and cp <= 0x5A) or (
            cp >= 0x61 and cp <= 0x7A
        )
        var number = cp >= 0x30 and cp <= 0x39
        var space = cp == 0x20 or (cp >= 0x09 and cp <= 0x0D)
        if which == CLASS_LETTER:
            return letter
        if which == CLASS_NUMBER:
            return number
        if which == CLASS_SPACE:
            return space
        return not (letter or number or space)
    if which == CLASS_LETTER:
        return classes.is_letter(cp)
    if which == CLASS_NUMBER:
        return classes.is_number(cp)
    if which == CLASS_SPACE:
        return classes.is_space(cp)
    return classes.is_other(cp)


def _run_end(
    data: List[UInt8], start: Int, classes: UnicodeClasses, which: Int
) raises -> Int:
    """End of the maximal run of codepoints in `which` beginning at `start`.

    An invalid UTF-8 lead byte ends every run: it is in no Unicode category,
    so it can only be its own pre-token.
    """
    var j = start
    var n = len(data)
    while j < n:
        # One-byte ASCII needs neither UTF-8 decoding nor a tuple result.
        # `_in_class` has the exact constant-time ASCII property map.
        var byte = Int(data[j])
        if byte < 0x80:
            if not _in_class(classes, byte, which):
                break
            j += 1
            continue
        var step = decode_codepoint(data, j)
        var cp = step[0]
        if cp < 0 or not _in_class(classes, cp, which):
            break
        j += step[1]
    return j


def _contraction_end(data: List[UInt8], i: Int) -> Int:
    """`'(?:[sdmt]|ll|ve|re)` at `i`, or -1.

    LOWERCASE ONLY, exactly as written: "IT'S" does not take this
    alternative, which is why the reference splits it as `IT`, `'`, `S`.
    The two-letter forms are tried before the one-letter ones because the
    alternation inside the group is ordered `[sdmt] | ll | ve | re` and
    those sets are disjoint in the first letter, so order cannot matter
    here -- the two-letter branch is only reachable when the one-letter
    branch's set does not contain the next byte.
    """
    var n = len(data)
    if i >= n or Int(data[i]) != 0x27:  # '\''
        return -1
    if i + 1 >= n:
        return -1
    var c1 = Int(data[i + 1])
    if c1 == 0x73 or c1 == 0x64 or c1 == 0x6D or c1 == 0x74:  # s d m t
        return i + 2
    if i + 2 >= n:
        return -1
    var c2 = Int(data[i + 2])
    if c1 == 0x6C and c2 == 0x6C:  # ll
        return i + 3
    if c1 == 0x76 and c2 == 0x65:  # ve
        return i + 3
    if c1 == 0x72 and c2 == 0x65:  # re
        return i + 3
    return -1


def pretoken_end(
    data: List[UInt8], i: Int, classes: UnicodeClasses
) raises -> Int:
    """End offset of the pre-token beginning at byte `i`. Always > `i`."""
    var n = len(data)

    # 1.  '(?:[sdmt]|ll|ve|re)
    var apos = _contraction_end(data, i)
    if apos > 0:
        return apos

    # 2, 3, 4.   ?\p{L}++ |  ?\p{N}++ |  ?[^\s\p{L}\p{N}]++
    # These classes are mutually exclusive.  Classify the first codepoint
    # once and enter only the run that can match; trying all three in source
    # order produced the same answer but repeated Unicode-table searches.
    var start = i
    if Int(data[i]) == 0x20 and i + 1 < n:
        start = i + 1
    var first = decode_codepoint(data, start)[0]
    if first >= 0:
        var cls = CLASS_OTHER
        if _in_class(classes, first, CLASS_LETTER):
            cls = CLASS_LETTER
        elif _in_class(classes, first, CLASS_NUMBER):
            cls = CLASS_NUMBER
        elif _in_class(classes, first, CLASS_SPACE):
            cls = CLASS_SPACE
        if cls != CLASS_SPACE:
            return _run_end(data, start, classes, cls)

    # 5.  \s++$   -- possessive: the maximal run must reach end of text
    var ws_end = _run_end(data, i, classes, CLASS_SPACE)
    if ws_end > i:
        if ws_end == n:
            return ws_end

        # 6.  \s+(?!\S)  -- the run minus its last codepoint, when there is
        #     more than one codepoint in it
        var last = i
        var j = i
        while j < ws_end:
            last = j
            j += decode_codepoint(data, j)[1]
        if last > i:
            return last

        # 7.  \s  -- a single whitespace codepoint
        return i + decode_codepoint(data, i)[1]

    # Nothing in the pattern can reach here on valid UTF-8: a codepoint is a
    # letter, a number, whitespace, or in the complement class of all three.
    # An invalid UTF-8 lead byte is in none of them and becomes its own
    # one-byte pre-token, which is this lane's behaviour and not tiktoken's
    # (its input cannot be invalid). NOT_IMPLEMENTED.tsv records it.
    return i + 1


def pretokenize(
    data: List[UInt8], classes: UnicodeClasses
) raises -> List[Int]:
    """Pre-token boundaries: `m + 1` offsets for `m` pre-tokens, first 0 and
    last `len(data)`. Returned as boundaries rather than as copied byte
    lists so the BPE loop can hash straight out of `data`."""
    var bounds = List[Int]()
    bounds.append(0)
    var i = 0
    var n = len(data)
    while i < n:
        var end = pretoken_end(data, i, classes)
        if end <= i or end > n:
            raise Error(
                "pretoken_end returned "
                + String(end)
                + " at offset "
                + String(i)
                + ": the scan must advance and must stay in bounds"
            )
        bounds.append(end)
        i = end
    return bounds^


# ------------------------------------------------ Llama 3 and Qwen 2 cuts
#
# lane/pyglue-text-io (2026-10-03): the Llama 3 and Qwen 2 patterns, moved
# here from `python/mojolearn/models/tokenizer.py` (where they ran in Python
# over `unicodedata`), so every pattern cuts in Mojo over the SAME pinned
# Unicode tables the GPT-2 cut uses:
#
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
#   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
#
# (Qwen 2: `\p{N}` one digit at a time). Leftmost-first over the seven
# alternatives, as the backtracking engines that compiled them do.

comptime PATTERN_GPT2 = 0
comptime PATTERN_LLAMA3 = 1
comptime PATTERN_QWEN2 = 2


def _cp_at(data: List[UInt8], i: Int) raises -> Tuple[Int, Int]:
    return decode_codepoint(data, i)


def _fold_letter(cp: Int) -> Int:
    """The ASCII lowercase letter a codepoint case-folds to, when that fold
    is one of the contraction letters (s t m d r v l e), else -1. The only
    non-ASCII codepoint folding to one of them is U+017F (long s -> s)."""
    if cp == 0x17F:
        return 0x73
    if cp >= 0x41 and cp <= 0x5A:
        return cp + 0x20
    if cp >= 0x61 and cp <= 0x7A:
        return cp
    return -1


def _contraction_end_ci(data: List[UInt8], i: Int) raises -> Int:
    """`(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `i`, or -1."""
    var n = len(data)
    if Int(data[i]) != 0x27 or i + 1 >= n:
        return -1
    var s1 = _cp_at(data, i + 1)
    var f1 = _fold_letter(s1[0])
    if f1 == 0x73 or f1 == 0x74 or f1 == 0x6D or f1 == 0x64:  # s t m d
        return i + 1 + s1[1]
    if (f1 == 0x72 or f1 == 0x76 or f1 == 0x6C) and i + 1 + s1[1] < n:
        var s2 = _cp_at(data, i + 1 + s1[1])
        var f2 = _fold_letter(s2[0])
        if (f1 == 0x72 and f2 == 0x65) or (f1 == 0x76 and f2 == 0x65) or (
            f1 == 0x6C and f2 == 0x6C
        ):
            return i + 1 + s1[1] + s2[1]
    return -1


def _cls(classes: UnicodeClasses, cp: Int, which: Int) -> Bool:
    if cp < 0:
        return False
    return _in_class(classes, cp, which)


def _is_crlf(cp: Int) -> Bool:
    return cp == 0x0D or cp == 0x0A


def pretoken_end_llama(
    data: List[UInt8], i: Int, classes: UnicodeClasses, max_digits: Int
) raises -> Int:
    """End of the pre-token at byte `i` under the Llama 3 (`max_digits` 3)
    or Qwen 2 (`max_digits` 1) pattern. Always > `i`."""
    var n = len(data)
    var e = _contraction_end_ci(data, i)
    if e > 0:
        return e
    var s0 = _cp_at(data, i)
    var cp0 = s0[0]
    var w0 = s0[1]
    # 2.  [^\r\n\p{L}\p{N}]?\p{L}+
    if _cls(classes, cp0, CLASS_LETTER):
        return _run_end(data, i, classes, CLASS_LETTER)
    if (
        cp0 >= 0
        and not _is_crlf(cp0)
        and not _cls(classes, cp0, CLASS_NUMBER)
        and i + w0 < n
    ):
        var cp1 = _cp_at(data, i + w0)[0]
        if _cls(classes, cp1, CLASS_LETTER):
            return _run_end(data, i + w0, classes, CLASS_LETTER)
    # 3.  \p{N}{1,max_digits}
    if _cls(classes, cp0, CLASS_NUMBER):
        var j = i
        var k = 0
        while j < n and k < max_digits:
            var s = _cp_at(data, j)
            if not _cls(classes, s[0], CLASS_NUMBER):
                break
            j += s[1]
            k += 1
        return j
    # 4.   ?[^\s\p{L}\p{N}]+[\r\n]*
    var start = i
    if cp0 == 0x20 and i + 1 < n:
        var cp1 = _cp_at(data, i + 1)[0]
        if _cls(classes, cp1, CLASS_OTHER):
            start = i + 1
    if start > i or _cls(classes, cp0, CLASS_OTHER):
        var j = _run_end(data, start, classes, CLASS_OTHER)
        if j > start:
            while j < n and (Int(data[j]) == 0x0D or Int(data[j]) == 0x0A):
                j += 1
            return j
    # 5.  \s*[\r\n]+  -- the run up to and including its last CR or LF
    var ws_end = _run_end(data, i, classes, CLASS_SPACE)
    if ws_end > i:
        var last_crlf = -1
        var j = i
        while j < ws_end:
            var s = _cp_at(data, j)
            if _is_crlf(s[0]):
                last_crlf = j + s[1]
            j += s[1]
        if last_crlf > 0:
            return last_crlf
        # 6.  \s+(?!\S)
        if ws_end == n:
            return ws_end
        var last = i
        j = i
        while j < ws_end:
            last = j
            j += _cp_at(data, j)[1]
        if last > i:
            return last
        # 7.  \s+
        return ws_end
    # an invalid UTF-8 lead byte is its own one-byte pre-token
    return i + 1


def pretokenize_pattern(
    data: List[UInt8], classes: UnicodeClasses, pattern: Int
) raises -> List[Int]:
    """`pretokenize` under `pattern` (PATTERN_GPT2, PATTERN_LLAMA3 or
    PATTERN_QWEN2): `m + 1` boundaries, first 0 and last `len(data)`."""
    if pattern == PATTERN_GPT2:
        return pretokenize(data, classes)
    if pattern != PATTERN_LLAMA3 and pattern != PATTERN_QWEN2:
        raise Error("pretokenize: unknown pattern code " + String(pattern))
    var digits = 3 if pattern == PATTERN_LLAMA3 else 1
    var bounds = List[Int]()
    bounds.append(0)
    var i = 0
    var n = len(data)
    while i < n:
        var end = pretoken_end_llama(data, i, classes, digits)
        if end <= i or end > n:
            raise Error(
                "pretoken_end_llama returned "
                + String(end)
                + " at offset "
                + String(i)
                + ": the scan must advance and must stay in bounds"
            )
        bounds.append(end)
        i = end
    return bounds^
