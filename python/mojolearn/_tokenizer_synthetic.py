# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A small synthetic byte-level BPE vocabulary that mojolearn generates itself,
and a second, pure Python encoder to hold `BpeTokenizer` to.

WHY IT EXISTS. mojolearn ships no third-party vocabulary (2026-09-15). The
tokenizer's gates (`pixi run check-tokenizer`,
`python/mojolearn/tests/test_tokenizer_surface.py`) and the `tokenizer`
identity lane still need a vocabulary with real merges, so this module trains
one, deterministically, from a corpus it builds out of an integer generator
and a fixed list of pieces. Nothing in it is read from a file.

WHAT IT EXERCISES. The 256 single bytes are ids 0 to 255 in byte order, so
every byte falls back to a token. The merges (up to `TARGET_TOKENS` tokens)
come from repeated most-frequent-pair merging over the corpus: invented
syllable words with and without a leading space, digits, contractions,
whitespace runs, a few non-ASCII letters, digits and symbols, and the bytes
of a few float64 values (so the identity lane's binary documents reach
merges too). `<|endoftext|>` is the id after the last rank, the same place
the GPT-2 format puts it.

THE REFERENCE ENCODER. `reference_encode` is the same algorithm as
`tokenizer/impl/pretokenize.mojo` and `tokenizer/impl/bpe.mojo`, written
again in Python from the pattern and the rule (lowest-ranked adjacent pair
first), with the letter and number classes from `unicodedata` and White_Space
as a literal list. `cases()` are texts whose codepoints have had the same
category in every Unicode version since 13.0, so the two encoders agree
whatever Python runs this.

Standalone on purpose (standard library only, no package imports), so the
Mojo gate can write its fixture without importing mojolearn:

    python3 python/mojolearn/_tokenizer_synthetic.py build/tokenizer_synthetic
"""
import functools
import json
import os
import struct
import sys
import unicodedata

ENDOFTEXT = "<|endoftext|>"
TARGET_TOKENS = 512
PAT_STR = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s"
FORMAT = "mojolearn.tokenizer-synthetic.v1"

#: The White_Space property (unchanged since Unicode 4.1).
WHITE_SPACE = (
    (0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085), (0x00A0, 0x00A0),
    (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F),
    (0x205F, 0x205F), (0x3000, 0x3000),
)

_SYLLABLES = ("ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ze", "qu", "an", "el",
              "in", "on", "ul", "st", "tr", "ch", "sh", "or", "ba", "de", "fi", "go")


def _lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield state >> 33


def corpus_pieces():
    """The training corpus as a list of byte strings, one per piece. BPE
    merges only inside a piece, as it only merges inside a pre-token."""
    rng = _lcg(20260915)
    pieces = []
    for _ in range(900):
        n = 1 + next(rng) % 3
        word = "".join(_SYLLABLES[next(rng) % len(_SYLLABLES)] for _ in range(n))
        if next(rng) % 5 == 0:
            word = word.capitalize()
        if next(rng) % 2 == 0:
            word = " " + word
        pieces.append(word.encode("utf-8"))
    fixed = (
        (" 2026", 6), ("2026", 4), (" 123", 5), ("0", 6), ("'s", 8), ("'re", 5), ("'ll", 4),
        ("\n\n", 6), ("   ", 5), ("  ", 5), ("...", 4), ("!!", 3), (" (", 4), (")", 4),
        ("é", 5), (" café", 4), ("ü", 3), (" zürn", 3), ("中文", 4), ("\u0661\u0662\u0663", 3), ("€", 3),
    )
    for text, count in fixed:
        pieces += [text.encode("utf-8")] * count
    for v in (0.0, 1.0, -1.0, 0.5, 2.0):
        pieces += [struct.pack("<d", v)] * 3
    return pieces


@functools.lru_cache(maxsize=1)
def _vocabulary():
    tokens = [bytes([b]) for b in range(256)]
    index = {t: i for i, t in enumerate(tokens)}
    grouped = {}
    for p in corpus_pieces():
        grouped[p] = grouped.get(p, 0) + 1
    seqs = [(list(p), c) for p, c in sorted(grouped.items())]
    while len(tokens) < TARGET_TOKENS:
        counts = {}
        for s, c in seqs:
            for a, b in zip(s, s[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + c
        best = None
        for pair, c in counts.items():
            if c >= 2 and (best is None or (-c, pair) < best):
                best = (-c, pair)
        if best is None:
            break
        a, b = best[1]
        merged = tokens[a] + tokens[b]
        new = index.get(merged)
        if new is None:
            new = len(tokens)
            tokens.append(merged)
            index[merged] = new
        for s, _ in seqs:
            k = 0
            while k < len(s) - 1:
                if s[k] == a and s[k + 1] == b:
                    s[k:k + 2] = [new]
                k += 1
    return tuple(tokens)


def vocabulary():
    """The token byte strings in rank order; rank = id. `<|endoftext|>` is
    not in it and takes id `len(vocabulary())`."""
    return list(_vocabulary())


# --------------------------------------------------------------------------
# the reference encoder
# --------------------------------------------------------------------------

def _decode_cp(data, i):
    """(codepoint, width), or (-1, 1) for a byte that begins no well-formed
    UTF-8 sequence (overlongs, surrogates and above U+10FFFF included)."""
    n = len(data)
    c = data[i]
    if c < 0x80:
        return c, 1
    if 0xC2 <= c <= 0xDF and i + 1 < n:
        b1 = data[i + 1]
        return (((c & 0x1F) << 6) | (b1 & 0x3F), 2) if 0x80 <= b1 <= 0xBF else (-1, 1)
    if 0xE0 <= c <= 0xEF and i + 2 < n:
        b1, b2 = data[i + 1], data[i + 2]
        lo = 0xA0 if c == 0xE0 else 0x80
        hi = 0x9F if c == 0xED else 0xBF
        if lo <= b1 <= hi and 0x80 <= b2 <= 0xBF:
            return ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F), 3
        return -1, 1
    if 0xF0 <= c <= 0xF4 and i + 3 < n:
        b1, b2, b3 = data[i + 1], data[i + 2], data[i + 3]
        lo = 0x90 if c == 0xF0 else 0x80
        hi = 0x8F if c == 0xF4 else 0xBF
        if lo <= b1 <= hi and 0x80 <= b2 <= 0xBF and 0x80 <= b3 <= 0xBF:
            return ((c & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F), 4
        return -1, 1
    return -1, 1


def _is_space(cp):
    return any(lo <= cp <= hi for lo, hi in WHITE_SPACE)


def _in_class(cp, which):
    cat = unicodedata.category(chr(cp))
    if which == "L":
        return cat.startswith("L")
    if which == "N":
        return cat.startswith("N")
    if which == "S":
        return _is_space(cp)
    return not (_is_space(cp) or cat.startswith("L") or cat.startswith("N"))


def _run_end(data, start, which):
    j = start
    while j < len(data):
        cp, w = _decode_cp(data, j)
        if cp < 0 or not _in_class(cp, which):
            break
        j += w
    return j


def _pretoken_end(data, i):
    n = len(data)
    if data[i] == 0x27 and i + 1 < n:
        c1 = data[i + 1]
        if c1 in b"sdmt":
            return i + 2
        if i + 2 < n and bytes(data[i + 1:i + 3]) in (b"ll", b"ve", b"re"):
            return i + 3
    for which in ("L", "N", "O"):
        start = i + 1 if data[i] == 0x20 and i + 1 < n else i
        end = _run_end(data, start, which)
        if end > start:
            return end
    ws_end = _run_end(data, i, "S")
    if ws_end > i:
        if ws_end == n:
            return ws_end
        last, j = i, i
        while j < ws_end:
            last = j
            j += _decode_cp(data, j)[1]
        if last > i:
            return last
        return i + _decode_cp(data, i)[1]
    return i + 1


def pretokenize(data):
    """Pre-token boundaries, first 0 and last len(data)."""
    bounds, i = [0], 0
    while i < len(data):
        i = _pretoken_end(data, i)
        bounds.append(i)
    return bounds


def _bpe(index, data, start, end, out):
    whole = index.get(bytes(data[start:end]))
    if whole is not None:
        out.append(whole)
        return
    bounds = list(range(start, end + 1))
    while len(bounds) > 2:
        best_rank, best_at = -1, -1
        for k in range(len(bounds) - 2):
            r = index.get(bytes(data[bounds[k]:bounds[k + 2]]))
            if r is not None and (best_at < 0 or r < best_rank):
                best_rank, best_at = r, k
        if best_at < 0:
            break
        del bounds[best_at + 1]
    for k in range(len(bounds) - 1):
        out.append(index[bytes(data[bounds[k]:bounds[k + 1]])])


def reference_encode(tokens, data, allow_endoftext=False):
    """The ids of `data` (bytes) under the rank table `tokens`."""
    index = {t: i for i, t in enumerate(tokens)}
    eot = ENDOFTEXT.encode()
    segments = data.split(eot) if allow_endoftext else [data]
    out = []
    for k, seg in enumerate(segments):
        if k > 0:
            out.append(len(tokens))
        bounds = pretokenize(seg)
        for a, b in zip(bounds, bounds[1:]):
            _bpe(index, seg, a, b, out)
    return out


def reference_decode(tokens, ids):
    eot = ENDOFTEXT.encode()
    return b"".join(eot if i == len(tokens) else tokens[i] for i in ids)


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

def cases():
    """(name, text, allow_endoftext). Every class, alternative and fallback
    the tokenizer has, on codepoints whose category is stable across the
    Unicode versions Python ships."""
    words = [p.decode("utf-8") for p in corpus_pieces()[:40]]
    return [
        ("empty", "", False),
        ("corpus_words", "".join(words[:12]), False),
        ("corpus_sentence", "".join(words[12:40]) + ".", False),
        ("merges_with_space", " kalo mine ruvi", False),
        ("capitalized", "Kalo Mine RUVI", False),
        ("contraction_lower", "it's we'll they're I'd", False),
        ("contraction_upper", "IT'S WE'LL", False),
        ("digits", "2026 123 0 45678", False),
        ("arabic_indic_digits", "\u0661\u0662\u0663 \u0664", False),
        ("latin1_letters", " café zürn é", False),
        ("combining_mark", "e\u0301 café", False),
        ("cjk", "中文 中文字", False),
        ("symbols", "€ (x) ... !! ??", False),
        ("emoji", "ok \U0001F642\U0001F642 ok", False),
        ("whitespace_runs", "a   b\t\tc\n\nd", False),
        ("trailing_whitespace", "a   ", False),
        ("nbsp_and_file_sep", "a\u00a0b\u001cc", False),
        ("raw_controls", "\u0000\u0001\u007f", False),
        ("endoftext_as_text", ENDOFTEXT, False),
        ("endoftext_as_special", ENDOFTEXT, True),
        ("endoftext_inside", "kalo " + ENDOFTEXT + " mine" + ENDOFTEXT, True),
        ("unseen_bytes", "xyzw QXJ", False),
    ]


def fixture():
    """The gate's fixture: the vocabulary's size and pattern, and each case's
    text with the reference encoder's ids."""
    tokens = vocabulary()
    out = []
    for name, text, allow in cases():
        raw = text.encode("utf-8")
        ids = reference_encode(tokens, raw, allow)
        out.append(dict(name=name, text=text, ids=ids, allow_endoftext=allow,
                        roundtrip_ok=reference_decode(tokens, ids) == raw))
    return dict(encoding=FORMAT, generator="python/mojolearn/_tokenizer_synthetic.py",
                pat_str=PAT_STR, n_vocab=len(tokens) + 1, cases=out)


def write_ranks(tokens, path):
    """`rank<TAB>hex` lines, the rank file `BpeTokenizer` loads."""
    with open(path, "w", encoding="ascii") as fh:
        for i, t in enumerate(tokens):
            fh.write(f"{i}\t{t.hex()}\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: _tokenizer_synthetic.py OUT_DIR\n")
        return 2
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    tokens = vocabulary()
    write_ranks(tokens, os.path.join(out_dir, "ranks.tsv"))
    fx = fixture()
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="ascii") as fh:
        json.dump(fx, fh, indent=1, ensure_ascii=True)
        fh.write("\n")
    merges = sum(1 for t in tokens if len(t) > 1)
    print(f"{out_dir}: {len(tokens)} ranks ({merges} merges) + {ENDOFTEXT}, {len(fx['cases'])} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
