# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A bitwise-deterministic byte-level BPE VOCABULARY TRAINER.

mojolearn ships a byte-level BPE tokenizer (`tokenizer/`) and no vocabulary.
This module TRAINS one, and it is the reference the Mojo trainer
(`tokenizer/train/bpe_train.mojo`) is held byte-for-byte to by
`pixi run check-bpe-trainer`. Two independent implementations of one stated
algorithm, asserted to write identical files, is the same arrangement
`_tokenizer_synthetic.py` and `tokenizer/checks/tokenizer_check.mojo` already
use for the encoder.

WHY A TRAINER AT ALL, given that Hugging Face's BPE trainer measured bitwise
reproducible (bench/results/tokenizer_determinism/README.md, 2026-09-16)?
Because a vocabulary is part of what a model IS, and everything mojolearn
publishes should be rebuildable bit for bit from source we control. The
measurement lane's finding is not that determinism is free; it is that HF's
BPE happens to have it while HF's unigram and SentencePiece's sampled path do
not. This trainer is deterministic BY CONSTRUCTION and is verified in our own
harness rather than inherited.

## THE ALGORITHM, stated once

1. The corpus is split into PRE-TOKENS by the GPT-2 pattern
   (`tokenizer/impl/pretokenize.mojo`; `PAT_STR` below must equal the
   tokenizer's, and `check_pattern` asserts it). BPE never merges across a
   pre-token boundary, so the corpus reduces to a multiset of pre-tokens.
2. Identical pre-tokens are GROUPED with a count, and the groups are held in
   ascending byte order. The corpus is a LIST OF DOCUMENTS, each
   pre-tokenized alone, so no pre-token spans a document join and DOCUMENT
   ORDER CANNOT REACH THE RESULT AT ALL. Nothing is sampled and nothing is
   shuffled.
3. The vocabulary starts as the 256 single bytes, id = byte value.
4. Repeat until the vocabulary reaches `vocab_size`, or no pair qualifies:
   count every ADJACENT PAIR across all groups, weighted by the group count,
   and merge the winner.
5. A merge appends the joined bytes as the next id and rewrites every group
   LEFT TO RIGHT, NON-OVERLAPPING.

## THE FOUR THINGS THAT MAKE IT DETERMINISTIC

**1. A total order on the tie-break.** The winner is the pair with the
highest count; ties are broken by the SMALLEST `(left_id, right_id)`
lexicographically. Distinct pairs have distinct `(left_id, right_id)`, so
this is a TOTAL order and never falls through to whatever a data structure
yields. `train()` returns `n_ties_broken`, the number of selections where two
or more pairs shared the top count, so the tie rule can be shown to be
REACHED rather than merely present -- a corpus that produces no tie proves
nothing about it (`ties_corpus()` is engineered to produce many).

**2. A pinned reduction order.** Counting is SINGLE-THREADED. There is no
thread count to vary and therefore no reduction order to get wrong. This is a
measured choice, not an oversight: the fixture trains in well under a second,
and a parallel count would have to merge per-shard counts in shard index
order to keep this property.

**3. Sorted iteration, never hash-map order.** The pair counts live in a dict
for O(1) accumulation, but SELECTION ITERATES `sorted(counts)`. No output and
no tie is ever settled by an iteration order.

**4. No floats anywhere in selection.** Counts are integers, the comparison is
on integers, and the merged ids are integers. There is no score, no
probability and no log-likelihood -- which is exactly where Hugging Face's
unigram trainer loses reproducibility.

One more rule, because it is a real choice and not an implementation detail:
when a pair's two sides are EQUAL (`aa` inside `aaa`), counting sees two
occurrences and rewriting applies one merge. Counting overlapping and
applying non-overlapping is what the established trainers do, and both halves
are deterministic; it is written down here so the Mojo side implements the
same thing rather than the other reading.

## TWO OUTPUT FORMATS

`write_ranks` writes OUR format, `rank<TAB>hex_of_token_bytes` -- the file
`tokenizer/impl/ranks.mojo` loads and `GPT2Tokenizer.from_ranks_file` reads.

`write_tokenizer_json` writes a `tokenizer.json` Hugging Face `tokenizers`
loads, so a model published with one of our vocabularies is usable by people
who do not use mojolearn. Its pre-tokenizer is a `Sequence` of `Split` on OUR
LITERAL PATTERN and `ByteLevel(add_prefix_space=false, use_regex=false)`,
rather than `ByteLevel(use_regex=true)` which would run Hugging Face's own
GPT-2 regex. Both were measured to reproduce our splits on every sample
tried, but the `Split` form CARRIES the pattern instead of depending on that
agreement continuing to hold. `tools/bpe_trainer_interop.py` measures the
round trip; the claim is not assumed.

Both writers are byte-exact and hand-rolled, including the JSON escaping, so
that the Mojo writer can reproduce them without either side depending on a
JSON library's formatting choices. The spelled tokens hold no control
characters by construction (the byte-to-unicode bijection maps every byte
into U+0021..U+00FF and U+0100..U+0143), so the only escapes that can occur
are `\\"`, `\\\\` and `\\uXXXX` for a codepoint above 0x7E.

Standalone on purpose (standard library only), so the Mojo gate can write the
reference artifacts without importing the package:

    python3 python/mojolearn/_bpe_trainer.py build/bpe_trainer
"""
import os
import sys

try:  # as a package module
    from . import _tokenizer_synthetic as _syn
except ImportError:  # run by path, as the Mojo gate runs it
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_tokenizer_synthetic",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tokenizer_synthetic.py"),
    )
    _syn = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_syn)

ENDOFTEXT = "<|endoftext|>"
PAT_STR = _syn.PAT_STR
FORMAT = "mojolearn.bpe-trainer.v1"

#: The tie-break, spelled for the artifacts and the docs. Changing this
#: string without changing `_select` is a lie, so the gate prints it.
TIE_BREAK = "highest count, then smallest (left_id, right_id)"

#: The sabotage arm (`tokenizer/train/bpe_train.mojo` carries the same one as
#: a build define). It breaks ONLY the tie-break, taking the LARGEST
#: `(left_id, right_id)` among the pairs sharing the top count, so a corpus
#: with no tie is unmoved by it -- which is the point: if the fixture's
#: hashes do not move under this, the fixture never reached the rule.
SABOTAGE_ENV = "MOJOLEARN_BPE_TRAINER_SABOTAGE"


def sabotaged():
    return os.environ.get(SABOTAGE_ENV, "") not in ("", "0")


# --------------------------------------------------------------------------
# the byte-to-unicode spelling (the format's, and `impl/byte_unicode.mojo`'s)
# --------------------------------------------------------------------------

def byte_to_char():
    """Byte -> the codepoint the GPT-2 format spells it with. The printable
    Latin-1 runs are fixed points; the other 68 bytes take U+0100 upward in
    byte order."""
    fixed = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
    out, extra = {}, 256
    for b in range(256):
        if b in fixed:
            out[b] = chr(b)
        else:
            out[b] = chr(extra)
            extra += 1
    return out


_B2C = byte_to_char()


def spell(token):
    """A token's bytes in the format's printable spelling."""
    return "".join(_B2C[b] for b in token)


# --------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------

def as_documents(corpus):
    """`corpus` as a list of documents. A single `bytes` is ONE document.

    Documents are the unit because a pre-token CANNOT SPAN TWO OF THEM: each
    is pre-tokenized alone, so presenting the same shards in a different
    order cannot silently weld two pre-tokens together at the join. That is
    what makes the corpus-order axis a real test rather than an accident of
    where the concatenation happened to fall."""
    if isinstance(corpus, (bytes, bytearray, memoryview)):
        return [bytes(corpus)]
    return [bytes(d) for d in corpus]


def pretoken_groups(corpus):
    """The corpus as `[(pre-token bytes, count)]` in ascending byte order.

    Sorted, so the groups the merge loop walks do not depend on the order the
    documents were presented in: a reversed or rotated document list reduces
    to the SAME list, byte for byte."""
    counts = {}
    for doc in as_documents(corpus):
        bounds = _syn.pretokenize(doc)
        for a, b in zip(bounds, bounds[1:]):
            piece = bytes(doc[a:b])
            counts[piece] = counts.get(piece, 0) + 1
    return sorted(counts.items())


def _pair_counts(seqs):
    """Adjacent-pair counts over `[(ids, count)]`, overlapping."""
    counts = {}
    for ids, c in seqs:
        for k in range(len(ids) - 1):
            key = (ids[k], ids[k + 1])
            counts[key] = counts.get(key, 0) + c
    return counts


def _select(counts, min_frequency, break_ties_high):
    """The winning pair, and whether this selection was a tie.

    Iterates `sorted(counts)` -- never the dict's own order -- and compares
    integers only. Returns `(pair, count, n_at_top)`, or `(None, 0, 0)` when
    nothing reaches `min_frequency`.
    """
    best_pair, best_count = None, 0
    for pair in sorted(counts):
        c = counts[pair]
        if c < min_frequency:
            continue
        if best_pair is None or c > best_count:
            best_pair, best_count = pair, c
        elif c == best_count and break_ties_high and pair > best_pair:
            # THE SABOTAGE: the opposite end of the same total order.
            best_pair = pair
    if best_pair is None:
        return None, 0, 0
    n_at_top = sum(1 for p, c in counts.items() if c == best_count and c >= min_frequency)
    return best_pair, best_count, n_at_top


def _apply(ids, a, b, new):
    """Rewrite one sequence, LEFT TO RIGHT, NON-OVERLAPPING."""
    out, k, n = [], 0, len(ids)
    while k < n:
        if k + 1 < n and ids[k] == a and ids[k + 1] == b:
            out.append(new)
            k += 2
        else:
            out.append(ids[k])
            k += 1
    return out


def train(corpus, vocab_size=512, min_frequency=2, break_ties_high=None):
    """Train a byte-level BPE vocabulary on `corpus` (bytes).

    Returns `(tokens, merges, stats)`: the token byte strings in rank order
    (rank = id, the 256 single bytes first), the merges as
    `[(left_id, right_id, new_id)]` in the order they were made, and a stats
    dict carrying `n_ties_broken`, which is how the tie rule is shown to have
    been REACHED.
    """
    if break_ties_high is None:
        break_ties_high = sabotaged()
    if vocab_size < 256:
        raise ValueError(f"mojolearn: vocab_size {vocab_size} is below the 256 single-byte tokens")
    if min_frequency < 1:
        raise ValueError(f"mojolearn: min_frequency {min_frequency} must be at least 1")

    tokens = [bytes([b]) for b in range(256)]
    index = {t: i for i, t in enumerate(tokens)}
    seqs = [(list(piece), c) for piece, c in pretoken_groups(corpus)]

    merges, n_ties = [], 0
    while len(tokens) < vocab_size:
        counts = _pair_counts(seqs)
        pair, _count, n_at_top = _select(counts, min_frequency, break_ties_high)
        if pair is None:
            break
        if n_at_top > 1:
            n_ties += 1
        a, b = pair
        merged = tokens[a] + tokens[b]
        if merged in index:
            # Unreachable on this algorithm (a merge that already exists
            # cannot be the top pair, because its parts were rewritten away),
            # but refused rather than silently renumbered: a duplicate token
            # would make the rank table ambiguous and `load_rank_table`
            # refuses it anyway.
            raise AssertionError(f"mojolearn: merge {merged.hex()} is already token {index[merged]}")
        new = len(tokens)
        tokens.append(merged)
        index[merged] = new
        merges.append((a, b, new))
        seqs = [(_apply(ids, a, b, new), c) for ids, c in seqs]

    stats = {
        "n_tokens": len(tokens),
        "n_merges": len(merges),
        "n_groups": len(seqs),
        "n_ties_broken": n_ties,
        "tie_break": TIE_BREAK,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
    }
    return tokens, merges, stats


# --------------------------------------------------------------------------
# format 1: ours
# --------------------------------------------------------------------------

def render_ranks(tokens):
    """`rank<TAB>hex` lines, the rank file the tokenizer loads."""
    return "".join(f"{i}\t{t.hex()}\n" for i, t in enumerate(tokens))


def write_ranks(tokens, path):
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(render_ranks(tokens))


# --------------------------------------------------------------------------
# format 2: tokenizer.json, for the ecosystem
# --------------------------------------------------------------------------

def json_string(s):
    """One JSON string literal, escaped by OUR rule so the Mojo writer can
    reproduce it exactly: `"` and `\\` take their short escapes, every
    codepoint above 0x7E takes `\\uXXXX` with LOWERCASE hex, and everything
    else is itself. No control character can occur in a spelled token; one in
    the PATTERN would, so it is refused by name rather than mis-escaped."""
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif cp < 0x20:
            raise ValueError(f"mojolearn: refusing to escape control character U+{cp:04X} in {s!r}")
        elif cp <= 0x7E:
            out.append(ch)
        elif cp <= 0xFFFF:
            out.append(f"\\u{cp:04x}")
        else:
            cp -= 0x10000
            out.append(f"\\u{0xD800 + (cp >> 10):04x}\\u{0xDC00 + (cp & 0x3FF):04x}")
    out.append('"')
    return "".join(out)


def render_tokenizer_json(tokens, merges):
    """A `tokenizer.json` Hugging Face `tokenizers` loads.

    The layout is fixed and hand-rolled -- one entry per line, no
    indentation -- so that the Mojo writer produces the SAME BYTES without
    either side inheriting a JSON library's formatting. `<|endoftext|>` is an
    added special token at id `len(tokens)`, the place the GPT-2 format puts
    it and the place `eot_id()` puts it.
    """
    eot_id = len(tokens)
    p = []
    a = p.append
    a("{\n")
    a('"version":"1.0",\n')
    a('"truncation":null,\n')
    a('"padding":null,\n')
    a('"added_tokens":[\n')
    a('{"id":%d,"content":%s,"single_word":false,"lstrip":false,"rstrip":false,'
      '"normalized":false,"special":true}\n' % (eot_id, json_string(ENDOFTEXT)))
    a("],\n")
    a('"normalizer":null,\n')
    a('"pre_tokenizer":{"type":"Sequence","pretokenizers":[\n')
    a('{"type":"Split","pattern":{"Regex":%s},"behavior":"Isolated","invert":false},\n'
      % json_string(PAT_STR))
    a('{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}\n')
    a("]},\n")
    a('"post_processor":null,\n')
    a('"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,'
      '"use_regex":false},\n')
    a('"model":{\n')
    a('"type":"BPE",\n')
    a('"dropout":null,\n')
    a('"unk_token":null,\n')
    a('"continuing_subword_prefix":null,\n')
    a('"end_of_word_suffix":null,\n')
    a('"fuse_unk":false,\n')
    a('"byte_fallback":false,\n')
    a('"ignore_merges":false,\n')
    a('"vocab":{\n')
    for i, t in enumerate(tokens):
        a("%s:%d%s\n" % (json_string(spell(t)), i, "," if i + 1 < len(tokens) else ""))
    a("},\n")
    a('"merges":[\n')
    for k, (x, y, _new) in enumerate(merges):
        a("[%s,%s]%s\n" % (json_string(spell(tokens[x])), json_string(spell(tokens[y])),
                           "," if k + 1 < len(merges) else ""))
    a("]\n")
    a("}\n")
    a("}\n")
    return "".join(p)


def write_tokenizer_json(tokens, merges, path):
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(render_tokenizer_json(tokens, merges))


# --------------------------------------------------------------------------
# the corpora (generated, never committed)
# --------------------------------------------------------------------------

_SYLLABLES = ("ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ze", "qu",
              "an", "el", "in", "on", "ul", "st", "tr", "ch", "sh", "or")


def _lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield state >> 33


def synthetic_corpus(n_words=600, seed=20260916):
    """A deterministic synthetic corpus. Generated from an integer
    recurrence, so it is reproducible anywhere and NOTHING IS COMMITTED -- no
    third-party text ships in this tree."""
    rng = _lcg(seed)
    out = []
    for _ in range(n_words):
        n = 1 + next(rng) % 3
        word = "".join(_SYLLABLES[next(rng) % len(_SYLLABLES)] for _ in range(n))
        if next(rng) % 7 == 0:
            word = word.capitalize()
        if next(rng) % 11 == 0:
            word += "'s"
        if next(rng) % 5 == 0:
            word += str(next(rng) % 100)
        out.append((" " if next(rng) % 2 else "") + word)
    fixed = ("\n", " café", " zürn", " 中文", " ١٢٣",
             " €", "...", "!!", " (x)", "\n\n", "   ")
    for k, text in enumerate(fixed):
        out += [text] * (3 + k % 4)
    return "".join(out).encode("utf-8")


def ties_corpus():
    """A corpus ENGINEERED TO PRODUCE TIES, so the tie-break is reached.

    Every two-letter word below occurs the same number of times, so their
    pairs all reach the top count together and the winner can only be settled
    by the stated total order. A trainer whose tie-break is whatever its data
    structure yields gives a DIFFERENT vocabulary here while agreeing with us
    on ordinary text, which is why the fixture carries this corpus as well as
    the synthetic one.
    """
    words = ("ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr", "st",
             "uv", "wx", "yz", "ba", "dc", "fe", "hg", "ji")
    out = []
    for _ in range(4):
        for w in words:
            out.append(" " + w)
    return "".join(out).encode("utf-8")


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------

def check_pattern():
    """The trainer must pre-tokenize with the pattern the TOKENIZER
    implements, or a vocabulary trained here is wrong for the encoder that
    reads it. `_syn.PAT_STR` is the fixture's, and
    `tokenizer/checks/tokenizer_check.mojo` already holds THAT to
    `encoding.mojo`'s `GPT2_PAT_STR`, so this closes the chain."""
    if PAT_STR != _syn.PAT_STR:
        raise AssertionError("mojolearn: the trainer's pattern is not the tokenizer's")
    return PAT_STR


def check_round_trip(tokens, corpus):
    """Every document encodes and decodes under the trained vocabulary. A
    trained table that cannot encode its own corpus is broken whatever its
    hashes say."""
    n_ids = 0
    for doc in as_documents(corpus):
        ids = _syn.reference_encode(tokens, doc, False)
        if _syn.reference_decode(tokens, ids) != doc:
            return False, n_ids
        n_ids += len(ids)
    return True, n_ids


# --------------------------------------------------------------------------
# the fixture the Mojo gate compares against
# --------------------------------------------------------------------------

#: (name, corpus, vocab_size, min_frequency). Small on purpose: a bitwise
#: check needs no large corpus, and every one of these trains in well under a
#: second on one core.
FIXTURES = (
    ("synthetic", None, 512, 2),
    ("ties", None, 300, 2),
    ("synthetic_small", None, 320, 3),
)


def fixture_corpus(name):
    if name == "ties":
        return ties_corpus()
    if name == "synthetic_small":
        return synthetic_corpus(n_words=200, seed=7)
    return synthetic_corpus()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: _bpe_trainer.py OUT_DIR\n")
        return 2
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    check_pattern()

    lines = []
    for name, _c, vocab_size, min_frequency in FIXTURES:
        corpus = fixture_corpus(name)
        with open(os.path.join(out_dir, f"{name}.corpus"), "wb") as fh:
            fh.write(corpus)
        tokens, merges, stats = train(corpus, vocab_size, min_frequency)
        write_ranks(tokens, os.path.join(out_dir, f"{name}.ranks.tsv"))
        write_tokenizer_json(tokens, merges, os.path.join(out_dir, f"{name}.tokenizer.json"))
        ok, n_ids = check_round_trip(tokens, corpus)
        if not ok:
            raise AssertionError(f"mojolearn: {name}: the trained vocabulary does not round trip its corpus")
        lines.append(f"{name}\t{len(corpus)}\t{stats['n_tokens']}\t{stats['n_merges']}"
                     f"\t{stats['n_ties_broken']}\t{stats['n_groups']}\t{n_ids}")
        print(f"{name}: {len(corpus)} corpus bytes, {stats['n_tokens']} tokens "
              f"({stats['n_merges']} merges), {stats['n_ties_broken']} ties broken, "
              f"{stats['n_groups']} pre-token groups, {n_ids} ids")
    with open(os.path.join(out_dir, "summary.tsv"), "w", encoding="ascii", newline="\n") as fh:
        fh.write("# name\tcorpus_bytes\tn_tokens\tn_merges\tn_ties_broken\tn_groups\tn_ids\n")
        fh.write("".join(l + "\n" for l in lines))
    if sabotaged():
        print(f"NOTE {SABOTAGE_ENV} is set: the tie-break is REVERSED in these artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
