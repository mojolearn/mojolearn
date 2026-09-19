#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AN INDEPENDENT ORACLE FOR THE `bpe-trainer` LANE (2026-09-16).

    pixi run check-bpe-trainer-oracle
    python3 tools/bpe_trainer_oracle_check.py [--sabotage-expected]

WHY THIS FILE EXISTS. The oracle/applicability audit found four lanes a
release record runs whose passing cell is compared against
nothing but a previous hash of the same code. `bpe-trainer` is one of them, and
it is the worst of the four on one axis: its lane body is pure Python integer
work, so every column of a release record computes the SAME BYTES BY
CONSTRUCTION and a cross-column diff adds no information at all. A defect
present when the reference hash was recorded is invisible forever.

WHAT THIS IS NOT. `pixi run check-bpe-trainer` already holds the Mojo trainer
(`tokenizer/train/bpe_train.mojo`) against `python/mojolearn/_bpe_trainer.py`
file byte for file byte, and that IS a second implementation. It is not this
one, for two reasons worth writing down rather than assuming:

  * the Mojo trainer was written AGAINST that Python file as its stated
    reference, so the two share every decision the reference made. A tie-break
    that is wrong in `_bpe_trainer.py` is wrong in `bpe_train.mojo` too, and
    the byte-for-byte check passes.
  * it needs a Mojo toolchain and a compile. It is not what the `bpe-trainer`
    CELL is compared against, on any column, ever.

WHAT THIS IS. A byte-level BPE trainer written HERE from the algorithm's
definition, by a different route from ours at every step where a route was
available, and held against `mojolearn.tokenizer.BpeVocabularyTrainer` on the
lane's own fixture corpus plus corpora chosen to reach the parts of the
algorithm the lane's corpus may not.

THE FOUR PLACES THE REFERENCE TAKES A DIFFERENT ROUTE

  ours (`python/mojolearn/_bpe_trainer.py`)   this reference
  ----------------------------------------   -----------------------------
  pre-tokens grouped into `(piece, count)`    one entry per pre-token
  pairs, sorted by bytes                      OCCURRENCE, ungrouped and
                                              unsorted. If grouping by count
                                              is not exactly equivalent to
                                              counting occurrences, the two
                                              disagree.
  selection scans `sorted(counts)` and        selection materializes every
  keeps the first pair at the top count       qualifying pair, takes the top
                                              count with `max`, sorts the
                                              pairs AT that count and takes
                                              the first. The tie-break is
                                              expressed as an explicit sort
                                              of the tied set rather than as
                                              a scan order.
  a hand-decoded UTF-8 table and a literal    CPython's own strict UTF-8
  White_Space range list                      decoder over each 1..4 byte
  (`_tokenizer_synthetic.py`)                 candidate, and White_Space
                                              derived from `unicodedata`
                                              (Zs, Zl, Zp, U+0009..U+000D,
                                              U+0085) rather than a list.
  `render_tokenizer_json` GENERATES the       the artifact is PARSED with
  artifact text                               `json.loads` and its vocab and
                                              merges are read back and held
                                              against the reference. A
                                              generator held against another
                                              generator cannot see a merge
                                              list that is well formed and
                                              wrong; a parse can.

WHAT THIS ORACLE CAN CATCH

  * a tie-break that is not the stated total order, in either direction, and
    a tie-break that is not reached at all on the fixture (arm TIES refuses
    when the lane's corpus produces zero ties, because an oracle for a rule
    that is never reached is inert);
  * a merge applied overlapping instead of left to right non-overlapping;
  * a pair counted non-overlapping instead of overlapping;
  * a `min_frequency` that is off by one, or applied to the wrong quantity;
  * a vocabulary that stops one merge early or late;
  * a pre-token boundary rule that differs from the pattern the tokenizer
    implements, including the `\\s+(?!\\S)` backtrack and the ` ?\\p{L}++`
    leading space;
  * a `tokenizer.json` whose vocab or merges disagree with the ranks file
    beside it, or that is not parseable JSON.

WHAT IT CANNOT CATCH

  * a defect in the GPT-2 PATTERN ITSELF. Both sides read `PAT_STR` from
    `_tokenizer_synthetic.py`; if the pattern string is wrong for the
    tokenizer that consumes the vocabulary, both agree on the wrong split.
    `_bpe_trainer.check_pattern()` is what holds the two pattern strings
    together, and `tokenizer/checks/tokenizer_check.mojo` is what holds the
    Mojo pre-tokenizer to the Python one.
  * anything about the MOJO trainer. This file never compiles or runs Mojo.
  * a corpus the lane never sees. It says nothing about vocabularies trained
    on text with properties the four corpora here do not have.
  * the byte-to-unicode spelling table, which both sides take from
    `_bpe_trainer.byte_to_char()`. A wrong entry there moves both sides
    together. (The spelling is a format fact the tokenizer lane's round trip
    covers.)

SEEN TO FAIL. `--sabotage-expected` inverts every comparison arm: it requires
at least one disagreement and exits nonzero on agreement. The evidence and the
printed values are emitted by this program.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def _load_identity_break():
    """`tools/identity_break.py` by path, for the lane's own fixture."""
    spec = importlib.util.spec_from_file_location(
        "_identity_break_fixture", ROOT / "tools" / "identity_break.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the reference pre-tokenizer, written from the pattern
# --------------------------------------------------------------------------
# PAT_STR = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s"
#
# The alternation is ordered and every quantifier is possessive, so a match at
# a position is the FIRST alternative that matches anything there, taken as
# long as it will go, with one exception: `\s+(?!\S)` is not possessive and
# backtracks, which is the rule that leaves the last space of a run to the
# word that follows it.

def _decode(data, i):
    """`(codepoint, width)` for well-formed UTF-8 at `i`, else `(-1, 1)`.

    The route is CPython's own strict decoder, which rejects overlongs,
    surrogates and everything above U+10FFFF without a table here.
    """
    n = len(data)
    for width in (1, 2, 3, 4):
        if i + width > n:
            break
        try:
            text = bytes(data[i:i + width]).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) == 1:
            return ord(text), width
    return -1, 1


def _white_space(cp):
    """The Unicode White_Space property, from its definition rather than a
    range list: the separator categories plus the five C0 controls and NEL."""
    if 0x09 <= cp <= 0x0D or cp == 0x85:
        return True
    return unicodedata.category(chr(cp)) in ("Zs", "Zl", "Zp")


def _cls(cp):
    """`L`, `N`, `S` or `O`, the four classes the pattern distinguishes."""
    category = unicodedata.category(chr(cp))
    if category[0] == "L":
        return "L"
    if category[0] == "N":
        return "N"
    return "S" if _white_space(cp) else "O"


def _run(data, start, want):
    """The end of the maximal run of class `want` beginning at `start`."""
    at = start
    while at < len(data):
        cp, width = _decode(data, at)
        if cp < 0 or _cls(cp) != want:
            break
        at += width
    return at


def ref_pretoken_end(data, i):
    """Where the pre-token beginning at byte `i` ends."""
    tail = bytes(data[i + 1:i + 3])
    if data[i] == 0x27:                                  # "'"
        if tail[:1] in (b"s", b"d", b"m", b"t"):
            return i + 2
        if len(tail) == 2 and tail in (b"ll", b"ve", b"re"):
            return i + 3
    for want in ("L", "N", "O"):
        # ` ?` before the class run: one leading space, never two.
        start = i + 1 if (data[i] == 0x20 and i + 1 < len(data)) else i
        end = _run(data, start, want)
        if end > start:
            return end
    space_end = _run(data, i, "S")
    if space_end > i:
        if space_end == len(data):
            return space_end                             # `\s++$`
        # `\s+(?!\S)`: the greedy run backtracks until the next character is
        # itself a space, which is the run minus its LAST space. When the run
        # is one character long there is nothing left to match and the final
        # bare `\s` takes exactly one.
        last, at = i, i
        while at < space_end:
            last = at
            at += _decode(data, at)[1]
        return last if last > i else i + _decode(data, i)[1]
    return i + 1                                         # an undecodable byte


def ref_pretokens(doc):
    """`doc` as a list of pre-token byte strings, in arrival order."""
    out, at = [], 0
    while at < len(doc):
        end = ref_pretoken_end(doc, at)
        assert end > at, f"reference pre-tokenizer stalled at byte {at}"
        out.append(bytes(doc[at:end]))
        at = end
    return out


# --------------------------------------------------------------------------
# the reference trainer, written from the algorithm
# --------------------------------------------------------------------------

def ref_apply(word, left, right, new):
    """One sequence rewritten LEFT TO RIGHT, NON-OVERLAPPING."""
    out, at, n = [], 0, len(word)
    while at < n:
        if at + 1 < n and word[at] == left and word[at + 1] == right:
            out.append(new)
            at += 2
        else:
            out.append(word[at])
            at += 1
    return tuple(out)


def ref_train(documents, vocab_size=512, min_frequency=2, tie="low"):
    """Byte-level BPE from the definition. Returns `(tokens, merges, n_ties)`.

    `words` holds ONE ENTRY PER PRE-TOKEN OCCURRENCE and is never grouped or
    sorted, so nothing here depends on the grouping our trainer uses.
    """
    words = [tuple(piece) for doc in documents for piece in ref_pretokens(doc)]
    tokens = [bytes([b]) for b in range(256)]
    merges, n_ties = [], 0
    while len(tokens) < vocab_size:
        counts = collections.Counter()
        for word in words:
            counts.update(zip(word, word[1:]))
        qualifying = [(count, pair) for pair, count in counts.items()
                      if count >= min_frequency]
        if not qualifying:
            break
        top = max(count for count, _ in qualifying)
        at_top = sorted(pair for count, pair in qualifying if count == top)
        if len(at_top) > 1:
            n_ties += 1
        left, right = at_top[-1] if tie == "high" else at_top[0]
        new = len(tokens)
        tokens.append(tokens[left] + tokens[right])
        merges.append((left, right, new))
        words = [ref_apply(word, left, right, new) for word in words]
    return tokens, merges, n_ties


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

class Report:
    def __init__(self, out):
        self.out = out
        self.failures = []
        self.disagreements = []

    def ok(self, arm, message, detail=""):
        print(f"  ok    {arm}: {message}{detail}", file=self.out)

    def fail(self, arm, message, detail="", disagreement=False):
        print(f"  FAIL  {arm}: {message}{detail}", file=self.out)
        self.failures.append(f"{arm}: {message}")
        if disagreement:
            self.disagreements.append(f"{arm}: {message}")

    def same(self, arm, message, ours, theirs, show=None):
        """Hold two values together, PRINTING BOTH when they differ."""
        if ours == theirs:
            return self.ok(arm, message)
        shown = show(ours, theirs) if show else f"\n          ours = {ours!r}\n          ref  = {theirs!r}"
        self.fail(arm, message, shown, disagreement=True)


def _first_difference(ours, theirs):
    """The first index at which two sequences differ, and both values."""
    for k in range(min(len(ours), len(theirs))):
        if ours[k] != theirs[k]:
            return k, ours[k], theirs[k]
    return min(len(ours), len(theirs)), None, None


def _show_tokens(ours, theirs):
    at, a, b = _first_difference(ours, theirs)
    lines = [f"\n          len ours={len(ours)} ref={len(theirs)}",
             f"\n          first difference at rank {at}"]
    if a is not None:
        lines.append(f"\n            ours = {a.hex()}  ({a!r})")
        lines.append(f"\n            ref  = {b.hex()}  ({b!r})")
    lo = max(0, at - 2)
    lines.append(f"\n          ranks {lo}..{at + 2}")
    for k in range(lo, min(at + 3, max(len(ours), len(theirs)))):
        o = ours[k].hex() if k < len(ours) else "-"
        r = theirs[k].hex() if k < len(theirs) else "-"
        lines.append(f"\n            [{k}] ours={o} ref={r}")
    return "".join(lines)


def _show_merges(ours, theirs):
    at, a, b = _first_difference(ours, theirs)
    return (f"\n          len ours={len(ours)} ref={len(theirs)}"
            f"\n          first difference at merge {at}"
            f"\n            ours = {a}"
            f"\n            ref  = {b}")


def _show_pieces(ours, theirs):
    at, a, b = _first_difference(ours, theirs)
    return (f"\n          count ours={len(ours)} ref={len(theirs)}"
            f"\n          first difference at pre-token {at}"
            f"\n            ours = {a!r}"
            f"\n            ref  = {b!r}")


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def corpora(identity_break, bpe):
    """The corpora, the lane's own first.

    `lane` is the EXACT bytes the `bpe-trainer` lane trains on: the first
    4,096 bytes of the `base` fixture viewed as bytes. An oracle that ran on
    a different corpus would say nothing about the recorded cell.
    """
    X = identity_break.fixture("base")[0]
    lane = memoryview(X.tobytes())[:4096].tobytes()
    ties = identity_break.fixture("ties")[0]
    return [
        ("lane", [lane], 320, 2),
        ("ties-corpus", [bpe.ties_corpus()], 300, 2),
        ("synthetic", [bpe.synthetic_corpus()], 400, 2),
        ("synthetic-minfreq5", [bpe.synthetic_corpus()], 320, 5),
        ("ties-fixture-bytes", [memoryview(ties.tobytes())[:4096].tobytes()], 300, 2),
        ("multi-document", [bpe.ties_corpus(), bpe.synthetic_corpus(n_words=120)], 300, 2),
    ]


def arm_pretokens(rep, bpe, cases):
    """Our pre-token grouping against the reference's, per corpus."""
    for name, documents, _vocab, _freq in cases:
        ours = [piece for doc in documents for piece in _our_pieces(bpe, doc)]
        theirs = [piece for doc in documents for piece in ref_pretokens(doc)]
        rep.same("PRETOKEN", f"{name}: the pre-token sequence agrees "
                             f"({len(theirs)} pre-tokens)", ours, theirs, _show_pieces)


def _our_pieces(bpe, doc):
    """Our pre-tokens for ONE document, in arrival order (not grouped)."""
    bounds = bpe._syn.pretokenize(doc)
    return [bytes(doc[a:b]) for a, b in zip(bounds, bounds[1:])]


def arm_train(rep, ml, bpe, cases, tie):
    """The vocabulary, the merge list and the tie counter, per corpus."""
    for name, documents, vocab_size, min_frequency in cases:
        trained = ml.tokenizer.BpeVocabularyTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency).train(documents)
        tokens, merges, n_ties = ref_train(documents, vocab_size, min_frequency, tie)
        rep.same("TRAIN", f"{name}: the {len(tokens)}-token vocabulary agrees",
                 list(trained.tokens), tokens, _show_tokens)
        rep.same("TRAIN", f"{name}: the {len(merges)} merges agree in order",
                 [tuple(m) for m in trained.merges], merges, _show_merges)
        rep.same("TRAIN", f"{name}: n_ties_broken agrees ({n_ties})",
                 int(trained.n_ties_broken), n_ties)


def arm_ties(rep, ml, bpe, cases):
    """EXPRESSIBILITY. The tie-break arm of this oracle is inert on a corpus
    that never produces a tie, so the lane's own corpus has to reach one."""
    for name, documents, vocab_size, min_frequency in cases:
        trained = ml.tokenizer.BpeVocabularyTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency).train(documents)
        n = int(trained.n_ties_broken)
        if name in ("lane", "ties-corpus"):
            if n > 0:
                rep.ok("TIES", f"{name}: the tie-break is REACHED", f" ({n} selections were ties)")
            else:
                rep.fail("TIES", f"{name}: the tie-break is NEVER REACHED on this corpus, so "
                                 "every tie-break check here is inert", f" (n_ties_broken={n})")
        else:
            rep.ok("TIES", f"{name}: n_ties_broken={n}")
        # The OTHER end of the same total order must give a different
        # vocabulary wherever a tie was broken, and the same one where none
        # was. That is what makes "the tie-break is reached" a measurement
        # rather than a counter we trust.
        other = ref_train(documents, vocab_size, min_frequency, "high")[0]
        low = ref_train(documents, vocab_size, min_frequency, "low")[0]
        if n > 0:
            rep.same("TIES", f"{name}: reversing the tie-break moves the vocabulary",
                     other != low, True)
        else:
            rep.same("TIES", f"{name}: with no tie, reversing the tie-break moves nothing",
                     other, low, _show_tokens)


def arm_artifacts(rep, ml, bpe, cases):
    """The two emitted files, read back by a PARSER rather than regenerated."""
    for name, documents, vocab_size, min_frequency in cases[:3]:
        trained = ml.tokenizer.BpeVocabularyTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency).train(documents)
        tokens, merges, _ = ref_train(documents, vocab_size, min_frequency, "low")

        ranks = {}
        for line in trained.render_ranks().splitlines():
            rank, hexed = line.split("\t")
            ranks[int(rank)] = bytes.fromhex(hexed)
        rep.same("ARTIFACT", f"{name}: the ranks file parses back to the reference vocabulary",
                 [ranks[k] for k in sorted(ranks)], tokens, _show_tokens)
        rep.same("ARTIFACT", f"{name}: the ranks file numbers 0..n-1 with no gap",
                 sorted(ranks), list(range(len(tokens))))

        blob = json.loads(trained.render_tokenizer_json())
        vocab = blob["model"]["vocab"]
        rep.same("ARTIFACT", f"{name}: tokenizer.json vocab agrees with the reference",
                 [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
                 [bpe.spell(t) for t in tokens], _show_pieces)
        rep.same("ARTIFACT", f"{name}: tokenizer.json ids are 0..n-1",
                 sorted(vocab.values()), list(range(len(tokens))))
        rep.same("ARTIFACT", f"{name}: tokenizer.json merges agree with the reference",
                 [tuple(pair) for pair in blob["model"]["merges"]],
                 [(bpe.spell(tokens[a]), bpe.spell(tokens[b])) for a, b, _ in merges],
                 _show_merges)
        rep.same("ARTIFACT", f"{name}: <|endoftext|> is added at id len(vocab)",
                 (blob["added_tokens"][0]["id"], blob["added_tokens"][0]["content"]),
                 (len(tokens), bpe.ENDOFTEXT))
        rep.same("ARTIFACT", f"{name}: the Split pre-tokenizer carries our own pattern",
                 blob["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], bpe.PAT_STR)


def arm_invariants(rep, ml, bpe, cases):
    """Facts the reference and ours must BOTH satisfy, so a shared defect in
    the two trainers has one more thing to get past."""
    for name, documents, vocab_size, min_frequency in cases:
        trained = ml.tokenizer.BpeVocabularyTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency).train(documents)
        tokens = list(trained.tokens)
        rep.same("INVARIANT", f"{name}: the first 256 tokens are the single bytes",
                 tokens[:256], [bytes([b]) for b in range(256)])
        rep.same("INVARIANT", f"{name}: no token occurs twice",
                 len(set(tokens)), len(tokens))
        rep.same("INVARIANT", f"{name}: the vocabulary never exceeds vocab_size",
                 len(tokens) <= vocab_size, True)
        bad = [(k, tokens[new], tokens[a] + tokens[b])
               for k, (a, b, new) in enumerate(trained.merges)
               if tokens[new] != tokens[a] + tokens[b]]
        rep.same("INVARIANT", f"{name}: every merged token is its two parts joined",
                 bad, [])
        late = [(a, b, new) for a, b, new in trained.merges if a >= new or b >= new]
        rep.same("INVARIANT", f"{name}: every merge names parts that already existed",
                 late, [])


def main(argv=None, out=sys.stdout):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sabotage-expected", action="store_true",
                        help="require at least one disagreement; exit nonzero on agreement")
    parser.add_argument("--only", default="", help="comma separated arm names")
    args = parser.parse_args(argv)

    import mojolearn as ml
    from mojolearn import _bpe_trainer as bpe

    identity_break = _load_identity_break()
    cases = corpora(identity_break, bpe)
    rep = Report(out)

    # THE REFERENCE ALWAYS FOLLOWS THE STATED RULE, never the running
    # implementation's. If it read `bpe.sabotaged()` and followed the sabotage
    # arm, the two sides would agree under sabotage and this oracle could not
    # fail, which is the failure mode the audit of 2026-09-16 found eight of.
    tie = "low"
    print(f"bpe_trainer_oracle_check: tie-break {bpe.TIE_BREAK!r}, "
          f"reference arm {tie!r} (always the stated rule), sabotage env "
          f"{bpe.SABOTAGE_ENV}={os.environ.get(bpe.SABOTAGE_ENV, '')!r}, "
          f"ours reports sabotaged={bpe.sabotaged()}", file=out)
    bpe.check_pattern()

    arms = [("PRETOKEN", lambda: arm_pretokens(rep, bpe, cases)),
            ("TRAIN", lambda: arm_train(rep, ml, bpe, cases, tie)),
            ("TIES", lambda: arm_ties(rep, ml, bpe, cases)),
            ("ARTIFACT", lambda: arm_artifacts(rep, ml, bpe, cases)),
            ("INVARIANT", lambda: arm_invariants(rep, ml, bpe, cases))]
    wanted = set(filter(None, args.only.split(",")))
    for name, run in arms:
        if wanted and name not in wanted:
            continue
        print(f"[{name}]", file=out)
        run()

    if args.sabotage_expected:
        if rep.disagreements:
            print(f"\nSABOTAGE CAUGHT: {len(rep.disagreements)} disagreement(s); "
                  "the first is", file=out)
            print(f"  {rep.disagreements[0]}", file=out)
            return 0
        print("\nSABOTAGE NOT CAUGHT: every arm agreed, so this oracle cannot see "
              "the defect it was run against", file=out)
        return 1
    if rep.failures:
        print(f"\nFAILED: {len(rep.failures)} check(s)", file=out)
        for line in rep.failures:
            print(f"  {line}", file=out)
        return 1
    print("\nOK: the independent reference agrees with the trainer on every corpus", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
