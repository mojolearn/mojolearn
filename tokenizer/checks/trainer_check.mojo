# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GATE: the Mojo vocabulary trainer against an independent Python
implementation of the same stated algorithm, FILE BYTE FOR FILE BYTE.

    pixi run check-bpe-trainer

The task first runs `python/mojolearn/_bpe_trainer.py`, which writes, per
fixture, the corpus it trained on and the two artifacts it produced:

    build/bpe_trainer/<name>.corpus
    build/bpe_trainer/<name>.ranks.tsv          OUR format
    build/bpe_trainer/<name>.tokenizer.json     the ecosystem's

This then trains on the SAME corpus with the SAME config in Mojo and requires
both renderings to equal those files byte for byte. Not "the same
vocabulary", not "the same merges in some order" -- the same bytes, because
the bytes are what a user ships beside a model and what a rebuild has to
reproduce.

WHAT THIS IS NOT. It is not a cross-vendor identity claim. Vocabulary
training is host-only everywhere and there is no GPU path in `tokenizer/`, so
there is no vendor column to compare. The property is that the same corpus
and config give the same vocabulary bytes on any machine, and the thing that
could actually be wrong -- and is therefore what is asserted -- is that two
independent implementations of the stated algorithm agree.

THE ARMS, and why each exists:

    check_agreement     the two implementations write identical bytes, per
                        fixture, per format. A mismatch prints the first
                        differing offset and both neighbourhoods.
    check_ties_reached  `n_ties_broken` is NON-ZERO. Without this the
                        sabotage below is INERT and a passing gate would mean
                        nothing: a tie-break that is never reached cannot be
                        broken by reversing it.
    check_round_trip    the trained table encodes and decodes its own corpus.
                        A vocabulary that cannot read the text it was built
                        from is broken whatever its hashes say.
    check_determinism   training the same corpus TWICE in one process gives
                        identical bytes, so a stale buffer or a reused table
                        cannot pass unnoticed.

THE SABOTAGE, which must be SEEN TO FAIL:

    pixi run check-bpe-trainer-sabotage

builds with `-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1`, which reverses ONLY the
tie-break. `check_agreement` must then fail. If it passes, the fixture never
reached a tie and the gate is not testing the rule it claims to test.
"""

from tokenizer.encoding import GPT2_ENDOFTEXT, GPT2_PAT_STR, string_bytes
from tokenizer.impl.bpe import bpe_append
from tokenizer.impl.pretokenize import pretokenize
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import (
    UnicodeClasses,
    builtin_unicode_classes,
)
from tokenizer.train.bpe_train import (
    BPE_TRAINER_SABOTAGE,
    TrainedVocabulary,
    train_bpe,
)
from tokenizer.train.emit import render_ranks, render_tokenizer_json

comptime BUILD_DIR = "build/bpe_trainer"


struct Fixture(Copyable, Movable):
    var name: String
    var vocab_size: Int
    var min_frequency: Int

    def __init__(out self, name: String, vocab_size: Int, min_frequency: Int):
        self.name = name
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency


def fixtures() -> List[Fixture]:
    """The same three the Python reference writes, with the same configs. A
    bitwise check needs no large corpus; every one of these trains in
    milliseconds on one core."""
    var out = List[Fixture]()
    out.append(Fixture(String("synthetic"), 512, 2))
    out.append(Fixture(String("ties"), 300, 2))
    out.append(Fixture(String("synthetic_small"), 320, 3))
    return out^


def read_file(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def _first_difference(got: String, want: String) -> Int:
    var g = got.as_bytes()
    var w = want.as_bytes()
    var n = len(g) if len(g) < len(w) else len(w)
    for i in range(n):
        if g[i] != w[i]:
            return i
    if len(g) != len(w):
        return n
    return -1


def _around(text: String, at: Int) -> String:
    """Sixty bytes of context around an offset, for the failure report."""
    var b = text.as_bytes()
    var lo = at - 30
    if lo < 0:
        lo = 0
    var hi = at + 30
    if hi > len(b):
        hi = len(b)
    var out = List[UInt8]()
    for i in range(lo, hi):
        var c = b[i]
        # Newlines and tabs are made visible so the report does not wrap.
        out.append(UInt8(0xB7) if c == 10 or c == 9 else c)
    var s = String(StringSlice(unsafe_from_utf8=Span(out)))
    _ = out
    _ = text
    return s^


def compare(label: String, got: String, want: String) raises -> Int:
    var at = _first_difference(got, want)
    if at < 0:
        return 0
    print("FAIL " + label + ": the two implementations wrote different bytes")
    print(
        "    lengths  ours "
        + String(got.byte_length())
        + ", reference "
        + String(want.byte_length())
        + "; first difference at byte "
        + String(at)
    )
    print("    ours      " + _around(got, at))
    print("    reference " + _around(want, at))
    return 1


def round_trip(
    vocab: TrainedVocabulary, corpus: List[UInt8], classes: UnicodeClasses
) raises -> Int:
    """Encode the corpus under the trained table and decode it back. Built
    on the SHIPPED encoder (`impl/bpe.mojo`), not a second copy, so this also
    says the trainer and the tokenizer agree about what a vocabulary is."""
    var table = RankTable()
    table._reserve(vocab.n_tokens())
    for id in range(vocab.n_tokens()):
        table.offset.append(len(table.arena))
        table.length.append(vocab.length[id])
        var at = vocab.offset[id]
        for k in range(vocab.length[id]):
            table.arena.append(vocab.arena[at + k])
        table._insert(id)

    var ids = List[Int]()
    var bounds = pretokenize(corpus, classes)
    for k in range(len(bounds) - 1):
        bpe_append(ids, table, corpus, bounds[k], bounds[k + 1])

    var back = List[UInt8]()
    for k in range(len(ids)):
        table.append_token_bytes(back, ids[k])
    if len(back) != len(corpus):
        print(
            "FAIL round trip: decoded "
            + String(len(back))
            + " bytes from "
            + String(len(corpus))
        )
        return 1
    for i in range(len(back)):
        if back[i] != corpus[i]:
            print("FAIL round trip: byte " + String(i) + " differs")
            return 1
    return 0


def main() raises:
    print(
        "trainer_check: the Mojo BPE vocabulary trainer against"
        " python/mojolearn/_bpe_trainer.py, byte for byte"
    )
    comptime if BPE_TRAINER_SABOTAGE:
        print(
            "  MOJOLEARN_BPE_TRAINER_SABOTAGE is compiled in: the tie-break"
            " is REVERSED. check_agreement MUST fail below."
        )
    var classes = builtin_unicode_classes()
    print("  " + classes.source_header)

    var bad = 0
    var total_ties = 0
    var fx = fixtures()
    for i in range(len(fx)):
        var f = fx[i].copy()
        var corpus_text = read_file(String(BUILD_DIR) + "/" + f.name + ".corpus")
        var corpus = string_bytes(corpus_text)

        # The Python reference trains on the corpus as ONE document; so do
        # we, or the two are not training on the same thing.
        var documents = List[List[UInt8]]()
        documents.append(corpus.copy())

        var vocab = train_bpe(
            documents, classes, f.vocab_size, f.min_frequency
        )
        total_ties += vocab.n_ties_broken

        var ranks = render_ranks(vocab)
        var tj = render_tokenizer_json(
            vocab, String(GPT2_PAT_STR), String(GPT2_ENDOFTEXT)
        )
        bad += compare(
            f.name + " ranks.tsv",
            ranks,
            read_file(String(BUILD_DIR) + "/" + f.name + ".ranks.tsv"),
        )
        bad += compare(
            f.name + " tokenizer.json",
            tj,
            read_file(String(BUILD_DIR) + "/" + f.name + ".tokenizer.json"),
        )
        bad += round_trip(vocab, corpus, classes)

        # Training the same corpus twice in one process must give the same
        # bytes: a reused buffer or a stale table shows up here and nowhere
        # else.
        var again = train_bpe(
            documents, classes, f.vocab_size, f.min_frequency
        )
        bad += compare(f.name + " repeat", render_ranks(again), ranks)

        print(
            "  "
            + f.name
            + ": "
            + String(len(corpus))
            + " corpus bytes, "
            + String(vocab.n_tokens())
            + " tokens ("
            + String(vocab.n_merges())
            + " merges), "
            + String(vocab.n_ties_broken)
            + " ties broken, "
            + String(vocab.n_groups)
            + " pre-token groups"
        )

    # THE REACH ARM. A tie-break that is never reached cannot be broken, so
    # a sabotage that reverses it would be inert and the gate would pass
    # while testing nothing.
    if total_ties == 0:
        print(
            "FAIL reach: no fixture broke a single tie, so the tie-break rule"
            " was never exercised and the sabotage arm is INERT"
        )
        bad += 1
    else:
        print(
            "  reach: "
            + String(total_ties)
            + " tie-broken selections across the fixtures, so the tie rule is"
            " exercised and the sabotage arm can move it"
        )

    comptime if BPE_TRAINER_SABOTAGE:
        if bad == 0:
            raise Error(
                "trainer_check: the SABOTAGE build agreed with the reference."
                " Reversing the tie-break changed nothing, so this gate does"
                " not test the rule it claims to test."
            )
        print(
            "trainer_check: SABOTAGE SEEN TO FAIL ("
            + String(bad)
            + " failures), which is what this build is for"
        )
    else:
        if bad != 0:
            raise Error(
                "trainer_check: "
                + String(bad)
                + " failures. Byte equality with the second implementation is"
                " the whole point of this gate: fix the trainer, do not"
                " loosen the assertion."
            )
        print("trainer_check: PASS")
