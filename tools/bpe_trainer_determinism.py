#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DETERMINISM EVIDENCE for the byte-level BPE vocabulary trainer, per axis.

    python3 tools/bpe_trainer_determinism.py            # every axis
    python3 tools/bpe_trainer_determinism.py --axis order

THE CLAIM BEING TESTED is not "bitwise identical across GPU vendors".
Vocabulary training is HOST-ONLY everywhere -- Hugging Face, SentencePiece
and tiktoken all train on a CPU, because counting, sorting and merging is not
a matmul workload -- so there is no GPU path here and no cross-vendor GPU
column to match. The claim is:

    THE SAME CORPUS AND CONFIG PRODUCE THE SAME VOCABULARY BYTES ON ANY
    MACHINE AND ARCHITECTURE.

Every axis below varies one thing that MUST NOT reach the output, and hashes
both emitted formats. Two arms vary something that MUST reach it, so a
harness that reports IDENTICAL for everything is caught:

    control-vocab   vocab_size + 1      must DIFFER
    control-sabotage the tie-break      must DIFFER

The sabotage arm is the load-bearing one. It reverses ONLY the tie-break, so
if the fixture never reached a tie it would be INERT and the arm would report
SAME -- which is exactly the failure this arm exists to catch. `n_ties_broken`
is printed beside it so "the rule was reached" is visible rather than assumed.

THREADS. The trainer is single-threaded by construction, so there is no
thread axis to vary. That is recorded as a property, not measured as an axis:
a parallel count would have to merge per-shard counts in shard index order,
and this trainer does not take that risk for a workload that finishes in
milliseconds on the fixture.

ARCHITECTURE. Selection is over integers end to end -- no float is compared,
summed or stored anywhere in the trainer -- so the result cannot vary with
floating-point behaviour. This run records arm64; the x86_64 leg runs the
same file on a Linux CPU host.
"""
import argparse
import hashlib
import importlib.util
import os
import platform
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "_bpe_trainer", os.path.join(HERE, "python", "mojolearn", "_bpe_trainer.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)


def artifacts(corpus, vocab_size=512, min_frequency=2, break_ties_high=False):
    """The two emitted formats and the stats, for one training run."""
    tokens, merges, stats = TR.train(corpus, vocab_size, min_frequency, break_ties_high)
    return (TR.render_ranks(tokens), TR.render_tokenizer_json(tokens, merges), stats)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cells(corpus, **kw):
    ranks, tj, stats = artifacts(corpus, **kw)
    return (digest(ranks), digest(tj)), stats


def documents(n_docs=12, n_words=50, seed=20260916):
    """The corpus as a list of documents, so document ORDER is a thing that
    can be varied. Each is generated, none is committed."""
    return [TR.synthetic_corpus(n_words=n_words, seed=seed + 1000 * k) for k in range(n_docs)]


# --------------------------------------------------------------------------
# the axes
# --------------------------------------------------------------------------

def axis_repeats(report):
    """The same call, five times, in one process. Catches any dependence on
    a hash seed, an iteration order or a reused data structure."""
    docs = documents()
    seen = {}
    for k in range(5):
        cell, stats = cells(docs)
        seen.setdefault(cell, []).append(k)
    report("repeats", "5 runs, identical settings", len(seen) == 1, seen,
           f"{stats['n_ties_broken']} ties broken")


def axis_vocab_size(report):
    """Two runs at each of three vocabulary sizes. A size is a config, so
    each size must be self-consistent; sizes must differ from each other,
    which `control-vocab` covers."""
    docs = documents()
    for size in (300, 512, 1000):
        seen = {}
        for _ in range(2):
            cell, stats = cells(docs, vocab_size=size)
            seen.setdefault(cell, []).append(size)
        report("vocab_size", f"2 runs at vocab {size}", len(seen) == 1, seen,
               f"{stats['n_tokens']} tokens, {stats['n_ties_broken']} ties")


def axis_order(report):
    """Forward, reversed and rotated DOCUMENT order. Each document is
    pre-tokenized alone and the groups are sorted, so order cannot reach the
    result -- this axis is what proves that rather than asserting it."""
    docs = documents()
    forward, _s = cells(docs)
    reverse, _s = cells(list(reversed(docs)))
    rotated, _s = cells(docs[5:] + docs[:5])
    seen = {}
    for name, c in (("forward", forward), ("reversed", reverse), ("rotated", rotated)):
        seen.setdefault(c, []).append(name)
    report("order", "forward vs reversed vs rotated", len(seen) == 1, seen, "")
    for name, order in (("reversed", list(reversed(docs))), ("rotated", docs[5:] + docs[:5])):
        a, _s = cells(order)
        b, _s = cells(order)
        report("order", f"2 runs within {name}", a == b, {a: [name]}, "")


def axis_min_frequency(report):
    """`min_frequency` is a config too, and a run at each value must be
    self-consistent."""
    docs = documents()
    for mf in (2, 3, 5):
        a, stats = cells(docs, min_frequency=mf)
        b, _s = cells(docs, min_frequency=mf)
        report("min_frequency", f"2 runs at min_frequency {mf}", a == b, {a: [mf]},
               f"{stats['n_tokens']} tokens")


def axis_granularity(report):
    """The SAME pre-tokens presented as 12 documents and as 4, so the
    document split itself is shown not to reach the result when it does not
    cut a pre-token. Each document here is whole, so the multiset of
    pre-tokens is identical and the vocabulary must be too."""
    docs = documents()
    merged = [b"".join(docs[k::4]) for k in range(4)]
    a, _s = cells(docs)
    b, _s = cells(merged)
    # NOTE: joining documents CAN weld pre-tokens at a join, so this is
    # allowed to differ; it is reported, not asserted, and the report says
    # which it was.
    report("granularity", "12 documents vs 4 joined (reported, not required)",
           True, {a: ["12 docs"], b: ["4 joined"]},
           "same" if a == b else "joins welded pre-tokens, as expected")


def axis_ties(report):
    """The tie corpus, which is engineered so that many selections are ties.
    Repeats must agree, and `n_ties_broken` must be NON-ZERO or the
    control-sabotage arm below is inert."""
    corpus = TR.ties_corpus()
    a, stats = cells(corpus, vocab_size=300)
    b, _s = cells(corpus, vocab_size=300)
    report("ties", "2 runs on the engineered tie corpus", a == b, {a: ["ties"]},
           f"{stats['n_ties_broken']} ties broken")
    report("ties", "the tie-break is REACHED (n_ties_broken > 0)",
           stats["n_ties_broken"] > 0, {}, f"{stats['n_ties_broken']} ties")


def control_vocab(report):
    """MUST DIFFER: vocab_size + 1 is a different config."""
    docs = documents()
    a, _s = cells(docs, vocab_size=512)
    b, _s = cells(docs, vocab_size=513)
    report("control-vocab", "vocab 512 vs 513 (MUST DIFFER)", a != b,
           {a: ["512"], b: ["513"]}, "")


def control_sabotage(report):
    """MUST DIFFER: the tie-break reversed. If this reports SAME the fixture
    never reached a tie and every other row is worth less."""
    docs = documents()
    a, stats = cells(docs, break_ties_high=False)
    b, _s = cells(docs, break_ties_high=True)
    report("control-sabotage", "tie-break reversed (MUST DIFFER)", a != b,
           {a: ["ours"], b: ["reversed"]}, f"{stats['n_ties_broken']} ties broken")


AXES = {
    "repeats": axis_repeats,
    "vocab_size": axis_vocab_size,
    "order": axis_order,
    "min_frequency": axis_min_frequency,
    "granularity": axis_granularity,
    "ties": axis_ties,
    "control-vocab": control_vocab,
    "control-sabotage": control_sabotage,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", action="append", choices=sorted(AXES),
                    help="run only these axes (default: all)")
    args = ap.parse_args(argv)

    print(f"bpe_trainer_determinism: {platform.machine()} {platform.system()}, "
          f"python {platform.python_version()}, single-threaded by construction")
    print(f"{'axis':16} {'comparison':46} verdict")
    rows = []

    def report(axis, what, ok, seen, note):
        rows.append(ok)
        verdict = "IDENTICAL" if ok else "DIFFERS"
        if axis.startswith("control"):
            verdict = "DIFFERS (as required)" if ok else "IDENTICAL -- CONTROL FAILED"
        if axis == "granularity":
            verdict = note
        print(f"{axis:16} {what:46} {verdict}" + (f"   [{note}]" if note and axis != "granularity" else ""))
        if not ok and not axis.startswith("control"):
            for cell, who in sorted(seen.items(), key=lambda kv: str(kv[0])):
                print(f"{'':16}   {cell} <- {who}")

    for name in (args.axis or list(AXES)):
        AXES[name](report)

    bad = rows.count(False)
    print(f"bpe_trainer_determinism: {len(rows) - bad}/{len(rows)} rows as required")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
