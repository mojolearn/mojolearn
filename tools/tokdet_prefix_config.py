#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hugging Face BPE's bitwise reproducibility is a property of ONE CONFIGURATION.

The rest of this lane measured `tokenizers` BPE in the byte-level, no-prefix
configuration and found it identical on every axis it could move. This check
moves the axis the rest of the lane never moved, the trainer's own
configuration, and the answer is different.

Three arms, each trained in FRESH PROCESSES, because the cause is a per-process
hash seed and a single process cannot see it:

    plain       BpeTrainer, no prefix          the CONTROL, must be 1 distinct
    prefix      BpeTrainer, continuing_subword_prefix="##"
    wordpiece   WordPieceTrainer

The control is what makes the other two readable, and it is the fail-first arm.
If `plain` also came back with many distinct artifacts, the corpus or this
harness would be the cause and nothing about the configuration could be
concluded from the other two.

Layers, the same separation the rest of the lane uses, because "the files
differ" and "the vocabularies differ" are different findings: the token set,
the token to id map, the merges as a set and as a sequence, and the pieces a
held-out string encodes to.

`tokenizers` is not a mojolearn dependency. Install it into a throwaway venv
outside the repo, and run one core:

    python3 -m venv /Users/andrewhendel/mojolearn-evidence/tokenizer-trainer-determinism/hfenv
    .../hfenv/bin/pip install 'tokenizers==0.23.2'
    RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
        nice -n 19 .../hfenv/bin/python tools/tokdet_prefix_config.py OUTDIR
"""
import hashlib
import json
import os
import subprocess
import sys

N_RUNS = 8
VOCAB_SIZE = 300
ARMS = ("plain", "prefix", "wordpiece")


def _lcg(seed):
    """A 64-bit LCG, so the corpus is a function of this file and nothing else.
    No library, no seed file, no committed corpus."""
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (x >> 33) % 1000000


def write_corpus(path):
    """Words built from shared stems and shared endings, so continuation
    pieces exist and count ties are common. Tie density is what the
    configuration difference acts through, so a corpus that never ties would
    make this check inert."""
    heads = ["work", "play", "read", "walk", "talk", "build", "count", "merge",
             "token", "train", "learn", "check", "stand", "sort", "hash"]
    tails = ["ing", "ed", "er", "ers", "s", "able", "ment", "ion", "ly", ""]
    words = [h + t for h in heads for t in tails]
    rnd = _lcg(20260916)
    lines = []
    for _ in range(4000):
        n = 4 + next(rnd) % 9
        lines.append(" ".join(words[next(rnd) % len(words)] for _ in range(n)))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def train(corpus, out_json, arm):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    if arm == "wordpiece":
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tr = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE,
                                       special_tokens=["[UNK]"], show_progress=False)
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        kw = dict(vocab_size=VOCAB_SIZE, show_progress=False)
        if arm == "prefix":
            kw["continuing_subword_prefix"] = "##"
        tr = trainers.BpeTrainer(**kw)
    tok.train([corpus], tr)
    tok.save(out_json)


def _model(out, arm, i):
    with open(os.path.join(out, f"{arm}.{i}.json"), encoding="utf-8") as fh:
        return json.load(fh)["model"]


def _digest(model):
    """Hash the MODEL, not the file, so a serializer or metadata difference
    cannot be read as a vocabulary difference."""
    payload = json.dumps({"vocab": model.get("vocab"), "merges": model.get("merges")},
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _merges(model):
    raw = model.get("merges") or []
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in raw]


def layers(out, arm, a, b, text):
    from tokenizers import Tokenizer

    ma, mb = _model(out, arm, a), _model(out, arm, b)
    ta = Tokenizer.from_file(os.path.join(out, f"{arm}.{a}.json"))
    tb = Tokenizer.from_file(os.path.join(out, f"{arm}.{b}.json"))
    va, vb = ma["vocab"], mb["vocab"]
    ga, gb = _merges(ma), _merges(mb)
    only_a, only_b = sorted(set(va) - set(vb)), sorted(set(vb) - set(va))
    map_diff = sum(1 for k in set(va) & set(vb) if va[k] != vb[k])
    seq_diff = sum(1 for x, y in zip(ga, gb) if x != y) + abs(len(ga) - len(gb))
    set_diff = len(set(ga) ^ set(gb))
    pieces = sum(1 for t in text if ta.encode(t).tokens != tb.encode(t).tokens)
    ids_only = sum(1 for t in text
                   if ta.encode(t).tokens == tb.encode(t).tokens
                   and ta.encode(t).ids != tb.encode(t).ids)
    print(f"  {arm} run{a} vs run{b}")
    print(f"    token set      {len(only_a)} only in A, {len(only_b)} only in B")
    print(f"    token->id      {map_diff} shared entries differ")
    print(f"    merges SET     {set_diff} symmetric-difference entries")
    print(f"    merges SEQ     {seq_diff} positions differ")
    print(f"    corpus lines   {pieces} differ in PIECES, {ids_only} in ids only"
          f"  (of {len(text)})")
    if not (only_a or only_b):
        return
    # Does a genuinely differing token reach the tokenization? Probe the words
    # the differing tokens came from rather than in-corpus lines, because a
    # tail token can be absent from every line and still be a real difference.
    probes = [t.replace("##", "") for t in only_a + only_b]
    moved = [p for p in probes if ta.encode(p).tokens != tb.encode(p).tokens]
    print(f"    only in A      {only_a}")
    print(f"    only in B      {only_b}")
    print(f"    {len(moved)} of {len(probes)} probe strings encode to different PIECES")
    for p in moved[:3]:
        print(f"      {p!r}  A {ta.encode(p).tokens}  B {tb.encode(p).tokens}")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--train-one":
        out, arm, idx = sys.argv[2], sys.argv[3], sys.argv[4]
        train(os.path.join(out, "corpus.txt"), os.path.join(out, f"{arm}.{idx}.json"), arm)
        return
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    print("corpus sha256", write_corpus(os.path.join(out, "corpus.txt")))
    env = dict(os.environ, RAYON_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    with open(os.path.join(out, "corpus.txt"), encoding="utf-8") as fh:
        text = [l.strip() for l in fh if l.strip()][:500]

    failures = []
    for arm in ARMS:
        digests = []
        for i in range(N_RUNS):
            subprocess.run([sys.executable, __file__, "--train-one", out, arm, str(i)],
                           check=True, env=env)
            digests.append(_digest(_model(out, arm, i)))
        n = len(set(digests))
        print(f"{arm:10s} {n} distinct vocabularies in {N_RUNS} fresh processes")
        layers(out, arm, 0, 1, text)
        if arm == "plain" and n != 1:
            failures.append("the no-prefix CONTROL was not reproducible, so this "
                            "corpus or this harness is the cause and the other two "
                            "arms prove nothing")
        if arm != "plain" and n == 1:
            failures.append(f"the {arm} arm was reproducible, which contradicts the "
                            "recorded finding")
    print()
    for f in failures:
        print("FAIL", f)
    print("PASS tokdet_prefix_config" if not failures else "FAIL tokdet_prefix_config")
    sys.exit(1 if failures else 0)


main()
