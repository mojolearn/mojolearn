#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE INTEROP CLAIM, MEASURED: does our emitted `tokenizer.json` load in
Hugging Face `tokenizers` and tokenize a held-out sample IDENTICALLY to us?

    python3 tools/bpe_trainer_interop.py BUILD_DIR --python HFENV/bin/python

`tokenizers` is NOT a mojolearn dependency and is not installed into the
repo's environment. `--python` names an interpreter in a THROWAWAY
environment that has it; this script re-executes itself under that
interpreter for the Hugging Face half. Nothing here ships in a wheel.

WHY THIS IS A MEASUREMENT AND NOT A FORMALITY. Our pre-tokenizer is the
GPT-2 pattern with POSSESSIVE quantifiers and `\\s++$` implemented as END OF
HAYSTACK. Hugging Face's `ByteLevel(use_regex=true)` runs its own GPT-2
regex, in which `$` is an ordinary regex anchor. Those two readings can
disagree on text ending in a newline, so "the ecosystem can read our file"
is not the same claim as "the ecosystem tokenizes the same way". This script
settles the second one, on held-out text the vocabulary was not trained on.

TWO SHAPES ARE COMPARED, because the choice is a real one:
  split      Sequence[ Split(OUR literal pattern), ByteLevel(use_regex=false) ]
             -- what we emit. Carries the pattern in the file.
  bytelevel  ByteLevel(use_regex=true)
             -- the ecosystem-standard shape, which relies on Hugging Face's
                regex agreeing with ours rather than stating ours.

THE CONTROL. A comparison that reports IDENTICAL because it compared nothing
is worthless, so `--self-test` perturbs one merge of a real trained
vocabulary and requires the comparison to FAIL. Run it before believing a
pass.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TR = _load("_bpe_trainer", os.path.join(HERE, "python", "mojolearn", "_bpe_trainer.py"))
SYN = _load("_tokenizer_synthetic", os.path.join(HERE, "python", "mojolearn", "_tokenizer_synthetic.py"))


def read_ranks(path):
    tokens = []
    with open(path, "r", encoding="ascii") as fh:
        for lineno, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            rank, hexbytes = line.split("\t")
            if int(rank) != lineno:
                raise ValueError(f"{path}:{lineno + 1}: rank {rank} out of order")
            tokens.append(bytes.fromhex(hexbytes))
    return tokens


NUL, SOH, DEL, FS = chr(0), chr(1), chr(0x7F), chr(0x1C)
ADVERSARIAL = {
    "trailing_nl": "abc" + chr(10),
    "trailing_nl2": "abc" + chr(10) * 2,
    "trailing_sp": "a   ",
    "interior_ws": "a  b   c",
    "single_sp": "a b",
    "sp_then_nl": "a " + chr(10),
    "contr_low": "it's we'll they're I'd",
    "contr_up": "IT'S WE'LL",
    "digits": "2026 123 0",
    "arabic_digits": "١٢٣ ٤",
    "latin1": " café zürn é",
    "combining": "é café",
    "cjk": "中文 中文字",
    "symbols": "€ (x) ... !! ??",
    "emoji": "ok \U0001F642\U0001F642 ok",
    "ws_runs": "a   b" + chr(9) * 2 + "c" + chr(10) * 2 + "d",
    "nbsp_fs": "a b" + FS + "c",
    "controls": NUL + SOH + DEL,
    "nl_then_sp": "a" + chr(10) + "  ",
    "tab_tail": "a" + chr(9),
    "only_ws": "   ",
    "two_lines": "line1" + chr(10) + "line2" + chr(10),
    "crlf": "a" + chr(13) + chr(10) + "b" + chr(13) + chr(10),
}


def held_out_samples():
    """Text the vocabularies were NOT trained on: a differently seeded draw
    from the same generator, split into lines, plus the adversarial set. The
    held-out half is what makes this more than re-reading training text."""
    out = dict(ADVERSARIAL)
    text = TR.synthetic_corpus(n_words=180, seed=31337).decode("utf-8")
    for k, chunk in enumerate(text.split("\n")):
        if chunk.strip():
            out[f"heldout_{k:02d}"] = chunk
    return out


# --------------------------------------------------------------------------
# the Hugging Face half, run under --python
# --------------------------------------------------------------------------

def hf_main(argv):
    from tokenizers import Tokenizer

    req = json.load(sys.stdin)
    spec = json.loads(req["tokenizer_json"])
    if req["shape"] == "bytelevel":
        spec["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False,
                                 "trim_offsets": True, "use_regex": True}
    tok = Tokenizer.from_str(json.dumps(spec))
    out = {}
    for name, text in req["samples"].items():
        enc = tok.encode(text, add_special_tokens=False)
        out[name] = {"ids": enc.ids, "tokens": enc.tokens,
                     "decoded_ok": tok.decode(enc.ids) == text}
    import tokenizers
    json.dump({"version": tokenizers.__version__, "results": out}, sys.stdout)
    return 0


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------

def run_hf(python, tokenizer_json, samples, shape):
    req = json.dumps({"tokenizer_json": tokenizer_json, "samples": samples, "shape": shape})
    p = subprocess.run([python, os.path.abspath(__file__), "--hf-child"],
                       input=req, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"the Hugging Face half failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout)


def compare(name, tokens, tokenizer_json, samples, python, shape):
    hf = run_hf(python, tokenizer_json, samples, shape)
    n_same = n_diff = 0
    first = []
    for sname, text in sorted(samples.items()):
        raw = text.encode("utf-8")
        ours_ids = SYN.reference_encode(tokens, raw, False)
        theirs = hf["results"][sname]
        if list(theirs["ids"]) == list(ours_ids):
            n_same += 1
        else:
            n_diff += 1
            if len(first) < 4:
                ours_pieces = [TR.spell(tokens[i]) for i in ours_ids]
                first.append((sname, ours_pieces, theirs["tokens"]))
    return hf["version"], n_same, n_diff, first


def _drop_first_merge(spec):
    """The most frequent pair in the corpus stops being a merge. Anything
    that used it retokenizes."""
    spec["model"]["merges"] = spec["model"]["merges"][1:]
    return "drop the most frequent merge"


def _truncate_merges(spec):
    """Almost the whole merge table goes away."""
    spec["model"]["merges"] = spec["model"]["merges"][:5]
    return "keep only the first 5 merges"


def _swap_two_ids(spec):
    """A pure RENUMBERING: the same pieces, two ids exchanged. Catches a
    comparison that comes down to pieces and never looks at ids."""
    v = spec["model"]["vocab"]
    ka, kb = TR.spell(b"a"), TR.spell(b"b")
    v[ka], v[kb] = v[kb], v[ka]
    return "swap the ids of 'a' and 'b'"


#: Each must make the comparison FAIL. A perturbation that leaves every
#: sample untouched is not a control -- the first version of this self-test
#: swapped two ADJACENT merges, which changed nothing, because merges 2 and 3
#: never compete on these samples and BPE applies them by rank. It reported
#: "the comparison can fail" while proving the opposite. Each perturbation
#: here has to reach the samples, and each is checked SEPARATELY so one
#: strong arm cannot cover for a dead one.
PERTURBATIONS = (_drop_first_merge, _truncate_merges, _swap_two_ids)


def self_test(build_dir, python):
    """THE CONTROL, watched failing before any pass below is believed."""
    path = os.path.join(build_dir, "synthetic.tokenizer.json")
    tokens = read_ranks(os.path.join(build_dir, "synthetic.ranks.tsv"))
    samples = held_out_samples()
    bad = 0
    for perturb in PERTURBATIONS:
        spec = json.loads(open(path, encoding="ascii").read())
        what = perturb(spec)
        _v, n_same, n_diff, _f = compare("self-test", tokens, json.dumps(spec),
                                         samples, python, "split")
        total = n_same + n_diff
        if n_diff == 0:
            print(f"  CONTROL {what:32} 0/{total} differ  <-- INERT; this arm proves nothing")
            bad += 1
        else:
            print(f"  CONTROL {what:32} {n_diff}/{total} differ  (the comparison can fail)")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--python", required=True,
                    help="an interpreter in a throwaway environment that has `tokenizers`")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    samples = held_out_samples()
    print(f"bpe_trainer_interop: {len(samples)} held-out samples "
          f"({len(ADVERSARIAL)} adversarial, {len(samples) - len(ADVERSARIAL)} generated)")

    bad = 0
    if args.self_test:
        bad += self_test(args.build_dir, args.python)

    for name, _c, _v, _m in TR.FIXTURES:
        tj = os.path.join(args.build_dir, f"{name}.tokenizer.json")
        if not os.path.isfile(tj):
            print(f"FAIL {name}: {tj} does not exist")
            bad += 1
            continue
        tokens = read_ranks(os.path.join(args.build_dir, f"{name}.ranks.tsv"))
        text = open(tj, encoding="ascii").read()
        for shape in ("split", "bytelevel"):
            version, n_same, n_diff, first = compare(name, tokens, text, samples, args.python, shape)
            tag = "OK  " if n_diff == 0 else "FAIL"
            print(f"  {tag} {name:16} {shape:10} tokenizers {version}: "
                  f"{n_same}/{n_same + n_diff} samples tokenize identically to ours")
            for sname, ours_pieces, their_pieces in first:
                print(f"       {sname}: ours={ours_pieces}")
                print(f"       {' ' * len(sname)}  theirs={their_pieces}")
            if shape == "split" and n_diff:
                bad += 1
    print("bpe_trainer_interop: " + ("PASS" if bad == 0 else f"{bad} FAILURES"))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--hf-child" in sys.argv:
        sys.exit(hf_main(sys.argv))
    sys.exit(main())
