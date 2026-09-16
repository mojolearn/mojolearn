#!/usr/bin/env python3
"""tools/tokdet_train_hf.py -- ONE Hugging Face `tokenizers` vocabulary train,
in its own process, writing byte-comparable artifacts and a record of how it
was configured.

This is a MEASUREMENT harness for the question "is this trainer bitwise
reproducible", so it is written to put its OWN nondeterminism at zero:

  * the initial alphabet is `sorted(...)`. ByteLevel.alphabet() comes back
    from a Rust HashSet, so its order is not promised; feeding that order
    straight to the trainer would inject a difference this harness created
    and then blame the trainer for it.
  * the file list arrives in argv order and is NOT sorted here, because file
    order is one of the axes under test.
  * nothing is seeded, because there is nothing here to seed: any run-to-run
    difference has to come from the trainer.

Thread count is NOT set here. Rayon builds its global pool the first time it
is used and reads RAYON_NUM_THREADS at that moment, so the only honest way to
vary it is a fresh process per setting with the variable already in the
environment. The caller sets it; this script records what it saw.

  python3 tools/tokdet_train_hf.py --out DIR --vocab-size N --model bpe \
      --corpus FILE [FILE ...]
"""

import argparse
import hashlib
import json
import os
import sys
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--model", choices=("bpe", "unigram"), default="bpe")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    os.makedirs(args.out, exist_ok=True)

    # sorted: see the module docstring. This is the harness refusing to be the
    # source of the difference it is looking for.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    specials = ["<unk>"]

    if args.model == "bpe":
        tok = Tokenizer(models.BPE(unk_token=None))
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=specials,
            initial_alphabet=alphabet,
            show_progress=False,
            min_frequency=2,
        )
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=specials,
            initial_alphabet=alphabet,
            show_progress=False,
            unk_token="<unk>",
        )
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    t0 = time.time()
    tok.train(list(args.corpus), trainer)
    elapsed = time.time() - t0

    tok.save(os.path.join(args.out, "tokenizer.json"))
    # model.save() writes the OTHER serialization (vocab.json + merges.txt for
    # BPE, unigram.json for Unigram). Two serializers can disagree about what
    # is stable, so both are recorded.
    model_files = tok.model.save(args.out)

    artifacts = {}
    for name in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, name)
        if os.path.isfile(p) and name != "record.json":
            artifacts[name] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}

    record = {
        "trainer": "huggingface-tokenizers",
        "library_version": tokenizers.__version__,
        "python": sys.version.split()[0],
        "model": args.model,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": tok.get_vocab_size(),
        "corpus_order": list(args.corpus),
        "corpus_sha256": [sha256_file(p) for p in args.corpus],
        "label": args.label,
        "env": {
            k: os.environ.get(k)
            for k in ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
        },
        "elapsed_sec": round(elapsed, 3),
        "model_files": [os.path.basename(p) for p in model_files],
        "artifacts": artifacts,
    }
    with open(os.path.join(args.out, "record.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: record[k] for k in ("vocab_size_actual", "elapsed_sec", "artifacts")}))


if __name__ == "__main__":
    main()
