#!/usr/bin/env python3
"""tools/tokdet_train_sp.py -- ONE SentencePiece vocabulary train, in its own
process, writing byte-comparable artifacts and a record of how it was
configured.

Two knobs here are not cosmetic and are exposed on purpose:

  * --shuffle / --seed. SentencePiece samples its training sentences when
    input_sentence_size > 0, and with shuffle_input_sentence on that sample is
    drawn from a random generator. That is a KNOWN and documented source of
    run-to-run difference, not the subtle kind we are hunting, so the matrix
    pins it off and runs one separate arm with it on to show the harness can
    see it.
  * --threads. SentencePiece takes its own --num_threads; it does not read
    RAYON_NUM_THREADS. Recorded either way.

  python3 tools/tokdet_train_sp.py --out DIR --vocab-size N --model-type bpe \
      --threads N --corpus FILE [FILE ...]
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
    ap.add_argument("--model-type", choices=("bpe", "unigram"), default="bpe")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--input-sentence-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import sentencepiece as spm

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, "sp")

    kwargs = dict(
        input=list(args.corpus),
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        num_threads=args.threads,
        character_coverage=0.9995,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=bool(args.shuffle),
        train_extremely_large_corpus=False,
        minloglevel=2,
    )
    if args.seed is not None:
        kwargs["random_seed"] = args.seed

    t0 = time.time()
    spm.SentencePieceTrainer.train(**kwargs)
    elapsed = time.time() - t0

    artifacts = {}
    for name in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, name)
        if os.path.isfile(p) and name != "record.json":
            artifacts[name] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}

    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    record = {
        "trainer": "sentencepiece",
        "library_version": spm.__version__,
        "python": sys.version.split()[0],
        "model": args.model_type,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": sp.get_piece_size(),
        "threads": args.threads,
        "shuffle_input_sentence": bool(args.shuffle),
        "input_sentence_size": args.input_sentence_size,
        "random_seed": args.seed,
        "corpus_order": list(args.corpus),
        "corpus_sha256": [sha256_file(p) for p in args.corpus],
        "label": args.label,
        "elapsed_sec": round(elapsed, 3),
        "artifacts": artifacts,
    }
    with open(os.path.join(args.out, "record.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: record[k] for k in ("vocab_size_actual", "elapsed_sec", "artifacts")}))


if __name__ == "__main__":
    main()
