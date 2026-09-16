#!/usr/bin/env python3
"""tools/tokdet_corpus.py -- cut a fixed, reproducible training corpus and a
held-out sample out of one source text file.

Every shard boundary lands on a newline, and the boundary is chosen by
scanning forward from an exact byte offset, so the same source file always
yields the same shards on every machine. Nothing here samples or shuffles.

The held-out sample comes from AFTER the training bytes, so the "does the
tokenization actually differ" layer is not just re-reading training text.

  python3 tools/tokdet_corpus.py --source FILE --out DIR --bytes N --shards S
"""

import argparse
import hashlib
import json
import os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bytes", type=int, default=16 << 20)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--sample-bytes", type=int, default=256 << 10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.source, "rb") as fh:
        raw = fh.read(args.bytes + args.sample_bytes + (1 << 20))

    train = raw[: args.bytes]
    # extend to the next newline so no shard ends mid-line
    nl = raw.find(b"\n", args.bytes)
    if nl != -1:
        train = raw[:nl]
    sample = raw[len(train) + 1 : len(train) + 1 + args.sample_bytes]
    nl = sample.rfind(b"\n")
    if nl != -1:
        sample = sample[:nl]

    step = len(train) // args.shards
    bounds = [0]
    for i in range(1, args.shards):
        at = raw.find(b"\n", i * step)
        bounds.append(at + 1 if at != -1 and at < len(train) else i * step)
    bounds.append(len(train))

    shards = []
    for i in range(args.shards):
        p = os.path.join(args.out, "shard_%02d.txt" % i)
        with open(p, "wb") as fh:
            fh.write(train[bounds[i] : bounds[i + 1]])
        shards.append(p)

    sample_path = os.path.join(args.out, "sample.txt")
    with open(sample_path, "wb") as fh:
        fh.write(sample)

    manifest = {
        "source": os.path.abspath(args.source),
        "source_sha256": sha256_file(args.source),
        "train_bytes": len(train),
        "shards": [
            {"path": p, "bytes": os.path.getsize(p), "sha256": sha256_file(p)} for p in shards
        ],
        "sample": {
            "path": sample_path,
            "bytes": os.path.getsize(sample_path),
            "sha256": sha256_file(sample_path),
        },
    }
    with open(os.path.join(args.out, "corpus_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
