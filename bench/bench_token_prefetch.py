#!/usr/bin/env python3
"""Benchmark ordered prefetch from a verified pretokenized mmap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

from mojolearn.lm_corpus import CORPUS_SCHEMA, TOKENS_SCHEMA, TokenBatches


def make_tokens(root, count):
    path = root / "tokens.i32"
    block = bytearray(1 << 20)
    for i in range(0, len(block), 4):
        block[i:i + 4] = ((i // 4) % 32000).to_bytes(4, "little")
    h = hashlib.sha256()
    with path.open("wb") as stream:
        left = count * 4
        while left:
            piece = memoryview(block)[:min(left, len(block))]
            stream.write(piece)
            h.update(piece)
            left -= len(piece)
    manifest = {"schema": TOKENS_SCHEMA, "sha256": h.hexdigest(),
                "bytes": count * 4, "tokens": count,
                "train_range": [0, count],
                "source": {"schema": CORPUS_SCHEMA, "sha256": "0" * 64},
                "vocabulary": {"schema": "bench", "sha256": "1" * 64,
                               "n_vocab": 32000}}
    (root / "manifest.json").write_text(json.dumps(manifest))


def consume(batch, delay):
    # A GPU/native train step releases the GIL.  Sleep models that interval;
    # the digest makes byte identity independently observable.
    time.sleep(delay)


def run(batches, start, steps, delay, prefetched):
    begin = time.perf_counter()
    source = batches.prefetch(start, steps, depth=2) if prefetched else (
        (step, batches.ids(step)) for step in range(start, start + steps))
    for step, batch in source:
        consume(batch, delay)
    return time.perf_counter() - begin


def exact_digest(batches, start, steps, prefetched):
    h = hashlib.sha256()
    source = batches.prefetch(start, steps, depth=2) if prefetched else (
        (step, batches.ids(step)) for step in range(start, start + steps))
    for step, batch in source:
        h.update(step.to_bytes(8, "little"))
        h.update(batch.tobytes())
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=64_000_000)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--length", type=int, default=2048)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--consume-ms", type=float, default=1.5)
args = parser.parse_args()
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    make_tokens(root, args.tokens)
    batches = TokenBatches(root, args.batch, args.length)
    for bad in ((-1, 1, 2), (0, -1, 2), (0, 1, 0), (0, 1, True)):
        try:
            list(batches.prefetch(bad[0], bad[1], depth=bad[2]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid prefetch arguments were accepted: %r" % (bad,))
    sync_sha = exact_digest(batches, 17, 8, False)
    prefetch_sha = exact_digest(batches, 17, 8, True)
    assert sync_sha == prefetch_sha
    samples = []
    for _ in range(5):
        sync = run(batches, 17, args.steps, args.consume_ms / 1000, False)
        ahead = run(batches, 17, args.steps, args.consume_ms / 1000, True)
        samples.append({"sync_seconds": sync, "prefetch_seconds": ahead})
    print(json.dumps({"shape": [args.batch, args.length + 1], "steps": args.steps,
                      "consume_ms": args.consume_ms, "samples": samples,
                      "sha256": sync_sha,
                      "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "pid": os.getpid()}))
