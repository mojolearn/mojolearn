#!/usr/bin/env python3
"""Time a built tokenizer host binding on a deterministic text batch.

Build two source revisions into separate directories and invoke this once for
each directory.  The printed digest covers both token ids and per-document
counts, so an A/B timing also proves that document boundaries and results
agree.
"""
import argparse
import array
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import os
import time


def _load(path):
    name = "_mojolearn_tokenizer_host"
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _address(value):
    if hasattr(value, "buffer_info"):
        return value.buffer_info()[0]
    return ctypes.addressof(ctypes.c_char.from_buffer(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binding", help="path to _mojolearn_tokenizer_host.so")
    parser.add_argument("ranks", help="rank<TAB>hex vocabulary")
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    module = _load(os.path.abspath(args.binding))
    handle = module.bpe_load(os.path.abspath(args.ranks))
    samples = (b" hello world 2026!", b" kalomine rusato", b"caf\xc3\xa9 and text",
               b"12345...", b"short document")
    documents = [samples[k % len(samples)] for k in range(args.documents)]
    text = bytearray(b"".join(documents))
    offsets = array.array("q", [0])
    for document in documents:
        offsets.append(offsets[-1] + len(document))
    ids = array.array("i", [0]) * max(len(text), 1)
    counts = array.array("q", [0]) * len(documents)
    dims = [len(documents), len(text), len(text), False]

    elapsed = []
    total = 0
    for _ in range(args.repeats):
        start = time.perf_counter()
        total = module.bpe_encode_batch(
            handle, _address(text), _address(offsets), _address(ids),
            _address(counts), dims)
        elapsed.append(time.perf_counter() - start)
    elapsed.sort()
    digest = hashlib.sha256(ids[:total].tobytes() + counts.tobytes()).hexdigest()
    print(f"bytes={len(text)} documents={len(documents)} ids={total} "
          f"median_s={elapsed[len(elapsed) // 2]:.6f} sha256={digest}")


if __name__ == "__main__":
    main()
