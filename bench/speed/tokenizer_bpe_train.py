#!/usr/bin/env python3
"""Time native BPE vocabulary training on a deterministic text corpus.

Pass a built ``_mojolearn_tokenizer_host`` shared library.  The digest covers
the complete vocabulary arena, token lengths, merge operands, and trainer
statistics, so two source revisions can be compared for both time and exact
output without installing either build into the package.
"""
import argparse
import array
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import os
import statistics
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


def _corpus(n_documents):
    # Natural-language-shaped ASCII with changing numeric/file-name fields;
    # repeated prose exercises ingestion while the fields keep the grouped
    # pre-token set nontrivial.  Unicode samples ensure the table path remains
    # represented in the same timed run.
    prose = (
        b"The quick brown fox analyzes deterministic training output; ",
        b"A worker reads records, validates order, and writes exact bytes. ",
        b"Model verification should be repeatable across every CPU machine. ",
        "Unicode café, Αθήνα, 東京 and naïve text remain in this corpus. ".encode(),
    )
    return [
        prose[k & 3]
        + b"document=" + str(k).encode()
        + b" shard=" + str((k * 17) % 1009).encode()
        + b" path=/data/part-" + str(k % 4093).encode() + b".jsonl\n"
        for k in range(n_documents)
    ]


def _run(module, text, offsets, vocab_size, min_frequency):
    handle = module.bpe_train(
        _address(text) if text else 0,
        _address(offsets),
        [len(offsets) - 1, len(text), vocab_size, min_frequency, False],
    )
    sizes = tuple(int(x) for x in module.bpe_trained_sizes(handle))
    n_tokens, arena_bytes, n_merges, _ties, _groups = sizes
    arena = bytearray(max(arena_bytes, 1))
    lengths = array.array("q", [0]) * n_tokens
    left = array.array("q", [0]) * max(n_merges, 1)
    right = array.array("q", [0]) * max(n_merges, 1)
    module.bpe_trained_copy(
        handle, _address(arena), _address(lengths), _address(left), _address(right)
    )
    h = hashlib.sha256()
    h.update(arena[:arena_bytes])
    h.update(lengths.tobytes())
    h.update(left[:n_merges].tobytes())
    h.update(right[:n_merges].tobytes())
    h.update(array.array("q", sizes).tobytes())
    return h.hexdigest(), sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binding")
    parser.add_argument("--documents", type=int, default=20_000)
    parser.add_argument("--vocab-size", type=int, default=768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    documents = _corpus(args.documents)
    text = bytearray(b"".join(documents))
    offsets = array.array("q", [0])
    for document in documents:
        offsets.append(offsets[-1] + len(document))
    module = _load(os.path.abspath(args.binding))
    elapsed, digest, sizes = [], None, None
    for _ in range(args.repeats):
        start = time.perf_counter()
        got_digest, got_sizes = _run(
            module, text, offsets, args.vocab_size, args.min_frequency
        )
        elapsed.append(time.perf_counter() - start)
        if digest is not None and (got_digest, got_sizes) != (digest, sizes):
            raise RuntimeError("native BPE training changed output between repeats")
        digest, sizes = got_digest, got_sizes
    print(
        f"bytes={len(text)} documents={len(documents)} vocab={args.vocab_size} "
        f"median_s={statistics.median(elapsed):.6f} min_s={min(elapsed):.6f} "
        f"sha256={digest} sizes={sizes} samples="
        + ",".join(f"{x:.6f}" for x in elapsed)
    )


if __name__ == "__main__":
    main()
