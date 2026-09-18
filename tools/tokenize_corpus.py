#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tokenize a pinned byte corpus ONCE, into a pinned int32 id array.

    python3 tools/tokenize_corpus.py \
        --corpus training/corpus/enwik8/input.txt \
        --ranks  <ranks.tsv of a trained vocabulary> \
        --vocabulary-name mojolearn-bpe-50257-v1 \
        --out    training/corpus/enwik8/tokens/mojolearn-bpe-50257-v1

writes `tokens.i32` (raw int32, little-endian, no header) and `manifest.json`
beside it, schema `mojolearn.byte-lm.tokens.v1`.

WHY A ONE-TIME ARTIFACT. Tokenizing is pure, deterministic and slow enough to
be worth doing once: `tools/tokenize_corpus.py --measure` prints the rate.
Every training run then reads ids, not text, and a run's data is pinned by one
sha256 exactly the way the byte corpus is.

WHY THE SCHEMA IS ITS OWN. `mojolearn.byte-lm.corpus.v1` describes a single
raw-byte file with `vocabulary: 256`; every id in it is its own byte, so the
file IS the ids and nothing else is needed to read it. An id array is
meaningless without the vocabulary that produced it, so this schema carries
the vocabulary's own sha256, its rank count and the recipe that made it. Two
id arrays with the same bytes and different vocabularies are different
corpora, and a manifest that could not tell them apart would be a manifest
that cannot fail. The name keeps the `byte-lm` family prefix because it names
the TRAINER that consumes it (`training/byte_lm.mojo`,
`mojolearn.LanguageModelTrainer`), which is vocabulary-agnostic and always
was -- it validates `0 <= id < vocab_size` and computes gradients for every
embedding and unembedding row. The alphabet is what changes, not the trainer.

DOCUMENTS, WHICH ARE PART OF THE ARTIFACT AND NOT AN IMPLEMENTATION DETAIL.
A byte-level BPE encoder never merges across a pre-token boundary, and the
GPT-2 pre-tokenizer's pattern can take a whitespace run or a word across any
byte offset you might pick to cut a 100 MB file into pieces. So the cut is
not an internal chunking convenience that happens to be invisible; it CHANGES
IDS at every boundary. It is therefore declared:

  1. The source manifest's `train_range`, `validation_range` and `test_range`
     boundaries are cuts. A document never spans a split, so the token index
     of each split is EXACT rather than approximate, and a reader that honours
     the split reads no test byte.
  2. Inside a range, a document is at most `--document-bytes` bytes and ends
     at the LAST 0x0A at or before that limit. A window holding no 0x0A is cut
     at the limit exactly.
  3. Each document is encoded ALONE (`encode_batch(docs)[k] == encode(docs[k])`
     id for id, whatever else is in the batch), and the ids are concatenated
     in document order.

`<|endoftext|>` is NOT inserted between documents and is not recognized in the
text (`allow_endoftext=False`): the byte corpora are continuous streams, not
collections whose boundaries a model should be told about, and inserting a
separator our byte path does not insert would make the two paths differ by
more than the alphabet. The id is reserved and reachable; nothing here emits
it. That is a choice, recorded in the manifest as `endoftext`.

NOTHING HERE IS COMMITTED TO THE REPOSITORY. The id array goes to R2 beside
the corpora (`tools/dataset_store.sh`); only the manifest, which is small and
ours, is committed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TOKENS_SCHEMA = "mojolearn.byte-lm.tokens.v1"
CORPUS_SCHEMA = "mojolearn.byte-lm.corpus.v1"
#: the default document limit: big enough that the artificial boundaries are
#: rare (about 96 of them in enwik8), small enough that one encode call's id
#: buffer is bounded.
DEFAULT_DOCUMENT_BYTES = 1 << 20
#: documents per `encode_batch` call
DEFAULT_BATCH_DOCUMENTS = 16


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_ranks(path):
    """`(n_ranks, sha256, bytes)` of a `rank<TAB>hex` file, checked for the
    shape `GPT2Tokenizer.from_ranks_file` needs: ascending ranks from 0."""
    raw = Path(path).read_bytes()
    lines = raw.decode("ascii").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for k, line in enumerate(lines):
        rank, _, _hex = line.partition("\t")
        if int(rank) != k:
            raise ValueError("%s:%d: rank %s is not %d; the file must ascend from 0" % (path, k + 1, rank, k))
    return len(lines), sha256_bytes(raw), len(raw)


def split_points(manifest, n_bytes):
    """The byte offsets a document may never span: 0, n_bytes, and every
    range boundary the source manifest declares."""
    points = {0, n_bytes}
    for key in ("train_range", "validation_range", "test_range"):
        r = manifest.get(key)
        if not r:
            continue
        for x in r:
            if 0 <= int(x) <= n_bytes:
                points.add(int(x))
    return sorted(points)


def documents(data, manifest, document_bytes):
    """`[(start, stop)]` by the declared rule. Half-open, contiguous, in
    order, covering the whole file."""
    out = []
    points = split_points(manifest, len(data))
    for lo, hi in zip(points, points[1:]):
        at = lo
        while at < hi:
            stop = min(at + document_bytes, hi)
            if stop < hi:
                cut = data.rfind(b"\n", at, stop)
                if cut != -1 and cut + 1 > at:
                    stop = cut + 1
            out.append((at, stop))
            at = stop
    return out


def tokenize(corpus_path, ranks_path, out_dir, vocabulary_name, vocabulary_extra,
             document_bytes, batch_documents, limit_bytes=None, write=True, progress=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    from mojolearn.tokenizer import GPT2Tokenizer

    corpus_path = Path(corpus_path)
    source_manifest_path = corpus_path.with_name("manifest.json")
    source_manifest = json.loads(source_manifest_path.read_bytes())
    if source_manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError("%s is not %s" % (source_manifest_path, CORPUS_SCHEMA))
    data = corpus_path.read_bytes()
    source_sha = sha256_bytes(data)
    if source_sha != source_manifest.get("sha256") or len(data) != source_manifest.get("bytes"):
        raise ValueError("pinned corpus length/SHA mismatch for %s" % corpus_path)
    if limit_bytes is not None:
        data = data[:limit_bytes]

    n_ranks, ranks_sha, ranks_bytes = read_ranks(ranks_path)
    tok = GPT2Tokenizer.from_ranks_file(os.fspath(ranks_path))
    if tok.n_vocab != n_ranks + 1:
        raise ValueError("ranks file has %d ranks but the tokenizer reports n_vocab %d" % (n_ranks, tok.n_vocab))

    docs = documents(data, source_manifest if limit_bytes is None else {}, document_bytes)
    boundaries = {}          # byte offset -> token offset, for every split point
    for p in split_points(source_manifest if limit_bytes is None else {}, len(data)):
        boundaries[p] = None
    import numpy as np

    chunks = []
    at_token, n_docs_done = 0, 0
    min_id, max_id = None, None
    above_255 = 0
    encode_seconds = 0.0
    start_wall = time.perf_counter()
    for k in range(0, len(docs), batch_documents):
        group = docs[k:k + batch_documents]
        t0 = time.perf_counter()
        encoded = tok.encode_batch([bytes(data[lo:hi]) for lo, hi in group])
        encode_seconds += time.perf_counter() - t0
        for (lo, _hi), ids in zip(group, encoded):
            if lo in boundaries:
                boundaries[lo] = at_token
            arr = np.asarray(ids, dtype=np.int32)
            chunks.append(arr)
            at_token += arr.size
            if arr.size:
                lo_id, hi_id = int(arr.min()), int(arr.max())
                min_id = lo_id if min_id is None else min(min_id, lo_id)
                max_id = hi_id if max_id is None else max(max_id, hi_id)
                above_255 += int((arr > 255).sum())
        n_docs_done += len(group)
        if progress is not None:
            progress(n_docs_done, len(docs), docs[min(k + batch_documents, len(docs)) - 1][1],
                     at_token, time.perf_counter() - start_wall)
    boundaries[len(data)] = at_token
    seconds = time.perf_counter() - start_wall

    ids_out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    del chunks
    payload = ids_out.astype("<i4", copy=False).tobytes()
    tokens_sha = sha256_bytes(payload)

    def token_range(key):
        r = source_manifest.get(key)
        if not r or limit_bytes is not None:
            return None
        lo, hi = int(r[0]), int(r[1])
        if lo not in boundaries or hi not in boundaries:
            return None
        return [boundaries[lo], boundaries[hi]]

    vocabulary = dict(name=vocabulary_name, ranks_sha256=ranks_sha, ranks_bytes=ranks_bytes,
                      n_ranks=n_ranks, n_vocab=tok.n_vocab, endoftext_id=tok.eot_token)
    vocabulary.update(vocabulary_extra or {})
    manifest = dict(
        schema=TOKENS_SCHEMA,
        corpus=corpus_path.parent.name,
        source=dict(schema=CORPUS_SCHEMA, path=str(corpus_path), sha256=source_sha, bytes=len(data),
                    source_url=source_manifest.get("source_url"),
                    manifest_sha256=sha256_bytes(source_manifest_path.read_bytes())),
        vocabulary=vocabulary,
        encoder="mojolearn.tokenizer.GPT2Tokenizer.encode_batch through the host binding "
                "_mojolearn_tokenizer_host (tokenizer/encoding.mojo); integers and tables only",
        endoftext="not inserted and not recognized (allow_endoftext=False); the byte corpora are "
                  "continuous streams and the byte path inserts no separator either",
        document_rule="documents never span a source range boundary (train/validation/test); inside a "
                      "range a document is at most %d bytes and ends at the last 0x0A at or before that "
                      "limit, or at the limit exactly when the window holds no 0x0A; each document is "
                      "encoded ALONE and the ids are concatenated in document order" % document_bytes,
        document_bytes=document_bytes,
        n_documents=len(docs),
        dtype="int32", byte_order="little",
        sha256=tokens_sha, bytes=len(payload), tokens=at_token,
        bytes_per_token=(len(data) / at_token) if at_token else None,
        min_id=min_id, max_id=max_id,
        ids_above_255=above_255,
        train_range=token_range("train_range"),
        validation_range=token_range("validation_range"),
        test_range=token_range("test_range"),
        byte_to_token_boundaries={str(k): v for k, v in sorted(boundaries.items())},
        train_batch_schedule="step k zero-based, row b: ids[(k*batch*length + b*length) % "
                             "(train_tokens - length - 1) + train_range[0] : +length+1]; targets shifted "
                             "one id; the modulus is the TRAIN range, so a run of any length reads no "
                             "validation or test id (tools/lm_step_memory_probe.py --tokens)",
        tool="tools/tokenize_corpus.py",
        tokenize_seconds=seconds,
        tokenize_encode_seconds=encode_seconds,
        tokenize_bytes_per_second=(len(data) / seconds) if seconds else None,
        encode_bytes_per_second=(len(data) / encode_seconds) if encode_seconds else None,
        scope="a one-time pinned artifact; the id array itself lives in R2 beside the corpora "
              "(tools/dataset_store.sh) and is not committed",
    )
    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokens.i32").write_bytes(payload)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="the pinned byte corpus file")
    ap.add_argument("--ranks", type=Path, required=True, help="rank<TAB>hex vocabulary file")
    ap.add_argument("--vocabulary-name", required=True)
    ap.add_argument("--vocabulary-json", type=Path, default=None,
                    help="the vocabulary's own manifest; its fields are folded into the tokens manifest")
    ap.add_argument("--out", type=Path, required=True, help="output directory for tokens.i32 + manifest.json")
    ap.add_argument("--document-bytes", type=int, default=DEFAULT_DOCUMENT_BYTES)
    ap.add_argument("--batch-documents", type=int, default=DEFAULT_BATCH_DOCUMENTS)
    ap.add_argument("--measure", type=int, default=None, metavar="BYTES",
                    help="tokenize only the first BYTES and write nothing; prints the rate")
    args = ap.parse_args()
    extra = {}
    if args.vocabulary_json is not None:
        extra = json.loads(args.vocabulary_json.read_bytes())

    def progress(done, total, at_byte, at_token, seconds):
        rate = at_byte / seconds if seconds else 0.0
        print("  %5d/%d documents  %12d bytes  %12d ids  %7.1f s  %8.3f MB/s"
              % (done, total, at_byte, at_token, seconds, rate / 1e6), flush=True)

    manifest, payload = tokenize(
        args.corpus, args.ranks, args.out, args.vocabulary_name, extra,
        args.document_bytes, args.batch_documents,
        limit_bytes=args.measure, write=args.measure is None, progress=progress)
    print(json.dumps({k: v for k, v in manifest.items() if k != "byte_to_token_boundaries"}, indent=2))


if __name__ == "__main__":
    main()
