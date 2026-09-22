#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FineWeb-Edu parquet shards (or plain text) to one pinned token stream.

    python3 tools/fineweb_tokens.py --vocab ranks.tsv --out DIR SHARD.parquet [SHARD.parquet ...]
    python3 tools/fineweb_tokens.py --vocab ranks.tsv --out DIR --text corpus.txt [--held-out-last N]
    python3 tools/fineweb_tokens.py --vocab ranks.tsv --out DIR SHARD.parquet --groups 20   # a sample

Writes `DIR/tokens.i32` (little-endian int32 ids, documents concatenated in
order, no separator inserted) and `DIR/manifest.json` with the schema
`mojolearn.lm_corpus.TokenBatches` reads (`mojolearn.byte-lm.tokens.v1`), so
the stream is a drop-in corpus for `tools/lm_segment.py` and the vocabulary
identity travels with it. The two files are what the R2 dataset store pins.

ONE DOCUMENT IS ONE FINEWEB ROW. `lm_corpus.prepare` cuts a text file into
documents at newlines at most 1 MiB apart because a text file has no other
boundary; a parquet shard has the real one, the row. Each row's `text` is
UTF-8 encoded and encoded ALONE by the pinned vocabulary, as `prepare` does
its documents, and the ids are appended in row order. So the stream is a
function of the shards, their order and the vocabulary, and nothing else.
`--text` takes a file produced by `tools/fineweb_text.py` (one document per
line) for a machine without pyarrow, with the same rule.

STREAMED, NOT HELD. `prepare` keeps the whole id array in memory before it
writes; 2.6B ids is 10.5 GB. This writes row groups as they are encoded and
hashes as it goes, so memory is one row group plus the tokenizer.

THE TRAIN AND HELD-OUT RANGES. With `--held-out-last N` the last N documents
of the stream form the validation range and are never read by
`TokenBatches`, whose schedule covers the train range only; the manifest
records both ranges in tokens. The plan holds out FineWeb-Edu shard 013 (the
short one) by passing it last with `--held-out-last` equal to its row count.

Needs pyarrow for parquet (mojolearn does not depend on it; `pip install
pyarrow` on the box); `--text` needs nothing beyond mojolearn.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TOKENS_SCHEMA = "mojolearn.byte-lm.tokens.v1"


def _documents_from_parquet(paths, groups):
    import pyarrow.parquet as pq
    for path in paths:
        f = pq.ParquetFile(path)
        n = f.metadata.num_row_groups if groups is None else min(groups, f.metadata.num_row_groups)
        for g in range(n):
            texts = f.read_row_group(g, columns=["text"]).column("text").to_pylist()
            yield str(path), g, [s.encode("utf-8") for s in texts]


def _documents_from_text(path, batch):
    """`tools/fineweb_text.py` output: every document is its text plus one
    0x0A, so a document is a line. Newlines inside a FineWeb document were
    already lost when the text file was written; the parquet path keeps them."""
    group, index = [], 0
    with open(path, "rb") as fh:
        for line in fh:
            group.append(line[:-1] if line.endswith(b"\n") else line)
            if len(group) == batch:
                yield str(path), index, group
                group, index = [], index + 1
    if group:
        yield str(path), index, group


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*", help="parquet shards, in order")
    ap.add_argument("--text", help="a fineweb_text.py text file instead of parquet")
    ap.add_argument("--vocab", required=True, help="the pinned rank file (ranks.tsv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=None, help="only the first N row groups of each shard")
    ap.add_argument("--held-out-last", type=int, default=0, help="the last N documents form the validation range")
    ap.add_argument("--text-batch", type=int, default=2048, help="--text: documents per encode batch")
    args = ap.parse_args(argv)
    if bool(args.shards) == bool(args.text):
        ap.error("give parquet shards or --text, not both")
    from mojolearn import tokenizer as tk
    from mojolearn._array import Array
    tok = tk.BpeTokenizer.from_ranks_file(args.vocab)
    identity = tok.identity
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = out / "tokens.i32"
    source = _documents_from_text(args.text, args.text_batch) if args.text else _documents_from_parquet(args.shards, args.groups)
    whole, at_token, n_docs, n_bytes, max_id = hashlib.sha256(), 0, 0, 0, -1
    doc_tokens = []  # token count per document, for the held-out cut
    inputs = []
    encode_seconds, t_start = 0.0, time.perf_counter()
    with stream.open("wb") as fh:
        for path, group, docs in source:
            t0 = time.perf_counter()
            encoded = tok.encode_batch(docs)
            encode_seconds += time.perf_counter() - t0
            for doc, ids in zip(docs, encoded):
                arr = Array.from_list(ids, "<i4")
                raw = arr.tobytes()
                fh.write(raw)
                whole.update(raw)
                at_token += arr.size
                doc_tokens.append(int(arr.size))
                if arr.size:
                    max_id = max(max_id, int(arr.max()))
                n_bytes += len(doc)
            n_docs += len(docs)
            if path not in [i["path"] for i in inputs]:
                inputs.append(dict(path=path, sha256=None))
            secs = time.perf_counter() - t_start
            print("%s group %d: %d documents, %d tokens, %.1f s, %.3f MB/s encode"
                  % (Path(path).name, group, n_docs, at_token, secs, n_bytes / max(encode_seconds, 1e-9) / 1e6), flush=True)
    for entry in inputs:
        h = hashlib.sha256()
        with open(entry["path"], "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        entry["sha256"] = h.hexdigest()
        entry["bytes"] = Path(entry["path"]).stat().st_size
    held = args.held_out_last
    if held < 0 or held > n_docs:
        raise SystemExit("--held-out-last must be within the document count")
    cut = at_token - sum(doc_tokens[n_docs - held:]) if held else at_token
    seconds = time.perf_counter() - t_start
    manifest = dict(
        schema=TOKENS_SCHEMA,
        source=dict(path=";".join(i["path"] for i in inputs), sha256=hashlib.sha256("".join(i["sha256"] for i in inputs).encode()).hexdigest(),
                    bytes=n_bytes, schema="fineweb-edu.parquet-rows.v1" if not args.text else "fineweb_text.lines.v1",
                    inputs=inputs, row_groups_per_shard=args.groups),
        vocabulary=identity,
        encoder="mojolearn.tokenizer.BpeTokenizer.encode_batch (host binding _mojolearn_tokenizer_host, "
                "tokenizer/encoding.mojo); integers and tables only",
        endoftext="not inserted and not recognized (allow_endoftext=False)",
        document_rule="one document per FineWeb-Edu row (or per line of a fineweb_text.py file), each encoded alone, "
                      "ids concatenated in row order",
        n_documents=n_docs, dtype="int32", byte_order="little",
        sha256=whole.hexdigest(), bytes=4 * at_token, tokens=at_token,
        bytes_per_token=(n_bytes / at_token) if at_token else None, max_id=max_id,
        ids_above_255=None,
        train_range=[0, cut], validation_range=[cut, at_token] if held else None,
        schedule="train-range-modulo.v1",
        tokenize_seconds=seconds, encode_seconds=encode_seconds,
        encode_bytes_per_second=(n_bytes / encode_seconds) if encode_seconds else None,
    )
    manifest = {k: v for k, v in manifest.items() if v is not None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote %s: %d documents, %d tokens (%.2f bytes/token), train [0, %d), sha256 %s, %.1f s (%.3f MB/s encode)"
          % (stream, n_docs, at_token, manifest.get("bytes_per_token") or 0, cut, whole.hexdigest()[:16], seconds,
             n_bytes / max(encode_seconds, 1e-9) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
