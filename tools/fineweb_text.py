#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `text` column of a FineWeb-Edu parquet shard as plain text files.

    python3 tools/fineweb_text.py SHARD.parquet OUT_DIR             # every row group
    python3 tools/fineweb_text.py SHARD.parquet OUT_DIR --groups 20 # the first 20
    python3 tools/fineweb_text.py SHARD.parquet OUT.txt --one-file --groups 20

One file per row group (`rgNNN.txt`), or one file holding them in order. Every
document is its UTF-8 text followed by one 0x0A. Nothing is filtered, sampled
or reordered, so the bytes are a function of the shard alone; the sha256 of the
files' concatenation is printed and is the same in both layouts.

`tokenizer/train/train_main.mojo` takes each file as ONE document, which is how
the GPT-3 Small run's vocabulary was trained (tools/dataset_store.sh,
vocab/mojolearn-bpe-fineweb-edu-50257-v1). Needs pyarrow, which mojolearn does
not depend on.
"""
import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--groups", type=int, default=None, help="the first N row groups (default all)")
    ap.add_argument("--one-file", action="store_true")
    args = ap.parse_args()
    f = pq.ParquetFile(args.shard)
    n = f.metadata.num_row_groups if args.groups is None else args.groups
    h, docs, size = hashlib.sha256(), 0, 0
    one = open(args.out, "wb") if args.one_file else None
    if one is None:
        args.out.mkdir(parents=True, exist_ok=True)
    for g in range(n):
        texts = f.read_row_group(g, columns=["text"]).column("text").to_pylist()
        b = b"".join(s.encode("utf-8") + b"\n" for s in texts)
        h.update(b); docs += len(texts); size += len(b)
        if one is None:
            (args.out / f"rg{g:03d}.txt").write_bytes(b)
        else:
            one.write(b)
    if one is not None:
        one.close()
    print(f"row_groups={n} documents={docs} bytes={size} sha256={h.hexdigest()}")


if __name__ == "__main__":
    main()
