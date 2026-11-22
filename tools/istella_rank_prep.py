#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Istella-S as the ranking dataset it is: the query ids the decoded npz drops.

`gbm-bench/istella/istella_speed.npz` (what every tree arm reads) was decoded
by `speed_gbdt_arm._decode_letor`, which strips `qid:` and keeps only the
grade and the 220 features, and keeps only the first 500,000 test rows. A
ranking fit needs the query of every row and a test split made of whole
queries, so this writes a SIDE FILE, `istella_rank.npz`, holding

    qid_train  int64 [2,043,304]   query id of every train.txt row, in order
    qid_test   int64 [681,250]     query id of every test.txt row, in order
    x_test     float32 [681,250, 220]  ALL of test.txt, decoded exactly as
                                        `_decode_letor` decodes it
    r_test     float32 [681,250]   the grade of every test.txt row

The train FEATURES are not duplicated: they stay in `istella_speed.npz`.

    python3 tools/istella_rank_prep.py <folder with train.txt, test.txt> \
        <istella_speed.npz> <out istella_rank.npz>

Refuses to write unless (1) the train grades parsed here equal the npz's
`r_train` element for element, (2) the first 500,000 decoded test rows equal
the npz's `x_test` and `r_test` bit for bit, and (3) the rows of every query
are consecutive in both files (CatBoost's and our `group_id` contract). So the
side file is aligned with the features the arms read, by construction.
"""

import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import speed_gbdt_arm as spec  # noqa: E402

_PREFIX = re.compile(rb"^(\d+) qid:(\d+) ", re.MULTILINE)


def grades_and_qids(path):
    """Grade and query id of every line, streaming 64 MB at a time."""
    rs, qs = [], []
    tail = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(64 << 20)
            if not block:
                break
            block = tail + block
            cut = block.rfind(b"\n")
            if cut < 0:
                tail = block
                continue
            chunk, tail = block[:cut + 1], block[cut + 1:]
            pairs = _PREFIX.findall(chunk)
            n_lines = chunk.count(b"\n")
            if len(pairs) != n_lines:
                raise RuntimeError("%s: %d prefixes for %d lines"
                                   % (path, len(pairs), n_lines))
            arr = np.array(pairs, dtype=np.int64)
            rs.append(arr[:, 0])
            qs.append(arr[:, 1])
    if tail.strip():
        raise RuntimeError("%s: trailing partial line" % path)
    return np.concatenate(rs), np.concatenate(qs)


def check_consecutive(name, qid):
    change = np.flatnonzero(np.diff(qid) != 0) + 1
    starts = np.concatenate([[0], change])
    firsts = qid[starts]
    if np.unique(firsts).size != firsts.size:
        raise RuntimeError("%s: a query's rows are not consecutive" % name)
    sizes = np.diff(np.concatenate([starts, [qid.size]]))
    print("%s: %d rows, %d queries, rows per query min %d median %d max %d"
          % (name, qid.size, starts.size, sizes.min(), int(np.median(sizes)),
             sizes.max()))
    return sizes


def main(argv):
    folder, speed_npz, out = argv[1:4]
    train_txt = spec._find_file(folder, "train.txt")
    test_txt = spec._find_file(folder, "test.txt")
    speed = np.load(speed_npz)

    r_tr_txt, qid_train = grades_and_qids(train_txt)
    r_tr_npz = speed["r_train"]
    if r_tr_txt.shape[0] != r_tr_npz.shape[0] or not np.array_equal(
            r_tr_txt.astype(np.float32), r_tr_npz):
        raise SystemExit("REFUSED: train.txt grades differ from the npz r_train")
    print("train grades equal the npz r_train on all %d rows" % r_tr_npz.size)
    check_consecutive("train", qid_train)

    x_te, r_te = spec._decode_letor(test_txt, spec.ISTELLA_FEATURES)
    r_te_chk, qid_test = grades_and_qids(test_txt)
    if not np.array_equal(r_te_chk.astype(np.float32), r_te):
        raise SystemExit("REFUSED: test grades from the two parsers differ")
    n = speed["x_test"].shape[0]
    if not (np.array_equal(x_te[:n], speed["x_test"])
            and np.array_equal(r_te[:n], speed["r_test"])):
        raise SystemExit("REFUSED: decoded test rows differ from the npz test")
    print("first %d decoded test rows equal the npz x_test and r_test bit for bit"
          % n)
    check_consecutive("test", qid_test)

    np.savez(out, qid_train=qid_train, qid_test=qid_test, x_test=x_te,
             r_test=r_te)
    print("wrote %s (%d bytes)" % (out, os.path.getsize(out)))


if __name__ == "__main__":
    main(sys.argv)
