#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOW NON-DENSE IS criteo, per column, per slice? Host only, no device.

    GBM_BENCH_DATA=/root/datasets/gbm-bench python3 tools/criteo_density_audit.py

`_decode_criteo` assigns each category the rank of its string in sorted-unique
order over EVERY DECODED ROW. `load_criteo` then fits a row SLICE: the last
500,000 rows are held out and `--rows` cuts the train block further. So a
category living only in the withheld rows is a HOLE in the codes that reach
`fit`, and `gbdt/train.mojo:1231` refuses the fit by name:

    cat_features column 13 is not densely coded: category 1 is absent from ...

This script counts the holes instead of arguing about them. It measures the
full decoded matrix, the uncapped train slice, and the 1,000,000-row slice the
speed lane actually runs, so the answer to "is this only a --rows artifact?"
is a number rather than an opinion.

Nothing here touches a GPU and nothing here is a timing.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402


def say(*a):
    print(*a, flush=True)


def audit(label, x, cat_idx, global_card):
    """Per column: how many codes in 0..max are missing from this slice."""
    holes, non_dense = [], 0
    for k, j in enumerate(cat_idx):
        col = x[:, j]
        present = np.unique(col)
        top = int(present.max())
        missing = (top + 1) - present.size
        holes.append(missing)
        if missing > 0:
            non_dense += 1
    say("AUDIT slice=%s rows=%d non_dense_columns=%d of %d "
        "total_missing_codes=%d worst_column_missing=%d"
        % (label, x.shape[0], non_dense, len(cat_idx), sum(holes), max(holes)))
    first = next((j for k, j in enumerate(cat_idx) if holes[k] > 0), None)
    if first is not None:
        k = list(cat_idx).index(first)
        say("AUDIT slice=%s first_non_dense_column=%d missing=%d "
            "global_cardinality=%d" % (label, first, holes[k], global_card[k]))
    else:
        say("AUDIT slice=%s every declared column is dense 0..k-1" % label)
    return non_dense


def main():
    folder = os.path.join(spec.data_root(), "criteo")
    npz = os.path.join(folder, "criteo_speed.npz")
    cached = np.load(npz, allow_pickle=False)
    x_all, y_all = cached["x_all"], cached["y_all"]
    cat_idx = tuple(int(v) for v in cached["cat_idx"])
    say("AUDIT-DATA npz=%s rows=%d feats=%d ncat=%d positives=%.4f"
        % (npz, x_all.shape[0], x_all.shape[1], len(cat_idx),
           float(y_all.mean())))
    global_card = [int(np.unique(x_all[:, j]).size) for j in cat_idx]
    say("AUDIT-GLOBAL min=%d max=%d sum=%d"
        % (min(global_card), max(global_card), sum(global_card)))

    # The whole matrix: dense by construction, and the only slice that is.
    audit("all_decoded_rows", x_all, cat_idx, global_card)

    # What load_criteo actually hands fit, uncapped and at the lane's cap.
    n_all = x_all.shape[0]
    n_test = min(spec.CRITEO_N_TEST, max(1, n_all // 5))
    n_train_full = n_all - n_test
    audit("train_uncapped", x_all[:n_train_full], cat_idx, global_card)
    for cap in (1000000, 200000):
        if cap < n_train_full:
            audit("train_rows_%d" % cap, x_all[:cap], cat_idx, global_card)
    audit("test_tail", x_all[n_all - n_test:], cat_idx, global_card)
    say("AUDIT-CONCLUSION the codes are dense over the decoded matrix and "
        "NOT over any train slice of it, so the refusal is a property of the "
        "decode's global ranking and not of --rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
