# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""miv_quality.py: quality-only check for the lane/apple-fast-miv defines
(x_prep/dmi_fast.mojo MI_REG_TIES / MI_REG_RANKMAJOR / MI_CLF_RANKMAJOR /
MI_FAST_FOLDS; MI_ALL deleted 2026-10-09). It times nothing.

  python tools/miv_quality.py dump <out.npz>          scores with the installed .so
  python tools/miv_quality.py cmp <off.npz> <on.npz> <label>

Fixture (elementwise, seed 0, no BLAS): 30k rows x 48 columns, a third
continuous, a third low-cardinality integer codes (long x-runs, the TIES
case), a third sparse (70% exact zeros). Targets: a continuous y, a 5-value
y (istella's ranking labels) and a 2-class y. Gate: the k = d // 2 selected
set identical and the scores' max relative difference < 1e-4."""
import sys

import numpy as np


def _data(n=30_000, d=48, seed=0):
    r = np.random.default_rng(seed)
    X = np.empty((n, d), dtype=np.float64)
    for j in range(d):  # glue: fixture columns
        t = j % 3
        if t == 0:
            X[:, j] = r.standard_normal(n)
        elif t == 1:
            X[:, j] = r.integers(0, 3 + j % 7, n)
        else:
            X[:, j] = np.where(r.random(n) < 0.7, 0.0, r.exponential(1.0, n))
    w = (np.arange(d) % 5 - 2) / 4.0
    s = np.zeros(n)
    for j in range(d):  # elementwise sum, no BLAS
        s += w[j] * np.tanh(X[:, j])
    y_c = s + 0.5 * r.standard_normal(n)
    y_5 = np.digitize(y_c, np.quantile(y_c, [0.2, 0.4, 0.6, 0.8])).astype(np.float64)
    y_2 = (y_c > np.median(y_c)).astype(np.int32)
    return X.astype(np.float32), y_c.astype(np.float32), y_5.astype(np.float32), y_2


def dump(out):
    from mojolearn._expansion_prep import mutual_info_classif, mutual_info_regression
    X, yc, y5, y2 = _data()
    res = {
        "reg_cont": np.asarray(mutual_info_regression(X, yc, random_state=0), dtype=np.float64),
        "reg_5val": np.asarray(mutual_info_regression(X, y5, random_state=0), dtype=np.float64),
        "clf_2cls": np.asarray(mutual_info_classif(X, y2, random_state=0), dtype=np.float64),
    }
    np.savez(out, **res)


def cmp(a_path, b_path, label):
    a, b = np.load(a_path), np.load(b_path)
    ok = True
    for key in sorted(a.files):
        x, y = a[key], b[key]
        k = x.size // 2
        sa = set(np.argsort(-x, kind="stable")[:k].tolist())
        sb = set(np.argsort(-y, kind="stable")[:k].tolist())
        den = np.maximum(np.abs(x), 1e-12)
        rel = float((np.abs(x - y) / den).max())
        # relative to the largest score as well (tiny MI values near 0)
        rel_scale = float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-12))
        same = sa == sb
        ok = ok and same and rel_scale < 1e-4
        print("MIV-Q %s %s exact=%s set_identical=%s sym_diff=%d max_elem_rel=%.3e max_rel_to_scale=%.3e" % (
            label, key, bool(np.array_equal(x, y)), same, len(sa ^ sb), rel, rel_scale))
    print("MIV-Q-SUMMARY %s %s" % (label, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        cmp(sys.argv[2], sys.argv[3], sys.argv[4])
