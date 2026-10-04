# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""opv_quality.py: quality-only check for the lane/apple-fast-opv OPTICS
defines (FRONTIER_DEVICE, LIVEBUF). Times nothing.

  python tools/opv_quality.py dump <out.npz>
  python tools/opv_quality.py cmp <off.npz> <on.npz> <label>

Fixtures (elementwise, seed 0): 6k x 20 with 12 blobs plus 10% noise
(istella-like width) and 6k x 6 with 60 tight blobs (taxi-like, many
clusters); OPTICS(min_samples=10, xi=0.05) as the board. Gate: labels_
identical, ordering_ identical, reachability_ / core_distances_ max rel
diff < 1e-5 (finite cells, the same inf pattern)."""
import sys

import numpy as np


def _blobs(n, d, k, spread, seed):
    r = np.random.default_rng(seed)
    c = r.uniform(-10, 10, (k, d))
    lab = r.integers(0, k, n)
    X = c[lab] + spread * r.standard_normal((n, d))
    noise = r.random(n) < 0.1
    X[noise] = r.uniform(-12, 12, (int(noise.sum()), d))
    return X.astype(np.float32)


def dump(out):
    from mojolearn._expansion_cluster import OPTICS
    res = {}
    for name, X in (("wide", _blobs(6000, 20, 12, 1.0, 0)), ("many", _blobs(6000, 6, 60, 0.3, 1))):
        m = OPTICS(min_samples=10, xi=0.05).fit(X)
        res[name + "_labels"] = np.asarray(m.labels_, dtype=np.int64)
        res[name + "_ordering"] = np.asarray(m.ordering_, dtype=np.int64)
        res[name + "_reach"] = np.asarray(m.reachability_, dtype=np.float64)
        res[name + "_core"] = np.asarray(m.core_distances_, dtype=np.float64)
    np.savez(out, **res)


def cmp(a_path, b_path, label):
    a, b = np.load(a_path), np.load(b_path)
    ok = True
    for k in sorted(a.files):
        x, y = a[k], b[k]
        if k.endswith("_labels") or k.endswith("_ordering"):
            same = bool(np.array_equal(x, y))
            ok = ok and same
            extra = " n_clusters=%d" % len(set(x.tolist()) - {-1}) if k.endswith("_labels") else ""
            print("OPV-Q %s %s identical=%s mismatches=%d%s" % (label, k, same, int((x != y).sum()), extra))
        else:
            fx, fy = np.isfinite(x), np.isfinite(y)
            pat = bool(np.array_equal(fx, fy))
            f = fx & fy
            rel = float((np.abs(x[f] - y[f]) / np.maximum(np.abs(x[f]), 1e-12)).max()) if f.any() else 0.0
            ok = ok and pat and rel < 1e-5
            print("OPV-Q %s %s exact=%s inf_pattern_same=%s max_rel=%.3e" % (
                label, k, bool(np.array_equal(x, y)), pat, rel))
    print("OPV-Q-SUMMARY %s %s" % (label, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        cmp(sys.argv[2], sys.argv[3], sys.argv[4])
