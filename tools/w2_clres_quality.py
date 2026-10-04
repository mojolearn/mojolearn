#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quality pair for lane apple-fast-w2-clres (MiniBatchKMeans, board shape).

  dump    <lane> <dataset> <out.npz>          one arm (the installed .so)
  compare <A.npz> <B.npz> <mode> <dataset>    arm A (main) vs arm B

Run by tools/w2_clres_quality.sh, which installs each arm's prebuilt .so.
Data and parameters are the board's (tools/bench_board_algos.py LANES,
_load_block, lane_arrays), so the fit is the timed one.

Modes and tolerances, fixed before any result:
  exact  (SUMCMP; default now, old arm = MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF):
         the compacted sum adds the
         same rows in the same order, so centers, counts, labels and inertia
         must be BIT-IDENTICAL to main.
  labrg  (MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG): only the last labelling
         pass changes (its 220-term float32 distance sums are reordered), so
         centers and counts must be bit-identical; inertia relative change
         <= 1e-5 (the worst-case relative error of reordering a d <= 220
         float32 sum of nonnegative terms is about d * 2^-24 = 1.3e-5 per
         row, and per-row errors do not add coherently); at most 1e-4 of the
         labels may differ, and every differing row must be a genuine near
         tie: its float64 distances to the two centers differ by <= 1e-5
         relative.
Prints W2CLRES-QUALITY ... PASS|FAIL and exits 1 on FAIL.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _bench():
    spec = importlib.util.spec_from_file_location("w2clres_bba", HERE / "bench_board_algos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data_dir():
    for b in ("board-0834", "board-0833"):
        d = Path.home() / b / "cache/algos-data/rows-full"
        if d.is_dir():
            return str(d)
    raise SystemExit("no board data directory (~/board-083x/cache/algos-data/rows-full)")


def _X(bba, lane, dataset):
    B, _ = bba._load_block(lane, dataset, _data_dir())
    D = bba.lane_arrays(lane, B)
    return np.ascontiguousarray(D["X"], dtype=np.float32)


def dump(lane, dataset, out):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    assert lane == "minibatch-kmeans", lane
    bba = _bench()
    import mojolearn as ml
    from mojolearn import _backend
    so = Path(_backend.binding("_mojolearn_x_cluster", "fast").__file__)
    X = _X(bba, lane, dataset)
    p = dict(bba.LANES[lane]["params"])
    m = ml.MiniBatchKMeans(**p).fit(X)
    np.savez(out, labels=np.asarray(m.labels_, dtype=np.int32),
             centers=np.asarray(m.cluster_centers_, dtype=np.float32),
             counts=np.asarray(m.counts_, dtype=np.float32),
             inertia=np.array([float(m.inertia_)]),
             so_sha=np.array([hashlib.sha256(so.read_bytes()).hexdigest()]))
    print("W2CLRES-DUMP lane=%s ds=%s so=%s inertia=%.9g n=%d" % (
        lane, dataset, hashlib.sha256(so.read_bytes()).hexdigest()[:16], float(m.inertia_), X.shape[0]))


def compare(fa, fb, mode, dataset):
    A, B = np.load(fa), np.load(fb)
    r = dict(mode=mode, dataset=dataset, so_A=str(A["so_sha"][0])[:16], so_B=str(B["so_sha"][0])[:16])
    ok = r["so_A"] != r["so_B"]          # the two arms really are different builds
    r["arms_differ"] = ok
    same = lambda k: A[k].shape == B[k].shape and A[k].tobytes() == B[k].tobytes()
    r["centers_identical"] = same("centers")
    r["counts_identical"] = same("counts")
    ia, ib = float(A["inertia"][0]), float(B["inertia"][0])
    r["inertia_A"], r["inertia_B"] = ia, ib
    r["inertia_rel"] = abs(ib - ia) / max(abs(ia), 1e-30)
    la, lb = A["labels"], B["labels"]
    diff = np.flatnonzero(la != lb) if la.shape == lb.shape else None
    r["labels_differ"] = None if diff is None else int(diff.size)
    if mode == "exact":
        ok = ok and r["centers_identical"] and r["counts_identical"] and same("labels") and same("inertia")
    elif mode == "labrg":
        ok = ok and r["centers_identical"] and r["counts_identical"] and diff is not None
        ok = ok and r["inertia_rel"] <= 1e-5 and diff.size <= 1e-4 * la.size
        if ok and diff.size:
            X = _X(_bench(), "minibatch-kmeans", dataset)[diff].astype(np.float64)
            C = A["centers"].astype(np.float64)
            da = ((X - C[la[diff]]) ** 2).sum(1)
            db = ((X - C[lb[diff]]) ** 2).sum(1)
            gap = np.abs(da - db) / np.maximum(np.maximum(da, db), 1e-30)
            r["max_tie_gap"] = float(gap.max())
            ok = ok and r["max_tie_gap"] <= 1e-5
    else:
        raise SystemExit("mode must be exact or labrg")
    r["verdict"] = "PASS" if ok else "FAIL"
    print("W2CLRES-QUALITY " + json.dumps(r, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["dump"] and len(a) == 4:
        dump(a[1], a[2], a[3])
    elif a[:1] == ["compare"] and len(a) == 5:
        sys.exit(compare(a[1], a[2], a[3], a[4]))
    else:
        raise SystemExit(__doc__)
