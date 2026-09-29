#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN on the bench board's taxi block past the dense bound (lane
hdbscan-sparse, DEVIATION 1620): our fit and scikit-learn's on the SAME
rows at the SAME settings, one run each, and how far the labels agree.

    python tools/hdbscan_sparse_taxi.py prep    --data DIR          (the board's dbscan-taxi block)
    python tools/hdbscan_sparse_taxi.py ours    --data DIR --out O  (mojolearn.HDBSCAN, fit timed)
    python tools/hdbscan_sparse_taxi.py sklearn --data DIR --out O  (sklearn.cluster.HDBSCAN, n_jobs=-1, fit timed)
    python tools/hdbscan_sparse_taxi.py compare --out O             (ARI, noise agreement, exact partition)

The rows and settings are tools/classical_two_datasets.py's hdbscan lane:
the dbscan block's first 100,000 rows (taxi's numeric columns,
standardized), min_samples=10, min_cluster_size=100, eom, alpha 1.0. The
times are one fit each on the box this runs on; no claim is made from
them beyond that measurement.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KW = dict(min_samples=10, min_cluster_size=100, metric="euclidean", cluster_selection_method="eom",
          cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False)


def _rows(args):
    z = np.load(os.path.join(args.data, "dbscan-taxi.npz"))
    return np.ascontiguousarray(z["X"][:args.rows], dtype=np.float32)


def prep(args):
    os.makedirs(args.data, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "classical_two_datasets.py"), "prep", "--data", args.data,
                    "--lanes", "hdbscan", "--datasets", "taxi"], check=True)
    X = _rows(args)
    print("prep: dbscan-taxi rows", X.shape, "sha of first rows", hash(X.tobytes()) & 0xFFFFFFFF)


def ours(args):
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import mojolearn as ml
    X = _rows(args)
    est = ml.HDBSCAN(max_cluster_size=0, **KW)
    t0 = time.perf_counter()
    est.fit(X)
    wall = time.perf_counter() - t0
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "ours_labels.npy"), np.asarray(est.labels_, dtype=np.int32))
    np.save(os.path.join(args.out, "ours_probs.npy"), np.asarray(est.probabilities_, dtype=np.float32))
    rec = dict(arm="ours", mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", "(default)"), rows=int(X.shape[0]),
               cols=int(X.shape[1]), fit_s=wall, n_clusters=int(est.n_clusters_), n_noise=int(est.n_outliers_),
               n_boruvka_rounds=int(est.n_boruvka_rounds_),
               labels_fnv=_fnv(np.asarray(est.labels_, dtype=np.int32)),
               probs_fnv=_fnv(np.asarray(est.probabilities_, dtype=np.float32)))
    print(json.dumps(rec))
    with open(os.path.join(args.out, "ours.json"), "w") as f:
        json.dump(rec, f, indent=1)


def sk(args):
    from sklearn.cluster import HDBSCAN
    X = _rows(args)
    # scikit-learn's min_samples counts the point itself; ours (cuML's
    # runner.h) does not: +1 selects the SAME k-th neighbour
    # (tools/classical_two_datasets.py SKLEARN_HDBSCAN_KW).
    est = HDBSCAN(max_cluster_size=None, n_jobs=-1, **dict(KW, min_samples=KW["min_samples"] + 1))
    t0 = time.perf_counter()
    est.fit(X)
    wall = time.perf_counter() - t0
    lab = np.asarray(est.labels_, dtype=np.int32)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "sk_labels.npy"), lab)
    np.save(os.path.join(args.out, "sk_probs.npy"), np.asarray(est.probabilities_, dtype=np.float32))
    import sklearn
    rec = dict(arm="sklearn", version=sklearn.__version__, rows=int(X.shape[0]), fit_s=wall,
               n_clusters=int(lab.max() + 1), n_noise=int((lab < 0).sum()))
    print(json.dumps(rec))
    with open(os.path.join(args.out, "sklearn.json"), "w") as f:
        json.dump(rec, f, indent=1)


def compare(args):
    from sklearn.metrics import adjusted_rand_score
    a = np.load(os.path.join(args.out, "ours_labels.npy"))
    b = np.load(os.path.join(args.out, "sk_labels.npy"))
    same_noise = float(np.mean((a < 0) == (b < 0)))
    # exact partition equality: the pairing of label values is one to one
    pairs = set(zip(a.tolist(), b.tolist()))
    exact = len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))
    # per-point agreement under the majority mapping of our clusters to theirs
    agree = 0
    for c in np.unique(a):
        m = a == c
        vals, cnt = np.unique(b[m], return_counts=True)
        agree += int(cnt.max())
    rec = dict(ari=float(adjusted_rand_score(b, a)), noise_set_agreement=same_noise,
               majority_label_agreement=agree / a.size, exact_same_partition=bool(exact),
               ours_clusters=int(a.max() + 1), sk_clusters=int(b.max() + 1),
               ours_noise=int((a < 0).sum()), sk_noise=int((b < 0).sum()))
    print(json.dumps(rec))
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump(rec, f, indent=1)


def _fnv(arr):
    h = 0xCBF29CE484222325
    for byte in np.ascontiguousarray(arr).tobytes():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("prep", "ours", "sklearn", "compare"))
    p.add_argument("--data", default=os.path.expanduser("~/hdbscan-sparse-data"))
    p.add_argument("--out", default=os.path.expanduser("~/hdbscan-sparse-out"))
    p.add_argument("--rows", type=int, default=100_000)
    args = p.parse_args()
    {"prep": prep, "ours": ours, "sklearn": sk, "compare": compare}[args.what](args)


if __name__ == "__main__":
    main()
