# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cluster lane's GPU speed board (phase C, lane `cluster`).

Times the fit of every cluster-family estimator at a realistic shape on the
two R2 datasets (taxi and HIGGS, eight standardized features each), under the
numeric mode of the environment (MOJOLEARN_NUMERIC_MODE). Each case runs once
to load, then REPS timed runs; the minimum is reported. Every case prints a
digest of its fitted labels (and centers / means where the estimator has
them), so a before and an after on IDENTICAL show the same bits by eye (the
lane check proves it by column), plus a quality number for the FAST check.

The row counts follow tools/bench_board_algos.py: the linear-cost fits at
1M rows, the quadratic ones at the board's `quad` (10,000) and `tiny` (5,000)
shapes, BayesianGaussianMixture at its `mid` (100,000).

    python bench/x_cluster_speed.py [--dataset taxi,higgs] [--reps 3] [--only name,...] [--scale 1.0] [--column gpu|cpu]

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XCSPEED <dataset> <case> <rows> <seconds> <digest> <quality>`.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _root():
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def _digest(*vs):
    h = hashlib.sha256()
    for v in vs:
        if v is None:
            h.update(b"-")
            continue
        a = np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v)
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load(dataset, n):
    if dataset == "taxi":
        x = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))["x"][:n, :8]
    else:
        x = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))["x"][:n, 1:9]
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def _np(a):
    return None if a is None else np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a)


def _inertia(x, lab):
    lab = np.asarray(lab).astype(np.int64)
    tot = 0.0
    for c in np.unique(lab):
        if c < 0:
            continue
        p = x[lab == c].astype(np.float64)
        tot += float(((p - p.mean(0)) ** 2).sum())
    return tot


def _silhouette(x, lab, seed=0, m=4000):
    """The mean silhouette over a fixed sample of up to `m` rows (noise, -1,
    left out), float64 NumPy; NaN with fewer than two clusters."""
    lab = np.asarray(lab).astype(np.int64)
    keep = np.flatnonzero(lab >= 0)
    if len(np.unique(lab[keep])) < 2:
        return float("nan")
    rs = np.random.RandomState(seed)
    idx = keep if len(keep) <= m else np.sort(rs.choice(keep, m, replace=False))
    p = x[idx].astype(np.float64)
    lb = lab[idx]
    sq = (p * p).sum(1)
    dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * p @ p.T, 0))
    labs = np.unique(lb)
    mean_to = np.stack([dm[:, lb == c].sum(1) for c in labs], 1)
    cnt = np.array([(lb == c).sum() for c in labs], dtype=np.float64)
    own = np.searchsorted(labs, lb)
    a = mean_to[np.arange(len(lb)), own] / np.maximum(cnt[own] - 1, 1)
    other = mean_to / cnt[None, :]
    other[np.arange(len(lb)), own] = np.inf
    b = other.min(1)
    s = np.where(cnt[own] > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(s.mean())


def cases(ml, seed):
    """name -> (rows, build, quality). `build()` makes the estimator; the
    board calls fit(X) and reads labels_. quality(x, est) -> float."""
    inert = lambda x, e: _inertia(x, _np(e.labels_))
    sil = lambda x, e: _silhouette(x, _np(e.labels_))
    return {
        "kmeans": (1_000_000, lambda: ml.KMeans(n_clusters=8, random_state=seed), inert),
        "minibatch-kmeans": (1_000_000, lambda: ml.MiniBatchKMeans(
            n_clusters=8, batch_size=4096, max_iter=100, n_init=1, random_state=seed), inert),
        "bisecting-kmeans": (1_000_000, lambda: ml.BisectingKMeans(n_clusters=8, random_state=seed), inert),
        "gmm": (1_000_000, lambda: ml.GaussianMixture(n_components=8, random_state=seed),
                lambda x, e: float(e.score(x))),
        "bayesian-gmm": (100_000, lambda: ml.BayesianGaussianMixture(
            n_components=8, max_iter=100, random_state=seed), lambda x, e: float(e.score(x))),
        "dbscan": (100_000, lambda: ml.DBSCAN(eps=0.5, min_samples=10), sil),
        "hdbscan": (40_000, lambda: ml.HDBSCAN(min_cluster_size=50), sil),
        "agglomerative": (10_000, lambda: ml.AgglomerativeClustering(n_clusters=8), sil),
        "agglomerative-ward": (10_000, lambda: ml.AgglomerativeClustering(n_clusters=8, linkage="ward"), sil),
        "spectral": (10_000, lambda: ml.SpectralClustering(n_clusters=8, random_state=seed), sil),
        "meanshift": (10_000, lambda: ml.MeanShift(bin_seeding=True), sil),
        "optics": (10_000, lambda: ml.OPTICS(min_samples=10, xi=0.05), sil),
        "affinity-prop": (5_000, lambda: ml.AffinityPropagation(random_state=seed), sil),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every row count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--column", choices=("gpu", "cpu"), default="gpu",
                    help="cpu: every binding call goes to its host binding (the CPU column), "
                         "for a GPU-vs-CPU digest at the timed shape")
    a = ap.parse_args()
    import mojolearn as ml
    if a.column == "cpu":
        from mojolearn import _backend
        gpu_binding = _backend.binding

        def _host_binding(name, mode=None):
            # the estimator families only; the base binding's helpers
            # (all_finite_f32, ...) stay where they are
            base = _backend._HOST_MODULES.get(name)
            if name == "_mojolearn" or base is None:
                return gpu_binding(name, mode)
            return _backend.load_host_module(base)

        _backend.binding = _host_binding
    table = cases(ml, a.seed)
    names = [s for s in a.only.split(",") if s] or list(table)
    for ds in a.dataset.split(","):
        full = load(ds, int(max(r for r, _, _ in table.values()) * a.scale))
        for name in names:
            rows, build, qual = table[name]
            x = np.ascontiguousarray(full[: int(rows * a.scale)])
            try:
                build().fit(x)
                best = float("inf")
                est = None
                for _ in range(a.reps):
                    est = build()
                    t0 = time.perf_counter()
                    est.fit(x)
                    best = min(best, time.perf_counter() - t0)
                centers = None
                for attr in ("cluster_centers_", "means_"):
                    if getattr(est, attr, None) is not None:
                        centers = getattr(est, attr)
                        break
                dg = _digest(_np(est.labels_) if hasattr(est, "labels_") else est.predict(x), centers)
                q = float("nan") if a.no_quality else qual(x, est)
                tag = "" if a.column == "gpu" else " cpu"
                print(f"XCSPEED{tag} {ds} {name} {len(x)} {best:.4f} {dg} {q:.6g}", flush=True)
            except Exception as e:  # one broken case never hides the others
                print(f"XCSPEED {ds} {name} {len(x)} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
