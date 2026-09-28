# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbors lane's second Apple speed board (lane `neighbors-apple2`).

The estimators round one did not time: the x_neighbors expansion
(LocalOutlierFactor, NearestCentroid, OneClassSVM, KernelPCA, the kernel
approximations, label propagation / spreading, KNNImputer, PageRank,
connected components, Louvain, SVGP), plus the large-k k-NN shapes
(SpectralEmbedding with the knn affinity at 20k rows, whose default
n_neighbors = n // 10 is 2,000, and NearestNeighbors at k = 2,000).

Same contract as `bench/x_neighbors_apple_speed.py`: taxi and HIGGS from R2,
the first eight columns standardized, one load run, then the minimum of REPS
timed runs of fit and of predict, a digest of the outputs, under the numeric
mode of the environment (MOJOLEARN_NUMERIC_MODE).

    python bench/x_neighbors_apple2_speed.py [--dataset taxi,higgs] [--reps 2] [--only name,...] [--scale 1.0]

Lines: `XN2SPEED <dataset> <case> <fit_rows> <query_rows> <fit_s> <predict_s> <digest> <quality>`.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from x_neighbors_apple_speed import _acc, _digest, _np, _r2, load  # noqa: E402

# (name, fit rows, query rows) at scale 1
CASES = [
    ("lof", 20_000, 5_000),
    ("ncentroid", 200_000, 50_000),
    ("ocsvm", 3_000, 10_000),
    ("kpca", 500, 10_000),
    ("pcs", 1_000, 200_000),
    ("achi2", 1_000, 1_000_000),
    ("skewed", 1_000, 1_000_000),
    ("labelprop", 5_000, 2_000),
    ("labelspread", 5_000, 2_000),
    ("knnimpute", 50_000, 5_000),
    ("pagerank", 5_000, 0),
    ("cc", 5_000, 0),
    ("louvain", 1_000, 0),
    ("svgp", 100_000, 100_000),
    ("spectral-knn", 20_000, 0),
    ("nn-k2000", 20_000, 2_000),
]
GAMMA = 0.125


def _graph(x, k=10):
    """A symmetric 0/1 kNN adjacency (float64 NumPy, fixed rows)."""
    a = x.astype(np.float64)
    n = len(a)
    d = (a * a).sum(1)[:, None] + (a * a).sum(1)[None, :] - 2.0 * a @ a.T
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1, kind="stable")[:, :k]
    g = np.zeros((n, n), np.float32)
    g[np.repeat(np.arange(n), k), nb.ravel()] = 1.0
    return np.ascontiguousarray(np.maximum(g, g.T))


def run_case(ml, name, x, yc, yr, xq, ycq, yrq):
    st = {}
    if name == "lof":
        def fit():
            st["m"] = ml.LocalOutlierFactor(n_neighbors=20, novelty=True).fit(x)
        return fit, (lambda: st["m"].score_samples(xq)), (lambda o: float(np.mean(_np(o))))
    if name == "ncentroid":
        def fit():
            st["m"] = ml.NearestCentroid().fit(x, yc)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _acc(ycq, _np(o)))
    if name == "ocsvm":
        def fit():
            st["m"] = ml.OneClassSVM(kernel="rbf", gamma=GAMMA, nu=0.1).fit(x)
        return fit, (lambda: st["m"].decision_function(xq)), (lambda o: float(np.mean(_np(o) > 0)))
    if name == "kpca":
        def fit():
            st["m"] = ml.KernelPCA(n_components=10, kernel="rbf", gamma=GAMMA).fit(x)
        return fit, (lambda: st["m"].transform(xq)), (lambda o: float(np.abs(_np(o)).mean()))
    if name == "pcs":
        def fit():
            st["m"] = ml.PolynomialCountSketch(gamma=GAMMA, degree=2, n_components=500, random_state=0).fit(x)
        return fit, (lambda: st["m"].transform(xq)), (lambda o: float(np.abs(_np(o)).mean()))
    if name == "achi2":
        xa, qa = np.abs(x), np.ascontiguousarray(np.abs(xq))

        def fit():
            st["m"] = ml.AdditiveChi2Sampler(sample_steps=2).fit(xa)
        return fit, (lambda: st["m"].transform(qa)), (lambda o: float(np.abs(_np(o)).mean()))
    if name == "skewed":
        xa, qa = np.abs(x), np.ascontiguousarray(np.abs(xq))

        def fit():
            st["m"] = ml.SkewedChi2Sampler(skewedness=1.0, n_components=500, random_state=0).fit(xa)
        return fit, (lambda: st["m"].transform(qa)), (lambda o: float(np.abs(_np(o)).mean()))
    if name in ("labelprop", "labelspread"):
        y = yc.copy()
        y[np.arange(len(y)) % 10 != 0] = -1
        cls = ml.LabelPropagation if name == "labelprop" else ml.LabelSpreading

        def fit():
            st["m"] = cls(kernel="knn", n_neighbors=7).fit(x, y)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _acc(ycq, _np(o)))
    if name == "knnimpute":
        rng = np.random.default_rng(0)
        xm = x.copy()
        xm[rng.random(xm.shape) < 0.1] = np.nan
        qm = xq.copy()
        qm[rng.random(qm.shape) < 0.1] = np.nan

        def fit():
            st["m"] = ml.KNNImputer(n_neighbors=5).fit(xm)
        return fit, (lambda: st["m"].transform(qm)), (lambda o: float(np.nanmean(np.abs(_np(o) - xq))))
    if name in ("pagerank", "cc", "louvain"):
        g = _graph(x)
        if name == "pagerank":
            def fit():
                st["m"] = ml.PageRank().fit(g)
            return fit, (lambda: st["m"].pagerank_), \
                (lambda o: float(np.max(_np(o))))
        if name == "cc":
            def fit():
                st["o"] = ml.connected_components(g)
            return fit, (lambda: st["o"]), (lambda o: float(len(o)))
        def fit():
            st["m"] = ml.Louvain().fit(g)
        return fit, (lambda: st["m"].labels_), (lambda o: float(len(set(_np(o).tolist()))))
    if name == "svgp":
        def fit():
            st["m"] = ml.SVGP(n_inducing=64, lengthscale=2.0, noise_variance=0.5).fit(x, yr)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _r2(yrq, _np(o)))
    if name == "spectral-knn":
        def fit():
            st["m"] = ml.SpectralEmbedding(n_components=2, affinity="nearest_neighbors", random_state=0).fit(x)
        return fit, (lambda: st["m"].embedding_), (lambda o: float(np.abs(_np(o)).mean()))
    if name == "nn-k2000":
        def fit():
            st["m"] = ml.NearestNeighbors(n_neighbors=2000).fit(x)

        def pred():
            return st["m"].kneighbors(xq)

        def qual(o):
            # the first 1,024 slots must be the k = 1,024 answer, slot for slot
            d2, i2 = ml.NearestNeighbors(n_neighbors=1024).fit(x).kneighbors(xq)
            i1 = _np(o[1])[:, :1024]
            return float((i1 == _np(i2)).mean())
        return fit, pred, qual
    raise SystemExit(f"unknown case {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-quality", action="store_true")
    a = ap.parse_args()
    import mojolearn as ml
    only = [s for s in a.only.split(",") if s]
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical")
    print(f"# x_neighbors_apple2_speed mode={mode} reps={a.reps} scale={a.scale}", flush=True)
    for ds in a.dataset.split(","):
        for name, nf, nq in CASES:
            if only and name not in only:
                continue
            nf = max(64, int(nf * a.scale))
            nq = max(64, int(nq * a.scale)) if nq else 64
            x, yc, yr = load(ds, nf + nq)
            xf, xq, ycf, ycq, yrf, yrq = x[:nf], x[nf:], yc[:nf], yc[nf:], yr[:nf], yr[nf:]
            try:
                fit, pred, qual = run_case(ml, name, xf, ycf, yrf, xq, ycq, yrq)
                fit()
                out = pred()
                tf, tp = [], []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    fit()
                    t1 = time.perf_counter()
                    out = pred()
                    t2 = time.perf_counter()
                    tf.append(t1 - t0)
                    tp.append(t2 - t1)
                dg = _digest(out)
                q = float("nan") if a.no_quality else qual(out)
                print(f"XN2SPEED {ds} {name} {nf} {nq} {min(tf):.4f} {min(tp):.4f} {dg} {q:.6f}", flush=True)
            except Exception as e:  # a failing case must not hide the others
                msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
                print(f"XN2SPEED {ds} {name} {nf} {nq} FAIL {type(e).__name__}: {msg}", flush=True)


if __name__ == "__main__":
    main()
