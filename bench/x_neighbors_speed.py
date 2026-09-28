# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbors lane's GPU speed board (GPU-speed lane, family `neighbors`).

Times every algorithm of the family (neighbors/, svm/, kernel_methods/,
gaussian_process/, density's KernelDensity, x_neighbors/) at a realistic
shape on the two R2 datasets, under the numeric mode of the environment
(MOJOLEARN_NUMERIC_MODE). The row-linear algorithms (k-NN, radius, KDE,
NearestCentroid, the samplers, RBFSampler) run at --rows (default 1M);
the quadratic and cubic ones (SVC/SVR, OneClassSVM, KernelRidge, KernelPCA,
GP, label propagation, LOF) at the per-case sizes below, which are what a
user fits them at. Each case runs once to load, then REPS timed runs; the
minimum is reported, with a digest of the result, so a before and an after
on IDENTICAL show the same bits by eye (the lane check proves it by column).

    python bench/x_neighbors_speed.py [--rows 1000000] [--reps 3] [--only a,b] [--data taxi|higgs]

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XNSPEED <case> <seconds> <digest>`.
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


def _digest(v):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
            return
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                walk(x[k])
            return
        a = np.asarray(x.toarray() if hasattr(x, "toarray") else x)
        if a.dtype == object:
            h.update(repr(a.tolist()).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    walk(v)
    return h.hexdigest()[:16]


def _load(n, which):
    if which == "taxi":
        z = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
        x = np.asarray(z["x"][:n], dtype=np.float32)
        y = np.asarray(z["fare"][:n], dtype=np.float32)
        lab = np.asarray(z["card"][:n]).astype(np.int64)
    else:
        z = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
        x = np.asarray(z["x"][:n], dtype=np.float32)
        y = x[:, 0].copy()
        lab = np.asarray(z["y"][:n]).astype(np.int64)
    # standardized features: every kernel / distance method is fitted on
    # scaled inputs in practice, and raw taxi columns span 1e5
    mu = x.mean(axis=0, dtype=np.float64)
    sd = x.std(axis=0, dtype=np.float64) + 1e-12
    x = ((x - mu) / sd).astype(np.float32)
    x = np.ascontiguousarray(x[:, :16])
    y = ((y - y.mean()) / (y.std() + 1e-12)).astype(np.float32)
    return x, y, lab


def cases(x, y, lab, rows):
    import mojolearn as ml
    from mojolearn import _expansion_neighbors as xn
    n = min(rows, len(x))
    X = x[:n]
    Q = x[-10_000:]
    lab3 = (lab % 3).astype(np.int64)

    def sub(m):
        m = min(m, len(x))
        return x[:m], y[:m], lab[:m], lab3[:m]

    X20, y20, l20, l20_3 = sub(20_000)
    X10, y10, l10, _ = sub(10_000)
    X5, y5, l5, _ = sub(5_000)
    Xpos = np.abs(X) + np.float32(0.01)
    Xmiss = X[:200_000].copy()
    Xmiss[::7, 3] = np.nan
    lp_y = l20.copy()
    lp_y[1000:] = -1

    def fit_predict(est, Xf, yf, Xp):
        est.fit(Xf, yf)
        return est.predict(Xp)

    return [
        ("knn_kneighbors", lambda: ml.NearestNeighbors(n_neighbors=10).fit(X).kneighbors(Q)),
        ("knn_classifier", lambda: fit_predict(ml.KNeighborsClassifier(n_neighbors=10), X, lab[:n], Q)),
        ("knn_regressor", lambda: fit_predict(ml.KNeighborsRegressor(n_neighbors=10), X, y[:n], Q)),
        ("radius_neighbors", lambda: ml.RadiusNeighbors(radius=0.5).fit(X).radius_neighbors(Q[:2000])),
        ("kde_score", lambda: ml.KernelDensity(bandwidth=0.5).fit(X).score_samples(Q)),
        ("nearest_centroid", lambda: fit_predict(xn.NearestCentroid(), X, lab[:n], X)),
        ("rbf_sampler", lambda: ml.RBFSampler(n_components=256, random_state=0).fit(X).transform(X)),
        ("poly_sketch", lambda: xn.PolynomialCountSketch(n_components=256, random_state=0).fit(X).transform(X)),
        ("additive_chi2", lambda: xn.AdditiveChi2Sampler().fit(Xpos).transform(Xpos)),
        ("skewed_chi2", lambda: xn.SkewedChi2Sampler(n_components=256, random_state=0).fit(Xpos).transform(Xpos)),
        ("nystroem", lambda: ml.Nystroem(n_components=512, random_state=0).fit(X).transform(X)),
        ("knn_imputer", lambda: xn.KNNImputer(n_neighbors=5).fit_transform(Xmiss)),
        ("lof", lambda: xn.LocalOutlierFactor(n_neighbors=20).fit_predict(X[:200_000])),
        ("svc_rbf", lambda: fit_predict(ml.SVC(kernel="rbf"), X20, l20, X20)),
        ("svc_multiclass", lambda: fit_predict(ml.SVC(kernel="rbf"), X10, l20_3[:len(X10)], X10)),
        ("svr_rbf", lambda: fit_predict(ml.SVR(kernel="rbf"), X20, y20, X20)),
        ("ocsvm", lambda: xn.OneClassSVM(nu=0.1).fit(X20).predict(X20)),
        ("kernel_ridge", lambda: fit_predict(ml.KernelRidge(kernel="rbf", alpha=1.0), X10, y10, X10)),
        ("kernel_pca", lambda: xn.KernelPCA(n_components=8, kernel="rbf").fit_transform(X5)),
        ("gpr", lambda: fit_predict(ml.GaussianProcessRegressor(), X5[:2000], y5[:2000], X5)),
        ("label_spreading", lambda: xn.LabelSpreading(kernel="knn", n_neighbors=10).fit(X20, lp_y).transduction_),
        ("label_propagation", lambda: xn.LabelPropagation(kernel="knn", n_neighbors=10).fit(X20, lp_y).transduction_),
        ("svgp", lambda: fit_predict(xn.SVGP(n_inducing=128), X20, y20, X20)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--data", default="taxi", choices=("taxi", "higgs"))
    a = ap.parse_args()
    t0 = time.time()
    x, y, lab = _load(a.rows + 10_000, a.data)
    print("XNSPEED-DATA data=%s rows=%d load_s=%.2f mode=%s" % (
        a.data, a.rows, time.time() - t0, os.environ.get("MOJOLEARN_NUMERIC_MODE", "default")), flush=True)
    only = set(s for s in a.only.split(",") if s)
    total = 0.0
    for name, fn in cases(x, y, lab, a.rows):
        if only and name not in only:
            continue
        try:
            v = fn()
            best = float("inf")
            for _ in range(a.reps):
                t = time.perf_counter()
                v = fn()
                best = min(best, time.perf_counter() - t)
            total += best
            print("XNSPEED %-20s %9.4f %s" % (name, best, _digest(v)), flush=True)
        except Exception as e:  # a case that fails is reported, never hidden
            print("XNSPEED %-20s   FAILED %s: %s" % (name, type(e).__name__, str(e)[:200]), flush=True)
    print("XNSPEED-TOTAL %.4f" % total, flush=True)


if __name__ == "__main__":
    main()
