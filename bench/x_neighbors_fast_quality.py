# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbors lane's paired FAST quality check (lane `neighbors-apple`).

For each dataset (taxi, HIGGS; the first eight columns standardized) and
each seed, a seeded row sample is split into fit and held-out rows; every
estimator of the family is fitted twice on the SAME rows, numeric_mode
'identical' (the reference arithmetic) and 'fast', and scored on the SAME
held-out rows. Lines:

    XNQUAL <dataset> <seed> <case> <identical score> <fast score> <fast - identical>

and a summary per (dataset, case): mean and worst paired difference. Scores:
accuracy (classifiers, k-NN), R^2 (regressors), recall@k against float64
NumPy (NearestNeighbors), mean log density (KDE), minus the error against
float64 NumPy (Nystroem's kernel approximation, KernelPCA's eigenvalues and
projection; lane neighbors-apple3).

    python bench/x_neighbors_fast_quality.py [--seeds 5] [--dataset taxi,higgs] [--only a,b]

Needs both tiers built (bindings/build_*.sh with and without
MOJOLEARN_NUMERIC_MODE=fast).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from x_neighbors_apple_speed import _acc, _kernel_err, _knn_recall, _np, _r2, _root  # noqa: E402

GAMMA = 0.125


_RAW = {}


def rows(dataset, n, seed):
    if dataset not in _RAW:
        if dataset == "taxi":
            z = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
            _RAW[dataset] = (z["x"][:2_000_000, :8], z["card"][:2_000_000].astype(np.int32), z["fare"][:2_000_000])
        else:
            z = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
            xx = z["x"][:2_000_000]
            _RAW[dataset] = (xx[:, :8], z["y"][:2_000_000].astype(np.int32), xx[:, 8])
    x, yc, yr = _RAW[dataset]
    idx = np.sort(np.random.default_rng(seed).choice(min(len(x), 2_000_000), n, replace=False))
    x = np.asarray(x[idx], dtype=np.float64)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    yr = np.asarray(yr[idx], dtype=np.float64)
    yr = (yr - yr.mean()) / (yr.std() or 1.0)
    return (np.ascontiguousarray(x, np.float32), np.ascontiguousarray(yc[idx], np.int32),
            np.ascontiguousarray(yr, np.float32))


def score(ml, case, mode, x, yc, yr, xq, ycq, yrq):
    if case == "svc":
        return _acc(ycq, _np(ml.SVC(C=1.0, kernel="rbf", gamma=GAMMA, numeric_mode=mode).fit(x, yc).predict(xq)))
    if case == "svr":
        return _r2(yrq, _np(ml.SVR(C=1.0, kernel="rbf", gamma=GAMMA, numeric_mode=mode).fit(x, yr).predict(xq)))
    if case == "krr":
        return _r2(yrq, _np(ml.KernelRidge(alpha=1.0, kernel="rbf", gamma=GAMMA, numeric_mode=mode).fit(x, yr).predict(xq)))
    if case == "gpr":
        k = ml.ConstantKernel(1.0) * ml.RBF(2.0) + ml.WhiteKernel(0.5)
        return _r2(yrq, _np(ml.GaussianProcessRegressor(kernel=k, numeric_mode=mode).fit(x, yr).predict(xq)))
    if case == "gpc":
        m = ml.GaussianProcessClassifier(kernel=ml.ConstantKernel(1.0) * ml.RBF(2.0), numeric_mode=mode).fit(x, yc)
        return _acc(ycq, np.argmax(_np(m.predict_proba(xq)), 1))
    if case == "knnc":
        return _acc(ycq, _np(ml.KNeighborsClassifier(n_neighbors=10, numeric_mode=mode).fit(x, yc).predict(xq)))
    if case == "knnr":
        return _r2(yrq, _np(ml.KNeighborsRegressor(n_neighbors=10, numeric_mode=mode).fit(x, yr).predict(xq)))
    if case == "nn":
        o = ml.NearestNeighbors(n_neighbors=10, numeric_mode=mode).fit(x).kneighbors(xq)
        return _knn_recall(x, xq, _np(o[1]), 10)
    if case == "kde":
        return float(np.mean(_np(ml.KernelDensity(bandwidth=0.5, numeric_mode=mode).fit(x).score_samples(xq))))
    if case == "nystroem":
        # minus the relative Frobenius error of the feature map's Gram
        # against the exact float64 RBF kernel on the held-out rows
        m = ml.Nystroem(kernel="rbf", gamma=GAMMA, n_components=300, random_state=0, numeric_mode=mode).fit(x)
        return -_kernel_err(xq, _np(m.transform(xq)), GAMMA)
    if case == "kpca":
        # minus the relative error of the top 16 eigenvalues and of the
        # projection of the held-out rows against float64 NumPy (the same
        # centered kernel, eigh, scikit-learn's transform)
        m = ml.KernelPCA(n_components=16, kernel="rbf", gamma=GAMMA, numeric_mode=mode).fit(x)
        return -_kpca_err(x, xq, _np(m.eigenvalues_), _np(m.transform(xq)), GAMMA)
    raise SystemExit(case)


def _rbf64(a, b, gamma):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.exp(-gamma * np.maximum(d, 0.0))


def _kpca_err(x, xq, vals, proj, gamma):
    """Eigenvalue error plus projection error, both relative, in float64.
    A projection column's sign is the solver's choice, so each column is
    compared up to its sign."""
    k = _rbf64(x, x, gamma)
    rows = k.mean(0)
    allm = rows.mean()
    kc = k - rows[None, :] - k.mean(1)[:, None] + allm
    w, v = np.linalg.eigh(kc)
    c = len(vals)
    w = w[::-1][:c]
    v = v[:, ::-1][:, :c]
    kq = _rbf64(xq, x, gamma)
    kqc = kq - rows[None, :] - kq.mean(1)[:, None] + allm
    ref = kqc @ (v / np.sqrt(np.maximum(w, 1e-300)))
    got = np.asarray(proj, np.float64)
    err = 0.0
    for j in range(c):
        e = min(np.linalg.norm(got[:, j] - ref[:, j]), np.linalg.norm(got[:, j] + ref[:, j]))
        err += e / max(np.linalg.norm(ref[:, j]), 1e-300)
    return float(np.linalg.norm(np.asarray(vals, np.float64) - w) / np.linalg.norm(w) + err / c)


SHAPES = {"svc": (5000, 2000), "svr": (5000, 2000), "krr": (4000, 2000), "gpr": (2000, 1000),
          "gpc": (1500, 1000), "knnc": (50000, 2000), "knnr": (50000, 2000), "nn": (50000, 1000),
          "kde": (50000, 1000), "nystroem": (4000, 1000), "kpca": (500, 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import mojolearn as ml
    only = set(filter(None, a.only.split(",")))
    for ds in a.dataset.split(","):
        for case, (nf, nq) in SHAPES.items():
            if only and case not in only:
                continue
            diffs = []
            for seed in range(a.seeds):
                x, yc, yr = rows(ds, nf + nq, seed)
                args = (x[:nf], yc[:nf], yr[:nf], x[nf:], yc[nf:], yr[nf:])
                try:
                    si = score(ml, case, "identical", *args)
                    sf = score(ml, case, "fast", *args)
                except Exception as e:  # report and go on
                    print(f"XNQUAL {ds} {seed} {case} FAIL {str(e).splitlines()[0][:160]}", flush=True)
                    continue
                diffs.append(sf - si)
                print(f"XNQUAL {ds} {seed} {case} {si:.6f} {sf:.6f} {sf - si:+.6f}", flush=True)
            if diffs:
                print(f"XNQUAL_SUMMARY {ds} {case} seeds={len(diffs)} mean_diff={np.mean(diffs):+.6f} "
                      f"worst_diff={min(diffs):+.6f}", flush=True)


if __name__ == "__main__":
    main()
