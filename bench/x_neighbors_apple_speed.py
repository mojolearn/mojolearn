# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbors lane's Apple speed board (lane `neighbors-apple`).

Times fit and predict of every estimator of the neighbors family (k-NN,
radius neighbors, KDE, SVC / SVR, KernelRidge, GP regressor / classifier,
Nystroem, RBFSampler) on the two R2 datasets (taxi and HIGGS, the first eight
columns standardized), under the numeric mode of the environment
(MOJOLEARN_NUMERIC_MODE). Each case runs once to load (compile caches, device
context), then REPS timed runs; the minimum of each phase is reported. Every
case prints a digest of its outputs, so a before and an after on IDENTICAL
show the same bits by eye (the lane check proves it by column), and a quality
number for the FAST check (accuracy, R^2, recall against float64 NumPy,
mean log density, kernel approximation error).

    python bench/x_neighbors_apple_speed.py [--dataset taxi,higgs] [--reps 3] [--only name,...] [--scale 1.0]

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XNSPEED <dataset> <case> <fit_rows> <query_rows> <fit_s> <predict_s> <digest> <quality>`.
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


def _np(a):
    if a is None:
        return None
    if isinstance(a, (list, tuple)):
        return [_np(v) for v in a]
    return np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a)


def _digest(*vs):
    h = hashlib.sha256()

    def put(v):
        if v is None:
            h.update(b"-")
        elif isinstance(v, (list, tuple)):
            for w in v:
                put(w)
        else:
            h.update(np.ascontiguousarray(_np(v)).tobytes())
    for v in vs:
        put(v)
    return h.hexdigest()[:16]


_CACHE = {}


def load(dataset, n):
    """(x float32 standardized [n, 8], class label int32, regression target float32)."""
    if dataset not in _CACHE:
        if dataset == "taxi":
            z = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
            x, yc, yr = z["x"][:, :8], z["card"].astype(np.int32), z["fare"]
        else:
            z = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
            x = z["x"][:, :8]
            yc = z["y"].astype(np.int32)
            yr = z["x"][:, 8]
        _CACHE[dataset] = (x, yc, yr)
    x, yc, yr = _CACHE[dataset]
    x = np.asarray(x[:n], dtype=np.float64)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    yr = np.asarray(yr[:n], dtype=np.float64)
    yr = (yr - yr.mean()) / (yr.std() or 1.0)
    return (np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(yc[:n], dtype=np.int32),
            np.ascontiguousarray(yr, dtype=np.float32))


def _r2(y, p):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64).reshape(y.shape)
    return 1.0 - float(((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300))


def _acc(y, p):
    return float((np.asarray(y).ravel() == np.asarray(p).ravel()).mean())


def _knn_recall(xi, xq, ind, k, m=300):
    """Mean recall@k of `ind` on the first `m` queries against float64 NumPy."""
    a = xi.astype(np.float64)
    an = (a * a).sum(1)
    hit = 0
    for i in range(min(m, len(xq))):
        q = xq[i].astype(np.float64)
        d = an - 2.0 * (a @ q)
        ref = set(np.argpartition(d, k)[:k].tolist())
        hit += len(ref & set(np.asarray(ind[i]).ravel().tolist()[:k]))
    return hit / (min(m, len(xq)) * k)


def _kernel_err(xs, feats, gamma):
    """Relative Frobenius error of feats feats^T against exact RBF on rows xs."""
    a = xs.astype(np.float64)
    d = (a * a).sum(1)[:, None] + (a * a).sum(1)[None, :] - 2.0 * a @ a.T
    k = np.exp(-gamma * np.maximum(d, 0.0))
    f = np.asarray(feats, np.float64)
    return float(np.linalg.norm(f @ f.T - k) / np.linalg.norm(k))


# (name, fit rows, query rows) at scale 1
CASES = [
    ("nn", 200_000, 10_000),
    ("knnc", 200_000, 10_000),
    ("knnr", 200_000, 10_000),
    ("radius", 100_000, 5_000),
    ("nn-ties", 100_000, 5_000),
    ("nn-k20", 200_000, 10_000),
    ("kde", 100_000, 2_000),
    ("svc", 10_000, 10_000),
    ("svr", 10_000, 10_000),
    ("krr", 10_000, 10_000),
    ("gpr", 3_000, 3_000),
    ("gpc", 3_000, 3_000),
    ("nystroem", 4_000, 100_000),
    ("rbf", 1_000_000, 1_000_000),
]
K = 10
GAMMA = 0.125   # 1 / n_features on the standardized data (gamma='scale' there)


def run_case(ml, name, x, yc, yr, xq, ycq, yrq):
    """(fit callable, predict callable returning outputs, quality fn(outputs))."""
    st = {}
    if name == "nn":
        def fit():
            st["m"] = ml.NearestNeighbors(n_neighbors=K).fit(x)

        def pred():
            return st["m"].kneighbors(xq)
        return fit, pred, lambda o: _knn_recall(x, xq, _np(o[1]), K)
    if name == "nn-ties":
        # coarse grid values: many exactly tied distances and duplicate rows
        xt = np.ascontiguousarray(np.round(x * 2.0) / 2.0, dtype=np.float32)
        qt = np.ascontiguousarray(np.round(xq * 2.0) / 2.0, dtype=np.float32)

        def fit():
            st["m"] = ml.NearestNeighbors(n_neighbors=K).fit(xt)

        def pred():
            return st["m"].kneighbors(qt)
        return fit, pred, lambda o: _knn_recall(xt, qt, _np(o[1]), K)
    if name == "nn-k20":
        def fit():
            st["m"] = ml.NearestNeighbors(n_neighbors=20).fit(x)

        def pred():
            return st["m"].kneighbors(xq)
        return fit, pred, lambda o: _knn_recall(x, xq, _np(o[1]), 20)
    if name == "knnc":
        def fit():
            st["m"] = ml.KNeighborsClassifier(n_neighbors=K).fit(x, yc)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _acc(ycq, _np(o)))
    if name == "knnr":
        def fit():
            st["m"] = ml.KNeighborsRegressor(n_neighbors=K).fit(x, yr)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _r2(yrq, _np(o)))
    if name == "radius":
        # the radius that holds ~K neighbours of a typical query (float64, fixed rows)
        a = x.astype(np.float64)
        kd = []
        for i in range(64):
            d = ((a - xq[i].astype(np.float64)) ** 2).sum(1)
            kd.append(np.partition(d, 3 * K)[3 * K])
        r = float(np.sqrt(np.median(kd)))

        def fit():
            st["m"] = ml.RadiusNeighbors(radius=r).fit(x)

        def pred():
            return st["m"].radius_neighbors(xq)
        return fit, pred, lambda o: float(np.mean([len(_np(v)) for v in o[1]]))
    if name == "kde":
        def fit():
            st["m"] = ml.KernelDensity(bandwidth=0.5).fit(x)
        return fit, (lambda: st["m"].score_samples(xq)), (lambda o: float(np.mean(_np(o))))
    if name == "svc":
        def fit():
            st["m"] = ml.SVC(C=1.0, kernel="rbf", gamma=GAMMA).fit(x, yc)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _acc(ycq, _np(o)))
    if name == "svr":
        def fit():
            st["m"] = ml.SVR(C=1.0, kernel="rbf", gamma=GAMMA).fit(x, yr)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _r2(yrq, _np(o)))
    if name == "krr":
        def fit():
            st["m"] = ml.KernelRidge(alpha=1.0, kernel="rbf", gamma=GAMMA).fit(x, yr)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _r2(yrq, _np(o)))
    if name == "gpr":
        def fit():
            kern = ml.ConstantKernel(1.0) * ml.RBF(2.0) + ml.WhiteKernel(0.5)
            st["m"] = ml.GaussianProcessRegressor(kernel=kern).fit(x, yr)
        return fit, (lambda: st["m"].predict(xq)), (lambda o: _r2(yrq, _np(o)))
    if name == "gpc":
        def fit():
            st["m"] = ml.GaussianProcessClassifier(kernel=ml.ConstantKernel(1.0) * ml.RBF(2.0)).fit(x, yc)
        return fit, (lambda: st["m"].predict_proba(xq)), (lambda o: _acc(ycq, np.argmax(_np(o), 1)))
    if name == "nystroem":
        def fit():
            st["m"] = ml.Nystroem(kernel="rbf", gamma=GAMMA, n_components=300, random_state=0).fit(x)
        return fit, (lambda: st["m"].transform(xq)), (lambda o: _kernel_err(xq[:1000], _np(o)[:1000], GAMMA))
    if name == "rbf":
        def fit():
            st["m"] = ml.RBFSampler(gamma=GAMMA, n_components=500, random_state=0).fit(x)
        return fit, (lambda: st["m"].transform(xq)), (lambda o: _kernel_err(xq[:1000], _np(o)[:1000], GAMMA))
    raise SystemExit(f"unknown case {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-quality", action="store_true")
    a = ap.parse_args()
    import mojolearn as ml
    only = set(filter(None, a.only.split(",")))
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical")
    print(f"# x_neighbors_speed mode={mode} reps={a.reps} scale={a.scale}", flush=True)
    for ds in a.dataset.split(","):
        for name, nf, nq in CASES:
            if only and name not in only:
                continue
            nf, nq = max(64, int(nf * a.scale)), max(64, int(nq * a.scale))
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
                print(f"XNSPEED {ds} {name} {nf} {nq} {min(tf):.4f} {min(tp):.4f} {dg} {q:.6f}", flush=True)
            except Exception as e:  # a failing case must not hide the others
                msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
                print(f"XNSPEED {ds} {name} {nf} {nq} FAIL {msg}", flush=True)


if __name__ == "__main__":
    main()
