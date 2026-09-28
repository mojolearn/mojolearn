# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_fast_quality.py -- the paired quality check of FAST's Lanczos
top-eigenpair route (lane/decomp-apple2, `_top_eig` / `_lanczos_top`) for
Isomap and ClassicalMDS: 2 datasets (swiss roll, correlated Gaussian) x 5
seeds at N rows. Every fit runs IDENTICAL (the exact dense Jacobi) and FAST
(Lanczos) on this box's GPU; both are scored against the float64 numpy
eigendecomposition of the SAME centred kernel (the reference both solvers
approximate): the largest relative eigenvalue error and the largest
sign-aligned eigenvector error over the components. FAST passes a fit when
both of its errors are at most max(1.5 x IDENTICAL's, 1e-5). When
scikit-learn imports, its fit (eigen_solver 'auto' = ARPACK here) is scored
the same way for the record. env: N (default 1000), SEEDS (default 5)."""
import os, sys, time, warnings
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
warnings.filterwarnings("ignore")
import numpy as np
import mojolearn as ml

N = int(os.environ.get("N", "1000"))
SEEDS = int(os.environ.get("SEEDS", "5"))
try:
    import sklearn.manifold as skm
except Exception:
    skm = None


def data(kind, seed):
    g = np.random.default_rng(seed)
    if kind == "swissroll":
        t = 1.5 * np.pi * (1 + 2 * g.random(N))
        x = np.stack([t * np.cos(t), 21 * g.random(N), t * np.sin(t)], 1)
        return (x + 0.05 * g.standard_normal(x.shape)).astype(np.float32)
    return (g.standard_normal((N, 12)) @ g.standard_normal((12, 12))).astype(np.float32)


def centred(D2):
    n = D2.shape[0]
    G = -0.5 * D2
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    return G


def score(G64, w, V):
    ew, ev = np.linalg.eigh(G64)
    nc = V.shape[1]
    ew, ev = ew[::-1][:nc], ev[:, ::-1][:, :nc]
    werr = float(np.max(np.abs(np.asarray(w, np.float64) - ew) / np.abs(ew)))
    verr = float(max(min(np.linalg.norm(V[:, i] - ev[:, i]), np.linalg.norm(V[:, i] + ev[:, i])) for i in range(nc)))
    return werr, verr


def ours_w(m):
    if hasattr(m, "eigenvalues_"):
        return np.asarray(m.eigenvalues_, np.float64).ravel()
    return np.asarray(m.eigenvalues_m_.out(), np.float64).ravel()


def fit(est_cls, mode, X, **kw):
    t = time.perf_counter()
    m = est_cls(numeric_mode=mode, **kw).fit(X)
    return m, time.perf_counter() - t


fails = 0
rows = 0
for kind in ("swissroll", "gauss"):
    for seed in range(SEEDS):
        X = data(kind, seed)
        for name in ("Isomap", "ClassicalMDS"):
            cls = getattr(ml, name)
            kw = {"n_neighbors": 10} if name == "Isomap" else {}
            mi, ti = fit(cls, "identical", X, **kw)
            mf, tf = fit(cls, "fast", X, **kw)
            if name == "Isomap":
                D = np.asarray(mi.dist_matrix_, np.float64)
                Df = np.asarray(mf.dist_matrix_, np.float64)
                same_graph = bool(np.array_equal(D, Df))
            else:
                X64 = X.astype(np.float64)
                sq = (X64 * X64).sum(1)
                D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X64 @ X64.T, 0))
                same_graph = True
            G = centred(D * D)
            emb_i, emb_f = np.asarray(mi.embedding_, np.float64), np.asarray(mf.embedding_, np.float64)
            wi, wf = ours_w(mi), ours_w(mf)
            si = score(G, wi, emb_i / np.sqrt(np.maximum(wi, 1e-30)))
            sf = score(G, wf, emb_f / np.sqrt(np.maximum(wf, 1e-30)))
            ok = sf[0] <= max(1.5 * si[0], 1e-5) and sf[1] <= max(1.5 * si[1], 1e-5) and same_graph
            fails += 0 if ok else 1
            rows += 1
            extra = ""
            if skm is not None and hasattr(skm, name):
                try:
                    t = time.perf_counter()
                    sm = getattr(skm, name)(n_components=2, **kw).fit(X)
                    ts = time.perf_counter() - t
                    ws = np.asarray(sm.eigenvalues_ if hasattr(sm, "eigenvalues_") else sm.kernel_pca_.eigenvalues_, np.float64)
                    ss = score(G, ws, np.asarray(sm.embedding_, np.float64) / np.sqrt(np.maximum(ws, 1e-30)))
                    extra = f"  sklearn {ts:7.3f}s w {ss[0]:.2e} v {ss[1]:.2e}"
                except Exception as e:
                    extra = f"  sklearn ERR {str(e)[:60]}"
            print(f"{name:13s} {kind:9s} seed {seed}  IDENTICAL {ti:8.3f}s w {si[0]:.2e} v {si[1]:.2e}   "
                  f"FAST {tf:7.3f}s w {sf[0]:.2e} v {sf[1]:.2e}  {'PASS' if ok else 'FAIL'}{extra}", flush=True)
print(f"FAST quality: {rows - fails}/{rows} PASS", flush=True)
