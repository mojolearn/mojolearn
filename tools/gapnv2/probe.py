"""lane/gap-nv-classical2 stage probe (GPU box only): where each algorithm's
wall time goes at the board's shapes, on synthetic data of the same shape.
python tools/gapnv2/probe.py <rsvd|ivf-sq|ivf-refine|lasso|enet|hdbscan|tsne> [istella|taxi]"""
import os, sys, time, functools
import numpy as np

algo = sys.argv[1]
ds = sys.argv[2] if len(sys.argv) > 2 else "istella"
d = 220 if ds == "istella" else 11
rng = np.random.default_rng(7)
T = {}


def wrap(obj, name, label=None):
    f = getattr(obj, name)
    lab = label or name

    @functools.wraps(f)
    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            dt = (time.perf_counter() - t) * 1e3
            c = T.setdefault(lab, [0, 0.0])
            c[0] += 1
            c[1] += dt
    setattr(obj, name, g)


def report(tag, total):
    print(f"PROBE {algo} {ds} {tag} total_ms={total:.1f}")
    for k, (n, ms) in sorted(T.items(), key=lambda kv: -kv[1][1]):
        print(f"PROBE   {k:28s} n={n:5d} ms={ms:9.1f}")
    T.clear()


import mojolearn as ml

if algo == "rsvd":
    from mojolearn import _expansion_decomp as xd
    X = rng.standard_normal((900_000, d), dtype=np.float32)
    for nm in ("mm", "orth", "rand", "colsum", "ew", "svd", "eigh", "absmax_flags"):
        if hasattr(xd._Kit, nm):
            wrap(xd._Kit, nm)
    for nm in ("from_input", "out", "cols", "rows", "take_cols"):
        wrap(xd._M, nm, "M." + nm)
    for r in range(2):
        t = time.perf_counter()
        xd.randomized_svd(X, 8, n_oversamples=10, n_iter=4, random_state=7)
        report(f"round{r}", (time.perf_counter() - t) * 1e3)
elif algo in ("ivf-sq", "ivf-refine"):
    os.environ["MOJOLEARN_ANN_STAGES"] = "1"
    X = rng.standard_normal((400_000, d), dtype=np.float32)
    Q = rng.standard_normal((4000, d), dtype=np.float32)
    for r in range(2):
        t = time.perf_counter()
        if algo == "ivf-sq":
            idx = ml.IVFSQIndex(n_lists=1024, n_probes=32, n_neighbors=10, kmeans_n_iters=20, random_state=7).fit(X)
            t1 = time.perf_counter()
            idx.search(Q)
        else:
            idx = ml.IVFPQIndex(n_lists=1024, n_probes=32, n_neighbors=40, pq_bits=8, pq_dim=max(1, d // 4),
                                kmeans_n_iters=20, random_state=7).fit(X)
            t1 = time.perf_counter()
            r_ = idx.search(Q)
            cand = r_[1] if isinstance(r_, tuple) else r_
            t2 = time.perf_counter()
            ml.refine(X, Q, cand, 10)
            print(f"PROBE {algo} refine_ms={(time.perf_counter() - t2) * 1e3:.1f}")
        print(f"PROBE {algo} {ds} round{r} fit_ms={(t1 - t) * 1e3:.1f} total_ms={(time.perf_counter() - t) * 1e3:.1f}")
elif algo in ("lasso", "enet"):
    X = rng.standard_normal((1_000_000, d), dtype=np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = (X @ w + rng.standard_normal(1_000_000).astype(np.float32)).astype(np.float32)
    for r in range(2):
        m = ml.Lasso(alpha=0.01, max_iter=1000, tol=1e-4) if algo == "lasso" else \
            ml.ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
        t = time.perf_counter()
        m.fit(X, y)
        print(f"PROBE {algo} {ds} round{r} fit_ms={(time.perf_counter() - t) * 1e3:.1f} n_iter={m.n_iter_}")
else:
    raise SystemExit("unknown algo " + algo)
