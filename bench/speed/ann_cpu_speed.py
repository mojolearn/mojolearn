"""The ann lane's CPU speed bench (lane/ann-cpu, 2026-09-28).

Times every ann algorithm's host (CPU) binding through the public classes,
once per cell, and prints a sha256 of every output, so one run is both the
before/after timing and the bit check (run it at MOJOLEARN_CPU_THREADS = 1, 3
and unset and the digests must match; before and after a change they must
match too).

    MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=python/mojolearn/host \
      pixi run python bench/speed/ann_cpu_speed.py --data higgs.npz --algos all --out r.json

Data: the first `--n` rows of `--data` (an .npz with an `X` array, staged
from R2 by tools/dataset_store.sh; HIGGS is 11M x 28), standardized per
column in float64 then cast to float32; queries are the next `--m` rows.
Without `--data`, a seeded Gaussian blob mixture (for a smoke only).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load(args, n, m):
    if args.data:
        z = np.load(args.data)
        key = next((k for k in ("X", "x") if k in z.files), z.files[0])
        x = np.asarray(z[key][: n + m], dtype=np.float64)
    else:
        rng = np.random.default_rng(7)
        c = rng.normal(size=(32, args.dim)) * 4.0
        lab = rng.integers(0, 32, size=n + m)
        x = c[lab] + rng.normal(size=(n + m, args.dim))
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    x = x.astype(np.float32)
    return np.ascontiguousarray(x[:n]), np.ascontiguousarray(x[n:n + m])


def gpu_fit(args, algo):
    """Fit `algo` in a child with the GPU route, return the unpickled
    estimator and the child's wall time (recorded as fit_s, GPU)."""
    import pickle
    import subprocess
    import sys
    import tempfile
    env = {k: v for k, v in os.environ.items() if k not in ("MOJOLEARN_VENDOR", "MOJOLEARN_HOST_DIR")}
    stem = tempfile.mktemp(prefix="ann_fit_")
    argv = [a for a in sys.argv[1:] if a != "--fit-on-gpu"]
    t = time.perf_counter()
    subprocess.run([sys.executable, "-u", __file__, *argv, "--algos", algo, "--fit-only", stem, "--out", ""],
                   env=env, check=True)
    dt = time.perf_counter() - t
    with open(f"{stem}.{algo}.pkl", "rb") as f:
        est = pickle.load(f)
    return est, dt


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--dim", type=int, default=28)
    ap.add_argument("--algos", default="all")
    ap.add_argument("--n", type=int, default=1_000_000, help="index rows (IVF family)")
    ap.add_argument("--m", type=int, default=1000, help="queries")
    ap.add_argument("--n-graph", type=int, default=50_000, help="rows for CAGRA (exact kNN build is O(n^2))")
    ap.add_argument("--n-tsne", type=int, default=10_000, help="rows for t-SNE (exact repulsion, O(n^2) per step)")
    ap.add_argument("--tsne-iter", type=int, default=300)
    ap.add_argument("--n-lists", type=int, default=1024)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--kmeans-iters", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-on-gpu", action="store_true",
                    help="IVF family: fit on the GPU in a child process (MOJOLEARN_VENDOR unset), pickle the "
                         "fitted estimator, then time only the CPU search (the fitted arrays are the same bits)")
    ap.add_argument("--fit-only", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    import mojolearn as ml

    algos = args.algos.split(",") if args.algos != "all" else [
        "ivf", "ivf_pq", "ivf_sq", "ivf_rabitq", "refine", "cagra", "tsne"]
    rec = dict(threads=os.environ.get("MOJOLEARN_CPU_THREADS", ""), vendor=os.environ.get("MOJOLEARN_VENDOR", ""),
               n=args.n, m=args.m, cells={})
    x, q = load(args, args.n, args.m)
    k = args.k

    def cell(name, **kv):
        rec["cells"][name] = kv
        print(json.dumps({name: kv}), flush=True)

    common = dict(n_lists=args.n_lists, n_probes=args.n_probes, n_neighbors=k, kmeans_n_iters=args.kmeans_iters)
    for a in algos:
        if a == "ivf":
            est = ml.IVFIndex(**common)
            if args.fit_only:
                est.fit(x)
                import pickle
                with open(f"{args.fit_only}.{a}.pkl", "wb") as f:
                    pickle.dump(est, f)
                continue
            if args.fit_on_gpu:
                est, tf = gpu_fit(args, a)
            else:
                _, tf = timed(lambda: est.fit(x))
            (d, i), ts = timed(lambda: est.search(q))
            # lane ann-apple3: the second search of the same queries (the index is resident and the
            # kernels are warm); its digest must equal the first's
            (d2, i2), ts2 = timed(lambda: est.search(q))
            cell(a, fit_s=round(tf, 3), search_s=round(ts, 3), search2_s=round(ts2, 4), out=digest(d, i),
                 out2=digest(d2, i2))
        elif a in ("ivf_pq", "ivf_sq", "ivf_rabitq", "refine"):
            cls = {"ivf_pq": ml.IVFPQIndex, "ivf_sq": ml.IVFSQIndex, "ivf_rabitq": ml.IVFRaBitQIndex,
                   "refine": ml.IVFPQIndex}[a]
            kw = dict(common)
            if cls is ml.IVFPQIndex:
                kw.update(pq_dim=min(14, args.dim), pq_bits=8, pq_kmeans_n_iters=args.kmeans_iters)
            if a == "refine":
                kw["n_neighbors"] = 4 * k
            est = cls(**kw)
            if args.fit_only:
                est.fit(x)
                import pickle
                with open(f"{args.fit_only}.{a}.pkl", "wb") as f:
                    pickle.dump(est, f)
                continue
            if args.fit_on_gpu:
                est, tf = gpu_fit(args, a)
            else:
                _, tf = timed(lambda: est.fit(x))
            (d, i), ts = timed(lambda: est.search(q))
            (d2, i2), ts2 = timed(lambda: est.search(q))
            if a == "refine":
                (rd, ri), tr = timed(lambda: ml.refine(x, q, i, k))
                (rd2, ri2), tr2 = timed(lambda: ml.refine(x, q, i2, k))
                cell(a, fit_s=round(tf, 3), search_s=round(ts, 3), search2_s=round(ts2, 4), refine_s=round(tr, 3),
                     refine2_s=round(tr2, 4), out=digest(rd, ri), out2=digest(rd2, ri2))
            else:
                parts = [est.centers_, est.codes_] if hasattr(est, "codes_") else [est.centers_]
                cell(a, fit_s=round(tf, 3), search_s=round(ts, 3), search2_s=round(ts2, 4), model=digest(*parts),
                     out=digest(d, i), out2=digest(d2, i2))
        elif a == "cagra":
            xg = x[: args.n_graph]
            est = ml.CagraIndex()
            _, tf = timed(lambda: est.fit(xg))
            (d, i), ts = timed(lambda: est.search(q))
            (d2, i2), ts2 = timed(lambda: est.search(q))
            cell(a, n=args.n_graph, fit_s=round(tf, 3), search_s=round(ts, 3), search2_s=round(ts2, 4),
                 model=digest(est.graph_), out=digest(d, i), out2=digest(d2, i2))
        elif a == "tsne":
            xt = x[: args.n_tsne]
            est = ml.TSNE(max_iter=args.tsne_iter)
            _, tf = timed(lambda: est.fit(xt))
            cell(a, n=args.n_tsne, iters=args.tsne_iter, fit_s=round(tf, 3),
                 out=digest(est.embedding_, np.float32(est.kl_divergence_)))
        else:
            raise SystemExit(f"unknown algo {a}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
