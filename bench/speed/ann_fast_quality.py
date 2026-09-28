"""The ann lane's FAST quality check (lane ann-apple, 2026-09-28).

A FAST change may move bits but never quality. This prints, per dataset and
seed, the quality of each ann algorithm under the binding the environment
selects (MOJOLEARN_NUMERIC_MODE), so a before commit and an after commit run
at the same seeds give the paired numbers:

  * IVF-PQ / IVF-SQ / IVF-RaBitQ / CAGRA: recall@k against the exact k-NN
    (numpy, float64), the index fitted with random_state = seed;
  * t-SNE: trustworthiness@10 (scikit-learn's formula) and the final KL, the
    initial embedding drawn with random_state = seed.

    MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run python \
      bench/speed/ann_fast_quality.py --data higgs.npz,taxi.npz --seeds 0,1,2,3,4

Rows: the first --n rows (standardized per column, float32) index, the next
--m rows query; t-SNE and CAGRA use the first --n-small rows.
"""
import argparse
import json
import time

import numpy as np


def load(path, n):
    z = np.load(path)
    key = next((k for k in ("X", "x") if k in z.files), z.files[0])
    x = np.asarray(z[key][:n], dtype=np.float64)
    sd = x.std(0)
    x = (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
    return np.ascontiguousarray(x.astype(np.float32))


def exact_knn(x, q, k):
    xx = (x.astype(np.float64) ** 2).sum(1)
    out = np.empty((len(q), k), dtype=np.int64)
    for s in range(0, len(q), 256):
        qq = q[s:s + 256].astype(np.float64)
        d = (qq ** 2).sum(1)[:, None] - 2 * qq @ x.astype(np.float64).T + xx[None, :]
        out[s:s + 256] = np.argsort(d, axis=1, kind="stable")[:, :k]
    return out


def recall(found, truth):
    k = truth.shape[1]
    hit = sum(len(set(f[:k]) & set(t)) for f, t in zip(found, truth))
    return hit / truth.size


def trustworthiness(x, y, k=10):
    n = len(x)
    dx = ((x[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dx, np.inf)
    rank = np.argsort(np.argsort(dx, axis=1), axis=1)
    dy = ((y[:, None, :].astype(np.float64) - y[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dy, np.inf)
    nn_y = np.argsort(dy, axis=1)[:, :k]
    r = np.take_along_axis(rank, nn_y, axis=1) + 1
    t = np.maximum(r - k, 0).sum()
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma separated .npz paths")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--algos", default="ivf_pq,ivf_sq,ivf_rabitq,cagra,tsne")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--n-small", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-lists", type=int, default=256)
    ap.add_argument("--n-probes", type=int, default=16)
    args = ap.parse_args()

    import mojolearn as ml

    seeds = [int(s) for s in args.seeds.split(",")]
    for path in args.data.split(","):
        name = path.rsplit("/", 1)[-1].split(".")[0]
        x_all = load(path, args.n + args.m)
        x, q = x_all[:args.n], x_all[args.n:]
        truth = exact_knn(x, q, args.k)
        xs = x[:args.n_small]
        qs = q
        truth_s = exact_knn(xs, qs, args.k)
        for a in args.algos.split(","):
            for seed in seeds:
                t = time.perf_counter()
                row = dict(data=name, algo=a, seed=seed)
                common = dict(n_lists=args.n_lists, n_probes=args.n_probes, n_neighbors=args.k, random_state=seed)
                if a == "ivf_pq":
                    est = ml.IVFPQIndex(pq_dim=min(14, x.shape[1]), pq_bits=8, **common).fit(x)
                    row["recall"] = recall(np.asarray(est.search(q)[1]), truth)
                elif a == "ivf_sq":
                    est = ml.IVFSQIndex(**common).fit(x)
                    row["recall"] = recall(np.asarray(est.search(q)[1]), truth)
                elif a == "ivf_rabitq":
                    est = ml.IVFRaBitQIndex(**common).fit(x)
                    row["recall"] = recall(np.asarray(est.search(q)[1]), truth)
                elif a == "cagra":
                    if seed != seeds[0]:
                        continue  # CAGRA has no seed: one row per dataset
                    est = ml.CagraIndex(n_neighbors=args.k).fit(xs)
                    row["recall"] = recall(np.asarray(est.search(qs)[1]), truth_s)
                elif a == "tsne":
                    est = ml.TSNE(init="random", random_state=seed, max_iter=500)
                    # lane ann-apple3: the estimator returns mojolearn's own array type, which
                    # refuses `y[:, None, :]`; the metric below is numpy's
                    y = np.asarray(est.fit_transform(xs[:1500]))
                    row["trust10"] = trustworthiness(xs[:1500], y)
                    row["kl"] = float(est.kl_divergence_)
                row["s"] = round(time.perf_counter() - t, 2)
                print("ANN-QUALITY " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
