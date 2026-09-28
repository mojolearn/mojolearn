#!/usr/bin/env python3
"""py-dn-ann timing: search wall time of IVF-Flat, IVF-PQ and DistributedIVFIndex,
fit excluded, plus a sha256 of every output so base and head can be compared.
env: SIZE (rows), DIM, BATCHES, BQ (queries per batch), WHAT (comma list)."""
import hashlib, os, sys, time
import numpy as np
import mojolearn as ml

n = int(os.environ.get("SIZE", "1000000")); dim = int(os.environ.get("DIM", "128"))
batches = int(os.environ.get("BATCHES", "20")); bq = int(os.environ.get("BQ", "500"))
what = set(os.environ.get("WHAT", "flat,pq,dist").split(","))
col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
rng = np.random.default_rng(7)
X = rng.standard_normal((n, dim), dtype=np.float32)
Q = rng.standard_normal((batches * bq, dim), dtype=np.float32)


def run(name, idx, search):
    h = hashlib.sha256()
    t0 = time.perf_counter(); first = None
    for b in range(batches):
        d, i = search(Q[b * bq:(b + 1) * bq])
        h.update(np.ascontiguousarray(np.asarray(d)).tobytes()); h.update(np.ascontiguousarray(np.asarray(i)).tobytes())
        if first is None:
            first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    print(f"BENCH {col} {name} n={n} dim={dim} batches={batches}x{bq} total_s={tot:.3f} first_s={first:.3f} "
          f"per_later_s={(tot - first) / max(batches - 1, 1):.4f} sha={h.hexdigest()[:16]}", flush=True)


if "flat" in what:
    t = time.perf_counter(); f = ml.IVFIndex(n_lists=1024, n_probes=8, n_neighbors=8, kmeans_n_iters=5).fit(X)
    print(f"FIT {col} ivf_flat {time.perf_counter() - t:.1f}s", flush=True)
    run("ivf_flat", f, f.search)
    if "dist" in what:
        from mojolearn.parallel_ivf import DistributedIVFIndex
        devs = tuple(int(v) for v in os.environ.get("DEVICES", "0,1").split(","))
        with DistributedIVFIndex.from_index(f, devices=devs) as di:
            run(f"dist_ivf_{len(devs)}dev", di, di.search)
if "pq" in what:
    t = time.perf_counter(); p = ml.IVFPQIndex(n_lists=1024, n_probes=8, pq_dim=16, kmeans_n_iters=5, pq_kmeans_n_iters=5).fit(X)
    print(f"FIT {col} ivf_pq {time.perf_counter() - t:.1f}s", flush=True)
    run("ivf_pq", p, p.search)
