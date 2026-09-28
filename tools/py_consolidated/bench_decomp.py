#!/usr/bin/env python3
"""bench_decomp.py: seconds and output sha256 for the py-decomp-nbrs decomp items,
run by the SAME script against the base tree and the head tree (PYTHONPATH picks
the tree). env ONLY=comma names; MOJOLEARN_VENDOR=cpu for the CPU column."""
import hashlib, os, sys, time
import numpy as np
import mojolearn as ml

failures = []

col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        m.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return m.hexdigest()[:16]


def mcd(n):
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((n, 8)) @ rng.standard_normal((8, 8))).astype(np.float32)
    X[: n // 20] += 6
    m = ml.MinCovDet(random_state=0).fit(X)
    return h(m.location_, m.covariance_, m.dist_, np.asarray(m.support_, dtype=np.int8))


def lda(n, v, method, it):
    rng = np.random.default_rng(2)
    C = np.floor(rng.random((n, v)) ** 4 * 4).astype(np.float32)
    m = ml.LatentDirichletAllocation(n_components=10, learning_method=method, max_iter=it, random_state=0).fit(C)
    return h(m.components_)


def mds(n, it):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, 6)).astype(np.float32)
    return h(ml.MDS(metric_mds=False, max_iter=it, n_init=1, random_state=0).fit_transform(X))


CASES = {
    "mcd-20k": lambda: mcd(20000),
    "mcd-200k": lambda: mcd(200000),
    "lda-online-20kx500": lambda: lda(20000, 500, "online", 2),
    "lda-batch-100kx1000": lambda: lda(100000, 1000, "batch", 3),
    "mds-nm-1500": lambda: mds(1500, 20),
}
for name, f in CASES.items():
    if ONLY and name not in ONLY:
        continue
    t = time.perf_counter()
    try:
        d = f()
        print(f"BENCH {col} {name} {time.perf_counter() - t:.2f}s {d}", flush=True)
    except Exception as e:
        failures.append(name)
        print(f"BENCH {col} {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

if failures:
    print("INCOMPLETE/FAILED cases:", ",".join(failures), flush=True)
    raise SystemExit(1)
