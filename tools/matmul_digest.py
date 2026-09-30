#!/usr/bin/env python3
"""SHA-256 of mojolearn.matmul at several seeded shapes (IDENTICAL), plus median wall per call: the A/B for gemm host-entry changes."""
import hashlib, json, statistics, sys, time
import numpy as np
from mojolearn import matmul
rng = np.random.default_rng(20260930); out = {}
for m, n, k in [(1, 1, 1), (7, 5, 3), (64, 64, 64), (256, 195, 13), (512, 384, 1024), (1024, 1024, 1024), (4096, 512, 256)]:
    a = rng.standard_normal((m, k)).astype(np.float32); b = rng.standard_normal((k, n)).astype(np.float32)
    ts = []
    for _ in range(7):
        t = time.perf_counter(); c = np.asarray(matmul(a, b)); ts.append((time.perf_counter() - t) * 1000)
    out["%dx%dx%d" % (m, n, k)] = {"sha": hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()[:16], "ms": statistics.median(ts[2:])}
    for ta, tb in [(True, False), (False, True)]:
        aa = np.ascontiguousarray(a.T) if ta else a; bb = np.ascontiguousarray(b.T) if tb else b
        c = np.asarray(matmul(aa, bb, transpose_a=ta, transpose_b=tb))
        out["%dx%dx%d.t%d%d" % (m, n, k, ta, tb)] = {"sha": hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()[:16]}
print(json.dumps(out, indent=1))
