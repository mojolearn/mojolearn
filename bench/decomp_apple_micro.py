# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_apple_micro.py -- the fixed and per-byte cost of one x_decomp
kit call on this GPU (lane/decomp-apple). Each entry is timed REPS times
after one warm call, at several shapes, and the median milliseconds per call
is printed with the bytes it moved (inputs up, output down), so a fixed cost
(sync, allocation) separates from a per-byte cost (copies, page faults).
env: REPS (default 10), MOJOLEARN_NUMERIC_MODE."""
import os, sys, time, statistics
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
import mojolearn._expansion_decomp as ed
from mojolearn._expansion_decomp import _M, _Kit
from mojolearn import _backend

REPS = int(os.environ.get("REPS", "10"))
k = _Kit(_backend.default_mode())


def filled(r, c):
    m = _M.zeros(r, c)
    for i in range(0, r * c, 97):
        m.s[i] = (i % 13) * 0.25 - 1.0
    return m


def t(name, fn, nbytes):
    fn()
    ts = []
    for _ in range(REPS):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    med = statistics.median(ts) * 1e3
    gbps = nbytes / (med * 1e-3) / 1e9 if med > 0 else 0.0
    print(f"{name:34s} {med:9.3f} ms  min {min(ts)*1e3:9.3f}  {nbytes/1e6:9.2f} MB  {gbps:6.2f} GB/s", flush=True)


print("x_decomp micro", k.b.x_decomp_vendor() if hasattr(k.b, "x_decomp_vendor") else "", "reps", REPS, flush=True)
for r, c in ((1, 1), (1000, 1), (100000, 1), (1000000, 1), (1000000, 5), (1000000, 28)):
    A, B = filled(r, c), filled(r, c)
    nb = 4 * r * c
    t(f"ew add {r}x{c}", lambda: k.ew("add", A, B), 3 * nb)
    t(f"ew scale {r}x{c}", lambda: k.ew("scale", A, s=2.0), 2 * nb)
    t(f"colsum {r}x{c}", lambda: k.colsum(A), nb)
    t(f"rowsum {r}x{c}", lambda: k.rowsum(A), nb)
for r, c, n in ((1000, 28, 5), (100000, 28, 5), (1000000, 28, 5), (1000000, 5, 5)):
    A, B = filled(r, c), filled(c, n)
    t(f"gemm {r}x{c} @ {c}x{n}", lambda: k.mm(A, B), 4 * (r * c + c * n + r * n))
    At = filled(r, c)
    t(f"gemm ta {c}x{r} @ {r}x{c}", lambda: k.mm(At, At, ta=True), 8 * r * c)
for r, c in ((100000, 5), (1000000, 5), (1000000, 10)):
    A = filled(r, c)
    for i in range(c):
        A.s[i * c + i] += 3.0
    t(f"orth {r}x{c}", lambda: k.orth(A), 8 * r * c)
    t(f"absmax_sign col {r}x{c}", lambda: k.absmax_flags(A, True), 4 * r * c)
for n in (28, 64, 256):
    S = filled(n, n)
    S = k.ew("add", S, S.T)
    t(f"eigh {n}", lambda: k.eigh(S), 12 * n * n)
