# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_jacobi_ab.py -- the x_decomp kit's two Jacobi solvers, the
shipped-before kernels (MOJOLEARN_XD_JACOBI=1) against x_decomp/jacobi2.mojo
(=2), in ONE process on one box (lane/decomp-apple2). Per shape: seconds of
one call for each arm and the sha256 of every output byte; the IDENTICAL
contract is that the two hashes are EQUAL. Then the manifold fits whose
time is those solvers (Isomap, ClassicalMDS, LocallyLinearEmbedding) at
N3 rows, ALS and MinCovDet (the LU and ALS switches), timed and hashed
per arm. env: EIGH (comma sizes), SVD (comma m:n), N3, N2 (MinCovDet
rows), FITS (0 skips the fits), ARMS (default "1,2")."""
import os, sys, time, hashlib, warnings
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
warnings.filterwarnings("ignore")
import numpy as np
import mojolearn as ml
from mojolearn._expansion_decomp import _M, _Kit
from mojolearn import _backend

ARMS = os.environ.get("ARMS", "1,2").split(",")
k = _Kit(_backend.default_mode())


def M(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    m = _M.zeros(a.shape[0], a.shape[1])
    for i, x in enumerate(a.ravel().tolist()):
        m.s[i] = x
    return m


def h(*ms):
    d = hashlib.sha256()
    for m in ms:
        if isinstance(m, np.ndarray):
            d.update(np.ascontiguousarray(m).tobytes())
        else:
            d.update(bytes(memoryview(m.s).cast("B")))
    return d.hexdigest()[:16]


OLD = {"MOJOLEARN_XD_JACOBI": "1", "MOJOLEARN_XD_LU_SERIAL": "0", "MOJOLEARN_XD_ALS_TEAM": "0"}
NEW = {"MOJOLEARN_XD_JACOBI": "2", "MOJOLEARN_XD_LU_SERIAL": "16", "MOJOLEARN_XD_ALS_TEAM": "1"}


def arm(name, fn):
    """arm 1 = every switch of this lane at its shipped-before value, arm 2 =
    every switch at the new default."""
    res = {}
    for a in ARMS:
        os.environ.update(OLD if a == "1" else NEW)
        t = time.perf_counter()
        try:
            out = fn()
            dt = time.perf_counter() - t
            res[a] = (dt, h(*out))
        except Exception as e:  # report, keep going
            res[a] = (float("nan"), "ERR " + str(e)[:120])
    line = "  ".join(f"arm{a} {res[a][0]:9.3f}s {res[a][1]}" for a in ARMS)
    hs = {res[a][1] for a in ARMS}
    verdict = "EQUAL" if len(hs) == 1 and not next(iter(hs)).startswith("ERR") else "DIFFER"
    sp = ""
    if "1" in res and "2" in res and res["2"][0] > 0:
        sp = f" speedup {res['1'][0] / res['2'][0]:.2f}x"
    print(f"{name:34s} {line}  {verdict}{sp}", flush=True)


rng = np.random.default_rng(7)
for n in [int(x) for x in os.environ.get("EIGH", "8,64,256,800,1500").split(",") if x]:
    B = rng.standard_normal((n, n)).astype(np.float32)
    S = M(B + B.T)
    arm(f"eigh {n}", lambda: k.eigh(S))
for mn in [x for x in os.environ.get("SVD", "200000:28,5000:64,1000:256,800:800,1500:1500").split(",") if x]:
    m, n = (int(v) for v in mn.split(":"))
    A = M(rng.standard_normal((m, n)).astype(np.float32))
    arm(f"svd {m}x{n}", lambda: k.svd(A))

if os.environ.get("FITS", "1") != "0":
    rr = np.random.default_rng(0)
    R = (rr.random((20000, 2000)) < 0.01).astype(np.float32)
    arm("ALS(32f,5it) 20000x2000", lambda: (ml.AlternatingLeastSquares(factors=32, iterations=5, random_state=0).fit(R).user_factors,))
    Xc = (rr.standard_normal((int(os.environ.get("N2", "20000")), 8)) @ rr.standard_normal((8, 8))).astype(np.float32)
    arm(f"MinCovDet {Xc.shape[0]}x8", lambda: (ml.MinCovDet(random_state=0).fit(Xc).covariance_,))
    n3 = int(os.environ.get("N3", "1500"))
    rg = np.random.default_rng(0)
    X = (rg.standard_normal((n3, 28)) @ rg.standard_normal((28, 28))).astype(np.float32)
    arm(f"Isomap(10nn) {n3}", lambda: (ml.Isomap(n_neighbors=10).fit_transform(X),))
    arm(f"ClassicalMDS {n3}", lambda: (ml.ClassicalMDS().fit_transform(X),))
    arm(f"LocallyLinearEmbedding(10nn) {n3}", lambda: (ml.LocallyLinearEmbedding(n_neighbors=10).fit_transform(X),))
