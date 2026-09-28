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
rows), FITS (0 skips the fits; a comma list of name fragments picks fits),
ARMS (default "1,2"). Round 3 (lane/decomp-apple3): arm d = the tree's own
default on this vendor (every new switch, MOJOLEARN_XD_JACOBI and
MOJOLEARN_XD_J2_U unset: on Metal the eigh is device_eigh, the svd is
jacobi2's), so "d,3" isolates the eigh kernel. The speedup printed is the
first arm over the last. QUALITY=1 scores every eigh and svd arm against
numpy's float64 solve of the same matrix (FAST's paired quality check).
An arm may carry its own switches: "d+MOJOLEARN_XD_PJ_EIGH_MIN=1" is arm d
with that variable set for the arm's calls only. EIGH_KIND=gram times the
eigh of a positive semidefinite Gram matrix (rank n / 2) beside the
indefinite B + B^T."""
import os, sys, time, hashlib, warnings
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
warnings.filterwarnings("ignore")
import numpy as np
import mojolearn as ml
from mojolearn._expansion_decomp import _M, _Kit
from mojolearn import _backend

ARMS = os.environ.get("ARMS", "1,2").split(",")
# arm 3 = arm 2 with the eigh kernel at unroll 1 (MOJOLEARN_XD_J2_U)
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
        if not isinstance(m, _M):
            d.update(np.ascontiguousarray(np.asarray(m, dtype=np.float32)).tobytes())
        else:
            d.update(bytes(memoryview(m.s).cast("B")))
    return d.hexdigest()[:16]


OLD = {"MOJOLEARN_XD_JACOBI": "1", "MOJOLEARN_XD_LU_SERIAL": "0", "MOJOLEARN_XD_ALS_TEAM": "0", "MOJOLEARN_XD_ORTH_DEV": "0"}
NEW = {"MOJOLEARN_XD_JACOBI": "2", "MOJOLEARN_XD_LU_SERIAL": "16", "MOJOLEARN_XD_ALS_TEAM": "1", "MOJOLEARN_XD_ORTH_DEV": "1"}


QUALITY = os.environ.get("QUALITY", "0") == "1"
FITS = os.environ.get("FITS", "1")


def arm(name, fn, quality=None):
    """arm 1 = every switch of this lane at its shipped-before value, arm 2 =
    every switch at the new default (eigh unroll 4), arm 3 = arm 2 at eigh
    unroll 1, arm d = the tree's default (see the module text). `quality`
    (QUALITY=1): a function of the arm's output returning a short string."""
    if name.split(" ")[0].split("(")[0] not in ("eigh", "svd", "orth") and FITS not in ("0", "1"):
        if not any(f and f in name for f in FITS.split(",")):
            return
    res = {}
    for a in ARMS:
        base, *extra = a.split("+")
        os.environ.update(OLD if base == "1" else NEW)
        os.environ["MOJOLEARN_XD_J2_U"] = "1" if base == "3" else "4"
        if base == "d":
            os.environ.pop("MOJOLEARN_XD_JACOBI", None)
            os.environ.pop("MOJOLEARN_XD_J2_U", None)
        for kv in extra:
            os.environ[kv.split("=", 1)[0]] = kv.split("=", 1)[1]
        t = time.perf_counter()
        try:
            out = fn()
            dt = time.perf_counter() - t
            q = ""
            if QUALITY and quality is not None:
                q = " [" + quality(out) + "]"
            res[a] = (dt, h(*out), q)
        except Exception as e:  # report, keep going
            res[a] = (float("nan"), "ERR " + str(e)[-300:], "")
        for kv in extra:
            os.environ.pop(kv.split("=", 1)[0], None)
    line = "  ".join(f"arm{a} {res[a][0]:9.3f}s {res[a][1]}{res[a][2]}" for a in ARMS)
    hs = {res[a][1] for a in ARMS}
    verdict = "EQUAL" if len(hs) == 1 and not next(iter(hs)).startswith("ERR") else "DIFFER"
    sp = ""
    first, last = res[ARMS[0]], res[ARMS[-1]]
    if len(ARMS) > 1 and last[0] > 0:
        sp = f" speedup {first[0] / last[0]:.2f}x"
    print(f"{name:34s} {line}  {verdict}{sp}", flush=True)


def eigh_quality(A):
    """Against numpy's float64 eigh of the same float32 matrix: the largest
    eigenvalue error over the largest |eigenvalue|, the residual
    ||A V - V diag(w)||_F / ||A||_F and ||V^T V - I||_F / sqrt(n)."""
    A64 = np.asarray(A.out(), np.float64)
    ref = np.linalg.eigvalsh(A64)

    def q(out):
        w = np.asarray(out[0].out(), np.float64).ravel()
        V = np.asarray(out[1].out(), np.float64)
        n = V.shape[0]
        werr = np.max(np.abs(w - ref)) / np.max(np.abs(ref))
        resid = np.linalg.norm(A64 @ V - V * w[None, :]) / np.linalg.norm(A64)
        orth = np.linalg.norm(V.T @ V - np.eye(n)) / np.sqrt(n)
        return f"w {werr:.2e} res {resid:.2e} orth {orth:.2e}"
    return q


def svd_quality(A):
    """Against numpy's float64 singular values of the same matrix: the
    largest singular value error over the largest singular value, and
    ||Vt Vt^T - I||_F / sqrt(n)."""
    A64 = np.asarray(A.out(), np.float64)
    ref = np.linalg.svd(A64, compute_uv=False)

    def q(out):
        s = np.asarray(out[0].out(), np.float64).ravel()
        Vt = np.asarray(out[1].out(), np.float64)
        n = Vt.shape[0]
        serr = np.max(np.abs(s - ref)) / np.max(np.abs(ref))
        keep = ref > 1e-6 * ref[0]
        srel = np.max(np.abs(s[keep] - ref[keep]) / ref[keep])
        orth = np.linalg.norm(Vt @ Vt.T - np.eye(n)) / np.sqrt(n)
        # the right vectors against A: ||A v_i|| is s_i
        av = np.linalg.norm(A64 @ Vt.T, axis=0)
        vres = np.max(np.abs(av - s)) / ref[0]
        return f"s {serr:.2e} srel {srel:.2e} orth {orth:.2e} Av {vres:.2e}"
    return q


rng = np.random.default_rng(7)
for n in [int(x) for x in os.environ.get("EIGH", "8,64,256,800,1500").split(",") if x]:
    B = rng.standard_normal((n, n)).astype(np.float32)
    S = M(B + B.T)
    arm(f"eigh {n}", lambda: k.eigh(S), eigh_quality(S) if QUALITY else None)
    if os.environ.get("EIGH_KIND", "") == "gram":
        C = rng.standard_normal((n, max(n // 2, 1))).astype(np.float32)
        G0 = C @ C.T
        G = M((G0 + G0.T) * np.float32(0.5))
        arm(f"eigh gram {n}", lambda: k.eigh(G), eigh_quality(G) if QUALITY else None)
for mn in [x for x in os.environ.get("SVD", "200000:28,5000:64,1000:256,800:800,1500:1500").split(",") if x]:
    m, n = (int(v) for v in mn.split(":"))
    A = M(rng.standard_normal((m, n)).astype(np.float32))
    arm(f"svd {m}x{n}", lambda: k.svd(A), svd_quality(A) if QUALITY else None)

for mn in [x for x in os.environ.get("ORTH", "200000:15").split(",") if x]:
    m, n = (int(v) for v in mn.split(":"))
    A = M(rng.standard_normal((m, n)).astype(np.float32))
    arm(f"orth {m}x{n}", lambda: (k.orth(A),))
if FITS != "0":
    Xr = (rng.standard_normal((200000, 28)) @ rng.standard_normal((28, 28))).astype(np.float32)
    arm("PCA(randomized,5) 200000x28", lambda: (ml.PCA(n_components=5, svd_solver="randomized", random_state=0).fit(Xr).components_,))
    arm("randomized_svd(5) 200000x28", lambda: tuple(np.asarray(a) for a in ml.randomized_svd(Xr, 5, random_state=0)))
    arm("FactorAnalysis(5) 200000x28", lambda: (ml.FactorAnalysis(n_components=5, max_iter=20).fit(Xr).components_,))
    arm("lstsq 200000x27", lambda: (np.asarray(ml.lstsq(Xr[:, :-1], Xr[:, -1])[0]),))
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
    # the per-point solves of the other three methods (tiny eigh calls); only when named in FITS
    n5 = int(os.environ.get("N5", "400"))
    X5 = X[:n5]
    if FITS not in ("0", "1"):
        arm(f"LLE-ltsa(10nn) {n5}", lambda: (ml.LocallyLinearEmbedding(n_neighbors=10, method="ltsa").fit_transform(X5),))
        arm(f"LLE-modified(10nn) {n5}", lambda: (ml.LocallyLinearEmbedding(n_neighbors=10, method="modified").fit_transform(X5),))
        arm(f"LLE-hessian(10nn) {n5}", lambda: (ml.LocallyLinearEmbedding(n_neighbors=10, method="hessian").fit_transform(X5),))
