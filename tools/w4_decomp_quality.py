#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane apple-fast-w4-decomp quality: dump one arm's outputs, compare A (main) vs B.

  dump    ALGO OUT.npz        (the installed .so is the arm)
  compare ALGO A.npz B.npz

ALGO, binding, define (B = -D <define>, A = main = none):
  pca             estimators   MOJOLEARN_PCA_FAST_POOL
  kernel-pca      x_neighbors  MOJOLEARN_KPCA_RESIDENT
  randomized-svd  x_decomp     MOJOLEARN_RSVD_FAST_DIRECT_IN
  lle             x_decomp     MOJOLEARN_LLE_FAST_DEV_LU

Every dump prints W4Q-CAPTURE {json} with the bound binary's sha256 and a
reach value read from the binary itself (B must have the candidate compiled
in, A must not), and W4Q-TIME lines with each call's time AND the time of
the first read of what it returned (none of these candidates moves where an
output lives; the first-read time is printed so that claim is checked).

Tolerances, fixed before any result (fixture w4q-v1):
- pca (no arithmetic change; the MMA Gram's f32 atomics make run-to-run bits
  differ, gl2p-pca-quality measured 3.4e-6 / 1.8e-6 rad between Gram arms):
  on a seeded 200,000 x 220 Istella-like matrix (d past the split-K Gram, so
  the MMA arm and the pooled buffer are what runs), fitted THREE times in
  one process (the second and third fits reuse the pooled, dirty buffer):
  per fit, mean_ max rel diff <= 1e-6, explained_variance_ max rel diff
  <= 1e-4, noise_variance_ rel diff <= 1e-4, largest principal angle between
  the component subspaces <= 1e-3 rad, and relative reconstruction error of
  the centered fixture by the 10 components (float64, numpy) |A - B| <=
  1e-5 * A. A NaN input must raise the same exception type on both arms.
- kernel-pca (KPCA_RESIDENT builds the RBF kernel as exp(-gamma sqdist) on
  x_decomp's kit; bits differ by design): tools/kpca_quality_ab.py's
  tolerances, eigenvalues_ max rel diff < 1e-4 and largest principal angle
  between the fit_transform column spaces < 1e-3 rad, on its seeded
  Istella-like 10,000 x 220 matrix AND a taxi-like 10,000 x 11 one
  (n_components=8, kernel='rbf', random_state=7, the board's arguments).
- randomized-svd (same words reach the device): seeded 300,000 x 220 and
  300,000 x 11 matrices, n_components=8, n_oversamples=10, n_iter=4,
  random_state=7 (the board's): S max rel diff <= 1e-4, largest principal
  angle between the Vt row spaces <= 1e-3 rad and between the U column
  spaces <= 1e-3 rad, relative rank-8 reconstruction error |A - B| <= 1e-4 *
  A; a NaN input and a 1-D input raise the same exception type and message on
  both arms; a wide (transposed) input is also compared under the same
  tolerances. Byte equality is reported, not required (the kit GEMM's MMA
  split-K may sum in either order).
- lle (the same launches on the same words; expected byte-identical):
  seeded 3,000 x 11 and 3,000 x 220 matrices, n_neighbors=10,
  n_components=2, random_state=7 (the board's arguments): embedding columns
  sign-aligned, max |A - B| <= 1e-4 * max |A|, largest principal angle <=
  1e-3 rad, and trustworthiness (k=15, scikit-learn) |A - B| <= 1e-3.
  Byte equality is reported.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "w4q-v1"
ALGOS = {
    "pca": ("_mojolearn_estimators", "MOJOLEARN_PCA_FAST_POOL"),
    "kernel-pca": ("_mojolearn_x_neighbors", "MOJOLEARN_KPCA_RESIDENT"),
    "randomized-svd": ("_mojolearn_x_decomp", "MOJOLEARN_RSVD_FAST_DIRECT_IN"),
    "lle": ("_mojolearn_x_decomp", "MOJOLEARN_LLE_FAST_DEV_LU"),
}


def _binding(modname):
    from mojolearn import _backend
    m = _backend.binding(modname, "fast")
    return hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(), m


def _reach(algo, mod):
    if algo == "pca":
        return int(mod.pca_fast_pool_on()) if hasattr(mod, "pca_fast_pool_on") else -1
    if algo == "kernel-pca":
        return int(mod.x_neighbors_kpca_resident()) if hasattr(mod, "x_neighbors_kpca_resident") else -1
    if not hasattr(mod, "x_decomp_w4_flags"):
        return -1
    bit = 2 if algo == "randomized-svd" else 1
    return 1 if int(mod.x_decomp_w4_flags()) & bit else 0


def _err(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (the type is part of the record)
        return type(e).__name__, str(e)[:300]
    return "", ""


def _timed(tag, call, read):
    """Run call(); then read(result) (the first touch of what it returned).
    Prints both times; returns the result."""
    t0 = time.perf_counter()
    r = call()
    t1 = time.perf_counter()
    read(r)
    t2 = time.perf_counter()
    print("W4Q-TIME %s call_ms=%.1f first_read_ms=%.1f" % (tag, 1e3 * (t1 - t0), 1e3 * (t2 - t1)), flush=True)
    return r


def _istella_like(rng, n, d):
    """Columns spanning orders of magnitude, correlated (Istella-S's raw
    features span seven)."""
    Z = rng.standard_normal((n, d)).astype(np.float32)
    Z *= (10.0 ** (-2.0 * np.arange(d) / max(d - 1, 1))).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = (Z @ Q.astype(np.float32)) * np.exp(rng.uniform(-3, 6, d)).astype(np.float32)
    return np.ascontiguousarray(X, dtype=np.float32)


def _kpca_data(d):
    # tools/kpca_quality_ab.py's fixture for d = 220 (same seed, same steps)
    rng = np.random.default_rng(7)
    n = 10_000
    Z = rng.standard_normal((n, d)).astype(np.float32)
    Z *= (10.0 ** (-2.0 * np.arange(d) / (d - 1))).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = Z @ Q.astype(np.float32)
    X /= X.std(axis=0, keepdims=True)
    return np.ascontiguousarray(X, dtype=np.float32)


def _read(x):
    return float(np.asarray(x, dtype=np.float64).sum())


def dump_pca(ml, out):
    rng = np.random.default_rng(20261004)
    X = _istella_like(rng, 200_000, 220)
    for i in range(3):
        est = _timed("pca-fit%d" % i,
                     lambda: ml.PCA(n_components=10, svd_solver="covariance_eigh", random_state=7).fit(X),
                     lambda e: _read(e.components_))
        out["fit%d_components" % i] = np.asarray(est.components_, np.float64)
        out["fit%d_ev" % i] = np.asarray(est.explained_variance_, np.float64)
        out["fit%d_mean" % i] = np.asarray(est.mean_, np.float64)
        out["fit%d_noise" % i] = np.array([float(est.noise_variance_)])
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    out["xc_norm2"] = np.array([float((Xc * Xc).sum())])
    for i in range(3):
        V = out["fit%d_components" % i]
        R = Xc - (Xc @ V.T) @ V
        out["fit%d_recon" % i] = np.array([float((R * R).sum()) / out["xc_norm2"][0]])
    bad = X[:5000].copy()
    bad[17, 3] = np.nan
    return {"nan": _err(lambda: ml.PCA(n_components=10, svd_solver="covariance_eigh").fit(bad))}


def dump_kernel_pca(ml, out):
    for tag, d in (("istella", 220), ("taxi", 11)):
        X = _kpca_data(d)
        est = ml.KernelPCA(n_components=8, kernel="rbf", random_state=7)
        T = _timed("kpca-%s" % tag, lambda: est.fit_transform(X), _read)
        out[tag + "_ev"] = np.asarray(est.eigenvalues_, np.float64)
        out[tag + "_t"] = np.asarray(T, np.float64)
    return {}


def _rsvd_fn(ml):
    fn = getattr(ml, "randomized_svd", None)
    if fn is None:
        fn = ml.linalg.randomized_svd
    return fn


def dump_randomized_svd(ml, out):
    fn = _rsvd_fn(ml)
    rng = np.random.default_rng(553)
    kw = dict(n_components=8, n_oversamples=10, n_iter=4, random_state=7)
    for tag, X in (("istella", _istella_like(rng, 300_000, 220)),
                   ("taxi", _istella_like(rng, 300_000, 11)),
                   ("wide", _istella_like(rng, 64, 3000))):
        U, S, Vt = _timed("rsvd-%s" % tag, lambda: fn(X, **kw), lambda r: _read(r[0]))
        U, S, Vt = (np.asarray(a, np.float64) for a in (U, S, Vt))
        out[tag + "_U"], out[tag + "_S"], out[tag + "_Vt"] = U, S, Vt
        Xd = X.astype(np.float64)
        R = Xd - (U * S) @ Vt
        out[tag + "_recon"] = np.array([float((R * R).sum() / (Xd * Xd).sum())])
    bad = _istella_like(rng, 2000, 30)
    bad[7, 2] = np.inf
    return {"nan": _err(lambda: fn(bad, **kw)),
            "one_d": _err(lambda: fn(np.ones(50, np.float32), **kw))}


def dump_lle(ml, out):
    from sklearn.manifold import trustworthiness
    rng = np.random.default_rng(8)
    for tag, d in (("taxi", 11), ("istella", 220)):
        X = _istella_like(rng, 3000, d)
        X /= X.std(axis=0, keepdims=True)
        est = ml.LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=7)
        E = np.asarray(_timed("lle-%s" % tag, lambda: est.fit_transform(X), _read), np.float64)
        out[tag + "_E"] = E
        out[tag + "_trust"] = np.array([float(trustworthiness(X, E, n_neighbors=15))])
    return {}


DUMPS = {"pca": dump_pca, "kernel-pca": dump_kernel_pca, "randomized-svd": dump_randomized_svd, "lle": dump_lle}


def dump(algo, path):
    import mojolearn as ml
    modname, define = ALGOS[algo]
    sha, mod = _binding(modname)
    out = {}
    errs = DUMPS[algo](ml, out)
    out["errs"] = np.array([json.dumps(errs, sort_keys=True).encode()])
    np.savez(path, **out)
    print("W4Q-CAPTURE " + json.dumps(dict(algo=algo, binding=modname, binding_sha256=sha,
                                           reach=_reach(algo, mod), fixture=FIXTURE), sort_keys=True), flush=True)


def _angle(A, B):
    """Largest principal angle (rad) between the column spaces of A and B."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    sv = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    return float(np.max(np.arccos(sv)))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))


def compare(algo, pa, pb):
    A, B = np.load(pa), np.load(pb)
    checks = []   # (name, value, limit, ok)

    def chk(name, value, limit):
        checks.append((name, value, limit, bool(value <= limit)))

    ea, eb = json.loads(A["errs"][0]), json.loads(B["errs"][0])
    same_err = ea == eb if algo == "randomized-svd" else {k: v[0] for k, v in ea.items()} == {k: v[0] for k, v in eb.items()}
    checks.append(("refusals_same", ea, eb, bool(same_err)))
    exact = all(np.array_equal(A[k], B[k]) for k in A.files if k != "errs")
    if algo == "pca":
        for i in range(3):
            f = "fit%d_" % i
            chk(f + "mean_rel", _rel(A[f + "mean"], B[f + "mean"]), 1e-6)
            chk(f + "ev_rel", _rel(A[f + "ev"], B[f + "ev"]), 1e-4)
            chk(f + "noise_rel", _rel(A[f + "noise"], B[f + "noise"]), 1e-4)
            chk(f + "angle_rad", _angle(A[f + "components"].T, B[f + "components"].T), 1e-3)
            ra, rb = float(A[f + "recon"][0]), float(B[f + "recon"][0])
            chk(f + "recon_absdiff_over_A", abs(ra - rb) / max(ra, 1e-300), 1e-5)
    elif algo == "kernel-pca":
        for tag in ("istella", "taxi"):
            chk(tag + "_ev_rel", _rel(A[tag + "_ev"], B[tag + "_ev"]), 1e-4)
            chk(tag + "_angle_rad", _angle(A[tag + "_t"], B[tag + "_t"]), 1e-3)
    elif algo == "randomized-svd":
        for tag in ("istella", "taxi", "wide"):
            chk(tag + "_S_rel", _rel(A[tag + "_S"], B[tag + "_S"]), 1e-4)
            chk(tag + "_Vt_angle_rad", _angle(A[tag + "_Vt"].T, B[tag + "_Vt"].T), 1e-3)
            chk(tag + "_U_angle_rad", _angle(A[tag + "_U"], B[tag + "_U"]), 1e-3)
            ra, rb = float(A[tag + "_recon"][0]), float(B[tag + "_recon"][0])
            chk(tag + "_recon_absdiff_over_A", abs(ra - rb) / max(ra, 1e-300), 1e-4)
    else:
        for tag in ("taxi", "istella"):
            Ea, Eb = A[tag + "_E"], B[tag + "_E"]
            sg = np.where(np.sum(Ea * Eb, axis=0) < 0, -1.0, 1.0)
            chk(tag + "_maxabs_over_max", float(np.max(np.abs(Ea - Eb * sg)) / max(np.max(np.abs(Ea)), 1e-300)), 1e-4)
            chk(tag + "_angle_rad", _angle(Ea, Eb), 1e-3)
            chk(tag + "_trust_absdiff", abs(float(A[tag + "_trust"][0]) - float(B[tag + "_trust"][0])), 1e-3)
    ok = all(c[3] for c in checks)
    for name, v, lim, good in checks:
        print("W4Q-CHECK %s %s value=%s limit=%s %s" % (algo, name, v, lim, "ok" if good else "FAIL"))
    print("W4Q-AB algo=%s status=%s byte_identical=%s checks=%d fixture=%s"
          % (algo, "PASS" if ok else "FAIL", exact, len(checks), FIXTURE))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3], sys.argv[4]))
    else:
        raise SystemExit(__doc__)
