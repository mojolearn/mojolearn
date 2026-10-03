# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QUALITY-ONLY A/B of a FAST PCA build switch (no timing).

  python tools/pca_quality_ab.py fit OUT.npz      fit mojolearn.PCA on the seeded
      istella-like 200,000 x 220 matrix with the installed estimators binding
  python tools/pca_quality_ab.py compare A.npz B.npz
      PCAQ max_rel_ev_diff=... max_angle_rad=... PASS|FAIL (1e-4, 1e-3)

The matrix: a decaying spectrum (singular scales 10^(-4 t / 219)), a random
rotation, log-normal column scales and offsets, float32, seed 7. PCA as the
classical board runs it: n_components=10, svd_solver='covariance_eigh'.
"""
import sys

import numpy as np


def data():
    rng = np.random.default_rng(7)
    n, d = 200_000, 220
    Z = rng.standard_normal((n, d)).astype(np.float32)
    Z *= (10.0 ** (-4.0 * np.arange(d) / (d - 1))).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = Z @ Q.astype(np.float32)
    X *= np.exp(rng.normal(0.0, 1.5, d)).astype(np.float32)
    X += rng.normal(0.0, 10.0, d).astype(np.float32)
    return np.ascontiguousarray(X, dtype=np.float32)


def fit(out):
    import mojolearn as ml
    est = ml.PCA(n_components=10, svd_solver="covariance_eigh", whiten=False, random_state=7)
    est.fit(data())
    np.savez(out, ev=np.asarray(est.explained_variance_, dtype=np.float64),
             comp=np.asarray(est.components_, dtype=np.float64))
    print("PCAQ-FIT", out, "ev0=%.6g" % float(np.asarray(est.explained_variance_)[0]))


def compare(a, b):
    A, B = np.load(a), np.load(b)
    rel = float(np.max(np.abs(A["ev"] - B["ev"]) / np.abs(A["ev"])))
    qa, _ = np.linalg.qr(A["comp"].T)
    qb, _ = np.linalg.qr(B["comp"].T)
    sv = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    ang = float(np.max(np.arccos(sv)))
    ok = rel < 1e-4 and ang < 1e-3
    print("PCAQ max_rel_ev_diff=%.3e max_angle_rad=%.3e %s" % (rel, ang, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
