# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QUALITY-ONLY A/B of a FAST KernelPCA build switch (no timing).

  python tools/kpca_quality_ab.py fit OUT.npz      KernelPCA(n_components=8,
      kernel='rbf', random_state=7) on a seeded istella-like 10,000 x 220 matrix
  python tools/kpca_quality_ab.py compare A.npz B.npz
      KPCAQ max_rel_eig_diff=... max_angle_rad=... PASS|FAIL (1e-4, 1e-3)
"""
import sys

import numpy as np


def data():
    rng = np.random.default_rng(7)
    n, d = 10_000, 220
    Z = rng.standard_normal((n, d)).astype(np.float32)
    Z *= (10.0 ** (-2.0 * np.arange(d) / (d - 1))).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = Z @ Q.astype(np.float32)
    X /= X.std(axis=0, keepdims=True)
    return np.ascontiguousarray(X, dtype=np.float32)


def fit(out):
    import mojolearn as ml
    est = ml.KernelPCA(n_components=8, kernel="rbf", random_state=7)
    T = np.asarray(est.fit_transform(data()), dtype=np.float64)
    np.savez(out, ev=np.asarray(est.eigenvalues_, dtype=np.float64), t=T)
    print("KPCAQ-FIT", out, "ev0=%.6g" % float(np.asarray(est.eigenvalues_)[0]))


def compare(a, b):
    A, B = np.load(a), np.load(b)
    rel = float(np.max(np.abs(A["ev"] - B["ev"]) / np.abs(A["ev"])))
    qa, _ = np.linalg.qr(A["t"])
    qb, _ = np.linalg.qr(B["t"])
    sv = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    ang = float(np.max(np.arccos(sv)))
    ok = rel < 1e-4 and ang < 1e-3
    print("KPCAQ max_rel_eig_diff=%.3e max_angle_rad=%.3e %s" % (rel, ang, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
