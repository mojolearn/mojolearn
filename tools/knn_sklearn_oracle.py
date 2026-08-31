#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""mojolearn's KNeighborsClassifier / KNeighborsRegressor against
scikit-learn's, on hashed fixtures, in whichever numeric mode the process
loaded.

    PYTHONPATH=python pixi run -e gbmbench python3 tools/knn_sklearn_oracle.py
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python pixi run -e gbmbench \\
        python3 tools/knn_sklearn_oracle.py

WHAT IS COMPARED, AND WHY THE FIXTURE IS SHAPED THIS WAY. The Mojo gates
(`neighbors/original/knn_classify_check.mojo`, `knn_regress_check.mojo`)
prove the device equals a host transcription of cuML's kernels; this file
proves that transcription is the ALGORITHM scikit-learn ships -- uniform
vote, argmax with the LOWEST class on a tie (`scipy.stats.mode`), mean of
the neighbours' targets. The fixture is hashed (no two distances equal to
float64, so the neighbour SET is the same in both libraries and the only
thing under test is the vote / mean), labels are `{-1, 0, 1}` hashed (a
negative label, so `classes_` is not `range(n)`), and the classifier runs
at k = 4 and k = 16 on two classes so that 2-2 / 8-8 VOTE ties happen on purpose
and scikit-learn's tie rule is checked against ours on every one. Nothing
here is uniform data. Exits non-zero on any disagreement.

Probabilities: scikit-learn forms `count / k` in float64 and ours is cuML's
serial float32 sum of `1/k` (DEVIATION 542); they agree to float32
rounding, which is what is asserted (atol 1e-6), not bit equality. The
regressor's mean likewise: float32 serial sum vs float64, atol 1e-5 on
targets in [-5, 5).
"""

import os
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SkClf
from sklearn.neighbors import KNeighborsRegressor as SkReg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))
import mojolearn  # noqa: E402


def hashed(n, d, salt):
    i = np.arange(n, dtype=np.uint64)[:, None]
    f = np.arange(d, dtype=np.uint64)[None, :]
    z = (i * np.uint64(0x9E3779B97F4A7C15) + f * np.uint64(0xBF58476D1CE4E5B9)
         + np.uint64(salt) * np.uint64(0x94D049BB133111EB))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z % np.uint64(1000000)).astype(np.float64) / 1e6).astype(np.float32)


def hashed_labels(n, n_classes, salt):
    i = np.arange(n, dtype=np.uint64)
    z = i * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt) * np.uint64(0xD1B54A32D192ED03)
    z = (z ^ (z >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(32))
    return (z % np.uint64(n_classes)).astype(np.int64) - 1


def main():
    np.seterr(over="ignore")  # the hashes wrap mod 2^64 by design
    n, d, nq = 3000, 8, 200
    X = hashed(n, d, 11)
    Xq = hashed(nq, d, 29)
    y3 = hashed_labels(n, 3, 101)
    y2 = hashed_labels(n, 2, 202)
    yreg = (hashed(n, 1, 77)[:, 0].astype(np.float64) * 10.0 - 5.0).astype(np.float32)
    mode = mojolearn.numeric_mode()
    bad = 0
    lines = [f"knn_sklearn_oracle: mode={mode}"]
    for y, nc in ((y3, 3), (y2, 2)):
        for k in (1, 4, 5, 16):
            ours = mojolearn.KNeighborsClassifier(n_neighbors=k).fit(X, y)
            sk = SkClf(n_neighbors=k, algorithm="brute").fit(X, y)
            p_ours, p_sk = ours.predict(Xq), sk.predict(Xq)
            pr_ours, pr_sk = ours.predict_proba(Xq), sk.predict_proba(Xq)
            cls_ok = np.array_equal(ours.classes_, sk.classes_)
            n_wrong = int((p_ours != p_sk).sum())
            pr_err = float(np.abs(pr_ours.astype(np.float64) - pr_sk).max())
            # how many rows were vote ties (so the tie rule was exercised)
            counts = np.round(pr_sk * k).astype(int)
            top = counts.max(1)
            ties = int(((counts == top[:, None]).sum(1) > 1).sum())
            ok = cls_ok and n_wrong == 0 and pr_err <= 1e-6
            bad += 0 if ok else 1
            lines.append(f"  clf {nc}-class k={k:2d}: predict {'==' if n_wrong == 0 else '!='} "
                         f"sklearn ({n_wrong} wrong of {nq}), proba max|diff| {pr_err:.1e}, "
                         f"classes_ {'==' if cls_ok else '!='}, vote ties {ties}")
            if k % 2 == 0 and nc == 2 and ties == 0:
                bad += 1
                lines.append("    NO VOTE TIES at an even k on two classes: the fixture gates nothing")
    for k in (1, 5, 50):
        ours = mojolearn.KNeighborsRegressor(n_neighbors=k).fit(X, yreg)
        sk = SkReg(n_neighbors=k, algorithm="brute").fit(X, yreg)
        err = float(np.abs(ours.predict(Xq).astype(np.float64) - sk.predict(Xq)).max())
        ok = err <= 1e-5
        bad += 0 if ok else 1
        lines.append(f"  reg k={k:2d}: max|ours - sklearn| {err:.1e} {'OK' if ok else 'FAIL'}")
    # multi-output through both libraries
    Y2 = np.stack([y3, y2], 1)
    ours = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X, Y2)
    sk = SkClf(n_neighbors=5, algorithm="brute").fit(X, Y2)
    mo_ok = np.array_equal(ours.predict(Xq), sk.predict(Xq))
    bad += 0 if mo_ok else 1
    lines.append(f"  clf multi-output (3-class, 2-class) predict {'==' if mo_ok else '!='} sklearn")
    # the refusals, by name
    for kw in (dict(weights="distance"), dict(metric="cosine"), dict(algorithm="kd_tree")):
        try:
            mojolearn.KNeighborsClassifier(**kw).fit(X, y3)
            bad += 1
            lines.append(f"  NOT REFUSED: {kw}")
        except ValueError as e:
            lines.append(f"  refused by name: {kw} -> {str(e)[:60]}...")
    print("\n".join(lines))
    print("knn_sklearn_oracle:", "OK" if bad == 0 else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
