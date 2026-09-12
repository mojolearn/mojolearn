#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHAT THE CTR PATH COSTS OUR ARM, at the benchmark's own scale.

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
      python3 tools/criteo_ours_cat_ab.py

DEVIATION 2634 gates the CTR target prep on "some column is declared
categorical". Its 0.9603 flip verdict was measured on taxi and Istella-S,
where the gate always takes the SKIP branch, so the branch that BUILDS the
prep had never been timed by anything. criteo is the only dataset here that
can time it.

This is one process, our arm only, 1,000,000 rows, the indices declared and
withheld ALTERNATING round by round -- alternating because a rented box may
throttle mid-run and blocks would hand one side the cold clocks and the other
the hot ones.

WHAT THE RATIO IS AND IS NOT. `declared/withheld` is the cost of running the
categorical path, which includes the CTR prep 2634 gates AND the CTR tables,
one-hot splits and target statistics the fit then builds. It is NOT the cost
of 2634 alone: that would need the same build with
`-D MOJOLEARN_2634_CTR_PREP_OFF=1` beside it. It is also NOT a comparison of
two ways to solve one problem -- the withheld arm solves an EASIER one, by
splitting 26 hashed-id columns as ordered numbers, and its quality says so.
Read it as the price of the feature, not as a regression.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402

ROWS = int(os.environ.get("CRITEO_AB_ROWS", "1000000"))
ROUNDS = int(os.environ.get("CRITEO_AB_ROUNDS", "3"))
LANES = os.environ.get("CRITEO_AB_LANES", "SymmetricTree,Depthwise").split(",")


def say(*a):
    print(*a, flush=True)


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def positive_column(p):
    a = np.asarray(p, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    return a.ravel()


def main():
    import mojolearn

    d = spec.load_criteo("shipped", ROWS)
    if d.name != "criteo":
        raise SystemExit("load_criteo fell back to %r" % d.name)
    x = np.ascontiguousarray(d.X_train, dtype=np.float32)
    y = np.ascontiguousarray(d.y_train, dtype=np.float32)
    xte = np.ascontiguousarray(d.X_test, dtype=np.float32)
    say("CATAB-DATA rows=%d feats=%d ncat=%d rounds=%d"
        % (x.shape[0], x.shape[1], len(d.cat_idx), ROUNDS))

    for policy in [p.strip() for p in LANES if p.strip()]:
        times = {True: [], False: []}
        quality = {}
        for r in range(ROUNDS + 1):          # round 0 is an untimed warm-up
            for cats in (True, False):
                params = dict(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    l2_leaf_reg=1.0, border_count=254, random_state=7,
                    bootstrap_type="No", grow_policy=policy, loss="Logloss")
                if policy == "Lossguide":
                    params["max_leaves"] = 31
                if cats:
                    params["cat_features"] = list(d.cat_idx)
                m = mojolearn.GradientBoosting(**params)
                t0 = time.perf_counter()
                m.fit(x, y)
                ms = (time.perf_counter() - t0) * 1000.0
                if r == 0:
                    say("CATAB-WARMUP policy=%s cats=%d ms=%.1f"
                        % (policy, cats, ms))
                    continue
                times[cats].append(ms)
                say("CATAB policy=%s cats=%d round=%d ms=%.1f"
                    % (policy, cats, r, ms))
                if r == ROUNDS:
                    score = positive_column(m.predict_proba(xte))
                    quality[cats] = (spec.auc(d.y_test, score),
                                     logloss(d.y_test, score))
        med = {c: float(np.median(v)) if v else float("nan")
               for c, v in times.items()}
        say("CATAB-MEDIAN policy=%s declared_ms=%.1f withheld_ms=%.1f "
            "ratio=%.4f" % (policy, med[True], med[False],
                            med[True] / med[False] if med[False] else
                            float("nan")))
        for cats in (True, False):
            if cats in quality:
                say("CATAB-QUALITY policy=%s cats=%d auc=%.6f logloss=%.6f"
                    % (policy, cats, quality[cats][0], quality[cats][1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
