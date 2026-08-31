# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The root split, computed on the host, against CatBoost's own answer.

WHY THIS EXISTS
---------------
`mojo_only/oracle_check.mojo` found that our port's trees match CatBoost run
with `score_function=L2` at depths 0 to 2, and diverge from CatBoost run at
its DEFAULT `Cosine` at depth 0, the root. Depth 0 is one leaf holding every
row, so the split is a pure argmax over a histogram both sides compute from
identical inputs. There is nothing accumulated to blame.

This script removes the GPU, the port and Mojo from the question entirely. It
recomputes the ROOT split in numpy directly from the oracle's own data, using
the formulas transcribed from `catboost/cuda/methods/kernel/score_calcers.cuh`,
and prints the argmax for each score function in float64 and float32.

WHAT IT ESTABLISHES, and the L2 line is the important one
---------------------------------------------------------
    l2      float64: feature 0 bin 7     <- MATCHES CatBoost's L2 run exactly
    l2      float32: feature 0 bin 7
    cosine  float64: feature 0 bin 7     <- CatBoost's Cosine run says bin 10
    cosine  float32: feature 0 bin 7

The L2 agreement is what makes this harness evidence rather than a second
opinion: the same code, the same data and the same convention reproduce
CatBoost's L2 root split. So the harness is right about L2, and its Cosine
disagrees with their Cosine.

That rules out three things at once. It is NOT float32 versus their double,
because float64 gives the same answer. It is NOT our GPU kernel, because no
kernel runs here. It is NOT the port's score-function selection, because this
script has no selection.

WHAT IS RULED OUT ON THEIR SIDE, read from their source at 54a8143a
-------------------------------------------------------------------
  * `normalize` is hardcoded `false` at the call site
    (`greedy_search_helper.cpp:467`), so `lambda = Lambda` and not
    `Lambda * weight`.
  * `Cosine` and `L2` are BOTH first order
    (`IsSecondOrderScoreFunction`, `enum_helpers.cpp:830-846`), so
    `ComputeTarget` hands both the identical target and the stat planes hold
    the same numbers. The two runs differ only in the calcer.
  * `NextFeature` seeds `Score = 0; DenumSqr = 1e-10f`
    (`score_calcers.cuh:146-149`), which this script reproduces.
  * `ScoreStdDev` is 0 because `random_strength` is 0, so the noise term in
    `GetScore` (`score_calcers.cuh:162-166`) is not in play.

So the remaining candidates are all in how `TCosineScoreCalcer` is FED rather
than what it computes. That is where to look next, and this script is the
place to test a hypothesis in seconds instead of a build.

USAGE
    pixi run -e bench python tools/score_probe.py
"""
import json

import numpy as np


def main() -> int:
    d = json.load(open("bench/oracle.json"))
    rows = d["config"]["rows"]
    feats = d["config"]["feats"]
    lam = d["config"]["l2_leaf_reg"]
    borders = [
        np.array(b["borders"], dtype=np.float64)
        for b in d["float_feature_borders"]
    ]

    # Rebuilt with the same seed and the same expression as the oracle, so
    # this is their matrix and not a lookalike.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, feats)).astype(np.float32)
    y = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 3]
        + 1.5 * x[:, 7] * x[:, 0]
        + 0.1 * rng.normal(size=rows)
    ).astype(np.float32)

    def root_argmax(dtype, mode):
        best = (-1e300, -1, -1)
        for f in range(feats):
            b = np.searchsorted(borders[f], x[:, f].astype(np.float64), "left")
            nb = len(borders[f])
            w = np.bincount(b, minlength=nb + 1).astype(dtype)
            s = np.bincount(
                b, weights=y.astype(np.float64), minlength=nb + 1
            ).astype(dtype)
            cw = np.cumsum(w).astype(dtype)
            cs = np.cumsum(s).astype(dtype)
            tw = dtype(w.sum())
            ts = dtype(s.sum())
            for i in range(nb):
                wl, sl = cw[i], cs[i]
                wr, sr = dtype(tw - wl), dtype(ts - cs[i])
                L = dtype(lam)
                if mode == "cosine":
                    mul = dtype(sl / (wl + L)) if wl > 0 else dtype(0)
                    mur = dtype(sr / (wr + L)) if wr > 0 else dtype(0)
                    sc = dtype(sl * mul + sr * mur)
                    dn = dtype(dtype(1e-10) + wl * mul * mul + wr * mur * mur)
                    g = dtype(sc / np.sqrt(dn)) if dn > 1e-15 else dtype(-1e30)
                else:
                    g = dtype(0)
                    if wl > 1e-20:
                        g = dtype(g + sl * sl / (wl + L))
                    if wr > 1e-20:
                        g = dtype(g + sr * sr / (wr + L))
                if g > best[0]:
                    best = (float(g), f, i)
        return best

    for mode in ("cosine", "l2"):
        for dt, name in ((np.float64, "float64"), (np.float32, "float32")):
            g, f, i = root_argmax(dt, mode)
            print(
                "%-7s %s: best split feature %d bin %d   gain %.9g"
                % (mode, name, f, i, g)
            )

    sp = d["trees"][0]["splits"][0]
    f = sp["float_feature_index"]
    bidx = -1
    for i, v in enumerate(borders[f]):
        if abs(v - sp["border"]) <= 1e-9:
            bidx = i
    print()
    print(
        "CatBoost's own root split: feature %d bin %d (border %.9g)"
        % (f, bidx, sp["border"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
