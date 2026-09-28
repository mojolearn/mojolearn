# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Paired FAST quality check for the QN solver (lane linear-apple2).

For each seed, a seeded 200k-row train / 100k-row held-out split of HIGGS
(LogisticRegression, LinearSVC) and of taxi (LinearSVR, LogisticRegression on
the card flag) is fitted under the environment's numeric mode and scored on
the held-out rows. Run once per arm (each arm's binding built in turn); the
arms are compared seed by seed.

    python bench/linear_apple_fast_quality.py --arm <label> [--seeds 0,1,2,3,4]

Lines: `QUAL <arm> <case> seed=<s> <metric>=<v> n_iter=<k>`.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from x_linear_speed import _load  # noqa: E402


def _logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    d = _load(1_000_000)
    from mojolearn import linear_model as LM
    from mojolearn import svm as SV
    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(1_000_000)
        tr, te = perm[:200_000], perm[200_000:300_000]
        hx, hy = d["hx"], d["hy"]
        tx, fare, card = d["tx"], d["fare"], d["card"]
        m = LM.LogisticRegression(max_iter=100).fit(hx[tr], hy[tr])
        p = np.asarray(m.predict_proba(hx[te]).tolist())[:, 1]
        print(f"QUAL {a.arm} logistic-higgs seed={seed} logloss={_logloss(p, hy[te]):.6f} "
              f"n_iter={getattr(m, 'n_iter_', '?')}", flush=True)
        m = LM.LogisticRegression(max_iter=100).fit(tx[tr], card[tr])
        p = np.asarray(m.predict_proba(tx[te]).tolist())[:, 1]
        print(f"QUAL {a.arm} logistic-taxi seed={seed} logloss={_logloss(p, card[te]):.6f} "
              f"n_iter={getattr(m, 'n_iter_', '?')}", flush=True)
        m = SV.LinearSVC(max_iter=100).fit(hx[tr], hy[tr])
        acc = float(np.mean(np.asarray(m.predict(hx[te]).tolist()) == hy[te]))
        print(f"QUAL {a.arm} linear-svc-higgs seed={seed} acc={acc:.6f} "
              f"n_iter={getattr(m, 'n_iter_', '?')}", flush=True)
        m = SV.LinearSVR(max_iter=100).fit(tx[tr], fare[tr])
        pr = np.asarray(m.predict(tx[te]).tolist(), dtype=np.float64)
        yt = fare[te].astype(np.float64)
        r2 = 1.0 - float(np.sum((yt - pr) ** 2) / np.sum((yt - yt.mean()) ** 2))
        print(f"QUAL {a.arm} linear-svr-taxi seed={seed} r2={r2:.6f} "
              f"n_iter={getattr(m, 'n_iter_', '?')}", flush=True)


if __name__ == "__main__":
    main()
