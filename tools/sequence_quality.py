#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S FAST QUALITY CHECK (lane/sequence-apple2): the paired
quality number a FAST change needs ("FAST never degrades quality"), printed
for whichever build MOJOLEARN_NUMERIC_MODE and SEQ_SPEED_PYTHON select, so
tools/sequence_apple_ab.sh runs it for every variant and mode beside the
timings. One QUAL line per case.

    lamb, adafactor   the speed harness's 4M-parameter, 10-step run against a
                      float64 NumPy statement of the same optimizer
                      (timm's Lamb, torch 2.5's Adafactor, as the kernels
                      state them): max and relative L2 error of the final
                      parameters. Lower is better.
    ets               AAA damped ETS on 10000 HIGGS series of 112 values:
                      fit the first 100, forecast 12, against the held out
                      12: mean negative log likelihood at the optimum
                      (info[5], lower is better), holdout MAE and sMAPE
                      (lower is better), mean Nelder-Mead iterations."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("SEQ_SPEED_PYTHON") or str(ROOT / "python"))
sys.path.insert(1, str(ROOT / "tools"))
from sequence_speed import load, series  # noqa: E402


def lamb_ref(p, grads, steps, lr=1e-3, b1=0.9, b2=0.999, eps=1e-6, wd=0.01, max_norm=1.0):
    p = p.astype(np.float64)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t in range(1, steps + 1):
        g = grads[(t - 1) % 2].astype(np.float64)
        clip = np.sqrt(np.sum(g * g)) / max_norm
        if clip > 1.0:
            g = g / clip
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        u = (m / bc1) / (np.sqrt(v) / np.sqrt(bc2) + eps) + wd * p
        wn, un = np.sqrt(np.sum(p * p)), np.sqrt(np.sum(u * u))
        r = wn / un if wn > 0 and un > 0 else 1.0
        p = p - lr * u * r
    return p


def adafactor_ref(p, grads, steps, lr=1e-2, beta2_decay=-0.8, eps1=float(np.finfo(np.float32).eps), eps2=1e-3, d=1.0):
    p = p.astype(np.float64)
    R, C = p.shape
    row = np.zeros(R)
    col = np.zeros(C)
    for t in range(1, steps + 1):
        g = grads[(t - 1) % 2].astype(np.float64)
        rho = min(lr, 1 / np.sqrt(t))
        alpha = max(eps2, np.sqrt(np.sum(p * p)) / np.sqrt(p.size)) * rho
        w = t ** beta2_decay
        row = row + w * ((g * g).mean(1) - row)
        col = col + w * ((g * g).mean(0) - col)
        ve = np.outer(row, col) / max(row.mean(), eps1)
        u = g / np.sqrt(np.maximum(ve, eps1 * eps1))
        den = max(1.0, np.sqrt(np.sum(u * u)) / (np.sqrt(u.size) * d))
        p = p - alpha / den * u
    return p


def q_optim(ml, name):
    P = 4_000_000
    rng = np.random.default_rng(0)
    p = rng.standard_normal(P).astype(np.float32) * np.float32(0.1)
    if name == "adafactor":
        side = int(np.sqrt(P))
        p = np.ascontiguousarray(p[: side * side].reshape(side, side))
    grads = [rng.standard_normal(p.shape).astype(np.float32) for _ in range(2)]
    ref = (lamb_ref if name == "lamb" else adafactor_ref)(p.copy(), grads, 10)
    opt = (ml.LAMB if name == "lamb" else ml.Adafactor)([p])
    for i in range(10):
        opt.step([grads[i % 2]])
    out = np.asarray(opt.params[0], dtype=np.float64)
    err = out - ref
    return dict(max_abs_err=float(np.abs(err).max()),
                rel_l2_err=float(np.sqrt(np.sum(err * err)) / np.sqrt(np.sum(ref * ref))))


def q_ets(ml, X):
    Y = series(X, 10000, 112)
    tr, te = np.ascontiguousarray(Y[:, :100]), Y[:, 100:]
    m = ml.ETS(season_length=12, model="AAA", damped=True).fit(tr)
    f = np.asarray(m.predict(12)["mean"], dtype=np.float64)
    info = np.asarray(m.info_, dtype=np.float64)
    ae = np.abs(f - te)
    smape = 2 * ae / (np.abs(f) + np.abs(te))
    ok = np.isfinite(f).all(1)
    return dict(nll_mean=float(info[:, 5].mean()), nll_median=float(np.median(info[:, 5])),
                mae=float(ae[ok].mean()), smape=float(smape[ok].mean()), nonfinite=int((~ok).sum()),
                iters_mean=float(info[:, 6].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/datasets/gbm-bench/higgs/higgs_speed.npz")
    ap.add_argument("--what", default="lamb,adafactor,ets")
    a = ap.parse_args()
    import mojolearn as ml
    X = None
    for w in a.what.split(","):
        try:
            if w in ("lamb", "adafactor"):
                r = q_optim(ml, w)
            elif w == "ets":
                if X is None:
                    X, _ = load(a.data, 1_000_000)
                r = q_ets(ml, X)
            else:
                raise KeyError(w)
        except Exception as e:
            r = dict(error=f"{type(e).__name__}: {e}")
        print("QUAL", json.dumps(dict(case=w, mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", ""), **r)), flush=True)


if __name__ == "__main__":
    main()
