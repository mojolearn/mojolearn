#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane py-consolidated: the small cross-tree timing cases for the speed claims
that have no reference arm in the head build (py-sequence's schedules, optimizer
replay, LayerNorm backward, Holt-Winters index slices and Theta repeat predict).

    python tools/py_consolidated/timing.py [--only a,b]

The tree is whatever `mojolearn` PYTHONPATH resolves (the base snapshot or the
head tree); the column is MOJOLEARN_VENDOR (cpu, else the GPU). Sizes shrink on
the CPU column. One line per case: `TIME <col> <case> <seconds> <digest>`. The
same public calls run on both trees, so equal digests mean equal outputs."""
import argparse
import hashlib
import os
import time

import numpy as np

CPU = os.environ.get("MOJOLEARN_VENDOR") == "cpu"
COL = "cpu" if CPU else "gpu"


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def seasonal(B, n, m, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    lvl = rng.uniform(5, 10, (B, 1)).astype(np.float32)
    amp = rng.uniform(0.5, 2, (B, 1)).astype(np.float32)
    y = lvl + amp * np.sin(2 * np.pi * t / m) + 0.01 * t + rng.standard_normal((B, n)).astype(np.float32) * 0.2
    return np.ascontiguousarray(y, dtype=np.float32)


def sched_exp(ml):
    e = ml.ExponentialLR(0.1, 0.99999)
    t0 = time.perf_counter()
    v = np.asarray([e.lr_at(t) for t in range(1, 3001)], dtype=np.float32)
    return time.perf_counter() - t0, digest(v)


def sched_onecycle(ml):
    s = ml.OneCycleLR(1e-2, 2000)
    t0 = time.perf_counter()
    v = np.asarray([s.lr_at(t) for t in range(1, 2001)], dtype=np.float32)
    return time.perf_counter() - t0, digest(v)


def optim_nadam(ml):
    n = 100_000 if CPU else 1_000_000
    rng = np.random.default_rng(5)
    G = [(rng.standard_normal(n, dtype=np.float32) * np.float32(1e-2)) for _ in range(4)]
    p = np.linspace(-1, 1, n, dtype=np.float32)
    opt = ml.NAdam([p])
    t0 = time.perf_counter()
    for k in range(200):
        opt.step([G[k & 3]])
    st = list(opt.state) if isinstance(getattr(opt, "state", None), list) else []
    return time.perf_counter() - t0, digest(p, *st)


def layernorm_bwd(ml):
    rng = np.random.default_rng(3)
    n = 50_000 if CPU else 500_000
    x = rng.standard_normal((n, 64), dtype=np.float32)
    dy = rng.standard_normal((n, 64), dtype=np.float32)
    ln = ml.LayerNorm(64)
    ln.weight[:] = np.linspace(0.5, 1.5, 64, dtype=np.float32)
    ln.bias[:] = np.linspace(-0.1, 0.1, 64, dtype=np.float32)
    ln.forward(x)
    t0 = time.perf_counter()
    dx = ln.backward(dy)
    return time.perf_counter() - t0, digest(dx, ln.weight_grad, ln.bias_grad)


def hw_index(ml):
    B = 20_000 if CPU else 100_000
    y = seasonal(B, 40, 4, 5)
    hw = ml.ExponentialSmoothing(y, seasonal_periods=4, ts_num=B)
    hw.fit()
    t0 = time.perf_counter()
    lv = np.asarray(hw.get_level(7))
    fc = np.asarray(hw.forecast(5, index=7))
    return time.perf_counter() - t0, digest(lv, fc, np.asarray(hw.sse_))


def theta_twice(ml):
    y = seasonal(5_000 if CPU else 20_000, 48, 12, 4)
    th = ml.AutoTheta(season_length=12)
    th.fit(y)
    t0 = time.perf_counter()
    f1 = th.predict(6)["mean"]
    f2 = th.predict(6)["mean"]
    return time.perf_counter() - t0, digest(f1, f2, np.asarray(th.model_))


CASES = dict(sched_exp_prefix3000=sched_exp, sched_onecycle_2000=sched_onecycle, optim_nadam_200steps=optim_nadam,
             layernorm_bwd=layernorm_bwd, hw_level_forecast_index=hw_index, theta_predict_twice=theta_twice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import mojolearn as ml
    only = [c for c in a.only.split(",") if c]
    failed = 0
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        try:
            s, d = fn(ml)
            print(f"TIME {COL} {name} {s:.4f} {d}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"TIME {COL} {name} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
