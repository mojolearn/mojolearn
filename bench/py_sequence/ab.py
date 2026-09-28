# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane py-sequence before/after: time and digest every touched path of the
sequence family, in ONE tree (run it once against the base tree and once
against the lane's, same box, same job) and on ONE arm.

    python bench/py_sequence/ab.py --arm gpu|cpu --out result.json [--only a,b] [--old]

--arm cpu runs under MOJOLEARN_VENDOR=cpu (every binding from its CPU host
build, as tools/algos_lane_check.py's CPU arm), --arm gpu on the device. --old marks the base tree: cases whose old cost is infeasible run a
bounded prefix there (named in the case). Only public API is used, so the
same file drives both trees. A digest is sha256 over the raw bytes of every
output array, in a fixed order."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def f32(vals):
    return np.asarray(vals, dtype=np.float32)


class Clock:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


# ---------------------------------------------------------------- cases
def case_sched(ml, old):
    out = {}
    T = 312_500
    sch = {
        "onecycle_cos": lambda: ml.OneCycleLR(1e-2, T),
        "onecycle_linear": lambda: ml.OneCycleLR(1e-2, T, anneal_strategy="linear"),
        "onecycle_3phase": lambda: ml.OneCycleLR(1e-2, 31_250, three_phase=True),
        "steplr": lambda: ml.StepLR(0.1, 1000, 0.9),
    }
    for name, mk in sch.items():
        s = mk()
        n = getattr(s, "total_steps", T)
        with Clock() as c:
            v = f32([s.lr_at(t) for t in range(1, n + 1)])
        out[f"sched_{name}_full{n}"] = dict(s=c.s, digest=digest(v))
    # ExponentialLR: the exact old path is O(t) bits per call; a full run
    # is infeasible on the base tree, so both trees do the same prefix and
    # the same single late calls; the lane's tree also does the full run.
    e = ml.ExponentialLR(0.1, 0.99999)
    with Clock() as c:
        v = f32([e.lr_at(t) for t in range(1, 3001)])
    out["sched_exp_prefix3000"] = dict(s=c.s, digest=digest(v))
    for t in (1_000, 10_000, 31_250):
        e = ml.ExponentialLR(0.1, 0.99999)
        with Clock() as c:
            v = f32([e.lr_at(t)])
        out[f"sched_exp_single_t{t}"] = dict(s=c.s, digest=digest(v))
    if not old:
        e = ml.ExponentialLR(0.1, 0.99999)
        with Clock() as c:
            v = f32([e.lr_at(t) for t in range(1, T + 1)])
        out[f"sched_exp_full{T}"] = dict(s=c.s, digest=digest(v))
        # the fast path against the exact reference, dense and near-tie
        from mojolearn import _x_sequence_sched as S
        bad = 0
        for s in (S.ExponentialLR(0.1, 0.99999), S.ExponentialLR(1 + 2 ** -24 + 2 ** -52, 1 - 2 ** -53),
                  S.StepLR(0.037, 2, 0.93), S.OneCycleLR(0.01, 2000), S.OneCycleLR(0.01, 2000, "linear"),
                  S.OneCycleLR(0.01, 1999, 0.37, three_phase=True)):
            n = min(2001, getattr(s, "total_steps", 2000) + 1)
            for t in range(1, n + 1):
                if np.float32(s.lr_at(t)).tobytes() != np.float32(s._exact_lr_at(t)).tobytes():
                    bad += 1
        out["sched_fast_vs_exact_dense"] = dict(s=0.0, digest="differ=%d" % bad)
    return out


def rnn_data(n, T, D, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D), dtype=np.float32)
    y = (X[:, :, 0].sum(axis=1) * np.float32(0.25)).astype(np.float32)
    return X, y


def case_rnn(ml, old):
    out = {}
    n, T, D = 1_000_000, 8, 4
    X, y = rnn_data(n, T, D)
    steps = (n + 31) // 32
    m = ml.LSTMRegressor(hidden_size=16, batch_size=32, max_epochs=1, optimizer="adam",
                         lr_schedule=ml.OneCycleLR(1e-2, steps), random_state=0)
    with Clock() as c:
        m.fit(X, y)
    out["rnn_lstm_1M_onecycle_fit"] = dict(s=c.s, digest=digest(m.params_, m.loss_curve_))
    with Clock() as c:
        p = m.predict(X[:200_000])
    out["rnn_lstm_predict_200k"] = dict(s=c.s, digest=digest(p))
    # the float32 order cap: 1M rows x 17 epochs = 17M > 2^24 order entries
    Xs, ys = rnn_data(n, 1, 1, seed=1)
    m2 = ml.GRURegressor(hidden_size=2, batch_size=n, max_epochs=17, optimizer="adam", random_state=0)
    try:
        with Clock() as c:
            m2.fit(Xs, ys)
        out["rnn_gru_1M_17epochs"] = dict(s=c.s, digest=digest(m2.params_, m2.loss_curve_))
    except Exception as e:     # the base tree refuses it
        out["rnn_gru_1M_17epochs"] = dict(s=0.0, digest="REFUSED: " + str(e)[:80])
    return out


def grads_for(n, k):
    rng = np.random.default_rng(100 + k)
    return rng.standard_normal(n, dtype=np.float32) * np.float32(1e-2)


def case_optim(ml, old):
    out = {}
    n = 1_000_000
    G = [grads_for(n, k) for k in range(4)]
    kinds = {
        "rmsprop_centered_mom": lambda p: ml.RMSprop([p], lr=1e-3, centered=True, momentum=0.9),
        "adagrad": lambda p: ml.Adagrad([p], lr=1e-2),
        "lion": lambda p: ml.Lion([p], lr=1e-4, weight_decay=0.1),
        "adamax": lambda p: ml.Adamax([p]),
        "nadam": lambda p: ml.NAdam([p]),
        "nadam_explr": None,
        "lamb": lambda p: ml.LAMB([p]),
    }
    steps = 2000
    for name, mk in kinds.items():
        p = np.linspace(-1, 1, n, dtype=np.float32)
        if name == "nadam_explr":
            opt = ml.NAdam([p])
            opt.lr_schedule = ml.ExponentialLR(2e-3, 0.999)
        else:
            opt = mk(p)
        with Clock() as c:
            for k in range(steps):
                opt.step([G[k & 3]])
        st = [s for s in getattr(opt, "state", [])] if isinstance(getattr(opt, "state", None), list) else []
        out[f"optim_{name}_{steps}steps_1M"] = dict(s=c.s, digest=digest(p, *st))
        # late steps: t jumps to 300,000 (the stateless replay's worst case)
        p2 = np.linspace(-1, 1, n, dtype=np.float32)
        opt2 = ml.NAdam([p2]) if name.startswith("nadam") else mk(p2)
        opt2.t = 300_000
        opt2.step([G[0]])            # the first step after the jump replays on both trees
        with Clock() as c:
            for k in range(1, 4):
                opt2.step([G[k]])
        st2 = [s for s in opt2.state] if isinstance(getattr(opt2, "state", None), list) else []
        out[f"optim_{name}_t300k_3steps"] = dict(s=c.s, digest=digest(p2, *st2))
    # multi-tensor param list (packed and scattered per step)
    ps = [np.linspace(-1, 1, 300_000, dtype=np.float32).reshape(600, 500),
          np.linspace(1, -1, 700_000, dtype=np.float32)]
    opt = ml.Adamax(ps)
    gs = [grads_for(300_000, 7).reshape(600, 500), grads_for(700_000, 8)]
    with Clock() as c:
        for _ in range(500):
            opt.step(gs)
    out["optim_adamax_2tensors_500steps"] = dict(s=c.s, digest=digest(*ps, *opt.state))
    return out


def case_layernorm(ml, old):
    out = {}
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1_000_000, 64), dtype=np.float32)
    dy = rng.standard_normal((1_000_000, 64), dtype=np.float32)
    ln = ml.LayerNorm(64)
    ln.weight[:] = np.linspace(0.5, 1.5, 64, dtype=np.float32)
    ln.bias[:] = np.linspace(-0.1, 0.1, 64, dtype=np.float32)
    with Clock() as c:
        y = ln.forward(x)
    out["layernorm_fwd_1Mx64"] = dict(s=c.s, digest=digest(y))
    with Clock() as c:
        dx = ln.backward(dy)
    out["layernorm_bwd_1Mx64"] = dict(s=c.s, digest=digest(dx, ln.weight_grad, ln.bias_grad))
    return out


def seasonal(B, n, m, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    lvl = rng.uniform(5, 10, (B, 1)).astype(np.float32)
    amp = rng.uniform(0.5, 2, (B, 1)).astype(np.float32)
    y = lvl + amp * np.sin(2 * np.pi * t / m) + 0.01 * t + rng.standard_normal((B, n)).astype(np.float32) * 0.2
    return np.ascontiguousarray(y, dtype=np.float32)


def case_forecast(ml, old):
    out = {}
    y = seasonal(20_000, 48, 12, 4)
    th = ml.AutoTheta(season_length=12)
    th.fit(y)
    with Clock() as c:
        f1 = th.predict(6)["mean"]
    t1 = c.s
    with Clock() as c:
        f2 = th.predict(6)["mean"]
    out["theta_20k_predict_twice"] = dict(s=t1 + c.s, s_second=c.s, digest=digest(f1, f2, th.info_, np.asarray(th.model_)))
    e = ml.ETS(season_length=12, model="AAA", damped=False)
    e.fit(y)
    with Clock() as c:
        g1 = e.predict(6)["mean"]
    t1 = c.s
    with Clock() as c:
        g2 = e.predict(6)["mean"]
    out["ets_20k_predict_twice"] = dict(s=t1 + c.s, s_second=c.s, digest=digest(g1, g2, e.info_))
    # Prophet: one long series
    N = 20_000
    days = np.arange(N, dtype=np.float64) / 24.0
    ys = np.stack([(np.sin(2 * np.pi * days / 7) + 0.001 * k * days).astype(np.float32) for k in range(64)])
    pr = ml.ProphetForecaster(max_iter=100)
    with Clock() as c:
        pr.fit(days, ys)
    out["prophet_64x20k_fit"] = dict(s=c.s, digest=digest(pr.params_, pr.info_))
    return out


def case_tsa(ml, old):
    out = {}
    B, n, m = 100_000, 40, 4
    y = seasonal(B, n, m, 5)
    hw = ml.ExponentialSmoothing(y, seasonal_periods=m, ts_num=B)
    with Clock() as c:
        hw.fit()
    out["hw_100k_fit"] = dict(s=c.s, digest=digest(np.asarray(hw.sse_)))
    with Clock() as c:
        lv = np.asarray(hw.get_level(7))
        fc = np.asarray(hw.forecast(5, index=7))
    out["hw_level_forecast_index"] = dict(s=c.s, digest=digest(lv, fc))
    with Clock() as c:
        L = np.asarray(hw.level_)
        S = np.asarray(hw.get_season())
    out["hw_components_all"] = dict(s=c.s, digest=digest(L, S))
    yk = seasonal(200_000, 64, 8, 6)
    from mojolearn import _tsa_impl
    with Clock() as c:
        fl = np.asarray(_tsa_impl.kpss_test(np.ascontiguousarray(yk.T), d=1))
    out["kpss_200k"] = dict(s=c.s, digest=digest(fl))
    return out


def case_autoarima(ml, old):
    out = {}
    ya = seasonal(500, 60, 4, 7)
    aa = ml.AutoARIMA(ya)
    with Clock() as c:
        aa.search(p=range(0, 3), q=range(0, 3), d=range(2), P=0, Q=0, maxiter=20)
    out["autoarima_500_search"] = dict(s=c.s, digest=digest(aa.order_, aa.ic_, aa.d_))
    return out


def case_parallel_hw(ml, old):
    out = {}
    from mojolearn import parallel_forecasting as pf
    B, n, m = 2_000, 40, 4
    y = seasonal(B, n, m, 8)
    hw = ml.ExponentialSmoothing(y, seasonal_periods=m, ts_num=B)
    hw.fit()
    with Clock() as c:
        r = np.asarray(pf.forecast_exponential_smoothing(hw, 6, devices=(0,), series_per_shard=500))
    out["parallel_hw_forecast_2k"] = dict(s=c.s, digest=digest(r))
    return out


CASES = dict(sched=case_sched, rnn=case_rnn, optim=case_optim, layernorm=case_layernorm,
             forecast=case_forecast, tsa=case_tsa, autoarima=case_autoarima, parallel_hw=case_parallel_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--old", action="store_true")
    a = ap.parse_args()
    import mojolearn as ml
    from mojolearn import _backend
    vendor = _backend.vendor()
    if (a.arm == "cpu") != (vendor == "cpu"):
        raise SystemExit(f"--arm {a.arm} but the selected vendor is {vendor} (set MOJOLEARN_VENDOR=cpu for the CPU arm)")
    res = {}
    only = [c for c in a.only.split(",") if c]
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        if a.arm == "cpu" and name == "parallel_hw":
            continue
        t0 = time.perf_counter()
        try:
            r = fn(ml, a.old)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {f"{name}_ERROR": dict(s=0.0, digest=f"{type(e).__name__}: {str(e)[:120]}")}
        for k, v in r.items():
            print(f"{a.arm} {k}: {v['s']:.3f}s {v['digest']}", flush=True)
        res.update(r)
        print(f"# {name} done in {time.perf_counter() - t0:.1f}s", flush=True)
        json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
