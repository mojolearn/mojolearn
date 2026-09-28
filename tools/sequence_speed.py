#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S SPEED HARNESS: every sequence-family algorithm at its
large shape, timed ONCE after a small warm-up (which pays the binding load
and the process-lifetime DeviceContext), with a sha256 of every output so a
speed change can show its bits did not move.

    MOJOLEARN_NUMERIC_MODE=identical pixi run python tools/sequence_speed.py \
        --data ~/data/higgs_speed.npz --out /root/speed/before.json [--algos lstm,gru]
    MOJOLEARN_VENDOR=cpu ...                  # the CPU column (host bindings)
    python tools/sequence_speed.py --compare before.json after.json

Data: HIGGS (gbm-bench/higgs/higgs_speed.npz from R2, docs/REMOTE_DATA_R2.md),
1M rows by default. The neural lanes read its rows (sequences cut from
consecutive rows); the forecasters read its columns as series (1M values as
`--series` series); nothing is downloaded here.

Shapes are the lane's working sizes at 1M rows; each record carries them,
the seconds, and the output digest. `--compare` prints before -> after and
refuses when a digest moved (IDENTICAL: the bits must not change)."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# SEQ_SPEED_PYTHON: another checkout's python/ (tools/sequence_apple_ab.sh
# times every variant with this one harness)
sys.path.insert(0, os.environ.get("SEQ_SPEED_PYTHON") or str(ROOT / "python"))


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def load(path, rows):
    z = np.load(os.path.expanduser(path))
    keys = list(z.keys())
    X = z["X_train"] if "X_train" in keys else z[[k for k in keys if k.lower().startswith("x")][0]]
    y = z["y_train"] if "y_train" in keys else z[[k for k in keys if k.lower().startswith("y")][0]]
    X = np.ascontiguousarray(X[:rows], dtype=np.float32)
    y = np.ascontiguousarray(y[:rows], dtype=np.float32)
    # standardise columns (float64 statistics, stored float32) so every
    # model trains on the scale it expects
    mu = X.astype(np.float64).mean(0)
    sd = X.astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    X = np.ascontiguousarray(((X - mu) / sd).astype(np.float32))
    return X, y


#: HIGGS's continuous columns (the b-tags 8, 12, 16, 20 are discrete)
CONTINUOUS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27]


def series(X, B, n):
    """B series of n values from the columns of X, each standardised then
    shifted positive (the multiplicative forecasters need y > 0)."""
    cols = [c for c in CONTINUOUS if c < X.shape[1]]
    flat = np.ascontiguousarray(X[:, cols].T).ravel()
    reps = -(-(B * n) // len(flat))
    v = np.tile(flat, reps)[: B * n].reshape(B, n)
    return np.ascontiguousarray(v + np.float32(10.0), dtype=np.float32)


# ------------------------------------------------------------------ cases
def case_recurrent(ml, X, y, kind, big):
    T = 16
    D = X.shape[1]
    n = (len(X) // T) if big else 256
    Xs = np.ascontiguousarray(X[: n * T].reshape(n, T, D))
    ys = np.ascontiguousarray(y[T - 1: n * T: T])
    cls = dict(lstm=ml.LSTMRegressor, gru=ml.GRURegressor, rnn=ml.RNNRegressor)[kind]
    m = cls(hidden_size=64, num_layers=1, learning_rate=1e-3, batch_size=512, max_epochs=1, random_state=0)
    t0 = time.perf_counter()
    m.fit(Xs, ys)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = m.predict(Xs)
    infer = time.perf_counter() - t0
    return dict(shape=f"{n}x{T}x{D} h64 b512 1 epoch adam", fit_s=fit, infer_s=infer,
                digest=digest(m.params_, m.loss_curve_, p))


def case_mlp(ml, X, y, big):
    n = len(X) if big else 2048
    m = ml.MLPClassifier(hidden_layer_sizes=(256,), batch_size=512, max_iter=1, shuffle=True,
                         random_state=0, tol=0.0, n_iter_no_change=1000)
    yy = (y[:n] > 0.5).astype(np.int64)
    t0 = time.perf_counter()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X[:n], yy)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = m.predict_proba(X[:n])
    infer = time.perf_counter() - t0
    return dict(shape=f"{n}x{X.shape[1]} (256,) b512 1 epoch adam", fit_s=fit, infer_s=infer,
                digest=digest(*[np.asarray(c) for c in m.coefs_], *[np.asarray(c) for c in m.intercepts_], p))


def case_moe(ml, X, y, big):
    n = len(X) if big else 1024
    D = 32
    x = np.zeros((n, D), dtype=np.float32)
    x[:, : X.shape[1]] = X[:n]
    m = ml.MoEBlock(D, 64, num_experts=8, top_k=2, random_state=0)
    t0 = time.perf_counter()
    out = m.forward(x)
    s = time.perf_counter() - t0
    out = out if not isinstance(out, tuple) else out
    arrs = out if isinstance(out, tuple) else (out,)
    return dict(shape=f"{n} tokens D{D} F64 E8 k2 forward", fit_s=s, digest=digest(*arrs))


def case_layernorm(ml, X, y, big):
    n = len(X) if big else 1024
    x = np.ascontiguousarray(X[:n])
    D = x.shape[1]
    w = np.linspace(0.5, 1.5, D, dtype=np.float32)
    b = np.linspace(-0.1, 0.1, D, dtype=np.float32)
    t0 = time.perf_counter()
    yv = ml.layer_norm_forward(x, D, w, b)
    dy = np.ascontiguousarray(x[:, ::-1])
    g = ml.layer_norm_backward(dy, x, D, w, b)
    s = time.perf_counter() - t0
    arrs = [yv] + (list(g) if isinstance(g, (tuple, list)) else [g])
    return dict(shape=f"{n}x{D} forward+backward", fit_s=s, digest=digest(*arrs))


def case_optim(ml, X, y, name, big):
    P = 4_000_000 if big else 4096
    rng = np.random.default_rng(0)
    p = rng.standard_normal(P).astype(np.float32) * np.float32(0.1)
    if name == "adafactor":
        side = int(np.sqrt(P))
        p = np.ascontiguousarray(p[: side * side].reshape(side, side))
    params = [p]
    cls = dict(rmsprop=ml.RMSprop, adagrad=ml.Adagrad, lion=ml.Lion, adamax=ml.Adamax, nadam=ml.NAdam,
               lamb=ml.LAMB, adafactor=ml.Adafactor)[name]
    opt = cls(params)
    steps = 10
    grads = [rng.standard_normal(p.shape).astype(np.float32) for _ in range(2)]
    t0 = time.perf_counter()
    for i in range(steps):
        opt.step([grads[i % 2]])
    s = time.perf_counter() - t0
    out = opt.params if hasattr(opt, "params") else params
    return dict(shape=f"{p.size} params x {steps} steps", fit_s=s, digest=digest(*[np.asarray(a) for a in out]))


def case_ts(ml, X, y, name, big, B_big, n_len):
    B = B_big if big else 8
    Y = series(X, B, n_len)
    t0 = time.perf_counter()
    if name == "stl":
        r = ml.STL(Y, period=12, robust=True).fit()
        outs = (r.seasonal, r.trend, r.resid, r.weights)
    elif name == "theta":
        m = ml.AutoTheta(season_length=12).fit(Y)
        outs = (m.predict(12)["mean"] if isinstance(m.predict(12), dict) else m.predict(12),)
    elif name == "croston":
        m = ml.CrostonOptimized().fit(np.where(Y > 10.8, Y, 0).astype(np.float32))
        p = m.predict(12)
        outs = (p["mean"] if isinstance(p, dict) else p,)
    elif name == "ets":
        m = ml.ETS(season_length=12, model="AAA", damped=True).fit(Y)
        p = m.predict(12)
        outs = (p["mean"] if isinstance(p, dict) else p,)
    elif name == "garch":
        m = ml.GARCH(1, 0, 1).fit(Y - np.float32(10.0), horizon=5)
        outs = (m.params_ if hasattr(m, "params_") else m.params, m.forecast(5))
    elif name == "arima":
        m = ml.ARIMA((1, 1, 1)).fit(Y)
        outs = (np.asarray(m.params) if hasattr(m, "params") else m.params_, m.forecast(12))
    elif name == "hw":
        m = ml.ExponentialSmoothing(Y, seasonal_periods=12, ts_num=B)
        m.fit()
        outs = (m.forecast(12),)
    elif name == "kpss":
        r = ml.kpss_test(np.ascontiguousarray(Y.T), return_statistic=True)
        outs = tuple(np.asarray(v) for v in (r if isinstance(r, tuple) else (r,)))
    elif name == "autoarima":
        m = ml.AutoARIMA(Y).search(d=range(2), p=range(2), q=range(2))
        m.fit()
        outs = (m.order_, m.forecast(12))
    elif name == "var":
        Yv = np.ascontiguousarray(series(X, 4, len(X) if big else 500).T)
        r = ml.VAR(Yv).fit(maxlags=2)
        outs = (r.params, r.sigma_u)
        B, n_len = Yv.shape[1], Yv.shape[0]
    elif name == "prophet":
        # SEQ_PROPHET_N: the A/B's shape (the 1M-point IDENTICAL fit is one
        # GPU thread, ~25 minutes on an M4 Pro)
        n = min(len(X), int(os.environ.get("SEQ_PROPHET_N", 1_000_000))) if big else 2000
        tt = np.arange(n, dtype=np.float64) / 24.0
        yy = np.ascontiguousarray(X[:n, 0] + np.float32(10.0))
        m = ml.ProphetForecaster().fit(tt, yy)
        outs = (m.predict(tt[-48:]),)
        B, n_len = 1, n
    else:
        raise KeyError(name)
    s = time.perf_counter() - t0
    flat = []
    for o in outs:
        if isinstance(o, dict):
            flat.extend(np.asarray(v) for _, v in sorted(o.items()))
        elif hasattr(o, "__dict__") and not isinstance(o, np.ndarray):
            flat.extend(np.asarray(v) for _, v in sorted(vars(o).items()) if isinstance(v, np.ndarray))
        else:
            flat.append(np.asarray(o))
    rec = dict(shape=f"{B} series x {n_len}", fit_s=s, digest=digest(*flat))
    it = None
    if name == "garch" and hasattr(m, "n_iter_"):
        it, cap = np.asarray(m.n_iter_), 4000
    elif name == "ets" and hasattr(m, "info_"):
        it, cap = np.asarray(m.info_)[:, 6].astype(np.int64), 1000
    if it is not None:
        # the work a GPU does: every 32-series simdgroup runs its slowest member
        g = it[: len(it) // 32 * 32].reshape(-1, 32).max(1)
        rec["iters"] = dict(mean=float(it.mean()), max=int(it.max()), p50=float(np.median(it)),
                            simd_max_mean=float(g.mean()), capped=int((it >= cap).sum()))
    return rec


TS = dict(stl=(10000, 100), theta=(10000, 100), croston=(10000, 100), ets=(10000, 100),
          garch=(10000, 100), arima=(10000, 100), hw=(10000, 100), kpss=(10000, 100),
          autoarima=(2000, 100), var=(0, 0), prophet=(0, 0))
#: every case; ALL is the default --algos
NEURAL = ["lstm", "gru", "rnn", "mlp", "moe", "layernorm"]
OPTIM = ["rmsprop", "adagrad", "lion", "adamax", "nadam", "lamb", "adafactor"]
ALL = NEURAL + OPTIM + list(TS)


def run_case(ml, X, y, name, big):
    if name in ("lstm", "gru", "rnn"):
        return case_recurrent(ml, X, y, name, big)
    if name == "mlp":
        return case_mlp(ml, X, y, big)
    if name == "moe":
        return case_moe(ml, X, y, big)
    if name == "layernorm":
        return case_layernorm(ml, X, y, big)
    if name in OPTIM:
        return case_optim(ml, X, y, name, big)
    B, n = TS[name]
    return case_ts(ml, X, y, name, big, B, n)


def compare(a, b):
    A = {r["algo"]: r for r in json.load(open(a))["records"]}
    Bd = {r["algo"]: r for r in json.load(open(b))["records"]}
    moved = []
    print(f"{'algo':12s} {'before s':>10s} {'after s':>10s} {'x':>7s}  bits")
    for k in A:
        if k not in Bd:
            continue
        ra, rb = A[k], Bd[k]
        if "error" in ra or "error" in rb:
            print(f"{k:12s} error: {ra.get('error', '')[:40]} | {rb.get('error', '')[:40]}")
            continue
        same = ra["digest"] == rb["digest"]
        if not same:
            moved.append(k)
        print(f"{k:12s} {ra['fit_s']:10.3f} {rb['fit_s']:10.3f} {ra['fit_s'] / max(rb['fit_s'], 1e-9):7.2f}  "
              f"{'same' if same else 'MOVED'}")
    if moved:
        print("BITS MOVED:", ", ".join(moved))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=next((p for p in ("~/data/higgs_speed.npz",
                    "~/datasets/gbm-bench/higgs/higgs_speed.npz") if os.path.exists(os.path.expanduser(p))),
                    "~/data/higgs_speed.npz"))
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--algos", default=",".join(ALL))
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    import mojolearn as ml
    X, y = load(args.data, args.rows)
    recs = []
    out = Path(os.path.expanduser(args.out)) if args.out else None
    for name in args.algos.split(","):
        rec = dict(algo=name)
        try:
            run_case(ml, X, y, name, False)          # warm-up: binding load, context, kernels
            if name in os.environ.get("SEQ_PROFILE", "").split(","):
                # a SPLIT, never a timing: where the host time goes (cProfile)
                import cProfile, io, pstats
                pr = cProfile.Profile()
                pr.enable()
                run_case(ml, X, y, name, True)
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
                print("\n".join("PROF " + name + " " + l for l in s.getvalue().splitlines() if l.strip()), flush=True)
            rec.update(run_case(ml, X, y, name, True))
        except Exception as e:                       # recorded, never hidden
            rec["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(rec), flush=True)
        recs.append(rec)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(
                mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", ""), vendor=os.environ.get("MOJOLEARN_VENDOR", "gpu"),
                host=platform.node(), rows=args.rows, data=args.data, records=recs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
