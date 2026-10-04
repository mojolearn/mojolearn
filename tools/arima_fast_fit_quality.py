#!/usr/bin/env python3
"""Capture/compare real AutoARIMA fits for the ARIMA evaluation experiments.

Run capture once per freshly built FAST Apple arm on the M3. No CPU model
or opponent is run. The comparison requires identical orders, criteria,
parameters, log-likelihoods, fitted values and forecasts, including bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def _wide_fixture(np, nobs=1392, horizon=48, n=48, seed=61):
    """Board-shaped: 48 series of mixed processes over 1,392 hours, so many
    chosen orders share one (d, r, k) refit group (ARIMA_FIT_GROUPS)."""
    rng = np.random.default_rng(seed)
    T = nobs + horizon
    e = rng.normal(size=(n, T)).astype(np.float64)
    y = np.zeros((n, T))
    t = np.arange(T)
    for i in range(n):
        kind = i % 6
        phi = .2 + .75 * rng.random()
        for s in range(3, T):
            if kind == 0:
                y[i, s] = phi * y[i, s - 1] + e[i, s]
            elif kind == 1:
                y[i, s] = .5 * y[i, s - 1] + .3 * y[i, s - 2] + e[i, s] + .4 * e[i, s - 1]
            elif kind == 2:
                y[i, s] = y[i, s - 1] + .05 + e[i, s]
            elif kind == 3:
                y[i, s] = e[i, s] + .6 * e[i, s - 1] - .3 * e[i, s - 2]
            elif kind == 4:
                y[i, s] = phi * y[i, s - 1] - .2 * y[i, s - 3] + e[i, s]
            else:
                y[i, s] = .9 * y[i, s - 1] + e[i, s] + .5 * e[i, s - 1]
        if kind == 5:
            y[i] += 20 * np.sin(2 * np.pi * t / 24) + 50
    return y.astype(np.float32), horizon


def capture(path, small=False, wide=False):
    import numpy as np
    import mojolearn as ml

    if os.environ.get("MOJOLEARN_NUMERIC_MODE") != "fast":
        raise RuntimeError("This check requires MOJOLEARN_NUMERIC_MODE=fast")
    binding = ml.ARIMA()._extension()
    if str(binding.arima_vendor()) != "metal" or int(binding.arima_numeric_mode()) != 0:
        raise RuntimeError("This check requires the FAST Metal binding")
    arrays = {}
    metrics = {}
    fixtures = ((23, 128),) if small else ((23, 512), (87, 2048))
    if wide:
        fixtures = fixtures + ((61, 1392),)
    for seed, nobs in fixtures:
        if seed == 61:
            y, horizon = _wide_fixture(np, nobs)
        else:
            rng = np.random.default_rng(seed)
            horizon = 24
            innovations = rng.normal(size=(4, nobs + horizon)).astype(np.float32)
            y = np.zeros_like(innovations)
            # Stationary, nearly nonstationary, integrated and mixed ARMA.
            for t in range(2, y.shape[1]):
                y[0, t] = .65 * y[0, t - 1] + innovations[0, t]
                y[1, t] = .985 * y[1, t - 1] + innovations[1, t]
                y[2, t] = y[2, t - 1] + .08 + innovations[2, t]
                y[3, t] = .55 * y[3, t - 1] - .15 * y[3, t - 2] + innovations[3, t] + .4 * innovations[3, t - 1]
        train = np.ascontiguousarray(y[:, :nobs])
        model = ml.AutoARIMA(train).search(s=1, d=range(2), p=range(4), q=range(4),
                                           P=range(1), D=range(1), Q=range(1), maxiter=20)
        model.fit(maxiter=1000)
        prefix = f"seed{seed}_n{nobs}"
        arrays[prefix + "/input"] = train
        for name in ("order_", "ic_", "d_"):
            arrays[prefix + "/" + name] = np.asarray(getattr(model, name))
        pred = np.asarray(model.predict(2, nobs))
        forecast = np.asarray(model.forecast(horizon))
        arrays[prefix + "/prediction"] = pred
        arrays[prefix + "/forecast"] = forecast
        if not np.isfinite(pred).all() or not np.isfinite(forecast).all():
            raise AssertionError("Nonfinite prediction/forecast: " + prefix)
        for group, fit in enumerate(model._fitted):
            for name in ("params_", "llf_", "n_iter_", "retcode_") + (("x_", "x0_", "fx_", "aic_") if wide else ()):
                value = np.asarray(getattr(fit, name))
                if not np.isfinite(value).all():
                    raise AssertionError("Nonfinite fitted value: " + prefix + "/" + name)
                arrays[f"{prefix}/group{group}/{name}"] = value
        metrics[prefix] = dict(forecast_rmse=float(np.sqrt(np.mean((forecast.astype(np.float64) - y[:, nobs:]) ** 2))),
                               fitted_rmse=float(np.sqrt(np.mean((pred.astype(np.float64) - y[:, 2:nobs]) ** 2))),
                               loglike_sum=float(sum(np.sum(np.asarray(fit.llf_, dtype=np.float64)) for fit in model._fitted)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print("ARIMA_FAST_CAPTURE " + json.dumps(dict(output=str(path), binding=str(binding.__file__),
          binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(), metrics=metrics), sort_keys=True))


def compare(a, b):
    import numpy as np
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise AssertionError("Different fitted groups or missing quality arrays")
        bad = [k for k in left.files if left[k].shape != right[k].shape or left[k].dtype != right[k].dtype
               or left[k].tobytes() != right[k].tobytes()]
        if bad:
            raise AssertionError("Fitted quality bytes differ: " + ", ".join(bad))
        print(f"ARIMA_FAST_QUALITY status=PASS arrays={len(left.files)} forecast=identical loglike=identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "compare"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--small", action="store_true", help="One 128-observation smoke fixture")
    parser.add_argument("--wide", action="store_true",
                        help="Add the board-shaped 48-series, 1392-observation fixture (seed61_n1392) and x/x0/fx/aic")
    args = parser.parse_args()
    if args.action == "capture" and len(args.paths) == 1:
        capture(args.paths[0], args.small, args.wide)
    elif args.action == "compare" and len(args.paths) == 2:
        compare(*args.paths)
    else:
        parser.error("capture needs one output .npz; compare needs two .npz paths")
