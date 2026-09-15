#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Agreement between `mojolearn.ARIMA(..., exog=...)` and statsmodels
`SARIMAX` on the regression coefficients and the forecasts (lane/arima-exog,
2026-09-15).

REPORTED, NOT TUNED TO. Nothing in the lane was changed to move a number
here, and no bound below is a gate: this prints what the two libraries say
about the same data and the differences between them. They cannot agree to
the last bit and are not meant to:

  * mojolearn is FLOAT32 on the device (DEVIATION 670) and statsmodels is
    float64 throughout.
  * The optimizers differ. statsmodels drives scipy's L-BFGS-B over its own
    starting values (Hannan-Rissanen); mojolearn runs cuML's own L-BFGS from
    `estimate_x0` with a 2^-10 forward difference (DEVIATION 679, 687). Two
    optimizers stopping in the same basin stop at different points in it.
  * The intercept is not the same parameter. cuML's `mu`, which this package
    spells `trend='c'`, is a STATE intercept: the filter does
    `alpha[n_diff] += mu` every step, so for an AR(1) the implied mean of the
    series is `mu / (1 - phi)`. statsmodels' `trend='c'` coefficient is the
    mean of the regression's intercept term. Both are printed, with the
    transform, and neither is called wrong.

What SHOULD agree closely, because it is the same estimand on the same data,
is `beta`: the regression coefficients of a regression with ARIMA errors.
That is the row to read.

    python3 statsmodels_agreement.py [--out agreement.json]

Needs statsmodels and a mojolearn install that can fit ARIMA (a GPU set, or
the reference arima host binding on a CPU-only install).
"""
import argparse
import json
import sys

import numpy as np


def make_data(seed=11, n=400, h=16, n_exog=2):
    """One series per case: a stationary AR(1) (or differenced seasonal)
    process plus a genuine regression component, so `beta` is identifiable
    and is not a fit of noise on noise. The regressors are their own AR(1)
    processes, as a real covariate would be, not white noise."""
    rng = np.random.default_rng(seed)
    total = n + h
    exog = np.empty((total, n_exog), dtype=np.float64)
    for j in range(n_exog):
        e = rng.standard_normal(total)
        x = np.empty(total)
        x[0] = e[0]
        for t in range(1, total):
            x[t] = 0.6 * x[t - 1] + e[t]
        exog[:, j] = x
    beta_true = np.array([2.0, -0.5])[:n_exog]
    u = np.empty(total)
    e = rng.standard_normal(total) * 0.5
    u[0] = e[0]
    for t in range(1, total):
        u[t] = 0.5 * u[t - 1] + e[t]
    y = exog @ beta_true + u
    return (y[:n].astype(np.float64), exog[:n].astype(np.float64),
            exog[n:].astype(np.float64), beta_true)


CASES = (
    dict(name="arima-exog (1,0,0) trend=n", order=(1, 0, 0),
         seasonal_order=(0, 0, 0, 0), trend="n"),
    dict(name="arima-exog (1,0,0) trend=c", order=(1, 0, 0),
         seasonal_order=(0, 0, 0, 0), trend="c"),
    dict(name="arima-exog-seasonal (1,1,0)(1,0,0,4)", order=(1, 1, 0),
         seasonal_order=(1, 0, 0, 4), trend="n"),
)


def ours(case, y, exog, fut, h):
    import mojolearn
    m = mojolearn.ARIMA(order=case["order"], seasonal_order=case["seasonal_order"],
                        trend=case["trend"]).fit(
        y.astype(np.float32).reshape(1, -1), exog.astype(np.float32)[None, :, :])
    out = dict(beta=np.asarray(m.beta_)[0].astype(np.float64).tolist(),
               sigma2=float(np.asarray(m.sigma2_)[0]),
               llf=float(np.asarray(m.llf_)[0]),
               forecast=np.asarray(m.forecast(h, exog=fut.astype(np.float32)[None, :, :]))[0]
               .astype(np.float64).tolist())
    out["ar"] = np.asarray(m.ar_)[0].astype(np.float64).tolist() if case["order"][0] else []
    out["sar"] = (np.asarray(m.sar_)[0].astype(np.float64).tolist()
                  if case["seasonal_order"][0] else [])
    if case["trend"] == "c":
        mu = float(np.asarray(m.mu_)[0])
        out["mu_state_intercept"] = mu
        phi = out["ar"][0] if out["ar"] else 0.0
        out["mu_implied_mean"] = mu / (1.0 - phi) if abs(1.0 - phi) > 1e-6 else float("nan")
    return out


def theirs(case, y, exog, fut, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(y, exog=exog, order=case["order"],
                    seasonal_order=case["seasonal_order"],
                    trend="c" if case["trend"] == "c" else "n",
                    simple_differencing=True)
    res = model.fit(disp=False)
    named = dict(zip(res.param_names, np.asarray(res.params, dtype=np.float64)))
    beta = [named[k] for k in res.param_names if k.startswith("x")]
    ar = [named[k] for k in res.param_names if k.startswith("ar.L")]
    sar = [named[k] for k in res.param_names if k.startswith("ar.S.L")]
    out = dict(beta=beta, ar=[a for a in ar if a not in sar], sar=sar,
               sigma2=float(named.get("sigma2", float("nan"))),
               llf=float(res.llf),
               forecast=np.asarray(res.forecast(steps=h, exog=fut), dtype=np.float64).tolist())
    if case["trend"] == "c":
        out["intercept"] = float(named.get("intercept", named.get("const", float("nan"))))
    return out


def compare(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    denom = np.maximum(np.abs(b), 1e-12)
    return dict(max_abs=float(np.max(np.abs(a - b))),
                max_rel=float(np.max(np.abs(a - b) / denom)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    h = args.steps
    y, exog, fut, beta_true = make_data(h=h)
    report = dict(beta_true=beta_true.tolist(), n_obs=int(y.size), steps=h, cases=[])
    for case in CASES:
        mine = ours(case, y, exog, fut, h)
        ref = theirs(case, y, exog, fut, h)
        row = dict(case=case["name"], ours=mine, statsmodels=ref,
                   beta=compare(mine["beta"], ref["beta"]),
                   ar=compare(mine["ar"], ref["ar"]),
                   sar=compare(mine["sar"], ref["sar"]),
                   forecast=compare(mine["forecast"], ref["forecast"]))
        report["cases"].append(row)
        print(f"== {case['name']}")
        print(f"   beta true       {np.asarray(beta_true)}")
        print(f"   beta ours       {np.asarray(mine['beta'])}")
        print(f"   beta statsmodels{np.asarray(ref['beta'])}")
        print(f"   beta            {row['beta']}")
        if row["ar"]:
            print(f"   ar              {row['ar']}  ours {mine['ar']} theirs {ref['ar']}")
        if row["sar"]:
            print(f"   sar             {row['sar']}")
        print(f"   forecast ({h})   {row['forecast']}")
        if case["trend"] == "c":
            print(f"   intercept: ours mu (state) {mine['mu_state_intercept']:.6g}, implied mean "
                  f"{mine['mu_implied_mean']:.6g}; statsmodels {ref['intercept']:.6g}")
        print(f"   sigma2 ours {mine['sigma2']:.6g}, statsmodels {ref['sigma2']:.6g}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
